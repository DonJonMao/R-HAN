from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mas_stage2.learning import OnlineLinearModel
from mas_stage2.types import Stage2RunResult, TurnTrace
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import UnionGraph
from stage2_gcr_plus.runtime_v2 import Stage2RuntimeV2
from stage2_phase1_semantic_safe_override.artifacts import ArtifactIR, answer_text
from stage2_phase1_semantic_safe_override.runtime import Phase1SemanticSafeOverrideRuntime
from stage2_phase1_semantic_safe_override.verifier import VerifierState, apply_anchor_pairwise_metrics, mean_residual

from .config import Phase2UnifiedLocalCorrectionConfig
from .correction import (
    CorrectionArtifact,
    DeltaPrediction,
    actual_delta,
    answer_first_gate,
    apply_correction_artifact,
    critique_summary,
    localize_units,
    mean_expected_drop,
    predict_delta,
    preserve_heatmap,
    propose_correction,
)


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))


class Phase2UnifiedLocalCorrectionRuntime(Phase1SemanticSafeOverrideRuntime):
    def __init__(self, config: Phase2UnifiedLocalCorrectionConfig, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._correction_value_model = OnlineLinearModel(learning_rate=config.correction_value_learning_rate)
        self._last_phase2_selection: Dict[str, Any] = {}

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update({"phase2_correction_value_model": self._correction_value_model.state_dict()})
        return payload

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if isinstance(state.get("phase2_correction_value_model"), dict):
            self._correction_value_model.load_state_dict(state["phase2_correction_value_model"])

    def _ensure_phase2_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_phase1_entry_fields(entry)
        entry.setdefault("phase2_answer_first_gate", 0.0)
        entry.setdefault("phase2_localization_map", {})
        entry.setdefault("phase2_preserve_map", {})
        entry.setdefault("phase2_critique", {})
        entry.setdefault("phase2_correction_artifact", {})
        entry.setdefault("phase2_delta_prediction", {})
        entry.setdefault("phase2_delta_actual", {})
        entry.setdefault("phase2_correction_value_features", {})
        entry.setdefault("phase2_correction_value", 0.0)
        entry.setdefault("phase2_correction_risk", 0.0)
        entry.setdefault("phase2_final_utility", float(entry.get("phase1_safe_utility", 0.0)))
        entry.setdefault("phase2_repair_round", 0)
        entry.setdefault("phase2_repair_parent_digest", "")

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        payload.update(
            {
                "phase2_answer_first_gate": float(entry.get("phase2_answer_first_gate", 0.0)),
                "phase2_localization_map": dict(entry.get("phase2_localization_map", {})),
                "phase2_preserve_map": dict(entry.get("phase2_preserve_map", {})),
                "phase2_critique": dict(entry.get("phase2_critique", {})),
                "phase2_correction_artifact": dict(entry.get("phase2_correction_artifact", {})),
                "phase2_delta_prediction": dict(entry.get("phase2_delta_prediction", {})),
                "phase2_delta_actual": dict(entry.get("phase2_delta_actual", {})),
                "phase2_correction_value": float(entry.get("phase2_correction_value", 0.0)),
                "phase2_correction_risk": float(entry.get("phase2_correction_risk", 0.0)),
                "phase2_final_utility": float(entry.get("phase2_final_utility", entry.get("phase1_safe_utility", 0.0))),
                "phase2_repair_round": int(entry.get("phase2_repair_round", 0)),
                "phase2_repair_parent_digest": str(entry.get("phase2_repair_parent_digest", "")),
            }
        )
        return payload

    def _copy_parent_entry(self, parent: Dict[str, Any], new_text: str, *, repair_round: int) -> Dict[str, Any]:
        entry = self._init_candidate_entry(new_text)
        self._ensure_phase2_entry_fields(entry)
        entry["stage1_anchor"] = False
        entry["source_roles"] = set(parent.get("source_roles", set()))
        entry["source_node_ids"] = set(parent.get("source_node_ids", set()))
        entry["turn_indices"] = set(parent.get("turn_indices", set()))
        entry["occurrence_count"] = int(parent.get("occurrence_count", 0))
        entry["sink_support"] = int(parent.get("sink_support", 0))
        entry["feedback_pass"] = float(parent.get("feedback_pass", 0.0))
        entry["feedback_challenge"] = float(parent.get("feedback_challenge", 0.0))
        entry["feedback_uncertain"] = float(parent.get("feedback_uncertain", 0.0))
        entry["feedback_pass_calibrated"] = float(parent.get("feedback_pass_calibrated", 0.0))
        entry["feedback_challenge_calibrated"] = float(parent.get("feedback_challenge_calibrated", 0.0))
        entry["feedback_uncertain_calibrated"] = float(parent.get("feedback_uncertain_calibrated", 0.0))
        entry["reviewer_event_count"] = float(parent.get("reviewer_event_count", 0.0))
        entry["reviewer_trust_sum"] = float(parent.get("reviewer_trust_sum", 0.0))
        entry["reviewer_mean_trust"] = float(parent.get("reviewer_mean_trust", 0.5))
        entry["origin_node_id"] = str(parent.get("origin_node_id", ""))
        entry["origin_turn_index"] = int(parent.get("origin_turn_index", -1))
        entry["origin_role"] = str(parent.get("origin_role", ""))
        entry["candidate_bank_source"] = "phase2_unified_local_correction"
        entry["parent_candidate_digest"] = str(parent.get("digest", ""))
        entry["repair_branch"] = True
        entry["repair_agent_id"] = "phase2_unified_local_correction"
        entry["repair_round"] = int(repair_round)
        entry["repair_parent_digest"] = str(parent.get("digest", ""))
        entry["repair_operator_type"] = "unified_local_correction"
        entry["recovery_subgraph_node_ids"] = list(parent.get("recovery_subgraph_node_ids", ()))
        entry["recovery_subgraph_edge_ids"] = list(parent.get("recovery_subgraph_edge_ids", ()))
        entry["verifier_snapshot"] = dict(parent.get("verifier_snapshot", {}))
        trigger_state = parent.get("phase1_verifier_state")
        entry["trigger_verifier_snapshot"] = (
            dict(trigger_state.to_dict()) if isinstance(trigger_state, VerifierState) else {}
        )
        entry["recovery_reinserted"] = True
        entry["provenance"] = [dict(item) for item in parent.get("provenance", ()) if isinstance(item, dict)]
        entry["phase2_repair_round"] = int(repair_round)
        entry["phase2_repair_parent_digest"] = str(parent.get("digest", ""))
        return entry

    def _correction_value_features(
        self,
        *,
        entry: Dict[str, Any],
        prediction: DeltaPrediction,
        state: VerifierState,
    ) -> Dict[str, float]:
        return {
            "bias": 1.0,
            "expected_drop_mean": mean_expected_drop(prediction),
            "expected_confidence_gain": float(prediction.expected_confidence_gain),
            "expected_progress_gain": float(prediction.expected_progress_gain),
            "expected_preserve_risk": -float(prediction.expected_preserve_risk),
            "typed_support_score": float(state.typed_support_score),
            "answer_consistency": float(state.answer_consistency_score),
            "overturn_risk": -float(state.overturn_risk),
            "residual_mean": -float(mean_residual(state)),
        }

    def _heuristic_correction_value(self, *, prediction: DeltaPrediction, state: VerifierState) -> float:
        return max(
            0.0,
            min(
                1.0,
                0.34 * mean_expected_drop(prediction)
                + 0.18 * float(prediction.expected_confidence_gain)
                + 0.18 * float(prediction.expected_progress_gain)
                - 0.16 * float(prediction.expected_preserve_risk)
                - 0.08 * float(state.overturn_risk)
                + 0.12 * float(state.typed_support_score)
                + 0.10 * float(state.answer_consistency_score),
            ),
        )

    def _compute_final_utility(self, entry: Dict[str, Any]) -> float:
        safe_utility = float(entry.get("phase1_safe_utility", 0.0))
        correction_value = float(entry.get("phase2_correction_value", 0.0))
        return max(0.0, min(1.0, safe_utility + float(self.config.correction_value_weight) * correction_value))

    def _annotate_phase2_bundle(self, bundle: Dict[str, Any]) -> Dict[str, Any]:
        candidates = [dict(entry) for entry in bundle.get("candidates", [])]
        for entry in candidates:
            self._ensure_phase2_entry_fields(entry)
            state = entry.get("phase1_verifier_state")
            prediction_payload = dict(entry.get("phase2_delta_prediction", {}))
            if not isinstance(state, VerifierState) or not prediction_payload:
                entry["phase2_correction_value"] = float(entry.get("phase2_correction_value", 0.0))
                entry["phase2_correction_risk"] = float(entry.get("phase2_correction_risk", 0.0))
                entry["phase2_final_utility"] = self._compute_final_utility(entry)
                continue
            prediction = DeltaPrediction(
                expected_residual_drop={str(k): float(v) for k, v in dict(prediction_payload.get("expected_residual_drop", {})).items()},
                expected_confidence_gain=float(prediction_payload.get("expected_confidence_gain", 0.0)),
                expected_progress_gain=float(prediction_payload.get("expected_progress_gain", 0.0)),
                expected_preserve_risk=float(prediction_payload.get("expected_preserve_risk", 0.0)),
                rationale=str(prediction_payload.get("rationale", "")),
            )
            features = self._correction_value_features(entry=entry, prediction=prediction, state=state)
            predicted, _ = self._correction_value_model.predict(features)
            heuristic = self._heuristic_correction_value(prediction=prediction, state=state)
            entry["phase2_correction_value_features"] = features
            entry["phase2_correction_value"] = max(0.0, min(1.0, 0.70 * heuristic + 0.30 * float(predicted)))
            entry["phase2_correction_risk"] = float(prediction.expected_preserve_risk)
            entry["phase2_final_utility"] = self._compute_final_utility(entry)

        candidates.sort(
            key=lambda entry: (
                float(entry.get("phase2_final_utility", 0.0)),
                float(entry.get("phase2_correction_value", 0.0)),
                float(entry.get("phase1_safe_override_score", 0.0)),
                -float(entry.get("phase1_overturn_risk", 1.0)),
                float(entry.get("phase1_confidence_score", 0.0)),
                -float(entry.get("phase1_residual_mean", 1.0)),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )
        serialized = [self._serialize_candidate_entry(entry) for entry in candidates]
        bundle["candidates"] = candidates
        bundle["candidates_serialized"] = serialized
        bundle["anchor"] = self._find_anchor_candidate(candidates)
        bundle["anchor_serialized"] = self._find_anchor_candidate(serialized)
        return bundle

    def _apply_local_corrections(
        self,
        bundle: Dict[str, Any],
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        total_turns: int,
    ) -> Dict[str, Any]:
        bank: Dict[str, Dict[str, Any]] = {
            str(entry.get("text", "")): dict(entry)
            for entry in bundle.get("candidates", [])
            if str(entry.get("text", "")).strip()
        }
        correction_traces: List[Dict[str, Any]] = []
        attempt_count = 0
        accepted_count = 0
        improvement_count = 0
        current_bundle = bundle
        for repair_round in range(1, max(1, int(self.config.correction_max_rounds)) + 1):
            anchor = current_bundle.get("anchor")
            anchor_artifact = anchor.get("phase1_artifact") if isinstance(anchor, dict) else None
            anchor_state = anchor.get("phase1_verifier_state") if isinstance(anchor, dict) else None
            frontier = list(current_bundle.get("candidates", []))[: max(1, int(self.config.correction_frontier_k))]
            changed = False
            seen_digests = {str(item.get("digest", "")) for item in bank.values()}
            for candidate in frontier:
                self._ensure_phase2_entry_fields(candidate)
                state = candidate.get("phase1_verifier_state")
                artifact = candidate.get("phase1_artifact")
                if not isinstance(state, VerifierState) or not isinstance(artifact, ArtifactIR):
                    continue
                if mean_residual(state) <= float(self.config.correction_min_trigger_residual):
                    continue
                safe_utility = float(candidate.get("phase1_safe_utility", 0.0))
                gate = answer_first_gate(artifact, state, safe_utility=safe_utility)
                preserve_map = preserve_heatmap(artifact, state, anchor_artifact=anchor_artifact)
                localization_map = localize_units(
                    artifact,
                    state,
                    preserve_map=preserve_map,
                    answer_first_score=gate,
                    previous_delta=dict(candidate.get("phase2_delta_actual", {})),
                )
                critique = critique_summary(
                    artifact,
                    state,
                    localization_map=localization_map,
                    preserve_map=preserve_map,
                    answer_first_score=gate,
                    top_k=int(self.config.localizer_top_k),
                )
                correction = propose_correction(
                    artifact,
                    state,
                    critique=critique,
                    anchor_artifact=anchor_artifact,
                    anchor_state=anchor_state if isinstance(anchor_state, VerifierState) else None,
                )
                if correction is None:
                    continue
                attempt_count += 1
                prediction = predict_delta(state, correction=correction)
                features = self._correction_value_features(entry=candidate, prediction=prediction, state=state)
                predicted_value, _ = self._correction_value_model.predict(features)
                heuristic_value = self._heuristic_correction_value(prediction=prediction, state=state)
                correction_value = max(0.0, min(1.0, 0.70 * heuristic_value + 0.30 * float(predicted_value)))
                if correction_value < float(self.config.correction_min_value):
                    continue
                edited_artifact = apply_correction_artifact(artifact, correction)
                if not edited_artifact.rendered_answer.strip() or edited_artifact.rendered_answer == artifact.rendered_answer:
                    continue
                new_entry = self._copy_parent_entry(candidate, edited_artifact.rendered_answer, repair_round=repair_round)
                if str(new_entry.get("digest", "")) in seen_digests:
                    continue
                self._enrich_candidate_entry(
                    new_entry,
                    question_text=question_text,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                    total_turns=total_turns,
                    anchor_artifact=anchor_artifact,
                )
                new_state = new_entry.get("phase1_verifier_state")
                if not isinstance(new_state, VerifierState):
                    continue
                new_entry["phase2_answer_first_gate"] = float(gate)
                new_entry["phase2_localization_map"] = dict(localization_map)
                new_entry["phase2_preserve_map"] = dict(preserve_map)
                new_entry["phase2_critique"] = dict(critique)
                new_entry["phase2_correction_artifact"] = correction.to_dict()
                new_entry["phase2_delta_prediction"] = prediction.to_dict()
                new_entry["phase2_delta_actual"] = actual_delta(state, new_state)
                new_entry["phase2_correction_value_features"] = features
                new_entry["phase2_correction_value"] = correction_value
                new_entry["phase2_correction_risk"] = float(prediction.expected_preserve_risk)
                new_entry["phase2_repair_round"] = int(repair_round)
                new_entry["phase2_repair_parent_digest"] = str(candidate.get("digest", ""))
                new_entry["phase2_final_utility"] = self._compute_final_utility(new_entry)
                self._merge_candidate_entry(bank, new_entry)
                accepted_count += 1
                actual_improvement = _mean(new_entry["phase2_delta_actual"].values()) > 0.0
                if actual_improvement:
                    improvement_count += 1
                seen_digests.add(str(new_entry.get("digest", "")))
                correction_traces.append(
                    {
                        "repair_round": repair_round,
                        "parent_digest": str(candidate.get("digest", "")),
                        "child_digest": str(new_entry.get("digest", "")),
                        "answer_first_gate": float(gate),
                        "correction_value": float(correction_value),
                        "correction_risk": float(prediction.expected_preserve_risk),
                        "correction_artifact": dict(new_entry.get("phase2_correction_artifact", {})),
                        "delta_prediction": dict(new_entry.get("phase2_delta_prediction", {})),
                        "delta_actual": dict(new_entry.get("phase2_delta_actual", {})),
                    }
                )
                changed = True
            if not changed:
                break
            current_bundle = Phase1SemanticSafeOverrideRuntime._score_bank(
                self,
                bank,
                question_text=question_text,
                dataset_profile=dataset_profile,
                total_turns=total_turns,
                metadata=metadata,
            )
        rescored = Phase1SemanticSafeOverrideRuntime._score_bank(
            self,
            bank,
            question_text=question_text,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
            metadata=metadata,
        )
        rescored.update(
            {
                "phase2_correction_attempt_count": int(attempt_count),
                "phase2_correction_accept_count": int(accepted_count),
                "phase2_correction_improvement_count": int(improvement_count),
                "phase2_correction_traces": correction_traces,
            }
        )
        return self._annotate_phase2_bundle(rescored)

    def _candidate_bank_bundle(
        self,
        *,
        question_text: str,
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, Any]:
        base_bundle = Phase1SemanticSafeOverrideRuntime._candidate_bank_bundle(
            self,
            question_text=question_text,
            turn_traces=turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        total_turns = max(1, len(turn_traces))
        corrected = self._apply_local_corrections(
            base_bundle,
            question_text=question_text,
            metadata=metadata,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
        )
        corrected["occurrences"] = dict(base_bundle.get("occurrences", {}))
        corrected["phase1_pairwise_audit"] = list(base_bundle.get("phase1_pairwise_audit", []))
        return corrected

    def _select_final_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not candidates:
            if anchor is not None:
                self._phase1_last_residual_mean = float(anchor.get("phase1_residual_mean", 0.5))
                return anchor, "phase2_unified_local_correction_empty_bank_preserve_anchor"
            return None, "phase2_unified_local_correction_empty_bank"
        best = dict(candidates[0])
        if anchor is None:
            self._phase1_last_residual_mean = float(best.get("phase1_residual_mean", 0.5))
            return best, "phase2_unified_local_correction_no_anchor"
        challengers = [dict(entry) for entry in candidates if entry.get("digest") != anchor.get("digest")]
        if not challengers:
            self._phase1_last_residual_mean = float(anchor.get("phase1_residual_mean", 0.5))
            return anchor, "phase2_unified_local_correction_anchor_wins_frontier"
        anchor_final_utility = float(anchor.get("phase2_final_utility", anchor.get("phase1_safe_utility", 0.0)))
        anchor_residual = float(anchor.get("phase1_residual_mean", 1.0))
        challengers.sort(
            key=lambda entry: (
                float(entry.get("phase2_final_utility", entry.get("phase1_safe_utility", 0.0))) - anchor_final_utility,
                float(entry.get("phase2_correction_value", 0.0)),
                float(entry.get("phase1_safe_override_score", 0.0)),
                -float(entry.get("phase1_overturn_risk", 1.0)),
                float(entry.get("phase1_confidence_score", 0.0)),
                -float(entry.get("phase1_residual_mean", 1.0)),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )
        qualified: List[Dict[str, Any]] = []
        any_pairwise_better = False
        for challenger in challengers:
            challenger_final_utility = float(challenger.get("phase2_final_utility", challenger.get("phase1_safe_utility", 0.0)))
            if challenger_final_utility <= anchor_final_utility:
                continue
            any_pairwise_better = True
            if self._is_catastrophic_answer_rewrite(challenger, anchor):
                continue
            if float(challenger.get("phase1_overturn_risk", 1.0)) >= float(self.config.overturn_threshold):
                continue
            if float(challenger.get("phase1_answer_consistency_score", 0.0)) < float(self.config.catastrophic_consistency_threshold):
                continue
            if challenger.get("phase2_correction_artifact") and float(challenger.get("phase2_correction_risk", 1.0)) >= float(self.config.correction_risk_threshold):
                continue
            qualified.append(challenger)
        if qualified:
            selected = max(
                qualified,
                key=lambda entry: (
                    float(entry.get("phase2_final_utility", 0.0)),
                    float(entry.get("phase2_correction_value", 0.0)),
                    float(entry.get("phase1_safe_override_score", 0.0)),
                    -float(entry.get("phase1_overturn_risk", 1.0)),
                    float(entry.get("phase1_confidence_score", 0.0)),
                    -float(entry.get("phase1_residual_mean", 1.0)),
                    str(entry.get("digest", "")),
                ),
            )
            self._phase1_last_residual_mean = float(selected.get("phase1_residual_mean", anchor_residual))
            if selected.get("phase2_correction_artifact"):
                return selected, "phase2_unified_local_correction_override_with_correction"
            return selected, "phase2_unified_local_correction_override_frontier"
        if not any_pairwise_better:
            self._phase1_last_residual_mean = anchor_residual
            return anchor, "phase2_unified_local_correction_anchor_wins_frontier"
        self._phase1_last_residual_mean = anchor_residual
        return anchor, "phase2_unified_local_correction_preserve_anchor_guard"

    def _record_selection_metadata(
        self,
        *,
        candidates: Sequence[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        strategy: str,
        bundle: Dict[str, Any],
    ) -> None:
        top = [dict(item) for item in list(candidates)[: self.config.candidate_max_k]]
        selected_overturn = float((selected or {}).get("phase1_overturn_risk", 0.0))
        selected_consistency = float((selected or {}).get("phase1_answer_consistency_score", 0.0))
        selected_safe_override = float((selected or {}).get("phase1_safe_override_score", 0.0))
        selected_pairwise_value = 0.0
        if selected is not None and anchor is not None:
            selected_pairwise_value = float((selected or {}).get("phase2_final_utility", 0.0)) - float(anchor.get("phase2_final_utility", anchor.get("phase1_safe_utility", 0.0)))
        self._last_phase2_selection = {
            "stage2_version": self.config.stage2_version,
            "selection_reason": strategy,
            "stage1_anchor_present": bool(anchor is not None),
            "stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "selected_candidate_digest": str((selected or {}).get("digest", "")),
            "selected_candidate_source": self._candidate_source_label(selected or {}),
            "candidate_count": int(len(candidates)),
            "soft_class_count": int(len(bundle.get("phase1_soft_classes", []))),
            "phase1_selected_overturn_risk": selected_overturn,
            "phase1_selected_answer_consistency": selected_consistency,
            "phase1_selected_safe_override_score": selected_safe_override,
            "phase2_selected_correction_value": float((selected or {}).get("phase2_correction_value", 0.0)),
            "phase2_selected_final_utility": float((selected or {}).get("phase2_final_utility", 0.0)),
            "phase2_selected_correction_risk": float((selected or {}).get("phase2_correction_risk", 0.0)),
            "phase1_selected_pairwise_value": selected_pairwise_value,
            "phase2_correction_attempt_count": int(bundle.get("phase2_correction_attempt_count", 0)),
            "phase2_correction_accept_count": int(bundle.get("phase2_correction_accept_count", 0)),
            "phase2_correction_improvement_count": int(bundle.get("phase2_correction_improvement_count", 0)),
            "phase2_top_candidates": top,
            "phase1_top_classes": list(bundle.get("phase1_soft_classes", [])),
            "phase1_pairwise_audit": list(bundle.get("phase1_pairwise_audit", [])),
            "phase2_correction_traces": list(bundle.get("phase2_correction_traces", [])),
            "phase1_halt_mass": float(
                self._phase1_controller_snapshot.halt_mass if self._phase1_controller_snapshot is not None else 0.0
            ),
        }
        self._last_phase1_selection = dict(self._last_phase2_selection)

    def _graph_faithfulness_metadata(self, result: Stage2RunResult) -> Dict[str, Any]:
        metadata = super()._graph_faithfulness_metadata(result)
        selected_source = str(self._last_phase2_selection.get("selected_candidate_source", ""))
        selected_digest = str(self._last_phase2_selection.get("selected_candidate_digest", ""))
        selected_entry = None
        for entry in self._last_candidate_bundle.get("candidates", []):
            if str(entry.get("digest", "")) == selected_digest:
                selected_entry = entry
                break
        metadata.update(
            {
                "graph_faithfulness_final_answer_source_type": selected_source,
                "graph_faithfulness_phase2_selected_correction_value": float(
                    (selected_entry or {}).get("phase2_correction_value", 0.0)
                ),
            }
        )
        return metadata

    def run(
        self,
        graph: UnionGraph,
        *,
        question_text: str,
        metadata: Optional[dict] = None,
        reference_answer: Optional[str] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        replay_dir: Optional[str] = None,
        learn: bool = False,
    ) -> Stage2RunResult:
        self._last_phase1_selection = {}
        self._last_phase2_selection = {}
        self._last_candidate_bundle = {}
        self._phase1_current_graph = graph
        self._phase1_controller_snapshot = None
        self._phase1_last_residual_mean = 0.5
        self._phase1_last_utility_delta = 0.0
        result = Stage2RuntimeV2.run(
            self,
            graph,
            question_text=question_text,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
            replay_dir=replay_dir,
            learn=learn,
        )
        if result.signature.startswith("stage2_v2|"):
            result.signature = "stage2_phase2_unified_local_correction|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_phase2_selection)
        result.metadata.update(self._graph_faithfulness_metadata(result))
        result.metadata["stage2_version"] = self.config.stage2_version
        result.metadata["phase2_protocol_family"] = "semantic_verification_local_correction"
        return result

    def save_replay_bundle(
        self,
        result: Stage2RunResult,
        replay_dir: str,
        *,
        question_text: str,
        metadata: Optional[dict],
    ) -> None:
        Stage2RuntimeV2.save_replay_bundle(self, result, replay_dir, question_text=question_text, metadata=metadata)
        with open(os.path.join(replay_dir, "phase2_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_phase2_selection, handle, ensure_ascii=False, indent=2)

    def learn_from_run(
        self,
        graph: UnionGraph,
        result: Stage2RunResult,
        *,
        dataset_profile: DatasetProfile,
        summary=None,
        reward_target: Optional[float] = None,
        question_text: Optional[str] = None,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, float]:
        stats = super().learn_from_run(
            graph,
            result,
            dataset_profile=dataset_profile,
            summary=summary,
            reward_target=reward_target,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        if not self.config.learning.enabled or not question_text:
            stats.update({"correction_value_updates": 0.0, "correction_value_steps": float(self._correction_value_model.steps)})
            return stats

        bundle = self._last_candidate_bundle or {}
        candidates = list(bundle.get("candidates", []))
        if not candidates:
            stats.update({"correction_value_updates": 0.0, "correction_value_steps": float(self._correction_value_model.steps)})
            return stats

        eval_cache: Dict[str, Dict[str, float]] = {}
        for entry in candidates:
            digest = str(entry.get("digest", ""))
            if not digest or digest in eval_cache:
                continue
            eval_cache[digest] = self._evaluate_candidate_summary(
                question_text=question_text,
                candidate_text=str(entry.get("text", "")),
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )

        by_digest = {str(entry.get("digest", "")): entry for entry in candidates}
        updates = 0
        for entry in candidates:
            features = entry.get("phase2_correction_value_features", {})
            if not isinstance(features, dict) or not features:
                continue
            digest = str(entry.get("digest", ""))
            parent_digest = str(entry.get("phase2_repair_parent_digest", "") or entry.get("parent_candidate_digest", ""))
            candidate_eval = eval_cache.get(digest, {"success": 0.0, "task_score": 0.0})
            parent_eval = eval_cache.get(parent_digest)
            if parent_eval is None:
                continue
            target = self._lexicographic_preference(candidate_eval, parent_eval)
            self._correction_value_model.update({str(name): float(value) for name, value in features.items()}, target)
            updates += 1

        stats.update(
            {
                "correction_value_updates": float(updates),
                "correction_value_steps": float(self._correction_value_model.steps),
            }
        )
        return stats
