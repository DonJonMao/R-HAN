from __future__ import annotations

import json
import os
import re
from dataclasses import asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mas_stage2.learning import OnlineLinearModel
from mas_stage2.types import ControllerState, EdgeActivation, ExportedMemoryMessage, FeedbackEvent, Stage2RunResult, TurnTrace
from mas_treesearch.prompting import build_system_prompt
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import PromptSlots, UnionGraph, UnionNode
from stage2_gcr_plus.runtime_v2 import Stage2RuntimeV2
from stage2_gcr_plus.runtime_v41 import Stage2RuntimeV41

from .artifacts import ArtifactIR, clamp01, cosine_similarity, lexical_feature_vector, pooled_artifact_vector, softmax
from .config import Phase3aUnifiedConfig
from .controller import UnifiedControllerSnapshot, build_controller_state
from .correction import (
    CorrectionArtifact,
    DeltaPrediction,
    actual_delta,
    apply_correction_artifact,
    delta_match_score,
    deterministic_critique,
    deterministic_proposal,
    localize_units,
    predict_delta,
    preserve_heatmap,
)
from .prompts import (
    PROMPT_ARTIFACT_PROPOSAL,
    PROMPT_CRITIQUE,
    PROMPT_DELTA,
    build_prompt_payload,
)
from .verifier import VerifierState, mean_residual, verify_artifact
from .artifacts import canonicalize_candidate, sparsemax


def _extract_json_object(raw_text: str) -> Optional[Dict[str, Any]]:
    text = str(raw_text or "").strip()
    if not text:
        return None
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start : end + 1])
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


class Phase3aUnifiedRuntime(Stage2RuntimeV41):
    def __init__(self, config: Phase3aUnifiedConfig, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._utility_model = OnlineLinearModel(learning_rate=config.utility_learning_rate)
        self._delta_model = OnlineLinearModel(learning_rate=config.delta_learning_rate)
        self._halt_model = OnlineLinearModel(learning_rate=config.halt_learning_rate)
        self._last_phase3a_selection: Dict[str, Any] = {}
        self._last_candidate_bundle: Dict[str, Any] = {}
        self._phase3a_current_graph: Optional[UnionGraph] = None
        self._phase3a_controller_snapshot: Optional[UnifiedControllerSnapshot] = None
        self._phase3a_last_residual_mean: float = 0.5
        self._phase3a_last_utility_delta: float = 0.0

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.update(
            {
                "phase3a_utility_model": self._utility_model.state_dict(),
                "phase3a_delta_model": self._delta_model.state_dict(),
                "phase3a_halt_model": self._halt_model.state_dict(),
            }
        )
        return payload

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        if isinstance(state.get("phase3a_utility_model"), dict):
            self._utility_model.load_state_dict(state["phase3a_utility_model"])
        if isinstance(state.get("phase3a_delta_model"), dict):
            self._delta_model.load_state_dict(state["phase3a_delta_model"])
        if isinstance(state.get("phase3a_halt_model"), dict):
            self._halt_model.load_state_dict(state["phase3a_halt_model"])

    @staticmethod
    def _json_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="critique_then_answer",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="strict",
            finalization="answer_only",
        )

    def _resolve_agent_id(self, preferred: str, *fallbacks: str) -> Optional[str]:
        candidates = [preferred, *fallbacks, "verifier", "critic", "reviser", "aggregator", "judge"]
        for agent_id in candidates:
            if agent_id and agent_id in self._by_id:
                return agent_id
        return None

    def _prompt_json(
        self,
        *,
        agent_id: str,
        instruction: str,
        question_text: str,
        metadata: Optional[dict],
        payload: Dict[str, Any],
        dataset_profile: DatasetProfile,
        extra_role_hint: str,
    ) -> Optional[Dict[str, Any]]:
        agent = self._by_id.get(agent_id)
        if agent is None:
            return None
        system_prompt = build_system_prompt(agent, self._json_slots(), extra_role_hint=extra_role_hint)
        user_prompt = instruction + "\n\n" + build_prompt_payload(question_text=question_text, metadata=metadata, payload=payload)
        raw = self.evaluator._cached_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
        )
        return _extract_json_object(raw)

    def _ensure_phase3a_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_v4_entry_fields(entry)
        self._ensure_v4_3_entry_fields(entry)
        entry.setdefault("phase3a_artifact", None)
        entry.setdefault("phase3a_verifier_state", None)
        entry.setdefault("phase3a_support_mean", 0.0)
        entry.setdefault("phase3a_raw_utility", 0.0)
        entry.setdefault("phase3a_adjusted_utility", 0.0)
        entry.setdefault("phase3a_frontier_weight", 0.0)
        entry.setdefault("phase3a_utility_features", {})
        entry.setdefault("phase3a_residual_mean", 1.0)
        entry.setdefault("phase3a_anchor_similarity", 0.0)
        entry.setdefault("phase3a_confidence_score", 0.0)
        entry.setdefault("phase3a_correction_artifact", {})
        entry.setdefault("phase3a_delta_prediction", {})
        entry.setdefault("phase3a_delta_actual", {})
        entry.setdefault("phase3a_delta_match", 0.0)
        entry.setdefault("phase3a_soft_class_id", -1)
        entry.setdefault("phase3a_soft_class_size", 1)

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        verifier_state = entry.get("phase3a_verifier_state")
        payload.update(
            {
                "phase3a_raw_utility": float(entry.get("phase3a_raw_utility", 0.0)),
                "phase3a_adjusted_utility": float(entry.get("phase3a_adjusted_utility", 0.0)),
                "phase3a_frontier_weight": float(entry.get("phase3a_frontier_weight", 0.0)),
                "phase3a_support_mean": float(entry.get("phase3a_support_mean", 0.0)),
                "phase3a_residual_mean": float(entry.get("phase3a_residual_mean", 0.0)),
                "phase3a_confidence_score": float(entry.get("phase3a_confidence_score", 0.0)),
                "phase3a_anchor_similarity": float(entry.get("phase3a_anchor_similarity", 0.0)),
                "phase3a_verifier_state": verifier_state.to_dict() if isinstance(verifier_state, VerifierState) else {},
                "phase3a_correction_artifact": dict(entry.get("phase3a_correction_artifact", {})),
                "phase3a_delta_prediction": dict(entry.get("phase3a_delta_prediction", {})),
                "phase3a_delta_actual": dict(entry.get("phase3a_delta_actual", {})),
                "phase3a_delta_match": float(entry.get("phase3a_delta_match", 0.0)),
                "phase3a_soft_class_id": int(entry.get("phase3a_soft_class_id", -1)),
                "phase3a_soft_class_size": int(entry.get("phase3a_soft_class_size", 1)),
            }
        )
        payload.pop("text", None)
        return payload

    def _build_turn_state(
        self,
        *,
        turn_index: int,
        total_turns: int,
        previous_feedback: Sequence[FeedbackEvent],
        active_edges: Sequence[EdgeActivation],
    ) -> ControllerState:
        graph = self._phase3a_current_graph
        task_nodes = self._task_nodes(graph) if graph is not None else []
        state, snapshot = build_controller_state(
            graph=graph,
            task_nodes=task_nodes,
            previous_feedback=previous_feedback,
            previous_snapshot=self._phase3a_controller_snapshot,
            turn_index=turn_index,
            total_turns=total_turns,
            last_residual_mean=self._phase3a_last_residual_mean,
            last_utility_delta=self._phase3a_last_utility_delta,
            active_edges=active_edges,
            node_top_k=int(self.config.controller_node_top_k),
        )
        self._phase3a_controller_snapshot = snapshot
        return state

    @staticmethod
    def _memory_view_for_record(record) -> str:
        feedback_type = str(getattr(record, "feedback_type", "")).strip().lower()
        record_type = str(getattr(record, "record_type", "")).strip().lower()
        if feedback_type in {"pass", "preserve", "keep"}:
            return "stable"
        if feedback_type in {"challenge", "reject", "conflict", "revise"}:
            return "failure"
        if "feedback" in record_type or "provenance" in record_type:
            return "provenance"
        return "global"

    def _prepare_turn_packages(
        self,
        task_nodes: Sequence[UnionNode],
        *,
        question_text: str,
        turn_state: ControllerState,
        current_turn: int,
    ) -> Dict[str, Dict[str, object]]:
        prepared: Dict[str, Dict[str, object]] = {}
        attention = dict(turn_state.metadata.get("phase3a_memory_view_attention", {}))
        for node in task_nodes:
            local_records = self._memory_store.get(node.node_id)
            records_by_id = {record.record_id: record for record in local_records}
            selected_items = self._selector.select(
                node,
                question_text,
                turn_state,
                local_records,
                current_turn=current_turn,
            )
            if selected_items:
                selected_items = sorted(
                    selected_items,
                    key=lambda item: (
                        float(item.score) + 0.25 * float(attention.get(self._memory_view_for_record(records_by_id.get(item.record_id)), 0.0)),
                        item.record_id,
                    ),
                    reverse=True,
                )
            local_latent = self._compose_latent(node, question_text, turn_state, selected_items, records_by_id)
            prepared[node.node_id] = {
                "records_by_id": records_by_id,
                "selected_items": selected_items,
                "local_latent": local_latent,
            }
        return prepared

    def _activate_edges_v2(
        self,
        graph: UnionGraph,
        prepared_states: Dict[str, Dict[str, object]],
        *,
        turn_index: int,
    ) -> List[EdgeActivation]:
        activations = Stage2RuntimeV2._activate_edges_v2(self, graph, prepared_states, turn_index=turn_index)
        participation = dict(
            self._phase3a_controller_snapshot.node_participation if self._phase3a_controller_snapshot is not None else {}
        )
        for activation in activations:
            beta = clamp01(
                0.60 * float(activation.score)
                + 0.20 * float(participation.get(activation.src, 0.0))
                + 0.20 * float(participation.get(activation.dst, 0.0))
            )
            activation.metadata["phase3a_edge_support"] = beta
            if beta >= float(self.config.controller_edge_threshold):
                activation.active = True
            activation.reason = f"{activation.reason},phase3a_beta={beta:.3f}"
        return activations

    def _active_task_nodes_v2(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
    ) -> List[UnionNode]:
        participation = dict(
            self._phase3a_controller_snapshot.node_participation if self._phase3a_controller_snapshot is not None else {}
        )
        active_incident = {
            node_id
            for activation in active_edges
            if activation.active
            for node_id in (activation.src, activation.dst)
        }
        protected_ids = {
            node_id
            for node_id in set(graph.root_node_ids) | set(graph.sink_node_ids)
            if node_id in graph.nodes and graph.nodes[node_id].node_type == "task"
        }
        selected = [
            node
            for node in task_nodes
            if float(participation.get(node.node_id, 0.0)) > 0.0
            or node.node_id in protected_ids
            or node.node_id in active_incident
        ]
        if selected:
            return selected
        return Stage2RuntimeV2._active_task_nodes_v2(self, graph, task_nodes, active_edges)

    def _candidate_anchor_artifact(self, metadata: Optional[dict], dataset_profile: DatasetProfile) -> Optional[ArtifactIR]:
        anchor_text = str((metadata or {}).get("stage1_anchor_output", "")).strip()
        if not anchor_text:
            return None
        return canonicalize_candidate(
            candidate_text=anchor_text,
            provenance=(),
            metadata=metadata,
            task_type=dataset_profile.task_type,
        )

    def _utility_features(
        self,
        *,
        entry: Dict[str, Any],
        verifier_state: VerifierState,
        artifact: ArtifactIR,
        anchor_similarity: float,
        total_turns: int,
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        return {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            "support_mean": float(_mean(verifier_state.support_map.values())),
            "confidence_score": float(verifier_state.confidence_score),
            "progress_score": float(verifier_state.progress_score),
            "anchor_similarity": float(anchor_similarity),
            "stage1_anchor": 1.0 if entry.get("stage1_anchor") else 0.0,
            "occurrence_log": float(entry.get("occurrence_count", 0.0)),
            "sink_support": float(entry.get("sink_support", 0.0)),
            "reviewer_mean_trust": float(entry.get("reviewer_mean_trust", 0.5)),
            "residual_parse": -float(verifier_state.residual_vector.get("r_parse", 0.0)),
            "residual_consistency": -float(verifier_state.residual_vector.get("r_consistency", 0.0)),
            "residual_completeness": -float(verifier_state.residual_vector.get("r_completeness", 0.0)),
            "residual_execution": -float(verifier_state.residual_vector.get("r_execution", 0.0)),
            "residual_constraint": -float(verifier_state.residual_vector.get("r_constraint", 0.0)),
            "residual_support": -float(verifier_state.residual_vector.get("r_support", 0.0)),
            "residual_preserve": -float(verifier_state.residual_vector.get("r_preserve", 0.0)),
            "unit_count": float(len(artifact.units)) / float(max(1, total_turns)),
        }

    def _heuristic_utility(
        self,
        *,
        entry: Dict[str, Any],
        verifier_state: VerifierState,
        anchor_similarity: float,
        artifact: ArtifactIR,
    ) -> float:
        support_mean = _mean(verifier_state.support_map.values())
        residual_mean = mean_residual(verifier_state)
        provenance_coverage = float(artifact.metadata.get("provenance_coverage", 0.0))
        anchor_bonus = 0.05 if entry.get("stage1_anchor") else 0.0
        return clamp01(
            0.42 * (1.0 - residual_mean)
            + 0.18 * verifier_state.confidence_score
            + 0.16 * support_mean
            + 0.12 * provenance_coverage
            + 0.07 * anchor_similarity
            + anchor_bonus
        )

    def _soft_cluster_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parent = list(range(len(candidates)))
        vectors = [pooled_artifact_vector(entry["phase3a_artifact"]) for entry in candidates]

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def union(left: int, right: int) -> None:
            left_root = find(left)
            right_root = find(right)
            if left_root != right_root:
                parent[right_root] = left_root

        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                if cosine_similarity(vectors[i], vectors[j]) >= float(self.config.similarity_threshold):
                    union(i, j)

        clusters: Dict[int, List[int]] = {}
        for index in range(len(candidates)):
            clusters.setdefault(find(index), []).append(index)

        summaries: List[Dict[str, Any]] = []
        for cluster_id, members in clusters.items():
            members_sorted = sorted(
                members,
                key=lambda idx: (
                    float(candidates[idx].get("phase3a_adjusted_utility", 0.0)),
                    str(candidates[idx].get("digest", "")),
                ),
                reverse=True,
            )
            representative = candidates[members_sorted[0]]
            for member_index in members:
                candidates[member_index]["phase3a_soft_class_id"] = int(cluster_id)
                candidates[member_index]["phase3a_soft_class_size"] = int(len(members))
            summaries.append(
                {
                    "class_id": int(cluster_id),
                    "size": int(len(members)),
                    "representative_digest": str(representative.get("digest", "")),
                    "representative_source": self._candidate_source_label(representative),
                    "representative_utility": float(representative.get("phase3a_adjusted_utility", 0.0)),
                    "contains_anchor": any(bool(candidates[idx].get("stage1_anchor", False)) for idx in members),
                }
            )
        summaries.sort(key=lambda item: (item["representative_utility"], item["size"]), reverse=True)
        return summaries

    def _enrich_candidate_entry(
        self,
        entry: Dict[str, Any],
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        total_turns: int,
        anchor_artifact: Optional[ArtifactIR],
    ) -> None:
        self._ensure_phase3a_entry_fields(entry)
        artifact = canonicalize_candidate(
            candidate_text=str(entry.get("text", "")),
            provenance=entry.get("provenance", ()),
            metadata=metadata,
            task_type=dataset_profile.task_type,
        )
        verifier_state = verify_artifact(
            artifact,
            question_text=question_text,
            metadata=metadata,
            task_type=dataset_profile.task_type,
            candidate_entry=entry,
            anchor_artifact=anchor_artifact,
            timeout_s=float(self.config.code_timeout_s),
            max_failed_examples=int(self.config.code_max_failed_examples),
        )
        anchor_similarity = 0.0
        if anchor_artifact is not None:
            anchor_similarity = clamp01((cosine_similarity(pooled_artifact_vector(artifact), pooled_artifact_vector(anchor_artifact)) + 1.0) * 0.5)
        features = self._utility_features(
            entry=entry,
            verifier_state=verifier_state,
            artifact=artifact,
            anchor_similarity=anchor_similarity,
            total_turns=total_turns,
            dataset_profile=dataset_profile,
        )
        predicted, uncertainty = self._utility_model.predict(features)
        heuristic = self._heuristic_utility(
            entry=entry,
            verifier_state=verifier_state,
            anchor_similarity=anchor_similarity,
            artifact=artifact,
        )
        raw_utility = clamp01(0.75 * heuristic + 0.25 * float(predicted))
        entry["phase3a_artifact"] = artifact
        entry["phase3a_verifier_state"] = verifier_state
        entry["phase3a_support_mean"] = _mean(verifier_state.support_map.values())
        entry["phase3a_utility_features"] = features
        entry["phase3a_raw_utility"] = raw_utility
        entry["phase3a_adjusted_utility"] = raw_utility
        entry["phase3a_residual_mean"] = mean_residual(verifier_state)
        entry["phase3a_confidence_score"] = verifier_state.confidence_score
        entry["phase3a_anchor_similarity"] = anchor_similarity
        entry["candidate_model_score"] = raw_utility
        entry["candidate_model_uncertainty"] = float(uncertainty)
        entry["support_score"] = raw_utility

    def _score_bank(
        self,
        bank: Dict[str, Dict[str, Any]],
        *,
        question_text: str,
        dataset_profile: DatasetProfile,
        total_turns: int,
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        anchor_artifact = self._candidate_anchor_artifact(metadata, dataset_profile)
        for entry in bank.values():
            if dataset_profile.task_type == "code_generation" and entry.get("parse_ok") is None:
                parse_ok, entry_point_ok = self._code_candidate_checks(entry["text"], metadata=metadata)
                entry["parse_ok"] = parse_ok
                entry["entry_point_ok"] = entry_point_ok
            if float(entry.get("reviewer_event_count", 0.0)) > 0.0:
                entry["reviewer_mean_trust"] = float(entry.get("reviewer_trust_sum", 0.0)) / float(entry["reviewer_event_count"])
            self._enrich_candidate_entry(
                entry,
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                total_turns=total_turns,
                anchor_artifact=anchor_artifact,
            )

        candidates = [dict(entry) for entry in bank.values()]
        vectors = [pooled_artifact_vector(entry["phase3a_artifact"]) for entry in candidates]
        raw_utilities = [float(entry.get("phase3a_raw_utility", 0.0)) for entry in candidates]
        frontier_prior = softmax(raw_utilities, temperature=float(self.config.redundancy_temperature))
        for index, entry in enumerate(candidates):
            penalty = 0.0
            for peer_index, peer_weight in enumerate(frontier_prior):
                if peer_index == index:
                    continue
                penalty += float(peer_weight) * max(0.0, cosine_similarity(vectors[index], vectors[peer_index]))
            entry["phase3a_adjusted_utility"] = float(entry["phase3a_raw_utility"]) - float(self.config.redundancy_gamma) * penalty

        candidates.sort(
            key=lambda entry: (
                float(entry.get("phase3a_adjusted_utility", 0.0)),
                float(entry.get("phase3a_confidence_score", 0.0)),
                -float(entry.get("phase3a_residual_mean", 1.0)),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )
        adjusted_utilities = [float(entry.get("phase3a_adjusted_utility", 0.0)) for entry in candidates]
        frontier_weights = softmax(adjusted_utilities, temperature=float(self.config.redundancy_temperature))
        for entry, weight in zip(candidates, frontier_weights):
            entry["phase3a_frontier_weight"] = float(weight)
        cluster_summary = self._soft_cluster_candidates(candidates)
        serialized = [self._serialize_candidate_entry(entry) for entry in candidates]
        return {
            "candidates": candidates,
            "candidates_serialized": serialized,
            "anchor": self._find_anchor_candidate(candidates),
            "anchor_serialized": self._find_anchor_candidate(serialized),
            "phase3a_soft_classes": cluster_summary,
        }

    def _copy_parent_entry(self, parent: Dict[str, Any], new_text: str, *, repair_round: int) -> Dict[str, Any]:
        entry = self._init_candidate_entry(new_text)
        self._ensure_phase3a_entry_fields(entry)
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
        entry["candidate_bank_source"] = "unified_correction_artifact"
        entry["parent_candidate_digest"] = str(parent.get("digest", ""))
        entry["repair_branch"] = True
        entry["repair_agent_id"] = "phase3a_unified"
        entry["repair_round"] = int(repair_round)
        entry["repair_parent_digest"] = str(parent.get("digest", ""))
        entry["repair_operator_type"] = "unified_local_correction"
        entry["recovery_subgraph_node_ids"] = list(parent.get("recovery_subgraph_node_ids", ()))
        entry["recovery_subgraph_edge_ids"] = list(parent.get("recovery_subgraph_edge_ids", ()))
        entry["verifier_snapshot"] = dict(parent.get("verifier_snapshot", {}))
        entry["trigger_verifier_snapshot"] = dict(parent.get("phase3a_verifier_state", VerifierState({}, {}, {}, 0, 0, 0, 0, 0, 0)).to_dict() if isinstance(parent.get("phase3a_verifier_state"), VerifierState) else {})
        entry["recovery_reinserted"] = True
        entry["provenance"] = [dict(item) for item in parent.get("provenance", ()) if isinstance(item, dict)]
        return entry

    def _maybe_llm_critique(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        artifact: ArtifactIR,
        verifier_state: VerifierState,
        localization_map: Dict[str, float],
        preserve_map: Dict[str, float],
    ) -> Optional[Dict[str, Any]]:
        agent_id = self._resolve_agent_id(self.config.critic_agent_id, self.config.verifier_agent_id)
        if agent_id is None:
            return None
        payload = {
            "artifact": artifact.to_dict(),
            "verifier_state": verifier_state.to_dict(),
            "unit_heatmap": localization_map,
            "preserve_heatmap": preserve_map,
        }
        return self._prompt_json(
            agent_id=agent_id,
            instruction=PROMPT_CRITIQUE,
            question_text=question_text,
            metadata=metadata,
            payload=payload,
            dataset_profile=dataset_profile,
            extra_role_hint="phase3a_critic",
        )

    def _maybe_llm_correction(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        artifact: ArtifactIR,
        critique: Dict[str, Any],
    ) -> Optional[CorrectionArtifact]:
        agent_id = self._resolve_agent_id(self.config.editor_agent_id, self.config.critic_agent_id)
        if agent_id is None:
            return None
        payload = {
            "artifact": artifact.to_dict(),
            "critique": critique,
        }
        raw = self._prompt_json(
            agent_id=agent_id,
            instruction=PROMPT_ARTIFACT_PROPOSAL,
            question_text=question_text,
            metadata=metadata,
            payload=payload,
            dataset_profile=dataset_profile,
            extra_role_hint="phase3a_editor",
        )
        if not isinstance(raw, dict):
            return None
        operation = str(raw.get("operation", "")).strip()
        if operation not in {"replace", "insert_before", "insert_after", "delete", "reorder"}:
            return None
        return CorrectionArtifact(
            target_units=[str(item) for item in raw.get("target_units", []) if str(item).strip()],
            operation=operation,
            new_units=[str(item) for item in raw.get("new_units", []) if str(item).strip()],
            preserve_units=[str(item) for item in raw.get("preserve_units", []) if str(item).strip()],
            expected_delta=dict(raw.get("expected_delta", {})),
            rationale=str(raw.get("rationale", "")),
        )

    def _maybe_llm_delta(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        original_artifact: ArtifactIR,
        edited_artifact: ArtifactIR,
        original_state: VerifierState,
        correction: CorrectionArtifact,
    ) -> Optional[DeltaPrediction]:
        agent_id = self._resolve_agent_id(self.config.delta_agent_id, self.config.verifier_agent_id)
        if agent_id is None:
            return None
        payload = {
            "original_artifact": original_artifact.to_dict(),
            "edited_artifact": edited_artifact.to_dict(),
            "original_state": original_state.to_dict(),
            "correction_artifact": correction.to_dict(),
        }
        raw = self._prompt_json(
            agent_id=agent_id,
            instruction=PROMPT_DELTA,
            question_text=question_text,
            metadata=metadata,
            payload=payload,
            dataset_profile=dataset_profile,
            extra_role_hint="phase3a_delta",
        )
        if not isinstance(raw, dict):
            return None
        return DeltaPrediction(
            expected_residual_drop={str(key): float(value) for key, value in dict(raw.get("expected_residual_drop", {})).items()},
            expected_confidence_gain=float(raw.get("expected_confidence_gain", 0.0)),
            expected_preserve_risk=float(raw.get("expected_preserve_risk", 0.0)),
            expected_frontier_shift=float(raw.get("expected_frontier_shift", 0.0)),
            rationale=str(raw.get("rationale", "")),
        )

    def _apply_unified_corrections(
        self,
        bank: Dict[str, Dict[str, Any]],
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        total_turns: int,
    ) -> Dict[str, Any]:
        bundle = self._score_bank(
            bank,
            question_text=question_text,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
            metadata=metadata,
        )
        correction_traces: List[Dict[str, Any]] = []
        attempt_count = 0
        accepted_count = 0
        improvement_count = 0
        last_utility_delta = 0.0
        for repair_round in range(1, max(1, int(self.config.correction_max_rounds)) + 1):
            anchor = bundle.get("anchor")
            anchor_artifact = anchor.get("phase3a_artifact") if isinstance(anchor, dict) else None
            frontier = list(bundle.get("candidates", []))[: max(1, int(self.config.correction_frontier_k))]
            changed = False
            for candidate in frontier:
                state = candidate.get("phase3a_verifier_state")
                artifact = candidate.get("phase3a_artifact")
                if not isinstance(state, VerifierState) or not isinstance(artifact, ArtifactIR):
                    continue
                if mean_residual(state) <= 0.18:
                    continue
                preserve_map = preserve_heatmap(artifact, state, anchor_artifact=anchor_artifact)
                localization_map = localize_units(
                    artifact,
                    state,
                    preserve_map=preserve_map,
                    previous_delta={},
                )
                critique = deterministic_critique(
                    artifact,
                    state,
                    localization_map=localization_map,
                    preserve_map=preserve_map,
                    top_k=int(self.config.localizer_top_k),
                )
                if self.config.use_prompt_correction:
                    llm_critique = self._maybe_llm_critique(
                        question_text=question_text,
                        metadata=metadata,
                        dataset_profile=dataset_profile,
                        artifact=artifact,
                        verifier_state=state,
                        localization_map=localization_map,
                        preserve_map=preserve_map,
                    )
                    if isinstance(llm_critique, dict):
                        critique = llm_critique
                correction = deterministic_proposal(
                    artifact,
                    state,
                    critique=critique,
                    anchor_artifact=anchor_artifact,
                    task_type=dataset_profile.task_type,
                )
                if self.config.use_prompt_correction:
                    llm_correction = self._maybe_llm_correction(
                        question_text=question_text,
                        metadata=metadata,
                        dataset_profile=dataset_profile,
                        artifact=artifact,
                        critique=critique,
                    )
                    if isinstance(llm_correction, CorrectionArtifact):
                        correction = llm_correction
                if correction is None:
                    continue
                attempt_count += 1
                edited_artifact = apply_correction_artifact(artifact, correction)
                if not edited_artifact.rendered_answer.strip() or edited_artifact.rendered_answer == artifact.rendered_answer:
                    continue
                delta_prediction = predict_delta(state, correction=correction)
                if self.config.use_prompt_delta:
                    llm_delta = self._maybe_llm_delta(
                        question_text=question_text,
                        metadata=metadata,
                        dataset_profile=dataset_profile,
                        original_artifact=artifact,
                        edited_artifact=edited_artifact,
                        original_state=state,
                        correction=correction,
                    )
                    if isinstance(llm_delta, DeltaPrediction):
                        delta_prediction = llm_delta
                new_entry = self._copy_parent_entry(candidate, edited_artifact.rendered_answer, repair_round=repair_round)
                new_entry["phase3a_correction_artifact"] = correction.to_dict()
                new_entry["phase3a_delta_prediction"] = delta_prediction.to_dict()
                self._enrich_candidate_entry(
                    new_entry,
                    question_text=question_text,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                    total_turns=total_turns,
                    anchor_artifact=anchor_artifact,
                )
                new_state = new_entry["phase3a_verifier_state"]
                if not isinstance(new_state, VerifierState):
                    continue
                new_entry["phase3a_delta_actual"] = actual_delta(state, new_state)
                new_entry["phase3a_delta_match"] = delta_match_score(delta_prediction, state, new_state)
                utility_delta = float(new_entry.get("phase3a_raw_utility", 0.0)) - float(candidate.get("phase3a_raw_utility", 0.0))
                residual_delta = float(candidate.get("phase3a_residual_mean", 1.0)) - float(new_entry.get("phase3a_residual_mean", 1.0))
                accept = (
                    residual_delta >= float(self.config.correction_accept_delta)
                    or utility_delta >= float(self.config.correction_accept_delta)
                )
                correction_traces.append(
                    {
                        "repair_round": repair_round,
                        "parent_digest": str(candidate.get("digest", "")),
                        "child_digest": str(new_entry.get("digest", "")),
                        "accepted": bool(accept),
                        "utility_delta": utility_delta,
                        "residual_delta": residual_delta,
                        "correction_artifact": dict(new_entry.get("phase3a_correction_artifact", {})),
                        "delta_prediction": dict(new_entry.get("phase3a_delta_prediction", {})),
                        "delta_actual": dict(new_entry.get("phase3a_delta_actual", {})),
                    }
                )
                if not accept:
                    continue
                self._merge_candidate_entry(bank, new_entry)
                accepted_count += 1
                if utility_delta > 0.0 or residual_delta > 0.0:
                    improvement_count += 1
                last_utility_delta = utility_delta
                changed = True
            if not changed:
                break
            bundle = self._score_bank(
                bank,
                question_text=question_text,
                dataset_profile=dataset_profile,
                total_turns=total_turns,
                metadata=metadata,
            )
        bundle.update(
            {
                "phase3a_correction_attempt_count": int(attempt_count),
                "phase3a_correction_accept_count": int(accepted_count),
                "phase3a_correction_improvement_count": int(improvement_count),
                "phase3a_correction_traces": correction_traces,
                "phase3a_last_utility_delta": float(last_utility_delta),
            }
        )
        self._phase3a_last_utility_delta = float(last_utility_delta)
        return bundle

    def _candidate_bank_bundle(
        self,
        *,
        question_text: str,
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, Any]:
        bank: Dict[str, Dict[str, Any]] = {}
        occurrence_map: Dict[Tuple[int, str], Dict[str, Any]] = {}
        feedback_by_occurrence = self._feedback_events_by_target_occurrence(turn_traces)
        stage1_output = str((metadata or {}).get("stage1_anchor_output", "")).strip()
        total_turns = max(1, len(turn_traces))

        def ensure_entry(text: str) -> Optional[Dict[str, Any]]:
            normalized = self._sanitize_candidate(question_text, text, metadata=metadata)
            if not normalized:
                return None
            entry = bank.get(normalized)
            if entry is None:
                entry = self._init_candidate_entry(normalized)
                self._ensure_phase3a_entry_fields(entry)
                bank[normalized] = entry
            return entry

        for turn_trace in turn_traces:
            sink_node_ids = set(turn_trace.sink_outputs)
            for node_trace in turn_trace.node_traces:
                if node_trace.role not in self._candidate_roles() and node_trace.node_id not in sink_node_ids:
                    continue
                normalized = self._sanitize_candidate(question_text, node_trace.output, metadata=metadata)
                if not normalized:
                    continue
                parse_ok = None
                entry_point_ok = None
                if dataset_profile.task_type == "code_generation":
                    parse_ok, entry_point_ok = self._code_candidate_checks(normalized, metadata=metadata)
                occurrence_key = (turn_trace.turn_index, node_trace.node_id)
                checker_snapshot = self._checker_snapshot(feedback_by_occurrence.get(occurrence_key, ()))
                admitted, admission_source = self._candidate_bank_admission(
                    role=node_trace.role,
                    is_sink=node_trace.node_id in sink_node_ids,
                    checker_snapshot=checker_snapshot,
                    is_recovery_output=self._is_recovery_output(node_trace),
                )
                occurrence_map[occurrence_key] = {
                    "digest": self._candidate_digest(normalized),
                    "role": node_trace.role,
                    "is_sink": node_trace.node_id in sink_node_ids,
                    "parse_ok": parse_ok,
                    "entry_point_ok": entry_point_ok,
                    "admitted_to_candidate_bank": bool(admitted),
                    "candidate_bank_source": admission_source,
                    "checker_labels": list(checker_snapshot["labels"]),
                }
                if not admitted:
                    continue
                entry = ensure_entry(normalized)
                if entry is None:
                    continue
                if dataset_profile.task_type == "code_generation":
                    entry["parse_ok"] = parse_ok
                    entry["entry_point_ok"] = entry_point_ok
                self._attach_candidate_provenance(
                    entry,
                    turn_index=turn_trace.turn_index,
                    node_trace=node_trace,
                    checker_snapshot=checker_snapshot,
                    admission_source=admission_source,
                )
                feedback_stats = self._aggregate_occurrence_feedback(
                    feedback_by_occurrence.get(occurrence_key, ()),
                    dataset_profile=dataset_profile,
                    target_role=node_trace.role,
                    target_is_sink=node_trace.node_id in sink_node_ids,
                    parse_ok=parse_ok,
                    entry_point_ok=entry_point_ok,
                )
                entry["occurrence_count"] += 1
                if node_trace.node_id in sink_node_ids:
                    entry["sink_support"] += 1
                entry["source_node_ids"].add(node_trace.node_id)
                entry["source_roles"].add(node_trace.role)
                entry["turn_indices"].add(turn_trace.turn_index)
                entry["feedback_pass"] += feedback_stats["pass_count"]
                entry["feedback_challenge"] += feedback_stats["challenge_count"]
                entry["feedback_uncertain"] += feedback_stats["uncertain_count"]
                entry["feedback_pass_calibrated"] += feedback_stats["pass_weight"]
                entry["feedback_challenge_calibrated"] += feedback_stats["challenge_weight"]
                entry["feedback_uncertain_calibrated"] += feedback_stats["uncertain_weight"]
                entry["reviewer_event_count"] += feedback_stats["event_count"]
                entry["reviewer_trust_sum"] += feedback_stats["trust_sum"]

        if stage1_output:
            anchor_entry = ensure_entry(stage1_output)
            if anchor_entry is not None:
                anchor_entry["stage1_anchor"] = True

        bundle = self._apply_unified_corrections(
            bank,
            question_text=question_text,
            metadata=metadata,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
        )
        bundle["occurrences"] = occurrence_map
        return bundle

    def _select_final_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not candidates:
            if anchor is not None:
                self._phase3a_last_residual_mean = float(anchor.get("phase3a_residual_mean", 0.5))
                return anchor, "phase3a_unified_empty_bank_preserve_anchor"
            return None, "phase3a_unified_empty_bank"
        best = dict(candidates[0])
        if anchor is None:
            self._phase3a_last_residual_mean = float(best.get("phase3a_residual_mean", 0.5))
            return best, "phase3a_unified_no_anchor"
        if best.get("digest") == anchor.get("digest"):
            self._phase3a_last_residual_mean = float(anchor.get("phase3a_residual_mean", 0.5))
            return anchor, "phase3a_unified_anchor_wins_frontier"
        anchor_utility = float(anchor.get("phase3a_adjusted_utility", 0.0))
        best_utility = float(best.get("phase3a_adjusted_utility", 0.0))
        anchor_preserve = float(anchor.get("phase3a_verifier_state").residual_vector.get("r_preserve", 0.0)) if isinstance(anchor.get("phase3a_verifier_state"), VerifierState) else 0.0
        best_state = best.get("phase3a_verifier_state")
        best_preserve = float(best_state.residual_vector.get("r_preserve", 1.0)) if isinstance(best_state, VerifierState) else 1.0
        best_confidence = float(best.get("phase3a_confidence_score", 0.0))
        anchor_confidence = float(anchor.get("phase3a_confidence_score", 0.0))
        if (
            best_utility >= anchor_utility + float(self.config.guard_margin)
            and best_preserve <= max(float(self.config.preserve_threshold), anchor_preserve + 0.10)
            and (best_confidence >= anchor_confidence or float(best.get("phase3a_residual_mean", 1.0)) + 0.05 < float(anchor.get("phase3a_residual_mean", 1.0)))
        ):
            self._phase3a_last_residual_mean = float(best.get("phase3a_residual_mean", 0.5))
            if best.get("repair_branch"):
                return best, "phase3a_unified_override_with_correction"
            return best, "phase3a_unified_override_frontier"
        self._phase3a_last_residual_mean = float(anchor.get("phase3a_residual_mean", 0.5))
        return anchor, "phase3a_unified_preserve_anchor_guard"

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
        self._last_phase3a_selection = {
            "stage2_version": self.config.stage2_version,
            "selection_reason": strategy,
            "stage1_anchor_present": bool(anchor is not None),
            "stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "selected_candidate_digest": str((selected or {}).get("digest", "")),
            "selected_candidate_source": self._candidate_source_label(selected or {}),
            "selected_is_correction": bool((selected or {}).get("repair_branch", False)),
            "candidate_count": int(len(candidates)),
            "soft_class_count": int(len(bundle.get("phase3a_soft_classes", []))),
            "correction_attempt_count": int(bundle.get("phase3a_correction_attempt_count", 0)),
            "correction_accept_count": int(bundle.get("phase3a_correction_accept_count", 0)),
            "correction_improvement_count": int(bundle.get("phase3a_correction_improvement_count", 0)),
            "phase3a_top_candidates": top,
            "phase3a_top_classes": list(bundle.get("phase3a_soft_classes", [])),
            "phase3a_correction_traces": list(bundle.get("phase3a_correction_traces", [])),
            "phase3a_halt_mass": float(
                self._phase3a_controller_snapshot.halt_mass if self._phase3a_controller_snapshot is not None else 0.0
            ),
        }

    def _finalize_answer(
        self,
        *,
        question_text: str,
        controller_state,
        sink_outputs: Dict[str, str],
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> Tuple[str, str]:
        del controller_state
        del sink_outputs
        del reference_answer
        bundle = self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        bundle["question_text"] = question_text
        bundle["metadata"] = metadata
        self._last_candidate_bundle = bundle
        candidates = bundle["candidates"]
        candidates_serialized = bundle["candidates_serialized"]
        anchor = bundle["anchor"]
        anchor_serialized = bundle["anchor_serialized"]
        selected, strategy = self._select_final_candidate(candidates, anchor)
        selected_serialized = None
        if selected is not None:
            for item in candidates_serialized:
                if item.get("digest") == selected.get("digest"):
                    selected_serialized = dict(item)
                    break
        self._record_selection_metadata(
            candidates=candidates_serialized,
            selected=selected_serialized,
            anchor=anchor_serialized,
            strategy=strategy,
            bundle=bundle,
        )
        return str((selected or {}).get("text", "")), strategy

    def _graph_faithfulness_metadata(self, result: Stage2RunResult) -> Dict[str, Any]:
        ratios_edge = []
        ratios_node = []
        task_node_count = max(1.0, float(result.metadata.get("task_node_count", 0.0)))
        for trace in result.turn_traces:
            active_edges = trace.active_edges
            ratios_edge.append(
                sum(1.0 for activation in active_edges if activation.active) / float(max(1, len(active_edges)))
                if active_edges
                else 1.0
            )
            active_nodes = trace.metadata.get("active_node_ids", [])
            ratios_node.append(float(len(active_nodes)) / task_node_count)
        selected_source = str(self._last_phase3a_selection.get("selected_candidate_source", ""))
        selected_digest = str(self._last_phase3a_selection.get("selected_candidate_digest", ""))
        selected_entry = None
        for entry in self._last_candidate_bundle.get("candidates", []):
            if str(entry.get("digest", "")) == selected_digest:
                selected_entry = entry
                break
        provenance_coverage = 0.0
        recovery_size = 0.0
        if selected_entry is not None:
            artifact = selected_entry.get("phase3a_artifact")
            if isinstance(artifact, ArtifactIR):
                provenance_coverage = float(artifact.metadata.get("provenance_coverage", 0.0))
            recovery_size = float(len(selected_entry.get("recovery_subgraph_node_ids", [])))
        return {
            "graph_faithfulness_active_edge_ratio_by_turn": ratios_edge,
            "graph_faithfulness_active_node_ratio_by_turn": ratios_node,
            "graph_faithfulness_candidate_provenance_coverage": provenance_coverage,
            "graph_faithfulness_recovery_subgraph_size": recovery_size,
            "graph_faithfulness_final_answer_source_type": selected_source,
        }

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
        self._last_phase3a_selection = {}
        self._last_candidate_bundle = {}
        self._phase3a_current_graph = graph
        self._phase3a_controller_snapshot = None
        self._phase3a_last_residual_mean = 0.5
        self._phase3a_last_utility_delta = 0.0
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
            result.signature = "stage2_phase3a_unified|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_phase3a_selection)
        result.metadata.update(self._graph_faithfulness_metadata(result))
        result.metadata["stage2_version"] = self.config.stage2_version
        result.metadata["phase3a_protocol_family"] = "unified_verifier_conditioned_refinement"
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
        with open(os.path.join(replay_dir, "phase3a_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_phase3a_selection, handle, ensure_ascii=False, indent=2)

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
        graph_ref = self._phase3a_current_graph or graph
        del reference_answer
        stats = dict(
            Stage2RuntimeV2.learn_from_run(
                self,
                graph_ref,
                result,
                dataset_profile=dataset_profile,
                summary=summary,
                reward_target=reward_target,
            )
        )
        if not self.config.learning.enabled or not question_text:
            stats.update(
                {
                    "utility_updates": 0.0,
                    "delta_updates": 0.0,
                    "halt_updates": 0.0,
                    "utility_steps": float(self._utility_model.steps),
                    "delta_steps": float(self._delta_model.steps),
                    "halt_steps": float(self._halt_model.steps),
                }
            )
            return stats

        bundle = self._last_candidate_bundle or self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=result.turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        target = self._summary_target(summary, reward_target)
        utility_updates = 0
        for entry in list(bundle.get("candidates", []))[: max(1, int(self.config.candidate_max_k))]:
            features = entry.get("phase3a_utility_features", {})
            if not isinstance(features, dict) or not features:
                continue
            local_target = clamp01(target * (1.0 - 0.45 * float(entry.get("phase3a_residual_mean", 1.0))))
            if str(entry.get("digest", "")) == str(self._last_phase3a_selection.get("selected_candidate_digest", "")):
                local_target = clamp01(target)
            self._utility_model.update({str(name): float(value) for name, value in features.items()}, local_target)
            utility_updates += 1

        delta_updates = 0
        for trace in bundle.get("phase3a_correction_traces", []):
            utility_delta = float(trace.get("utility_delta", 0.0))
            residual_delta = float(trace.get("residual_delta", 0.0))
            features = {
                "bias": 1.0,
                "utility_delta": utility_delta,
                "residual_delta": residual_delta,
                "accepted": 1.0 if trace.get("accepted", False) else 0.0,
            }
            delta_target = clamp01(0.5 + 0.5 * max(utility_delta, residual_delta))
            self._delta_model.update(features, delta_target)
            delta_updates += 1

        halt_updates = 0
        halt_mass = float(self._last_phase3a_selection.get("phase3a_halt_mass", 0.0))
        halt_features = {
            "bias": 1.0,
            "halt_mass": halt_mass,
            "selected_is_correction": 1.0 if self._last_phase3a_selection.get("selected_is_correction", False) else 0.0,
            "selected_anchor": 1.0 if self._last_phase3a_selection.get("stage1_anchor_used", False) else 0.0,
            "candidate_count": float(self._last_phase3a_selection.get("candidate_count", 0)),
        }
        halt_target = 1.0 if halt_mass >= float(self.config.halt_threshold) else 0.0
        self._halt_model.update(halt_features, halt_target)
        halt_updates += 1
        stats.update(
            {
                "base_target": float(target),
                "utility_updates": float(utility_updates),
                "delta_updates": float(delta_updates),
                "halt_updates": float(halt_updates),
                "utility_steps": float(self._utility_model.steps),
                "delta_steps": float(self._delta_model.steps),
                "halt_steps": float(self._halt_model.steps),
            }
        )
        return stats


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))
