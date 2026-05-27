from __future__ import annotations

import json
import math
import os
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_stage2.learning import OnlineLinearModel
from .runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import FeedbackEvent, Stage2RunResult, TurnTrace
from .runtime_v3 import Stage2RuntimeV3
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import EvalSummary, UnionGraph

from .config import Stage2V31Config


class Stage2RuntimeV31(Stage2RuntimeV3):
    def __init__(self, config: Stage2V31Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._pairwise_model = OnlineLinearModel(learning_rate=config.learning.edge_learning_rate)
        self._override_model = self._pairwise_model
        self._last_v3_1_selection: Dict[str, Any] = {}
        self._last_candidate_bundle: Dict[str, Any] = {}

    def state_dict(self) -> dict:
        payload = super().state_dict()
        payload.pop("v3_candidate_model", None)
        payload.pop("v3_override_model", None)
        payload.pop("v3_reviewer_model", None)
        payload.update(
            {
                "v3_1_candidate_model": self._candidate_model.state_dict(),
                "v3_1_pairwise_model": self._pairwise_model.state_dict(),
                "v3_1_reviewer_model": self._reviewer_model.state_dict(),
            }
        )
        return payload

    def load_state_dict(self, state: dict) -> None:
        Stage2RuntimeV2.load_state_dict(self, state)
        candidate_state = state.get("v3_1_candidate_model") or state.get("v3_candidate_model")
        if isinstance(candidate_state, dict):
            self._candidate_model.load_state_dict(candidate_state)
        pairwise_state = state.get("v3_1_pairwise_model") or state.get("v3_override_model")
        if isinstance(pairwise_state, dict):
            self._pairwise_model.load_state_dict(pairwise_state)
        reviewer_state = state.get("v3_1_reviewer_model") or state.get("v3_reviewer_model")
        if isinstance(reviewer_state, dict):
            self._reviewer_model.load_state_dict(reviewer_state)

    @staticmethod
    def _field_count(entry: Dict[str, Any], name: str) -> int:
        value = entry.get(name, ()) if isinstance(entry, dict) else ()
        if value is None:
            return 0
        return len(value)

    @staticmethod
    def _quality_score(entry: Dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        return float(entry.get("candidate_model_score", entry.get("support_score", 0.0)))

    @staticmethod
    def _occurrence_saturation(entry: Dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        count = max(0.0, float(entry.get("occurrence_count", 0.0)))
        return float(1.0 - math.exp(-count / 3.0))

    @staticmethod
    def _sink_ratio(entry: Dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        occurrence = max(1.0, float(entry.get("occurrence_count", 0.0)))
        return float(entry.get("sink_support", 0.0)) / occurrence

    def _source_diversity(self, entry: Dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        node_count = self._field_count(entry, "source_node_ids")
        role_count = self._field_count(entry, "source_roles")
        return float(math.log1p(float(node_count)) + 0.5 * math.log1p(float(role_count)))

    def _review_volume(self, entry: Dict[str, Any]) -> float:
        if not isinstance(entry, dict):
            return 0.0
        if "feedback_pass_calibrated" in entry:
            return (
                float(entry.get("feedback_pass_calibrated", 0.0))
                + float(entry.get("feedback_challenge_calibrated", 0.0))
                + float(entry.get("feedback_uncertain_calibrated", 0.0))
            )
        return (
            float(entry.get("feedback_pass", 0.0))
            + float(entry.get("feedback_challenge", 0.0))
            + float(entry.get("feedback_uncertain", 0.0))
        )

    def _review_consensus(self, entry: Dict[str, Any]) -> float:
        volume = self._review_volume(entry)
        if volume <= 0.0:
            return 0.0
        return float(self._review_advantage(entry)) / volume

    def _candidate_entry_features(
        self,
        entry: Dict[str, Any],
        *,
        dataset_profile: DatasetProfile,
        total_turns: int,
    ) -> Dict[str, float]:
        review_balance = self._review_advantage(entry)
        review_volume = self._review_volume(entry)
        source_node_count = self._field_count(entry, "source_node_ids")
        source_role_count = self._field_count(entry, "source_roles")
        turn_count = self._field_count(entry, "turn_indices")
        features = {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            "occurrence_log": math.log1p(float(entry.get("occurrence_count", 0.0))),
            "occurrence_saturation": self._occurrence_saturation(entry),
            "sink_support_log": math.log1p(float(entry.get("sink_support", 0.0))),
            "sink_ratio": self._sink_ratio(entry),
            "turn_ratio": turn_count / max(1.0, float(total_turns)),
            "source_node_count": math.log1p(float(source_node_count)),
            "source_role_count": math.log1p(float(source_role_count)),
            "source_diversity": self._source_diversity(entry),
            "feedback_pass_raw": float(entry.get("feedback_pass", 0.0)),
            "feedback_challenge_raw": float(entry.get("feedback_challenge", 0.0)),
            "feedback_uncertain_raw": float(entry.get("feedback_uncertain", 0.0)),
            "feedback_pass_calibrated": float(entry.get("feedback_pass_calibrated", 0.0)),
            "feedback_challenge_calibrated": float(entry.get("feedback_challenge_calibrated", 0.0)),
            "feedback_uncertain_calibrated": float(entry.get("feedback_uncertain_calibrated", 0.0)),
            "review_margin": float(review_balance),
            "review_volume": float(review_volume),
            "review_consensus": float(self._review_consensus(entry)),
            "reviewer_event_count": math.log1p(float(entry.get("reviewer_event_count", 0.0))),
            "reviewer_mean_trust": float(entry.get("reviewer_mean_trust", 0.5)),
            "text_length": min(1.5, math.log1p(float(entry.get("text_length", 0.0))) / 6.5),
            "line_count": min(1.5, math.log1p(float(entry.get("line_count", 0.0))) / 4.5),
            "parse_ok": 1.0 if bool(entry.get("parse_ok")) else 0.0,
            "entry_point_ok": 1.0 if bool(entry.get("entry_point_ok")) else 0.0,
        }
        for role in entry.get("source_roles", ()):
            features[f"source_role::{role}"] = 1.0
        return features

    def _override_features(
        self,
        challenger: Dict[str, Any],
        anchor: Dict[str, Any],
        *,
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        challenger_quality = self._quality_score(challenger)
        anchor_quality = self._quality_score(anchor)
        challenger_review = self._review_advantage(challenger)
        anchor_review = self._review_advantage(anchor)
        challenger_consensus = self._review_consensus(challenger)
        anchor_consensus = self._review_consensus(anchor)
        challenger_diversity = self._source_diversity(challenger)
        anchor_diversity = self._source_diversity(anchor)
        challenger_sink_ratio = self._sink_ratio(challenger)
        anchor_sink_ratio = self._sink_ratio(anchor)
        challenger_occurrence = self._occurrence_saturation(challenger)
        anchor_occurrence = self._occurrence_saturation(anchor)
        challenger_uncertainty = float(challenger.get("candidate_model_uncertainty", 0.0))
        anchor_uncertainty = float(anchor.get("candidate_model_uncertainty", 0.0))
        return {
            "bias": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            "challenger_quality": challenger_quality,
            "anchor_quality": anchor_quality,
            "quality_margin": challenger_quality - anchor_quality,
            "challenger_review_margin": float(challenger_review),
            "anchor_review_margin": float(anchor_review),
            "review_margin": float(challenger_review - anchor_review),
            "challenger_review_consensus": float(challenger_consensus),
            "anchor_review_consensus": float(anchor_consensus),
            "review_consensus_margin": float(challenger_consensus - anchor_consensus),
            "challenger_sink_ratio": float(challenger_sink_ratio),
            "anchor_sink_ratio": float(anchor_sink_ratio),
            "sink_ratio_margin": float(challenger_sink_ratio - anchor_sink_ratio),
            "challenger_source_diversity": float(challenger_diversity),
            "anchor_source_diversity": float(anchor_diversity),
            "source_diversity_margin": float(challenger_diversity - anchor_diversity),
            "challenger_occurrence_saturation": float(challenger_occurrence),
            "anchor_occurrence_saturation": float(anchor_occurrence),
            "occurrence_saturation_margin": float(challenger_occurrence - anchor_occurrence),
            "challenger_uncertainty": challenger_uncertainty,
            "anchor_uncertainty": anchor_uncertainty,
            "uncertainty_margin": float(anchor_uncertainty - challenger_uncertainty),
            "challenger_parse_ok": 1.0 if bool(challenger.get("parse_ok")) else 0.0,
            "anchor_parse_ok": 1.0 if bool(anchor.get("parse_ok")) else 0.0,
            "challenger_entry_point_ok": 1.0 if bool(challenger.get("entry_point_ok")) else 0.0,
            "anchor_entry_point_ok": 1.0 if bool(anchor.get("entry_point_ok")) else 0.0,
        }

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        payload.update(
            {
                "source_node_count": self._field_count(entry, "source_node_ids"),
                "source_role_count": self._field_count(entry, "source_roles"),
                "turn_count": self._field_count(entry, "turn_indices"),
                "quality_score": self._quality_score(entry),
                "sink_ratio": self._sink_ratio(entry),
                "source_diversity": self._source_diversity(entry),
                "review_consensus": self._review_consensus(entry),
                "occurrence_saturation": self._occurrence_saturation(entry),
            }
        )
        return payload

    def _candidate_sort_key(self, dataset_profile: DatasetProfile):
        if dataset_profile.task_type == "code_generation":
            return lambda entry: (
                1 if self._is_valid_code_candidate(entry) else 0,
                self._quality_score(entry),
                self._review_consensus(entry),
                self._sink_ratio(entry),
                self._source_diversity(entry),
                self._review_advantage(entry),
                str(entry.get("digest", "")),
            )
        return lambda entry: (
            self._quality_score(entry),
            self._review_consensus(entry),
            self._sink_ratio(entry),
            self._source_diversity(entry),
            self._review_advantage(entry),
            str(entry.get("digest", "")),
        )

    def _pairwise_candidate_comparisons(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> List[Dict[str, Any]]:
        if anchor is None:
            return []
        comparisons: List[Dict[str, Any]] = []
        for entry in candidates:
            if str(entry.get("digest", "")) == str(anchor.get("digest", "")):
                continue
            probability, uncertainty, _ = self._override_probability(entry, anchor, dataset_profile=dataset_profile)
            comparisons.append(
                {
                    "candidate": dict(entry),
                    "digest": str(entry.get("digest", "")),
                    "source": self._candidate_source_label(entry),
                    "pairwise_probability": float(probability),
                    "pairwise_uncertainty": float(uncertainty),
                    "quality_score": self._quality_score(entry),
                    "model_uncertainty": float(entry.get("candidate_model_uncertainty", 0.0)),
                    "review_advantage": float(self._review_advantage(entry)),
                    "review_consensus": float(self._review_consensus(entry)),
                    "sink_support": int(entry.get("sink_support", 0)),
                    "sink_ratio": float(self._sink_ratio(entry)),
                    "source_diversity": float(self._source_diversity(entry)),
                    "is_valid_code": self._is_valid_code_candidate(entry) if dataset_profile.task_type == "code_generation" else True,
                }
            )
        comparisons.sort(
            key=lambda item: (
                float(item["pairwise_probability"]),
                float(item["quality_score"]),
                float(item["review_consensus"]),
                float(item["sink_ratio"]),
                float(item["source_diversity"]),
                str(item["digest"]),
            ),
            reverse=True,
        )
        return comparisons

    @staticmethod
    def _pairwise_public_view(item: Dict[str, Any]) -> Dict[str, Any]:
        return {key: value for key, value in item.items() if key != "candidate"}

    def _pairwise_extra(self, item: Optional[Dict[str, Any]], *, prefix: str) -> Dict[str, Any]:
        if not isinstance(item, dict):
            return {
                f"{prefix}_probability": 0.0,
                f"{prefix}_uncertainty": 0.0,
                f"{prefix}_candidate_digest": "",
                f"{prefix}_candidate_source": "",
            }
        return {
            f"{prefix}_probability": float(item.get("pairwise_probability", 0.0)),
            f"{prefix}_uncertainty": float(item.get("pairwise_uncertainty", 0.0)),
            f"{prefix}_candidate_digest": str(item.get("digest", "")),
            f"{prefix}_candidate_source": str(item.get("source", "")),
        }

    def _select_against_anchor(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        *,
        dataset_profile: DatasetProfile,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        if not candidates:
            reason = "v3_1_empty_bank_use_stage1" if anchor is not None else "v3_1_empty_bank"
            return anchor, reason, {
                "v3_1_pairwise_candidate_count": 0,
                "v3_1_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v3_1_best_pairwise"),
            }
        best_quality = dict(candidates[0])
        if anchor is None:
            return best_quality, "v3_1_no_stage1_anchor", {
                "v3_1_pairwise_candidate_count": 0,
                "v3_1_top_pairwise_candidates": [],
                **self._pairwise_extra(None, prefix="v3_1_best_pairwise"),
            }
        comparisons = self._pairwise_candidate_comparisons(candidates, anchor, dataset_profile=dataset_profile)
        top_pairwise = [self._pairwise_public_view(item) for item in comparisons[: self.config.max_logged_candidates]]
        best_pairwise = comparisons[0] if comparisons else None
        extra = {
            "v3_1_pairwise_candidate_count": int(len(comparisons)),
            "v3_1_top_pairwise_candidates": top_pairwise,
            **self._pairwise_extra(best_pairwise, prefix="v3_1_best_pairwise"),
        }
        if not comparisons:
            return anchor, "v3_1_only_stage1_anchor", extra
        chosen_item = best_pairwise
        if dataset_profile.task_type == "code_generation":
            anchor_valid = self._is_valid_code_candidate(anchor)
            valid_comparisons = [item for item in comparisons if bool(item.get("is_valid_code", False))]
            extra["v3_1_valid_pairwise_candidate_count"] = int(len(valid_comparisons))
            if not anchor_valid and valid_comparisons:
                chosen_item = valid_comparisons[0]
                extra.update(self._pairwise_extra(chosen_item, prefix="v3_1_decision_pairwise"))
                return dict(chosen_item["candidate"]), "v3_1_code_replace_invalid_anchor", extra
            if not valid_comparisons:
                extra.update(self._pairwise_extra(best_pairwise, prefix="v3_1_decision_pairwise"))
                return anchor, "v3_1_code_no_valid_challenger", extra
            chosen_item = valid_comparisons[0]
            extra.update(self._pairwise_extra(chosen_item, prefix="v3_1_decision_pairwise"))
            if float(chosen_item.get("pairwise_probability", 0.0)) > 0.5:
                return dict(chosen_item["candidate"]), "v3_1_code_pairwise_override_stage1", extra
            return anchor, "v3_1_code_pairwise_preserve_stage1", extra
        extra.update(self._pairwise_extra(chosen_item, prefix="v3_1_decision_pairwise"))
        if float(chosen_item.get("pairwise_probability", 0.0)) > 0.5:
            return dict(chosen_item["candidate"]), "v3_1_pairwise_override_stage1", extra
        return anchor, "v3_1_pairwise_preserve_stage1", extra

    def _record_selection_metadata(
        self,
        *,
        task_type: str,
        candidates: Sequence[Dict[str, Any]],
        selected: Optional[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        strategy: str,
        selection_extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        top = [dict(item) for item in list(candidates)[: self.config.max_logged_candidates]]
        for item in top:
            item.pop("text", None)
        extra = dict(selection_extra or {})
        self._last_v3_1_selection = {
            "stage2_version": "v3.1",
            "v3_1_task_type": task_type,
            "v3_1_candidate_count": int(len(candidates)),
            "v3_1_stage1_anchor_present": bool(anchor is not None),
            "v3_1_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "v3_1_selection_reason": strategy,
            "v3_1_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v3_1_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v3_1_selected_quality_score": self._quality_score(selected or {}),
            "v3_1_selected_support_score": float((selected or {}).get("support_score", 0.0)),
            "v3_1_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v3_1_selected_sink_support": int((selected or {}).get("sink_support", 0)),
            "v3_1_selected_sink_ratio": self._sink_ratio(selected or {}),
            "v3_1_selected_source_diversity": self._source_diversity(selected or {}),
            "v3_1_selected_review_advantage": float(self._review_advantage(selected or {})),
            "v3_1_selected_review_consensus": float(self._review_consensus(selected or {})),
            "v3_1_stage1_anchor_digest": str((anchor or {}).get("digest", "")),
            "v3_1_stage1_anchor_quality_score": self._quality_score(anchor or {}),
            "v3_1_stage1_support_score": float((anchor or {}).get("support_score", 0.0)),
            "v3_1_stage1_model_uncertainty": float((anchor or {}).get("candidate_model_uncertainty", 0.0)),
            "v3_1_stage1_sink_support": int((anchor or {}).get("sink_support", 0)),
            "v3_1_stage1_sink_ratio": self._sink_ratio(anchor or {}),
            "v3_1_stage1_source_diversity": self._source_diversity(anchor or {}),
            "v3_1_stage1_review_advantage": float(self._review_advantage(anchor or {})),
            "v3_1_stage1_review_consensus": float(self._review_consensus(anchor or {})),
            "v3_1_candidate_model_steps": float(self._candidate_model.steps),
            "v3_1_pairwise_model_steps": float(self._pairwise_model.steps),
            "v3_1_reviewer_model_steps": float(self._reviewer_model.steps),
            "v3_1_top_candidates": top,
        }
        self._last_v3_1_selection.update(extra)

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
        self._last_candidate_bundle = bundle
        candidates = bundle["candidates"]
        candidates_serialized = bundle["candidates_serialized"]
        anchor = bundle["anchor"]
        anchor_serialized = bundle["anchor_serialized"]
        selected, strategy, extra = self._select_against_anchor(candidates, anchor, dataset_profile=dataset_profile)
        selected_serialized = None
        if selected is not None:
            for item in candidates_serialized:
                if item.get("digest") == selected.get("digest"):
                    selected_serialized = dict(item)
                    break
        final_answer = str((selected or {}).get("text", "")) if selected is not None else ""
        self._record_selection_metadata(
            task_type=dataset_profile.task_type,
            candidates=candidates_serialized,
            selected=selected_serialized,
            anchor=anchor_serialized,
            strategy=strategy,
            selection_extra=extra,
        )
        return final_answer, strategy

    @staticmethod
    def _relative_reviewer_target(event_type: str, candidate_target: float, reference_target: float) -> float:
        margin = float(candidate_target) - float(reference_target)
        if event_type in {"pass", "preserve"}:
            if margin > 0.0:
                return 1.0
            if margin < 0.0:
                return 0.0
            return 0.5
        if event_type in {"challenge", "reject", "conflict"}:
            if margin < 0.0:
                return 1.0
            if margin > 0.0:
                return 0.0
            return 0.5
        return max(0.0, 1.0 - abs(margin))

    def learn_from_run(
        self,
        graph: UnionGraph,
        result: Stage2RunResult,
        *,
        dataset_profile: DatasetProfile,
        summary: Optional[EvalSummary] = None,
        reward_target: Optional[float] = None,
        question_text: Optional[str] = None,
        reference_answer: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, float]:
        stats = dict(
            Stage2RuntimeV2.learn_from_run(
                self,
                graph,
                result,
                dataset_profile=dataset_profile,
                summary=summary,
                reward_target=reward_target,
            )
        )
        if not self.config.learning.enabled or not question_text:
            stats.update(
                {
                    "candidate_updates": 0.0,
                    "pairwise_updates": 0.0,
                    "override_updates": 0.0,
                    "reviewer_updates": 0.0,
                    "candidate_steps": float(self._candidate_model.steps),
                    "pairwise_steps": float(self._pairwise_model.steps),
                    "override_steps": float(self._pairwise_model.steps),
                    "reviewer_steps": float(self._reviewer_model.steps),
                }
            )
            return stats

        bundle = self._last_candidate_bundle or self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=result.turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        ranked_candidates = list(bundle.get("candidates", []))
        if not ranked_candidates:
            stats.update(
                {
                    "candidate_updates": 0.0,
                    "pairwise_updates": 0.0,
                    "override_updates": 0.0,
                    "reviewer_updates": 0.0,
                    "candidate_steps": float(self._candidate_model.steps),
                    "pairwise_steps": float(self._pairwise_model.steps),
                    "override_steps": float(self._pairwise_model.steps),
                    "reviewer_steps": float(self._reviewer_model.steps),
                }
            )
            return stats

        learning_candidates: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for entry in ranked_candidates:
            digest = str(entry.get("digest", ""))
            if digest in seen:
                continue
            learning_candidates.append(entry)
            seen.add(digest)
            if len(learning_candidates) >= self.config.max_logged_candidates:
                break
        anchor = bundle.get("anchor")
        if isinstance(anchor, dict) and str(anchor.get("digest", "")) not in seen:
            learning_candidates.append(anchor)

        candidate_targets = self._evaluate_candidate_targets(
            learning_candidates,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )

        candidate_updates = 0
        for entry in learning_candidates:
            features = entry.get("candidate_features")
            if not isinstance(features, dict) or not features:
                continue
            target = candidate_targets.get(str(entry.get("digest", "")))
            if target is None:
                continue
            self._candidate_model.update({str(name): float(value) for name, value in features.items()}, target)
            candidate_updates += 1

        pairwise_updates = 0
        for left in learning_candidates:
            left_digest = str(left.get("digest", ""))
            left_target = candidate_targets.get(left_digest)
            if left_target is None:
                continue
            for right in learning_candidates:
                right_digest = str(right.get("digest", ""))
                if right_digest == left_digest:
                    continue
                right_target = candidate_targets.get(right_digest)
                if right_target is None:
                    continue
                if left_target > right_target:
                    pairwise_target = 1.0
                elif left_target < right_target:
                    pairwise_target = 0.0
                else:
                    pairwise_target = 0.5
                features = self._override_features(left, right, dataset_profile=dataset_profile)
                self._pairwise_model.update(features, pairwise_target)
                pairwise_updates += 1

        reviewer_updates = 0
        occurrences = bundle.get("occurrences", {})
        candidate_by_digest = {str(entry.get("digest", "")): entry for entry in learning_candidates}
        anchor_digest = str(anchor.get("digest", "")) if isinstance(anchor, dict) else ""
        anchor_target = candidate_targets.get(anchor_digest)
        best_non_anchor_target: Optional[float] = None
        for entry in learning_candidates:
            digest = str(entry.get("digest", ""))
            if digest == anchor_digest:
                continue
            target = candidate_targets.get(digest)
            if target is None:
                continue
            if best_non_anchor_target is None or target > best_non_anchor_target:
                best_non_anchor_target = target

        for turn_trace in result.turn_traces:
            for event in turn_trace.feedback_events:
                occurrence = occurrences.get((event.turn_index, event.target_node_id))
                if not isinstance(occurrence, dict):
                    continue
                digest = str(occurrence.get("digest", ""))
                candidate_entry = candidate_by_digest.get(digest)
                candidate_target = candidate_targets.get(digest)
                if candidate_entry is None or candidate_target is None:
                    continue
                if digest == anchor_digest:
                    reference_target = best_non_anchor_target if best_non_anchor_target is not None else candidate_target
                else:
                    reference_target = anchor_target if anchor_target is not None else 0.5
                features = self._reviewer_features(
                    event,
                    dataset_profile=dataset_profile,
                    target_role=str(occurrence.get("role", "")),
                    target_is_sink=bool(occurrence.get("is_sink", False)),
                    parse_ok=occurrence.get("parse_ok"),
                    entry_point_ok=occurrence.get("entry_point_ok"),
                )
                reviewer_target = self._relative_reviewer_target(event.event_type, candidate_target, reference_target)
                self._reviewer_model.update(features, reviewer_target)
                reviewer_updates += 1

        stats.update(
            {
                "candidate_updates": float(candidate_updates),
                "pairwise_updates": float(pairwise_updates),
                "override_updates": float(pairwise_updates),
                "reviewer_updates": float(reviewer_updates),
                "candidate_steps": float(self._candidate_model.steps),
                "pairwise_steps": float(self._pairwise_model.steps),
                "override_steps": float(self._pairwise_model.steps),
                "reviewer_steps": float(self._reviewer_model.steps),
            }
        )
        return stats

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
        self._last_v3_1_selection = {}
        self._last_candidate_bundle = {}
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
            result.signature = "stage2_v3_1|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v3_1_selection)
        result.metadata["stage2_version"] = "v3.1"
        result.metadata["v3_1_all_task_nodes_each_turn"] = True
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
        with open(os.path.join(replay_dir, "v3_1_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v3_1_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v3_1_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
