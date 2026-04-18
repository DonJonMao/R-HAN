from __future__ import annotations

import json
import os
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mas_stage2.learning import OnlineLinearModel
from mas_stage2.types import ControllerState, EdgeActivation, FeedbackEvent, Stage2RunResult, TurnTrace
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import UnionGraph, UnionNode
from stage2_gcr_plus.runtime_v2 import Stage2RuntimeV2
from stage2_gcr_plus.runtime_v41 import Stage2RuntimeV41
from stage2_phase3a_unified.runtime import Phase3aUnifiedRuntime

from .artifacts import ArtifactIR, answer_text, canonicalize_candidate, clamp01, cosine_similarity, pooled_artifact_vector, softmax
from .config import Phase1SemanticSafeOverrideConfig
from .controller import UnifiedControllerSnapshot, build_controller_state
from .verifier import (
    VerifierState,
    apply_anchor_pairwise_metrics,
    compute_overturn_risk,
    mean_residual,
    verify_artifact,
)


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))


class Phase1SemanticSafeOverrideRuntime(Phase3aUnifiedRuntime):
    def __init__(self, config: Phase1SemanticSafeOverrideConfig, evaluator, agent_pool, embedder):
        Stage2RuntimeV41.__init__(self, config, evaluator, agent_pool, embedder)
        self.config = config
        self._utility_model = OnlineLinearModel(learning_rate=config.utility_learning_rate)
        self._overturn_model = OnlineLinearModel(learning_rate=config.overturn_learning_rate)
        self._safe_override_model = OnlineLinearModel(learning_rate=config.safe_override_learning_rate)
        self._last_phase1_selection: Dict[str, Any] = {}
        self._last_candidate_bundle: Dict[str, Any] = {}
        self._phase1_current_graph: Optional[UnionGraph] = None
        self._phase1_controller_snapshot: Optional[UnifiedControllerSnapshot] = None
        self._phase1_last_residual_mean: float = 0.5
        self._phase1_last_utility_delta: float = 0.0

    def state_dict(self) -> dict:
        payload = Stage2RuntimeV41.state_dict(self)
        payload.update(
            {
                "phase1_utility_model": self._utility_model.state_dict(),
                "phase1_overturn_model": self._overturn_model.state_dict(),
                "phase1_safe_override_model": self._safe_override_model.state_dict(),
            }
        )
        return payload

    def load_state_dict(self, state: dict) -> None:
        Stage2RuntimeV41.load_state_dict(self, state)
        if isinstance(state.get("phase1_utility_model"), dict):
            self._utility_model.load_state_dict(state["phase1_utility_model"])
        if isinstance(state.get("phase1_overturn_model"), dict):
            self._overturn_model.load_state_dict(state["phase1_overturn_model"])
        if isinstance(state.get("phase1_safe_override_model"), dict):
            self._safe_override_model.load_state_dict(state["phase1_safe_override_model"])

    def _ensure_phase1_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_v4_entry_fields(entry)
        entry.setdefault("phase1_artifact", None)
        entry.setdefault("phase1_verifier_state", None)
        entry.setdefault("phase1_support_mean", 0.0)
        entry.setdefault("phase1_typed_support_score", 0.0)
        entry.setdefault("phase1_raw_utility", 0.0)
        entry.setdefault("phase1_adjusted_utility", 0.0)
        entry.setdefault("phase1_safe_utility", 0.0)
        entry.setdefault("phase1_frontier_weight", 0.0)
        entry.setdefault("phase1_utility_features", {})
        entry.setdefault("phase1_residual_mean", 1.0)
        entry.setdefault("phase1_confidence_score", 0.0)
        entry.setdefault("phase1_answer_consistency_score", 0.0)
        entry.setdefault("phase1_anchor_similarity", 0.0)
        entry.setdefault("phase1_answer_similarity", 0.0)
        entry.setdefault("phase1_answer_delta", 0.0)
        entry.setdefault("phase1_overturn_risk", 0.0)
        entry.setdefault("phase1_safe_override_score", 0.0)
        entry.setdefault("phase1_safe_override_features", {})
        entry.setdefault("phase1_soft_class_id", -1)
        entry.setdefault("phase1_soft_class_size", 1)

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super(Phase3aUnifiedRuntime, self)._serialize_candidate_entry(entry)
        verifier_state = entry.get("phase1_verifier_state")
        payload.update(
            {
                "phase1_raw_utility": float(entry.get("phase1_raw_utility", 0.0)),
                "phase1_adjusted_utility": float(entry.get("phase1_adjusted_utility", 0.0)),
                "phase1_safe_utility": float(entry.get("phase1_safe_utility", 0.0)),
                "phase1_frontier_weight": float(entry.get("phase1_frontier_weight", 0.0)),
                "phase1_support_mean": float(entry.get("phase1_support_mean", 0.0)),
                "phase1_typed_support_score": float(entry.get("phase1_typed_support_score", 0.0)),
                "phase1_residual_mean": float(entry.get("phase1_residual_mean", 0.0)),
                "phase1_confidence_score": float(entry.get("phase1_confidence_score", 0.0)),
                "phase1_answer_consistency_score": float(entry.get("phase1_answer_consistency_score", 0.0)),
                "phase1_anchor_similarity": float(entry.get("phase1_anchor_similarity", 0.0)),
                "phase1_answer_similarity": float(entry.get("phase1_answer_similarity", 0.0)),
                "phase1_answer_delta": float(entry.get("phase1_answer_delta", 0.0)),
                "phase1_overturn_risk": float(entry.get("phase1_overturn_risk", 0.0)),
                "phase1_safe_override_score": float(entry.get("phase1_safe_override_score", 0.0)),
                "phase1_verifier_state": verifier_state.to_dict() if isinstance(verifier_state, VerifierState) else {},
                "phase1_soft_class_id": int(entry.get("phase1_soft_class_id", -1)),
                "phase1_soft_class_size": int(entry.get("phase1_soft_class_size", 1)),
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
        graph = self._phase1_current_graph
        task_nodes = self._task_nodes(graph) if graph is not None else []
        state, snapshot = build_controller_state(
            graph=graph,
            task_nodes=task_nodes,
            previous_feedback=previous_feedback,
            previous_snapshot=self._phase1_controller_snapshot,
            turn_index=turn_index,
            total_turns=total_turns,
            last_residual_mean=self._phase1_last_residual_mean,
            last_utility_delta=self._phase1_last_utility_delta,
            active_edges=active_edges,
            node_top_k=int(self.config.controller_node_top_k),
        )
        self._phase1_controller_snapshot = snapshot
        return state

    def _memory_view_for_record(self, record) -> str:
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
        attention = dict(turn_state.metadata.get("phase1_memory_view_attention", {}))
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
            self._phase1_controller_snapshot.node_participation if self._phase1_controller_snapshot is not None else {}
        )
        for activation in activations:
            beta = clamp01(
                0.60 * float(activation.score)
                + 0.20 * float(participation.get(activation.src, 0.0))
                + 0.20 * float(participation.get(activation.dst, 0.0))
            )
            activation.metadata["phase1_edge_support"] = beta
            if beta >= float(self.config.controller_edge_threshold):
                activation.active = True
            activation.reason = f"{activation.reason},phase1_beta={beta:.3f}"
        return activations

    def _active_task_nodes_v2(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
    ) -> List[UnionNode]:
        participation = dict(
            self._phase1_controller_snapshot.node_participation if self._phase1_controller_snapshot is not None else {}
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
            dataset_name=dataset_profile.name,
            task_type=dataset_profile.task_type,
            answer_format=dataset_profile.answer_format,
            task_subtype=str((metadata or {}).get("task", "")),
            answer_role_threshold=float(self.config.answer_role_threshold),
            evidence_role_threshold=float(self.config.evidence_role_threshold),
        )

    def _utility_features(
        self,
        *,
        entry: Dict[str, Any],
        verifier_state: VerifierState,
        artifact: ArtifactIR,
        total_turns: int,
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        return {
            "bias": 1.0,
            f"dataset::{dataset_profile.name}": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            f"answer_format::{dataset_profile.answer_format}": 1.0,
            f"task_subtype::{str(artifact.metadata.get('task_subtype', '')) or 'unknown'}": 1.0,
            "support_mean": float(_mean(verifier_state.support_map.values())),
            "confidence_score": float(verifier_state.confidence_score),
            "progress_score": float(verifier_state.progress_score),
            "answer_consistency": float(verifier_state.answer_consistency_score),
            "answer_valid": 1.0 if artifact.answer_object.valid else 0.0,
            "schema_valid": 1.0 if bool(artifact.answer_object.fields.get("contract_valid", artifact.answer_object.fields.get("schema_valid", True))) else 0.0,
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
        artifact: ArtifactIR,
    ) -> float:
        support_mean = _mean(verifier_state.support_map.values())
        residual_mean = mean_residual(verifier_state)
        provenance_coverage = float(artifact.metadata.get("provenance_coverage", 0.0))
        return clamp01(
            0.36 * (1.0 - residual_mean)
            + 0.18 * verifier_state.confidence_score
            + 0.16 * verifier_state.answer_consistency_score
            + 0.12 * support_mean
            + 0.10 * provenance_coverage
            + 0.08 * verifier_state.anchor_similarity
        )

    def _safe_override_features(
        self,
        *,
        entry: Dict[str, Any],
        anchor: Optional[Dict[str, Any]],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        state = entry.get("phase1_verifier_state")
        if not isinstance(state, VerifierState):
            return {"bias": 1.0}
        anchor_safe = float(anchor.get("phase1_safe_utility", 0.0)) if isinstance(anchor, dict) else 0.0
        anchor_confidence = float(anchor.get("phase1_confidence_score", 0.0)) if isinstance(anchor, dict) else 0.0
        anchor_residual = float(anchor.get("phase1_residual_mean", 1.0)) if isinstance(anchor, dict) else 1.0
        catastrophic = 1.0 if self._is_catastrophic_answer_rewrite(entry, anchor) else 0.0
        return {
            "bias": 1.0,
            f"dataset::{dataset_profile.name}": 1.0,
            f"task::{dataset_profile.task_type}": 1.0,
            f"answer_format::{dataset_profile.answer_format}": 1.0,
            f"task_subtype::{str(entry.get('phase1_artifact').metadata.get('task_subtype', '')) if isinstance(entry.get('phase1_artifact'), ArtifactIR) else 'unknown'}": 1.0,
            "safe_utility_gap": float(entry.get("phase1_safe_utility", 0.0)) - anchor_safe,
            "adjusted_utility_gap": float(entry.get("phase1_adjusted_utility", 0.0)) - float(anchor.get("phase1_adjusted_utility", 0.0) if isinstance(anchor, dict) else 0.0),
            "confidence_gap": float(entry.get("phase1_confidence_score", 0.0)) - anchor_confidence,
            "residual_advantage": anchor_residual - float(entry.get("phase1_residual_mean", 1.0)),
            "answer_consistency": float(state.answer_consistency_score),
            "overturn_risk": -float(state.overturn_risk),
            "preserve_risk": -float(state.preserve_risk),
            "answer_delta": -float(state.answer_delta),
            "catastrophic_rewrite": -catastrophic,
        }

    def _heuristic_safe_override_score(self, *, entry: Dict[str, Any], anchor: Optional[Dict[str, Any]]) -> float:
        state = entry.get("phase1_verifier_state")
        if not isinstance(state, VerifierState):
            return 0.0
        anchor_safe = float(anchor.get("phase1_safe_utility", 0.0)) if isinstance(anchor, dict) else 0.0
        score = (
            0.35 * clamp01(float(entry.get("phase1_safe_utility", 0.0)) - anchor_safe + 0.5)
            + 0.22 * float(state.answer_consistency_score)
            + 0.18 * float(entry.get("phase1_confidence_score", 0.0))
            + 0.12 * clamp01(float(anchor.get("phase1_residual_mean", 1.0)) - float(entry.get("phase1_residual_mean", 1.0)) + 0.5)
            - 0.13 * float(state.overturn_risk)
        )
        if self._is_catastrophic_answer_rewrite(entry, anchor):
            score -= 0.35
        return clamp01(score)

    def _soft_cluster_candidates(self, candidates: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        parent = list(range(len(candidates)))
        vectors = [pooled_artifact_vector(entry["phase1_artifact"]) for entry in candidates]

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
                    float(candidates[idx].get("phase1_safe_utility", 0.0)),
                    str(candidates[idx].get("digest", "")),
                ),
                reverse=True,
            )
            representative = candidates[members_sorted[0]]
            for member_index in members:
                candidates[member_index]["phase1_soft_class_id"] = int(cluster_id)
                candidates[member_index]["phase1_soft_class_size"] = int(len(members))
            summaries.append(
                {
                    "class_id": int(cluster_id),
                    "size": int(len(members)),
                    "representative_digest": str(representative.get("digest", "")),
                    "representative_source": self._candidate_source_label(representative),
                    "representative_safe_utility": float(representative.get("phase1_safe_utility", 0.0)),
                    "contains_anchor": any(bool(candidates[idx].get("stage1_anchor", False)) for idx in members),
                }
            )
        summaries.sort(key=lambda item: (item["representative_safe_utility"], item["size"]), reverse=True)
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
        self._ensure_phase1_entry_fields(entry)
        artifact = canonicalize_candidate(
            candidate_text=str(entry.get("text", "")),
            provenance=entry.get("provenance", ()),
            metadata=metadata,
            dataset_name=dataset_profile.name,
            task_type=dataset_profile.task_type,
            answer_format=dataset_profile.answer_format,
            task_subtype=str((metadata or {}).get("task", "")),
            answer_role_threshold=float(self.config.answer_role_threshold),
            evidence_role_threshold=float(self.config.evidence_role_threshold),
        )
        verifier_state = verify_artifact(
            artifact,
            question_text=question_text,
            metadata=metadata,
            dataset_name=dataset_profile.name,
            task_type=dataset_profile.task_type,
            answer_format=dataset_profile.answer_format,
            task_subtype=str((metadata or {}).get("task", "")),
            candidate_entry=entry,
            anchor_artifact=anchor_artifact,
            timeout_s=float(self.config.code_timeout_s),
            max_failed_examples=int(self.config.code_max_failed_examples),
        )
        features = self._utility_features(
            entry=entry,
            verifier_state=verifier_state,
            artifact=artifact,
            total_turns=total_turns,
            dataset_profile=dataset_profile,
        )
        predicted, uncertainty = self._utility_model.predict(features)
        heuristic = self._heuristic_utility(
            entry=entry,
            verifier_state=verifier_state,
            artifact=artifact,
        )
        raw_utility = clamp01(0.75 * heuristic + 0.25 * float(predicted))
        entry["phase1_artifact"] = artifact
        entry["phase1_verifier_state"] = verifier_state
        entry["phase1_support_mean"] = _mean(verifier_state.support_map.values())
        entry["phase1_typed_support_score"] = verifier_state.typed_support_score
        entry["phase1_utility_features"] = features
        entry["phase1_raw_utility"] = raw_utility
        entry["phase1_adjusted_utility"] = raw_utility
        entry["phase1_safe_utility"] = raw_utility
        entry["phase1_residual_mean"] = mean_residual(verifier_state)
        entry["phase1_confidence_score"] = verifier_state.confidence_score
        entry["phase1_answer_consistency_score"] = verifier_state.answer_consistency_score
        entry["phase1_anchor_similarity"] = verifier_state.anchor_similarity
        entry["phase1_answer_similarity"] = verifier_state.answer_similarity
        entry["phase1_answer_delta"] = verifier_state.answer_delta
        entry["phase1_overturn_risk"] = 0.0
        entry["phase1_safe_override_score"] = 0.0
        entry["candidate_model_score"] = raw_utility
        entry["candidate_model_uncertainty"] = float(uncertainty)
        entry["support_score"] = raw_utility

    def _compute_safe_utility(self, entry: Dict[str, Any]) -> float:
        state = entry.get("phase1_verifier_state")
        if not isinstance(state, VerifierState):
            return float(entry.get("phase1_adjusted_utility", 0.0))
        value = (
            float(entry.get("phase1_adjusted_utility", 0.0))
            - float(self.config.overturn_penalty_weight) * float(state.overturn_risk)
            - float(self.config.preserve_penalty_weight) * float(state.preserve_risk)
            + float(self.config.answer_consistency_bonus) * float(state.answer_consistency_score)
        )
        return clamp01(value)

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
        anchor = self._find_anchor_candidate(candidates)
        anchor_state = anchor.get("phase1_verifier_state") if isinstance(anchor, dict) else None
        anchor_artifact = anchor.get("phase1_artifact") if isinstance(anchor, dict) else anchor_artifact

        if isinstance(anchor_state, VerifierState) and isinstance(anchor_artifact, ArtifactIR):
            updated_anchor_state = apply_anchor_pairwise_metrics(
                anchor_artifact,
                anchor_state,
                anchor_artifact=anchor_artifact,
                anchor_state=anchor_state,
            )
            anchor["phase1_verifier_state"] = updated_anchor_state
            anchor["phase1_answer_consistency_score"] = updated_anchor_state.answer_consistency_score
            anchor["phase1_answer_delta"] = updated_anchor_state.answer_delta
            anchor["phase1_anchor_similarity"] = updated_anchor_state.anchor_similarity
            anchor["phase1_answer_similarity"] = updated_anchor_state.answer_similarity
            anchor_state = updated_anchor_state

        for entry in candidates:
            state = entry.get("phase1_verifier_state")
            artifact = entry.get("phase1_artifact")
            if not isinstance(state, VerifierState) or not isinstance(artifact, ArtifactIR):
                continue
            state = apply_anchor_pairwise_metrics(
                artifact,
                state,
                anchor_artifact=anchor_artifact if isinstance(anchor_artifact, ArtifactIR) else None,
                anchor_state=anchor_state if isinstance(anchor_state, VerifierState) else None,
            )
            risk_features = {
                "bias": 1.0,
                "answer_delta": float(state.answer_delta),
                "candidate_residual": float(mean_residual(state)),
                "anchor_residual": float(mean_residual(anchor_state)) if isinstance(anchor_state, VerifierState) else 0.0,
                "confidence_gap": float(state.confidence_score) - float(anchor_state.confidence_score) if isinstance(anchor_state, VerifierState) else 0.0,
                "answer_consistency": float(state.answer_consistency_score),
                "typed_support_gap": float(state.typed_support_score) - float(anchor_state.typed_support_score) if isinstance(anchor_state, VerifierState) else 0.0,
                "preserve_risk": float(state.preserve_risk),
            }
            predicted_risk, _ = self._overturn_model.predict(risk_features)
            heuristic_risk = compute_overturn_risk(
                artifact,
                state,
                anchor_artifact=anchor_artifact,
                anchor_state=anchor_state if isinstance(anchor_state, VerifierState) else None,
            )
            overturn_risk = clamp01(0.75 * heuristic_risk + 0.25 * float(predicted_risk))
            new_state = replace(state, overturn_risk=overturn_risk)
            entry["phase1_verifier_state"] = new_state
            entry["phase1_overturn_risk"] = overturn_risk
            entry["phase1_answer_consistency_score"] = new_state.answer_consistency_score
            entry["phase1_typed_support_score"] = new_state.typed_support_score
            entry["phase1_answer_delta"] = new_state.answer_delta
            entry["phase1_anchor_similarity"] = new_state.anchor_similarity
            entry["phase1_answer_similarity"] = new_state.answer_similarity

        vectors = [pooled_artifact_vector(entry["phase1_artifact"]) for entry in candidates]
        raw_utilities = [float(entry.get("phase1_raw_utility", 0.0)) for entry in candidates]
        frontier_prior = softmax(raw_utilities, temperature=float(self.config.redundancy_temperature))
        for index, entry in enumerate(candidates):
            penalty = 0.0
            for peer_index, peer_weight in enumerate(frontier_prior):
                if peer_index == index:
                    continue
                penalty += float(peer_weight) * max(0.0, cosine_similarity(vectors[index], vectors[peer_index]))
            entry["phase1_adjusted_utility"] = float(entry["phase1_raw_utility"]) - float(self.config.redundancy_gamma) * penalty
            entry["phase1_safe_utility"] = self._compute_safe_utility(entry)

        anchor = self._find_anchor_candidate(candidates)
        for entry in candidates:
            features = self._safe_override_features(entry=entry, anchor=anchor, dataset_profile=dataset_profile)
            predicted_score, _ = self._safe_override_model.predict(features)
            heuristic_score = self._heuristic_safe_override_score(entry=entry, anchor=anchor)
            entry["phase1_safe_override_features"] = features
            entry["phase1_safe_override_score"] = clamp01(0.65 * heuristic_score + 0.35 * float(predicted_score))
            entry["phase1_safe_utility"] = self._compute_safe_utility(entry)

        candidates.sort(
            key=lambda entry: (
                float(entry.get("phase1_safe_utility", 0.0)),
                float(entry.get("phase1_safe_override_score", 0.0)),
                float(entry.get("phase1_confidence_score", 0.0)),
                -float(entry.get("phase1_residual_mean", 1.0)),
                str(entry.get("digest", "")),
            ),
            reverse=True,
        )
        safe_utilities = [float(entry.get("phase1_safe_utility", 0.0)) for entry in candidates]
        frontier_weights = softmax(safe_utilities, temperature=float(self.config.redundancy_temperature))
        for entry, weight in zip(candidates, frontier_weights):
            entry["phase1_frontier_weight"] = float(weight)
        cluster_summary = self._soft_cluster_candidates(candidates)
        serialized = [self._serialize_candidate_entry(entry) for entry in candidates]
        return {
            "candidates": candidates,
            "candidates_serialized": serialized,
            "anchor": self._find_anchor_candidate(candidates),
            "anchor_serialized": self._find_anchor_candidate(serialized),
            "phase1_soft_classes": cluster_summary,
        }

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
                self._ensure_phase1_entry_fields(entry)
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

        bundle = self._score_bank(
            bank,
            question_text=question_text,
            dataset_profile=dataset_profile,
            total_turns=total_turns,
            metadata=metadata,
        )
        bundle["occurrences"] = occurrence_map
        return bundle

    def _is_catastrophic_answer_rewrite(self, entry: Dict[str, Any], anchor: Optional[Dict[str, Any]]) -> bool:
        if anchor is None or bool(entry.get("stage1_anchor", False)):
            return False
        answer_delta = float(entry.get("phase1_answer_delta", 0.0))
        answer_consistency = float(entry.get("phase1_answer_consistency_score", 0.0))
        return (
            answer_delta > float(self.config.catastrophic_answer_delta_threshold)
            and answer_consistency < float(self.config.catastrophic_consistency_threshold)
        )

    def _select_final_candidate(
        self,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
    ) -> Tuple[Optional[Dict[str, Any]], str]:
        if not candidates:
            if anchor is not None:
                self._phase1_last_residual_mean = float(anchor.get("phase1_residual_mean", 0.5))
                return anchor, "phase1_semantic_safe_override_empty_bank_preserve_anchor"
            return None, "phase1_semantic_safe_override_empty_bank"
        best = dict(candidates[0])
        if anchor is None:
            self._phase1_last_residual_mean = float(best.get("phase1_residual_mean", 0.5))
            return best, "phase1_semantic_safe_override_no_anchor"
        challengers = [dict(entry) for entry in candidates if entry.get("digest") != anchor.get("digest")]
        if not challengers:
            self._phase1_last_residual_mean = float(anchor.get("phase1_residual_mean", 0.5))
            return anchor, "phase1_semantic_safe_override_anchor_wins_frontier"
        anchor_safe_utility = float(anchor.get("phase1_safe_utility", 0.0))
        anchor_confidence = float(anchor.get("phase1_confidence_score", 0.0))
        anchor_residual = float(anchor.get("phase1_residual_mean", 1.0))
        challengers.sort(
            key=lambda entry: (
                float(entry.get("phase1_safe_utility", 0.0)) - anchor_safe_utility,
                float(entry.get("phase1_safe_override_score", 0.0)),
                float(entry.get("phase1_answer_consistency_score", 0.0)),
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
            challenger_safe_utility = float(challenger.get("phase1_safe_utility", 0.0))
            if challenger_safe_utility <= anchor_safe_utility:
                continue
            any_pairwise_better = True
            if self._is_catastrophic_answer_rewrite(challenger, anchor):
                continue
            if float(challenger.get("phase1_safe_override_score", 0.0)) < float(self.config.safe_override_threshold):
                continue
            if float(challenger.get("phase1_overturn_risk", 1.0)) >= float(self.config.overturn_threshold):
                continue
            if float(challenger.get("phase1_answer_consistency_score", 0.0)) < float(self.config.catastrophic_consistency_threshold):
                continue
            challenger_confidence = float(challenger.get("phase1_confidence_score", 0.0))
            challenger_residual = float(challenger.get("phase1_residual_mean", 1.0))
            if not (challenger_confidence >= anchor_confidence or challenger_residual + 0.05 < anchor_residual):
                continue
            qualified.append(challenger)
        if qualified:
            selected = max(
                qualified,
                key=lambda entry: (
                    float(entry.get("phase1_safe_utility", 0.0)),
                    float(entry.get("phase1_safe_override_score", 0.0)),
                    -float(entry.get("phase1_overturn_risk", 1.0)),
                    float(entry.get("phase1_answer_consistency_score", 0.0)),
                    float(entry.get("phase1_confidence_score", 0.0)),
                    -float(entry.get("phase1_residual_mean", 1.0)),
                    str(entry.get("digest", "")),
                ),
            )
            self._phase1_last_residual_mean = float(selected.get("phase1_residual_mean", anchor_residual))
            return selected, "phase1_semantic_safe_override_override_frontier"
        if not any_pairwise_better:
            self._phase1_last_residual_mean = anchor_residual
            return anchor, "phase1_semantic_safe_override_anchor_wins_frontier"
        self._phase1_last_residual_mean = anchor_residual
        return anchor, "phase1_semantic_safe_override_preserve_anchor_guard"

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
            selected_pairwise_value = float((selected or {}).get("phase1_safe_utility", 0.0)) - float(anchor.get("phase1_safe_utility", 0.0))
        self._last_phase1_selection = {
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
            "phase1_selected_pairwise_value": selected_pairwise_value,
            "phase1_top_candidates": top,
            "phase1_top_classes": list(bundle.get("phase1_soft_classes", [])),
            "phase1_pairwise_audit": list(bundle.get("phase1_pairwise_audit", [])),
            "phase1_halt_mass": float(
                self._phase1_controller_snapshot.halt_mass if self._phase1_controller_snapshot is not None else 0.0
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
        selected_source = str(self._last_phase1_selection.get("selected_candidate_source", ""))
        selected_digest = str(self._last_phase1_selection.get("selected_candidate_digest", ""))
        selected_entry = None
        for entry in self._last_candidate_bundle.get("candidates", []):
            if str(entry.get("digest", "")) == selected_digest:
                selected_entry = entry
                break
        provenance_coverage = 0.0
        answer_signature = ""
        if selected_entry is not None:
            artifact = selected_entry.get("phase1_artifact")
            if isinstance(artifact, ArtifactIR):
                provenance_coverage = float(artifact.metadata.get("provenance_coverage", 0.0))
                answer_signature = artifact.answer_signature
        return {
            "graph_faithfulness_active_edge_ratio_by_turn": ratios_edge,
            "graph_faithfulness_active_node_ratio_by_turn": ratios_node,
            "graph_faithfulness_candidate_provenance_coverage": provenance_coverage,
            "graph_faithfulness_final_answer_source_type": selected_source,
            "graph_faithfulness_final_answer_signature": answer_signature,
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
        self._last_phase1_selection = {}
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
            result.signature = "stage2_phase1_semantic_safe_override|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_phase1_selection)
        result.metadata.update(self._graph_faithfulness_metadata(result))
        result.metadata["stage2_version"] = self.config.stage2_version
        result.metadata["phase1_protocol_family"] = "semantic_verification_safe_override"
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
        with open(os.path.join(replay_dir, "phase1_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_phase1_selection, handle, ensure_ascii=False, indent=2)

    @staticmethod
    def _lexicographic_preference(left: Dict[str, float], right: Dict[str, float], *, eps: float = 1e-9) -> float:
        left_success = float(left.get("success", 0.0))
        right_success = float(right.get("success", 0.0))
        if left_success > right_success + eps:
            return 1.0
        if left_success < right_success - eps:
            return 0.0
        left_task = float(left.get("task_score", 0.0))
        right_task = float(right.get("task_score", 0.0))
        if left_task > right_task + eps:
            return 1.0
        if left_task < right_task - eps:
            return 0.0
        return 0.5

    def _evaluate_candidate_summary(
        self,
        *,
        question_text: str,
        candidate_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Dict[str, float]:
        summary = self._evaluator.evaluate_output(
            question_text,
            candidate_text,
            tier="tier2",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        return {
            "success": float(summary.mean_success),
            "task_score": float(summary.mean_task_score),
            "safety_penalty": float(summary.mean_safety_penalty),
        }

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
        del graph
        del summary
        del reward_target
        stats = {
            "enabled": 0.0,
            "selector_updates": 0.0,
            "edge_updates": 0.0,
            "controller_updates": 0.0,
            "phase1_frozen_base_runtime_learning": 1.0,
        }
        if not self.config.learning.enabled or not question_text:
            stats.update(
                {
                    "utility_updates": 0.0,
                    "overturn_updates": 0.0,
                    "safe_override_updates": 0.0,
                    "utility_steps": float(self._utility_model.steps),
                    "overturn_steps": float(self._overturn_model.steps),
                    "safe_override_steps": float(self._safe_override_model.steps),
                }
            )
            return stats

        bundle = self._last_candidate_bundle or self._candidate_bank_bundle(
            question_text=question_text,
            turn_traces=result.turn_traces,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        anchor = bundle.get("anchor")
        anchor_digest = str((anchor or {}).get("digest", ""))
        candidate_pool: List[Dict[str, Any]] = []
        if isinstance(anchor, dict):
            candidate_pool.append(anchor)
        for entry in list(bundle.get("candidates", [])):
            if str(entry.get("digest", "")) == anchor_digest:
                continue
            candidate_pool.append(entry)

        candidate_eval: Dict[str, Dict[str, float]] = {}
        for entry in candidate_pool:
            digest = str(entry.get("digest", ""))
            if not digest:
                continue
            candidate_eval[digest] = self._evaluate_candidate_summary(
                question_text=question_text,
                candidate_text=str(entry.get("text", "")),
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )

        anchor_eval = candidate_eval.get(anchor_digest, {"success": 0.0, "task_score": 0.0, "safety_penalty": 0.0})
        pairwise_audit: List[Dict[str, Any]] = []
        challenger_labels: Dict[str, float] = {}
        for entry in candidate_pool:
            digest = str(entry.get("digest", ""))
            eval_summary = candidate_eval.get(digest, {"success": 0.0, "task_score": 0.0, "safety_penalty": 0.0})
            if digest == anchor_digest:
                continue
            label = self._lexicographic_preference(eval_summary, anchor_eval)
            challenger_labels[digest] = label
            pairwise_audit.append(
                {
                    "digest": digest,
                    "source": self._candidate_source_label(entry),
                    "success": float(eval_summary.get("success", 0.0)),
                    "task_score": float(eval_summary.get("task_score", 0.0)),
                    "pairwise_label_vs_anchor": label,
                    "better_than_anchor": bool(label > 0.5),
                }
            )

        bundle["phase1_pairwise_audit"] = pairwise_audit
        self._last_candidate_bundle = bundle
        if isinstance(result.metadata, dict):
            result.metadata["phase1_pairwise_audit"] = pairwise_audit
        if isinstance(self._last_phase1_selection, dict):
            self._last_phase1_selection["phase1_pairwise_audit"] = pairwise_audit

        any_better = any(label > 0.5 for label in challenger_labels.values())
        any_tie = any(abs(label - 0.5) <= 1e-9 for label in challenger_labels.values())
        positive_pairs = sum(1 for label in challenger_labels.values() if label > 0.5)
        mined_pairs = len(challenger_labels)

        utility_updates = 0
        overturn_updates = 0
        safe_override_updates = 0
        for entry in candidate_pool:
            digest = str(entry.get("digest", ""))
            features = entry.get("phase1_utility_features", {})
            if isinstance(features, dict) and features:
                if digest == anchor_digest:
                    if any_better:
                        local_target = 0.0
                    elif any_tie:
                        local_target = 0.5
                    else:
                        local_target = 1.0
                else:
                    local_target = challenger_labels.get(digest, 0.0)
                self._utility_model.update({str(name): float(value) for name, value in features.items()}, local_target)
                utility_updates += 1

            state = entry.get("phase1_verifier_state")
            if isinstance(state, VerifierState) and digest != anchor_digest:
                ovr_features = {
                    "bias": 1.0,
                    "answer_delta": float(state.answer_delta),
                    "candidate_residual": float(mean_residual(state)),
                    "anchor_residual": float(anchor.get("phase1_residual_mean", 1.0)) if isinstance(anchor, dict) else 1.0,
                    "confidence_gap": float(state.confidence_score) - float(anchor.get("phase1_confidence_score", 0.0) if isinstance(anchor, dict) else 0.0),
                    "answer_consistency": float(state.answer_consistency_score),
                    "typed_support_gap": float(state.typed_support_score) - float((anchor.get("phase1_verifier_state").typed_support_score) if isinstance(anchor, dict) and isinstance(anchor.get("phase1_verifier_state"), VerifierState) else 0.0),
                    "preserve_risk": float(state.preserve_risk),
                }
                pairwise_target = challenger_labels.get(digest, 0.0)
                ovr_target = clamp01(1.0 - pairwise_target)
                self._overturn_model.update(ovr_features, ovr_target)
                overturn_updates += 1

            safe_features = entry.get("phase1_safe_override_features", {})
            if isinstance(safe_features, dict) and safe_features and digest != anchor_digest:
                safe_target = challenger_labels.get(digest, 0.0)
                self._safe_override_model.update({str(name): float(value) for name, value in safe_features.items()}, safe_target)
                safe_override_updates += 1

        stats.update(
            {
                "pairwise_learning_protocol": 1.0,
                "pairwise_mined_candidates": float(len(candidate_pool)),
                "pairwise_mined_pairs": float(mined_pairs),
                "pairwise_positive_pairs": float(positive_pairs),
                "utility_updates": float(utility_updates),
                "overturn_updates": float(overturn_updates),
                "safe_override_updates": float(safe_override_updates),
                "utility_steps": float(self._utility_model.steps),
                "overturn_steps": float(self._overturn_model.steps),
                "safe_override_steps": float(self._safe_override_model.steps),
            }
        )
        return stats
