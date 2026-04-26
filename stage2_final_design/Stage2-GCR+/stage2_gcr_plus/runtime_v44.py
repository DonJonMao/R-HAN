from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .runtime_v2 import Stage2RuntimeV2
from mas_stage2.types import ControllerState, EdgeActivation, FeedbackEvent, Stage2RunResult, TurnTrace
from .runtime_v43 import Stage2RuntimeV43
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import PromptSlots, UnionGraph, UnionNode

from .code_repair import CodeRepairEval, build_failure_summary, evaluate_code_candidate
from .config import Stage2V44Config
from .deductive_reasoning import (
    DEDUCTIVE_DATASETS,
    DeductiveEval,
    answer_only_from_artifact,
    build_deductive_repair_prompt,
    dataset_name_from,
    deductive_dominates,
    _has_deterministic_fatal,
    make_deductive_eval,
    parse_deductive_artifact,
    repair_operator_for_residual,
    verify_deductive_artifact,
)


@dataclass(frozen=True)
class ReasoningEval:
    normalized_answer: str
    fatal_contradictions: Tuple[str, ...]
    major_contradictions: Tuple[str, ...]
    critical_support: int
    contradiction_cluster: str

    @property
    def class_key(self) -> Tuple[Any, ...]:
        return (
            self.normalized_answer,
            tuple(sorted(self.fatal_contradictions)),
            self.contradiction_cluster,
        )

    @property
    def lexicographic_key(self) -> Tuple[int, int, int, int]:
        return (
            -len(self.fatal_contradictions),
            -len(self.major_contradictions),
            int(self.critical_support),
            int(bool(self.normalized_answer)),
        )

    def dominates(self, other: "ReasoningEval") -> bool:
        return self.lexicographic_key > other.lexicographic_key


@dataclass(frozen=True)
class GraphConstraintEval:
    normalized_structure: str
    broken_blocks: Tuple[str, ...]
    fatal_blocks: Tuple[str, ...]
    repair_locus: str
    verified_blocks: int

    @property
    def class_key(self) -> Tuple[Any, ...]:
        return (
            tuple(sorted(self.broken_blocks)),
            self.repair_locus,
            self.normalized_structure,
        )

    @property
    def rank_key(self) -> Tuple[int, int, int, int]:
        return (
            -len(self.fatal_blocks),
            -len(self.broken_blocks),
            int(self.verified_blocks),
            int(bool(self.normalized_structure)),
        )

    def dominates(self, other: "GraphConstraintEval") -> bool:
        mine = self.rank_key
        theirs = other.rank_key
        return mine > theirs


class Stage2RuntimeV44(Stage2RuntimeV43):
    def __init__(self, config: Stage2V44Config, evaluator, agent_pool, embedder):
        super().__init__(config, evaluator, agent_pool, embedder)
        self.config = config
        self._last_v4_4_selection: Dict[str, Any] = {}
        self._v4_4_route_family: str = ""
        self._v4_4_execution_mode_hint: str = "lean"
        self._v4_4_focus_node_ids: set[str] = set()
        self._v4_4_sink_guard_ids: set[str] = set()
        self._v4_4_protected_ids: set[str] = set()
        self._v4_4_champion_provenance_ids: set[str] = set()

    @staticmethod
    def _graph_repair_slots() -> PromptSlots:
        return PromptSlots(
            reasoning_mode="stepwise",
            upstream_usage="summary",
            output_style="raw",
            verification_mode="off",
            finalization="answer_only",
        )

    def _ensure_v4_4_entry_fields(self, entry: Dict[str, Any]) -> None:
        self._ensure_v4_3_entry_fields(entry)
        entry.setdefault("v4_4_class_key", ())
        entry.setdefault("v4_4_class_size", 1)
        entry.setdefault("v4_4_route_family", "")
        entry.setdefault("v4_4_execution_mode", "")
        entry.setdefault("promotion_inspector_decision", "")
        entry.setdefault("promotion_inspector_rationale", "")
        entry.setdefault("stage1_anchor_binding_legalized", False)
        entry.setdefault("stage1_anchor_binding_kind", "")
        entry.setdefault("stage1_anchor_binding_source_graph_id", "")
        entry.setdefault("anchor_bound_node_ids", [])
        entry.setdefault("anchor_bound_edge_ids", [])
        entry.setdefault("deductive_norm_answer", "")
        entry.setdefault("deductive_first_bad_step", None)
        entry.setdefault("deductive_verified_prefix_len", 0)
        entry.setdefault("deductive_residual_kind", "")
        entry.setdefault("deductive_final_consistent", False)
        entry.setdefault("deductive_repair_locus", "")
        entry.setdefault("deductive_parser_confidence", "")
        entry.setdefault("deductive_final_source", "")
        entry.setdefault("deductive_step_source", "")
        entry.setdefault("deductive_contract_ok", False)
        entry.setdefault("deductive_checked_equation_count", 0)
        entry.setdefault("deductive_fatal_count", 0)
        entry.setdefault("deductive_local_count", 0)
        entry.setdefault("deductive_repair_operator_type", "")
        entry.setdefault("deductive_repair_accepted", False)
        entry.setdefault("deductive_repair_rejected_by_dominance", False)

    @staticmethod
    def _stable_entry_tiebreak(entry: Dict[str, Any]) -> Tuple[int, str]:
        return (
            int(bool(entry.get("stage1_anchor", False))),
            str(entry.get("digest", "")),
        )

    def _serialize_candidate_entry(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        payload = super()._serialize_candidate_entry(entry)
        raw_class_key = entry.get("v4_4_class_key", ())
        public_class_key = [raw_class_key] if isinstance(raw_class_key, str) else list(raw_class_key)
        payload.update(
            {
                "v4_4_class_key": public_class_key,
                "v4_4_class_size": int(entry.get("v4_4_class_size", 1)),
                "v4_4_route_family": str(entry.get("v4_4_route_family", "")),
                "v4_4_execution_mode": str(entry.get("v4_4_execution_mode", "")),
                "promotion_inspector_decision": str(entry.get("promotion_inspector_decision", "")),
                "promotion_inspector_rationale": str(entry.get("promotion_inspector_rationale", "")),
                "candidate_bank_source": str(entry.get("candidate_bank_source", "")),
                "origin_node_id": str(entry.get("origin_node_id", "")),
                "origin_turn_index": int(entry.get("origin_turn_index", -1)),
                "origin_role": str(entry.get("origin_role", "")),
                "parent_candidate_digest": str(entry.get("parent_candidate_digest", "")),
                "repair_operator_type": str(entry.get("repair_operator_type", "")),
                "recovery_subgraph_node_ids": list(entry.get("recovery_subgraph_node_ids", ())),
                "recovery_subgraph_edge_ids": list(entry.get("recovery_subgraph_edge_ids", ())),
                "trigger_verifier_snapshot": dict(entry.get("trigger_verifier_snapshot", {})),
                "provenance": [dict(item) for item in entry.get("provenance", ()) if isinstance(item, dict)],
                "recovery_reinserted": bool(entry.get("recovery_reinserted", False)),
                "stage1_anchor_binding_legalized": bool(entry.get("stage1_anchor_binding_legalized", False)),
                "stage1_anchor_binding_kind": str(entry.get("stage1_anchor_binding_kind", "")),
                "stage1_anchor_binding_source_graph_id": str(entry.get("stage1_anchor_binding_source_graph_id", "")),
                "anchor_bound_node_ids": list(entry.get("anchor_bound_node_ids", ())),
                "anchor_bound_edge_ids": list(entry.get("anchor_bound_edge_ids", ())),
                "deductive_norm_answer": str(entry.get("deductive_norm_answer", "")),
                "deductive_first_bad_step": entry.get("deductive_first_bad_step", None),
                "deductive_verified_prefix_len": int(entry.get("deductive_verified_prefix_len", 0)),
                "deductive_residual_kind": str(entry.get("deductive_residual_kind", "")),
                "deductive_final_consistent": bool(entry.get("deductive_final_consistent", False)),
                "deductive_repair_locus": str(entry.get("deductive_repair_locus", "")),
                "deductive_parser_confidence": str(entry.get("deductive_parser_confidence", "")),
                "deductive_final_source": str(entry.get("deductive_final_source", "")),
                "deductive_step_source": str(entry.get("deductive_step_source", "")),
                "deductive_contract_ok": bool(entry.get("deductive_contract_ok", False)),
                "deductive_checked_equation_count": int(entry.get("deductive_checked_equation_count", 0)),
                "deductive_fatal_count": int(entry.get("deductive_fatal_count", 0)),
                "deductive_local_count": int(entry.get("deductive_local_count", 0)),
                "deductive_repair_operator_type": str(entry.get("deductive_repair_operator_type", "")),
                "deductive_repair_accepted": bool(entry.get("deductive_repair_accepted", False)),
                "deductive_repair_rejected_by_dominance": bool(entry.get("deductive_repair_rejected_by_dominance", False)),
            }
        )
        return payload

    def _budget_bucket(self, metadata: Optional[dict]) -> str:
        value = str((metadata or {}).get("v4_4_budget_bucket") or self.config.default_budget_bucket).strip().lower()
        if value not in {"tight", "normal", "ample"}:
            return "normal"
        return value

    def _apply_budget_bucket(self, base_mode: str, budget_bucket: str) -> str:
        if budget_bucket == "tight":
            if base_mode == "full":
                return "lean"
            return base_mode
        return base_mode

    def _initial_mode_hint(self, budget_bucket: str) -> str:
        if budget_bucket == "tight":
            return "lean"
        if budget_bucket == "ample":
            return "full"
        return "lean"

    def _dataset_name(self, dataset_profile: DatasetProfile, metadata: Optional[dict]) -> str:
        return dataset_name_from(dataset_profile, metadata)

    def _is_deductive_dataset(self, dataset_profile: DatasetProfile, metadata: Optional[dict]) -> bool:
        return self._dataset_name(dataset_profile, metadata) in DEDUCTIVE_DATASETS

    def _route_family(self, dataset_profile: DatasetProfile, metadata: Optional[dict]) -> str:
        task_type = str(dataset_profile.task_type)
        dataset_name = self._dataset_name(dataset_profile, metadata)
        if self._is_deductive_dataset(dataset_profile, metadata):
            return "deductive_reasoning"
        if task_type == "code_generation":
            return "code_repair"
        if task_type == "graph_reasoning":
            return "graph_constrained"
        if task_type == "structured_list" or dataset_name == "knowledge_crosswords":
            return "graph_constrained"
        return "adversarial"

    def _sanitize_candidate(
        self,
        question_text: str,
        raw_text: str,
        *,
        metadata: Optional[dict],
    ) -> str:
        metadata_name = str(
            (metadata or {}).get("dataset_name")
            or (metadata or {}).get("dataset")
            or (metadata or {}).get("mas_dataset_name")
            or ""
        ).strip().lower()
        if str(getattr(self, "_v4_4_route_family", "")) == "deductive_reasoning" or metadata_name in DEDUCTIVE_DATASETS:
            return MultiFidelityEvaluator._strip_hidden_reasoning(str(raw_text or "")).strip()
        return super()._sanitize_candidate(question_text, raw_text, metadata=metadata)

    def _role_answer_contract(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> str:
        del reference_answer
        if self._is_deductive_dataset(dataset_profile, metadata):
            if node.role in {"solver", "solver_a", "solver_b", "generator", "reviser", "aggregator"} or node.node_id in graph.sink_node_ids:
                return (
                    "Output exactly this internal derivation contract:\n"
                    "SOLUTION:\n"
                    "1. ...\n"
                    "2. ...\n"
                    "3. ...\n\n"
                    "FINAL: <number_or_expression>\n\n"
                    "Keep numbered derivation steps internally. The final evaluator will receive only FINAL."
                )
            if node.role in {"verifier", "critic", "judge"}:
                return (
                    "Output exactly four lines:\n"
                    "VERDICT: pass|challenge|uncertain\n"
                    "LOCUS: step=<id>|final|parse\n"
                    "ISSUE: arithmetic_mismatch|algebra_mismatch|unsupported_transition|missing_final|none\n"
                    "FIX: <one sentence>"
                )
        return super()._role_answer_contract(
            graph,
            node,
            question_text=question_text,
            reference_answer=None,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )

    def _postprocess_node_output(
        self,
        node: UnionNode,
        raw_output: str,
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        answer_contract: str,
    ) -> str:
        del reference_answer
        if self._is_deductive_dataset(dataset_profile, metadata):
            return MultiFidelityEvaluator._strip_hidden_reasoning(str(raw_output or "")).strip()
        return super()._postprocess_node_output(
            node,
            raw_output,
            question_text=question_text,
            reference_answer=None,
            metadata=metadata,
            dataset_profile=dataset_profile,
            answer_contract=answer_contract,
        )

    def _feedback_records(
        self,
        graph: UnionGraph,
        events: Sequence[FeedbackEvent],
        *,
        episode_id: str,
    ) -> List[Any]:
        records = super()._feedback_records(graph, events, episode_id=episode_id)
        if str(getattr(self, "_v4_4_route_family", "")) != "deductive_reasoning":
            return records
        for record in records:
            parsed = self._parse_tagged_output(str(record.text))
            issue = str(parsed.get("issue", "")).strip() or str(record.metadata.get("residual_kind", "")).strip()
            locus = str(parsed.get("locus", "")).strip()
            first_bad_step = None
            step_match = re.search(r"step\s*=\s*(\d+)", locus, flags=re.IGNORECASE)
            if step_match:
                first_bad_step = int(step_match.group(1))
            record.metadata.update(
                {
                    "morph_type": "deductive_reasoning",
                    "dataset_name": "",
                    "residual_kind": issue,
                    "first_bad_step": first_bad_step,
                    "verified_prefix_len": max(0, int(first_bad_step) - 1) if first_bad_step is not None else 0,
                    "repair_locus": f"step_{first_bad_step}_suffix" if first_bad_step is not None else (locus or "parse"),
                    "operator_type": str(record.metadata.get("operator_type", "")),
                }
            )
        return records

    def _build_turn_state(
        self,
        *,
        turn_index: int,
        total_turns: int,
        previous_feedback: Sequence[FeedbackEvent],
        active_edges: Sequence[EdgeActivation],
    ) -> ControllerState:
        state = Stage2RuntimeV2._build_turn_state(
            self,
            turn_index=turn_index,
            total_turns=total_turns,
            previous_feedback=previous_feedback,
            active_edges=active_edges,
        )
        focus_ids = {
            event.target_node_id
            for event in previous_feedback
            if event.event_type in {"challenge", "reject", "conflict"}
        }
        if not focus_ids:
            focus_ids = {
                event.target_node_id
                for event in previous_feedback
                if event.event_type in {"pass", "preserve"}
            }
        self._v4_4_focus_node_ids = set(focus_ids)
        state.mode = self._v4_4_execution_mode_hint
        state.focus = ",".join(sorted(focus_ids)[:3])
        state.metadata.update(
            {
                "v4_4_route_family": self._v4_4_route_family,
                "v4_4_execution_mode_hint": self._v4_4_execution_mode_hint,
                "v4_4_focus_count": len(focus_ids),
            }
        )
        return state

    def _route_role_allowed(self, role: str, route_family: str, execution_mode: str) -> bool:
        role = str(role)
        verifier_roles = {"verifier", "critic", "judge"}
        repair_roles = {"solver", "solver_a", "solver_b", "generator", "reviser", "router", "aggregator"}
        graph_roles = {"solver", "solver_a", "solver_b", "generator", "aggregator", "reviser", "verifier", "critic", "judge"}
        adversarial_roles = {"solver", "solver_a", "solver_b", "generator", "verifier", "critic", "judge", "aggregator", "reviser", "router"}

        if route_family == "code_repair":
            allowed = verifier_roles | repair_roles
            if execution_mode != "full":
                allowed.discard("solver_b")
            return role in allowed
        if route_family == "graph_constrained":
            allowed = set(graph_roles)
            if execution_mode != "full":
                allowed.discard("solver_b")
                allowed.discard("judge")
            return role in allowed
        if route_family == "deductive_reasoning":
            allowed = {
                "solver",
                "solver_a",
                "generator",
                "reviser",
                "verifier",
                "critic",
                "judge",
                "aggregator",
                "router",
            }
            if execution_mode != "full":
                allowed.discard("solver_b")
                allowed.discard("judge")
            return role in allowed
        allowed = set(adversarial_roles)
        if execution_mode != "full":
            allowed.discard("solver_b")
            allowed.discard("router")
        return role in allowed

    @staticmethod
    def _checker_roles() -> set[str]:
        return {"tester", "verifier", "critic", "judge", "checker"}

    @staticmethod
    def _sink_guard_roles() -> set[str]:
        return {"aggregator", "router", "reviser"}

    @staticmethod
    def _active_neighbors(active_edges: Sequence[EdgeActivation]) -> Dict[str, set[str]]:
        neighbors: Dict[str, set[str]] = {}
        for activation in active_edges:
            if not activation.active:
                continue
            neighbors.setdefault(activation.src, set()).add(activation.dst)
            neighbors.setdefault(activation.dst, set()).add(activation.src)
        return neighbors

    def _checker_node_ids(self, task_nodes: Sequence[UnionNode]) -> set[str]:
        return {
            node.node_id
            for node in task_nodes
            if node.node_type == "task" and str(node.role) in self._checker_roles()
        }

    def _checker_nodes_in_verifier_state(
        self,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
    ) -> set[str]:
        checker_ids = self._checker_node_ids(task_nodes)
        if not checker_ids:
            return set()
        neighbors = self._active_neighbors(active_edges)
        focus_ids = set(self._v4_4_focus_node_ids)
        involved: set[str] = set()
        for node_id in checker_ids:
            if node_id in focus_ids or any(peer in focus_ids for peer in neighbors.get(node_id, set())):
                involved.add(node_id)
        return involved

    def _sink_guard_ids(
        self,
        graph: UnionGraph,
        active_edges: Sequence[EdgeActivation],
    ) -> set[str]:
        sink_ids = {
            node_id
            for node_id in graph.sink_node_ids
            if node_id in graph.nodes and graph.nodes[node_id].node_type == "task"
        }
        guard_ids = set(sink_ids)
        for activation in active_edges:
            if not activation.active or activation.dst not in sink_ids:
                continue
            src_node = graph.nodes.get(activation.src)
            if src_node is None or src_node.node_type != "task":
                continue
            if str(src_node.role) in self._sink_guard_roles():
                guard_ids.add(activation.src)
        return guard_ids

    def _protected_node_ids(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
    ) -> set[str]:
        task_ids = {node.node_id for node in task_nodes if node.node_type == "task"}
        sink_ids = {
            node_id
            for node_id in graph.sink_node_ids
            if node_id in task_ids
        }
        champion_ids = {
            node_id
            for node_id in self._v4_4_champion_provenance_ids
            if node_id in task_ids
        }
        checker_state_ids = self._checker_nodes_in_verifier_state(task_nodes, active_edges)
        return sink_ids | champion_ids | checker_state_ids

    def _active_task_nodes_v2(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
    ) -> List[UnionNode]:
        base_nodes = Stage2RuntimeV2._active_task_nodes_v2(self, graph, task_nodes, active_edges)
        if not base_nodes:
            base_nodes = list(task_nodes)
        base_ids = {node.node_id for node in base_nodes}
        sink_guard_ids = self._sink_guard_ids(graph, active_edges)
        protected_ids = self._protected_node_ids(graph, task_nodes, active_edges)
        self._v4_4_sink_guard_ids = set(sink_guard_ids)
        self._v4_4_protected_ids = set(protected_ids)
        guarded_ids = sink_guard_ids | protected_ids
        neighbors = self._active_neighbors(active_edges)

        selected: List[UnionNode] = []
        seen_role_pairs: set[Tuple[str, str]] = set()
        for node in task_nodes:
            if node.node_id not in base_ids and node.node_id not in guarded_ids:
                continue
            role_match = self._route_role_allowed(node.role, self._v4_4_route_family, self._v4_4_execution_mode_hint)
            if not role_match and node.node_id not in guarded_ids:
                continue
            touches_failure = (
                not self._v4_4_focus_node_ids
                or node.node_id in self._v4_4_focus_node_ids
                or any(peer in self._v4_4_focus_node_ids for peer in neighbors.get(node.node_id, set()))
            )
            if not touches_failure and node.node_id not in guarded_ids and node.role not in {"aggregator", "router", "judge"}:
                continue
            novelty_key = (node.agent_id, node.role)
            adds_novel_signal = (
                novelty_key not in seen_role_pairs
                or node.node_id in guarded_ids
                or node.node_id in self._v4_4_focus_node_ids
                or self._v4_4_execution_mode_hint == "full"
            )
            if not adds_novel_signal:
                continue
            seen_role_pairs.add(novelty_key)
            selected.append(node)

        if not selected:
            return base_nodes
        selected_ids = {node.node_id for node in selected}
        for node in task_nodes:
            if node.node_id in guarded_ids and node.node_id not in selected_ids:
                selected.append(node)
        return selected

    @staticmethod
    def _public_class_key(key: Tuple[Any, ...]) -> List[str]:
        return [str(item) for item in key]

    @staticmethod
    def _code_patch_locus(feedback: CodeRepairEval) -> str:
        if feedback.failing_examples:
            return str(feedback.failing_examples[0])
        if feedback.failure_kind:
            return str(feedback.failure_kind)
        return "stable"

    @staticmethod
    def _candidate_provenance_node_ids(entry: Dict[str, Any]) -> set[str]:
        node_ids = {
            str(item.get("node_id", ""))
            for item in entry.get("provenance", ())
            if isinstance(item, dict) and str(item.get("node_id", ""))
        }
        if entry.get("origin_node_id"):
            node_ids.add(str(entry.get("origin_node_id", "")))
        node_ids.update(
            str(node_id)
            for node_id in entry.get("anchor_bound_node_ids", ())
            if str(node_id)
        )
        return node_ids

    @staticmethod
    def _stage1_topology_priors(metadata: Optional[dict], graph: UnionGraph) -> List[Tuple[str, float]]:
        payload = dict(metadata or {})
        signatures = list(payload.get("stage1_selected_topology_signatures", ()))
        scores = list(payload.get("stage1_selected_topology_scores", ()))
        pairs: List[Tuple[str, float]] = []
        if signatures and len(signatures) == len(scores):
            for signature, score in zip(signatures, scores):
                text = str(signature).strip()
                if not text:
                    continue
                pairs.append((text, float(score)))
        if pairs:
            return pairs
        fallback = list(payload.get("stage1_source_topology_signatures", ())) or list(graph.source_topology_signatures)
        return [(str(signature), float(len(fallback) - idx)) for idx, signature in enumerate(fallback) if str(signature).strip()]

    @staticmethod
    def _induced_task_edge_ids(graph: UnionGraph, node_ids: Sequence[str]) -> List[str]:
        allowed = {str(node_id) for node_id in node_ids if str(node_id)}
        edge_ids = [
            Stage2RuntimeV44._edge_id(edge)
            for edge in Stage2RuntimeV44._task_edges(graph)
            if edge.src in allowed and edge.dst in allowed
        ]
        return sorted(set(edge_ids))

    def _checker_neighbor_node_ids(self, graph: UnionGraph, seed_node_ids: Sequence[str]) -> set[str]:
        seeds = {str(node_id) for node_id in seed_node_ids if str(node_id)}
        if not seeds:
            return set()
        checker_ids = self._checker_node_ids(self._task_nodes(graph))
        if not checker_ids:
            return set()
        neighbors: set[str] = set()
        for edge in self._task_edges(graph):
            if edge.src in seeds and edge.dst in checker_ids:
                neighbors.add(edge.dst)
            if edge.dst in seeds and edge.src in checker_ids:
                neighbors.add(edge.src)
        return neighbors

    def _shortest_task_path_nodes(
        self,
        graph: UnionGraph,
        *,
        start_ids: Sequence[str],
        target_ids: Sequence[str],
    ) -> List[str]:
        starts = [str(node_id) for node_id in start_ids if str(node_id) in graph.nodes]
        targets = {str(node_id) for node_id in target_ids if str(node_id) in graph.nodes}
        if not starts or not targets:
            return []
        if any(node_id in targets for node_id in starts):
            return []

        adjacency: Dict[str, List[str]] = {}
        for edge in self._task_edges(graph):
            adjacency.setdefault(edge.src, []).append(edge.dst)

        frontier: List[str] = []
        parent: Dict[str, Optional[str]] = {}
        for node_id in starts:
            if node_id in parent:
                continue
            parent[node_id] = None
            frontier.append(node_id)

        found: Optional[str] = None
        index = 0
        while index < len(frontier):
            node_id = frontier[index]
            index += 1
            for next_id in adjacency.get(node_id, ()):
                if next_id in parent:
                    continue
                parent[next_id] = node_id
                if next_id in targets:
                    found = next_id
                    index = len(frontier)
                    break
                frontier.append(next_id)
        if found is None:
            return []

        path: List[str] = []
        cursor: Optional[str] = found
        while cursor is not None:
            path.append(cursor)
            cursor = parent.get(cursor)
        path.reverse()
        return path

    def _legalize_stage1_anchor_recovery_seed(
        self,
        *,
        graph: UnionGraph,
        anchor_entry: Dict[str, Any],
        metadata: Optional[dict],
    ) -> bool:
        if not bool(anchor_entry.get("stage1_anchor", False)):
            return False
        if self._entry_has_recovery_seed_provenance(anchor_entry):
            return True

        priors = self._stage1_topology_priors(metadata, graph)
        if not priors:
            return False
        best_source_graph_id = max(priors, key=lambda item: (float(item[1]), item[0]))[0]
        if not best_source_graph_id:
            return False

        prior_node_ids = {
            node.node_id
            for node in graph.nodes.values()
            if node.node_type == "task" and best_source_graph_id in node.source_graph_ids
        }
        if not prior_node_ids:
            return False

        checker_neighbors = self._checker_neighbor_node_ids(graph, prior_node_ids)
        sink_targets = {
            node_id
            for node_id in set(self._v4_4_sink_guard_ids) | set(graph.sink_node_ids)
            if node_id in graph.nodes and graph.nodes[node_id].node_type == "task"
        }
        bound_node_ids = set(prior_node_ids) | checker_neighbors | set(self._v4_4_sink_guard_ids) | set(self._v4_4_focus_node_ids)
        if sink_targets and not (bound_node_ids & sink_targets):
            bound_node_ids.update(
                self._shortest_task_path_nodes(
                    graph,
                    start_ids=sorted(bound_node_ids),
                    target_ids=sorted(sink_targets),
                )
            )

        if not bound_node_ids:
            return False

        edge_ids = self._induced_task_edge_ids(graph, sorted(bound_node_ids))
        distances = self._sink_distance_lookup(graph)
        origin_type_priority = {
            "proposal": 0,
            "checker": 1,
            "aggregator": 2,
            "sink": 3,
        }
        origin_node_id = min(
            prior_node_ids,
            key=lambda node_id: (
                int(origin_type_priority.get(self._runtime_node_type(graph, graph.nodes[node_id]), 9)),
                int(distances.get(node_id, 10**9)),
                -int(graph.nodes[node_id].support_count),
                node_id,
            ),
        )
        anchor_entry["origin_node_id"] = str(origin_node_id)
        anchor_entry["origin_turn_index"] = 0
        anchor_entry["origin_role"] = str(graph.nodes[origin_node_id].role)
        anchor_entry["candidate_bank_source"] = str(anchor_entry.get("candidate_bank_source") or "stage1_anchor")
        anchor_entry["provenance"] = [
            {
                "node_id": str(node_id),
                "turn_index": 0,
                "role": str(graph.nodes[node_id].role),
                "binding_kind": "best_original_graph_prior",
                "source_graph_id": best_source_graph_id,
            }
            for node_id in sorted(prior_node_ids)
        ]
        anchor_entry["stage1_anchor_binding_legalized"] = True
        anchor_entry["stage1_anchor_binding_kind"] = "best_original_graph_prior"
        anchor_entry["stage1_anchor_binding_source_graph_id"] = best_source_graph_id
        anchor_entry["anchor_bound_node_ids"] = sorted(bound_node_ids)
        anchor_entry["anchor_bound_edge_ids"] = list(edge_ids)
        return True

    def _build_code_recovery_context(
        self,
        *,
        graph: UnionGraph,
        turn_traces: Sequence[TurnTrace],
        target_entry: Dict[str, Any],
        target_feedback: CodeRepairEval,
    ) -> Dict[str, Any]:
        provenance_node_ids = self._candidate_provenance_node_ids(target_entry)
        node_ids = set(provenance_node_ids) | set(self._v4_4_focus_node_ids) | set(self._v4_4_sink_guard_ids)
        edge_ids = [
            str(edge_id)
            for edge_id in target_entry.get("anchor_bound_edge_ids", ())
            if str(edge_id)
        ]
        if not edge_ids:
            edge_ids = self._induced_task_edge_ids(graph, sorted(node_ids))
        verifier_labels: List[str] = []
        if turn_traces:
            latest_turn = turn_traces[-1]
            target_nodes = set(node_ids)
            if not edge_ids:
                for activation in latest_turn.active_edges:
                    if not activation.active:
                        continue
                    if activation.src in target_nodes or activation.dst in target_nodes:
                        edge_ids.append(str(activation.edge_id))
            for event in latest_turn.feedback_events:
                if event.target_node_id in target_nodes:
                    verifier_labels.append(str(event.event_type))
        return {
            "recovery_subgraph_node_ids": sorted(node_ids),
            "recovery_subgraph_edge_ids": sorted(set(edge_ids)),
            "trigger_verifier_snapshot": {
                "labels": sorted(set(verifier_labels)),
                "failure_kind": str(target_feedback.failure_kind),
                "passed": int(target_feedback.passed),
                "total": int(target_feedback.total),
            },
        }

    @staticmethod
    def _recovery_entry_has_required_fields(entry: Dict[str, Any]) -> bool:
        return bool(
            entry.get("origin_node_id")
            and int(entry.get("origin_turn_index", -1)) >= 0
            and entry.get("origin_role")
            and entry.get("parent_candidate_digest")
            and entry.get("repair_operator_type")
            and list(entry.get("recovery_subgraph_node_ids", ()))
            and list(entry.get("provenance", ()))
            and dict(entry.get("trigger_verifier_snapshot", {}))
        )

    @staticmethod
    def _entry_has_recovery_seed_provenance(entry: Dict[str, Any]) -> bool:
        return bool(
            entry.get("origin_node_id")
            and int(entry.get("origin_turn_index", -1)) >= 0
            and entry.get("origin_role")
            and list(entry.get("provenance", ()))
        )

    def _assert_recovery_entry_invariants(self, entry: Dict[str, Any]) -> None:
        if not entry.get("repair_branch"):
            return
        if not bool(entry.get("recovery_reinserted", False)):
            raise ValueError("Selected recovery output must be reinserted before final selection.")
        if not self._recovery_entry_has_required_fields(entry):
            raise ValueError("Selected recovery output is missing required provenance fields.")

    def _verify_code_pool(
        self,
        *,
        anchor: Optional[Dict[str, Any]],
        candidates: Sequence[Dict[str, Any]],
        metadata: Optional[dict],
    ) -> Tuple[List[Tuple[Dict[str, Any], CodeRepairEval]], Optional[Tuple[Dict[str, Any], CodeRepairEval]]]:
        pool: List[Tuple[Dict[str, Any], CodeRepairEval]] = []
        seen: set[str] = set()
        ordered: List[Dict[str, Any]] = []
        if anchor is not None:
            ordered.append(anchor)
        ordered.extend(list(candidates))
        anchor_pair = None
        anchor_digest = str((anchor or {}).get("digest", ""))
        for entry in ordered:
            digest = str(entry.get("digest", ""))
            if digest in seen:
                continue
            seen.add(digest)
            verified_entry, feedback = self._prepare_verified_entry(str(entry.get("text", "")), metadata=metadata, parent=entry)
            self._ensure_v4_4_entry_fields(verified_entry)
            verified_entry["v4_4_route_family"] = "code_repair"
            pool.append((verified_entry, feedback))
            if digest and digest == anchor_digest:
                anchor_pair = (verified_entry, feedback)
        pool.sort(key=lambda item: self._verified_rank_key(item[0], item[1]), reverse=True)
        return pool, anchor_pair

    def _collapse_code_classes(
        self,
        pool: Sequence[Tuple[Dict[str, Any], CodeRepairEval]],
        *,
        anchor_digest: str,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for entry, feedback in pool:
            key = (
                bool(feedback.syntax_ok),
                bool(feedback.entry_point_ok),
                int(feedback.passed),
                int(feedback.total),
                str(feedback.failure_kind),
                self._code_patch_locus(feedback),
            )
            payload = grouped.get(key)
            rank_key = (
                self._verified_rank_key(entry, feedback),
                *self._stable_entry_tiebreak(entry),
            )
            if payload is None:
                payload = {
                    "key": key,
                    "representative": entry,
                    "feedback": feedback,
                    "size": 0,
                    "contains_anchor": False,
                    "repair_locus": key[-1],
                    "rank_key": rank_key,
                }
                grouped[key] = payload
            payload["size"] += 1
            payload["contains_anchor"] = bool(payload["contains_anchor"] or str(entry.get("digest", "")) == anchor_digest)
            if rank_key > payload["rank_key"]:
                payload["representative"] = entry
                payload["feedback"] = feedback
                payload["rank_key"] = rank_key

        collapsed = list(grouped.values())
        collapsed.sort(
            key=lambda item: (
                self._verified_rank_key(item["representative"], item["feedback"]),
                *self._stable_entry_tiebreak(item["representative"]),
            ),
            reverse=True,
        )
        for item in collapsed:
            rep = item["representative"]
            self._ensure_v4_4_entry_fields(rep)
            rep["v4_4_class_key"] = tuple(item["key"])
            rep["v4_4_class_size"] = int(item["size"])
            rep["v4_4_route_family"] = "code_repair"
        return collapsed

    def _reasoning_major_flags(
        self,
        entry: Dict[str, Any],
        *,
        normalized_answer: str,
    ) -> Tuple[str, ...]:
        major: List[str] = []
        if not normalized_answer:
            return tuple(major)
        if float(self._review_advantage(entry)) < 0.0:
            major.append("review_pressure")
        if float(self._sink_ratio(entry)) <= 0.0:
            major.append("no_sink_support")
        if float(self._source_diversity(entry)) <= 0.0:
            major.append("single_source")
        return tuple(sorted(set(major)))

    def _evaluate_reasoning_candidate(
        self,
        *,
        question_text: str,
        entry: Dict[str, Any],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> ReasoningEval:
        text = str(entry.get("text", "")).strip()
        task_type = str(dataset_profile.task_type)
        fatal: List[str] = []
        normalized_answer = ""
        contradiction_cluster = "stable"
        if task_type == "mcq":
            normalized = self.evaluator._normalize_mcq_output(question_text, text, metadata=metadata)
            option, _ = self.evaluator._parse_mcq_answer(normalized)
            if option is None:
                fatal.append("invalid_option")
                contradiction_cluster = "invalid_option"
            else:
                normalized_answer = f"OPTION - {option}"
        elif task_type == "numeric":
            value = self.evaluator._extract_prediction_target(text)
            if value is None:
                fatal.append("missing_numeric_target")
                contradiction_cluster = "missing_numeric_target"
            else:
                normalized_answer = str(value)
        elif task_type == "math_expression":
            expr = self.evaluator._normalize_math_expression(text)
            if not expr:
                fatal.append("empty_math_expression")
                contradiction_cluster = "empty_math_expression"
            else:
                normalized_answer = expr
        elif task_type == "boolean":
            value = self.evaluator._normalize_yes_no_output(text)
            if value not in {"yes", "no"}:
                fatal.append("unsupported_yes_no")
                contradiction_cluster = "unsupported_yes_no"
            else:
                normalized_answer = value
        else:
            cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text).strip()
            if not cleaned:
                fatal.append("empty_answer")
                contradiction_cluster = "empty_answer"
            else:
                lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
                normalized_answer = lines[-1] if lines else cleaned

        major = self._reasoning_major_flags(entry, normalized_answer=normalized_answer)
        critical_support = 0
        if int(entry.get("sink_support", 0)) > 0:
            critical_support += 1
        if float(self._review_advantage(entry)) >= 0.0:
            critical_support += 1
        if float(self._source_diversity(entry)) > 0.0:
            critical_support += 1
        return ReasoningEval(
            normalized_answer=normalized_answer,
            fatal_contradictions=tuple(sorted(set(fatal))),
            major_contradictions=major,
            critical_support=critical_support,
            contradiction_cluster=contradiction_cluster,
        )

    def _collapse_reasoning_classes(
        self,
        *,
        question_text: str,
        candidates: Sequence[Dict[str, Any]],
        anchor_digest: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for entry in candidates:
            self._ensure_v4_4_entry_fields(entry)
            evaluation = self._evaluate_reasoning_candidate(
                question_text=question_text,
                entry=entry,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            key = evaluation.class_key
            payload = grouped.get(key)
            rank_key = (
                evaluation.lexicographic_key,
                *self._stable_entry_tiebreak(entry),
            )
            if payload is None:
                payload = {
                    "key": key,
                    "representative": entry,
                    "evaluation": evaluation,
                    "size": 0,
                    "contains_anchor": False,
                    "rank_key": rank_key,
                }
                grouped[key] = payload
            payload["size"] += 1
            payload["contains_anchor"] = bool(payload["contains_anchor"] or str(entry.get("digest", "")) == anchor_digest)
            if rank_key > payload["rank_key"]:
                payload["representative"] = entry
                payload["evaluation"] = evaluation
                payload["rank_key"] = rank_key
        collapsed = list(grouped.values())
        collapsed.sort(
            key=lambda item: (
                item["evaluation"].lexicographic_key,
                *self._stable_entry_tiebreak(item["representative"]),
            ),
            reverse=True,
        )
        for item in collapsed:
            rep = item["representative"]
            rep["v4_4_class_key"] = tuple(item["key"])
            rep["v4_4_class_size"] = int(item["size"])
            rep["v4_4_route_family"] = "adversarial"
        return collapsed

    def _attach_deductive_eval(self, entry: Dict[str, Any], evaluation: DeductiveEval) -> None:
        self._ensure_v4_4_entry_fields(entry)
        entry["v4_4_route_family"] = "deductive_reasoning"
        entry["deductive_eval"] = evaluation
        entry["deductive_norm_answer"] = str(evaluation.artifact.normalized_final_answer or "")
        entry["deductive_first_bad_step"] = evaluation.residual.first_bad_step
        entry["deductive_verified_prefix_len"] = int(evaluation.residual.verified_prefix_len)
        entry["deductive_residual_kind"] = str(evaluation.residual.residual_kind or "")
        entry["deductive_final_consistent"] = bool(evaluation.residual.final_consistent)
        entry["deductive_repair_locus"] = str(evaluation.residual.repair_locus or "")
        entry["deductive_parser_confidence"] = str(evaluation.artifact.parser_confidence)
        entry["deductive_final_source"] = str(evaluation.artifact.final_source)
        entry["deductive_step_source"] = str(evaluation.artifact.step_source)
        entry["deductive_contract_ok"] = bool(evaluation.artifact.contract_ok)
        entry["deductive_checked_equation_count"] = int(evaluation.residual.checked_equation_count)
        entry["deductive_fatal_count"] = int(len(evaluation.residual.fatal))
        entry["deductive_local_count"] = int(len(evaluation.residual.local))
        entry["v4_4_class_key"] = repr(evaluation.class_key)

    def _evaluate_deductive_candidate(
        self,
        *,
        question_text: str,
        entry: Dict[str, Any],
        dataset_profile: DatasetProfile,
        metadata: Optional[dict],
    ) -> DeductiveEval:
        artifact = parse_deductive_artifact(
            str(entry.get("text", "")),
            dataset_name=self._dataset_name(dataset_profile, metadata),
        )
        residual = verify_deductive_artifact(
            question_text,
            artifact,
            dataset_profile,
            dict(metadata or {}),
        )
        return make_deductive_eval(artifact, residual, entry)

    def _prepare_deductive_verified_entry(
        self,
        text: str,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        parent: Optional[Dict[str, Any]] = None,
        repair_agent_id: str = "",
        repair_round: int = -1,
        recovery_subgraph_node_ids: Optional[Sequence[str]] = None,
        recovery_subgraph_edge_ids: Optional[Sequence[str]] = None,
        trigger_verifier_snapshot: Optional[Dict[str, Any]] = None,
        repair_operator_type: str = "",
    ) -> Tuple[Dict[str, Any], DeductiveEval]:
        entry = self._init_candidate_entry(text)
        if parent is not None:
            entry["stage1_anchor"] = bool(parent.get("stage1_anchor", False))
            entry["source_roles"] = set(parent.get("source_roles", set()))
            entry["source_node_ids"] = set(parent.get("source_node_ids", set()))
            entry["turn_indices"] = set(parent.get("turn_indices", set()))
            entry["candidate_model_score"] = float(parent.get("candidate_model_score", 0.5))
            entry["candidate_model_uncertainty"] = float(parent.get("candidate_model_uncertainty", 0.2))
            entry["support_score"] = float(parent.get("support_score", 0.5))
            entry["sink_support"] = int(parent.get("sink_support", 0))
            entry["occurrence_count"] = int(parent.get("occurrence_count", 0))
            entry["feedback_pass"] = float(parent.get("feedback_pass", 0.0))
            entry["feedback_challenge"] = float(parent.get("feedback_challenge", 0.0))
            entry["feedback_uncertain"] = float(parent.get("feedback_uncertain", 0.0))
            entry["feedback_pass_calibrated"] = float(parent.get("feedback_pass_calibrated", 0.0))
            entry["feedback_challenge_calibrated"] = float(parent.get("feedback_challenge_calibrated", 0.0))
            entry["feedback_uncertain_calibrated"] = float(parent.get("feedback_uncertain_calibrated", 0.0))
            entry["reviewer_event_count"] = float(parent.get("reviewer_event_count", 0.0))
            entry["reviewer_mean_trust"] = float(parent.get("reviewer_mean_trust", 0.5))
        self._ensure_v4_4_entry_fields(entry)
        if parent is not None:
            entry["origin_node_id"] = str(parent.get("origin_node_id", ""))
            entry["origin_turn_index"] = int(parent.get("origin_turn_index", -1))
            entry["origin_role"] = str(parent.get("origin_role", ""))
            entry["candidate_bank_source"] = str(parent.get("candidate_bank_source", ""))
            entry["provenance"] = [dict(item) for item in parent.get("provenance", ()) if isinstance(item, dict)]
            entry["parent_candidate_digest"] = str(parent.get("parent_candidate_digest", ""))
            entry["repair_operator_type"] = str(parent.get("repair_operator_type", ""))
            entry["recovery_subgraph_node_ids"] = list(parent.get("recovery_subgraph_node_ids", ()))
            entry["recovery_subgraph_edge_ids"] = list(parent.get("recovery_subgraph_edge_ids", ()))
            entry["trigger_verifier_snapshot"] = dict(parent.get("trigger_verifier_snapshot", {}))
            entry["recovery_reinserted"] = bool(parent.get("recovery_reinserted", False))
        if repair_agent_id:
            entry["repair_branch"] = True
            entry["repair_agent_id"] = repair_agent_id
            entry["repair_round"] = repair_round
            entry["repair_parent_digest"] = str((parent or {}).get("digest", ""))
            entry["parent_candidate_digest"] = str((parent or {}).get("digest", ""))
            entry["repair_operator_type"] = repair_operator_type
            entry["deductive_repair_operator_type"] = repair_operator_type
            entry["recovery_subgraph_node_ids"] = list(recovery_subgraph_node_ids or ())
            entry["recovery_subgraph_edge_ids"] = list(recovery_subgraph_edge_ids or ())
            entry["trigger_verifier_snapshot"] = dict(trigger_verifier_snapshot or {})
            entry["recovery_reinserted"] = False
            if not entry.get("origin_node_id") and entry["recovery_subgraph_node_ids"]:
                entry["origin_node_id"] = str(entry["recovery_subgraph_node_ids"][0])
            if int(entry.get("origin_turn_index", -1)) < 0:
                entry["origin_turn_index"] = 0
            if not entry.get("origin_role"):
                entry["origin_role"] = str((parent or {}).get("origin_role") or "solver")
            if not entry.get("provenance"):
                entry["provenance"] = [
                    {
                        "node_id": str(entry.get("origin_node_id", "")),
                        "turn_index": int(entry.get("origin_turn_index", 0)),
                        "role": str(entry.get("origin_role", "")),
                        "binding_kind": "deductive_recovery_seed",
                    }
                ]
        evaluation = self._evaluate_deductive_candidate(
            question_text=question_text,
            entry=entry,
            dataset_profile=dataset_profile,
            metadata=metadata,
        )
        self._attach_deductive_eval(entry, evaluation)
        return entry, evaluation

    def _verify_deductive_pool(
        self,
        question_text: str,
        anchor_entry: Optional[Dict[str, Any]],
        candidate_entries: Sequence[Dict[str, Any]],
        dataset_profile: DatasetProfile,
        metadata: Optional[dict],
    ) -> List[Dict[str, Any]]:
        verified: List[Dict[str, Any]] = []
        seen: set[str] = set()
        ordered: List[Dict[str, Any]] = []
        if anchor_entry is not None:
            ordered.append(anchor_entry)
        ordered.extend(list(candidate_entries))
        for entry in ordered:
            digest = str(entry.get("digest", ""))
            if digest in seen:
                continue
            seen.add(digest)
            evaluation = self._evaluate_deductive_candidate(
                question_text=question_text,
                entry=entry,
                dataset_profile=dataset_profile,
                metadata=metadata,
            )
            self._attach_deductive_eval(entry, evaluation)
            verified.append(entry)
        verified.sort(key=self._deductive_rank_key, reverse=True)
        return verified

    def _deductive_rank_key(self, entry: Dict[str, Any]) -> tuple:
        fatal_count = int(entry.get("deductive_fatal_count", 0))
        local_count = int(entry.get("deductive_local_count", 0))
        checked_equations = int(entry.get("deductive_checked_equation_count", 0))
        final_source = str(entry.get("deductive_final_source") or "")
        explicit_final = final_source in {"final_line", "gsm_hash", "boxed"}

        return (
            -fatal_count,
            int(explicit_final),
            int(bool(entry.get("deductive_contract_ok", False))),
            int(bool(entry.get("deductive_final_consistent", False))),
            -local_count,
            checked_equations,
            int(bool(entry.get("deductive_norm_answer"))),
            int(bool(entry.get("stage1_anchor", False))),
            str(entry.get("digest", "")),
        )

    def _collapse_deductive_classes(self, verified_entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        groups: Dict[str, List[Dict[str, Any]]] = {}
        for entry in verified_entries:
            key = str(entry.get("v4_4_class_key", ""))
            groups.setdefault(key, []).append(entry)

        reps: List[Dict[str, Any]] = []
        for key, members in groups.items():
            members.sort(
                key=lambda item: (
                    self._deductive_rank_key(item),
                    *self._stable_entry_tiebreak(item),
                ),
                reverse=True,
            )
            rep = members[0]
            rep["v4_4_class_key"] = key
            rep["v4_4_class_size"] = len(members)
            reps.append(rep)

        reps.sort(
            key=lambda item: (
                self._deductive_rank_key(item),
                *self._stable_entry_tiebreak(item),
            ),
            reverse=True,
        )
        return reps

    def _deductive_mode(
        self,
        *,
        anchor_eval: Optional[DeductiveEval],
        collapsed: Sequence[Dict[str, Any]],
        budget_bucket: str,
    ) -> str:
        if anchor_eval is not None and not anchor_eval.residual.fatal and not any(
            deductive_dominates(item["deductive_eval"], anchor_eval) for item in collapsed if "deductive_eval" in item
        ):
            return self._apply_budget_bucket("bypass", budget_bucket)
        if len(collapsed) >= 2:
            return self._apply_budget_bucket("full", budget_bucket)
        if anchor_eval is not None and anchor_eval.residual.fatal:
            return self._apply_budget_bucket("lean", budget_bucket)
        return self._apply_budget_bucket("lean", budget_bucket)

    def _build_deductive_recovery_context(
        self,
        graph: UnionGraph,
        turn_traces: Sequence[TurnTrace],
        target_entry: Dict[str, Any],
        target_eval: DeductiveEval,
    ) -> dict:
        provenance_node_ids = self._candidate_provenance_node_ids(target_entry)
        node_ids = set(provenance_node_ids) | set(self._v4_4_focus_node_ids) | set(self._v4_4_sink_guard_ids)
        if not node_ids:
            node_ids = set(str(node_id) for node_id in target_entry.get("source_node_ids", ()) if str(node_id))
        edge_ids = [
            str(edge_id)
            for edge_id in target_entry.get("anchor_bound_edge_ids", ())
            if str(edge_id)
        ]
        if not edge_ids and node_ids:
            edge_ids = self._induced_task_edge_ids(graph, sorted(node_ids))
        verifier_labels: List[str] = []
        if turn_traces:
            latest_turn = turn_traces[-1]
            target_nodes = set(node_ids)
            if not edge_ids:
                for activation in latest_turn.active_edges:
                    if not activation.active:
                        continue
                    if activation.src in target_nodes or activation.dst in target_nodes:
                        edge_ids.append(str(activation.edge_id))
            for event in latest_turn.feedback_events:
                if event.target_node_id in target_nodes:
                    verifier_labels.append(str(event.event_type))
        verified_prefix_len = int(target_eval.residual.verified_prefix_len)
        first_bad_step = target_eval.residual.first_bad_step
        stable_loci = [f"step_{index}" for index in range(1, verified_prefix_len + 1)]
        unstable_locus = f"step_{first_bad_step}" if first_bad_step is not None else str(target_eval.residual.repair_locus or "")
        return {
            "morph_type": "deductive_reasoning",
            "recovery_subgraph_node_ids": sorted(node_ids),
            "recovery_subgraph_edge_ids": sorted(set(edge_ids)),
            "trigger_verifier_snapshot": {
                "labels": sorted(set(verifier_labels)),
                "residual_kind": str(target_eval.residual.residual_kind or ""),
                "first_bad_step": target_eval.residual.first_bad_step,
                "verified_prefix_len": int(target_eval.residual.verified_prefix_len),
                "normalized_answer": str(target_eval.artifact.normalized_final_answer or ""),
                "final_consistent": bool(target_eval.residual.final_consistent),
                "fatal_count": int(len(target_eval.residual.fatal)),
                "local_count": int(len(target_eval.residual.local)),
                "final_source": str(target_eval.artifact.final_source),
                "contract_ok": bool(target_eval.artifact.contract_ok),
                "checked_equation_count": int(target_eval.residual.checked_equation_count),
            },
            "stable_loci": stable_loci,
            "unstable_locus": unstable_locus,
            "repair_locus": str(target_eval.residual.repair_locus or ""),
        }

    def _generate_deductive_repair_branches(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        current_entry: Dict[str, Any],
        current_eval: DeductiveEval,
        repair_round: int,
        recovery_context: Dict[str, Any],
    ) -> List[Tuple[Dict[str, Any], DeductiveEval]]:
        operator_type = repair_operator_for_residual(current_eval)
        if not operator_type:
            return []
        if operator_type in {"deductive_algebra_suffix_patch", "deductive_transition_suffix_patch"} and current_eval.artifact.parser_confidence == "low":
            return []
        prompt = build_deductive_repair_prompt(
            operator_type=operator_type,
            question_text=render_question_text(question_text, metadata=metadata),
            artifact=current_eval.artifact,
            residual=current_eval.residual,
        )
        seen: set[str] = {str(current_entry.get("text", "")).strip()}
        branches: List[Tuple[Dict[str, Any], DeductiveEval]] = []
        for agent_id in self._select_graph_repair_agents():
            agent = self._by_id[agent_id]
            system_prompt = build_system_prompt(agent, self._graph_repair_slots(), extra_role_hint=operator_type)
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            repaired = self._sanitize_candidate(question_text, raw_output, metadata=metadata)
            if not repaired or repaired in seen:
                continue
            seen.add(repaired)
            branch_entry, branch_eval = self._prepare_deductive_verified_entry(
                repaired,
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                parent=current_entry,
                repair_agent_id=agent_id,
                repair_round=repair_round,
                recovery_subgraph_node_ids=recovery_context.get("recovery_subgraph_node_ids", ()),
                recovery_subgraph_edge_ids=recovery_context.get("recovery_subgraph_edge_ids", ()),
                trigger_verifier_snapshot=recovery_context.get("trigger_verifier_snapshot", {}),
                repair_operator_type=operator_type,
            )
            branches.append((branch_entry, branch_eval))
        branches.sort(
            key=lambda item: (
                item[1].rank_key,
                *self._stable_entry_tiebreak(item[0]),
            ),
            reverse=True,
        )
        return branches

    @staticmethod
    def _deductive_eval_for_entry(entry: Optional[Dict[str, Any]]) -> Optional[DeductiveEval]:
        if not entry:
            return None
        evaluation = entry.get("deductive_eval")
        return evaluation if isinstance(evaluation, DeductiveEval) else None

    def _deductive_is_safe_improvement(self, new_entry: Dict[str, Any], old_entry: Optional[Dict[str, Any]]) -> bool:
        new_eval = self._deductive_eval_for_entry(new_entry)
        old_eval = self._deductive_eval_for_entry(old_entry)
        if new_eval is None:
            return False
        if old_eval is None:
            return True
        return deductive_dominates(new_eval, old_eval)

    def _deductive_has_deterministic_fatal(self, evaluation: DeductiveEval) -> bool:
        return _has_deterministic_fatal(evaluation)

    def _deductive_has_repairable_deterministic_fatal(self, evaluation: DeductiveEval) -> bool:
        if not self._deductive_has_deterministic_fatal(evaluation):
            return False
        operator_type = repair_operator_for_residual(evaluation)
        return bool(operator_type and operator_type != "deductive_transition_suffix_patch")

    def _deductive_anchor_is_safe(self, evaluation: DeductiveEval) -> bool:
        if not evaluation.artifact.normalized_final_answer:
            return False
        if self._deductive_has_deterministic_fatal(evaluation):
            return False
        return True

    def _deductive_entry_is_explicit_clean_witness(self, entry: Dict[str, Any]) -> bool:
        final_source = str(entry.get("deductive_final_source") or "")
        if final_source not in {"final_line", "gsm_hash", "boxed"}:
            return False
        if not bool(entry.get("deductive_contract_ok", False)):
            return False
        if int(entry.get("deductive_fatal_count", 0)) != 0:
            return False
        if not bool(entry.get("deductive_final_consistent", False)):
            return False
        if int(entry.get("deductive_checked_equation_count", 0)) <= 0:
            return False
        return bool(entry.get("deductive_norm_answer"))

    def _best_same_answer_deductive_candidate(
        self,
        collapsed: Sequence[Dict[str, Any]],
        anchor_eval: DeductiveEval,
    ) -> Optional[Dict[str, Any]]:
        anchor_answer = str(anchor_eval.artifact.normalized_final_answer or "")
        if not anchor_answer:
            return None
        for entry in collapsed:
            if str(entry.get("deductive_norm_answer") or "") == anchor_answer:
                return entry
        return None

    def _best_cross_answer_certificate(
        self,
        collapsed: Sequence[Dict[str, Any]],
        anchor_eval: DeductiveEval,
    ) -> Optional[Dict[str, Any]]:
        anchor_answer = str(anchor_eval.artifact.normalized_final_answer or "")
        classes: Dict[str, List[Dict[str, Any]]] = {}
        for entry in collapsed:
            answer = str(entry.get("deductive_norm_answer") or "")
            if not answer:
                continue
            classes.setdefault(answer, []).append(entry)

        anchor_class = classes.get(anchor_answer, [])
        if any(self._deductive_entry_is_explicit_clean_witness(entry) for entry in anchor_class):
            return None

        best_candidate: Optional[Dict[str, Any]] = None
        best_key: tuple = ()
        for answer, entries in classes.items():
            if answer == anchor_answer:
                continue
            witnesses = [entry for entry in entries if self._deductive_entry_is_explicit_clean_witness(entry)]
            independent_origins = {
                str(entry.get("origin_node_id") or entry.get("origin_role") or entry.get("digest") or "")
                for entry in witnesses
            }
            if len(independent_origins) < 2:
                continue
            rep = sorted(witnesses, key=self._deductive_rank_key, reverse=True)[0]
            key = self._deductive_rank_key(rep)
            if best_candidate is None or key > best_key:
                best_candidate = rep
                best_key = key
        return best_candidate

    def _deductive_anchor_or_best_non_regressive(
        self,
        anchor_entry: Optional[Dict[str, Any]],
        collapsed: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if anchor_entry is not None:
            return anchor_entry
        return collapsed[0]

    def _select_best_deductive_candidate(
        self,
        question_text: str,
        anchor_entry: Optional[Dict[str, Any]],
        candidate_entries: Sequence[Dict[str, Any]],
        dataset_profile: DatasetProfile,
        metadata: Optional[dict],
        *,
        graph: Optional[UnionGraph] = None,
        turn_traces: Sequence[TurnTrace] = (),
        budget_bucket: str = "normal",
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        repair_rounds_run = 0
        repair_branch_count = 0
        repair_accepted = 0
        repair_rejected_by_dominance = 0
        recovery_target_digest = ""
        recovery_subgraph_node_count = 0
        recovery_subgraph_edge_count = 0
        anchor_repair_entries: List[Dict[str, Any]] = []
        selected_anchor_repair: Optional[Dict[str, Any]] = None

        if anchor_entry is not None:
            anchor_eval_initial = self._evaluate_deductive_candidate(
                question_text=question_text,
                entry=anchor_entry,
                dataset_profile=dataset_profile,
                metadata=metadata,
            )
            self._attach_deductive_eval(anchor_entry, anchor_eval_initial)
            if (
                graph is not None
                and self._deductive_has_repairable_deterministic_fatal(anchor_eval_initial)
            ):
                recovery_target_digest = str(anchor_entry.get("digest", ""))
                recovery_context = self._build_deductive_recovery_context(
                    graph,
                    turn_traces,
                    anchor_entry,
                    anchor_eval_initial,
                )
                recovery_subgraph_node_count = len(recovery_context["recovery_subgraph_node_ids"])
                recovery_subgraph_edge_count = len(recovery_context["recovery_subgraph_edge_ids"])
                repair_rounds_run = 1
                branches = self._generate_deductive_repair_branches(
                    question_text=question_text,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                    current_entry=anchor_entry,
                    current_eval=anchor_eval_initial,
                    repair_round=repair_rounds_run,
                    recovery_context=recovery_context,
                )
                repair_branch_count += len(branches)
                approved: List[Dict[str, Any]] = []
                for branch_entry, branch_eval in branches:
                    if not deductive_dominates(branch_eval, anchor_eval_initial):
                        branch_entry["deductive_repair_rejected_by_dominance"] = True
                        repair_rejected_by_dominance += 1
                        continue
                    if str(branch_entry.get("repair_operator_type", "")) == "deductive_transition_suffix_patch":
                        branch_entry["deductive_repair_rejected_by_dominance"] = True
                        repair_rejected_by_dominance += 1
                        continue
                    branch_entry["deductive_repair_accepted"] = True
                    branch_entry["recovery_reinserted"] = True
                    branch_entry["candidate_bank_source"] = "recovery_output"
                    self._assert_recovery_entry_invariants(branch_entry)
                    repair_accepted += 1
                    approved.append(branch_entry)
                if approved:
                    approved.sort(key=self._deductive_rank_key, reverse=True)
                    anchor_repair_entries.extend(approved)
                    selected_anchor_repair = approved[0]

        verified = self._verify_deductive_pool(
            question_text,
            anchor_entry,
            list(candidate_entries) + anchor_repair_entries,
            dataset_profile,
            metadata,
        )
        if not verified:
            return anchor_entry, "v4_4_deductive_empty_bank", {
                "v4_4_protocol_family": "deductive_reasoning",
                "v4_4_execution_mode": "bypass",
                "v4_4_budget_bucket": budget_bucket,
                "v4_4_collapsed_class_count": 0,
            }
        collapsed = self._collapse_deductive_classes(verified)
        anchor_digest = str((anchor_entry or {}).get("digest", ""))
        anchor_verified = next((entry for entry in verified if str(entry.get("digest", "")) == anchor_digest), None)
        anchor_eval = self._deductive_eval_for_entry(anchor_verified)
        execution_mode = self._deductive_mode(anchor_eval=anchor_eval, collapsed=collapsed, budget_bucket=budget_bucket)

        best = collapsed[0] if collapsed else anchor_verified
        selected_entry = best
        selected_reason = "v4_4_deductive_best_available" if best is not None else "v4_4_deductive_empty_bank"
        if selected_anchor_repair is not None:
            selected_entry = selected_anchor_repair
            selected_reason = "v4_4_deductive_override_anchor_repair"
        elif anchor_verified is not None and anchor_eval is not None:
            if self._deductive_anchor_is_safe(anchor_eval):
                same_answer = self._best_same_answer_deductive_candidate(collapsed, anchor_eval)
                if (
                    same_answer is not None
                    and same_answer is not anchor_verified
                    and self._deductive_is_safe_improvement(same_answer, anchor_verified)
                ):
                    selected_entry = same_answer
                    selected_reason = "v4_4_deductive_same_answer_enrichment"
                else:
                    certified = self._best_cross_answer_certificate(collapsed, anchor_eval)
                    if certified is not None:
                        selected_entry = certified
                        selected_reason = "v4_4_deductive_override_answer_class_certificate"
                    else:
                        selected_entry = anchor_verified
                        selected_reason = "v4_4_deductive_preserve_anchor_no_deterministic_fatal"
            elif best is not None and self._deductive_is_safe_improvement(best, anchor_verified):
                selected_entry = best
                selected_reason = "v4_4_deductive_override_residual_dominance"
            else:
                selected_entry = anchor_verified
                selected_reason = "v4_4_deductive_anchor_guard_preserve"

        selected_eval = self._deductive_eval_for_entry(selected_entry)
        if (
            anchor_verified is not None
            and selected_entry is not anchor_verified
            and selected_reason != "v4_4_deductive_override_answer_class_certificate"
            and not self._deductive_is_safe_improvement(selected_entry, anchor_verified)
        ):
            selected_entry = anchor_verified
            selected_eval = self._deductive_eval_for_entry(selected_entry)
            selected_reason = "v4_4_deductive_anchor_guard_preserve"

        selected_eval = selected_eval or self._deductive_eval_for_entry(best)
        parser_coverage = (
            float(sum(1 for entry in verified if str(entry.get("deductive_parser_confidence", "")) != "low")) / float(len(verified))
            if verified
            else 0.0
        )
        extra = {
            "v4_4_protocol_family": "deductive_reasoning",
            "v4_4_execution_mode": execution_mode,
            "v4_4_budget_bucket": budget_bucket,
            "v4_4_stage1_anchor_present": bool(anchor_verified is not None),
            "v4_4_stage1_anchor_used": bool(anchor_verified is not None and str(selected_entry.get("digest", "")) == str(anchor_verified.get("digest", ""))),
            "v4_4_candidate_count": int(len(candidate_entries)),
            "v4_4_collapsed_class_count": int(len(collapsed)),
            "v4_4_selected_candidate_digest": str(selected_entry.get("digest", "")),
            "v4_4_selected_candidate_source": self._candidate_source_label(selected_entry),
            "v4_4_selected_quality_score": float(self._quality_score(selected_entry)),
            "v4_4_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
            "deductive_parser_coverage": float(parser_coverage),
            "deductive_selected_residual_kind": str((selected_eval.residual.residual_kind if selected_eval else "") or ""),
            "deductive_selected_first_bad_step": (selected_eval.residual.first_bad_step if selected_eval else None),
            "deductive_selected_verified_prefix_len": int(selected_eval.residual.verified_prefix_len if selected_eval else 0),
            "deductive_selected_final_consistent": bool(selected_eval.residual.final_consistent if selected_eval else False),
            "deductive_selected_final_source": str(selected_eval.artifact.final_source if selected_eval else ""),
            "deductive_selected_contract_ok": bool(selected_eval.artifact.contract_ok if selected_eval else False),
            "deductive_selected_checked_equation_count": int(selected_eval.residual.checked_equation_count if selected_eval else 0),
            "deductive_repair_operator_type": str(selected_entry.get("deductive_repair_operator_type", selected_entry.get("repair_operator_type", ""))),
            "deductive_repair_accepted": bool(selected_entry.get("deductive_repair_accepted", False)),
            "deductive_repair_rejected_by_dominance": bool(repair_rejected_by_dominance > 0),
            "v4_4_repair_rounds_run": int(repair_rounds_run),
            "v4_4_repair_branch_count": int(repair_branch_count),
            "v4_4_repair_improvement_count": int(repair_accepted),
            "v4_4_recovery_target_digest": recovery_target_digest,
            "v4_4_recovery_subgraph_node_count": int(recovery_subgraph_node_count),
            "v4_4_recovery_subgraph_edge_count": int(recovery_subgraph_edge_count),
            "v4_4_top_classes": [
                {
                    "class_key": [str(item.get("v4_4_class_key", ""))],
                    "size": int(item.get("v4_4_class_size", 1)),
                    "contains_anchor": bool(anchor_digest and str(item.get("digest", "")) == anchor_digest),
                    "representative_digest": str(item.get("digest", "")),
                    "residual_kind": str(item.get("deductive_residual_kind", "")),
                    "first_bad_step": item.get("deductive_first_bad_step", None),
                    "verified_prefix_len": int(item.get("deductive_verified_prefix_len", 0)),
                    "final_consistent": bool(item.get("deductive_final_consistent", False)),
                    "parser_confidence": str(item.get("deductive_parser_confidence", "")),
                    "final_source": str(item.get("deductive_final_source", "")),
                    "contract_ok": bool(item.get("deductive_contract_ok", False)),
                    "checked_equation_count": int(item.get("deductive_checked_equation_count", 0)),
                }
                for item in collapsed[: self.config.max_logged_candidates]
            ],
        }
        if anchor_verified is not None and anchor_eval is not None:
            extra.update(
                {
                    "v4_4_stage1_anchor_digest": str(anchor_verified.get("digest", "")),
                    "v4_4_stage1_anchor_residual_kind": str(anchor_eval.residual.residual_kind or ""),
                    "v4_4_stage1_anchor_first_bad_step": anchor_eval.residual.first_bad_step,
                    "v4_4_stage1_anchor_verified_prefix_len": int(anchor_eval.residual.verified_prefix_len),
                    "v4_4_stage1_anchor_final_consistent": bool(anchor_eval.residual.final_consistent),
                    "v4_4_stage1_anchor_final_source": str(anchor_eval.artifact.final_source),
                    "v4_4_stage1_anchor_contract_ok": bool(anchor_eval.artifact.contract_ok),
                    "v4_4_stage1_anchor_checked_equation_count": int(anchor_eval.residual.checked_equation_count),
                }
            )
        return selected_entry, selected_reason, extra

    def _evaluate_graph_candidate(
        self,
        *,
        question_text: str,
        entry: Dict[str, Any],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> GraphConstraintEval:
        text = str(entry.get("text", "")).strip()
        cleaned = MultiFidelityEvaluator._strip_hidden_reasoning(text).strip()
        dataset_name = str((metadata or {}).get("mas_dataset_name") or dataset_profile.name or "").strip().lower()
        broken: List[str] = []
        fatal: List[str] = []
        verified_blocks = 0
        normalized_structure = ""
        repair_locus = "stable"

        if dataset_name == "knowledge_crosswords" or dataset_profile.task_type == "structured_list":
            parsed = self.evaluator._extract_json_list(cleaned)
            if parsed is None:
                fatal.append("invalid_json_list")
            else:
                normalized_structure = parsed
                verified_blocks += 1
                values = [str(item).strip() for item in json.loads(parsed)]
                expected_len = 0
                if isinstance((metadata or {}).get("blanks"), list):
                    expected_len = len((metadata or {}).get("blanks") or [])
                elif isinstance((metadata or {}).get("answer_all"), list):
                    expected_len = len((metadata or {}).get("answer_all") or [])
                if expected_len and len(values) != expected_len:
                    broken.append("length_mismatch")
                elif expected_len:
                    verified_blocks += 1
                if any(not item for item in values):
                    broken.append("empty_slot")
                else:
                    verified_blocks += 1
            if fatal or broken:
                repair_locus = (fatal or broken)[0]
            return GraphConstraintEval(
                normalized_structure=normalized_structure,
                broken_blocks=tuple(sorted(set(fatal + broken))),
                fatal_blocks=tuple(sorted(set(fatal))),
                repair_locus=repair_locus,
                verified_blocks=verified_blocks,
            )

        task = str((metadata or {}).get("task") or "").strip()
        pred_json = self.evaluator._safe_json(cleaned) if cleaned else None
        if task:
            if not isinstance(pred_json, dict):
                fatal.append("invalid_graph_json")
            if task in {"connectivity", "cycle"}:
                answer = None
                if isinstance(pred_json, dict):
                    raw = pred_json.get("answer")
                    answer = self.evaluator._normalize_yes_no_output(str(raw)) if raw is not None else None
                if answer is None:
                    answer = self.evaluator._normalize_yes_no_output(cleaned)
                if answer not in {"yes", "no"}:
                    broken.append("invalid_answer")
                else:
                    normalized_structure = json.dumps({"answer": answer}, ensure_ascii=False, sort_keys=True)
                    verified_blocks += 1
            elif task in {"flow", "matching"}:
                value = None
                if isinstance(pred_json, dict):
                    for key in ("max_flow", "count", "answer"):
                        raw = pred_json.get(key)
                        if isinstance(raw, (int, float)):
                            value = raw
                            break
                if value is None:
                    value = self.evaluator._extract_prediction_target(cleaned)
                if value is None:
                    broken.append("missing_numeric_output")
                else:
                    normalized_structure = str(value)
                    verified_blocks += 1
            elif task == "topology":
                order: List[int] = []
                if isinstance(pred_json, dict) and isinstance(pred_json.get("order"), list):
                    for item in pred_json["order"]:
                        try:
                            order.append(int(item))
                        except Exception:
                            continue
                if not order:
                    order = self.evaluator._extract_sequence_numbers(cleaned)
                if not order:
                    broken.append("missing_order")
                else:
                    constraints = [
                        (int(a), int(b))
                        for a, b in re.findall(r"node (\d+) should be visited before node (\d+)", question_text, flags=re.IGNORECASE)
                    ]
                    position = {node: idx for idx, node in enumerate(order)}
                    if len(position) != len(order):
                        broken.append("duplicate_node")
                    if constraints and not all(a in position and b in position and position[a] < position[b] for a, b in constraints):
                        broken.append("constraint_violation")
                    normalized_structure = json.dumps({"order": order}, ensure_ascii=False, sort_keys=True)
                    verified_blocks += 1 + int("constraint_violation" not in broken)
            elif task in {"hamilton", "shortest_path"}:
                path: List[int] = []
                if isinstance(pred_json, dict) and isinstance(pred_json.get("path"), list):
                    for item in pred_json["path"]:
                        try:
                            path.append(int(item))
                        except Exception:
                            continue
                if not path:
                    path = self.evaluator._extract_sequence_numbers(cleaned)
                if not path:
                    broken.append("missing_path")
                else:
                    edges = self.evaluator._parse_nlgraph_edges(question_text)
                    invalid_edge = any((src, dst) not in edges for src, dst in zip(path, path[1:]))
                    if invalid_edge:
                        broken.append("invalid_edge")
                    if task == "hamilton" and len(path) != len(set(path)):
                        broken.append("duplicate_node")
                    computed_weight = self.evaluator._path_weight(path, edges) if len(path) > 1 else None
                    if task == "shortest_path" and isinstance(pred_json, dict) and "total_weight" in pred_json and computed_weight is not None:
                        try:
                            stated_weight = int(pred_json["total_weight"])
                        except Exception:
                            stated_weight = None
                        if stated_weight is not None and stated_weight != computed_weight:
                            broken.append("weight_mismatch")
                    normalized_structure = json.dumps({"path": path}, ensure_ascii=False, sort_keys=True)
                    verified_blocks += 1 + int("invalid_edge" not in broken)
            elif task == "GNN":
                if isinstance(pred_json, dict) and isinstance(pred_json.get("node_embeddings"), dict):
                    normalized_structure = json.dumps(pred_json["node_embeddings"], ensure_ascii=False, sort_keys=True)
                    verified_blocks += 1
                else:
                    broken.append("missing_node_embeddings")
            else:
                if not cleaned:
                    fatal.append("empty_graph_answer")
                else:
                    normalized_structure = cleaned
                    verified_blocks += 1
        else:
            if not cleaned:
                fatal.append("empty_graph_answer")
            else:
                normalized_structure = cleaned
                verified_blocks += 1

        if fatal or broken:
            repair_locus = (fatal or broken)[0]
        return GraphConstraintEval(
            normalized_structure=normalized_structure,
            broken_blocks=tuple(sorted(set(fatal + broken))),
            fatal_blocks=tuple(sorted(set(fatal))),
            repair_locus=repair_locus,
            verified_blocks=verified_blocks,
        )

    def _collapse_graph_classes(
        self,
        *,
        question_text: str,
        candidates: Sequence[Dict[str, Any]],
        anchor_digest: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> List[Dict[str, Any]]:
        grouped: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        for entry in candidates:
            self._ensure_v4_4_entry_fields(entry)
            evaluation = self._evaluate_graph_candidate(
                question_text=question_text,
                entry=entry,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            key = evaluation.class_key
            payload = grouped.get(key)
            rank_key = (
                evaluation.rank_key,
                *self._stable_entry_tiebreak(entry),
            )
            if payload is None:
                payload = {
                    "key": key,
                    "representative": entry,
                    "evaluation": evaluation,
                    "size": 0,
                    "contains_anchor": False,
                    "rank_key": rank_key,
                }
                grouped[key] = payload
            payload["size"] += 1
            payload["contains_anchor"] = bool(payload["contains_anchor"] or str(entry.get("digest", "")) == anchor_digest)
            if rank_key > payload["rank_key"]:
                payload["representative"] = entry
                payload["evaluation"] = evaluation
                payload["rank_key"] = rank_key
        collapsed = list(grouped.values())
        collapsed.sort(
            key=lambda item: (
                item["evaluation"].rank_key,
                *self._stable_entry_tiebreak(item["representative"]),
            ),
            reverse=True,
        )
        for item in collapsed:
            rep = item["representative"]
            rep["v4_4_class_key"] = tuple(item["key"])
            rep["v4_4_class_size"] = int(item["size"])
            rep["v4_4_route_family"] = "graph_constrained"
        return collapsed

    def _select_graph_repair_agents(self) -> List[str]:
        selected: List[str] = []
        for agent_id in self.config.graph_repair_agent_ids:
            if agent_id in self._by_id and agent_id not in selected:
                selected.append(agent_id)
        if selected:
            return selected[: self.config.graph_full_branch_cap]
        for fallback in ("coder", "reasoner", "planner", "verifier"):
            if fallback in self._by_id and fallback not in selected:
                selected.append(fallback)
        return selected[: self.config.graph_full_branch_cap]

    def _build_graph_repair_prompt(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        current_entry: Dict[str, Any],
        evaluation: GraphConstraintEval,
    ) -> str:
        broken = ", ".join(evaluation.broken_blocks) if evaluation.broken_blocks else "none"
        return (
            "You are repairing a graph-structured answer after local verification found a repair locus.\n\n"
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Current answer:\n{str(current_entry.get('text', '')).strip()}\n\n"
            f"Broken blocks: {broken}\n"
            f"Primary repair locus: {evaluation.repair_locus}\n\n"
            "Requirements:\n"
            "- Return only the corrected final answer in the required format.\n"
            "- Fix the primary repair locus first.\n"
            "- Preserve already-satisfied local structure.\n"
        )

    def _prepare_graph_verified_entry(
        self,
        text: str,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        parent: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], GraphConstraintEval]:
        entry = self._init_candidate_entry(text)
        self._ensure_v4_4_entry_fields(entry)
        if parent is not None:
            entry["source_roles"] = set(parent.get("source_roles", set()))
            entry["source_node_ids"] = set(parent.get("source_node_ids", set()))
            entry["turn_indices"] = set(parent.get("turn_indices", set()))
            entry["candidate_model_score"] = float(parent.get("candidate_model_score", 0.5))
            entry["candidate_model_uncertainty"] = float(parent.get("candidate_model_uncertainty", 0.2))
            entry["support_score"] = float(parent.get("support_score", 0.5))
            entry["sink_support"] = int(parent.get("sink_support", 0))
            entry["occurrence_count"] = int(parent.get("occurrence_count", 0))
            entry["feedback_pass"] = float(parent.get("feedback_pass", 0.0))
            entry["feedback_challenge"] = float(parent.get("feedback_challenge", 0.0))
            entry["feedback_uncertain"] = float(parent.get("feedback_uncertain", 0.0))
        evaluation = self._evaluate_graph_candidate(
            question_text=question_text,
            entry=entry,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        entry["v4_4_route_family"] = "graph_constrained"
        return entry, evaluation

    def _generate_graph_repair_branches(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        current_entry: Dict[str, Any],
        evaluation: GraphConstraintEval,
    ) -> List[Tuple[Dict[str, Any], GraphConstraintEval]]:
        seen: set[str] = {str(current_entry.get("text", "")).strip()}
        branches: List[Tuple[Dict[str, Any], GraphConstraintEval]] = []
        for agent_id in self._select_graph_repair_agents():
            agent = self._by_id[agent_id]
            system_prompt = build_system_prompt(agent, self._graph_repair_slots(), extra_role_hint="graph_repair")
            user_prompt = self._build_graph_repair_prompt(
                question_text=question_text,
                metadata=metadata,
                current_entry=current_entry,
                evaluation=evaluation,
            )
            raw_output = self.evaluator._cached_chat(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
            )
            repaired = self._sanitize_candidate(question_text, raw_output, metadata=metadata)
            if not repaired or repaired in seen:
                continue
            seen.add(repaired)
            branch_entry, branch_eval = self._prepare_graph_verified_entry(
                repaired,
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                parent=current_entry,
            )
            branches.append((branch_entry, branch_eval))
        branches.sort(
            key=lambda item: (
                item[1].rank_key,
                *self._stable_entry_tiebreak(item[0]),
            ),
            reverse=True,
        )
        return branches

    def _inspect_code_promotion(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        current_entry: Dict[str, Any],
        current_feedback: CodeRepairEval,
        candidate_entry: Dict[str, Any],
        candidate_feedback: CodeRepairEval,
    ) -> Tuple[bool, str]:
        if candidate_feedback.fully_passed:
            return True, "fully_passed"
        inspector_id = self.config.inspector_agent_id
        if inspector_id not in self._by_id:
            inspector_id = "verifier" if "verifier" in self._by_id else next(iter(self._by_id))
        agent = self._by_id[inspector_id]
        system_prompt = build_system_prompt(agent, self._inspector_slots(), extra_role_hint="promotion_inspector")
        current_summary = build_failure_summary(
            current_feedback,
            metadata=metadata,
            max_examples=int(self.config.repair_max_failed_examples),
        )
        candidate_summary = build_failure_summary(
            candidate_feedback,
            metadata=metadata,
            max_examples=int(self.config.repair_max_failed_examples),
        )
        user_prompt = (
            "You are approving whether a partial code-repair branch may replace the current checkpoint.\n\n"
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Current checkpoint code:\n{str(current_entry.get('text', '')).strip()}\n\n"
            f"Current verifier summary:\n{current_summary}\n\n"
            f"Candidate repair code:\n{str(candidate_entry.get('text', '')).strip()}\n\n"
            f"Candidate verifier summary:\n{candidate_summary}\n\n"
            "Return exactly three lines:\n"
            "REPAIRED: yes|no\n"
            "DRIFT: yes|no\n"
            "RATIONALE: <short justification>\n"
        )
        raw_output = self.evaluator._cached_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            runtime=self.evaluator._resolve_runtime("tier2", dataset_profile),
        )
        parsed = self._parse_tagged_output(raw_output)
        repaired = str(parsed.get("repaired", "")).strip().lower() == "yes"
        drift = str(parsed.get("drift", "")).strip().lower() == "yes"
        rationale = str(parsed.get("rationale", "")).strip()
        return repaired and not drift, rationale

    @staticmethod
    def _is_recoverable_code_feedback(feedback: CodeRepairEval) -> bool:
        failure_kind = str(feedback.failure_kind or "").strip()
        has_failure_signal = bool(feedback.failing_examples) or failure_kind not in {"", "no_dataset_tests"}
        return bool(not feedback.fully_passed and has_failure_signal)

    def _best_code_recovery_target(
        self,
        *,
        champion_entry: Dict[str, Any],
        champion_feedback: CodeRepairEval,
        anchor_pair: Optional[Tuple[Dict[str, Any], CodeRepairEval]],
    ) -> Optional[Tuple[Dict[str, Any], CodeRepairEval, str]]:
        if self._is_recoverable_code_feedback(champion_feedback) and self._entry_has_recovery_seed_provenance(champion_entry):
            return champion_entry, champion_feedback, "champion"
        if (
            anchor_pair is not None
            and self._is_recoverable_code_feedback(anchor_pair[1])
            and self._entry_has_recovery_seed_provenance(anchor_pair[0])
        ):
            return anchor_pair[0], anchor_pair[1], "anchor"
        return None

    def _code_mode(
        self,
        *,
        anchor_pair: Optional[Tuple[Dict[str, Any], CodeRepairEval]],
        challenger_classes: Sequence[Dict[str, Any]],
        budget_bucket: str,
    ) -> str:
        if anchor_pair is None:
            base = "lean"
            return self._apply_budget_bucket(base, budget_bucket)
        _, anchor_feedback = anchor_pair
        anchor_recoverable = self._is_recoverable_code_feedback(anchor_feedback)
        surviving = [
            item
            for item in challenger_classes
            if item["feedback"].syntax_ok and item["feedback"].entry_point_ok
        ]
        if anchor_feedback.fully_passed and not any(item["feedback"].dominates(anchor_feedback) for item in surviving):
            return self._apply_budget_bucket("bypass", budget_bucket)
        if len(surviving) >= 2:
            return self._apply_budget_bucket("full", budget_bucket)
        if len(surviving) == 1 or surviving:
            return self._apply_budget_bucket("lean", budget_bucket)
        if anchor_recoverable:
            return self._apply_budget_bucket("lean", budget_bucket)
        return self._apply_budget_bucket("bypass", budget_bucket)

    def _graph_mode(
        self,
        *,
        anchor_eval: Optional[GraphConstraintEval],
        challenger_classes: Sequence[Dict[str, Any]],
        budget_bucket: str,
    ) -> str:
        surviving = [item for item in challenger_classes if not item["evaluation"].fatal_blocks]
        if anchor_eval is not None and not anchor_eval.broken_blocks and not any(
            item["evaluation"].dominates(anchor_eval) for item in surviving
        ):
            return self._apply_budget_bucket("bypass", budget_bucket)
        if len(surviving) >= 2 or (anchor_eval is not None and len(anchor_eval.broken_blocks) >= 2):
            return self._apply_budget_bucket("full", budget_bucket)
        if surviving:
            return self._apply_budget_bucket("lean", budget_bucket)
        return self._apply_budget_bucket("bypass", budget_bucket)

    def _reasoning_mode(
        self,
        *,
        anchor_eval: Optional[ReasoningEval],
        challenger_classes: Sequence[Dict[str, Any]],
        budget_bucket: str,
    ) -> str:
        surviving = [item for item in challenger_classes if not item["evaluation"].fatal_contradictions]
        if anchor_eval is not None and not anchor_eval.fatal_contradictions and not any(
            item["evaluation"].dominates(anchor_eval) for item in surviving
        ):
            return self._apply_budget_bucket("bypass", budget_bucket)
        if len(surviving) >= 2:
            return self._apply_budget_bucket("full", budget_bucket)
        if surviving:
            return self._apply_budget_bucket("lean", budget_bucket)
        return self._apply_budget_bucket("bypass", budget_bucket)

    def _select_code_repair_against_anchor_v44(
        self,
        *,
        graph: UnionGraph,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        budget_bucket: str,
        turn_traces: Sequence[TurnTrace] = (),
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        full_pool, anchor_pair = self._verify_code_pool(anchor=anchor, candidates=candidates, metadata=metadata)
        if not full_pool:
            return anchor, "v4_4_code_empty_bank", {
                "v4_4_protocol_family": "code_repair",
                "v4_4_execution_mode": "bypass",
                "v4_4_budget_bucket": budget_bucket,
                "v4_4_collapsed_class_count": 0,
            }

        anchor_recoverable_but_blocked_no_provenance_count = 0
        anchor_recovery_seed_legalized = False
        anchor_recovery_seed_binding_kind = ""
        anchor_recovery_seed_source_graph = ""
        if anchor_pair is not None:
            anchor_entry, anchor_feedback = anchor_pair
            anchor_recoverable = self._is_recoverable_code_feedback(anchor_feedback)
            anchor_missing_seed = anchor_recoverable and not self._entry_has_recovery_seed_provenance(anchor_entry)
            if anchor_missing_seed:
                anchor_recovery_seed_legalized = self._legalize_stage1_anchor_recovery_seed(
                    graph=graph,
                    anchor_entry=anchor_entry,
                    metadata=metadata,
                )
                if anchor_recovery_seed_legalized:
                    anchor_recovery_seed_binding_kind = str(anchor_entry.get("stage1_anchor_binding_kind", ""))
                    anchor_recovery_seed_source_graph = str(anchor_entry.get("stage1_anchor_binding_source_graph_id", ""))
                else:
                    anchor_recoverable_but_blocked_no_provenance_count = 1

        anchor_digest = str((anchor_pair[0] if anchor_pair is not None else {}).get("digest", ""))
        classes = self._collapse_code_classes(full_pool, anchor_digest=anchor_digest)
        challenger_classes = [item for item in classes if not item["contains_anchor"]]
        execution_mode = self._code_mode(anchor_pair=anchor_pair, challenger_classes=challenger_classes, budget_bucket=budget_bucket)

        if anchor_pair is None:
            selected_entry, selected_feedback = classes[0]["representative"], classes[0]["feedback"]
            selected_reason = "v4_4_code_no_anchor"
        elif execution_mode == "bypass":
            selected_entry, selected_feedback = anchor_pair
            selected_reason = "v4_4_code_bypass_stable_anchor"
        else:
            seed_classes = challenger_classes if challenger_classes else classes
            seed_cap = 1 if execution_mode == "lean" else max(1, int(self.config.repair_seed_top_k))
            seed_pool = [(item["representative"], item["feedback"]) for item in seed_classes[:seed_cap]]
            if not seed_pool:
                seed_pool = [anchor_pair]

            selected_entry, selected_feedback = anchor_pair
            seed_best_entry, seed_best_feedback = seed_pool[0]
            if seed_best_feedback.dominates(selected_feedback):
                selected_entry, selected_feedback = seed_best_entry, seed_best_feedback
                selected_reason = "v4_4_code_override_verified_class"
            else:
                selected_reason = "v4_4_code_preserve_anchor_after_class_collapse"

            repair_round_limit = 1 if execution_mode == "lean" else max(1, int(self.config.repair_rounds))
            repair_rounds_run = 0
            repair_branch_count = 0
            repair_improvement_count = 0
            restart_count = 0
            inspector_approval_count = 0
            reinserted_recovery_count = 0
            recovery_target_source = "none"
            recovery_target_digest = ""
            recovery_subgraph_node_count = 0
            recovery_subgraph_edge_count = 0
            seed_index = 0

            while repair_rounds_run < repair_round_limit:
                if selected_feedback.fully_passed:
                    break
                recovery_target = self._best_code_recovery_target(
                    champion_entry=selected_entry,
                    champion_feedback=selected_feedback,
                    anchor_pair=anchor_pair,
                )
                if recovery_target is None:
                    if execution_mode == "full" and seed_index + 1 < len(seed_pool):
                        seed_index += 1
                        selected_entry, selected_feedback = seed_pool[seed_index]
                        selected_reason = "v4_4_code_restart_to_verified_class"
                        restart_count += 1
                        continue
                    break
                recovery_entry, recovery_feedback, recovery_target_source = recovery_target
                recovery_target_digest = str(recovery_entry.get("digest", ""))
                recovery_context = self._build_code_recovery_context(
                    graph=graph,
                    turn_traces=turn_traces,
                    target_entry=recovery_entry,
                    target_feedback=recovery_feedback,
                )
                recovery_subgraph_node_count = len(recovery_context["recovery_subgraph_node_ids"])
                recovery_subgraph_edge_count = len(recovery_context["recovery_subgraph_edge_ids"])
                repair_rounds_run += 1
                branches = self._generate_repair_branches(
                    question_text=question_text,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                    current_entry=recovery_entry,
                    feedback=recovery_feedback,
                    repair_round=repair_rounds_run,
                    recovery_subgraph_node_ids=recovery_context["recovery_subgraph_node_ids"],
                    recovery_subgraph_edge_ids=recovery_context["recovery_subgraph_edge_ids"],
                    trigger_verifier_snapshot=recovery_context["trigger_verifier_snapshot"],
                    repair_operator_type="code_repair_patch",
                )
                repair_branch_count += len(branches)
                approved: List[Tuple[Dict[str, Any], CodeRepairEval]] = []
                for branch_entry, branch_feedback in branches:
                    if not branch_feedback.dominates(recovery_feedback):
                        continue
                    self._ensure_v4_4_entry_fields(branch_entry)
                    if branch_feedback.fully_passed:
                        branch_entry["promotion_inspector_decision"] = "approved_fully_passed"
                        approved.append((branch_entry, branch_feedback))
                        continue
                    inspect_ok, inspect_rationale = self._inspect_code_promotion(
                        question_text=question_text,
                        metadata=metadata,
                        dataset_profile=dataset_profile,
                        current_entry=recovery_entry,
                        current_feedback=recovery_feedback,
                        candidate_entry=branch_entry,
                        candidate_feedback=branch_feedback,
                    )
                    branch_entry["promotion_inspector_decision"] = "approve" if inspect_ok else "reject"
                    branch_entry["promotion_inspector_rationale"] = inspect_rationale
                    if inspect_ok:
                        inspector_approval_count += 1
                        approved.append((branch_entry, branch_feedback))
                if approved:
                    reinserted_recovery_count += len(approved)
                    for branch_entry, _ in approved:
                        branch_entry["recovery_reinserted"] = True
                        branch_entry["candidate_bank_source"] = "recovery_output"
                        self._assert_recovery_entry_invariants(branch_entry)
                    full_pool.extend(list(approved))
                    classes = self._collapse_code_classes(full_pool, anchor_digest=anchor_digest)
                    selected_entry = classes[0]["representative"]
                    selected_feedback = classes[0]["feedback"]
                    selected_reason = "v4_4_code_reinsert_recollapse"
                    if selected_feedback.dominates(recovery_feedback):
                        repair_improvement_count += 1
                    continue
                if execution_mode == "full" and seed_index + 1 < len(seed_pool):
                    seed_index += 1
                    selected_entry, selected_feedback = seed_pool[seed_index]
                    selected_reason = "v4_4_code_restart_to_verified_class"
                    restart_count += 1
                    continue
                break

            if not selected_feedback.dominates(anchor_pair[1]):
                selected_entry, selected_feedback = anchor_pair
                selected_reason = "v4_4_code_anchor_guard_preserve"

            extra = {
                "v4_4_protocol_family": "code_repair",
                "v4_4_execution_mode": execution_mode,
                "v4_4_budget_bucket": budget_bucket,
                "v4_4_stage1_anchor_present": True,
                "v4_4_stage1_anchor_used": bool(str(selected_entry.get("digest", "")) == str(anchor_pair[0].get("digest", ""))),
                "v4_4_candidate_count": int(len(candidates)),
                "v4_4_collapsed_class_count": int(len(classes)),
                "v4_4_selected_candidate_digest": str(selected_entry.get("digest", "")),
                "v4_4_selected_candidate_source": self._candidate_source_label(selected_entry),
                "v4_4_selected_quality_score": float(self._quality_score(selected_entry)),
                "v4_4_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
                "v4_4_selected_is_repair_branch": bool(selected_entry.get("repair_branch", False)),
                "v4_4_selected_visible_tests_passed": int(selected_feedback.passed),
                "v4_4_selected_visible_tests_total": int(selected_feedback.total),
                "v4_4_selected_failure_kind": str(selected_feedback.failure_kind),
                "v4_4_repair_rounds_run": int(repair_rounds_run),
                "v4_4_repair_branch_count": int(repair_branch_count),
                "v4_4_repair_improvement_count": int(repair_improvement_count),
                "v4_4_restart_count": int(restart_count),
                "v4_4_inspector_approval_count": int(inspector_approval_count),
                "v4_4_reinserted_recovery_count": int(reinserted_recovery_count),
                "v4_4_recovery_target_source": recovery_target_source,
                "v4_4_recovery_target_digest": recovery_target_digest,
                "v4_4_recovery_subgraph_node_count": int(recovery_subgraph_node_count),
                "v4_4_recovery_subgraph_edge_count": int(recovery_subgraph_edge_count),
                "v4_4_stage1_anchor_digest": str(anchor_pair[0].get("digest", "")),
                "v4_4_stage1_anchor_visible_tests_passed": int(anchor_pair[1].passed),
                "v4_4_stage1_anchor_visible_tests_total": int(anchor_pair[1].total),
                "v4_4_stage1_anchor_failure_kind": str(anchor_pair[1].failure_kind),
                "v4_4_anchor_recovery_seed_legalized": bool(anchor_recovery_seed_legalized),
                "v4_4_anchor_recovery_seed_binding_kind": anchor_recovery_seed_binding_kind,
                "v4_4_anchor_recovery_seed_source_graph": anchor_recovery_seed_source_graph,
                "v4_4_anchor_recoverable_but_blocked_no_provenance_count": int(
                    anchor_recoverable_but_blocked_no_provenance_count
                ),
                "v4_4_candidate_provenance_coverage": (
                    float(
                        sum(
                            1
                            for entry, _ in full_pool
                            if self._entry_has_recovery_seed_provenance(entry)
                            or bool(entry.get("anchor_bound_node_ids"))
                        )
                    )
                    / float(len(full_pool))
                    if full_pool
                    else 0.0
                ),
                "v4_4_top_classes": [
                    {
                        "class_key": self._public_class_key(tuple(item["key"])),
                        "size": int(item["size"]),
                        "contains_anchor": bool(item["contains_anchor"]),
                        "representative_digest": str(item["representative"].get("digest", "")),
                        "repair_locus": str(item["repair_locus"]),
                        "passed": int(item["feedback"].passed),
                        "total": int(item["feedback"].total),
                        "failure_kind": str(item["feedback"].failure_kind),
                    }
                    for item in classes[: self.config.max_logged_candidates]
                ],
            }
            return selected_entry, selected_reason, extra

        extra = {
            "v4_4_protocol_family": "code_repair",
            "v4_4_execution_mode": execution_mode,
            "v4_4_budget_bucket": budget_bucket,
            "v4_4_stage1_anchor_present": bool(anchor_pair is not None),
            "v4_4_stage1_anchor_used": bool(anchor_pair is not None and selected_entry.get("digest") == anchor_pair[0].get("digest")),
            "v4_4_candidate_count": int(len(candidates)),
            "v4_4_collapsed_class_count": int(len(classes)),
            "v4_4_selected_candidate_digest": str(selected_entry.get("digest", "")),
            "v4_4_selected_candidate_source": self._candidate_source_label(selected_entry),
            "v4_4_selected_quality_score": float(self._quality_score(selected_entry)),
            "v4_4_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
            "v4_4_selected_is_repair_branch": bool(selected_entry.get("repair_branch", False)),
            "v4_4_selected_visible_tests_passed": int(selected_feedback.passed),
            "v4_4_selected_visible_tests_total": int(selected_feedback.total),
            "v4_4_selected_failure_kind": str(selected_feedback.failure_kind),
            "v4_4_repair_rounds_run": 0,
            "v4_4_repair_branch_count": 0,
            "v4_4_repair_improvement_count": 0,
            "v4_4_restart_count": 0,
            "v4_4_inspector_approval_count": 0,
            "v4_4_reinserted_recovery_count": 0,
            "v4_4_recovery_target_source": "none",
            "v4_4_recovery_target_digest": "",
            "v4_4_recovery_subgraph_node_count": 0,
            "v4_4_recovery_subgraph_edge_count": 0,
            "v4_4_top_classes": [
                {
                    "class_key": self._public_class_key(tuple(item["key"])),
                    "size": int(item["size"]),
                    "contains_anchor": bool(item["contains_anchor"]),
                    "representative_digest": str(item["representative"].get("digest", "")),
                    "repair_locus": str(item["repair_locus"]),
                    "passed": int(item["feedback"].passed),
                    "total": int(item["feedback"].total),
                    "failure_kind": str(item["feedback"].failure_kind),
                }
                for item in classes[: self.config.max_logged_candidates]
            ],
        }
        if anchor_pair is not None:
            extra.update(
                {
                    "v4_4_stage1_anchor_digest": str(anchor_pair[0].get("digest", "")),
                    "v4_4_stage1_anchor_visible_tests_passed": int(anchor_pair[1].passed),
                    "v4_4_stage1_anchor_visible_tests_total": int(anchor_pair[1].total),
                    "v4_4_stage1_anchor_failure_kind": str(anchor_pair[1].failure_kind),
                }
            )
        return selected_entry, selected_reason, extra

    def _select_graph_against_anchor_v44(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        budget_bucket: str,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        if anchor is not None:
            ordered.append(anchor)
        ordered.extend([item for item in candidates if str(item.get("digest", "")) != str((anchor or {}).get("digest", ""))])
        if not ordered:
            return anchor, "v4_4_graph_empty_bank", {
                "v4_4_protocol_family": "graph_constrained",
                "v4_4_execution_mode": "bypass",
                "v4_4_budget_bucket": budget_bucket,
                "v4_4_collapsed_class_count": 0,
            }
        anchor_digest = str((anchor or {}).get("digest", ""))
        classes = self._collapse_graph_classes(
            question_text=question_text,
            candidates=ordered,
            anchor_digest=anchor_digest,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        anchor_eval = None
        if anchor is not None:
            anchor_eval = self._evaluate_graph_candidate(
                question_text=question_text,
                entry=anchor,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
        challenger_classes = [item for item in classes if not item["contains_anchor"]]
        execution_mode = self._graph_mode(anchor_eval=anchor_eval, challenger_classes=challenger_classes, budget_bucket=budget_bucket)

        if anchor is not None and execution_mode == "bypass":
            selected_entry = anchor
            selected_eval = anchor_eval
            strategy = "v4_4_graph_bypass_stable_anchor"
            graph_branch_count = 0
            graph_restart_count = 0
        else:
            seed_classes = challenger_classes if challenger_classes else classes
            seed_cap = 1 if execution_mode == "lean" else max(1, int(self.config.graph_seed_top_k))
            seed_pool = [(item["representative"], item["evaluation"]) for item in seed_classes[:seed_cap]]
            if anchor is not None and anchor_eval is not None:
                selected_entry, selected_eval = anchor, anchor_eval
            else:
                selected_entry, selected_eval = seed_pool[0]
            best_seed_entry, best_seed_eval = seed_pool[0]
            strategy = "v4_4_graph_preserve_anchor_after_collapse"
            if anchor is None or best_seed_eval.dominates(selected_eval):
                selected_entry, selected_eval = best_seed_entry, best_seed_eval
                strategy = "v4_4_graph_override_collapsed_representative"

            graph_branch_count = 0
            graph_restart_count = 0
            if execution_mode == "full" and selected_eval is not None and selected_eval.broken_blocks:
                round_limit = max(1, int(self.config.graph_repair_rounds))
                seed_index = 0
                for _ in range(round_limit):
                    branches = self._generate_graph_repair_branches(
                        question_text=question_text,
                        metadata=metadata,
                        dataset_profile=dataset_profile,
                        current_entry=selected_entry,
                        evaluation=selected_eval,
                    )
                    graph_branch_count += len(branches)
                    improved = [
                        (entry, evaluation)
                        for entry, evaluation in branches
                        if evaluation.dominates(selected_eval)
                    ]
                    if improved:
                        improved.sort(
                            key=lambda item: (
                                item[1].rank_key,
                                *self._stable_entry_tiebreak(item[0]),
                            ),
                            reverse=True,
                        )
                        selected_entry, selected_eval = improved[0]
                        strategy = "v4_4_graph_override_local_repair"
                        continue
                    if seed_index + 1 < len(seed_pool):
                        seed_index += 1
                        selected_entry, selected_eval = seed_pool[seed_index]
                        graph_restart_count += 1
                        strategy = "v4_4_graph_restart_to_next_class"
                        continue
                    break

            if anchor is not None and anchor_eval is not None and not selected_eval.dominates(anchor_eval):
                selected_entry, selected_eval = anchor, anchor_eval
                strategy = "v4_4_graph_anchor_guard_preserve"

        extra = {
            "v4_4_protocol_family": "graph_constrained",
            "v4_4_execution_mode": execution_mode,
            "v4_4_budget_bucket": budget_bucket,
            "v4_4_stage1_anchor_present": bool(anchor is not None),
            "v4_4_stage1_anchor_used": bool(anchor is not None and str(selected_entry.get("digest", "")) == str(anchor.get("digest", ""))),
            "v4_4_candidate_count": int(len(candidates)),
            "v4_4_collapsed_class_count": int(len(classes)),
            "v4_4_selected_candidate_digest": str(selected_entry.get("digest", "")),
            "v4_4_selected_candidate_source": self._candidate_source_label(selected_entry),
            "v4_4_selected_quality_score": float(self._quality_score(selected_entry)),
            "v4_4_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
            "v4_4_selected_broken_block_count": int(len(selected_eval.broken_blocks)),
            "v4_4_selected_repair_locus": str(selected_eval.repair_locus),
            "v4_4_selected_verified_blocks": int(selected_eval.verified_blocks),
            "v4_4_graph_branch_count": int(graph_branch_count),
            "v4_4_graph_restart_count": int(graph_restart_count),
            "v4_4_top_classes": [
                {
                    "class_key": self._public_class_key(tuple(item["key"])),
                    "size": int(item["size"]),
                    "contains_anchor": bool(item["contains_anchor"]),
                    "representative_digest": str(item["representative"].get("digest", "")),
                    "broken_blocks": list(item["evaluation"].broken_blocks),
                    "repair_locus": str(item["evaluation"].repair_locus),
                    "verified_blocks": int(item["evaluation"].verified_blocks),
                }
                for item in classes[: self.config.max_logged_candidates]
            ],
        }
        if anchor is not None and anchor_eval is not None:
            extra.update(
                {
                    "v4_4_stage1_anchor_digest": str(anchor.get("digest", "")),
                    "v4_4_stage1_anchor_broken_block_count": int(len(anchor_eval.broken_blocks)),
                    "v4_4_stage1_anchor_repair_locus": str(anchor_eval.repair_locus),
                    "v4_4_stage1_anchor_verified_blocks": int(anchor_eval.verified_blocks),
                }
            )
        return selected_entry, strategy, extra

    def _select_reasoning_against_anchor_v44(
        self,
        *,
        question_text: str,
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
        candidates: Sequence[Dict[str, Any]],
        anchor: Optional[Dict[str, Any]],
        budget_bucket: str,
    ) -> Tuple[Optional[Dict[str, Any]], str, Dict[str, Any]]:
        ordered: List[Dict[str, Any]] = []
        if anchor is not None:
            ordered.append(anchor)
        ordered.extend([item for item in candidates if str(item.get("digest", "")) != str((anchor or {}).get("digest", ""))])
        if not ordered:
            return anchor, "v4_4_reasoning_empty_bank", {
                "v4_4_protocol_family": "adversarial",
                "v4_4_execution_mode": "bypass",
                "v4_4_budget_bucket": budget_bucket,
                "v4_4_collapsed_class_count": 0,
            }

        anchor_digest = str((anchor or {}).get("digest", ""))
        classes = self._collapse_reasoning_classes(
            question_text=question_text,
            candidates=ordered,
            anchor_digest=anchor_digest,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        anchor_eval = None
        if anchor is not None:
            anchor_eval = self._evaluate_reasoning_candidate(
                question_text=question_text,
                entry=anchor,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
        challenger_classes = [item for item in classes if not item["contains_anchor"]]
        execution_mode = self._reasoning_mode(anchor_eval=anchor_eval, challenger_classes=challenger_classes, budget_bucket=budget_bucket)

        calibration_rounds = 0
        calibration_probability = 0.0
        calibration_votes = 0
        if anchor is not None and anchor_eval is not None and not anchor_eval.fatal_contradictions:
            selected_entry = anchor
            selected_eval = anchor_eval
            strategy = "v4_4_reasoning_stabilize_preserve_anchor"
        elif anchor is not None and execution_mode == "bypass":
            selected_entry = anchor
            selected_eval = anchor_eval
            strategy = "v4_4_reasoning_bypass_stable_anchor"
        else:
            cap = (
                int(self.config.adversarial_lean_hypothesis_cap)
                if execution_mode == "lean"
                else int(self.config.adversarial_full_hypothesis_cap)
            )
            surviving = [item for item in challenger_classes if not item["evaluation"].fatal_contradictions]
            active = surviving[: max(1, cap)]
            if not active:
                selected_entry = anchor if anchor is not None else classes[0]["representative"]
                selected_eval = anchor_eval if anchor_eval is not None else classes[0]["evaluation"]
                strategy = "v4_4_reasoning_preserve_no_surviving_class"
            else:
                best = active[0]
                selected_entry = best["representative"]
                selected_eval = best["evaluation"]
                strategy = "v4_4_reasoning_override_ach_lexicographic"
                if anchor is not None and anchor_eval is not None:
                    if selected_eval.dominates(anchor_eval):
                        pass
                    elif selected_eval.lexicographic_key == anchor_eval.lexicographic_key:
                        calibration = self._calibrate_challenger_against_anchor(
                            question_text=question_text,
                            metadata=metadata,
                            dataset_profile=dataset_profile,
                            anchor=anchor,
                            challenger=selected_entry,
                        )
                        calibration_rounds = int(calibration["rounds"])
                        calibration_probability = float(calibration["probability"])
                        calibration_votes = int(calibration["challenger_votes"])
                        if calibration["override"]:
                            strategy = "v4_4_reasoning_override_calibrated_tie"
                        else:
                            selected_entry = anchor
                            selected_eval = anchor_eval
                            strategy = "v4_4_reasoning_preserve_calibrated_tie"
                    else:
                        selected_entry = anchor
                        selected_eval = anchor_eval
                        strategy = "v4_4_reasoning_anchor_guard_preserve"

        extra = {
            "v4_4_protocol_family": "adversarial",
            "v4_4_execution_mode": execution_mode,
            "v4_4_budget_bucket": budget_bucket,
            "v4_4_stage1_anchor_present": bool(anchor is not None),
            "v4_4_stage1_anchor_used": bool(anchor is not None and str(selected_entry.get("digest", "")) == str(anchor.get("digest", ""))),
            "v4_4_candidate_count": int(len(candidates)),
            "v4_4_collapsed_class_count": int(len(classes)),
            "v4_4_selected_candidate_digest": str(selected_entry.get("digest", "")),
            "v4_4_selected_candidate_source": self._candidate_source_label(selected_entry),
            "v4_4_selected_quality_score": float(self._quality_score(selected_entry)),
            "v4_4_selected_model_uncertainty": float(selected_entry.get("candidate_model_uncertainty", 0.0)),
            "v4_4_selected_fatal_count": int(len(selected_eval.fatal_contradictions)),
            "v4_4_selected_major_count": int(len(selected_eval.major_contradictions)),
            "v4_4_selected_critical_support": int(selected_eval.critical_support),
            "v4_4_calibration_rounds": int(calibration_rounds),
            "v4_4_calibration_challenger_votes": int(calibration_votes),
            "v4_4_calibration_probability": float(calibration_probability),
            "v4_4_top_classes": [
                {
                    "class_key": self._public_class_key(tuple(item["key"])),
                    "size": int(item["size"]),
                    "contains_anchor": bool(item["contains_anchor"]),
                    "representative_digest": str(item["representative"].get("digest", "")),
                    "fatal_contradictions": list(item["evaluation"].fatal_contradictions),
                    "major_contradictions": list(item["evaluation"].major_contradictions),
                    "critical_support": int(item["evaluation"].critical_support),
                }
                for item in classes[: self.config.max_logged_candidates]
            ],
        }
        if anchor is not None and anchor_eval is not None:
            extra.update(
                {
                    "v4_4_stage1_anchor_digest": str(anchor.get("digest", "")),
                    "v4_4_stage1_anchor_fatal_count": int(len(anchor_eval.fatal_contradictions)),
                    "v4_4_stage1_anchor_major_count": int(len(anchor_eval.major_contradictions)),
                    "v4_4_stage1_anchor_critical_support": int(anchor_eval.critical_support),
                }
            )
        return selected_entry, strategy, extra

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
        route_family = str(extra.get("v4_4_protocol_family", self._v4_4_route_family))
        execution_mode = str(extra.get("v4_4_execution_mode", ""))
        self._last_v4_4_selection = {
            "stage2_version": "v4.4",
            "v4_4_task_type": task_type,
            "v4_4_route_family": route_family,
            "v4_4_execution_mode": execution_mode,
            "v4_4_selection_reason": strategy,
            "v4_4_candidate_count": int(len(candidates)),
            "v4_4_stage1_anchor_present": bool(anchor is not None),
            "v4_4_stage1_anchor_used": bool(anchor is not None and selected is not None and anchor.get("digest") == selected.get("digest")),
            "v4_4_selected_candidate_digest": str((selected or {}).get("digest", "")),
            "v4_4_selected_candidate_source": self._candidate_source_label(selected or {}),
            "v4_4_selected_quality_score": self._quality_score(selected or {}),
            "v4_4_selected_model_uncertainty": float((selected or {}).get("candidate_model_uncertainty", 0.0)),
            "v4_4_top_candidates": top,
        }
        self._last_v4_4_selection.update(extra)

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

        route_family = self._route_family(dataset_profile, metadata)
        budget_bucket = self._budget_bucket(metadata)
        self._v4_4_route_family = route_family

        if route_family == "code_repair":
            current_graph = getattr(self, "_v4_4_current_graph", None)
            if current_graph is None:
                raise ValueError("Stage2RuntimeV44 code repair finalizer requires the active union graph context.")
            selected, strategy, extra = self._select_code_repair_against_anchor_v44(
                graph=current_graph,
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                candidates=candidates,
                anchor=anchor,
                budget_bucket=budget_bucket,
                turn_traces=turn_traces,
            )
        elif route_family == "deductive_reasoning":
            selected, strategy, extra = self._select_best_deductive_candidate(
                question_text,
                anchor,
                candidates,
                dataset_profile,
                metadata,
                graph=getattr(self, "_v4_4_current_graph", None),
                turn_traces=turn_traces,
                budget_bucket=budget_bucket,
            )
        elif route_family == "graph_constrained":
            selected, strategy, extra = self._select_graph_against_anchor_v44(
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                candidates=candidates,
                anchor=anchor,
                budget_bucket=budget_bucket,
            )
        else:
            selected, strategy, extra = self._select_reasoning_against_anchor_v44(
                question_text=question_text,
                metadata=metadata,
                dataset_profile=dataset_profile,
                candidates=candidates,
                anchor=anchor,
                budget_bucket=budget_bucket,
            )

        selected_serialized = None
        if selected is not None:
            selected_serialized = self._serialize_candidate_entry(selected)
            for index, item in enumerate(candidates_serialized):
                if item.get("digest") == selected.get("digest"):
                    candidates_serialized[index] = dict(selected_serialized)
                    break
            else:
                candidates_serialized.append(dict(selected_serialized))
            self._assert_recovery_entry_invariants(selected)

        if selected is not None and route_family == "deductive_reasoning":
            selected_eval = self._deductive_eval_for_entry(selected)
            if selected_eval is None:
                selected_eval = self._evaluate_deductive_candidate(
                    question_text=question_text,
                    entry=selected,
                    dataset_profile=dataset_profile,
                    metadata=metadata,
                )
            final_answer = answer_only_from_artifact(selected_eval.artifact)
        else:
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

    def _graph_faithfulness_metrics(self, result: Stage2RunResult) -> Dict[str, Any]:
        edge_ratios: List[float] = []
        node_ratios: List[float] = []
        for turn_trace in result.turn_traces:
            total_edges = len(turn_trace.active_edges)
            active_edges = sum(1 for edge in turn_trace.active_edges if edge.active)
            edge_ratios.append(float(active_edges) / float(total_edges) if total_edges else 0.0)
            active_node_ids = set(turn_trace.metadata.get("active_node_ids", ()))
            skipped_node_ids = set(turn_trace.metadata.get("skipped_node_ids", ()))
            total_nodes = len(active_node_ids | skipped_node_ids)
            node_ratios.append(float(len(active_node_ids)) / float(total_nodes) if total_nodes else 0.0)
        candidates = list(self._last_candidate_bundle.get("candidates_serialized", ()))
        provenance_count = sum(1 for item in candidates if item.get("provenance"))
        selected_digest = str(self._last_v4_4_selection.get("v4_4_selected_candidate_digest", ""))
        selected_entry = next((item for item in candidates if str(item.get("digest", "")) == selected_digest), {})
        provenance_coverage = self._last_v4_4_selection.get("v4_4_candidate_provenance_coverage")
        if provenance_coverage is None:
            provenance_coverage = float(provenance_count) / float(len(candidates)) if candidates else 0.0
        recovery_subgraph_size = int(
            len(selected_entry.get("recovery_subgraph_node_ids", ()))
            or self._last_v4_4_selection.get("v4_4_recovery_subgraph_node_count", 0)
        )
        return {
            "graph_faithfulness_active_edge_ratio_by_turn": edge_ratios,
            "graph_faithfulness_active_node_ratio_by_turn": node_ratios,
            "graph_faithfulness_candidate_provenance_coverage": float(provenance_coverage),
            "graph_faithfulness_sink_path_provenance_length": int(len(selected_entry.get("provenance", ()))),
            "graph_faithfulness_recovery_subgraph_size": recovery_subgraph_size,
            "graph_faithfulness_final_answer_source_type": str(
                selected_entry.get("candidate_bank_source")
                or self._last_v4_4_selection.get("v4_4_selected_candidate_source", "")
            ),
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
        self._last_v4_4_selection = {}
        self._last_candidate_bundle = {}
        self._v4_4_route_family = self._route_family(dataset_profile, metadata)
        self._v4_4_execution_mode_hint = self._initial_mode_hint(self._budget_bucket(metadata))
        self._v4_4_focus_node_ids = set()
        self._v4_4_current_graph = graph
        try:
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
        finally:
            self._v4_4_current_graph = None
        if result.signature.startswith("stage2_v2|"):
            result.signature = "stage2_v4_4|" + result.signature[len("stage2_v2|") :]
        result.metadata.update(self._last_v4_4_selection)
        result.metadata.update(self._graph_faithfulness_metrics(result))
        result.metadata["stage2_version"] = "v4.4"
        result.metadata["v4_4_all_task_nodes_each_turn"] = False
        result.metadata["v4_4_route_family"] = self._v4_4_route_family
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
        with open(os.path.join(replay_dir, "v4_4_selection.json"), "w", encoding="utf-8") as handle:
            json.dump(self._last_v4_4_selection, handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                    "metadata": result.metadata | self._last_v4_4_selection,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
