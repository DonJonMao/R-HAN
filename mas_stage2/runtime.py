from __future__ import annotations

import json
import os
import time
from collections import Counter
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_treesearch.agents import AgentPool, default_agent_pool
from mas_treesearch.cache import DictCache
from mas_treesearch.clients import CachedEmbedder
from mas_treesearch.config import TieredEvalConfig
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.prompting import build_system_prompt, render_question_text
from mas_treesearch.types import EvalSummary, PromptSlots, UnionEdge, UnionGraph, UnionNode

from .config import Stage2RuntimeConfig
from .controller import GlobalController
from .learning import OnlineLinearModel, controller_features, edge_features
from .memory import (
    ExportMessageBuilder,
    LocalMemoryComposer,
    MemoryBriefVerbalizer,
    PrivateEpisodeMemoryStore,
    RoleAwareMemorySelector,
)
from .types import (
    ControllerState,
    EdgeActivation,
    ExportedMemoryMessage,
    FeedbackEvent,
    MemoryRecord,
    NodeTurnTrace,
    Stage2RunResult,
    TurnTrace,
)


def _truncate(text: str, limit: int) -> str:
    cleaned = (text or "").strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


class Stage2Runtime:
    def __init__(
        self,
        config: Stage2RuntimeConfig,
        evaluator: MultiFidelityEvaluator,
        agent_pool: AgentPool,
        embedder: CachedEmbedder,
    ):
        self.config = config
        self.evaluator = evaluator
        self.agent_pool = agent_pool
        self.embedder = embedder
        self._by_id = agent_pool.by_id()
        self._memory_store = PrivateEpisodeMemoryStore(config.memory)
        self._selector_model = OnlineLinearModel(learning_rate=config.learning.selector_learning_rate)
        self._edge_model = OnlineLinearModel(learning_rate=config.learning.edge_learning_rate)
        self._controller_model = OnlineLinearModel(learning_rate=config.learning.controller_learning_rate)
        self._selector = RoleAwareMemorySelector(
            config.memory,
            embedder,
            learned_model=self._selector_model,
            learned_weight=config.learning.selector_model_weight,
        )
        self._composer = LocalMemoryComposer(config.memory, embedder)
        self._exporter = ExportMessageBuilder(config.memory, embedder)
        self._verbalizer = MemoryBriefVerbalizer(config.memory)
        self._controller = GlobalController()
        self._sink_distance_cache: Dict[Tuple[Any, ...], Dict[str, int]] = {}

    @staticmethod
    def _prompt_slots(node: UnionNode) -> PromptSlots:
        payload = node.metadata.get("prompt_slots", {})
        if not isinstance(payload, dict):
            return PromptSlots()
        return PromptSlots(
            reasoning_mode=str(payload.get("reasoning_mode", PromptSlots().reasoning_mode)),
            upstream_usage=str(payload.get("upstream_usage", PromptSlots().upstream_usage)),
            output_style=str(payload.get("output_style", PromptSlots().output_style)),
            verification_mode=str(payload.get("verification_mode", PromptSlots().verification_mode)),
            finalization=str(payload.get("finalization", PromptSlots().finalization)),
        )

    @staticmethod
    def _task_nodes(graph: UnionGraph) -> List[UnionNode]:
        nodes = [node for node in graph.nodes.values() if node.node_type == "task"]
        nodes.sort(key=lambda node: (node.topo_level_mean, -node.support_count, node.node_id))
        return nodes

    @staticmethod
    def _task_edges(graph: UnionGraph) -> List[UnionEdge]:
        return [
            edge
            for edge in graph.edges
            if edge.src in graph.nodes
            and edge.dst in graph.nodes
            and graph.nodes[edge.src].node_type == "task"
            and graph.nodes[edge.dst].node_type == "task"
        ]

    @staticmethod
    def _edge_id(edge: UnionEdge) -> str:
        return f"{edge.src}->{edge.dst}"

    def _feedback_summary(self, events: Sequence[FeedbackEvent]) -> Dict[str, Dict[str, int]]:
        by_target: Dict[str, Dict[str, int]] = {}
        for event in events:
            bucket = by_target.setdefault(event.target_node_id, {})
            bucket[event.event_type] = bucket.get(event.event_type, 0) + 1
        return by_target

    def _token_cost_from_estimate(self, token_estimate: int) -> float:
        return float(token_estimate) * float(self.evaluator.config.token_cost_per_word)

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _summary_target(self, summary: Optional[EvalSummary], reward_target: Optional[float]) -> float:
        if reward_target is not None:
            return self._clamp01(reward_target)
        if summary is None:
            return 0.5
        return self._clamp01(0.55 * float(summary.mean_success) + 0.45 * float(summary.mean_task_score))

    def _feedback_adjusted_target(
        self,
        base_target: float,
        bucket: Dict[str, int],
        *,
        is_sink: bool,
    ) -> float:
        target = float(base_target)
        positive = int(bucket.get("pass", 0)) + int(bucket.get("preserve", 0))
        negative = int(bucket.get("challenge", 0)) + int(bucket.get("reject", 0)) + int(bucket.get("conflict", 0))
        uncertain = int(bucket.get("uncertain", 0))
        target += self.config.learning.positive_feedback_bonus * min(1.0, float(positive))
        target -= self.config.learning.negative_feedback_penalty * min(1.0, float(negative))
        target -= self.config.learning.uncertain_penalty * min(1.0, float(uncertain))
        if is_sink and positive > negative:
            target += self.config.learning.sink_bonus
        return self._clamp01(target)

    def _controller_role_adjustments(
        self,
        dataset_profile: DatasetProfile,
        controller_state: Optional[ControllerState],
        *,
        turn_index: int,
        total_turns: int,
        feedback_events: Sequence[FeedbackEvent],
    ) -> Dict[str, float]:
        if not self.config.learning.enabled:
            return {}
        support_count = sum(1 for event in feedback_events if event.event_type in {"pass", "preserve"})
        challenge_count = sum(1 for event in feedback_events if event.event_type in {"challenge", "reject", "conflict"})
        uncertain_count = sum(1 for event in feedback_events if event.event_type == "uncertain")
        if controller_state is None:
            base_roles = {role: 1.0 for role in getattr(self._controller, "_default_roles", ())}
            controller_state = ControllerState(
                turn_index=0,
                mode="explore",
                focus="bootstrap",
                uncertainty=0.35,
                role_weights=base_roles,
                summary="MODE=explore",
            )
        adjustments: Dict[str, float] = {}
        for role in controller_state.role_weights:
            features = controller_features(
                role,
                dataset_profile,
                controller_state=controller_state,
                turn_index=turn_index,
                total_turns=total_turns,
                support_count=support_count,
                challenge_count=challenge_count,
                uncertain_count=uncertain_count,
            )
            predicted, _ = self._controller_model.predict(features)
            adjustments[role] = self.config.learning.controller_model_weight * ((predicted - 0.5) * 2.0)
        return adjustments

    def state_dict(self) -> dict:
        return {
            "selector_model": self._selector_model.state_dict(),
            "edge_model": self._edge_model.state_dict(),
            "controller_model": self._controller_model.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        selector_state = state.get("selector_model")
        if isinstance(selector_state, dict):
            self._selector_model.load_state_dict(selector_state)
        edge_state = state.get("edge_model")
        if isinstance(edge_state, dict):
            self._edge_model.load_state_dict(edge_state)
        controller_state = state.get("controller_model")
        if isinstance(controller_state, dict):
            self._controller_model.load_state_dict(controller_state)

    def _edge_score(
        self,
        graph: UnionGraph,
        edge: UnionEdge,
        controller_state: ControllerState,
        feedback_counts: Dict[str, Dict[str, int]],
    ) -> Tuple[float, str, Dict[str, object]]:
        src_node = graph.nodes.get(edge.src)
        dst_node = graph.nodes.get(edge.dst)
        src_role = src_node.role if src_node is not None else ""
        dst_role = dst_node.role if dst_node is not None else ""
        role_boost = (
            (controller_state.role_weights.get(src_role, 1.0) - 1.0)
            + (controller_state.role_weights.get(dst_role, 1.0) - 1.0)
        ) * self.config.graph.controller_role_boost
        src_feedback = feedback_counts.get(edge.src, {})
        support_count = int(src_feedback.get("pass", 0) + src_feedback.get("preserve", 0))
        challenge_count = int(src_feedback.get("challenge", 0) + src_feedback.get("reject", 0) + src_feedback.get("conflict", 0))
        support_bonus = self.config.graph.support_bonus * float(support_count)
        challenge_penalty = self.config.graph.challenge_penalty * float(challenge_count)
        heuristic_score = (
            0.35 * edge.initial_keep_logit
            + 0.20 * edge.support_ratio
            + 0.15 * edge.avg_parent_score
            + 0.10 * edge.best_parent_score
            + role_boost
            + support_bonus
            - challenge_penalty
        )
        features = edge_features(
            edge,
            src_role=src_role,
            dst_role=dst_role,
            controller_state=controller_state,
            support_count=support_count,
            challenge_count=challenge_count,
        )
        learned_score = 0.5
        if self.config.learning.enabled:
            learned_score, _ = self._edge_model.predict(features)
            learned_score = self._clamp01(learned_score)
        score = heuristic_score + self.config.learning.edge_model_weight * ((learned_score - 0.5) * 2.0)
        score = max(0.0, min(1.0, score))
        reason = (
            f"prior={edge.initial_keep_logit:.2f},role={role_boost:.2f},support={support_bonus:.2f},"
            f"challenge=-{challenge_penalty:.2f},learned={learned_score:.2f}"
        )
        return score, reason, {
            "features": features,
            "heuristic_score": float(heuristic_score),
            "learned_score": float(learned_score),
            "src_role": src_role,
            "dst_role": dst_role,
        }

    def _activate_edges(
        self,
        graph: UnionGraph,
        controller_state: ControllerState,
        previous_feedback: Sequence[FeedbackEvent],
        *,
        turn_index: int,
    ) -> List[EdgeActivation]:
        feedback_counts = self._feedback_summary(previous_feedback)
        activations: List[EdgeActivation] = []
        task_edges = self._task_edges(graph)
        grouped: Dict[str, List[Tuple[float, UnionEdge, str, Dict[str, object]]]] = {}
        scored_edges: Dict[str, Tuple[float, str, Dict[str, object]]] = {}
        for edge in task_edges:
            score, reason, metadata = self._edge_score(graph, edge, controller_state, feedback_counts)
            edge_id = self._edge_id(edge)
            scored_edges[edge_id] = (score, reason, metadata)
            grouped.setdefault(edge.dst, []).append((score, edge, reason, metadata))
        active_ids: set[str] = set()
        for dst, items in grouped.items():
            items.sort(key=lambda item: (item[0], item[1].support_ratio, item[1].src), reverse=True)
            kept = [item for item in items if item[0] >= self.config.graph.soft_prune_threshold]
            if not kept:
                kept = items[: self.config.graph.min_incoming_edges]
            if len(kept) > self.config.graph.soft_prune_top_k:
                kept = kept[: self.config.graph.soft_prune_top_k]
            for _, edge, _, _ in kept:
                active_ids.add(self._edge_id(edge))
        for edge in task_edges:
            edge_id = self._edge_id(edge)
            score, reason, metadata = scored_edges[edge_id]
            active = edge_id in active_ids
            if turn_index + 1 >= self.config.graph.hard_prune_after_turn and score < self.config.graph.soft_prune_threshold:
                active = False
            activations.append(
                EdgeActivation(
                    edge_id=edge_id,
                    src=edge.src,
                    dst=edge.dst,
                    score=score,
                    active=active,
                    reason=reason,
                    metadata=dict(metadata),
                )
            )
        return activations

    def _neighbour_exports(
        self,
        node_id: str,
        active_edges: Sequence[EdgeActivation],
        previous_exports: Dict[str, ExportedMemoryMessage],
    ) -> List[ExportedMemoryMessage]:
        messages = [
            previous_exports[activation.src]
            for activation in active_edges
            if activation.active and activation.dst == node_id and activation.src in previous_exports
        ]
        messages.sort(key=lambda item: (item.confidence, item.node_id), reverse=True)
        return messages[: self.config.memory.max_neighbour_exports]

    @staticmethod
    def _code_candidate_roles() -> set[str]:
        return {"solver", "solver_a", "solver_b", "generator", "reviser", "aggregator"}

    @staticmethod
    def _code_feedback_roles() -> set[str]:
        return {"critic", "verifier", "judge"}

    @staticmethod
    def _runtime_checker_roles() -> set[str]:
        return {"tester", "verifier", "critic", "judge", "checker"}

    def _sink_distance_lookup(self, graph: UnionGraph) -> Dict[str, int]:
        if not hasattr(self, "_sink_distance_cache"):
            self._sink_distance_cache = {}
        cache_key = (
            tuple(sorted(graph.nodes)),
            tuple(sorted((edge.src, edge.dst) for edge in graph.edges)),
            tuple(sorted(graph.sink_node_ids)),
        )
        cached = self._sink_distance_cache.get(cache_key)
        if cached is not None:
            return cached
        reverse_adj: Dict[str, List[str]] = {}
        for edge in graph.edges:
            reverse_adj.setdefault(edge.dst, []).append(edge.src)
        distances: Dict[str, int] = {node_id: 10**9 for node_id in graph.nodes}
        frontier: List[str] = [node_id for node_id in graph.sink_node_ids if node_id in graph.nodes]
        for node_id in frontier:
            distances[node_id] = 0
        index = 0
        while index < len(frontier):
            node_id = frontier[index]
            index += 1
            base_distance = distances[node_id]
            for upstream in reverse_adj.get(node_id, ()):
                if upstream not in distances:
                    continue
                if base_distance + 1 >= distances[upstream]:
                    continue
                distances[upstream] = base_distance + 1
                frontier.append(upstream)
        self._sink_distance_cache[cache_key] = distances
        return distances

    def _runtime_node_type(self, graph: UnionGraph, node: UnionNode) -> str:
        if node.node_id in graph.sink_node_ids:
            return "sink"
        role = str(node.role)
        if role in self._runtime_checker_roles():
            return "checker"
        distances = self._sink_distance_lookup(graph)
        node_distance = distances.get(node.node_id, 10**9)
        if node_distance >= 10**9:
            return "proposal"
        candidate_shell = [
            graph_node
            for graph_node in graph.nodes.values()
            if graph_node.node_type == "task"
            and graph_node.node_id not in graph.sink_node_ids
            and str(graph_node.role) not in self._runtime_checker_roles()
            and distances.get(graph_node.node_id, 10**9) < 10**9
        ]
        if not candidate_shell:
            return "proposal"
        nearest_shell = min(distances.get(graph_node.node_id, 10**9) for graph_node in candidate_shell)
        shell_nodes = [
            graph_node
            for graph_node in candidate_shell
            if distances.get(graph_node.node_id, 10**9) == nearest_shell
        ]
        shell_support_max = max((int(graph_node.support_count) for graph_node in shell_nodes), default=-1)
        node_support = int(node.support_count)
        if node_distance == nearest_shell and node_support >= shell_support_max:
            return "aggregator"
        # Allow explicit aggregation/router roles one shell farther from sink when
        # they remain well supported by Stage1. This keeps routing topology-aware
        # without collapsing back to role-only typing.
        if role in {"aggregator", "router"} and node_distance <= nearest_shell + 1:
            support_floor = shell_support_max if node_distance == nearest_shell else max(0, shell_support_max - 1)
            if node_support >= support_floor:
                return "aggregator"
        if role == "reviser" and node_distance == nearest_shell and node_support >= max(0, shell_support_max - 1):
            return "aggregator"
        return "proposal"

    def _annotate_runtime_node(self, graph: UnionGraph, node: UnionNode) -> UnionNode:
        metadata = dict(node.metadata or {})
        distances = self._sink_distance_lookup(graph)
        metadata.update(
            {
                "runtime_node_type": self._runtime_node_type(graph, node),
                "sink_distance": int(distances.get(node.node_id, 10**9)),
                "stage1_support": int(node.support_count),
                "is_sink_runtime": bool(node.node_id in graph.sink_node_ids),
            }
        )
        node.metadata = metadata
        return node

    def _priority_node_ids(
        self,
        task_nodes: Sequence[UnionNode],
        controller_state: ControllerState,
    ) -> set[str]:
        priority_roles = {
            role
            for role, _ in sorted(
                controller_state.role_weights.items(),
                key=lambda item: (item[1], item[0]),
                reverse=True,
            )[:2]
        }
        return {node.node_id for node in task_nodes if node.role in priority_roles}

    @staticmethod
    def _feedback_recovery_node_ids(previous_feedback: Sequence[FeedbackEvent]) -> set[str]:
        recovery_ids: set[str] = set()
        for event in previous_feedback:
            if event.event_type in {"challenge", "reject", "conflict", "revise"}:
                recovery_ids.add(event.target_node_id)
                recovery_ids.add(event.source_node_id)
        return recovery_ids

    def _active_task_nodes(
        self,
        graph: UnionGraph,
        task_nodes: Sequence[UnionNode],
        active_edges: Sequence[EdgeActivation],
        controller_state: ControllerState,
        previous_feedback: Sequence[FeedbackEvent],
        *,
        turn_index: int,
    ) -> List[UnionNode]:
        if turn_index <= 0:
            return list(task_nodes)
        incident_ids: set[str] = set()
        for activation in active_edges:
            if not activation.active:
                continue
            incident_ids.add(activation.src)
            incident_ids.add(activation.dst)
        preserved_ids = {
            event.target_node_id
            for event in previous_feedback
            if event.event_type in {"pass", "preserve"}
        }
        recovery_ids = self._feedback_recovery_node_ids(previous_feedback)
        protected_ids = {
            node_id
            for node_id in graph.sink_node_ids
            if node_id in graph.nodes and graph.nodes[node_id].node_type == "task"
        }
        protected_ids |= self._priority_node_ids(task_nodes, controller_state)
        active_ids = incident_ids | preserved_ids | recovery_ids | protected_ids
        if not active_ids:
            return list(task_nodes)
        return [node for node in task_nodes if node.node_id in active_ids]

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
        if dataset_profile.task_type == "code_generation":
            if node.role in self._code_feedback_roles():
                return (
                    "Do not output Python code.\n"
                    "Output exactly three lines:\n"
                    "VERDICT: pass|challenge|uncertain\n"
                    "ISSUES: <short text>\n"
                    "FIX: <short text or none>"
                )
            if node.role == "router":
                return (
                    "Do not output Python code.\n"
                    "Output exactly two lines:\n"
                    "FOCUS: <short text>\n"
                    "PLAN: <short text>"
                )
            if node.role in self._code_candidate_roles():
                return self.evaluator._output_contract(
                    question_text,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
        if node.role in {"aggregator", "reviser", "verifier", "judge"} or node.node_id in graph.sink_node_ids:
            return self.evaluator._output_contract(
                question_text,
                reference_answer=reference_answer,
                metadata=metadata,
            )
        return ""

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
        if dataset_profile.task_type == "code_generation":
            if node.role in self._code_candidate_roles():
                return self.evaluator._sanitize_final_output(
                    question_text,
                    raw_output,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
            return self.evaluator._strip_hidden_reasoning(raw_output)
        if answer_contract:
            return self.evaluator._sanitize_final_output(
                question_text,
                raw_output,
                reference_answer=reference_answer,
                metadata=metadata,
            )
        return self.evaluator._strip_hidden_reasoning(raw_output)

    def _sink_candidate_counts(
        self,
        turn_traces: Sequence[TurnTrace],
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Counter[str]:
        candidates: List[str] = []
        for turn_trace in turn_traces:
            for text in turn_trace.sink_outputs.values():
                if not text.strip():
                    continue
                candidate = self.evaluator._sanitize_final_output(
                    question_text,
                    text,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
                if candidate.strip():
                    candidates.append(candidate)
        return Counter(candidates)

    def _preserve_numeric_candidate(
        self,
        turn_traces: Sequence[TurnTrace],
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
    ) -> Optional[str]:
        counts = self._sink_candidate_counts(
            turn_traces,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        if not counts:
            return None
        choice, count = max(counts.items(), key=lambda item: (item[1], -len(item[0]), item[0]))
        if count >= 2 or sum(counts.values()) == 1:
            return choice
        return None

    def _preserve_consensus_candidate(
        self,
        turn_traces: Sequence[TurnTrace],
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Optional[str]:
        counts = self._sink_candidate_counts(
            turn_traces,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        if not counts:
            return None
        choice, count = max(counts.items(), key=lambda item: (item[1], -len(item[0]), item[0]))
        total = sum(counts.values())
        if total == 1 or len(counts) == 1:
            return choice
        if dataset_profile.task_type in {"mcq", "boolean"} and count >= 2:
            return choice
        if dataset_profile.task_type in {"graph_reasoning", "structured_list"} and count * 2 > total and count >= 2:
            return choice
        return None

    def _select_best_code_candidate(
        self,
        turn_traces: Sequence[TurnTrace],
        *,
        question_text: str,
        reference_answer: Optional[str],
        metadata: Optional[dict],
        dataset_profile: DatasetProfile,
    ) -> Optional[str]:
        candidates: List[Tuple[float, float, float, int, str]] = []
        seen: set[str] = set()
        for turn_trace in turn_traces:
            for node_trace in turn_trace.node_traces:
                if node_trace.role not in self._code_candidate_roles() or not node_trace.output.strip():
                    continue
                candidate = self.evaluator._sanitize_final_output(
                    question_text,
                    node_trace.output,
                    reference_answer=reference_answer,
                    metadata=metadata,
                )
                if not candidate or candidate in seen:
                    continue
                seen.add(candidate)
                summary = self.evaluator.evaluate_output(
                    question_text,
                    candidate,
                    tier="tier2",
                    reference_answer=reference_answer,
                    metadata=metadata,
                    dataset_profile=dataset_profile,
                )
                candidates.append(
                    (
                        summary.mean_success,
                        summary.mean_task_score,
                        -summary.mean_safety_penalty,
                        -len(candidate.splitlines()),
                        candidate,
                    )
                )
        if not candidates:
            return None
        candidates.sort(reverse=True)
        return candidates[0][-1]

    @staticmethod
    def _role_instruction(
        node: UnionNode,
        *,
        turn_index: int,
        total_turns: int,
        dataset_profile: DatasetProfile,
        controller_state: ControllerState,
        final_turn: bool,
    ) -> str:
        role_map = {
            "solver": "Propose the strongest current candidate answer.",
            "solver_a": "Propose one concrete candidate from your angle.",
            "solver_b": "Propose a materially different candidate when possible.",
            "generator": "Draft a concrete candidate that downstream agents can inspect.",
            "critic": "Identify the most likely flaw, inconsistency, or missing condition.",
            "reviser": "Repair the current candidate using the strongest feedback signals.",
            "verifier": "Check correctness, edge cases, and output contract compliance.",
            "aggregator": "Merge surviving candidates into one stronger answer.",
            "judge": "Decide which candidate is more reliable and why.",
            "router": "Identify the next subproblem and route attention accordingly.",
        }
        parts = [
            f"Turn {turn_index + 1}/{total_turns}.",
            role_map.get(node.role, "Execute your assigned role."),
            f"Controller mode: {controller_state.mode}.",
            f"Current focus: {controller_state.focus}",
        ]
        if dataset_profile.task_type == "code_generation":
            if node.role in {"critic", "verifier", "judge"}:
                parts.append("Do not output Python code. Return a compact review using VERDICT, ISSUES, and FIX lines.")
            elif node.role == "router":
                parts.append("Do not output Python code. Route the next repair focus in short text.")
            else:
                parts.append("If you propose a candidate, output executable Python that preserves the required function signature.")
        elif dataset_profile.task_type in {"numeric", "math_expression"}:
            parts.append("Prioritize mathematical correctness over stylistic variation.")
        if final_turn:
            parts.append("Be decisive and avoid leaving unresolved branches.")
        return " ".join(parts)

    def _run_task_node(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
        controller_state: ControllerState,
        active_edges: Sequence[EdgeActivation],
        previous_exports: Dict[str, ExportedMemoryMessage],
        episode_id: str,
        current_turn: int,
    ) -> Tuple[NodeTurnTrace, MemoryRecord, ExportedMemoryMessage]:
        node = self._annotate_runtime_node(graph, node)
        local_records = self._memory_store.get(node.node_id)
        records_by_id = {record.record_id: record for record in local_records}
        selected_items = self._selector.select(
            node,
            question_text,
            controller_state,
            local_records,
            current_turn=current_turn,
        )
        local_latent = self._composer.compose(
            node,
            records_by_id,
            selected_items,
            controller_state,
            current_turn=current_turn,
        )
        neighbour_exports = self._neighbour_exports(node.node_id, active_edges, previous_exports)
        memory_brief = self._verbalizer.build(local_latent, neighbour_exports, controller_state)
        task_context = self.evaluator._task_context(
            question_text,
            dataset_profile=dataset_profile,
            metadata=metadata,
            role=node.role,
        )
        answer_contract = self._role_answer_contract(
            graph,
            node,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
        )
        agent = self._by_id[node.agent_id]
        system_prompt = build_system_prompt(agent, self._prompt_slots(node), extra_role_hint=node.role)
        user_prompt = (
            f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
            f"Memory brief:\n{memory_brief}\n\n"
            f"Current task:\n{self._role_instruction(node, turn_index=current_turn, total_turns=self.config.graph.turn_count, dataset_profile=dataset_profile, controller_state=controller_state, final_turn=current_turn + 1 == self.config.graph.turn_count)}"
        )
        if task_context:
            user_prompt += f"\n\nTask context:\n{task_context}"
        if answer_contract:
            user_prompt += f"\n\nOutput contract:\n{answer_contract}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        raw_output = self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))
        output = self._postprocess_node_output(
            node,
            raw_output,
            question_text=question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
            answer_contract=answer_contract,
        )
        output_record = MemoryRecord(
            record_id=f"{node.node_id}::turn{current_turn}::output",
            episode_id=episode_id,
            turn_index=current_turn,
            owner_node_id=node.node_id,
            agent_id=node.agent_id,
            role=node.role,
            record_type="self_output",
            text=output,
            embedding=self.embedder.embed(output),
            token_estimate=max(1, len(output.split())),
            feedback_type="unresolved",
            confidence=min(1.0, 0.45 + 0.10 * len(selected_items)),
            source_node_id=node.node_id,
            metadata={
                "selected_record_ids": [item.record_id for item in selected_items],
                "neighbour_sources": [message.node_id for message in neighbour_exports],
                "runtime_node_type": str(node.metadata.get("runtime_node_type", "")),
            },
        )
        export_message = self._exporter.build(node, local_latent, output)
        trace = NodeTurnTrace(
            node_id=node.node_id,
            agent_id=node.agent_id,
            role=node.role,
            active_incoming_edge_ids=[edge.edge_id for edge in active_edges if edge.active and edge.dst == node.node_id],
            selected_records=list(selected_items),
            neighbour_sources=[message.node_id for message in neighbour_exports],
            memory_brief=memory_brief,
            output=output,
            local_latent_summary=local_latent.summary,
            exported_summary=export_message.summary if self.config.replay.save_exports else "",
            prompt_excerpt=_truncate(user_prompt, self.config.replay.max_prompt_chars) if self.config.replay.save_prompts else "",
            metadata={
                "selected_record_count": len(selected_items),
                "runtime_node_type": str(node.metadata.get("runtime_node_type", "")),
                "sink_distance": int(node.metadata.get("sink_distance", 10**9)),
                "stage1_support": int(node.metadata.get("stage1_support", node.support_count)),
            },
        )
        return trace, output_record, export_message

    @staticmethod
    def _classify_feedback_text(text: str, role: str) -> Tuple[str, float]:
        lowered = text.lower()
        verdict_match = None
        for line in text.splitlines():
            if line.strip().lower().startswith("verdict:"):
                verdict_match = line.split(":", 1)[1].strip().lower()
                break
        if verdict_match in {"pass", "challenge", "uncertain", "reject", "conflict"}:
            event_type = verdict_match
            confidence = 0.78 if event_type in {"pass", "challenge", "reject", "conflict"} else 0.52
            return event_type, confidence
        challenge_words = (
            "error",
            "wrong",
            "incorrect",
            "invalid",
            "bug",
            "issue",
            "fail",
            "violate",
            "mismatch",
            "missing",
            "conflict",
        )
        support_words = (
            "correct",
            "valid",
            "pass",
            "sound",
            "consistent",
            "works",
            "satisfy",
            "looks good",
        )
        challenge_hits = sum(1 for word in challenge_words if word in lowered)
        support_hits = sum(1 for word in support_words if word in lowered)
        if role == "critic" and challenge_hits == 0:
            challenge_hits = 1 if lowered.strip() else 0
        if challenge_hits > support_hits:
            return "challenge", min(0.95, 0.55 + 0.08 * challenge_hits)
        if support_hits > challenge_hits:
            return "pass", min(0.95, 0.55 + 0.08 * support_hits)
        return "uncertain", 0.45

    def _extract_feedback_events(
        self,
        graph: UnionGraph,
        node_traces: Sequence[NodeTurnTrace],
        *,
        turn_index: int,
    ) -> List[FeedbackEvent]:
        events: List[FeedbackEvent] = []
        for trace in node_traces:
            if trace.role not in {"critic", "verifier", "judge"}:
                continue
            polarity, confidence = self._classify_feedback_text(trace.output, trace.role)
            for edge_id in trace.active_incoming_edge_ids:
                src, _, _ = edge_id.partition("->")
                if src not in graph.nodes or graph.nodes[src].node_type != "task":
                    continue
                events.append(
                    FeedbackEvent(
                        event_id=f"{trace.node_id}::{src}::turn{turn_index}",
                        turn_index=turn_index,
                        source_node_id=trace.node_id,
                        target_node_id=src,
                        source_kind=trace.role,
                        event_type=polarity,
                        confidence=confidence,
                        detail=_truncate(trace.output, 220),
                    )
                )
        sink_texts = [trace.output.strip() for trace in node_traces if trace.node_id in graph.sink_node_ids and trace.output.strip()]
        if len(set(sink_texts)) > 1:
            for trace in node_traces:
                if trace.node_id in graph.sink_node_ids:
                    events.append(
                        FeedbackEvent(
                            event_id=f"global_controller::{trace.node_id}::turn{turn_index}",
                            turn_index=turn_index,
                            source_node_id="global_controller",
                            target_node_id=trace.node_id,
                            source_kind="controller",
                            event_type="conflict",
                            confidence=0.70,
                            detail="Sink outputs disagree and require consolidation.",
                        )
                    )
        return events

    def _feedback_records(
        self,
        graph: UnionGraph,
        events: Sequence[FeedbackEvent],
        *,
        episode_id: str,
    ) -> List[MemoryRecord]:
        records: List[MemoryRecord] = []
        for event in events:
            node = graph.nodes.get(event.target_node_id)
            if node is None:
                continue
            text = f"{event.source_kind} {event.event_type}: {event.detail}"
            records.append(
                MemoryRecord(
                    record_id=f"{event.event_id}::memory",
                    episode_id=episode_id,
                    turn_index=event.turn_index,
                    owner_node_id=event.target_node_id,
                    agent_id=node.agent_id,
                    role=node.role,
                    record_type="feedback",
                    text=text,
                    embedding=self.embedder.embed(text),
                    token_estimate=max(1, len(text.split())),
                    feedback_type=event.event_type,
                    confidence=event.confidence,
                    source_node_id=event.source_node_id,
                    metadata=dict(event.metadata),
                )
            )
        return records

    def _finalize_answer(
        self,
        *,
        question_text: str,
        controller_state: ControllerState,
        sink_outputs: Dict[str, str],
        turn_traces: Sequence[TurnTrace],
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> Tuple[str, str]:
        if dataset_profile.task_type == "code_generation":
            preserved = self._select_best_code_candidate(
                turn_traces,
                question_text=question_text,
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            if preserved:
                return preserved, "preserve_code_candidate"
        if dataset_profile.task_type in {"numeric", "math_expression"}:
            preserved = self._preserve_numeric_candidate(
                turn_traces,
                question_text=question_text,
                reference_answer=reference_answer,
                metadata=metadata,
            )
            if preserved is not None:
                return preserved, "preserve_numeric_consensus"
        if dataset_profile.task_type in {"mcq", "boolean", "graph_reasoning", "structured_list"}:
            preserved = self._preserve_consensus_candidate(
                turn_traces,
                question_text=question_text,
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=dataset_profile,
            )
            if preserved is not None:
                return preserved, f"preserve_{dataset_profile.task_type}_consensus"
        sink_block = "\n\n".join(
            f"[{node_id}]\n{_truncate(text, self.config.finalizer_max_chars)}"
            for node_id, text in sink_outputs.items()
            if text.strip()
        )
        if not sink_block:
            sink_block = "No sink outputs available."
        answer_contract = self.evaluator._output_contract(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the final answer synthesizer of a three-layer MAS graph. "
                    "Prefer keeping the best sink output with minimal rewriting when it already satisfies the contract. "
                    "Return only the final answer."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                    f"Controller summary:\n{controller_state.summary}\n\n"
                    f"Sink outputs:\n{sink_block}\n\n"
                    f"Output contract:\n{answer_contract}"
                ),
            },
        ]
        raw_output = self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))
        return (
            self.evaluator._sanitize_final_output(
                question_text,
                raw_output,
                reference_answer=reference_answer,
                metadata=metadata,
            ),
            "llm_finalizer",
        )

    def learn_from_run(
        self,
        graph: UnionGraph,
        result: Stage2RunResult,
        *,
        dataset_profile: DatasetProfile,
        summary: Optional[EvalSummary] = None,
        reward_target: Optional[float] = None,
    ) -> Dict[str, float]:
        if not self.config.learning.enabled:
            return {"enabled": 0.0, "selector_updates": 0.0, "edge_updates": 0.0, "controller_updates": 0.0}
        base_target = self._summary_target(summary, reward_target)
        selector_updates = 0
        edge_updates = 0
        controller_updates = 0
        total_turns = max(1, len(result.turn_traces))
        for turn_trace in result.turn_traces:
            feedback_by_target = self._feedback_summary(turn_trace.feedback_events)
            support_count = sum(1 for event in turn_trace.feedback_events if event.event_type in {"pass", "preserve"})
            challenge_count = sum(1 for event in turn_trace.feedback_events if event.event_type in {"challenge", "reject", "conflict"})
            uncertain_count = sum(1 for event in turn_trace.feedback_events if event.event_type == "uncertain")
            for node_trace in turn_trace.node_traces:
                node_bucket = feedback_by_target.get(node_trace.node_id, {})
                local_target = self._feedback_adjusted_target(
                    base_target,
                    node_bucket,
                    is_sink=node_trace.node_id in graph.sink_node_ids,
                )
                for item in node_trace.selected_records:
                    payload = dict(item.metadata) if isinstance(item.metadata, dict) else {}
                    features = payload.get("features")
                    if isinstance(features, dict) and features:
                        self._selector_model.update({str(name): float(value) for name, value in features.items()}, local_target)
                        selector_updates += 1
                role_feats = controller_features(
                    node_trace.role,
                    dataset_profile,
                    controller_state=turn_trace.controller_state,
                    turn_index=turn_trace.turn_index,
                    total_turns=total_turns,
                    support_count=support_count,
                    challenge_count=challenge_count,
                    uncertain_count=uncertain_count,
                )
                self._controller_model.update(role_feats, local_target)
                controller_updates += 1
            for activation in turn_trace.active_edges:
                payload = dict(activation.metadata) if isinstance(activation.metadata, dict) else {}
                features = payload.get("features")
                if not isinstance(features, dict) or not features:
                    continue
                edge_bucket = feedback_by_target.get(activation.src, {})
                edge_target = self._feedback_adjusted_target(
                    base_target,
                    edge_bucket,
                    is_sink=activation.dst in graph.sink_node_ids,
                )
                if not activation.active:
                    edge_target = max(0.0, edge_target - 0.05)
                self._edge_model.update({str(name): float(value) for name, value in features.items()}, edge_target)
                edge_updates += 1
        return {
            "enabled": 1.0,
            "base_target": float(base_target),
            "selector_updates": float(selector_updates),
            "edge_updates": float(edge_updates),
            "controller_updates": float(controller_updates),
            "selector_steps": float(self._selector_model.steps),
            "edge_steps": float(self._edge_model.steps),
            "controller_steps": float(self._controller_model.steps),
        }

    def save_replay_bundle(
        self,
        result: Stage2RunResult,
        replay_dir: str,
        *,
        question_text: str,
        metadata: Optional[dict],
    ) -> None:
        os.makedirs(replay_dir, exist_ok=True)
        with open(os.path.join(replay_dir, "question.json"), "w", encoding="utf-8") as handle:
            json.dump({"question_text": question_text, "metadata": metadata or {}}, handle, ensure_ascii=False, indent=2)
        for trace in result.turn_traces:
            with open(os.path.join(replay_dir, f"turn_{trace.turn_index}.json"), "w", encoding="utf-8") as handle:
                json.dump(asdict(trace), handle, ensure_ascii=False, indent=2)
        with open(os.path.join(replay_dir, "final_answer.json"), "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "final_answer": result.final_answer,
                    "signature": result.signature,
                    "final_controller_state": asdict(result.final_controller_state),
                    "memory_record_counts": result.memory_record_counts,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def run(
        self,
        graph: UnionGraph,
        *,
        question_text: str,
        metadata: Optional[dict] = None,
        reference_answer: Optional[str] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        replay_dir: Optional[str] = None,
    ) -> Stage2RunResult:
        if self.config.allow_cross_agent_raw_memory:
            raise ValueError("Stage2Runtime requires graph-mediated memory access; cross-agent raw memory is disabled.")
        self._memory_store = PrivateEpisodeMemoryStore(self.config.memory)
        episode_id = str((metadata or {}).get("question_id") or (metadata or {}).get("id") or abs(hash(question_text)))
        start = time.perf_counter()
        task_nodes = self._task_nodes(graph)
        controller_state = self._controller.bootstrap(
            graph,
            dataset_profile,
            role_adjustments=self._controller_role_adjustments(
                dataset_profile,
                None,
                turn_index=0,
                total_turns=self.config.graph.turn_count,
                feedback_events=[],
            ),
        )
        previous_feedback: List[FeedbackEvent] = []
        previous_exports: Dict[str, ExportedMemoryMessage] = {}
        turn_traces: List[TurnTrace] = []
        final_sink_outputs: Dict[str, str] = {}
        turn_token_estimates: List[int] = []
        turn_token_costs: List[float] = []

        for turn_index in range(self.config.graph.turn_count):
            active_edges = self._activate_edges(graph, controller_state, previous_feedback, turn_index=turn_index)
            active_task_nodes = self._active_task_nodes(
                graph,
                task_nodes,
                active_edges,
                controller_state,
                previous_feedback,
                turn_index=turn_index,
            )
            active_node_ids = {node.node_id for node in active_task_nodes}
            skipped_node_ids = [node.node_id for node in task_nodes if node.node_id not in active_node_ids]
            node_traces: List[NodeTurnTrace] = []
            current_exports: Dict[str, ExportedMemoryMessage] = {}
            sink_outputs: Dict[str, str] = {}
            turn_token_estimate = 0
            for node in active_task_nodes:
                trace, output_record, export_message = self._run_task_node(
                    graph,
                    node,
                    question_text=question_text,
                    metadata=metadata,
                    reference_answer=reference_answer,
                    dataset_profile=dataset_profile,
                    controller_state=controller_state,
                    active_edges=active_edges,
                    previous_exports=previous_exports,
                    episode_id=episode_id,
                    current_turn=turn_index,
                )
                self._memory_store.add(output_record)
                turn_token_estimate += int(output_record.token_estimate)
                current_exports[node.node_id] = export_message
                node_traces.append(trace)
                if node.node_id in graph.sink_node_ids:
                    sink_outputs[node.node_id] = trace.output
            feedback_events = self._extract_feedback_events(graph, node_traces, turn_index=turn_index)
            feedback_records = self._feedback_records(graph, feedback_events, episode_id=episode_id)
            for record in feedback_records:
                self._memory_store.add(record)
                turn_token_estimate += int(record.token_estimate)
            turn_token_cost = self._token_cost_from_estimate(turn_token_estimate)
            turn_token_estimates.append(turn_token_estimate)
            turn_token_costs.append(turn_token_cost)
            controller_state = self._controller.update(
                controller_state,
                turn_index=turn_index,
                total_turns=self.config.graph.turn_count,
                feedback_events=feedback_events,
                sink_outputs=sink_outputs,
                role_adjustments=self._controller_role_adjustments(
                    dataset_profile,
                    controller_state,
                    turn_index=turn_index,
                    total_turns=self.config.graph.turn_count,
                    feedback_events=feedback_events,
                ),
            )
            turn_traces.append(
                TurnTrace(
                    turn_index=turn_index,
                    controller_state=controller_state,
                    active_edges=active_edges,
                    node_traces=node_traces,
                    feedback_events=feedback_events,
                    sink_outputs=sink_outputs,
                    metadata={
                        "elapsed_s": time.perf_counter() - start,
                        "active_node_ids": sorted(active_node_ids),
                        "skipped_node_ids": skipped_node_ids,
                        "turn_token_estimate": turn_token_estimate,
                        "turn_token_cost": turn_token_cost,
                    },
                )
            )
            previous_feedback = feedback_events
            previous_exports = current_exports
            final_sink_outputs = sink_outputs

        final_answer, finalizer_strategy = self._finalize_answer(
            question_text=question_text,
            controller_state=controller_state,
            sink_outputs=final_sink_outputs,
            turn_traces=turn_traces,
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
        )
        result = Stage2RunResult(
            final_answer=final_answer,
            final_controller_state=controller_state,
            turn_traces=turn_traces,
            memory_record_counts=self._memory_store.counts(),
            signature="stage2|" + "||".join(graph.source_topology_signatures),
            metadata={
                "elapsed_s": time.perf_counter() - start,
                "task_node_count": len(task_nodes),
                "source_graph_count": len(graph.source_topology_signatures),
                "turn_count": len(turn_traces),
                "finalizer_strategy": finalizer_strategy,
                "approx_token_cost": self._memory_store.total_token_estimate() * self.evaluator.config.token_cost_per_word,
                "turn_token_estimates": list(turn_token_estimates),
                "turn_token_costs": list(turn_token_costs),
            },
        )
        if replay_dir:
            self.save_replay_bundle(result, replay_dir, question_text=question_text, metadata=metadata)
        return result


def build_default_stage2_runtime(
    *,
    config: Optional[Stage2RuntimeConfig] = None,
    runtime_config: Optional[TieredEvalConfig] = None,
    agent_pool: Optional[AgentPool] = None,
) -> Stage2Runtime:
    resolved_config = config or Stage2RuntimeConfig()
    resolved_runtime = runtime_config or TieredEvalConfig()
    resolved_pool = agent_pool or default_agent_pool()
    embedder = CachedEmbedder(resolved_runtime.embedding, DictCache())
    evaluator = MultiFidelityEvaluator(resolved_runtime, resolved_pool)
    return Stage2Runtime(resolved_config, evaluator, resolved_pool, embedder)
