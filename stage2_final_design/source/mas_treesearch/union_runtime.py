from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .agents import AgentPool
from .clients import CachedEmbedder
from .config import UnionRuntimeConfig
from .evaluator import MultiFidelityEvaluator
from .gating import cosine
from .profiles import DEFAULT_PROFILE, DatasetProfile
from .prompting import build_system_prompt, render_question_text
from .reward import risk_adjusted_score
from .types import EvalSummary, MemoryChunk, PromptSlots, SearchNode, UnionEdge, UnionGraph, UnionNode


@dataclass
class UnionRuntimeResult:
    summary: EvalSummary
    signature: str
    turn_traces: List[Dict[str, object]]


def _truncate_text(text: str, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."

class GraphMerger:
    def __init__(self, config: UnionRuntimeConfig):
        self.config = config

    @staticmethod
    def _node_id(role: str, agent_id: str) -> str:
        return f"{role}@{agent_id}"

    def merge(self, topologies: Sequence[SearchNode]) -> UnionGraph:
        node_acc: Dict[str, Dict[str, object]] = {}
        edge_acc: Dict[Tuple[str, str], Dict[str, object]] = {}
        graph_count = max(1, len(topologies))
        source_signatures = [node.compiled.signature() for node in topologies]

        for node in topologies:
            graph_id = node.compiled.signature()
            graph_score = risk_adjusted_score(node.tier2, 0.0) if node.tier2 is not None else (node.proxy_score or 0.0)
            role_by_agent = {agent_id: role for role, agent_id in node.state.role_to_agent.items()}
            incoming_agents = {dst for _, dst in node.compiled.edges}
            root_agents = {agent_id for agent_id in node.compiled.nodes if agent_id not in incoming_agents}
            sink_agents = set(node.compiled.sinks)
            topo_level = {role: idx for idx, role in enumerate(node.compiled.execution_roles)}

            for role, agent_id in node.state.role_to_agent.items():
                union_node_id = self._node_id(role, agent_id)
                acc = node_acc.setdefault(
                    union_node_id,
                    {
                        "agent_id": agent_id,
                        "role": role,
                        "source_graph_ids": [],
                        "score_sum": 0.0,
                        "support_count": 0,
                        "root_hits": 0.0,
                        "sink_hits": 0.0,
                        "level_values": [],
                        "best_prompt_score": float("-inf"),
                        "prompt_slots": asdict(PromptSlots()),
                    },
                )
                acc["source_graph_ids"].append(graph_id)
                acc["score_sum"] = float(acc["score_sum"]) + graph_score
                acc["support_count"] = int(acc["support_count"]) + 1
                acc["root_hits"] = float(acc["root_hits"]) + (1.0 if agent_id in root_agents else 0.0)
                acc["sink_hits"] = float(acc["sink_hits"]) + (1.0 if agent_id in sink_agents else 0.0)
                acc["level_values"].append(float(topo_level.get(role, len(node.compiled.execution_roles))))
                if graph_score >= float(acc["best_prompt_score"]):
                    acc["best_prompt_score"] = graph_score
                    acc["prompt_slots"] = asdict(node.state.role_to_prompt.get(role, PromptSlots()))

            for src_agent, dst_agent in node.compiled.edges:
                src_role = role_by_agent.get(src_agent, src_agent)
                dst_role = role_by_agent.get(dst_agent, dst_agent)
                edge_key = (self._node_id(src_role, src_agent), self._node_id(dst_role, dst_agent))
                acc = edge_acc.setdefault(
                    edge_key,
                    {
                        "source_graph_ids": [],
                        "support_count": 0,
                        "score_sum": 0.0,
                        "best_score": float("-inf"),
                        "level_deltas": [],
                    },
                )
                acc["source_graph_ids"].append(graph_id)
                acc["support_count"] = int(acc["support_count"]) + 1
                acc["score_sum"] = float(acc["score_sum"]) + graph_score
                acc["best_score"] = max(float(acc["best_score"]), graph_score)
                acc["level_deltas"].append(float(topo_level.get(dst_role, 1) - topo_level.get(src_role, 0)))

        nodes: Dict[str, UnionNode] = {}
        for node_id, acc in node_acc.items():
            levels = list(acc["level_values"])
            level_mean = sum(levels) / max(1, len(levels))
            level_var = sum((value - level_mean) ** 2 for value in levels) / max(1, len(levels))
            support_count = int(acc["support_count"])
            avg_graph_score = float(acc["score_sum"]) / max(1, support_count)
            nodes[node_id] = UnionNode(
                node_id=node_id,
                agent_id=str(acc["agent_id"]),
                role=str(acc["role"]),
                node_type="task",
                source_graph_ids=list(acc["source_graph_ids"]),
                support_count=support_count,
                avg_graph_score=avg_graph_score,
                root_frequency=float(acc["root_hits"]) / graph_count,
                sink_frequency=float(acc["sink_hits"]) / graph_count,
                topo_level_mean=level_mean,
                topo_level_var=level_var,
                state_vector=[
                    support_count / graph_count,
                    avg_graph_score,
                    float(acc["root_hits"]) / graph_count,
                    float(acc["sink_hits"]) / graph_count,
                    level_mean,
                    level_var,
                ],
                metadata={"prompt_slots": dict(acc["prompt_slots"])},
            )

        edges: List[UnionEdge] = []
        incoming_counts: Dict[str, int] = {}
        outgoing_counts: Dict[str, int] = {}
        for (src, dst), acc in edge_acc.items():
            support_count = int(acc["support_count"])
            support_ratio = support_count / graph_count
            avg_parent_score = float(acc["score_sum"]) / max(1, support_count)
            level_deltas = list(acc["level_deltas"])
            level_delta_mean = sum(level_deltas) / max(1, len(level_deltas))
            edge = UnionEdge(
                src=src,
                dst=dst,
                edge_type="task",
                source_graph_ids=list(acc["source_graph_ids"]),
                support_count=support_count,
                support_ratio=support_ratio,
                avg_parent_score=avg_parent_score,
                best_parent_score=float(acc["best_score"]),
                initial_keep_logit=0.45 + 0.35 * support_ratio + 0.20 * avg_parent_score,
                dynamic_keep_weight=0.0,
                level_delta_mean=level_delta_mean,
                latency_prior=max(0.0, level_delta_mean - 1.0),
                token_cost_prior=0.12 * support_count,
            )
            edges.append(edge)
            incoming_counts[dst] = incoming_counts.get(dst, 0) + 1
            outgoing_counts[src] = outgoing_counts.get(src, 0) + 1

        root_node_ids = sorted(
            [node_id for node_id in nodes if incoming_counts.get(node_id, 0) == 0],
            key=lambda node_id: (nodes[node_id].topo_level_mean, -nodes[node_id].support_count, node_id),
        )
        sink_node_ids = sorted(
            [node_id for node_id in nodes if outgoing_counts.get(node_id, 0) == 0],
            key=lambda node_id: (nodes[node_id].topo_level_mean, -nodes[node_id].support_count, node_id),
        )

        special_nodes = {
            "super_source": UnionNode(
                node_id="super_source",
                agent_id="super_source",
                role="super_source",
                node_type="global",
                source_graph_ids=list(source_signatures),
                support_count=graph_count,
                avg_graph_score=0.0,
            ),
            "global_controller": UnionNode(
                node_id="global_controller",
                agent_id="global_controller",
                role="global_controller",
                node_type="global",
                source_graph_ids=list(source_signatures),
                support_count=graph_count,
                avg_graph_score=0.0,
            ),
            "super_sink": UnionNode(
                node_id="super_sink",
                agent_id="super_sink",
                role="super_sink",
                node_type="global",
                source_graph_ids=list(source_signatures),
                support_count=graph_count,
                avg_graph_score=0.0,
            ),
        }
        nodes.update(special_nodes)

        for root_id in root_node_ids:
            edges.append(
                UnionEdge(
                    src="super_source",
                    dst=root_id,
                    edge_type="source",
                    source_graph_ids=list(source_signatures),
                    support_count=graph_count,
                    support_ratio=1.0,
                    avg_parent_score=0.0,
                    best_parent_score=0.0,
                    initial_keep_logit=1.0,
                    dynamic_keep_weight=1.0,
                )
            )
            edges.append(
                UnionEdge(
                    src="global_controller",
                    dst=root_id,
                    edge_type="controller",
                    source_graph_ids=list(source_signatures),
                    support_count=graph_count,
                    support_ratio=1.0,
                    avg_parent_score=0.0,
                    best_parent_score=0.0,
                    initial_keep_logit=1.0,
                    dynamic_keep_weight=1.0,
                )
            )
        for sink_id in sink_node_ids:
            edges.append(
                UnionEdge(
                    src=sink_id,
                    dst="global_controller",
                    edge_type="feedback",
                    source_graph_ids=list(source_signatures),
                    support_count=graph_count,
                    support_ratio=1.0,
                    avg_parent_score=0.0,
                    best_parent_score=0.0,
                    initial_keep_logit=1.0,
                    dynamic_keep_weight=1.0,
                )
            )
            edges.append(
                UnionEdge(
                    src=sink_id,
                    dst="super_sink",
                    edge_type="finalize",
                    source_graph_ids=list(source_signatures),
                    support_count=graph_count,
                    support_ratio=1.0,
                    avg_parent_score=0.0,
                    best_parent_score=0.0,
                    initial_keep_logit=1.0,
                    dynamic_keep_weight=1.0,
                )
            )
        edges.append(
            UnionEdge(
                src="global_controller",
                dst="super_sink",
                edge_type="finalize",
                source_graph_ids=list(source_signatures),
                support_count=graph_count,
                support_ratio=1.0,
                avg_parent_score=0.0,
                best_parent_score=0.0,
                initial_keep_logit=1.0,
                dynamic_keep_weight=1.0,
            )
        )

        return UnionGraph(
            nodes=nodes,
            edges=edges,
            source_topology_signatures=list(source_signatures),
            root_node_ids=root_node_ids,
            sink_node_ids=sink_node_ids,
            metadata={
                "graph_count": graph_count,
                "task_node_count": len([node for node in nodes.values() if node.node_type == "task"]),
            },
        )


class UnionRuntime:
    def __init__(
        self,
        config: UnionRuntimeConfig,
        evaluator: MultiFidelityEvaluator,
        agent_pool: AgentPool,
        embedder: CachedEmbedder,
    ):
        self.config = config
        self.evaluator = evaluator
        self.agent_pool = agent_pool
        self.embedder = embedder
        self._by_id = agent_pool.by_id()

    @staticmethod
    def _prompt_slots(node: UnionNode) -> PromptSlots:
        payload = node.metadata.get("prompt_slots", {})
        if not isinstance(payload, dict):
            return PromptSlots()
        values = {
            "reasoning_mode": payload.get("reasoning_mode", PromptSlots().reasoning_mode),
            "upstream_usage": payload.get("upstream_usage", PromptSlots().upstream_usage),
            "output_style": payload.get("output_style", PromptSlots().output_style),
            "verification_mode": payload.get("verification_mode", PromptSlots().verification_mode),
            "finalization": payload.get("finalization", PromptSlots().finalization),
        }
        return PromptSlots(**values)

    @staticmethod
    def _incoming_edges(graph: UnionGraph, node_id: str) -> List[UnionEdge]:
        return [edge for edge in graph.edges if edge.dst == node_id]

    @staticmethod
    def _outgoing_edges(graph: UnionGraph, node_id: str) -> List[UnionEdge]:
        return [edge for edge in graph.edges if edge.src == node_id]

    def _task_order(self, graph: UnionGraph) -> List[UnionNode]:
        task_nodes = [node for node in graph.nodes.values() if node.node_type == "task"]
        task_nodes.sort(key=lambda node: (node.topo_level_mean, -node.support_count, node.node_id))
        return task_nodes

    def _edge_weight(
        self,
        edge: UnionEdge,
        latest_outputs: Dict[str, str],
        *,
        turn_index: int,
    ) -> float:
        if edge.edge_type in {"source", "finalize"}:
            return 1.0
        if edge.edge_type == "controller":
            return 0.55 + 0.10 * turn_index
        if edge.edge_type == "feedback":
            return 0.70
        src_output = latest_outputs.get(edge.src, "")
        output_bonus = 0.08 if src_output.strip() else 0.0
        return max(
            0.0,
            min(
                1.0,
                0.45 * edge.support_ratio + 0.25 * edge.avg_parent_score + 0.20 * edge.best_parent_score + output_bonus,
            ),
        )

    def _active_incoming(
        self,
        graph: UnionGraph,
        node_id: str,
        latest_outputs: Dict[str, str],
        *,
        turn_index: int,
    ) -> List[UnionEdge]:
        incoming = self._incoming_edges(graph, node_id)
        if not incoming:
            return []
        weighted: List[Tuple[float, UnionEdge]] = []
        for edge in incoming:
            weight = self._edge_weight(edge, latest_outputs, turn_index=turn_index)
            edge.dynamic_keep_weight = weight
            weighted.append((weight, edge))
        weighted.sort(key=lambda item: (item[0], item[1].support_ratio, item[1].src), reverse=True)
        keep = [edge for weight, edge in weighted if weight >= self.config.edge_prune_threshold]
        if not keep:
            keep = [weighted[0][1]]
        return keep

    def _memory_candidates(
        self,
        chunks: Sequence[MemoryChunk],
        node_id: str,
        incoming_edges: Sequence[UnionEdge],
    ) -> List[MemoryChunk]:
        allowed_node_ids = {node_id, "global_controller"}
        allowed_node_ids.update(edge.src for edge in incoming_edges if edge.src not in {"super_source", "super_sink"})
        return [chunk for chunk in chunks if chunk.metadata.get("node_id") in allowed_node_ids]

    def _score_memory(
        self,
        query_vec: Sequence[float],
        chunk: MemoryChunk,
        *,
        current_turn: int,
    ) -> float:
        recency = 1.0 / max(1.0, 1.0 + (current_turn - chunk.turn_index))
        return (
            0.55 * cosine(query_vec, chunk.embedding)
            + 0.15 * chunk.confidence
            + 0.15 * chunk.importance
            + 0.10 * chunk.novelty
            + 0.05 * recency
        )

    def _select_memories(
        self,
        question_text: str,
        controller_summary: str,
        *,
        chunks: Sequence[MemoryChunk],
        node_id: str,
        incoming_edges: Sequence[UnionEdge],
        current_turn: int,
    ) -> List[MemoryChunk]:
        candidates = self._memory_candidates(chunks, node_id, incoming_edges)
        if not candidates:
            return []
        query_vec = self.embedder.embed(f"{question_text}\n{controller_summary}".strip())
        ranked = sorted(
            candidates,
            key=lambda chunk: (
                self._score_memory(query_vec, chunk, current_turn=current_turn),
                chunk.turn_index,
                chunk.chunk_id,
            ),
            reverse=True,
        )
        return ranked[: max(1, self.config.memory_top_k)]

    @staticmethod
    def _memory_block(chunks: Sequence[MemoryChunk], *, limit: int) -> str:
        if not chunks:
            return "No selected memory."
        blocks = []
        for chunk in chunks:
            label = f"{chunk.role}@{chunk.agent_id}#t{chunk.turn_index}"
            blocks.append(f"[{label}]\n{_truncate_text(chunk.text, limit)}")
        return "\n\n".join(blocks)

    def _role_instruction(
        self,
        node: UnionNode,
        *,
        turn_index: int,
        dataset_profile: DatasetProfile,
        metadata: Optional[dict],
        final_turn: bool,
    ) -> str:
        mapping = {
            "solver": "Produce the strongest current candidate answer.",
            "solver_a": "Produce one strong candidate from your preferred angle.",
            "solver_b": "Produce a materially different candidate when possible.",
            "generator": "Draft a concrete candidate that others can inspect.",
            "critic": "Find the most likely bug, gap, or violated constraint.",
            "reviser": "Revise the candidate by fixing the most important issues.",
            "verifier": "Check correctness, edge cases, and output contract.",
            "aggregator": "Merge upstream candidates into one stronger candidate.",
            "judge": "Compare competing candidates and decide which is more reliable.",
            "router": "Identify the key subproblem and what should be attempted next.",
        }
        parts = [
            f"Turn {turn_index + 1}/{self.config.turn_count}.",
            mapping.get(node.role, "Execute your assigned role in the current union graph."),
        ]
        if dataset_profile.task_type == "code_generation":
            entry_point = str((metadata or {}).get("entry_point") or "").strip()
            parts.append(
                "This is a Python code-generation task. Prefer a complete, executable implementation that handles edge cases."
            )
            if entry_point:
                parts.append(f"The required function name is `{entry_point}`.")
            if node.role in {"critic", "verifier", "judge"}:
                parts.append("Focus on signature fidelity, corner cases, and hidden-test failure modes.")
            elif node.role in {"aggregator", "reviser", "solver", "solver_a", "solver_b", "generator"}:
                parts.append("Keep the implementation simple, deterministic, and free of explanatory prose.")
        elif dataset_profile.task_type in {"numeric", "math_expression"}:
            parts.append("Prioritize mathematical correctness over stylistic variety.")
        if final_turn:
            parts.append("Prefer a decisive output rather than open questions.")
        return " ".join(parts)

    def _run_task_node(
        self,
        graph: UnionGraph,
        node: UnionNode,
        *,
        question_text: str,
        controller_summary: str,
        current_turn: int,
        chunks: List[MemoryChunk],
        latest_outputs: Dict[str, str],
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> Tuple[str, Dict[str, object], List[MemoryChunk]]:
        incoming = self._active_incoming(graph, node.node_id, latest_outputs, turn_index=current_turn)
        selected_memories = self._select_memories(
            question_text,
            controller_summary,
            chunks=chunks,
            node_id=node.node_id,
            incoming_edges=incoming,
            current_turn=current_turn,
        )
        answer_contract = ""
        if node.node_id in graph.sink_node_ids or node.role in {"aggregator", "reviser", "verifier", "judge"}:
            answer_contract = self.evaluator._output_contract(
                question_text,
                reference_answer=reference_answer,
                metadata=metadata,
            )
        memory_block = self._memory_block(selected_memories, limit=self.config.memory_max_chars)
        task_context = self.evaluator._task_context(
            question_text,
            dataset_profile=dataset_profile,
            metadata=metadata,
            role=node.role,
        )
        agent = self._by_id[node.agent_id]
        messages = [
            {
                "role": "system",
                "content": build_system_prompt(agent, self._prompt_slots(node), extra_role_hint=node.role),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                    f"Global controller state:\n{_truncate_text(controller_summary, self.config.controller_max_chars)}\n\n"
                    f"Selected memory:\n{memory_block}\n\n"
                    f"Current task:\n{self._role_instruction(node, turn_index=current_turn, dataset_profile=dataset_profile, metadata=metadata, final_turn=current_turn + 1 == self.config.turn_count)}\n"
                    + (f"\n\nTask context:\n{task_context}" if task_context else "")
                    + (f"\n\nOutput contract:\n{answer_contract}" if answer_contract else "")
                ),
            },
        ]
        raw_output = self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))
        if answer_contract:
            output = self.evaluator._sanitize_final_output(
                question_text,
                raw_output,
                reference_answer=reference_answer,
                metadata=metadata,
            )
        else:
            output = self.evaluator._strip_hidden_reasoning(raw_output)
        latest_outputs[node.node_id] = output
        confidence = max(0.15, min(1.0, 0.45 + 0.20 * node.avg_graph_score + 0.10 * node.support_count))
        importance = 0.40 + (0.20 if node.node_id in graph.sink_node_ids else 0.0)
        new_chunk = MemoryChunk(
            chunk_id=f"{node.node_id}::turn{current_turn}",
            agent_id=node.agent_id,
            role=node.role,
            turn_index=current_turn,
            source_type="union_node",
            text=output,
            embedding=self.embedder.embed(output),
            token_estimate=max(1, len(output.split())),
            confidence=confidence,
            importance=importance,
            novelty=max(0.0, 1.0 - (len(selected_memories) / max(1, self.config.memory_top_k * 2))),
            metadata={
                "node_id": node.node_id,
                "selected_memory_ids": [chunk.chunk_id for chunk in selected_memories],
                "incoming_edges": [f"{edge.src}->{edge.dst}" for edge in incoming],
            },
        )
        return output, {
            "node_id": node.node_id,
            "agent_id": node.agent_id,
            "role": node.role,
            "incoming_edges": [f"{edge.src}->{edge.dst}" for edge in incoming],
            "selected_memory_ids": [chunk.chunk_id for chunk in selected_memories],
            "output": output,
        }, [new_chunk]

    def _controller_summary(
        self,
        graph: UnionGraph,
        *,
        question_text: str,
        sink_outputs: Dict[str, str],
        controller_summary: str,
        current_turn: int,
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> str:
        sink_block = "\n\n".join(
            f"[{node_id}]\n{_truncate_text(text, self.config.controller_max_chars)}"
            for node_id, text in sink_outputs.items()
            if text.strip()
        )
        if not sink_block:
            sink_block = "No sink output available."
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the global controller of a multi-agent union graph. "
                    "Maintain a compact shared state for the next round. "
                    "Output three labeled sections only: PROVISIONAL_ANSWER, OPEN_ISSUES, NEXT_FOCUS."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                    f"Previous controller state:\n{_truncate_text(controller_summary, self.config.controller_max_chars)}\n\n"
                    f"Current sink outputs:\n{sink_block}\n\n"
                    f"Round: {current_turn + 1}/{self.config.turn_count}\n"
                    "Write a better provisional answer, list the remaining risks, and say what the next round should focus on."
                ),
            },
        ]
        return self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))

    def _finalize_answer(
        self,
        *,
        question_text: str,
        controller_summary: str,
        sink_outputs: Dict[str, str],
        metadata: Optional[dict],
        reference_answer: Optional[str],
        dataset_profile: DatasetProfile,
    ) -> str:
        sink_block = "\n\n".join(
            f"[{node_id}]\n{_truncate_text(text, self.config.controller_max_chars)}"
            for node_id, text in sink_outputs.items()
            if text.strip()
        )
        answer_contract = self.evaluator._output_contract(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are the final controller of a multi-agent union graph. "
                    "Produce the final answer only, strictly following the output contract."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{render_question_text(question_text, metadata=metadata)}\n\n"
                    f"Controller summary:\n{_truncate_text(controller_summary, self.config.controller_max_chars)}\n\n"
                    f"Sink outputs:\n{sink_block}\n\n"
                    f"Output contract:\n{answer_contract}"
                ),
            },
        ]
        output = self.evaluator._cached_chat(messages, runtime=self.evaluator._resolve_runtime("tier2", dataset_profile))
        return self.evaluator._sanitize_final_output(
            question_text,
            output,
            reference_answer=reference_answer,
            metadata=metadata,
        )

    def run(
        self,
        graph: UnionGraph,
        *,
        question_text: str,
        metadata: Optional[dict] = None,
        reference_answer: Optional[str] = None,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> UnionRuntimeResult:
        start = time.perf_counter()
        task_order = self._task_order(graph)
        latest_outputs: Dict[str, str] = {}
        controller_summary = "PROVISIONAL_ANSWER:\nNone yet.\n\nOPEN_ISSUES:\nNo shared state yet.\n\nNEXT_FOCUS:\nEstablish a reliable first candidate."
        all_chunks: List[MemoryChunk] = []
        turn_traces: List[Dict[str, object]] = []

        for turn_index in range(self.config.turn_count):
            node_traces: List[Dict[str, object]] = []
            sink_outputs: Dict[str, str] = {}
            for node in task_order:
                output, trace, new_chunks = self._run_task_node(
                    graph,
                    node,
                    question_text=question_text,
                    controller_summary=controller_summary,
                    current_turn=turn_index,
                    chunks=all_chunks,
                    latest_outputs=latest_outputs,
                    metadata=metadata,
                    reference_answer=reference_answer,
                    dataset_profile=dataset_profile,
                )
                node_traces.append(trace)
                all_chunks.extend(new_chunks)
                if node.node_id in graph.sink_node_ids:
                    sink_outputs[node.node_id] = output

            controller_summary = self._controller_summary(
                graph,
                question_text=question_text,
                sink_outputs=sink_outputs,
                controller_summary=controller_summary,
                current_turn=turn_index,
                metadata=metadata,
                reference_answer=reference_answer,
                dataset_profile=dataset_profile,
            )
            latest_outputs["global_controller"] = controller_summary
            all_chunks.append(
                MemoryChunk(
                    chunk_id=f"global_controller::turn{turn_index}",
                    agent_id="global_controller",
                    role="global_controller",
                    turn_index=turn_index,
                    source_type="controller",
                    text=controller_summary,
                    embedding=self.embedder.embed(controller_summary),
                    token_estimate=max(1, len(controller_summary.split())),
                    confidence=0.65,
                    importance=0.75,
                    novelty=0.20,
                    metadata={"node_id": "global_controller"},
                )
            )
            turn_traces.append(
                {
                    "turn_index": turn_index,
                    "controller_summary": controller_summary,
                    "node_traces": node_traces,
                    "sink_outputs": sink_outputs,
                }
            )

        final_answer = self._finalize_answer(
            question_text=question_text,
            controller_summary=controller_summary,
            sink_outputs=turn_traces[-1].get("sink_outputs", {}) if turn_traces else {},
            metadata=metadata,
            reference_answer=reference_answer,
            dataset_profile=dataset_profile,
        )
        total_latency = time.perf_counter() - start
        total_token_cost = (
            sum(chunk.token_estimate for chunk in all_chunks) * self.evaluator.config.token_cost_per_word
        )
        size_penalty = self.config.union_size_penalty_scale * max(0.0, len(task_order) - 4)
        summary = self.evaluator.evaluate_output(
            question_text,
            final_answer,
            tier="tier2",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=dataset_profile,
            custom_metrics={
                "union_turns": float(self.config.turn_count),
                "union_task_nodes": float(len(task_order)),
                "union_source_graphs": float(len(graph.source_topology_signatures)),
            },
            size_penalty=size_penalty,
            latency=total_latency,
            token_cost=total_token_cost,
        )
        signature = "union|" + "||".join(graph.source_topology_signatures)
        return UnionRuntimeResult(summary=summary, signature=signature, turn_traces=turn_traces)
