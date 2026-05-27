from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from mas_treesearch.types import UnionGraph, UnionNode

from .contracts import contract_residual, parse_answer_object
from .pipeline import RollbackPreparedStage1Artifact


@dataclass
class TraceStep:
    index: int
    node_id: str
    brief_key: str
    block_id: int
    state_vector: List[float]
    action_summary: str
    output_summary: str
    provenance_node_ids: List[str]
    provenance_edge_ids: List[str]
    local_signal: Dict[str, float]
    brief: Dict[str, Any]
    contract_residual: Dict[str, float]


@dataclass
class RollbackTrace:
    steps: List[TraceStep]
    macro_blocks: Dict[int, List[int]]
    empty_state: Dict[str, Any] = field(default_factory=lambda: {"node_id": "__empty__", "brief_key": "__empty__"})


def _topological_nodes(graph: UnionGraph) -> List[UnionNode]:
    indegree: Dict[str, int] = {node_id: 0 for node_id in graph.nodes}
    outgoing: Dict[str, List[str]] = defaultdict(list)
    for edge in graph.edges:
        outgoing[edge.src].append(edge.dst)
        indegree[edge.dst] = indegree.get(edge.dst, 0) + 1
    queue = deque([graph.nodes[node_id] for node_id, degree in indegree.items() if degree == 0])
    ordered: List[UnionNode] = []
    seen = set()
    while queue:
        node = queue.popleft()
        if node.node_id in seen:
            continue
        seen.add(node.node_id)
        ordered.append(node)
        for dst in outgoing.get(node.node_id, []):
            indegree[dst] -= 1
            if indegree[dst] <= 0 and dst in graph.nodes:
                queue.append(graph.nodes[dst])
    if len(ordered) < len(graph.nodes):
        for node_id, node in graph.nodes.items():
            if node_id not in seen:
                ordered.append(node)
    return ordered


def _prompt_brief(node: UnionNode) -> Dict[str, Any]:
    prompt_slots = dict(node.metadata.get("prompt_slots", {}) if isinstance(node.metadata, dict) else {})
    return {
        "agent_id": node.agent_id,
        "role": node.role,
        "reasoning_mode": str(prompt_slots.get("reasoning_mode", "unknown")),
        "output_style": str(prompt_slots.get("output_style", "raw")),
        "verification_mode": str(prompt_slots.get("verification_mode", "off")),
        "finalization": str(prompt_slots.get("finalization", "answer_only")),
    }


def _brief_key(brief: Dict[str, Any]) -> str:
    return "|".join(
        [
            str(brief.get("agent_id", "")),
            str(brief.get("role", "")),
            str(brief.get("reasoning_mode", "")),
            str(brief.get("output_style", "")),
            str(brief.get("verification_mode", "")),
            str(brief.get("finalization", "")),
        ]
    )


def build_rollback_trace(
    prepared: RollbackPreparedStage1Artifact,
    *,
    question_text: str,
    candidate_text: str,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> RollbackTrace:
    metadata = dict(metadata or {})
    if prepared.base_prepared is None:
        answer_obj = parse_answer_object(
            candidate_text,
            dataset_name=dataset_name,
            answer_format=answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        residual = contract_residual(
            answer_obj,
            answer_format=answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        return RollbackTrace(
            steps=[
                TraceStep(
                    index=1,
                    node_id="anchor@fallback",
                    brief_key="fallback|answer_only",
                    block_id=1,
                    state_vector=[1.0, 0.0, 0.0],
                    action_summary="fallback_answer_emit",
                    output_summary=candidate_text,
                    provenance_node_ids=["anchor@fallback"],
                    provenance_edge_ids=[],
                    local_signal={"support_count": 1.0, "avg_graph_score": 0.0, "sink_frequency": 1.0},
                    brief={"agent_id": "fallback", "role": "anchor", "reasoning_mode": "direct", "output_style": "raw"},
                    contract_residual={
                        "parse": residual.parse,
                        "completeness": residual.completeness,
                        "constraint": residual.constraint,
                        "execution": residual.execution,
                    },
                )
            ],
            macro_blocks={1: [1]},
        )
    graph = prepared.base_prepared.union_graph
    ordered_nodes = _topological_nodes(graph)
    outgoing_edges: Dict[str, List[str]] = defaultdict(list)
    incoming_edges: Dict[str, List[str]] = defaultdict(list)
    for edge_index, edge in enumerate(graph.edges):
        edge_id = f"e{edge_index}:{edge.src}->{edge.dst}"
        outgoing_edges[edge.src].append(edge_id)
        incoming_edges[edge.dst].append(edge_id)
    steps: List[TraceStep] = []
    last_key = None
    current_block = 0
    for index, node in enumerate(ordered_nodes, start=1):
        brief = _prompt_brief(node)
        brief_key = _brief_key(brief)
        if brief_key != last_key:
            current_block += 1
            last_key = brief_key
        is_sink_like = node.node_id in set(graph.sink_node_ids) or node.sink_frequency > 0.0 or node.role in {"reviser", "verifier", "aggregator"}
        output_summary = candidate_text if is_sink_like else f"{node.role}:{node.agent_id}:{brief.get('reasoning_mode', 'unknown')}"
        answer_obj = parse_answer_object(
            output_summary,
            dataset_name=dataset_name,
            answer_format=answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        residual = contract_residual(
            answer_obj,
            answer_format=answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        steps.append(
            TraceStep(
                index=index,
                node_id=node.node_id,
                brief_key=brief_key,
                block_id=current_block,
                state_vector=[float(x) for x in list(node.state_vector)],
                action_summary=f"{node.role}:{node.agent_id}",
                output_summary=output_summary,
                provenance_node_ids=[node.node_id],
                provenance_edge_ids=sorted(outgoing_edges.get(node.node_id, []) + incoming_edges.get(node.node_id, [])),
                local_signal={
                    "support_count": float(node.support_count),
                    "avg_graph_score": float(node.avg_graph_score),
                    "root_frequency": float(node.root_frequency),
                    "sink_frequency": float(node.sink_frequency),
                },
                brief=brief,
                contract_residual={
                    "parse": residual.parse,
                    "completeness": residual.completeness,
                    "constraint": residual.constraint,
                    "execution": residual.execution,
                },
            )
        )
    macro_blocks: Dict[int, List[int]] = defaultdict(list)
    for step in steps:
        macro_blocks[step.block_id].append(step.index)
    return RollbackTrace(steps=steps, macro_blocks=dict(macro_blocks))
