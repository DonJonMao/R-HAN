from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Sequence, Set

from mas_stage2.types import ControllerState, EdgeActivation, FeedbackEvent
from mas_treesearch.types import UnionGraph, UnionNode

from .artifacts import clamp01, sparsemax


@dataclass
class UnifiedControllerSnapshot:
    node_participation: Dict[str, float]
    memory_view_attention: Dict[str, float]
    halt_mass: float
    plateau_count: int
    focus_ids: Set[str]

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["focus_ids"] = sorted(self.focus_ids)
        return payload


def _feedback_focus_ids(previous_feedback: Sequence[FeedbackEvent]) -> Set[str]:
    focus = {
        event.target_node_id
        for event in previous_feedback
        if event.event_type in {"challenge", "reject", "conflict", "revise"}
    }
    if focus:
        return focus
    return {
        event.target_node_id
        for event in previous_feedback
        if event.event_type in {"pass", "preserve"}
    }


def build_memory_view_attention(
    previous_feedback: Sequence[FeedbackEvent],
    *,
    uncertainty: float,
) -> Dict[str, float]:
    support_count = sum(1 for event in previous_feedback if event.event_type in {"pass", "preserve"})
    challenge_count = sum(1 for event in previous_feedback if event.event_type in {"challenge", "reject", "conflict", "revise"})
    raw = (
        0.35 + 0.12 * support_count,
        0.35 + 0.12 * challenge_count + 0.20 * uncertainty,
        0.25 + 0.10 * len(previous_feedback),
        0.20 + 0.18 * uncertainty,
    )
    weights = sparsemax(raw)
    return {
        name: float(weight)
        for name, weight in zip(("stable", "failure", "provenance", "global"), weights)
    }


def build_controller_state(
    *,
    graph: Optional[UnionGraph],
    task_nodes: Sequence[UnionNode],
    previous_feedback: Sequence[FeedbackEvent],
    previous_snapshot: Optional[UnifiedControllerSnapshot],
    turn_index: int,
    total_turns: int,
    last_residual_mean: float,
    last_utility_delta: float,
    active_edges: Sequence[EdgeActivation],
    node_top_k: int,
) -> tuple[ControllerState, UnifiedControllerSnapshot]:
    focus_ids = _feedback_focus_ids(previous_feedback)
    total_feedback = max(1, len(previous_feedback))
    challenge_count = sum(1 for event in previous_feedback if event.event_type in {"challenge", "reject", "conflict", "revise"})
    uncertain_count = sum(1 for event in previous_feedback if event.event_type == "uncertain")
    uncertainty = 0.35 if not previous_feedback else clamp01((challenge_count + uncertain_count) / float(total_feedback))
    memory_view_attention = build_memory_view_attention(previous_feedback, uncertainty=uncertainty)
    plateau_count = 0
    if previous_snapshot is not None:
        plateau_count = int(previous_snapshot.plateau_count)
        if abs(float(last_utility_delta)) <= 0.01:
            plateau_count += 1
        else:
            plateau_count = 0
    halt_mass = clamp01(
        0.30 * max(0.0, last_residual_mean - 0.20)
        + 0.35 * (1.0 - max(0.0, last_utility_delta))
        + 0.10 * uncertainty
        + 0.12 * min(1.0, plateau_count / 2.0)
        + 0.13 * float(turn_index + 1) / float(max(1, total_turns))
    )

    node_scores = []
    protected_ids = set(graph.root_node_ids if graph is not None else []) | set(graph.sink_node_ids if graph is not None else [])
    active_incident = {
        node_id
        for activation in active_edges
        if activation.active
        for node_id in (activation.src, activation.dst)
    }
    for node in task_nodes:
        score = 0.15 + 0.08 * float(node.support_count)
        if node.node_id in focus_ids:
            score += 0.25
        if node.node_id in protected_ids:
            score += 0.20
        if node.node_id in active_incident:
            score += 0.10
        if node.role in {"aggregator", "verifier", "critic"}:
            score += 0.05
        node_scores.append(score)
    node_weights = sparsemax(node_scores)
    if node_top_k > 0:
        ranked = sorted(
            zip(task_nodes, node_weights),
            key=lambda item: (item[1], item[0].node_id),
            reverse=True,
        )
        keep_ids = {node.node_id for node, _ in ranked[: min(node_top_k, len(ranked))]}
        keep_ids |= protected_ids
        node_participation = {
            node.node_id: (float(weight) if node.node_id in keep_ids else 0.0)
            for node, weight in zip(task_nodes, node_weights)
        }
    else:
        node_participation = {
            node.node_id: float(weight)
            for node, weight in zip(task_nodes, node_weights)
        }
    role_weights: Dict[str, float] = {}
    for node in task_nodes:
        role_weights[node.role] = role_weights.get(node.role, 0.0) + float(node_participation.get(node.node_id, 0.0))
    parts = [
        f"TURN={turn_index + 1}/{total_turns}",
        f"HALT_MASS={halt_mass:.3f}",
        f"UNCERTAINTY={uncertainty:.3f}",
        f"RESIDUAL={last_residual_mean:.3f}",
        f"UTILITY_DELTA={last_utility_delta:.3f}",
        f"FOCUS={','.join(sorted(focus_ids)) or 'none'}",
        "MEMORY=" + ",".join(f"{name}:{weight:.2f}" for name, weight in memory_view_attention.items()),
    ]
    state = ControllerState(
        turn_index=turn_index,
        mode="phase1_semantic_safe_override",
        focus=",".join(sorted(focus_ids)[:3]),
        uncertainty=uncertainty,
        role_weights={role: clamp01(weight + 0.5) for role, weight in role_weights.items()},
        summary="\n".join(parts),
        metadata={
            "phase1_halt_mass": halt_mass,
            "phase1_memory_view_attention": dict(memory_view_attention),
            "phase1_focus_ids": sorted(focus_ids),
            "phase1_last_residual_mean": last_residual_mean,
            "phase1_last_utility_delta": last_utility_delta,
            "phase1_plateau_count": plateau_count,
            "phase1_node_participation": dict(node_participation),
        },
    )
    snapshot = UnifiedControllerSnapshot(
        node_participation=node_participation,
        memory_view_attention=memory_view_attention,
        halt_mass=halt_mass,
        plateau_count=plateau_count,
        focus_ids=focus_ids,
    )
    return state, snapshot
