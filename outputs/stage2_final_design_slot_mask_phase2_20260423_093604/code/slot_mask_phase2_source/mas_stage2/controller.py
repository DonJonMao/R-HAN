from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from mas_treesearch.profiles import DEFAULT_PROFILE, DatasetProfile
from mas_treesearch.types import UnionGraph

from .types import ControllerState, FeedbackEvent


class GlobalController:
    def __init__(self) -> None:
        self._default_roles = (
            "solver",
            "solver_a",
            "solver_b",
            "generator",
            "critic",
            "reviser",
            "verifier",
            "aggregator",
            "judge",
            "router",
        )

    @staticmethod
    def _merge_role_adjustments(role_weights: Dict[str, float], role_adjustments: Optional[Dict[str, float]]) -> Dict[str, float]:
        if not role_adjustments:
            return role_weights
        merged = dict(role_weights)
        for role, delta in role_adjustments.items():
            merged[role] = min(1.60, max(0.70, merged.get(role, 1.0) + float(delta)))
        return merged

    def _base_role_weights(self, dataset_profile: DatasetProfile) -> Dict[str, float]:
        weights = {role: 1.0 for role in self._default_roles}
        task_type = dataset_profile.task_type
        if task_type == "code_generation":
            weights["verifier"] = 1.20
            weights["critic"] = 1.10
            weights["reviser"] = 1.08
            weights["aggregator"] = 1.05
        elif task_type in {"numeric", "math_expression"}:
            weights["verifier"] = 1.12
            weights["judge"] = 1.08
            weights["critic"] = 1.04
        elif task_type == "graph_reasoning":
            weights["verifier"] = 1.12
            weights["judge"] = 1.10
            weights["aggregator"] = 1.06
        elif task_type == "mcq":
            weights["judge"] = 1.12
            weights["critic"] = 1.08
        return weights

    def bootstrap(
        self,
        graph: UnionGraph,
        dataset_profile: DatasetProfile = DEFAULT_PROFILE,
        *,
        role_adjustments: Optional[Dict[str, float]] = None,
    ) -> ControllerState:
        role_weights = self._merge_role_adjustments(self._base_role_weights(dataset_profile), role_adjustments)
        focus = "Establish a reliable first candidate and collect validation signals."
        summary = "\n".join(
            [
                "MODE=explore",
                f"FOCUS={focus}",
                "PRIORITY_ROLES="
                + ",".join(
                    role for role, weight in sorted(role_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)[:4]
                ),
                f"TASK_NODES={len([node for node in graph.nodes.values() if node.node_type == 'task'])}",
            ]
        )
        return ControllerState(
            turn_index=0,
            mode="explore",
            focus=focus,
            uncertainty=0.35,
            role_weights=role_weights,
            summary=summary,
        )

    @staticmethod
    def _event_count(events: Iterable[FeedbackEvent], kinds: set[str]) -> int:
        return sum(1 for event in events if event.event_type in kinds)

    def update(
        self,
        previous: ControllerState,
        *,
        turn_index: int,
        total_turns: int,
        feedback_events: List[FeedbackEvent],
        sink_outputs: Dict[str, str],
        role_adjustments: Optional[Dict[str, float]] = None,
    ) -> ControllerState:
        challenge_count = self._event_count(feedback_events, {"challenge", "reject", "conflict"})
        support_count = self._event_count(feedback_events, {"pass", "preserve"})
        uncertain_count = self._event_count(feedback_events, {"uncertain"})
        total_events = max(1, len(feedback_events))
        uncertainty = min(1.0, (challenge_count + uncertain_count) / total_events)
        turns_left = max(0, total_turns - (turn_index + 1))
        if turns_left <= 1:
            mode = "finalize"
            focus = "Concentrate surviving evidence and produce a decisive answer."
        elif challenge_count > support_count:
            mode = "tighten"
            focus = "Increase verification pressure and remove shaky communication paths."
        else:
            mode = "refine"
            focus = "Keep the strongest partial answer and only revise the unstable parts."
        role_weights = dict(previous.role_weights)
        if challenge_count > 0:
            role_weights["verifier"] = min(1.45, role_weights.get("verifier", 1.0) + 0.08)
            role_weights["critic"] = min(1.35, role_weights.get("critic", 1.0) + 0.05)
        if support_count > challenge_count:
            role_weights["aggregator"] = min(1.35, role_weights.get("aggregator", 1.0) + 0.06)
            role_weights["judge"] = min(1.30, role_weights.get("judge", 1.0) + 0.04)
        if mode == "finalize":
            role_weights["aggregator"] = max(role_weights.get("aggregator", 1.0), 1.20)
            role_weights["judge"] = max(role_weights.get("judge", 1.0), 1.15)
        role_weights = self._merge_role_adjustments(role_weights, role_adjustments)
        provisional = next((text for text in sink_outputs.values() if text.strip()), "None yet.")
        priority_roles = ",".join(
            role for role, weight in sorted(role_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)[:4]
        )
        summary = "\n".join(
            [
                f"MODE={mode}",
                f"FOCUS={focus}",
                f"UNCERTAINTY={uncertainty:.3f}",
                f"PRIORITY_ROLES={priority_roles}",
                f"PROVISIONAL={provisional[:220]}",
            ]
        )
        return ControllerState(
            turn_index=turn_index,
            mode=mode,
            focus=focus,
            uncertainty=uncertainty,
            role_weights=role_weights,
            summary=summary,
            metadata={
                "challenge_count": challenge_count,
                "support_count": support_count,
                "uncertain_count": uncertain_count,
            },
        )
