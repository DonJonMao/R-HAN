from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple

Edge = Tuple[str, str]
Vector = List[float]


class WorkflowTemplate(str, Enum):
    DIRECT = "direct"
    PARALLEL_VOTE = "parallel_vote"
    SOLVE_VERIFY = "solve_verify"
    DEBATE_JUDGE = "debate_judge"
    CRITIQUE_REVISE = "critique_revise"
    ROUTE_SOLVE = "route_solve"


class PromptSlotName(str, Enum):
    REASONING = "reasoning_mode"
    UPSTREAM = "upstream_usage"
    OUTPUT = "output_style"
    VERIFY = "verification_mode"
    FINAL = "finalization"


@dataclass(frozen=True)
class PromptSlots:
    reasoning_mode: str = "direct"
    upstream_usage: str = "summary"
    output_style: str = "raw"
    verification_mode: str = "off"
    finalization: str = "answer_only"

    def with_slot(self, slot_name: str, value: str) -> "PromptSlots":
        kwargs = {
            "reasoning_mode": self.reasoning_mode,
            "upstream_usage": self.upstream_usage,
            "output_style": self.output_style,
            "verification_mode": self.verification_mode,
            "finalization": self.finalization,
        }
        if slot_name not in kwargs:
            raise KeyError(f"Unknown prompt slot: {slot_name}")
        kwargs[slot_name] = value
        return PromptSlots(**kwargs)

    def complexity(self) -> float:
        score = 0.0
        if self.reasoning_mode != "direct":
            score += 0.25
        if self.upstream_usage != "summary":
            score += 0.15
        if self.output_style != "raw":
            score += 0.10
        if self.verification_mode != "off":
            score += 0.25
        if self.finalization != "answer_only":
            score += 0.10
        return score


@dataclass(frozen=True)
class ArchitectureState:
    template: WorkflowTemplate
    role_to_agent: Dict[str, str]
    role_to_prompt: Dict[str, PromptSlots]
    prompt_edit_count: int = 0
    non_prompt_steps_since_edit: int = 0

    def active_agents(self) -> List[str]:
        seen: List[str] = []
        for aid in self.role_to_agent.values():
            if aid and aid not in seen:
                seen.append(aid)
        return seen


@dataclass
class CompiledArchitecture:
    state: ArchitectureState
    nodes: List[str]
    edges: List[Edge]
    sinks: List[str]
    execution_roles: List[str]

    def signature(self) -> str:
        roles = ",".join(f"{k}:{self.state.role_to_agent.get(k,'')}" for k in sorted(self.state.role_to_agent))
        prompts = ",".join(
            f"{role}:{self.state.role_to_prompt[role]}"
            for role in sorted(self.state.role_to_prompt)
        )
        edges = ",".join(f"{src}>{dst}" for src, dst in sorted(self.edges))
        return f"{self.state.template.value}|{roles}|{prompts}|{edges}"


@dataclass
class TaskEvaluation:
    task_score: float
    success: float
    latency: float
    token_cost: float
    safety_penalty: float
    raw_output: str = ""
    trace: List[Dict[str, str]] = field(default_factory=list)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    debug_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalSummary:
    tier: str
    mean_reward: float
    reward_std: float
    mean_task_score: float
    mean_success: float
    mean_latency: float
    mean_token_cost: float
    mean_safety_penalty: float
    evaluations: List[TaskEvaluation] = field(default_factory=list)


@dataclass
class SearchRecord:
    state_signature: str
    parent_signature: Optional[str]
    action: str
    proxy_score: float
    tier1_score: Optional[float]
    tier2_score: Optional[float]


@dataclass
class SearchNodeStats:
    visits: int = 0
    q_mean: float = 0.0
    q_max: float = float("-inf")
    proxy_mean: float = 0.0
    tier1_mean: float = 0.0
    tier2_mean: float = 0.0
    tier2_std: float = 0.0


@dataclass
class SearchNode:
    state: ArchitectureState
    compiled: CompiledArchitecture
    parent_signature: Optional[str]
    action_from_parent: str
    stats: SearchNodeStats = field(default_factory=SearchNodeStats)
    children: List[str] = field(default_factory=list)
    unexpanded_actions: List["EditAction"] = field(default_factory=list)
    proxy_score: Optional[float] = None
    proxy_uncertainty: Optional[float] = None
    tier1: Optional[EvalSummary] = None
    tier2: Optional[EvalSummary] = None


@dataclass(frozen=True)
class EditAction:
    kind: str
    payload: Dict[str, str]

    def describe(self) -> str:
        items = ",".join(f"{k}={v}" for k, v in sorted(self.payload.items()))
        return f"{self.kind}({items})"


@dataclass
class SearchResult:
    question_text: str
    selected_agents: List[str]
    root_signatures: List[str]
    best_node: SearchNode
    top_nodes: List[SearchNode]
    records: List[SearchRecord]
    nodes: Dict[str, SearchNode]
