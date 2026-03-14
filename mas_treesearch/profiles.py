from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple

from .types import PromptSlots, WorkflowTemplate


def _slots(
    *,
    reasoning_mode: str = "direct",
    upstream_usage: str = "summary",
    output_style: str = "raw",
    verification_mode: str = "off",
    finalization: str = "answer_only",
) -> PromptSlots:
    return PromptSlots(
        reasoning_mode=reasoning_mode,
        upstream_usage=upstream_usage,
        output_style=output_style,
        verification_mode=verification_mode,
        finalization=finalization,
    )


@dataclass(frozen=True)
class SearchOverrides:
    search_iterations: Optional[int] = None
    candidate_core_k: Optional[int] = None
    candidate_explore_k: Optional[int] = None
    candidate_max_k: Optional[int] = None
    tier1_top_fraction: Optional[float] = None
    tier2_top_fraction: Optional[float] = None
    final_top_k: Optional[int] = None
    max_prompt_edits_per_state: Optional[int] = None
    prompt_edit_cooldown: Optional[int] = None


@dataclass(frozen=True)
class RuntimeOverrides:
    tier1_repeats: Optional[int] = None
    tier2_repeats: Optional[int] = None
    tier1_max_tokens: Optional[int] = None
    tier2_max_tokens: Optional[int] = None
    tier1_judge_max_tokens: Optional[int] = None
    tier2_judge_max_tokens: Optional[int] = None


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    task_type: str
    answer_format: str
    root_templates: Tuple[str, ...]
    allowed_templates: Tuple[str, ...]
    role_prompt_overrides: Dict[str, PromptSlots] = field(default_factory=dict)
    role_agent_preferences: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    required_agent_ids: Tuple[str, ...] = field(default_factory=tuple)
    search_overrides: SearchOverrides = field(default_factory=SearchOverrides)
    runtime_overrides: RuntimeOverrides = field(default_factory=RuntimeOverrides)
    proxy_prompt_penalty_scale: float = 1.0
    reward_prompt_penalty_scale: float = 1.0
    reward_size_penalty_scale: float = 1.0
    notes: str = ""

    def prompt_for_role(self, role: str, base: PromptSlots) -> PromptSlots:
        return self.role_prompt_overrides.get(role, base)


def _all_templates() -> Tuple[str, ...]:
    return tuple(template.value for template in WorkflowTemplate)


def _merge_dicts(*mappings: Dict[str, PromptSlots]) -> Dict[str, PromptSlots]:
    merged: Dict[str, PromptSlots] = {}
    for mapping in mappings:
        merged.update(mapping)
    return merged


def _role_slots(roles: Tuple[str, ...], **kwargs: str) -> Dict[str, PromptSlots]:
    return {role: _slots(**kwargs) for role in roles}


def _role_agents(**kwargs: Tuple[str, ...]) -> Dict[str, Tuple[str, ...]]:
    return dict(kwargs)


_MCQ_TEMPLATES = ("solve_verify", "route_solve", "parallel_vote", "critique_revise")
_REASONING_TEMPLATES = ("solve_verify", "critique_revise", "parallel_vote", "route_solve")
_LIGHT_TEMPLATES = ("direct", "solve_verify", "critique_revise", "route_solve")
_OPEN_TEMPLATES = ("route_solve", "solve_verify", "critique_revise", "parallel_vote")
_CODE_TEMPLATES = ("route_solve", "solve_verify", "critique_revise")

_SOLVER_ROLES = ("solver", "solver_a", "solver_b", "generator")
_VERIFY_ROLES = ("verifier", "critic", "judge")
_SINK_ROLES = ("aggregator", "reviser")

_MCQ_ROLE_PREFS = _role_agents(
    solver=("reasoner", "debater_a", "math"),
    solver_a=("reasoner", "math", "debater_a"),
    solver_b=("debater_b", "reasoner", "math"),
    generator=("reasoner", "debater_a", "math"),
    verifier=("verifier", "skeptic", "summarizer"),
    critic=("verifier", "skeptic", "reasoner"),
    aggregator=("summarizer", "verifier", "reasoner"),
    reviser=("summarizer", "reasoner", "verifier"),
    router=("planner", "verifier", "reasoner"),
    judge=("verifier", "summarizer", "skeptic"),
)

_MATH_ROLE_PREFS = _role_agents(
    solver=("math", "reasoner", "coder"),
    solver_a=("math", "reasoner", "coder"),
    solver_b=("reasoner", "math", "coder"),
    generator=("math", "reasoner", "coder"),
    verifier=("verifier", "math", "coder"),
    critic=("verifier", "math", "skeptic"),
    aggregator=("summarizer", "verifier", "math"),
    reviser=("math", "summarizer", "verifier"),
    router=("planner", "math", "reasoner"),
    judge=("verifier", "math", "summarizer"),
)

_GRAPH_ROLE_PREFS = _role_agents(
    solver=("coder", "reasoner", "math"),
    solver_a=("coder", "reasoner", "math"),
    solver_b=("reasoner", "coder", "math"),
    generator=("coder", "reasoner", "math"),
    verifier=("verifier", "coder", "reasoner"),
    critic=("verifier", "coder", "skeptic"),
    aggregator=("verifier", "summarizer", "coder"),
    reviser=("coder", "summarizer", "verifier"),
    router=("planner", "coder", "reasoner"),
    judge=("verifier", "summarizer", "coder"),
)

_STRUCTURED_ROLE_PREFS = _role_agents(
    solver=("researcher", "reasoner", "summarizer"),
    solver_a=("researcher", "reasoner", "summarizer"),
    solver_b=("reasoner", "researcher", "summarizer"),
    generator=("researcher", "reasoner", "summarizer"),
    verifier=("verifier", "researcher", "summarizer"),
    critic=("verifier", "researcher", "skeptic"),
    aggregator=("summarizer", "verifier", "researcher"),
    reviser=("summarizer", "researcher", "verifier"),
    router=("planner", "researcher", "reasoner"),
    judge=("verifier", "summarizer", "researcher"),
)

_CODE_ROLE_PREFS = _role_agents(
    solver=("coder", "reasoner", "planner"),
    solver_a=("coder", "reasoner", "planner"),
    solver_b=("coder", "reasoner", "planner"),
    generator=("coder", "reasoner", "planner"),
    verifier=("verifier", "coder", "reasoner"),
    critic=("verifier", "coder", "skeptic"),
    aggregator=("verifier", "summarizer", "coder"),
    reviser=("coder", "summarizer", "verifier"),
    router=("planner", "coder", "verifier"),
    judge=("verifier", "coder", "summarizer"),
)


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "mmlu": DatasetProfile(
        name="mmlu",
        task_type="mcq",
        answer_format="option",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="direct", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MCQ_ROLE_PREFS,
        required_agent_ids=("verifier",),
        search_overrides=SearchOverrides(search_iterations=8),
        notes="High-risk MCQ. Final answer must be an option index only.",
    ),
    "popqa": DatasetProfile(
        name="popqa",
        task_type="mcq",
        answer_format="option_or_option_confidence",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="direct", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MCQ_ROLE_PREFS,
        required_agent_ids=("verifier",),
        search_overrides=SearchOverrides(search_iterations=8),
        notes="MCQ with possible abstain option and sometimes confidence line.",
    ),
    "cqa": DatasetProfile(
        name="cqa",
        task_type="mcq",
        answer_format="option",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="direct", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MCQ_ROLE_PREFS,
        required_agent_ids=("verifier",),
        search_overrides=SearchOverrides(
            search_iterations=10,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            max_prompt_edits_per_state=2,
        ),
        notes="Music-domain MCQ with fixed answer format.",
    ),
    "mmlu_pro": DatasetProfile(
        name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
        root_templates=("route_solve", "solve_verify", "critique_revise", "parallel_vote"),
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MCQ_ROLE_PREFS,
        required_agent_ids=("verifier",),
        search_overrides=SearchOverrides(
            search_iterations=10,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            max_prompt_edits_per_state=2,
        ),
        notes="Long-form professional MCQ with options stored in metadata.",
    ),
    "gsm8k": DatasetProfile(
        name="gsm8k",
        task_type="numeric",
        answer_format="number",
        root_templates=_REASONING_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MATH_ROLE_PREFS,
        required_agent_ids=("math", "verifier"),
        search_overrides=SearchOverrides(search_iterations=8),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.85,
        reward_prompt_penalty_scale=0.85,
        notes="Math word problem. Final answer should be the final number only.",
    ),
    "multiarith": DatasetProfile(
        name="multiarith",
        task_type="numeric",
        answer_format="number",
        root_templates=("solve_verify", "critique_revise", "route_solve"),
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MATH_ROLE_PREFS,
        required_agent_ids=("math", "verifier"),
        search_overrides=SearchOverrides(search_iterations=6),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.9,
        reward_prompt_penalty_scale=0.9,
        notes="Short arithmetic reasoning. Final answer should be a single number.",
    ),
    "math": DatasetProfile(
        name="math",
        task_type="math_expression",
        answer_format="math_expression",
        root_templates=_REASONING_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", upstream_usage="quote_then_reason", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_MATH_ROLE_PREFS,
        required_agent_ids=("math", "verifier"),
        search_overrides=SearchOverrides(
            search_iterations=14,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=3, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        proxy_prompt_penalty_scale=0.6,
        reward_prompt_penalty_scale=0.6,
        reward_size_penalty_scale=0.85,
        notes="Competition-style math. Prefer a concise final expression extracted from the final boxed answer.",
    ),
    "nlgraph": DatasetProfile(
        name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        root_templates=("solve_verify", "route_solve", "critique_revise"),
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", output_style="json", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", output_style="json", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", output_style="json", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_GRAPH_ROLE_PREFS,
        required_agent_ids=("coder", "verifier", "reasoner"),
        search_overrides=SearchOverrides(
            search_iterations=12,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=640),
        proxy_prompt_penalty_scale=0.7,
        reward_prompt_penalty_scale=0.7,
        notes="Graph reasoning. Return canonical JSON matching the requested graph subtask.",
    ),
    "normad": DatasetProfile(
        name="normad",
        task_type="boolean",
        answer_format="yes_no",
        root_templates=_LIGHT_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="direct", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_STRUCTURED_ROLE_PREFS,
        required_agent_ids=("verifier",),
        search_overrides=SearchOverrides(search_iterations=5),
        notes="Social norm judgement. Output exactly yes or no.",
    ),
    "knowledge_crosswords": DatasetProfile(
        name="knowledge_crosswords",
        task_type="structured_list",
        answer_format="json_list",
        root_templates=("route_solve", "solve_verify", "critique_revise"),
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", output_style="json", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", output_style="json", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_STRUCTURED_ROLE_PREFS,
        required_agent_ids=("researcher", "verifier", "summarizer"),
        search_overrides=SearchOverrides(
            search_iterations=12,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.7,
        reward_prompt_penalty_scale=0.7,
        notes="Fill multiple blanks and return a JSON list following blank order.",
    ),
    "gaia": DatasetProfile(
        name="gaia",
        task_type="generic",
        answer_format="question_defined",
        root_templates=_OPEN_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_STRUCTURED_ROLE_PREFS,
        required_agent_ids=("researcher", "verifier", "planner"),
        search_overrides=SearchOverrides(
            search_iterations=12,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768),
        proxy_prompt_penalty_scale=0.7,
        reward_prompt_penalty_scale=0.7,
        notes="Open-domain tool-leaning QA. Respect the explicit answer format in the prompt.",
    ),
    "qasper": DatasetProfile(
        name="qasper",
        task_type="generic",
        answer_format="short_span",
        root_templates=_OPEN_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="direct", upstream_usage="quote_then_reason", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_STRUCTURED_ROLE_PREFS,
        required_agent_ids=("researcher", "verifier"),
        search_overrides=SearchOverrides(
            search_iterations=12,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.7,
        reward_prompt_penalty_scale=0.7,
        notes="Short answer over paper metadata/context. Prefer concise span extraction.",
    ),
    "humaneval": DatasetProfile(
        name="humaneval",
        task_type="code_generation",
        answer_format="python_code",
        root_templates=_CODE_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_CODE_ROLE_PREFS,
        required_agent_ids=("coder", "verifier", "planner"),
        search_overrides=SearchOverrides(
            search_iterations=14,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        proxy_prompt_penalty_scale=0.65,
        reward_prompt_penalty_scale=0.65,
        reward_size_penalty_scale=0.85,
        notes="Python code generation benchmark. Return only executable Python code.",
    ),
    "mbpp": DatasetProfile(
        name="mbpp",
        task_type="code_generation",
        answer_format="python_code",
        root_templates=_CODE_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides=_merge_dicts(
            _role_slots(_SOLVER_ROLES, reasoning_mode="stepwise", finalization="answer_only"),
            _role_slots(_VERIFY_ROLES, reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            _role_slots(_SINK_ROLES + ("router",), reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        ),
        role_agent_preferences=_CODE_ROLE_PREFS,
        required_agent_ids=("coder", "verifier", "planner"),
        search_overrides=SearchOverrides(
            search_iterations=12,
            candidate_core_k=5,
            candidate_explore_k=3,
            candidate_max_k=8,
            tier1_top_fraction=0.4,
            tier2_top_fraction=0.5,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        proxy_prompt_penalty_scale=0.7,
        reward_prompt_penalty_scale=0.7,
        reward_size_penalty_scale=0.9,
        notes="Python programming tasks with visible unit tests. Return only executable Python code.",
    ),
}


DEFAULT_PROFILE = DatasetProfile(
    name="default",
    task_type="generic",
    answer_format="short_text",
    root_templates=("direct", "solve_verify", "critique_revise", "parallel_vote"),
    allowed_templates=_all_templates(),
    role_prompt_overrides={},
    notes="Fallback profile when no dataset-specific profile exists.",
)


def resolve_dataset_profile(dataset_name: Optional[str]) -> DatasetProfile:
    if not dataset_name:
        return DEFAULT_PROFILE
    return DATASET_PROFILES.get(dataset_name, DEFAULT_PROFILE)


def list_supported_datasets() -> Tuple[str, ...]:
    return tuple(sorted(DATASET_PROFILES.keys()))


def profile_summary(dataset_names: Iterable[str]) -> Dict[str, Dict[str, object]]:
    summary: Dict[str, Dict[str, object]] = {}
    for dataset_name in dataset_names:
        profile = resolve_dataset_profile(dataset_name)
        summary[dataset_name] = {
            "task_type": profile.task_type,
            "answer_format": profile.answer_format,
            "root_templates": list(profile.root_templates),
            "notes": profile.notes,
        }
    return summary
