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
class StructurePrior:
    selector_quality_weight: float = 0.65
    selector_structure_weight: float = 0.35
    selector_diversity_scale: float = 1.0
    proxy_task_weight: float = 0.67
    proxy_structure_weight: float = 0.33
    proxy_shape_weight: float = 0.45
    proxy_redundancy_weight: float = 0.35
    proxy_affordability_weight: float = 0.20
    topology_quality_weight: float = 0.40
    diversity_quality_weight: float = 0.35
    deployability_weight: float = 0.25
    execution_probe_weight: float = 0.20
    coverage_mix: float = 0.55
    complementarity_mix: float = 0.60
    target_task_nodes: int = 5
    target_task_edges: int = 10
    preferred_sink_count: int = 1
    max_root_count: int = 2


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
    structure_prior: StructurePrior = field(default_factory=StructurePrior)
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
        search_overrides=SearchOverrides(search_iterations=8, final_top_k=4),
        proxy_prompt_penalty_scale=0.88,
        reward_prompt_penalty_scale=0.88,
        reward_size_penalty_scale=0.92,
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
        search_overrides=SearchOverrides(search_iterations=8, final_top_k=4),
        proxy_prompt_penalty_scale=0.86,
        reward_prompt_penalty_scale=0.86,
        reward_size_penalty_scale=0.90,
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
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        proxy_prompt_penalty_scale=0.82,
        reward_prompt_penalty_scale=0.82,
        reward_size_penalty_scale=0.88,
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
            search_iterations=8,
            candidate_core_k=4,
            candidate_explore_k=2,
            candidate_max_k=6,
            tier1_top_fraction=0.35,
            tier2_top_fraction=0.45,
            final_top_k=4,
            max_prompt_edits_per_state=1,
        ),
        runtime_overrides=RuntimeOverrides(tier2_max_tokens=576, tier2_judge_max_tokens=160),
        structure_prior=StructurePrior(
            selector_quality_weight=0.72,
            selector_structure_weight=0.28,
            selector_diversity_scale=0.72,
            proxy_task_weight=0.74,
            proxy_structure_weight=0.26,
            proxy_shape_weight=0.50,
            proxy_redundancy_weight=0.12,
            proxy_affordability_weight=0.38,
            topology_quality_weight=0.38,
            diversity_quality_weight=0.16,
            deployability_weight=0.46,
            execution_probe_weight=0.10,
            coverage_mix=0.58,
            complementarity_mix=0.40,
            target_task_nodes=4,
            target_task_edges=6,
            preferred_sink_count=1,
            max_root_count=1,
        ),
        proxy_prompt_penalty_scale=0.88,
        reward_prompt_penalty_scale=0.88,
        reward_size_penalty_scale=0.94,
        notes="Long-form professional MCQ. Prefer light verify-and-decide structures over broad redundant graphs.",
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
        search_overrides=SearchOverrides(search_iterations=8, final_top_k=4),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        structure_prior=StructurePrior(
            selector_quality_weight=0.72,
            selector_structure_weight=0.28,
            selector_diversity_scale=0.75,
            proxy_task_weight=0.72,
            proxy_structure_weight=0.28,
            proxy_shape_weight=0.52,
            proxy_redundancy_weight=0.18,
            proxy_affordability_weight=0.30,
            topology_quality_weight=0.42,
            diversity_quality_weight=0.18,
            deployability_weight=0.40,
            execution_probe_weight=0.12,
            coverage_mix=0.60,
            complementarity_mix=0.45,
            target_task_nodes=4,
            target_task_edges=6,
            preferred_sink_count=1,
            max_root_count=1,
        ),
        proxy_prompt_penalty_scale=0.75,
        reward_prompt_penalty_scale=0.75,
        reward_size_penalty_scale=0.85,
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
        search_overrides=SearchOverrides(search_iterations=6, final_top_k=4),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.78,
        reward_prompt_penalty_scale=0.78,
        reward_size_penalty_scale=0.86,
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
            final_top_k=6,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=3, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        structure_prior=StructurePrior(
            selector_quality_weight=0.62,
            selector_structure_weight=0.38,
            selector_diversity_scale=1.10,
            proxy_task_weight=0.64,
            proxy_structure_weight=0.36,
            proxy_shape_weight=0.48,
            proxy_redundancy_weight=0.27,
            proxy_affordability_weight=0.25,
            topology_quality_weight=0.40,
            diversity_quality_weight=0.32,
            deployability_weight=0.28,
            execution_probe_weight=0.16,
            coverage_mix=0.50,
            complementarity_mix=0.55,
            target_task_nodes=5,
            target_task_edges=8,
            preferred_sink_count=1,
            max_root_count=2,
        ),
        proxy_prompt_penalty_scale=0.52,
        reward_prompt_penalty_scale=0.52,
        reward_size_penalty_scale=0.78,
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
            search_iterations=10,
            candidate_core_k=4,
            candidate_explore_k=2,
            candidate_max_k=6,
            tier1_top_fraction=0.35,
            tier2_top_fraction=0.45,
            final_top_k=5,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=512, tier2_judge_max_tokens=160),
        structure_prior=StructurePrior(
            selector_quality_weight=0.66,
            selector_structure_weight=0.34,
            selector_diversity_scale=0.85,
            proxy_task_weight=0.68,
            proxy_structure_weight=0.32,
            proxy_shape_weight=0.42,
            proxy_redundancy_weight=0.18,
            proxy_affordability_weight=0.40,
            topology_quality_weight=0.34,
            diversity_quality_weight=0.20,
            deployability_weight=0.46,
            execution_probe_weight=0.22,
            coverage_mix=0.46,
            complementarity_mix=0.48,
            target_task_nodes=4,
            target_task_edges=7,
            preferred_sink_count=1,
            max_root_count=1,
        ),
        proxy_prompt_penalty_scale=0.60,
        reward_prompt_penalty_scale=0.60,
        reward_size_penalty_scale=0.88,
        notes="Graph reasoning with reliable task checker. Favor compact coder-reasoner-verifier structures and exact JSON outputs.",
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
        search_overrides=SearchOverrides(search_iterations=5, final_top_k=3),
        proxy_prompt_penalty_scale=0.92,
        reward_prompt_penalty_scale=0.92,
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
            search_iterations=10,
            candidate_core_k=4,
            candidate_explore_k=2,
            candidate_max_k=6,
            tier1_top_fraction=0.35,
            tier2_top_fraction=0.45,
            final_top_k=5,
            max_prompt_edits_per_state=1,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=576, tier2_judge_max_tokens=160),
        structure_prior=StructurePrior(
            selector_quality_weight=0.63,
            selector_structure_weight=0.37,
            selector_diversity_scale=0.95,
            proxy_task_weight=0.66,
            proxy_structure_weight=0.34,
            proxy_shape_weight=0.40,
            proxy_redundancy_weight=0.16,
            proxy_affordability_weight=0.44,
            topology_quality_weight=0.34,
            diversity_quality_weight=0.24,
            deployability_weight=0.42,
            execution_probe_weight=0.22,
            coverage_mix=0.48,
            complementarity_mix=0.52,
            target_task_nodes=4,
            target_task_edges=7,
            preferred_sink_count=1,
            max_root_count=1,
        ),
        proxy_prompt_penalty_scale=0.62,
        reward_prompt_penalty_scale=0.62,
        reward_size_penalty_scale=0.90,
        notes="Structured blank filling. Favor compact researcher-verifier-summarizer workflows that preserve JSON list order.",
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
            final_top_k=6,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768),
        proxy_prompt_penalty_scale=0.62,
        reward_prompt_penalty_scale=0.62,
        reward_size_penalty_scale=0.84,
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
            final_top_k=6,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2),
        proxy_prompt_penalty_scale=0.60,
        reward_prompt_penalty_scale=0.60,
        reward_size_penalty_scale=0.84,
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
            final_top_k=6,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        structure_prior=StructurePrior(
            selector_quality_weight=0.58,
            selector_structure_weight=0.42,
            selector_diversity_scale=1.15,
            proxy_task_weight=0.60,
            proxy_structure_weight=0.40,
            proxy_shape_weight=0.38,
            proxy_redundancy_weight=0.42,
            proxy_affordability_weight=0.20,
            topology_quality_weight=0.34,
            diversity_quality_weight=0.42,
            deployability_weight=0.24,
            execution_probe_weight=0.18,
            coverage_mix=0.50,
            complementarity_mix=0.65,
            target_task_nodes=7,
            target_task_edges=13,
            preferred_sink_count=1,
            max_root_count=2,
        ),
        proxy_prompt_penalty_scale=0.50,
        reward_prompt_penalty_scale=0.50,
        reward_size_penalty_scale=0.75,
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
            final_top_k=6,
            max_prompt_edits_per_state=2,
        ),
        runtime_overrides=RuntimeOverrides(tier2_repeats=2, tier2_max_tokens=768, tier2_judge_max_tokens=192),
        structure_prior=StructurePrior(
            selector_quality_weight=0.60,
            selector_structure_weight=0.40,
            selector_diversity_scale=1.08,
            proxy_task_weight=0.62,
            proxy_structure_weight=0.38,
            proxy_shape_weight=0.40,
            proxy_redundancy_weight=0.40,
            proxy_affordability_weight=0.20,
            topology_quality_weight=0.35,
            diversity_quality_weight=0.40,
            deployability_weight=0.25,
            execution_probe_weight=0.17,
            coverage_mix=0.52,
            complementarity_mix=0.62,
            target_task_nodes=6,
            target_task_edges=12,
            preferred_sink_count=1,
            max_root_count=2,
        ),
        proxy_prompt_penalty_scale=0.55,
        reward_prompt_penalty_scale=0.55,
        reward_size_penalty_scale=0.80,
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
            "structure_prior": {
                "target_task_nodes": profile.structure_prior.target_task_nodes,
                "target_task_edges": profile.structure_prior.target_task_edges,
                "proxy_task_weight": profile.structure_prior.proxy_task_weight,
                "proxy_structure_weight": profile.structure_prior.proxy_structure_weight,
                "selector_quality_weight": profile.structure_prior.selector_quality_weight,
                "selector_structure_weight": profile.structure_prior.selector_structure_weight,
            },
            "notes": profile.notes,
        }
    return summary
