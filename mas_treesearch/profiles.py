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
class DatasetProfile:
    name: str
    task_type: str
    answer_format: str
    root_templates: Tuple[str, ...]
    allowed_templates: Tuple[str, ...]
    role_prompt_overrides: Dict[str, PromptSlots] = field(default_factory=dict)
    notes: str = ""

    def prompt_for_role(self, role: str, base: PromptSlots) -> PromptSlots:
        return self.role_prompt_overrides.get(role, base)


def _all_templates() -> Tuple[str, ...]:
    return tuple(template.value for template in WorkflowTemplate)


_MCQ_TEMPLATES = ("solve_verify", "route_solve", "parallel_vote", "critique_revise")
_REASONING_TEMPLATES = ("solve_verify", "critique_revise", "parallel_vote", "route_solve")
_LIGHT_TEMPLATES = ("direct", "solve_verify", "critique_revise", "route_solve")
_OPEN_TEMPLATES = ("route_solve", "solve_verify", "critique_revise", "parallel_vote")


DATASET_PROFILES: Dict[str, DatasetProfile] = {
    "mmlu": DatasetProfile(
        name="mmlu",
        task_type="mcq",
        answer_format="option",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="direct", verification_mode="off", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
            "reviser": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="High-risk MCQ. Final answer must be an option index only.",
    ),
    "popqa": DatasetProfile(
        name="popqa",
        task_type="mcq",
        answer_format="option_or_option_confidence",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="direct", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="MCQ with possible abstain option and sometimes confidence line.",
    ),
    "cqa": DatasetProfile(
        name="cqa",
        task_type="mcq",
        answer_format="option",
        root_templates=_MCQ_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="direct", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Music-domain MCQ with fixed answer format.",
    ),
    "mmlu_pro": DatasetProfile(
        name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
        root_templates=("route_solve", "solve_verify", "critique_revise", "parallel_vote"),
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
            "router": _slots(reasoning_mode="direct", finalization="answer_only"),
        },
        notes="Long-form professional MCQ with options stored in metadata.",
    ),
    "gsm8k": DatasetProfile(
        name="gsm8k",
        task_type="numeric",
        answer_format="number",
        root_templates=_REASONING_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "generator": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "reviser": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Math word problem. Final answer should be the final number only.",
    ),
    "nlgraph": DatasetProfile(
        name="nlgraph",
        task_type="graph_reasoning",
        answer_format="short_text",
        root_templates=_REASONING_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Graph reasoning. Usually requires a yes/no plus path or ordering.",
    ),
    "normad": DatasetProfile(
        name="normad",
        task_type="boolean",
        answer_format="yes_no",
        root_templates=_LIGHT_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="direct", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Social norm judgement. Output exactly yes or no.",
    ),
    "knowledge_crosswords": DatasetProfile(
        name="knowledge_crosswords",
        task_type="structured_list",
        answer_format="json_list",
        root_templates=("route_solve", "solve_verify", "parallel_vote", "critique_revise"),
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "solver": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(
                reasoning_mode="direct",
                output_style="json",
                verification_mode="light",
                finalization="answer_only",
            ),
            "reviser": _slots(
                reasoning_mode="direct",
                output_style="json",
                verification_mode="light",
                finalization="answer_only",
            ),
        },
        notes="Fill multiple blanks and return a JSON list following blank order.",
    ),
    "gaia": DatasetProfile(
        name="gaia",
        task_type="generic",
        answer_format="question_defined",
        root_templates=_OPEN_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "router": _slots(reasoning_mode="direct", finalization="answer_only"),
            "solver": _slots(reasoning_mode="stepwise", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Open-domain tool-leaning QA. Respect the explicit answer format in the prompt.",
    ),
    "qasper": DatasetProfile(
        name="qasper",
        task_type="generic",
        answer_format="short_span",
        root_templates=_OPEN_TEMPLATES,
        allowed_templates=_all_templates(),
        role_prompt_overrides={
            "router": _slots(reasoning_mode="direct", finalization="answer_only"),
            "solver": _slots(reasoning_mode="direct", upstream_usage="quote_then_reason", finalization="answer_only"),
            "verifier": _slots(reasoning_mode="direct", verification_mode="strict", finalization="answer_only"),
            "aggregator": _slots(reasoning_mode="direct", verification_mode="light", finalization="answer_only"),
        },
        notes="Short answer over paper metadata/context. Prefer concise span extraction.",
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
