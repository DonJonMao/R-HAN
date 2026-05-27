from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .types import CompiledArchitecture, WorkflowTemplate


@dataclass(frozen=True)
class TemplateSpec:
    required_roles: Tuple[str, ...]
    edges: Tuple[Tuple[str, str], ...]
    sinks: Tuple[str, ...]
    execution_roles: Tuple[str, ...]


TEMPLATE_SPECS: Dict[WorkflowTemplate, TemplateSpec] = {
    WorkflowTemplate.DIRECT: TemplateSpec(
        required_roles=("solver",),
        edges=(),
        sinks=("solver",),
        execution_roles=("solver",),
    ),
    WorkflowTemplate.SOLVE_VERIFY: TemplateSpec(
        required_roles=("solver", "verifier"),
        edges=(("solver", "verifier"),),
        sinks=("verifier",),
        execution_roles=("solver", "verifier"),
    ),
    WorkflowTemplate.PARALLEL_VOTE: TemplateSpec(
        required_roles=("solver_a", "solver_b", "aggregator"),
        edges=(("solver_a", "aggregator"), ("solver_b", "aggregator")),
        sinks=("aggregator",),
        execution_roles=("solver_a", "solver_b", "aggregator"),
    ),
    WorkflowTemplate.CRITIQUE_REVISE: TemplateSpec(
        required_roles=("generator", "critic", "reviser"),
        edges=(("generator", "critic"), ("generator", "reviser"), ("critic", "reviser")),
        sinks=("reviser",),
        execution_roles=("generator", "critic", "reviser"),
    ),
    WorkflowTemplate.DEBATE_JUDGE: TemplateSpec(
        required_roles=("debater_a", "debater_b", "judge"),
        edges=(("debater_a", "judge"), ("debater_b", "judge")),
        sinks=("judge",),
        execution_roles=("debater_a", "debater_b", "judge"),
    ),
    WorkflowTemplate.ROUTE_SOLVE: TemplateSpec(
        required_roles=("router", "solver", "aggregator"),
        edges=(("router", "solver"), ("solver", "aggregator")),
        sinks=("aggregator",),
        execution_roles=("router", "solver", "aggregator"),
    ),
}


def roles_for_template(template: WorkflowTemplate) -> List[str]:
    return list(TEMPLATE_SPECS[template].required_roles)


def sink_roles_for_template(template: WorkflowTemplate) -> List[str]:
    return list(TEMPLATE_SPECS[template].sinks)
