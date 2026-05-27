from __future__ import annotations

from typing import Dict, List

from .types import ArchitectureState, CompiledArchitecture
from .operators import TEMPLATE_SPECS


def compile_architecture(state: ArchitectureState) -> CompiledArchitecture:
    spec = TEMPLATE_SPECS[state.template]
    nodes: List[str] = []
    for role in spec.required_roles:
        agent_id = state.role_to_agent.get(role, "")
        if agent_id and agent_id not in nodes:
            nodes.append(agent_id)
    edge_set = set()
    edges: List[tuple[str, str]] = []
    for src, dst in spec.edges:
        src_agent = state.role_to_agent.get(src, "")
        dst_agent = state.role_to_agent.get(dst, "")
        if not src_agent or not dst_agent or src_agent == dst_agent:
            continue
        edge = (src_agent, dst_agent)
        if edge not in edge_set:
            edge_set.add(edge)
            edges.append(edge)
    sinks = [state.role_to_agent[role] for role in spec.sinks if state.role_to_agent.get(role)]
    execution_roles = [role for role in spec.execution_roles if state.role_to_agent.get(role)]
    return CompiledArchitecture(
        state=state,
        nodes=nodes,
        edges=edges,
        sinks=sinks,
        execution_roles=execution_roles,
    )


def architecture_signature(state: ArchitectureState) -> str:
    compiled = compile_architecture(state)
    return compiled.signature()
