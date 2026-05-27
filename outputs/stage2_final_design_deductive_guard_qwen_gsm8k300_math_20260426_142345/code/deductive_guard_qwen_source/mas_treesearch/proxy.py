from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .agents import AgentPool
from .gating import cosine
from .profiles import DEFAULT_PROFILE, DatasetProfile
from .types import ArchitectureState, CompiledArchitecture, PromptSlots, Vector, WorkflowTemplate


@dataclass
class ProxyScore:
    score: float
    uncertainty: float
    features: Dict[str, float]


class StaticProxyScorer:
    """Cheap structural scorer used for tier-0 filtering."""

    def __init__(self, agent_pool: AgentPool, agent_vectors: Dict[str, Vector]):
        self.agent_pool = agent_pool
        self._by_id = agent_pool.by_id()
        self.agent_vectors = agent_vectors

    def _alignment(self, question_vector: Vector, agent_ids: Sequence[str]) -> float:
        if not agent_ids:
            return 0.0
        vals = [cosine(question_vector, self.agent_vectors[aid]) for aid in agent_ids if aid in self.agent_vectors]
        return sum(vals) / max(1, len(vals))

    def _diversity(self, agent_ids: Sequence[str]) -> float:
        if len(agent_ids) <= 1:
            return 0.0
        pairs = []
        for i, aid in enumerate(agent_ids):
            for bid in agent_ids[i + 1 :]:
                pairs.append(1.0 - cosine(self.agent_vectors[aid], self.agent_vectors[bid]))
        return sum(pairs) / max(1, len(pairs))

    def _verification_bonus(self, template: WorkflowTemplate, state: ArchitectureState) -> float:
        bonus = 0.0
        if template in {WorkflowTemplate.SOLVE_VERIFY, WorkflowTemplate.CRITIQUE_REVISE}:
            for role in ("verifier", "critic", "judge"):
                aid = state.role_to_agent.get(role)
                if aid and "verification" in self._by_id[aid].capabilities:
                    bonus += 0.12
        return bonus

    @staticmethod
    def _template_prior(template: WorkflowTemplate, profile: DatasetProfile) -> float:
        task_type = profile.task_type
        common = {
            WorkflowTemplate.DIRECT: 0.48,
            WorkflowTemplate.SOLVE_VERIFY: 0.58,
            WorkflowTemplate.PARALLEL_VOTE: 0.55,
            WorkflowTemplate.CRITIQUE_REVISE: 0.57,
            WorkflowTemplate.DEBATE_JUDGE: 0.53,
            WorkflowTemplate.ROUTE_SOLVE: 0.50,
        }
        if profile.name == "mmlu_pro":
            return {
                WorkflowTemplate.DIRECT: 0.28,
                WorkflowTemplate.SOLVE_VERIFY: 0.72,
                WorkflowTemplate.PARALLEL_VOTE: 0.60,
                WorkflowTemplate.CRITIQUE_REVISE: 0.62,
                WorkflowTemplate.DEBATE_JUDGE: 0.64,
                WorkflowTemplate.ROUTE_SOLVE: 0.52,
            }.get(template, common.get(template, 0.50))
        if profile.name == "nlgraph":
            return {
                WorkflowTemplate.DIRECT: 0.22,
                WorkflowTemplate.SOLVE_VERIFY: 0.70,
                WorkflowTemplate.PARALLEL_VOTE: 0.40,
                WorkflowTemplate.CRITIQUE_REVISE: 0.69,
                WorkflowTemplate.DEBATE_JUDGE: 0.46,
                WorkflowTemplate.ROUTE_SOLVE: 0.66,
            }.get(template, common.get(template, 0.50))
        if profile.name == "knowledge_crosswords":
            return {
                WorkflowTemplate.DIRECT: 0.24,
                WorkflowTemplate.SOLVE_VERIFY: 0.68,
                WorkflowTemplate.PARALLEL_VOTE: 0.38,
                WorkflowTemplate.CRITIQUE_REVISE: 0.66,
                WorkflowTemplate.DEBATE_JUDGE: 0.42,
                WorkflowTemplate.ROUTE_SOLVE: 0.71,
            }.get(template, common.get(template, 0.50))
        if task_type == "code_generation":
            return {
                WorkflowTemplate.DIRECT: 0.30,
                WorkflowTemplate.SOLVE_VERIFY: 0.68,
                WorkflowTemplate.PARALLEL_VOTE: 0.50,
                WorkflowTemplate.CRITIQUE_REVISE: 0.72,
                WorkflowTemplate.DEBATE_JUDGE: 0.52,
                WorkflowTemplate.ROUTE_SOLVE: 0.74,
            }.get(template, 0.50)
        if task_type == "mcq":
            return {
                WorkflowTemplate.DIRECT: 0.44,
                WorkflowTemplate.SOLVE_VERIFY: 0.66,
                WorkflowTemplate.PARALLEL_VOTE: 0.58,
                WorkflowTemplate.CRITIQUE_REVISE: 0.57,
                WorkflowTemplate.DEBATE_JUDGE: 0.60,
                WorkflowTemplate.ROUTE_SOLVE: 0.60,
            }.get(template, common.get(template, 0.50))
        if task_type in {"numeric", "math_expression"}:
            return {
                WorkflowTemplate.DIRECT: 0.40,
                WorkflowTemplate.SOLVE_VERIFY: 0.65,
                WorkflowTemplate.PARALLEL_VOTE: 0.52,
                WorkflowTemplate.CRITIQUE_REVISE: 0.62,
                WorkflowTemplate.DEBATE_JUDGE: 0.50,
                WorkflowTemplate.ROUTE_SOLVE: 0.61,
            }.get(template, common.get(template, 0.50))
        if task_type in {"graph_reasoning", "structured_list"}:
            return {
                WorkflowTemplate.DIRECT: 0.34,
                WorkflowTemplate.SOLVE_VERIFY: 0.64,
                WorkflowTemplate.PARALLEL_VOTE: 0.50,
                WorkflowTemplate.CRITIQUE_REVISE: 0.64,
                WorkflowTemplate.DEBATE_JUDGE: 0.48,
                WorkflowTemplate.ROUTE_SOLVE: 0.67,
            }.get(template, common.get(template, 0.50))
        if task_type == "boolean":
            return {
                WorkflowTemplate.DIRECT: 0.55,
                WorkflowTemplate.SOLVE_VERIFY: 0.63,
                WorkflowTemplate.PARALLEL_VOTE: 0.50,
                WorkflowTemplate.CRITIQUE_REVISE: 0.56,
                WorkflowTemplate.DEBATE_JUDGE: 0.55,
                WorkflowTemplate.ROUTE_SOLVE: 0.54,
            }.get(template, common.get(template, 0.50))
        return common.get(template, 0.50)

    def _task_structure_bonus(
        self,
        compiled: CompiledArchitecture,
        profile: DatasetProfile,
        metadata: Optional[dict],
    ) -> float:
        state = compiled.state
        active = set(state.active_agents())
        roles = set(state.role_to_agent)
        task_type = profile.task_type
        bonus = 0.0

        if profile.name == "mmlu_pro":
            if "verifier" in active:
                bonus += 0.08
            if "reasoner" in active:
                bonus += 0.05
            if "planner" in active:
                bonus += 0.03
            if "judge" in roles or "aggregator" in roles:
                bonus += 0.05
            if {"verifier", "reasoner"} <= active:
                bonus += 0.04
            if state.template == WorkflowTemplate.DIRECT:
                bonus -= 0.12
            if len(active) >= 6:
                bonus -= 0.04
            return bonus

        if profile.name == "nlgraph":
            if "coder" in active:
                bonus += 0.08
            if "reasoner" in active:
                bonus += 0.07
            if "verifier" in active:
                bonus += 0.07
            if {"coder", "verifier"} <= active:
                bonus += 0.04
            if "router" in roles:
                bonus += 0.03
            task = str((metadata or {}).get("task") or "").strip()
            if task in {"hamilton", "topology", "shortest_path"} and {"coder", "reasoner"} & active:
                bonus += 0.03
            if state.template in {WorkflowTemplate.DIRECT, WorkflowTemplate.PARALLEL_VOTE}:
                bonus -= 0.06
            if len(active) >= 6:
                bonus -= 0.03
            return bonus

        if profile.name == "knowledge_crosswords":
            if "researcher" in active:
                bonus += 0.08
            if "summarizer" in active:
                bonus += 0.07
            if "verifier" in active:
                bonus += 0.06
            if {"researcher", "summarizer"} <= active:
                bonus += 0.05
            blank_count = len((metadata or {}).get("blanks") or [])
            if blank_count >= 4 and "router" in roles:
                bonus += 0.03
            if state.template == WorkflowTemplate.PARALLEL_VOTE:
                bonus -= 0.06
            if len(active) >= 6:
                bonus -= 0.03
            return bonus

        if task_type == "code_generation":
            if "coder" in active:
                bonus += 0.08
            if "verifier" in active:
                bonus += 0.07
            if "planner" in active:
                bonus += 0.05
            if {"coder", "verifier"} <= active:
                bonus += 0.05
            if {"router", "verifier"} <= roles:
                bonus += 0.03
            if isinstance(metadata, dict) and (
                isinstance(metadata.get("test"), str) or isinstance(metadata.get("test_list"), list)
            ):
                if "verifier" in roles or "critic" in roles:
                    bonus += 0.04
            if state.template == WorkflowTemplate.DIRECT:
                bonus -= 0.10
        elif task_type == "mcq":
            if "verifier" in active:
                bonus += 0.06
            if "judge" in roles or "aggregator" in roles:
                bonus += 0.03
        elif task_type in {"numeric", "math_expression"}:
            if "math" in active:
                bonus += 0.08
            if "verifier" in active:
                bonus += 0.05
            if state.template == WorkflowTemplate.DIRECT:
                bonus -= 0.06
        elif task_type == "graph_reasoning":
            if "coder" in active:
                bonus += 0.06
            if "reasoner" in active:
                bonus += 0.05
            if "verifier" in active:
                bonus += 0.05
        elif task_type == "structured_list":
            if "researcher" in active:
                bonus += 0.05
            if "summarizer" in active:
                bonus += 0.05
            if "verifier" in active:
                bonus += 0.04
        elif task_type == "boolean":
            if "verifier" in active:
                bonus += 0.05
        else:
            if "researcher" in active:
                bonus += 0.03
            if "verifier" in active:
                bonus += 0.03

        return bonus

    def _prompt_fit(self, state: ArchitectureState, profile: DatasetProfile) -> float:
        score = 0.0
        task_type = profile.task_type
        for role, slots in state.role_to_prompt.items():
            if slots.finalization == "answer_only":
                score += 0.01
            if role in {"verifier", "critic", "judge"} and slots.verification_mode in {"light", "strict"}:
                score += 0.02
            if profile.name == "mmlu_pro":
                if slots.output_style != "json":
                    score += 0.012
                if role in {"solver", "solver_a", "solver_b", "generator"} and slots.reasoning_mode in {
                    "direct",
                    "stepwise",
                }:
                    score += 0.012
                if role in {"verifier", "judge"} and slots.verification_mode == "strict":
                    score += 0.018
                continue
            if profile.name == "nlgraph":
                if slots.output_style == "json":
                    score += 0.03
                if role in {"solver", "solver_a", "solver_b", "generator"} and slots.reasoning_mode == "stepwise":
                    score += 0.02
                if role in {"verifier", "critic", "judge"} and slots.verification_mode == "strict":
                    score += 0.02
                continue
            if profile.name == "knowledge_crosswords":
                if slots.output_style == "json":
                    score += 0.028
                if role in {"solver", "solver_a", "solver_b", "generator"} and slots.reasoning_mode == "stepwise":
                    score += 0.018
                if role in {"verifier", "critic", "judge"} and slots.verification_mode == "strict":
                    score += 0.02
                if role in {"aggregator", "reviser"} and slots.finalization == "answer_only":
                    score += 0.012
                continue
            if task_type == "code_generation":
                if slots.output_style == "raw":
                    score += 0.012
                if role in {"solver", "solver_a", "solver_b", "generator", "reviser"} and slots.reasoning_mode in {
                    "stepwise",
                    "critique_then_answer",
                }:
                    score += 0.018
            elif task_type in {"graph_reasoning", "structured_list"}:
                if slots.output_style == "json":
                    score += 0.02
            elif task_type == "mcq":
                if slots.verification_mode == "strict" and role in {"verifier", "judge"}:
                    score += 0.02
                if slots.output_style != "json":
                    score += 0.01
            elif task_type in {"numeric", "math_expression"}:
                if role in {"solver", "solver_a", "solver_b", "generator"} and slots.reasoning_mode == "stepwise":
                    score += 0.015
            elif task_type == "boolean":
                if slots.reasoning_mode == "direct":
                    score += 0.01
            elif profile.answer_format == "short_span" and slots.upstream_usage == "quote_then_reason":
                score += 0.015
        return score

    @staticmethod
    def _size_penalty(active_count: int, profile: DatasetProfile) -> float:
        extra = max(0, active_count - 4)
        base = 0.08
        if profile.name == "mmlu_pro":
            base = 0.10
        elif profile.name == "nlgraph":
            base = 0.07
        elif profile.name == "knowledge_crosswords":
            base = 0.08
        elif profile.task_type == "code_generation":
            base = 0.05
        elif profile.task_type in {"graph_reasoning", "structured_list", "math_expression"}:
            base = 0.06
        elif profile.task_type == "boolean":
            base = 0.09
        return max(0.0, base * extra)

    @staticmethod
    def _critical_role_groups(profile: DatasetProfile) -> list[set[str]]:
        if profile.task_type == "code_generation":
            return [
                {"solver", "generator", "solver_a", "solver_b"},
                {"critic", "verifier", "judge"},
                {"reviser", "aggregator"},
                {"router"},
            ]
        if profile.task_type in {"numeric", "math_expression"}:
            return [
                {"solver", "generator", "solver_a", "solver_b"},
                {"critic", "verifier", "judge"},
                {"reviser", "aggregator"},
            ]
        return [
            {"solver", "generator", "solver_a", "solver_b"},
            {"critic", "verifier", "judge"},
            {"reviser", "aggregator"},
        ]

    def _topology_shape_quality(self, compiled: CompiledArchitecture, profile: DatasetProfile) -> float:
        state = compiled.state
        roles = set(state.role_to_agent)
        groups = self._critical_role_groups(profile)
        group_hits = sum(1 for group in groups if roles & group)
        role_coverage = group_hits / max(1, len(groups))
        sink_clarity = 1.0 if len(compiled.sinks) == 1 else max(0.3, 1.0 - 0.25 * max(0, len(compiled.sinks) - 1))
        edge_budget = max(2, len(compiled.execution_roles) + 1)
        edge_score = max(0.0, 1.0 - max(0, len(compiled.edges) - edge_budget) / max(3, edge_budget))
        return max(0.0, min(1.0, 0.45 * role_coverage + 0.30 * sink_clarity + 0.25 * edge_score))

    def _redundancy_readiness(self, compiled: CompiledArchitecture, profile: DatasetProfile) -> float:
        state = compiled.state
        roles = set(state.role_to_agent)
        active = set(state.active_agents())
        score = 0.0
        if {"critic", "verifier", "judge"} & roles:
            score += 0.30
        if {"reviser", "aggregator"} & roles:
            score += 0.25
        if "router" in roles:
            score += 0.12
        if profile.name == "mmlu_pro":
            if "reasoner" in active:
                score += 0.12
            if "summarizer" in active:
                score += 0.06
        elif profile.name == "nlgraph":
            if "coder" in active:
                score += 0.16
            if "reasoner" in active:
                score += 0.10
        elif profile.name == "knowledge_crosswords":
            if "researcher" in active:
                score += 0.16
            if "summarizer" in active:
                score += 0.10
        elif profile.task_type == "code_generation":
            if "coder" in active:
                score += 0.18
            if "planner" in active:
                score += 0.10
        elif profile.task_type in {"numeric", "math_expression"}:
            if "math" in active:
                score += 0.18
        return max(0.0, min(1.0, score))

    def _runtime_affordability(self, compiled: CompiledArchitecture, profile: DatasetProfile) -> float:
        prior = profile.structure_prior
        active_count = len(compiled.state.active_agents())
        edge_count = len(compiled.edges)
        target_nodes = max(2, prior.target_task_nodes)
        target_edges = max(2, prior.target_task_edges)
        node_score = max(0.0, 1.0 - max(0, active_count - target_nodes) / max(2, target_nodes))
        edge_score = max(0.0, 1.0 - max(0, edge_count - target_edges) / max(4, target_edges))
        return max(0.0, min(1.0, 0.55 * node_score + 0.45 * edge_score))

    def score(
        self,
        compiled: CompiledArchitecture,
        question_vector: Vector,
        profile: DatasetProfile = DEFAULT_PROFILE,
        metadata: Optional[dict] = None,
    ) -> ProxyScore:
        state = compiled.state
        active = state.active_agents()
        alignment = self._alignment(question_vector, active)
        diversity = self._diversity(active)
        size_penalty = self._size_penalty(len(active), profile)
        prompt_complexity = sum(slot.complexity() for slot in state.role_to_prompt.values()) / max(
            1, len(state.role_to_prompt)
        )
        prompt_penalty = 0.10 * prompt_complexity
        template_prior = self._template_prior(state.template, profile)
        verification_bonus = self._verification_bonus(state.template, state)
        required_bonus = 0.0
        if profile.required_agent_ids:
            required_hits = sum(1 for aid in profile.required_agent_ids if aid in active)
            required_bonus = 0.06 * (required_hits / max(1, len(profile.required_agent_ids)))
        preferred_bonus = 0.0
        for role, preferred_ids in profile.role_agent_preferences.items():
            agent_id = state.role_to_agent.get(role)
            if agent_id and agent_id in preferred_ids:
                preferred_bonus += 0.02
        structure_bonus = self._task_structure_bonus(compiled, profile, metadata)
        prompt_fit = self._prompt_fit(state, profile)
        prior = profile.structure_prior
        task_score = (
            0.48 * alignment
            + 0.12 * diversity
            + template_prior
            + verification_bonus
            + required_bonus
            + preferred_bonus
            + structure_bonus
            + prompt_fit
            - size_penalty
            - (prompt_penalty * profile.proxy_prompt_penalty_scale)
        )
        topology_shape = self._topology_shape_quality(compiled, profile)
        redundancy_readiness = self._redundancy_readiness(compiled, profile)
        runtime_affordability = self._runtime_affordability(compiled, profile)
        structure_total = max(1e-6, prior.proxy_shape_weight + prior.proxy_redundancy_weight + prior.proxy_affordability_weight)
        shape_weight = prior.proxy_shape_weight / structure_total
        redundancy_weight = prior.proxy_redundancy_weight / structure_total
        affordability_weight = prior.proxy_affordability_weight / structure_total
        structure_score = (
            shape_weight * topology_shape
            + redundancy_weight * redundancy_readiness
            + affordability_weight * runtime_affordability
        )
        score = prior.proxy_task_weight * task_score + prior.proxy_structure_weight * structure_score
        uncertainty = (
            0.08
            + 0.18 * max(0.0, 1.0 - alignment)
            + 0.08 * prompt_complexity
            + max(0.0, 0.03 - structure_bonus)
        )
        return ProxyScore(
            score=score,
            uncertainty=uncertainty,
            features={
                "alignment": alignment,
                "diversity": diversity,
                "size_penalty": size_penalty,
                "prompt_complexity": prompt_complexity,
                "verification_bonus": verification_bonus,
                "required_bonus": required_bonus,
                "preferred_bonus": preferred_bonus,
                "structure_bonus": structure_bonus,
                "prompt_fit": prompt_fit,
                "task_proxy": task_score,
                "topology_shape": topology_shape,
                "redundancy_readiness": redundancy_readiness,
                "runtime_affordability": runtime_affordability,
                "structure_proxy": structure_score,
            },
        )
