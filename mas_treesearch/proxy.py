from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

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

    def score(
        self,
        compiled: CompiledArchitecture,
        question_vector: Vector,
        profile: DatasetProfile = DEFAULT_PROFILE,
    ) -> ProxyScore:
        state = compiled.state
        active = state.active_agents()
        alignment = self._alignment(question_vector, active)
        diversity = self._diversity(active)
        size_penalty = max(0.0, 0.08 * max(0, len(active) - 4))
        prompt_complexity = sum(slot.complexity() for slot in state.role_to_prompt.values()) / max(
            1, len(state.role_to_prompt)
        )
        prompt_penalty = 0.10 * prompt_complexity
        template_prior = {
            WorkflowTemplate.DIRECT: 0.48,
            WorkflowTemplate.SOLVE_VERIFY: 0.58,
            WorkflowTemplate.PARALLEL_VOTE: 0.55,
            WorkflowTemplate.CRITIQUE_REVISE: 0.57,
            WorkflowTemplate.DEBATE_JUDGE: 0.53,
            WorkflowTemplate.ROUTE_SOLVE: 0.50,
        }.get(state.template, 0.50)
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

        score = (
            0.50 * alignment
            + 0.15 * diversity
            + template_prior
            + verification_bonus
            + required_bonus
            + preferred_bonus
            - size_penalty
            - (prompt_penalty * profile.proxy_prompt_penalty_scale)
        )
        uncertainty = 0.08 + 0.20 * max(0.0, 1.0 - alignment) + 0.10 * prompt_complexity
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
            },
        )
