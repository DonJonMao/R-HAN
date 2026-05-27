from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .agents import AgentPool
from .clients import CachedEmbedder
from .config import SearchConfig
from .profiles import DEFAULT_PROFILE, DatasetProfile
from .types import Vector


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class AgentSelection:
    question_vector: Vector
    candidate_agent_ids: List[str]
    core_agent_ids: List[str]
    explore_agent_ids: List[str]
    scores: Dict[str, float]


class TaskConditioner:
    """Task-conditioned candidate agent selector with embedding caching."""

    def __init__(self, agent_pool: AgentPool, embedder: CachedEmbedder, config: SearchConfig):
        self.agent_pool = agent_pool
        self.embedder = embedder
        self.config = config
        self._agent_vectors: Dict[str, Vector] = {
            agent_id: self.embedder.embed(text)
            for agent_id, text in self.agent_pool.texts()
        }

    @property
    def agent_vectors(self) -> Dict[str, Vector]:
        return self._agent_vectors

    def encode_question(self, question_text: str) -> Vector:
        return self.embedder.embed(question_text)

    def _incremental_diversity(
        self,
        agent_id: str,
        selected: Sequence[str],
    ) -> float:
        if not selected:
            return 0.0
        cur = self._agent_vectors[agent_id]
        sims = [cosine(cur, self._agent_vectors[sid]) for sid in selected]
        return 1.0 - (sum(sims) / max(1, len(sims)))

    def select(self, question_text: str, profile: DatasetProfile = DEFAULT_PROFILE) -> AgentSelection:
        q = self.encode_question(question_text)
        preferred_agent_ids = {
            agent_id
            for preferred in profile.role_agent_preferences.values()
            for agent_id in preferred
            if agent_id in self._agent_vectors
        }
        required_agent_ids = [
            agent_id for agent_id in profile.required_agent_ids if agent_id in self._agent_vectors
        ]
        core_k = profile.search_overrides.candidate_core_k or self.config.candidate_core_k
        explore_k = profile.search_overrides.candidate_explore_k or self.config.candidate_explore_k
        max_k = profile.search_overrides.candidate_max_k or self.config.candidate_max_k
        core_k = max(core_k, len(required_agent_ids))
        max_k = max(max_k, core_k, len(required_agent_ids))
        relevance = {
            agent_id: (
                cosine(vec, q)
                + (0.20 if agent_id in required_agent_ids else 0.0)
                + (0.08 if agent_id in preferred_agent_ids else 0.0)
            )
            for agent_id, vec in self._agent_vectors.items()
        }
        ordered = sorted(relevance.items(), key=lambda x: (-x[1], x[0]))

        core: List[str] = []
        remaining = [aid for aid, _ in ordered]
        while remaining and len(core) < core_k:
            best_id = ""
            best_score = float("-inf")
            for aid in remaining:
                score = relevance[aid] + 0.35 * self._incremental_diversity(aid, core)
                if score > best_score:
                    best_score = score
                    best_id = aid
            if not best_id:
                break
            core.append(best_id)
            remaining = [aid for aid in remaining if aid != best_id]

        for agent_id in required_agent_ids:
            if agent_id in core:
                continue
            if agent_id in remaining:
                core.append(agent_id)
                remaining = [aid for aid in remaining if aid != agent_id]

        explore = remaining[: explore_k]
        candidate = core + [aid for aid in explore if aid not in core]
        for agent_id in required_agent_ids:
            if agent_id not in candidate:
                candidate.append(agent_id)
        if len(candidate) > max_k:
            keep = set(required_agent_ids)
            trimmed: List[str] = []
            for agent_id in candidate:
                if len(trimmed) >= max_k and agent_id not in keep:
                    continue
                if agent_id not in trimmed:
                    trimmed.append(agent_id)
            candidate = trimmed[: max_k] if len(trimmed) > max_k else trimmed
        return AgentSelection(
            question_vector=q,
            candidate_agent_ids=candidate,
            core_agent_ids=core,
            explore_agent_ids=explore,
            scores=relevance,
        )
