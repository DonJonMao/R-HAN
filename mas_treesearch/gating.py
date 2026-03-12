from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from .agents import AgentPool
from .clients import CachedEmbedder
from .config import SearchConfig
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

    def select(self, question_text: str) -> AgentSelection:
        q = self.encode_question(question_text)
        relevance = {
            agent_id: cosine(vec, q)
            for agent_id, vec in self._agent_vectors.items()
        }
        ordered = sorted(relevance.items(), key=lambda x: (-x[1], x[0]))

        core: List[str] = []
        remaining = [aid for aid, _ in ordered]
        while remaining and len(core) < self.config.candidate_core_k:
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

        explore = remaining[: self.config.candidate_explore_k]
        candidate = core + [aid for aid in explore if aid not in core]
        if len(candidate) > self.config.candidate_max_k:
            candidate = candidate[: self.config.candidate_max_k]
        return AgentSelection(
            question_vector=q,
            candidate_agent_ids=candidate,
            core_agent_ids=core,
            explore_agent_ids=explore,
            scores=relevance,
        )
