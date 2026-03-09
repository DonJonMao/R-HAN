from __future__ import annotations

from collections import OrderedDict
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .math_utils import cosine, dot
from .questioning import question_signature
from .types import MASConfig, Vector
from .vectorizer import HashingVectorizer


@dataclass
class TaskConditioningResult:
    question_text: str
    question_vector: Vector
    selected_agent_ids: List[str]
    gating_scores: Dict[str, float]


@dataclass
class _GatingState:
    w_rel: float = 0.0
    w_div: float = 0.0
    w_pair: List[float] = field(default_factory=list)
    agent_bias: Dict[str, float] = field(default_factory=dict)
    ema_feedback: float = 0.0
    ema_initialized: bool = False


class LearnableSecondOrderGater:
    """Online-learned second-order gating model with per-question states."""

    def __init__(self, config: MASConfig, dim: int):
        self.config = config
        self.dim = dim
        self._states: "OrderedDict[str, _GatingState]" = OrderedDict()

    @staticmethod
    def _clip(x: float, lim: float) -> float:
        if lim <= 0.0:
            return x
        return max(-lim, min(lim, x))

    def _normalize_feedback(self, score: float) -> float:
        scale = max(1e-6, self.config.gating_feedback_scale)
        return math.tanh(score / scale)

    def question_key(self, question_text: Optional[str], question_vector: Vector) -> str:
        if not self.config.gating_question_isolated:
            return "__global__"
        return question_signature(
            question_text=question_text,
            question_vector=question_vector,
            vector_decimals=self.config.question_signature_hash_decimals,
        )

    def _state(self, q_key: str) -> _GatingState:
        state = self._states.get(q_key)
        if state is None:
            state = _GatingState(w_pair=[0.0] * self.dim)
            self._states[q_key] = state
            max_entries = max(0, int(self.config.gating_state_max_entries))
            if max_entries > 0:
                while len(self._states) > max_entries:
                    self._states.popitem(last=False)
            return state

        self._states.move_to_end(q_key)
        return state

    @staticmethod
    def _pair_feat(a: Vector, b: Vector, q: Vector) -> Vector:
        # q-conditioned second-order interaction feature.
        return [(ai * qi) * (bi * qi) for ai, bi, qi in zip(a, b, q)]

    def _pair_stats(
        self,
        selected: List[str],
        agent_vectors: Dict[str, Vector],
        q: Vector,
    ) -> tuple[Vector, float]:
        if len(selected) < 2:
            return [0.0] * self.dim, 0.0

        pair_sum = [0.0] * self.dim
        div_sum = 0.0
        cnt = 0
        for i in range(len(selected)):
            a = agent_vectors[selected[i]]
            for j in range(i + 1, len(selected)):
                b = agent_vectors[selected[j]]
                feat = self._pair_feat(a, b, q)
                pair_sum = [x + y for x, y in zip(pair_sum, feat)]
                div_sum += 1.0 - cosine(a, b)
                cnt += 1
        inv = 1.0 / max(1, cnt)
        return [x * inv for x in pair_sum], div_sum * inv

    def _bias(self, state: _GatingState, aid: str) -> float:
        if aid not in state.agent_bias:
            state.agent_bias[aid] = 0.0
        return state.agent_bias[aid]

    def incremental_score(
        self,
        aid: str,
        selected: List[str],
        agent_vectors: Dict[str, Vector],
        q: Vector,
        q_key: str,
    ) -> float:
        state = self._state(q_key)
        vec = agent_vectors[aid]
        rel = dot(vec, q)
        score = state.w_rel * rel + self._bias(state, aid)

        if selected:
            pair_vals = []
            div_vals = []
            for sid in selected:
                svec = agent_vectors[sid]
                feat = self._pair_feat(vec, svec, q)
                pair_vals.append(dot(state.w_pair, feat))
                div_vals.append(1.0 - cosine(vec, svec))
            score += (sum(pair_vals) / len(pair_vals)) + state.w_div * (sum(div_vals) / len(div_vals))
        return score

    def predict_set_value(
        self,
        selected: List[str],
        agent_vectors: Dict[str, Vector],
        q: Vector,
        q_key: str,
    ) -> float:
        if not selected:
            return 0.0
        state = self._state(q_key)
        rel_mean = sum(dot(agent_vectors[aid], q) for aid in selected) / len(selected)
        bias_mean = sum(self._bias(state, aid) for aid in selected) / len(selected)
        pair_feat_mean, div_mean = self._pair_stats(selected, agent_vectors, q)
        return state.w_rel * rel_mean + bias_mean + dot(state.w_pair, pair_feat_mean) + state.w_div * div_mean

    def update(
        self,
        selected: List[str],
        agent_vectors: Dict[str, Vector],
        q: Vector,
        q_key: str,
        target_score: float,
    ) -> None:
        if not selected:
            return
        state = self._state(q_key)
        norm_target = self._normalize_feedback(target_score)
        target_clip = max(0.0, self.config.gating_target_clip)

        if not state.ema_initialized:
            # Keep first-step signal non-zero but avoid over-shooting.
            first_gain = max(0.0, self.config.gating_first_update_gain)
            centered_target = self._clip(first_gain * norm_target, target_clip)
            state.ema_feedback = norm_target
            state.ema_initialized = True
        else:
            prev = state.ema_feedback
            centered_target = self._clip(norm_target - prev, target_clip)
            alpha = min(max(self.config.gating_feedback_ema, 0.0), 1.0)
            state.ema_feedback = (1.0 - alpha) * prev + alpha * norm_target

        pred = self.predict_set_value(selected, agent_vectors, q, q_key=q_key)
        err = pred - centered_target

        rel_mean = sum(dot(agent_vectors[aid], q) for aid in selected) / len(selected)
        pair_feat_mean, div_mean = self._pair_stats(selected, agent_vectors, q)

        lr = self.config.gating_learning_rate
        l2 = self.config.gating_l2
        grad_scale = self._clip(2.0 * err, max(0.0, self.config.gating_grad_clip))

        state.w_rel -= lr * (grad_scale * rel_mean + l2 * state.w_rel)
        state.w_div -= lr * (grad_scale * div_mean + l2 * state.w_div)
        state.w_pair = [
            w - lr * (grad_scale * g + l2 * w)
            for w, g in zip(state.w_pair, pair_feat_mean)
        ]

        bias_grad = grad_scale * (1.0 / len(selected))
        for aid in selected:
            state.agent_bias[aid] = self._bias(state, aid) - lr * bias_grad


class TaskConditioner:
    """Question conditioning + subset gating.

    Uses:
    - deterministic greedy heuristic selector
    - optional online-learned second-order scorer
    """

    def __init__(self, config: MASConfig, vectorizer: HashingVectorizer):
        self.config = config
        self.vectorizer = vectorizer
        self.gater = LearnableSecondOrderGater(
            config=config,
            dim=config.embedding_dim,
        )

    def encode_question(self, question_text: Optional[str]) -> Vector:
        if not question_text:
            return [0.0] * self.config.embedding_dim
        return self.vectorizer.vectorize_text(question_text)

    def _score_agent(self, agent_vector: Vector, q: Vector) -> float:
        return dot(agent_vector, q)

    @staticmethod
    def _q_complement(a: Vector, b: Vector, q: Vector) -> float:
        # Complementarity on q-relevant dimensions.
        return sum(abs(x - y) * abs(qi) for x, y, qi in zip(a, b, q)) / max(1, len(q))

    @staticmethod
    def _diversity(a: Vector, b: Vector) -> float:
        return 1.0 - cosine(a, b)

    def _heuristic_score(
        self,
        aid: str,
        selected: List[str],
        agent_vectors: Dict[str, Vector],
        q: Vector,
        relevance: Dict[str, float],
    ) -> float:
        rel = relevance[aid]
        if not selected:
            return self.config.gating_relevance_weight * rel

        comps = [
            self._q_complement(agent_vectors[aid], agent_vectors[sid], q)
            for sid in selected
        ]
        divs = [
            self._diversity(agent_vectors[aid], agent_vectors[sid])
            for sid in selected
        ]
        comp = sum(comps) / len(comps)
        div = sum(divs) / len(divs)
        return (
            self.config.gating_relevance_weight * rel
            + self.config.gating_complement_weight * comp
            + self.config.gating_diversity_weight * div
        )

    def select_agents(
        self,
        question_text: Optional[str],
        agent_vectors: Dict[str, Vector],
        agent_ids: List[str],
        top_k: Optional[int] = None,
    ) -> TaskConditioningResult:
        q = self.encode_question(question_text)
        q_text = question_text or ""
        relevance = {aid: self._score_agent(agent_vectors[aid], q) for aid in agent_ids}
        q_key = self.gater.question_key(q_text, q)

        if not self.config.enable_subset_selection:
            selected = list(agent_ids)
        elif (not q_text) and (top_k is None):
            # Preserve legacy behavior when task text is absent.
            selected = list(agent_ids)
        else:
            k = top_k if top_k is not None else self.config.agent_subset_top_k
            if k <= 0:
                k = len(agent_ids)
            k = max(self.config.agent_subset_min_k, min(k, len(agent_ids)))

            # Deterministic ordering to keep subset selection reproducible.
            remaining = sorted(agent_ids)
            selected = []
            eps = 1e-12
            while remaining and len(selected) < k:
                best_aid = ""
                best_val = float("-inf")
                for aid in remaining:
                    val = self._heuristic_score(aid, selected, agent_vectors, q, relevance)
                    if self.config.enable_learnable_gating:
                        val += self.gater.incremental_score(
                            aid=aid,
                            selected=selected,
                            agent_vectors=agent_vectors,
                            q=q,
                            q_key=q_key,
                        )

                    if (val > best_val + eps) or (abs(val - best_val) <= eps and (not best_aid or aid < best_aid)):
                        best_val = val
                        best_aid = aid

                if not best_aid:
                    break
                selected.append(best_aid)
                remaining = [aid for aid in remaining if aid != best_aid]

        return TaskConditioningResult(
            question_text=q_text,
            question_vector=q,
            selected_agent_ids=selected,
            gating_scores=relevance,
        )

    def update_from_feedback(
        self,
        question_text: Optional[str],
        question_vector: Vector,
        selected_agent_ids: List[str],
        feedback_score: float,
        agent_vectors: Dict[str, Vector],
    ) -> None:
        if not self.config.enable_learnable_gating:
            return
        if not selected_agent_ids:
            return
        q_key = self.gater.question_key(question_text, question_vector)
        self.gater.update(
            selected=selected_agent_ids,
            agent_vectors=agent_vectors,
            q=question_vector,
            q_key=q_key,
            target_score=feedback_score,
        )
