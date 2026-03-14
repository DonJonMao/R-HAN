from __future__ import annotations

import hashlib
import inspect
import json
import math
import random
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional, Protocol, Sequence, Tuple

from .questioning import question_signature
from .scoring import ScoreModel
from .types import DAGState, MASConfig, RewardBreakdown, TaskEvaluation, Vector


class MASTaskEvaluator(Protocol):
    """Runs a full MAS task and returns outcome metrics.

    Implementations should support `active_agent_ids` to enable contribution
    estimation by agent ablation.
    """

    def evaluate(
        self,
        dag: DAGState,
        active_agent_ids: Optional[Sequence[str]] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[Vector] = None,
    ) -> TaskEvaluation:
        ...


def _stable_exp(x: float, clip_min: float, clip_max: float) -> float:
    return math.exp(min(max(x, clip_min), clip_max))


@dataclass
class _EvalCache:
    # Cache utility by active agent set.
    values: Dict[Tuple[str, ...], float]


@dataclass
class _EvalCallSpec:
    has_active_kw: bool
    has_question_text_kw: bool
    has_question_vector_kw: bool
    active_positional: bool


class MASRewardModel:
    """Combines BIC + task utility + agent contribution into training reward."""

    def __init__(self, config: MASConfig, scorer: ScoreModel):
        self.config = config
        self.scorer = scorer
        self.rng = random.Random(config.random_seed)
        self._true_eval_cache: OrderedDict[
            Tuple[str, str, Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]],
            RewardBreakdown,
        ] = OrderedDict()
        self._question_prior_cache: OrderedDict[Tuple[str, str], Tuple[float, float]] = OrderedDict()
        self._eval_call_spec_cache: Dict[type, _EvalCallSpec] = {}

    @staticmethod
    def _graph_signature(dag: DAGState) -> Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]:
        # Include node identity mapping to avoid collisions across node subsets/orders.
        nodes = tuple(str(x) for x in dag.nodes)
        named_edges: list[Tuple[str, str]] = []
        for src, dst in sorted(dag.edges):
            if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                named_edges.append((nodes[src], nodes[dst]))
            else:
                named_edges.append((f"idx:{src}", f"idx:{dst}"))
        return nodes, tuple(sorted(named_edges))

    def _question_signature(self, question_text: Optional[str], question_vector: Optional[Vector]) -> str:
        return question_signature(
            question_text=question_text,
            question_vector=question_vector,
            vector_decimals=self.config.question_signature_hash_decimals,
        )

    @staticmethod
    def _sig_value(value, depth: int = 0):
        if depth >= 3:
            return "<max-depth>"
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, (list, tuple)):
            return [MASRewardModel._sig_value(v, depth + 1) for v in list(value)[:16]]
        if isinstance(value, set):
            vals = [MASRewardModel._sig_value(v, depth + 1) for v in list(value)[:16]]
            return sorted(vals, key=lambda x: repr(x))
        if isinstance(value, dict):
            out = {}
            cnt = 0
            for k in sorted(value.keys(), key=lambda x: repr(x)):
                if cnt >= 16:
                    break
                if not isinstance(k, (str, int, float, bool)):
                    continue
                out[str(k)] = MASRewardModel._sig_value(value[k], depth + 1)
                cnt += 1
            return out
        return f"<{value.__class__.__module__}.{value.__class__.__qualname__}>"

    @staticmethod
    def _init_param_snapshot(evaluator: MASTaskEvaluator) -> Dict[str, object]:
        out: Dict[str, object] = {}
        try:
            sig = inspect.signature(evaluator.__class__.__init__)
        except (TypeError, ValueError):
            return out

        for name, p in sig.parameters.items():
            if name == "self":
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                continue
            if not hasattr(evaluator, name):
                continue
            val = getattr(evaluator, name)
            if callable(val):
                continue
            out[name] = MASRewardModel._sig_value(val)
        return out

    @staticmethod
    def _evaluator_signature(evaluator: Optional[MASTaskEvaluator]) -> str:
        if evaluator is None:
            return "none"

        custom = getattr(evaluator, "cache_signature", None)
        if callable(custom):
            try:
                val = custom()
                if isinstance(val, str) and val:
                    return f"custom::{val}"
            except Exception:
                pass

        cls = evaluator.__class__
        parts = [f"{cls.__module__}.{cls.__qualname__}"]
        for attr in ("cache_version", "version", "model_version"):
            if hasattr(evaluator, attr):
                v = getattr(evaluator, attr)
                if isinstance(v, (str, int, float, bool)):
                    parts.append(f"{attr}={v}")

        snap = MASRewardModel._init_param_snapshot(evaluator)
        if snap:
            payload = json.dumps(snap, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
            digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
            parts.append(f"init={digest}")
        return "|".join(parts)

    def _touch_question_prior(
        self,
        key: Tuple[str, str],
        value: Optional[Tuple[float, float]] = None,
    ) -> Optional[Tuple[float, float]]:
        if value is None:
            cached = self._question_prior_cache.get(key)
            if cached is not None:
                self._question_prior_cache.move_to_end(key)
            return cached

        self._question_prior_cache[key] = value
        self._question_prior_cache.move_to_end(key)
        max_entries = max(0, int(self.config.reward_question_prior_max_entries))
        if max_entries > 0:
            while len(self._question_prior_cache) > max_entries:
                self._question_prior_cache.popitem(last=False)
        return value

    def _touch_true_eval_cache(
        self,
        key: Tuple[str, str, Tuple[Tuple[str, ...], Tuple[Tuple[str, str], ...]]],
        value: Optional[RewardBreakdown] = None,
    ) -> Optional[RewardBreakdown]:
        if value is None:
            cached = self._true_eval_cache.get(key)
            if cached is not None:
                self._true_eval_cache.move_to_end(key)
            return cached

        self._true_eval_cache[key] = value
        self._true_eval_cache.move_to_end(key)
        max_entries = max(0, int(self.config.reward_true_eval_cache_max_entries))
        if max_entries > 0:
            while len(self._true_eval_cache) > max_entries:
                self._true_eval_cache.popitem(last=False)
        return value

    @staticmethod
    def _resolve_eval_call_spec(evaluator: MASTaskEvaluator) -> _EvalCallSpec:
        sig = inspect.signature(evaluator.evaluate)
        params = [p for name, p in sig.parameters.items() if name != "self"]
        has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
        names = {p.name for p in params}

        has_active_kw = has_kwargs or ("active_agent_ids" in names)
        has_q_text_kw = has_kwargs or ("question_text" in names)
        has_q_vec_kw = has_kwargs or ("question_vector" in names)

        active_positional = False
        if not has_active_kw:
            positional = [
                p
                for p in params
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            # Expected: evaluate(dag, active_agent_ids, ...)
            active_positional = len(positional) >= 2

        return _EvalCallSpec(
            has_active_kw=has_active_kw,
            has_question_text_kw=has_q_text_kw,
            has_question_vector_kw=has_q_vec_kw,
            active_positional=active_positional,
        )

    def _safe_call_evaluator(
        self,
        evaluator: MASTaskEvaluator,
        dag: DAGState,
        active_agent_ids: Sequence[str],
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> TaskEvaluation:
        cls = evaluator.__class__
        spec = self._eval_call_spec_cache.get(cls)
        if spec is None:
            spec = self._resolve_eval_call_spec(evaluator)
            self._eval_call_spec_cache[cls] = spec

        kwargs = {}
        if spec.has_question_text_kw:
            kwargs["question_text"] = question_text
        if spec.has_question_vector_kw:
            kwargs["question_vector"] = question_vector

        if spec.has_active_kw:
            kwargs["active_agent_ids"] = active_agent_ids
            return evaluator.evaluate(dag, **kwargs)
        if spec.active_positional:
            return evaluator.evaluate(dag, active_agent_ids, **kwargs)

        raise TypeError(
            "Evaluator.evaluate must accept `active_agent_ids` (keyword or second positional argument)."
        )

    def utility(self, ev: TaskEvaluation) -> float:
        cfg = self.config
        return (
            cfg.utility_task_score_weight * ev.task_score
            + cfg.utility_success_weight * ev.success
            - cfg.utility_latency_weight * ev.latency
            - cfg.utility_token_weight * ev.token_cost
            - cfg.utility_safety_weight * ev.safety_penalty
        )

    def _bic_term(self, bic_score: float) -> float:
        scale = max(1e-6, self.config.bic_normalize_scale)
        return math.tanh(bic_score / scale)

    def _eval_utility(
        self,
        evaluator: MASTaskEvaluator,
        dag: DAGState,
        active_agent_ids: Sequence[str],
        cache: _EvalCache,
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> float:
        key = tuple(sorted(active_agent_ids))
        if key in cache.values:
            return cache.values[key]
        if not active_agent_ids:
            cache.values[key] = 0.0
            return 0.0
        out = self._safe_call_evaluator(
            evaluator=evaluator,
            dag=dag,
            active_agent_ids=active_agent_ids,
            question_text=question_text,
            question_vector=question_vector,
        )
        utility = self.utility(out)
        cache.values[key] = utility
        return utility

    def _loo_contributions(
        self,
        evaluator: MASTaskEvaluator,
        dag: DAGState,
        full_utility: float,
        agent_ids: Sequence[str],
        cache: _EvalCache,
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> Dict[str, float]:
        contributions: Dict[str, float] = {}
        for aid in agent_ids:
            active = [x for x in agent_ids if x != aid]
            minus_utility = self._eval_utility(
                evaluator,
                dag,
                active,
                cache,
                question_text=question_text,
                question_vector=question_vector,
            )
            contributions[aid] = full_utility - minus_utility
        return contributions

    def _shapley_contributions(
        self,
        evaluator: MASTaskEvaluator,
        dag: DAGState,
        agent_ids: Sequence[str],
        cache: _EvalCache,
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> Dict[str, float]:
        contributions: Dict[str, float] = {aid: 0.0 for aid in agent_ids}
        if not agent_ids:
            return contributions

        perms = max(1, self.config.shapley_permutations)
        for _ in range(perms):
            order = list(agent_ids)
            self.rng.shuffle(order)

            active: list[str] = []
            utility_prev = self._eval_utility(
                evaluator,
                dag,
                active,
                cache,
                question_text=question_text,
                question_vector=question_vector,
            )
            for aid in order:
                active_next = active + [aid]
                utility_next = self._eval_utility(
                    evaluator,
                    dag,
                    active_next,
                    cache,
                    question_text=question_text,
                    question_vector=question_vector,
                )
                contributions[aid] += utility_next - utility_prev
                active = active_next
                utility_prev = utility_next

        for aid in contributions:
            contributions[aid] /= perms
        return contributions

    def estimate_agent_contributions(
        self,
        dag: DAGState,
        evaluator: Optional[MASTaskEvaluator],
        full_utility: float,
        question_text: Optional[str],
        question_vector: Optional[Vector],
    ) -> Dict[str, float]:
        if evaluator is None:
            return {}
        agent_ids = list(dag.nodes)
        cache = _EvalCache(values={})

        # Cache full set utility first.
        cache.values[tuple(sorted(agent_ids))] = full_utility

        mode = self.config.contribution_mode.lower()
        if mode in {"none", "off", "disabled"}:
            return {}
        if mode == "shapley":
            return self._shapley_contributions(
                evaluator,
                dag,
                agent_ids,
                cache,
                question_text=question_text,
                question_vector=question_vector,
            )
        return self._loo_contributions(
            evaluator,
            dag,
            full_utility,
            agent_ids,
            cache,
            question_text=question_text,
            question_vector=question_vector,
        )

    def _question_alignment(
        self,
        dag: DAGState,
        question_vector: Optional[Vector],
        agent_vectors: Optional[Dict[str, Vector]],
    ) -> float:
        if not question_vector or not agent_vectors:
            return 0.0
        vals: list[float] = []
        for aid in dag.nodes:
            if aid in agent_vectors:
                vals.append(sum(a * b for a, b in zip(question_vector, agent_vectors[aid])))
        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def _size_penalty(self, dag: DAGState) -> float:
        cfg = self.config
        exempt = set(cfg.size_penalty_exempt_agents)
        effective = sum(1 for aid in dag.nodes if aid not in exempt)
        effective = max(0, effective - cfg.size_penalty_free_count)
        t = max(0, cfg.size_penalty_stage1_threshold)
        r1 = max(0.0, cfg.size_penalty_stage1_rate)
        r2 = max(0.0, cfg.size_penalty_stage2_rate)

        if effective <= t:
            penalty = r1 * effective
        else:
            penalty = r1 * t + r2 * (effective - t)
        return -penalty

    def score_and_reward(
        self,
        dag: DAGState,
        evaluator: Optional[MASTaskEvaluator] = None,
        use_true_evaluator: bool = True,
        proxy_task_utility: Optional[float] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[Vector] = None,
        agent_vectors: Optional[Dict[str, Vector]] = None,
    ) -> RewardBreakdown:
        sig = self._graph_signature(dag)
        qsig = self._question_signature(question_text, question_vector)
        esig = self._evaluator_signature(evaluator)
        prior_key = (esig, qsig)
        cache_key = (esig, qsig, sig)
        if not use_true_evaluator:
            cached = self._touch_true_eval_cache(cache_key)
        else:
            cached = None
        if cached is not None:
            task_utility = cached.task_utility
            task_score = cached.task_score
            task_success = cached.task_success
            if self.config.use_task_score_as_bic:
                bic_score = task_utility
            else:
                bic_score = self.scorer.score(dag)
            bic_term = self._bic_term(bic_score)
            question_alignment_term = self._question_alignment(dag, question_vector, agent_vectors)
            size_penalty_term = self._size_penalty(dag)
            cfg = self.config
            task_weight = 0.0 if cfg.use_task_score_as_bic else cfg.reward_task_weight
            total_score = (
                task_weight * cached.task_utility
                + cfg.reward_bic_weight * bic_term
                + cfg.reward_contrib_weight * cached.contribution_term
                + cfg.reward_question_weight * question_alignment_term
                + cfg.reward_size_penalty_weight * size_penalty_term
            )
            reward = _stable_exp(cfg.reward_temperature * total_score, cfg.reward_clip_min, cfg.reward_clip_max)
            out_cached = RewardBreakdown(
                bic_score=bic_score,
                bic_term=bic_term,
                task_utility=task_utility,
                task_score=task_score,
                task_success=task_success,
                task_safety_penalty=cached.task_safety_penalty,
                contribution_term=cached.contribution_term,
                total_score=total_score,
                reward=reward,
                question_alignment_term=question_alignment_term,
                size_penalty_term=size_penalty_term,
                agent_contributions=dict(cached.agent_contributions),
                component_terms={
                    "task": task_weight * cached.task_utility,
                    "bic": cfg.reward_bic_weight * bic_term,
                    "contrib": cfg.reward_contrib_weight * cached.contribution_term,
                    "question": cfg.reward_question_weight * question_alignment_term,
                    "size_penalty": cfg.reward_size_penalty_weight * size_penalty_term,
                },
            )
            self._touch_question_prior(prior_key, (cached.task_utility, cached.contribution_term))
            return out_cached

        task_utility = 0.0
        task_score = 0.0
        task_success = 0.0
        task_safety_penalty = 0.0
        contributions: Dict[str, float] = {}
        contribution_term = 0.0
        cfg = self.config
        did_true_eval = False

        if use_true_evaluator and evaluator is not None:
            full_eval = self._safe_call_evaluator(
                evaluator=evaluator,
                dag=dag,
                active_agent_ids=list(dag.nodes),
                question_text=question_text,
                question_vector=question_vector,
            )
            task_utility = self.utility(full_eval)
            task_score = full_eval.task_score
            task_success = full_eval.success
            task_safety_penalty = full_eval.safety_penalty
            did_true_eval = True
            contributions = self.estimate_agent_contributions(
                dag,
                evaluator,
                task_utility,
                question_text=question_text,
                question_vector=question_vector,
            )
            positive_contrib = [max(0.0, v) for v in contributions.values()]
            contribution_term = (sum(positive_contrib) / len(positive_contrib)) if positive_contrib else 0.0
        elif use_true_evaluator and evaluator is None and proxy_task_utility is not None:
            task_utility = proxy_task_utility
        elif not use_true_evaluator:
            miss_policy = cfg.reward_cache_miss_policy.lower()
            if miss_policy == "proxy_only":
                if proxy_task_utility is not None:
                    task_utility = proxy_task_utility
            elif miss_policy == "true_utility":
                if evaluator is not None:
                    full_eval = self._safe_call_evaluator(
                        evaluator=evaluator,
                        dag=dag,
                        active_agent_ids=list(dag.nodes),
                        question_text=question_text,
                        question_vector=question_vector,
                    )
                    task_utility = self.utility(full_eval)
                    task_score = full_eval.task_score
                    task_success = full_eval.success
                    task_safety_penalty = full_eval.safety_penalty
                    did_true_eval = True
                    if cfg.reward_cache_miss_true_eval_contrib:
                        contributions = self.estimate_agent_contributions(
                            dag,
                            evaluator,
                            task_utility,
                            question_text=question_text,
                            question_vector=question_vector,
                        )
                        positive_contrib = [max(0.0, v) for v in contributions.values()]
                        contribution_term = (sum(positive_contrib) / len(positive_contrib)) if positive_contrib else 0.0
                elif proxy_task_utility is not None:
                    task_utility = proxy_task_utility
            else:
                # Default/recommended: prior_or_proxy
                if proxy_task_utility is not None:
                    task_utility = proxy_task_utility
                prior = self._touch_question_prior(prior_key)
                if prior is not None:
                    prior_task, prior_contrib = prior
                    if proxy_task_utility is None:
                        task_utility = prior_task
                    contribution_term = prior_contrib
        elif proxy_task_utility is not None:
            task_utility = proxy_task_utility

        if cfg.use_task_score_as_bic:
            bic_score = task_utility
        else:
            bic_score = self.scorer.score(dag)
        bic_term = self._bic_term(bic_score)

        question_alignment_term = self._question_alignment(dag, question_vector, agent_vectors)
        size_penalty_term = self._size_penalty(dag)

        task_weight = 0.0 if cfg.use_task_score_as_bic else cfg.reward_task_weight
        total_score = (
            task_weight * task_utility
            + cfg.reward_bic_weight * bic_term
            + cfg.reward_contrib_weight * contribution_term
            + cfg.reward_question_weight * question_alignment_term
            + cfg.reward_size_penalty_weight * size_penalty_term
        )
        reward = _stable_exp(cfg.reward_temperature * total_score, cfg.reward_clip_min, cfg.reward_clip_max)

        out = RewardBreakdown(
            bic_score=bic_score,
            bic_term=bic_term,
            task_utility=task_utility,
            task_score=task_score,
            task_success=task_success,
            task_safety_penalty=task_safety_penalty,
            contribution_term=contribution_term,
            total_score=total_score,
            reward=reward,
            question_alignment_term=question_alignment_term,
            size_penalty_term=size_penalty_term,
            agent_contributions=contributions,
            component_terms={
                "task": task_weight * task_utility,
                "bic": cfg.reward_bic_weight * bic_term,
                "contrib": cfg.reward_contrib_weight * contribution_term,
                "question": cfg.reward_question_weight * question_alignment_term,
                "size_penalty": cfg.reward_size_penalty_weight * size_penalty_term,
            },
        )
        self._touch_question_prior(prior_key, (task_utility, contribution_term))
        if use_true_evaluator or did_true_eval:
            self._touch_true_eval_cache(cache_key, out)
        return out
