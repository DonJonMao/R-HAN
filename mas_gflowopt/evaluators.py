from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .agent_pool import AgentPool, default_agent_pool
from .types import DAGState, TaskEvaluation


@dataclass
class HeuristicEvaluatorConfig:
    """Config for deterministic, no-LLM MAS task evaluation."""

    target_density: float = 0.25
    safety_over_size_penalty: float = 0.05
    success_threshold: float = 0.55


class HeuristicMASTaskEvaluator:
    """A practical baseline evaluator for MAS graph optimization.

    This evaluator is intentionally deterministic and lightweight:
    - no LLM calls
    - supports `active_agent_ids` for contribution estimation
    - uses question-domain matching + graph-structure quality
    """

    _DOMAIN_KEYWORDS: Dict[str, Tuple[str, ...]] = {
        "medicine": ("medical", "diagnos", "patient", "clinic", "safety", "doctor", "treat"),
        "finance": ("finance", "risk", "portfolio", "cost", "market", "investment", "budget"),
        "software": ("software", "code", "system", "api", "service", "program", "deploy"),
        "math": ("proof", "optimiz", "model", "equation", "math", "theorem", "inference"),
    }

    _DOMAIN_REQUIREMENTS: Dict[str, Set[str]] = {
        "medicine": {"doctor", "algo_designer", "programmer"},
        "finance": {"finance_expert", "algo_designer"},
        "software": {"programmer", "algo_designer"},
        "math": {"mathematician", "algo_designer"},
    }

    _BASE_DEPENDENCIES: Tuple[Tuple[str, str], ...] = (
        ("algo_designer", "programmer"),
        ("mathematician", "algo_designer"),
        ("doctor", "algo_designer"),
        ("finance_expert", "algo_designer"),
        ("algo_designer", "doctor"),
        ("algo_designer", "finance_expert"),
    )

    def __init__(
        self,
        agent_pool: Optional[AgentPool] = None,
        config: Optional[HeuristicEvaluatorConfig] = None,
    ):
        self.agent_pool = agent_pool or default_agent_pool()
        self.config = config or HeuristicEvaluatorConfig()

        self._domain_by_agent: Dict[str, str] = {}
        self._role_by_agent: Dict[str, str] = {}
        for a in self.agent_pool.agents:
            self._domain_by_agent[a.agent_id] = str(a.metadata.get("domain", "")).strip().lower()
            self._role_by_agent[a.agent_id] = a.role

    def cache_signature(self) -> str:
        # Stable signature for reward-cache partitioning.
        parts = []
        for aid in sorted(self._domain_by_agent.keys()):
            parts.append(f"{aid}:{self._domain_by_agent.get(aid,'')}")
        return "heuristic-v1|" + "|".join(parts)

    @staticmethod
    def _clip(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def _infer_domains(self, question_text: Optional[str]) -> Set[str]:
        txt = (question_text or "").lower()
        out: Set[str] = set()
        for dom, kws in self._DOMAIN_KEYWORDS.items():
            if any(k in txt for k in kws):
                out.add(dom)
        return out

    def _active_edges(self, dag: DAGState, active: Set[str]) -> List[Tuple[int, int]]:
        out: List[Tuple[int, int]] = []
        for src, dst in dag.edges:
            if 0 <= src < len(dag.nodes) and 0 <= dst < len(dag.nodes):
                if dag.nodes[src] in active and dag.nodes[dst] in active:
                    out.append((src, dst))
        return out

    def _longest_path_len(self, node_ids: Sequence[str], edges: Sequence[Tuple[int, int]], active: Set[str]) -> int:
        idx_active = [i for i, aid in enumerate(node_ids) if aid in active]
        if not idx_active:
            return 0

        adj: Dict[int, List[int]] = {i: [] for i in idx_active}
        indeg: Dict[int, int] = {i: 0 for i in idx_active}
        for src, dst in edges:
            if src in adj and dst in adj:
                adj[src].append(dst)
                indeg[dst] += 1

        queue = [i for i in idx_active if indeg[i] == 0]
        order: List[int] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nxt in adj[cur]:
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(idx_active):
            # Should not happen for DAG states; keep robust fallback.
            return max(1, len(edges) // 2)

        dp = {i: 0 for i in idx_active}
        for u in order:
            for v in adj[u]:
                dp[v] = max(dp[v], dp[u] + 1)
        return max(dp.values()) if dp else 0

    def _dependency_score(self, active: Set[str], edges: Sequence[Tuple[int, int]], nodes: Sequence[str]) -> float:
        observed = set()
        for src, dst in edges:
            if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
                observed.add((nodes[src], nodes[dst]))

        candidates = [
            pair
            for pair in self._BASE_DEPENDENCIES
            if pair[0] in active and pair[1] in active
        ]
        if not candidates:
            return 0.5
        hit = sum(1 for p in candidates if p in observed)
        return hit / len(candidates)

    def _coverage_score(self, active: Set[str], domains: Set[str]) -> float:
        required: Set[str] = {"algo_designer"}
        for d in domains:
            required |= self._DOMAIN_REQUIREMENTS.get(d, set())
        if not required:
            return 1.0
        hit = sum(1 for aid in required if aid in active)
        return hit / len(required)

    def _compute_features(
        self,
        dag: DAGState,
        active_agent_ids: Optional[Sequence[str]],
        question_text: Optional[str],
    ) -> Tuple[Dict[str, float], float, float, float, Set[str], int, int]:
        active = set(active_agent_ids or dag.nodes)
        active &= set(dag.nodes)
        n_all = max(1, len(dag.nodes))
        n_active = len(active)
        active_ratio = n_active / n_all

        active_edges = self._active_edges(dag, active)
        e_active = len(active_edges)
        denom = max(1, n_active * max(1, n_active - 1))
        density = e_active / denom
        cohesion = 1.0 - abs(density - self.config.target_density)
        cohesion = self._clip(cohesion, 0.0, 1.0)

        path_len = self._longest_path_len(dag.nodes, active_edges, active)
        coord = 0.0 if n_active <= 1 else self._clip(path_len / max(1, n_active - 1), 0.0, 1.0)

        domains = self._infer_domains(question_text)
        coverage = self._coverage_score(active, domains)
        dep_score = self._dependency_score(active, active_edges, dag.nodes)

        unique_domains = {
            self._domain_by_agent.get(aid, "")
            for aid in active
            if self._domain_by_agent.get(aid, "")
        }
        diversity = self._clip(len(unique_domains) / 4.0, 0.0, 1.0)

        safety_penalty = 0.0
        if "medicine" in domains and "doctor" not in active:
            safety_penalty += 0.65
        if "finance" in domains and "finance_expert" not in active:
            safety_penalty += 0.45
        if "software" in domains and "programmer" not in active:
            safety_penalty += 0.35
        if "algo_designer" not in active:
            safety_penalty += 0.20
        if coverage < 0.5:
            safety_penalty += 0.15
        safety_penalty += self.config.safety_over_size_penalty * max(0, n_active - 5)
        safety_penalty = self._clip(safety_penalty, 0.0, 2.0)

        latency = max(0.05, 1.6 - 0.9 * coord + 0.06 * n_active + 0.02 * e_active)
        q_len = len((question_text or "").split())
        token_cost = 0.12 * n_active + 0.025 * e_active + 0.002 * q_len

        features = {
            "coverage": coverage,
            "coordination": coord,
            "dependency": dep_score,
            "cohesion": cohesion,
            "diversity": diversity,
            "active_ratio": active_ratio,
        }
        return features, safety_penalty, latency, token_cost, domains, n_active, e_active

    def evaluate(
        self,
        dag: DAGState,
        active_agent_ids: Optional[Sequence[str]] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[List[float]] = None,
    ) -> TaskEvaluation:
        features, safety_penalty, latency, token_cost, domains, n_active, e_active = self._compute_features(
            dag, active_agent_ids, question_text
        )
        coverage = features["coverage"]
        coord = features["coordination"]
        dep_score = features["dependency"]
        cohesion = features["cohesion"]
        diversity = features["diversity"]
        active_ratio = features["active_ratio"]

        task_score = (
            0.35 * coverage
            + 0.25 * coord
            + 0.20 * dep_score
            + 0.10 * cohesion
            + 0.10 * diversity
        )
        task_score += 0.05 * active_ratio
        task_score = self._clip(task_score, 0.0, 1.5)

        success = 1.0 if (task_score - 0.35 * safety_penalty) >= self.config.success_threshold else 0.0
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
            custom_metrics={
                "coverage": coverage,
                "coordination": coord,
                "dependency": dep_score,
                "cohesion": cohesion,
                "diversity": diversity,
                "active_agents": float(n_active),
                "active_edges": float(e_active),
            },
        )


@dataclass
class TemplateWeightedEvaluatorConfig(HeuristicEvaluatorConfig):
    learning_rate: float = 0.05
    l2: float = 1e-4
    weight_clip: float = 3.0
    template_weights: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "general": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
            "medicine": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
            "finance": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
            "software": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
            "math": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
            "multi": {
                "coverage": 0.35,
                "coordination": 0.25,
                "dependency": 0.20,
                "cohesion": 0.10,
                "diversity": 0.10,
                "active_ratio": 0.05,
            },
        }
    )
    template_bias: Dict[str, float] = field(default_factory=lambda: {"general": 0.0, "multi": 0.0})


class TemplateWeightedMASTaskEvaluator(HeuristicMASTaskEvaluator):
    def __init__(
        self,
        agent_pool: Optional[AgentPool] = None,
        config: Optional[TemplateWeightedEvaluatorConfig] = None,
    ):
        super().__init__(agent_pool=agent_pool, config=config or TemplateWeightedEvaluatorConfig())
        cfg = self.config
        self._template_weights: Dict[str, Dict[str, float]] = {
            key: dict(val) for key, val in cfg.template_weights.items()
        }
        self._template_bias: Dict[str, float] = dict(cfg.template_bias)

    def _template_key(self, domains: Set[str]) -> str:
        if not domains:
            return "general"
        if len(domains) == 1:
            only = next(iter(domains))
            return only if only in self._template_weights else "general"
        return "multi" if "multi" in self._template_weights else "general"

    def _weights_for(self, key: str) -> Dict[str, float]:
        if key not in self._template_weights:
            self._template_weights[key] = dict(self._template_weights.get("general", {}))
        return self._template_weights[key]

    def evaluate(
        self,
        dag: DAGState,
        active_agent_ids: Optional[Sequence[str]] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[List[float]] = None,
    ) -> TaskEvaluation:
        features, safety_penalty, latency, token_cost, domains, n_active, e_active = self._compute_features(
            dag, active_agent_ids, question_text
        )
        key = self._template_key(domains)
        weights = self._weights_for(key)
        bias = self._template_bias.get(key, 0.0)
        task_score = bias + sum(weights.get(k, 0.0) * v for k, v in features.items())
        task_score = self._clip(task_score, 0.0, 1.5)
        success = 1.0 if (task_score - 0.35 * safety_penalty) >= self.config.success_threshold else 0.0
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
            custom_metrics={
                "coverage": features["coverage"],
                "coordination": features["coordination"],
                "dependency": features["dependency"],
                "cohesion": features["cohesion"],
                "diversity": features["diversity"],
                "active_agents": float(n_active),
                "active_edges": float(e_active),
            },
        )

    def update_from_feedback(
        self,
        dag: DAGState,
        target_score: float,
        active_agent_ids: Optional[Sequence[str]] = None,
        question_text: Optional[str] = None,
    ) -> float:
        features, _, _, _, domains, _, _ = self._compute_features(dag, active_agent_ids, question_text)
        key = self._template_key(domains)
        weights = self._weights_for(key)
        bias = self._template_bias.get(key, 0.0)
        pred = bias + sum(weights.get(k, 0.0) * v for k, v in features.items())
        err = pred - target_score
        lr = self.config.learning_rate
        l2 = self.config.l2
        clip = abs(self.config.weight_clip)
        for k, v in features.items():
            w = weights.get(k, 0.0)
            w = w - lr * (err * v + l2 * w)
            if clip > 0.0:
                w = self._clip(w, -clip, clip)
            weights[k] = w
        bias = bias - lr * err
        self._template_bias[key] = bias
        return pred


@dataclass
class LLMExecutionConfig:
    api_base: str = "http://localhost:8039"
    model: str = ""
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    temperature: float = 0.3
    max_tokens: int = 768
    max_retries: int = 2
    judge_model: Optional[str] = None
    judge_temperature: float = 0.2
    judge_max_tokens: int = 256
    token_cost_per_word: float = 0.00001
    success_threshold: float = 0.6


class LLMExecutionMASTaskEvaluator:
    def __init__(
        self,
        agent_pool: Optional[AgentPool] = None,
        config: Optional[LLMExecutionConfig] = None,
    ):
        self.agent_pool = agent_pool or default_agent_pool()
        self.config = config or LLMExecutionConfig()
        self._agent_by_id = {a.agent_id: a for a in self.agent_pool.agents}

    def cache_signature(self) -> str:
        cfg = self.config
        parts = [cfg.api_base, cfg.model, str(cfg.judge_model or cfg.model)]
        return "llm-exec-v1|" + "|".join(parts)

    def _post_chat(self, messages: List[Dict[str, str]], model: str, temperature: float, max_tokens: int) -> str:
        base = self.config.api_base.rstrip("/")
        url = f"{base}/v1/chat/completions"
        payload = json.dumps(
            {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        if self.config.api_key:
            req.add_header("Authorization", f"Bearer {self.config.api_key}")
        last_err: Optional[Exception] = None
        for _ in range(max(0, int(self.config.max_retries)) + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.config.timeout_s) as resp:
                    body = resp.read().decode("utf-8")
                data = json.loads(body)
                msg = data.get("choices", [{}])[0].get("message", {})
                content = msg.get("content", "")
                if content:
                    return content
            except Exception as exc:
                last_err = exc
        raise RuntimeError(f"LLM request failed: {last_err}")

    def _topo_order(self, dag: DAGState, active: Set[str]) -> Tuple[List[str], Dict[str, List[str]]]:
        nodes = [n for n in dag.nodes if n in active]
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [
            (dag.nodes[src], dag.nodes[dst])
            for src, dst in dag.edges
            if 0 <= src < len(dag.nodes)
            and 0 <= dst < len(dag.nodes)
            and dag.nodes[src] in active
            and dag.nodes[dst] in active
        ]
        indeg = {n: 0 for n in nodes}
        outgoing: Dict[str, List[str]] = {n: [] for n in nodes}
        incoming: Dict[str, List[str]] = {n: [] for n in nodes}
        for src, dst in edges:
            if src in outgoing and dst in indeg:
                outgoing[src].append(dst)
                incoming[dst].append(src)
                indeg[dst] += 1
        queue = [n for n in nodes if indeg[n] == 0]
        order: List[str] = []
        while queue:
            cur = queue.pop(0)
            order.append(cur)
            for nxt in outgoing.get(cur, []):
                indeg[nxt] -= 1
                if indeg[nxt] == 0:
                    queue.append(nxt)
        if len(order) != len(nodes):
            return nodes, incoming
        return order, incoming

    def _safe_json(self, text: str) -> Optional[dict]:
        try:
            return json.loads(text)
        except Exception:
            pass
        if "{" in text and "}" in text:
            snippet = text[text.find("{") : text.rfind("}") + 1]
            try:
                return json.loads(snippet)
            except Exception:
                return None
        return None

    def evaluate(
        self,
        dag: DAGState,
        active_agent_ids: Optional[Sequence[str]] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[List[float]] = None,
    ) -> TaskEvaluation:
        if not self.config.model:
            raise ValueError("LLMExecutionConfig.model is required.")
        active = set(active_agent_ids or dag.nodes)
        order, incoming = self._topo_order(dag, active)
        if not order:
            return TaskEvaluation(task_score=0.0, success=0.0)

        start = time.perf_counter()
        outputs: Dict[str, str] = {}
        total_words = 0
        for aid in order:
            agent = self._agent_by_id.get(aid)
            if agent is None:
                continue
            upstream_ids = incoming.get(aid, [])
            upstream_text = "\n".join(
                f"{uid}: {outputs.get(uid,'')}" for uid in upstream_ids if outputs.get(uid, "")
            )
            user_msg = (
                f"任务：{question_text or ''}\n"
                f"角色：{agent.role}\n"
                f"背景：{agent.profile}\n"
                f"上游输出：\n{upstream_text}\n"
                "请给出你的具体产出。"
            )
            content = self._post_chat(
                messages=[{"role": "system", "content": agent.system_prompt}, {"role": "user", "content": user_msg}],
                model=self.config.model,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            outputs[aid] = content
            total_words += len(content.split())

        sinks = [aid for aid in order if not any(aid in incoming.get(x, []) for x in order)]
        final_output = "\n".join(outputs.get(aid, "") for aid in sinks if outputs.get(aid, ""))
        judge_model = self.config.judge_model or self.config.model
        judge_prompt = (
            "请根据任务和输出给出评分，返回JSON对象，字段为："
            "task_score(0-1), success(0或1), safety_penalty(0-1)。"
        )
        judge_input = f"任务：{question_text or ''}\n输出：\n{final_output}"
        judge_text = self._post_chat(
            messages=[{"role": "system", "content": judge_prompt}, {"role": "user", "content": judge_input}],
            model=judge_model,
            temperature=self.config.judge_temperature,
            max_tokens=self.config.judge_max_tokens,
        )
        parsed = self._safe_json(judge_text) or {}
        task_score = float(parsed.get("task_score", 0.0))
        safety_penalty = float(parsed.get("safety_penalty", 0.0))
        success = float(parsed.get("success", 1.0 if task_score >= self.config.success_threshold else 0.0))

        latency = time.perf_counter() - start
        token_cost = total_words * self.config.token_cost_per_word
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
            custom_metrics={
                "active_agents": float(len(order)),
                "active_edges": float(
                    sum(
                        1
                        for src, dst in dag.edges
                        if 0 <= src < len(dag.nodes)
                        and 0 <= dst < len(dag.nodes)
                        and dag.nodes[src] in active
                        and dag.nodes[dst] in active
                    )
                ),
                "output_words": float(total_words),
            },
        )
