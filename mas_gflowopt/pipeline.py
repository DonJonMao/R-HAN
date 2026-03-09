from __future__ import annotations

from typing import Optional

from .agent_pool import AgentPool, default_agent_pool
from .conditioning import TaskConditioner
from .gflownet import GFlowNetSampler
from .optimizer import ContinuousDiscreteOptimizer
from .representation import GraphRepresentationModel
from .reward import MASTaskEvaluator, MASRewardModel
from .scoring import BICScorer, ScoreModel
from .types import GFlowNetTrainingStats, MASConfig, OptimizationOutput
from .vectorizer import HashingVectorizer, build_openai_embedding_encoder


class MASGFlowPipeline:
    """End-to-end skeleton of MAS + GFlowOpt-like process."""

    def __init__(
        self,
        config: Optional[MASConfig] = None,
        agent_pool: Optional[AgentPool] = None,
        scorer: Optional[ScoreModel] = None,
    ):
        self.config = config or MASConfig()
        self.agent_pool = agent_pool or default_agent_pool()
        self.vectorizer = HashingVectorizer(dim=self.config.embedding_dim)
        if self.config.embedding_api_base and self.config.embedding_model:
            self.vectorizer.set_text_encoder(
                build_openai_embedding_encoder(
                    api_base=self.config.embedding_api_base,
                    model=self.config.embedding_model,
                    timeout_s=self.config.embedding_timeout_s,
                    api_key=self.config.embedding_api_key,
                )
            )
        self.repr_model = GraphRepresentationModel(
            self.vectorizer,
            smooth_alpha=0.70,
            message_steps=self.config.repr_message_steps,
            attention_edge_bias=self.config.repr_attention_edge_bias,
            attention_self_bias=self.config.repr_attention_self_bias,
            seed=self.config.random_seed,
        )
        self.conditioner = TaskConditioner(self.config, self.vectorizer)
        self.scorer = scorer or BICScorer()
        self.reward_model = MASRewardModel(self.config, self.scorer)

        self.sampler = GFlowNetSampler(
            config=self.config,
            repr_model=self.repr_model,
            scorer=self.scorer,
            reward_model=self.reward_model,
        )
        self.optimizer = ContinuousDiscreteOptimizer(
            config=self.config,
            scorer=self.scorer,
        )

    def _should_update_gating(self, stage: str, evaluator: Optional[MASTaskEvaluator]) -> bool:
        if not self.config.enable_learnable_gating:
            return False
        if self.config.gating_require_true_evaluator_for_update and evaluator is None:
            return False

        policy = self.config.gating_update_policy.lower()
        if policy == "none":
            return False
        if stage == "train" and policy not in {"train_only", "train_and_run"}:
            return False
        if stage == "run" and policy != "train_and_run":
            return False

        if stage == "train" and not self.config.gating_update_on_train:
            return False
        if stage == "run" and not self.config.gating_update_on_run:
            return False
        return True

    def _feedback_from_samples(self, sampled_dags) -> float:
        if not sampled_dags:
            return 0.0
        metric = self.config.gating_feedback_metric.lower()

        total_scores = [
            s.reward_breakdown.total_score
            for s in sampled_dags
            if s.reward_breakdown is not None
        ]
        rewards = [s.reward for s in sampled_dags]

        if metric == "mean_reward":
            return sum(rewards) / max(1, len(rewards))

        if metric == "topk_mean_total_score":
            base = total_scores if total_scores else rewards
            if not base:
                return 0.0
            k = max(1, int(len(base) * max(0.0, min(1.0, self.config.gating_feedback_topk_frac))))
            top_vals = sorted(base, reverse=True)[:k]
            return sum(top_vals) / len(top_vals)

        # default: mean_total_score
        if total_scores:
            return sum(total_scores) / len(total_scores)
        return sum(rewards) / max(1, len(rewards))

    def _feedback_from_history(self, history: list[GFlowNetTrainingStats]) -> float:
        if not history:
            return 0.0
        metric = self.config.gating_feedback_metric.lower()
        total_scores = [h.mean_total_score for h in history]
        terminal_rewards = [h.mean_terminal_reward for h in history]

        if metric == "mean_reward":
            return sum(terminal_rewards) / max(1, len(terminal_rewards))

        if metric == "topk_mean_total_score":
            if not total_scores:
                return 0.0
            k = max(
                1,
                int(
                    len(total_scores)
                    * max(0.0, min(1.0, self.config.gating_feedback_topk_frac))
                ),
            )
            top_vals = sorted(total_scores, reverse=True)[:k]
            return sum(top_vals) / len(top_vals)

        return sum(total_scores) / max(1, len(total_scores))

    def train(
        self,
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        agent_top_k: Optional[int] = None,
        task_tag: Optional[str] = None,
    ) -> list[GFlowNetTrainingStats]:
        q_text = question_text if self.config.enable_task_conditioning else None
        full_agent_vectors = self.agent_pool.vectorize(self.vectorizer)
        all_nodes = self.agent_pool.as_nodes()
        cond = self.conditioner.select_agents(
            question_text=q_text,
            agent_vectors=full_agent_vectors,
            agent_ids=all_nodes,
            top_k=agent_top_k,
        )
        nodes = cond.selected_agent_ids
        agent_vectors = {aid: full_agent_vectors[aid] for aid in nodes}
        self.sampler.initialize()
        history = self.sampler.train(
            nodes=nodes,
            agent_vectors=agent_vectors,
            evaluator=evaluator,
            question_text=cond.question_text,
            question_vector=cond.question_vector,
            task_tag=task_tag,
        )
        if self._should_update_gating("train", evaluator) and history:
            self.conditioner.update_from_feedback(
                question_text=cond.question_text,
                question_vector=cond.question_vector,
                selected_agent_ids=nodes,
                feedback_score=self._feedback_from_history(history),
                agent_vectors=full_agent_vectors,
            )
        return history

    def run(
        self,
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        agent_top_k: Optional[int] = None,
        task_tag: Optional[str] = None,
    ) -> OptimizationOutput:
        q_text = question_text if self.config.enable_task_conditioning else None
        full_agent_vectors = self.agent_pool.vectorize(self.vectorizer)
        all_nodes = self.agent_pool.as_nodes()
        cond = self.conditioner.select_agents(
            question_text=q_text,
            agent_vectors=full_agent_vectors,
            agent_ids=all_nodes,
            top_k=agent_top_k,
        )
        nodes = cond.selected_agent_ids
        agent_vectors = {aid: full_agent_vectors[aid] for aid in nodes}

        self.sampler.initialize()
        sampled_dags = self.sampler.sample_batch(
            nodes=nodes,
            agent_vectors=agent_vectors,
            num_dags=self.config.num_sampled_dags,
            evaluator=evaluator,
            question_text=cond.question_text,
            question_vector=cond.question_vector,
            task_tag=task_tag,
        )
        out = self.optimizer.optimize(
            sampled_dags=sampled_dags,
            evaluator=evaluator,
            reward_model=self.reward_model,
            question_text=cond.question_text,
            question_vector=cond.question_vector,
            agent_vectors=agent_vectors,
        )
        if self._should_update_gating("run", evaluator) and sampled_dags:
            feedback = self._feedback_from_samples(sampled_dags)
            self.conditioner.update_from_feedback(
                question_text=cond.question_text,
                question_vector=cond.question_vector,
                selected_agent_ids=nodes,
                feedback_score=feedback,
                agent_vectors=full_agent_vectors,
            )
        return out

    def train_and_run(
        self,
        evaluator: Optional[MASTaskEvaluator] = None,
        question_text: Optional[str] = None,
        agent_top_k: Optional[int] = None,
        task_tag: Optional[str] = None,
    ) -> tuple[list[GFlowNetTrainingStats], OptimizationOutput]:
        history = self.train(
            evaluator=evaluator,
            question_text=question_text,
            agent_top_k=agent_top_k,
            task_tag=task_tag,
        )
        out = self.run(
            evaluator=evaluator,
            question_text=question_text,
            agent_top_k=agent_top_k,
            task_tag=task_tag,
        )
        return history, out
