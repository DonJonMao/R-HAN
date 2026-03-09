from __future__ import annotations

from typing import List, Optional

from .clustering import kmeans
from .discrete_refine import hill_climb_bic, hill_climb_objective, match_nearest_dag
from .proxy import train_proxy
from .reward import MASTaskEvaluator, MASRewardModel
from .scoring import ScoreModel
from .types import (
    ClusterSeed,
    MASConfig,
    OptimizationOutput,
    OptimizedRepresentation,
    ProxyPair,
    SampledDAG,
)


class ContinuousDiscreteOptimizer:
    """Implements the user's MAS adaptation of paper's stage-2/3."""

    def __init__(self, config: MASConfig, scorer: ScoreModel):
        self.config = config
        self.scorer = scorer

    def optimize(
        self,
        sampled_dags: List[SampledDAG],
        evaluator: Optional[MASTaskEvaluator] = None,
        reward_model: Optional[MASRewardModel] = None,
        question_text: Optional[str] = None,
        question_vector: Optional[List[float]] = None,
        agent_vectors: Optional[dict[str, List[float]]] = None,
    ) -> OptimizationOutput:
        if not sampled_dags:
            raise ValueError("No sampled DAGs provided.")

        pairs = [ProxyPair(z=s.z, reward=s.reward, graph=s.graph) for s in sampled_dags]
        proxy = train_proxy(self.config, pairs)

        vectors = [s.z for s in sampled_dags]
        centers, cluster_members = kmeans(
            vectors=vectors,
            k=self.config.cluster_count,
            max_iters=self.config.kmeans_iters,
            seed=self.config.random_seed,
        )

        cluster_seeds: List[ClusterSeed] = []
        optimized: List[OptimizedRepresentation] = []

        for cid, center in enumerate(centers):
            members = cluster_members[cid]
            cluster_seeds.append(ClusterSeed(cluster_id=cid, center_z=center, member_indices=members))
            z_opt = proxy.ascent(
                z0=center,
                steps=self.config.gradient_steps,
                lr=self.config.gradient_lr,
            )
            optimized.append(
                OptimizedRepresentation(
                    cluster_id=cid,
                    start_z=center,
                    optimized_z=z_opt,
                    proxy_score=proxy.predict(z_opt),
                )
            )

        optimized.sort(key=lambda x: x.proxy_score, reverse=True)
        top = optimized[: self.config.top_optimized_count]
        z_best_star = top[0].optimized_z if top else sampled_dags[0].z

        matched = match_nearest_dag(z_best_star, sampled_dags)

        refine_mode = self.config.refine_objective.lower()
        use_composite = (
            reward_model is not None
            and evaluator is not None
            and (refine_mode in {"composite", "auto"} or self.config.use_task_score_as_bic)
        )
        if use_composite:
            def composite_obj(dag):
                rb = reward_model.score_and_reward(
                    dag,
                    evaluator=evaluator,
                    use_true_evaluator=True,
                    question_text=question_text,
                    question_vector=question_vector,
                    agent_vectors=agent_vectors,
                )
                return rb.total_score

            refined, refined_score = hill_climb_objective(
                init_dag=matched,
                objective_fn=composite_obj,
                max_iters=self.config.hc_max_iters,
            )
        else:
            refined, refined_score = hill_climb_bic(
                init_dag=matched,
                scorer=self.scorer,
                max_iters=self.config.hc_max_iters,
            )

        return OptimizationOutput(
            sampled_dags=sampled_dags,
            cluster_seeds=cluster_seeds,
            top_optimized_representations=top,
            z_best_star=z_best_star,
            matched_discrete_dag=matched,
            refined_best_dag=refined,
            refined_best_score=refined_score,
        )
