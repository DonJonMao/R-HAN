from __future__ import annotations

from mas_gflowopt import MASConfig, MASGFlowPipeline, TaskEvaluation
from mas_gflowopt.reward import MASTaskEvaluator
from mas_gflowopt.types import DAGState


class ToyMASEvaluator(MASTaskEvaluator):
    """Small deterministic evaluator for local smoke tests."""

    def evaluate(self, dag: DAGState, active_agent_ids=None, question_text=None, question_vector=None) -> TaskEvaluation:
        active = set(active_agent_ids or dag.nodes)
        active_ratio = len(active) / max(1, len(dag.nodes))

        # Reward edges that connect active agents only.
        active_edges = 0
        for src, dst in dag.edges:
            if dag.nodes[src] in active and dag.nodes[dst] in active:
                active_edges += 1

        task_score = 0.15 * active_edges + 0.30 * active_ratio
        success = 1.0 if active_edges >= 2 else 0.0
        latency = max(0.0, 1.2 - 0.08 * active_edges)
        token_cost = 0.2 + 0.05 * len(active)
        safety_penalty = 0.0
        return TaskEvaluation(
            task_score=task_score,
            success=success,
            latency=latency,
            token_cost=token_cost,
            safety_penalty=safety_penalty,
        )


def main() -> None:
    cfg = MASConfig(
        gflownet_train_epochs=5,
        gflownet_batch_size=6,
        num_sampled_dags=20,
        contribution_mode="loo",
    )
    pipeline = MASGFlowPipeline(config=cfg)
    evaluator = ToyMASEvaluator()

    history, out = pipeline.train_and_run(
        evaluator=evaluator,
        question_text="Design a medically safe and cost-aware diagnosis workflow.",
        agent_top_k=4,
    )
    print("epochs:", len(history))
    print("last_total_loss:", round(history[-1].total_loss, 6))
    print("last_mean_reward:", round(history[-1].mean_terminal_reward, 6))
    print("sampled_dags:", len(out.sampled_dags))
    print("refined_best_score:", round(out.refined_best_score, 6))


if __name__ == "__main__":
    main()
