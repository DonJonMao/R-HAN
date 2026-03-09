from mas_gflowopt import MASConfig, MASGFlowPipeline


def main() -> None:
    config = MASConfig(
        num_sampled_dags=30,
        cluster_count=5,
        top_optimized_count=5,
        reward_every_step=True,
        allow_backtracking=False,
    )
    pipeline = MASGFlowPipeline(config=config)
    out = pipeline.run()

    print("sampled_dags:", len(out.sampled_dags))
    print("clusters:", len(out.cluster_seeds))
    print("top_optimized_representations:", len(out.top_optimized_representations))
    print("matched_discrete_dag edges:", len(out.matched_discrete_dag.edges))
    print("refined_best_dag edges:", len(out.refined_best_dag.edges))
    print("refined_best_score:", round(out.refined_best_score, 4))


if __name__ == "__main__":
    main()
