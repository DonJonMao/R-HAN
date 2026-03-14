from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from demo_train import build_llm_evaluator, run_test, run_train, summarize_metrics
from mas_gflowopt import MASConfig, MASGFlowPipeline
from mas_treesearch import build_processed_datasets, list_processed_datasets, load_processed_split


def build_config(args: argparse.Namespace) -> MASConfig:
    true_eval_interval = args.true_eval_interval
    true_eval_budget = args.true_eval_budget
    true_eval_terminal_always = True
    if args.one_eval_per_trajectory:
        true_eval_interval = 0
        true_eval_budget = 0
        true_eval_terminal_always = True

    return MASConfig(
        gflownet_train_epochs=args.gflownet_train_epochs,
        gflownet_batch_size=args.gflownet_batch_size,
        num_sampled_dags=args.num_sampled_dags,
        contribution_mode=args.contribution_mode,
        embedding_api_base=args.embedding_api_base or None,
        embedding_model=args.embedding_model or None,
        embedding_api_key=args.embedding_api_key or None,
        true_eval_interval=true_eval_interval,
        true_eval_budget_per_trajectory=true_eval_budget,
        true_eval_terminal_always=true_eval_terminal_always,
        enable_refine=not args.disable_refine,
        early_stop_metric=args.early_stop_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup_epochs=args.early_stop_warmup_epochs,
    )


def limit_items(items: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    return items[:limit]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_gflowopt_processed",
        help="Processed per-dataset root. Will be created if missing and --prepare-data is set.",
    )
    parser.add_argument(
        "--unified-root",
        default="/mnt/nvme/projects/R-HAN/dataset/unified_mixed",
        help="Mixed unified root used when --prepare-data is enabled.",
    )
    parser.add_argument("--prepare-data", action="store_true", help="Build processed per-dataset JSONL files first.")
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name to run. Repeatable.")
    parser.add_argument("--all-datasets", action="store_true", help="Run every processed dataset.")
    parser.add_argument("--max-train-per-dataset", type=int, default=1000)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--agent-top-k", type=int, default=6)
    parser.add_argument("--gflownet-train-epochs", type=int, default=5)
    parser.add_argument("--gflownet-batch-size", type=int, default=6)
    parser.add_argument("--num-sampled-dags", type=int, default=20)
    parser.add_argument("--contribution-mode", default="none")
    parser.add_argument("--true-eval-interval", type=int, default=6)
    parser.add_argument("--true-eval-budget", type=int, default=4)
    parser.add_argument("--one-eval-per-trajectory", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--disable-refine", action="store_true")
    parser.add_argument("--batch-eval", action="store_true")
    parser.add_argument("--batch-eval-workers", type=int, default=6)
    parser.add_argument("--batch-eval-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--periodic-test-size", type=int, default=100)
    parser.add_argument("--early-stop-metric", default="total_loss")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--early-stop-warmup-epochs", type=int, default=1)
    parser.add_argument("--embedding-api-base", default="")
    parser.add_argument("--embedding-model", default="")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/outputs/mas_gflowopt_dataset_train",
        help="Directory where per-dataset summaries will be written.",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    if args.prepare_data or not data_root.exists():
        manifest = build_processed_datasets(args.unified_root, args.data_root)
        print(f"[prepare] wrote processed datasets to {args.data_root}", flush=True)
        print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2), flush=True)

    datasets = list(args.dataset)
    if args.all_datasets or not datasets:
        datasets = list_processed_datasets(args.data_root)
    if not datasets:
        raise ValueError("No datasets selected. Use --dataset or --all-datasets.")

    cfg = build_config(args)
    evaluator = build_llm_evaluator(args.batch_eval, args.batch_eval_workers)

    ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    output_root = Path(args.output_root) / ts
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_config.json").write_text(
        json.dumps(vars(args), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_report: dict[str, Any] = {
        "generated_at": ts,
        "data_root": args.data_root,
        "datasets": {},
    }

    for dataset_name in datasets:
        dataset_dir = output_root / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)

        all_train = load_processed_split(args.data_root, dataset_name, "train")
        all_test = load_processed_split(args.data_root, dataset_name, "test")
        train_limit = min(len(all_train), max(0, int(args.max_train_per_dataset)))
        test_limit = min(
            len(all_test),
            max(1, int(train_limit * max(0.0, args.test_ratio))) if train_limit > 0 else 0,
        )
        train_samples = limit_items(all_train, train_limit)
        test_samples = limit_items(all_test, test_limit)
        periodic_test_samples = test_samples
        if args.periodic_test_size > 0:
            periodic_test_samples = test_samples[: args.periodic_test_size]

        print(
            f"\n=== Dataset: {dataset_name} train={len(train_samples)} test={len(test_samples)} ===",
            flush=True,
        )
        dataset_start = time.time()
        pipeline = MASGFlowPipeline(config=cfg)
        cumulative_metrics: list[dict[str, float]] = []
        cumulative_by_dataset: dict[str, dict[str, float]] = {}
        cumulative_by_category: dict[str, dict[str, float]] = {}

        def run_periodic_test(step_idx: int) -> None:
            if not periodic_test_samples:
                return
            print(f"[test] start_eval dataset={dataset_name} train_items={step_idx}", flush=True)
            run_test(
                pipeline,
                evaluator,
                periodic_test_samples,
                args.agent_top_k,
                0,
                args.batch_eval_size,
                True,
                cumulative_metrics,
                cumulative_by_dataset,
                cumulative_by_category,
                "cumulative",
            )

        train_rewards = run_train(
            pipeline,
            evaluator,
            train_samples,
            args.agent_top_k,
            args.log_every,
            args.batch_eval_size,
            args.eval_every,
            run_periodic_test if args.eval_every > 0 else None,
        )
        train_seconds = time.time() - dataset_start
        train_summary = {
            "count": len(train_samples),
            "avg_reward": sum(train_rewards) / max(1, len(train_rewards)),
            "seconds": train_seconds,
            "avg_s": train_seconds / max(1, len(train_samples)),
        }
        print(f"[train-dataset-summary] {json.dumps(train_summary, ensure_ascii=False)}", flush=True)

        test_metrics = run_test(
            pipeline,
            evaluator,
            periodic_test_samples,
            args.agent_top_k,
            0,
            args.batch_eval_size,
            True,
            cumulative_metrics,
            cumulative_by_dataset,
            cumulative_by_category,
            "cumulative",
        )
        test_summary = summarize_metrics(test_metrics)
        dataset_report = {
            "dataset": dataset_name,
            "train_limit": train_limit,
            "test_limit": test_limit,
            "train_summary": train_summary,
            "test_summary": test_summary,
        }
        (dataset_dir / "report.json").write_text(
            json.dumps(dataset_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        final_report["datasets"][dataset_name] = dataset_report

    final_path = output_root / "report.json"
    final_path.write_text(json.dumps(final_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== Final Report ===", flush=True)
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)
    print(f"[report] wrote {final_path}", flush=True)


if __name__ == "__main__":
    main()
