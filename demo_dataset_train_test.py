from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, List, Optional

from mas_treesearch import (
    SearchConfig,
    TieredEvalConfig,
    TreeSearchMASPipeline,
    build_processed_datasets,
    list_processed_datasets,
    load_processed_split,
)


@dataclass
class PhaseSummary:
    split: str
    count: int
    avg_reward: float
    avg_task_score: float
    avg_success: float
    avg_latency: float
    avg_token_cost: float


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _sample_items(items: List[dict[str, Any]], limit: int, seed: int, shuffle: bool) -> List[dict[str, Any]]:
    if shuffle:
        rng = random.Random(seed)
        items = list(items)
        rng.shuffle(items)
    if limit >= 0:
        return items[:limit]
    return items


def _summarize_results(split: str, rows: List[dict[str, float]]) -> PhaseSummary:
    return PhaseSummary(
        split=split,
        count=len(rows),
        avg_reward=_mean([row["reward"] for row in rows]),
        avg_task_score=_mean([row["task_score"] for row in rows]),
        avg_success=_mean([row["success"] for row in rows]),
        avg_latency=_mean([row["latency"] for row in rows]),
        avg_token_cost=_mean([row["token_cost"] for row in rows]),
    )


def _run_phase(
    pipeline: TreeSearchMASPipeline,
    items: Iterable[dict[str, Any]],
    *,
    split: str,
    learn: bool,
    log_every: int,
) -> tuple[PhaseSummary, List[dict[str, Any]]]:
    rows: List[dict[str, Any]] = []
    items_list = list(items)
    for idx, item in enumerate(items_list, start=1):
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=item.get("metadata"),
            dataset_name=item.get("source_dataset"),
            learn=learn,
        )
        summary = result.best_node.tier2
        if summary is None:
            raise RuntimeError(f"No tier2 summary for {item.get('id', '')}")
        row = {
            "id": item.get("id", ""),
            "split": split,
            "dataset": item.get("source_dataset", ""),
            "category": item.get("category", ""),
            "reward": float(summary.mean_reward),
            "task_score": float(summary.mean_task_score),
            "success": float(summary.mean_success),
            "latency": float(summary.mean_latency),
            "token_cost": float(summary.mean_token_cost),
            "signature": result.best_node.compiled.signature(),
            "output": summary.evaluations[0].raw_output if summary.evaluations else "",
        }
        rows.append(row)
        if log_every > 0 and (idx == 1 or idx % log_every == 0 or idx == len(items_list)):
            print(
                f"[{split}] {idx}/{len(items_list)} "
                f"id={row['id']} reward={row['reward']:.4f} "
                f"task={row['task_score']:.4f} success={row['success']:.4f}",
                flush=True,
            )
    return _summarize_results(split, rows), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed",
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
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--test-split", default="test")
    parser.add_argument("--max-train", type=int, default=50)
    parser.add_argument("--max-test", type=int, default=50)
    parser.add_argument("--train-seed", type=int, default=7)
    parser.add_argument("--test-seed", type=int, default=17)
    parser.add_argument("--train-shuffle", action="store_true")
    parser.add_argument("--test-shuffle", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--search-iterations", type=int, default=8)
    parser.add_argument("--candidate-core-k", type=int, default=4)
    parser.add_argument("--candidate-explore-k", type=int, default=2)
    parser.add_argument("--candidate-max-k", type=int, default=6)
    parser.add_argument("--tier1-max-tokens", type=int, default=192)
    parser.add_argument("--tier2-max-tokens", type=int, default=512)
    parser.add_argument("--tier1-repeats", type=int, default=1)
    parser.add_argument("--tier2-repeats", type=int, default=2)
    parser.add_argument("--disable-learned-prior", action="store_true")
    parser.add_argument("--disable-learned-value", action="store_true")
    parser.add_argument("--debug-judge", action="store_true")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--report-file", default="", help="Optional JSON report path.")
    args = parser.parse_args()

    if args.prepare_data or not Path(args.data_root).exists():
        manifest = build_processed_datasets(args.unified_root, args.data_root)
        print(f"[prepare] wrote processed datasets to {args.data_root}", flush=True)
        print(json.dumps(manifest["datasets"], ensure_ascii=False, indent=2), flush=True)

    datasets = list(args.dataset)
    if args.all_datasets or not datasets:
        datasets = list_processed_datasets(args.data_root)
    if not datasets:
        raise ValueError("No datasets selected. Use --dataset or --all-datasets.")

    search_config = SearchConfig(
        search_iterations=args.search_iterations,
        candidate_core_k=args.candidate_core_k,
        candidate_explore_k=args.candidate_explore_k,
        candidate_max_k=args.candidate_max_k,
        enable_learned_edit_prior=not args.disable_learned_prior,
        enable_learned_value_model=not args.disable_learned_value,
    )
    runtime_config = TieredEvalConfig()
    runtime_config.tier1.max_tokens = args.tier1_max_tokens
    runtime_config.tier2.max_tokens = args.tier2_max_tokens
    runtime_config.tier1.repeats = args.tier1_repeats
    runtime_config.tier2.repeats = args.tier2_repeats
    runtime_config.debug_judge = args.debug_judge

    report: dict[str, Any] = {"datasets": {}}
    for dataset_name in datasets:
        print(f"\n=== Dataset: {dataset_name} ===", flush=True)
        pipeline = TreeSearchMASPipeline(
            search_config=search_config,
            runtime_config=runtime_config,
        )
        dataset_report: dict[str, Any] = {}

        if not args.skip_train:
            train_items = load_processed_split(args.data_root, dataset_name, args.train_split)
            train_items = _sample_items(train_items, args.max_train, args.train_seed, args.train_shuffle)
            train_summary, train_rows = _run_phase(
                pipeline,
                train_items,
                split=args.train_split,
                learn=True,
                log_every=args.log_every,
            )
            dataset_report["train"] = asdict(train_summary)
            dataset_report["train_samples"] = train_rows[: min(5, len(train_rows))]
            print(f"[train-summary] {json.dumps(asdict(train_summary), ensure_ascii=False)}", flush=True)

        if not args.skip_test:
            test_items = load_processed_split(args.data_root, dataset_name, args.test_split)
            test_items = _sample_items(test_items, args.max_test, args.test_seed, args.test_shuffle)
            test_summary, test_rows = _run_phase(
                pipeline,
                test_items,
                split=args.test_split,
                learn=False,
                log_every=args.log_every,
            )
            dataset_report["test"] = asdict(test_summary)
            dataset_report["test_samples"] = test_rows[: min(5, len(test_rows))]
            print(f"[test-summary] {json.dumps(asdict(test_summary), ensure_ascii=False)}", flush=True)

        report["datasets"][dataset_name] = dataset_report

    if args.report_file:
        report_path = Path(args.report_file)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[report] wrote {report_path}", flush=True)

    print("\n=== Final Report ===", flush=True)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
