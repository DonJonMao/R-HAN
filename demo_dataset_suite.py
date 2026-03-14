from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mas_treesearch import (
    SearchConfig,
    TieredEvalConfig,
    TreeSearchMASPipeline,
    list_processed_datasets,
    load_processed_split,
)


SPLITS = ("train", "validation", "test")
METRIC_KEYS = (
    "reward",
    "task_score",
    "success",
    "latency",
    "token_cost",
    "safety_penalty",
)


@dataclass
class RunningStats:
    count: int = 0
    reward: float = 0.0
    task_score: float = 0.0
    success: float = 0.0
    latency: float = 0.0
    token_cost: float = 0.0
    safety_penalty: float = 0.0

    def add(self, row: Dict[str, float]) -> None:
        self.count += 1
        for key in METRIC_KEYS:
            setattr(self, key, getattr(self, key) + float(row[key]))

    def mean_dict(self) -> Dict[str, float]:
        if self.count <= 0:
            return {key: 0.0 for key in METRIC_KEYS}
        return {
            key: float(getattr(self, key) / self.count)
            for key in METRIC_KEYS
        }


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _safe_dataset_dir(output_root: Path, dataset_name: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in dataset_name)
    return output_root / safe


def _mean(values: List[float]) -> float:
    return sum(values) / max(1, len(values))


def _load_dataset_pool(data_root: str, dataset_name: str) -> tuple[List[Dict[str, Any]], Dict[str, int]]:
    items: List[Dict[str, Any]] = []
    split_sizes: Dict[str, int] = {}
    for split in SPLITS:
        split_items = load_processed_split(data_root, dataset_name, split)
        split_sizes[split] = len(split_items)
        items.extend(split_items)
    return items, split_sizes


def _sample_and_split(
    items: List[Dict[str, Any]],
    *,
    max_pool_size: int,
    seed: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, int]]:
    rng = random.Random(seed)
    shuffled = list(items)
    rng.shuffle(shuffled)
    used_all = len(shuffled) <= max_pool_size
    selected = shuffled if used_all else shuffled[:max_pool_size]

    if len(selected) <= 1:
        return selected, [], {
            "selected_count": len(selected),
            "train_count": len(selected),
            "test_count": 0,
            "used_all": int(used_all),
        }

    test_count = max(1, len(selected) // 11)
    if test_count >= len(selected):
        test_count = 1
    test_items = selected[:test_count]
    train_items = selected[test_count:]
    return train_items, test_items, {
        "selected_count": len(selected),
        "train_count": len(train_items),
        "test_count": len(test_items),
        "used_all": int(used_all),
    }


def _row_from_result(
    dataset_name: str,
    split: str,
    item: Dict[str, Any],
    result: Any,
) -> Dict[str, Any]:
    summary = result.best_node.tier2
    if summary is None:
        raise RuntimeError(f"No tier2 summary for dataset={dataset_name} id={item.get('id', '')}")
    return {
        "id": item.get("id", ""),
        "split": split,
        "dataset": dataset_name,
        "category": item.get("category", ""),
        "reward": float(summary.mean_reward),
        "task_score": float(summary.mean_task_score),
        "success": float(summary.mean_success),
        "latency": float(summary.mean_latency),
        "token_cost": float(summary.mean_token_cost),
        "safety_penalty": float(summary.mean_safety_penalty),
        "signature": result.best_node.compiled.signature(),
        "output": summary.evaluations[0].raw_output if summary.evaluations else "",
    }


def _summary_from_rows(name: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "name": name,
        "count": len(rows),
        "avg_reward": _mean([float(row["reward"]) for row in rows]),
        "avg_task_score": _mean([float(row["task_score"]) for row in rows]),
        "avg_success": _mean([float(row["success"]) for row in rows]),
        "avg_latency": _mean([float(row["latency"]) for row in rows]),
        "avg_token_cost": _mean([float(row["token_cost"]) for row in rows]),
        "avg_safety_penalty": _mean([float(row["safety_penalty"]) for row in rows]),
    }


def _periodic_eval_items(
    test_items: List[Dict[str, Any]],
    *,
    eval_round: int,
    eval_size: int,
) -> List[Dict[str, Any]]:
    if not test_items:
        return []
    size = min(eval_size, len(test_items))
    start = (eval_round * size) % len(test_items)
    if start + size <= len(test_items):
        return test_items[start : start + size]
    overflow = start + size - len(test_items)
    return test_items[start:] + test_items[:overflow]


def _print_running(prefix: str, stats: RunningStats, *, elapsed_s: float, extra: str = "") -> None:
    metrics = stats.mean_dict()
    msg = (
        f"{prefix} count={stats.count} "
        f"avg_reward={metrics['reward']:.4f} "
        f"avg_task={metrics['task_score']:.4f} "
        f"avg_success={metrics['success']:.4f} "
        f"avg_latency={metrics['latency']:.4f} "
        f"avg_token={metrics['token_cost']:.4f} "
        f"avg_safety={metrics['safety_penalty']:.4f} "
        f"elapsed_s={elapsed_s:.1f}"
    )
    if extra:
        msg += f" {extra}"
    print(msg, flush=True)


def _run_dataset(
    dataset_name: str,
    *,
    data_root: str,
    output_root: Path,
    search_config: SearchConfig,
    runtime_config: TieredEvalConfig,
    max_pool_size: int,
    split_seed: int,
    log_every: int,
    periodic_eval_every: int,
    periodic_eval_size: int,
) -> Dict[str, Any]:
    pipeline = TreeSearchMASPipeline(
        search_config=search_config,
        runtime_config=runtime_config,
    )
    dataset_dir = _safe_dataset_dir(output_root, dataset_name)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    all_items, original_split_sizes = _load_dataset_pool(data_root, dataset_name)
    train_items, test_items, sampled_meta = _sample_and_split(
        all_items,
        max_pool_size=max_pool_size,
        seed=split_seed,
    )

    print(
        f"\n=== Dataset: {dataset_name} | original={len(all_items)} "
        f"selected={sampled_meta['selected_count']} train={len(train_items)} test={len(test_items)} ===",
        flush=True,
    )

    train_stats = RunningStats()
    periodic_test_stats = RunningStats()
    train_rows: List[Dict[str, Any]] = []
    periodic_test_rows: List[Dict[str, Any]] = []
    start_time = time.time()
    eval_round = 0

    for idx, item in enumerate(train_items, start=1):
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=item.get("metadata"),
            dataset_name=item.get("source_dataset"),
            learn=True,
        )
        row = _row_from_result(dataset_name, "train", item, result)
        train_rows.append(row)
        train_stats.add(row)

        if log_every > 0 and (idx % log_every == 0 or idx == len(train_items)):
            elapsed_s = time.time() - start_time
            extra = f"progress={idx}/{len(train_items)} avg_s_per_item={elapsed_s / max(1, idx):.2f}"
            _print_running(f"[train][{dataset_name}]", train_stats, elapsed_s=elapsed_s, extra=extra)

        if periodic_eval_every > 0 and test_items and idx % periodic_eval_every == 0:
            eval_items = _periodic_eval_items(
                test_items,
                eval_round=eval_round,
                eval_size=periodic_eval_size,
            )
            eval_round += 1
            for eval_item in eval_items:
                eval_result = pipeline.search(
                    eval_item["question"],
                    reference_answer=eval_item.get("answer"),
                    metadata=eval_item.get("metadata"),
                    dataset_name=eval_item.get("source_dataset"),
                    learn=False,
                )
                eval_row = _row_from_result(dataset_name, "periodic_test", eval_item, eval_result)
                periodic_test_rows.append(eval_row)
                periodic_test_stats.add(eval_row)
            elapsed_s = time.time() - start_time
            extra = f"after_train={idx} window={len(eval_items)}"
            _print_running(
                f"[periodic-test][{dataset_name}]",
                periodic_test_stats,
                elapsed_s=elapsed_s,
                extra=extra,
            )

    checkpoint_path = dataset_dir / "checkpoint.json"
    pipeline.save_checkpoint(
        str(checkpoint_path),
        metadata={
            "dataset": dataset_name,
            "saved_at": _utc_now(),
            "selected_count": sampled_meta["selected_count"],
            "train_count": len(train_items),
            "test_count": len(test_items),
            "original_split_sizes": original_split_sizes,
        },
    )
    print(f"[checkpoint][{dataset_name}] {checkpoint_path}", flush=True)

    final_test_rows: List[Dict[str, Any]] = []
    final_test_stats = RunningStats()
    final_test_start = time.time()
    for idx, item in enumerate(test_items, start=1):
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=item.get("metadata"),
            dataset_name=item.get("source_dataset"),
            learn=False,
        )
        row = _row_from_result(dataset_name, "final_test", item, result)
        final_test_rows.append(row)
        final_test_stats.add(row)
        if log_every > 0 and (idx % log_every == 0 or idx == len(test_items)):
            elapsed_s = time.time() - final_test_start
            extra = f"progress={idx}/{len(test_items)} avg_s_per_item={elapsed_s / max(1, idx):.2f}"
            _print_running(f"[final-test][{dataset_name}]", final_test_stats, elapsed_s=elapsed_s, extra=extra)

    dataset_report = {
        "dataset": dataset_name,
        "saved_at": _utc_now(),
        "original_split_sizes": original_split_sizes,
        "sampling": sampled_meta,
        "train_summary": _summary_from_rows("train", train_rows),
        "periodic_test_summary": _summary_from_rows("periodic_test", periodic_test_rows),
        "final_test_summary": _summary_from_rows("final_test", final_test_rows),
        "checkpoint_path": str(checkpoint_path),
        "train_examples": train_rows[: min(5, len(train_rows))],
        "periodic_test_examples": periodic_test_rows[: min(5, len(periodic_test_rows))],
        "final_test_examples": final_test_rows[: min(5, len(final_test_rows))],
    }

    report_path = dataset_dir / "report.json"
    report_path.write_text(json.dumps(dataset_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[report][{dataset_name}] {report_path}", flush=True)
    return dataset_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset name to run. Repeatable.")
    parser.add_argument("--all-datasets", action="store_true", help="Run every dataset under --data-root.")
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/outputs/mas_treesearch_dataset_suite",
    )
    parser.add_argument("--max-pool-size", type=int, default=1000)
    parser.add_argument("--split-seed", type=int, default=7)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--periodic-eval-every", type=int, default=100)
    parser.add_argument("--periodic-eval-size", type=int, default=10)
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
    args = parser.parse_args()

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
    runtime_config.chat.api_base = os.getenv("LLM_API_BASE", runtime_config.chat.api_base)
    runtime_config.chat.model = os.getenv("LLM_MODEL") or runtime_config.chat.model
    runtime_config.embedding.api_base = os.getenv("EMBED_API_BASE", runtime_config.embedding.api_base)
    runtime_config.embedding.model = os.getenv("EMBED_MODEL", runtime_config.embedding.model)
    runtime_config.tier1.max_tokens = args.tier1_max_tokens
    runtime_config.tier2.max_tokens = args.tier2_max_tokens
    runtime_config.tier1.repeats = args.tier1_repeats
    runtime_config.tier2.repeats = args.tier2_repeats
    runtime_config.debug_judge = args.debug_judge

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    run_report = {
        "started_at": _utc_now(),
        "data_root": args.data_root,
        "output_root": str(output_root),
        "datasets": {},
        "search_config": asdict(search_config),
        "runtime_config": asdict(runtime_config),
        "effective_env": {
            "LLM_API_BASE": os.getenv("LLM_API_BASE", ""),
            "EMBED_API_BASE": os.getenv("EMBED_API_BASE", ""),
            "LLM_MODEL": os.getenv("LLM_MODEL", ""),
            "EMBED_MODEL": os.getenv("EMBED_MODEL", ""),
            "LLM_MAX_TOKENS": os.getenv("LLM_MAX_TOKENS", ""),
            "LLM_JUDGE_MAX_TOKENS": os.getenv("LLM_JUDGE_MAX_TOKENS", ""),
        },
    }

    for dataset_name in datasets:
        dataset_report = _run_dataset(
            dataset_name,
            data_root=args.data_root,
            output_root=output_root,
            search_config=search_config,
            runtime_config=runtime_config,
            max_pool_size=args.max_pool_size,
            split_seed=args.split_seed,
            log_every=args.log_every,
            periodic_eval_every=args.periodic_eval_every,
            periodic_eval_size=args.periodic_eval_size,
        )
        run_report["datasets"][dataset_name] = dataset_report

        suite_report_path = output_root / "suite_report.json"
        suite_report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")

    run_report["finished_at"] = _utc_now()
    suite_report_path = output_root / "suite_report.json"
    suite_report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[suite-report] {suite_report_path}", flush=True)
    print(json.dumps(run_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
