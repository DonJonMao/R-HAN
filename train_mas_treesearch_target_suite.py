from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mas_treesearch import SearchConfig, TieredEvalConfig, TreeSearchMASPipeline, load_processed_split


DEFAULT_DATASET_PLAN: Dict[str, Dict[str, int]] = {
    "multiarith": {"max_train": 493, "max_validation": 57, "max_test": 50, "periodic_every": 25, "periodic_size": 10},
    "humaneval": {"max_train": 125, "max_validation": 23, "max_test": 16, "periodic_every": 20, "periodic_size": 8},
    "mbpp": {"max_train": 500, "max_validation": 100, "max_test": 93, "periodic_every": 50, "periodic_size": 20},
    "gsm8k": {"max_train": 800, "max_validation": 200, "max_test": 200, "periodic_every": 100, "periodic_size": 20},
    "mmlu": {"max_train": 1000, "max_validation": 300, "max_test": 300, "periodic_every": 100, "periodic_size": 30},
    "nlgraph": {"max_train": 600, "max_validation": 150, "max_test": 150, "periodic_every": 50, "periodic_size": 20},
    "knowledge_crosswords": {"max_train": 500, "max_validation": 150, "max_test": 150, "periodic_every": 50, "periodic_size": 20},
    "normad": {"max_train": 800, "max_validation": 200, "max_test": 200, "periodic_every": 100, "periodic_size": 20},
    "math": {"max_train": 600, "max_validation": 150, "max_test": 200, "periodic_every": 75, "periodic_size": 20},
}


@dataclass
class RunningStats:
    count: int = 0
    reward: float = 0.0
    task_score: float = 0.0
    success: float = 0.0
    latency: float = 0.0
    token_cost: float = 0.0
    safety_penalty: float = 0.0

    def add(self, row: Dict[str, Any]) -> None:
        self.count += 1
        self.reward += float(row["reward"])
        self.task_score += float(row["task_score"])
        self.success += float(row["success"])
        self.latency += float(row["latency"])
        self.token_cost += float(row["token_cost"])
        self.safety_penalty += float(row["safety_penalty"])

    def mean_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "reward": 0.0,
                "task_score": 0.0,
                "success": 0.0,
                "latency": 0.0,
                "token_cost": 0.0,
                "safety_penalty": 0.0,
            }
        return {
            "reward": self.reward / self.count,
            "task_score": self.task_score / self.count,
            "success": self.success / self.count,
            "latency": self.latency / self.count,
            "token_cost": self.token_cost / self.count,
            "safety_penalty": self.safety_penalty / self.count,
        }

    def state_dict(self) -> Dict[str, float]:
        return {
            "count": int(self.count),
            "reward": float(self.reward),
            "task_score": float(self.task_score),
            "success": float(self.success),
            "latency": float(self.latency),
            "token_cost": float(self.token_cost),
            "safety_penalty": float(self.safety_penalty),
        }

    @classmethod
    def from_state_dict(cls, payload: Dict[str, Any] | None) -> "RunningStats":
        payload = payload or {}
        return cls(
            count=int(payload.get("count", 0)),
            reward=float(payload.get("reward", 0.0)),
            task_score=float(payload.get("task_score", 0.0)),
            success=float(payload.get("success", 0.0)),
            latency=float(payload.get("latency", 0.0)),
            token_cost=float(payload.get("token_cost", 0.0)),
            safety_penalty=float(payload.get("safety_penalty", 0.0)),
        )


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_json_dict(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(items: List[Dict[str, Any]], limit: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    picked = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(picked)
    if limit >= 0:
        return picked[:limit]
    return picked


def _sampled_split_ids(items: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("id", "")) for item in items]


def _validate_resume_ids(current_items: List[Dict[str, Any]], saved_ids: List[str], *, dataset_name: str, split: str) -> None:
    current_ids = _sampled_split_ids(current_items)
    if saved_ids and current_ids != list(saved_ids):
        raise ValueError(
            f"Resume split mismatch for dataset={dataset_name} split={split}: "
            f"current sampled ids differ from checkpoint."
        )


def _row_from_result(dataset_name: str, split: str, item: Dict[str, Any], result: Any) -> Dict[str, Any]:
    summary = result.best_node.tier2
    if summary is None:
        raise RuntimeError(f"No tier2 summary for dataset={dataset_name} split={split} id={item.get('id', '')}")
    signature = result.best_node.compiled.signature()
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
        "signature": signature,
        "output": summary.evaluations[0].raw_output if summary.evaluations else "",
    }


def _print_row(prefix: str, idx: int, total: int, row: Dict[str, Any], *, elapsed_s: float) -> None:
    print(
        f"{prefix} {idx}/{total} "
        f"id={row['id']} reward={row['reward']:.4f} task={row['task_score']:.4f} "
        f"success={row['success']:.4f} latency={row['latency']:.2f} token={row['token_cost']:.4f} "
        f"safety={row['safety_penalty']:.4f} elapsed_s={elapsed_s:.1f} signature={row['signature']}",
        flush=True,
    )


def _print_summary(prefix: str, stats: RunningStats, *, elapsed_s: float) -> None:
    metrics = stats.mean_dict()
    print(
        f"{prefix} count={stats.count} avg_reward={metrics['reward']:.4f} "
        f"avg_task={metrics['task_score']:.4f} avg_success={metrics['success']:.4f} "
        f"avg_latency={metrics['latency']:.2f} avg_token={metrics['token_cost']:.4f} "
        f"avg_safety={metrics['safety_penalty']:.4f} elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _periodic_window(items: List[Dict[str, Any]], *, round_idx: int, window: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    size = min(window, len(items))
    start = (round_idx * size) % len(items)
    if start + size <= len(items):
        return items[start : start + size]
    overflow = start + size - len(items)
    return items[start:] + items[:overflow]


def _runtime_config_dict(runtime_config: TieredEvalConfig) -> Dict[str, Any]:
    return asdict(runtime_config)


def _run_eval_phase(
    pipeline: TreeSearchMASPipeline,
    dataset_name: str,
    items: Iterable[Dict[str, Any]],
    *,
    split: str,
) -> tuple[List[Dict[str, Any]], RunningStats]:
    rows: List[Dict[str, Any]] = []
    stats = RunningStats()
    start_time = time.time()
    items_list = list(items)
    for idx, item in enumerate(items_list, start=1):
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=item.get("metadata"),
            dataset_name=item.get("source_dataset"),
            learn=False,
        )
        row = _row_from_result(dataset_name, split, item, result)
        rows.append(row)
        stats.add(row)
        _print_row(f"[{split}][{dataset_name}]", idx, len(items_list), row, elapsed_s=time.time() - start_time)
    _print_summary(f"[{split}-summary][{dataset_name}]", stats, elapsed_s=time.time() - start_time)
    return rows, stats


def _checkpoint_metadata(
    *,
    dataset_name: str,
    seed: int,
    plan: Dict[str, int],
    effective_runtime: Dict[str, Any],
    train_items: List[Dict[str, Any]],
    validation_items: List[Dict[str, Any]],
    test_items: List[Dict[str, Any]],
    train_rows: List[Dict[str, Any]],
    periodic_validation_rows: List[Dict[str, Any]],
    train_stats: RunningStats,
    periodic_validation_stats: RunningStats,
    train_index_completed: int,
    validation_round: int,
    train_phase_complete: bool,
    phase: str,
) -> Dict[str, Any]:
    return {
        "dataset": dataset_name,
        "saved_at": _utc_now(),
        "seed": int(seed),
        "plan": dict(plan),
        "effective_runtime_config": effective_runtime,
        "train_ids": _sampled_split_ids(train_items),
        "validation_ids": _sampled_split_ids(validation_items),
        "test_ids": _sampled_split_ids(test_items),
        "train_rows": list(train_rows),
        "periodic_validation_rows": list(periodic_validation_rows),
        "train_stats": train_stats.state_dict(),
        "periodic_validation_stats": periodic_validation_stats.state_dict(),
        "train_index_completed": int(train_index_completed),
        "validation_round": int(validation_round),
        "train_phase_complete": bool(train_phase_complete),
        "phase": phase,
    }


def _save_dataset_checkpoint(
    pipeline: TreeSearchMASPipeline,
    checkpoint_path: Path,
    *,
    metadata: Dict[str, Any],
) -> None:
    pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
    print(f"[checkpoint][{metadata['dataset']}] {checkpoint_path}", flush=True)


def _suite_progress_path(output_root: Path) -> Path:
    return output_root / "suite_progress.json"


def _run_dataset(
    dataset_name: str,
    *,
    data_root: str,
    output_root: Path,
    search_config: SearchConfig,
    runtime_config: TieredEvalConfig,
    plan: Dict[str, int],
    seed: int,
    resume: bool,
    checkpoint_every: int,
) -> Dict[str, Any]:
    train_items = _sample(load_processed_split(data_root, dataset_name, "train"), plan["max_train"], seed, True)
    validation_items = _sample(
        load_processed_split(data_root, dataset_name, "validation"),
        plan["max_validation"],
        seed + 1,
        True,
    )
    test_items = _sample(load_processed_split(data_root, dataset_name, "test"), plan["max_test"], seed + 2, True)
    if not train_items:
        raise ValueError(f"No train items found for dataset={dataset_name}")

    dataset_output_root = output_root / dataset_name
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = dataset_output_root / "checkpoint.json"

    pipeline = TreeSearchMASPipeline(search_config=search_config, runtime_config=runtime_config)
    effective_runtime = _runtime_config_dict(pipeline.runtime_config)

    train_rows: List[Dict[str, Any]] = []
    periodic_validation_rows: List[Dict[str, Any]] = []
    train_stats = RunningStats()
    periodic_validation_stats = RunningStats()
    train_index_completed = 0
    validation_round = 0
    train_phase_complete = False

    if resume and checkpoint_path.exists():
        metadata = pipeline.load_checkpoint(str(checkpoint_path))
        _validate_resume_ids(train_items, list(metadata.get("train_ids", [])), dataset_name=dataset_name, split="train")
        _validate_resume_ids(
            validation_items,
            list(metadata.get("validation_ids", [])),
            dataset_name=dataset_name,
            split="validation",
        )
        _validate_resume_ids(test_items, list(metadata.get("test_ids", [])), dataset_name=dataset_name, split="test")
        train_rows = list(metadata.get("train_rows", []))
        periodic_validation_rows = list(metadata.get("periodic_validation_rows", []))
        train_stats = RunningStats.from_state_dict(metadata.get("train_stats"))
        periodic_validation_stats = RunningStats.from_state_dict(metadata.get("periodic_validation_stats"))
        train_index_completed = int(metadata.get("train_index_completed", 0))
        validation_round = int(metadata.get("validation_round", 0))
        train_phase_complete = bool(metadata.get("train_phase_complete", False))
        print(
            f"[resume][{dataset_name}] checkpoint={checkpoint_path} "
            f"train_index_completed={train_index_completed} validation_round={validation_round} "
            f"train_phase_complete={train_phase_complete}",
            flush=True,
        )

    print(
        f"\n=== DATASET {dataset_name} train={len(train_items)} validation={len(validation_items)} test={len(test_items)} ===",
        flush=True,
    )
    print(f"[plan][{dataset_name}] {json.dumps(plan, ensure_ascii=False)}", flush=True)
    print(f"[runtime][{dataset_name}] {json.dumps(effective_runtime, ensure_ascii=False)}", flush=True)

    train_start = time.time()
    if not checkpoint_path.exists():
        metadata = _checkpoint_metadata(
            dataset_name=dataset_name,
            seed=seed,
            plan=plan,
            effective_runtime=effective_runtime,
            train_items=train_items,
            validation_items=validation_items,
            test_items=test_items,
            train_rows=train_rows,
            periodic_validation_rows=periodic_validation_rows,
            train_stats=train_stats,
            periodic_validation_stats=periodic_validation_stats,
            train_index_completed=train_index_completed,
            validation_round=validation_round,
            train_phase_complete=train_phase_complete,
            phase="init",
        )
        _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=metadata)

    if not train_phase_complete:
        train_slice = train_items[train_index_completed:]
        base_train_index = train_index_completed
        for offset, item in enumerate(train_slice, start=1):
            idx = base_train_index + offset
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
            train_index_completed = idx
            _print_row(f"[train][{dataset_name}]", idx, len(train_items), row, elapsed_s=time.time() - train_start)

            should_run_periodic = validation_items and (idx % plan["periodic_every"] == 0 or idx == len(train_items))
            if should_run_periodic:
                window = _periodic_window(
                    validation_items,
                    round_idx=validation_round,
                    window=plan["periodic_size"],
                )
                validation_round += 1
                for eval_idx, eval_item in enumerate(window, start=1):
                    eval_result = pipeline.search(
                        eval_item["question"],
                        reference_answer=eval_item.get("answer"),
                        metadata=eval_item.get("metadata"),
                        dataset_name=eval_item.get("source_dataset"),
                        learn=False,
                    )
                    eval_row = _row_from_result(dataset_name, "periodic_validation", eval_item, eval_result)
                    periodic_validation_rows.append(eval_row)
                    periodic_validation_stats.add(eval_row)
                    _print_row(
                        f"[periodic-validation][{dataset_name}]",
                        eval_idx,
                        len(window),
                        eval_row,
                        elapsed_s=time.time() - train_start,
                    )
                _print_summary(
                    f"[periodic-validation-summary][{dataset_name}]",
                    periodic_validation_stats,
                    elapsed_s=time.time() - train_start,
                )

            checkpoint_due = checkpoint_every > 0 and idx % checkpoint_every == 0
            train_complete = idx == len(train_items)
            if checkpoint_due or train_complete:
                metadata = _checkpoint_metadata(
                    dataset_name=dataset_name,
                    seed=seed,
                    plan=plan,
                    effective_runtime=effective_runtime,
                    train_items=train_items,
                    validation_items=validation_items,
                    test_items=test_items,
                    train_rows=train_rows,
                    periodic_validation_rows=periodic_validation_rows,
                    train_stats=train_stats,
                    periodic_validation_stats=periodic_validation_stats,
                    train_index_completed=train_index_completed,
                    validation_round=validation_round,
                    train_phase_complete=train_complete,
                    phase="train",
                )
                _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=metadata)

        train_phase_complete = True

    _print_summary(f"[train-summary][{dataset_name}]", train_stats, elapsed_s=time.time() - train_start)

    metadata = _checkpoint_metadata(
        dataset_name=dataset_name,
        seed=seed,
        plan=plan,
        effective_runtime=effective_runtime,
        train_items=train_items,
        validation_items=validation_items,
        test_items=test_items,
        train_rows=train_rows,
        periodic_validation_rows=periodic_validation_rows,
        train_stats=train_stats,
        periodic_validation_stats=periodic_validation_stats,
        train_index_completed=train_index_completed,
        validation_round=validation_round,
        train_phase_complete=True,
        phase="post_train",
    )
    _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=metadata)

    validation_rows, validation_stats = _run_eval_phase(
        pipeline,
        dataset_name,
        validation_items,
        split="validation",
    )
    test_rows, test_stats = _run_eval_phase(
        pipeline,
        dataset_name,
        test_items,
        split="test",
    )

    report = {
        "dataset": dataset_name,
        "saved_at": _utc_now(),
        "plan": plan,
        "effective_runtime_config": effective_runtime,
        "train_summary": train_stats.mean_dict() | {"count": train_stats.count},
        "periodic_validation_summary": periodic_validation_stats.mean_dict() | {"count": periodic_validation_stats.count},
        "validation_summary": validation_stats.mean_dict() | {"count": validation_stats.count},
        "test_summary": test_stats.mean_dict() | {"count": test_stats.count},
        "checkpoint_path": str(checkpoint_path),
        "train_examples": train_rows[: min(5, len(train_rows))],
        "periodic_validation_examples": periodic_validation_rows[: min(5, len(periodic_validation_rows))],
        "validation_examples": validation_rows[: min(5, len(validation_rows))],
        "test_examples": test_rows[: min(5, len(test_rows))],
    }
    report_path = dataset_output_root / "report.json"
    _write_json(report_path, report)
    print(f"[report][{dataset_name}] {report_path}", flush=True)

    metadata = _checkpoint_metadata(
        dataset_name=dataset_name,
        seed=seed,
        plan=plan,
        effective_runtime=effective_runtime,
        train_items=train_items,
        validation_items=validation_items,
        test_items=test_items,
        train_rows=train_rows,
        periodic_validation_rows=periodic_validation_rows,
        train_stats=train_stats,
        periodic_validation_stats=periodic_validation_stats,
        train_index_completed=train_index_completed,
        validation_round=validation_round,
        train_phase_complete=True,
        phase="completed",
    )
    _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=metadata)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_target_suite_20260314",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/outputs/mas_treesearch_target_suite_train_20260314",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to run. Repeatable.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--search-iterations", type=int, default=6)
    parser.add_argument("--candidate-core-k", type=int, default=4)
    parser.add_argument("--candidate-explore-k", type=int, default=2)
    parser.add_argument("--candidate-max-k", type=int, default=6)
    parser.add_argument("--tier1-max-tokens", type=int, default=160)
    parser.add_argument("--tier2-max-tokens", type=int, default=384)
    parser.add_argument("--tier1-repeats", type=int, default=1)
    parser.add_argument("--tier2-repeats", type=int, default=1)
    parser.add_argument("--disable-learned-prior", action="store_true")
    parser.add_argument("--disable-learned-value", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    datasets = list(args.dataset) if args.dataset else list(DEFAULT_DATASET_PLAN.keys())
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = _suite_progress_path(output_root)

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

    suite_report = _load_json_dict(output_root / "suite_report.json") if args.resume else None
    if suite_report is None:
        suite_report = {
            "started_at": _utc_now(),
            "data_root": args.data_root,
            "output_root": str(output_root),
            "datasets": {},
            "search_config": asdict(search_config),
            "requested_runtime_config": asdict(runtime_config),
            "effective_runtime_config": None,
        }
    else:
        suite_report.pop("finished_at", None)

    suite_progress = _load_json_dict(progress_path) if args.resume else None
    if suite_progress is None:
        suite_progress = {
            "started_at": suite_report.get("started_at", _utc_now()),
            "data_root": args.data_root,
            "output_root": str(output_root),
            "datasets": {},
        }
        for dataset_name, dataset_report in dict(suite_report.get("datasets", {})).items():
            suite_progress["datasets"][dataset_name] = {
                "status": "completed",
                "seed": None,
                "checkpoint_path": str((output_root / dataset_name / "checkpoint.json")),
                "updated_at": dataset_report.get("saved_at", _utc_now()),
            }
    else:
        suite_progress.pop("finished_at", None)

    for index, dataset_name in enumerate(datasets):
        existing = suite_progress["datasets"].get(dataset_name, {})
        if existing.get("status") == "completed":
            print(f"[resume][{dataset_name}] already completed, skipping", flush=True)
            continue

        plan = dict(DEFAULT_DATASET_PLAN[dataset_name])
        suite_progress["datasets"][dataset_name] = {
            "status": "running",
            "seed": args.seed + index * 100,
            "checkpoint_path": str(output_root / dataset_name / "checkpoint.json"),
            "updated_at": _utc_now(),
        }
        _write_json(progress_path, suite_progress)

        report = _run_dataset(
            dataset_name,
            data_root=args.data_root,
            output_root=output_root,
            search_config=search_config,
            runtime_config=runtime_config,
            plan=plan,
            seed=args.seed + index * 100,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        )
        suite_report["datasets"][dataset_name] = report
        if suite_report["effective_runtime_config"] is None:
            suite_report["effective_runtime_config"] = report.get("effective_runtime_config")
        _write_json(output_root / "suite_report.json", suite_report)

        suite_progress["datasets"][dataset_name] = {
            "status": "completed",
            "seed": args.seed + index * 100,
            "checkpoint_path": report.get("checkpoint_path", ""),
            "updated_at": _utc_now(),
        }
        _write_json(progress_path, suite_progress)

    suite_report["finished_at"] = _utc_now()
    suite_report_path = output_root / "suite_report.json"
    _write_json(suite_report_path, suite_report)
    suite_progress["finished_at"] = _utc_now()
    _write_json(progress_path, suite_progress)
    print(f"[suite-report] {suite_report_path}", flush=True)
    print(json.dumps(suite_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
