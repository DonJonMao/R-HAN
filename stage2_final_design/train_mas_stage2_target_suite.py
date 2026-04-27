from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from mas_stage2 import (
    Stage2MASPipeline,
    Stage2RuntimeConfig,
    Stage2V2Config,
    load_prepared_stage1_artifact,
    save_prepared_stage1_artifact,
)
from mas_treesearch import (
    SearchConfig,
    TieredEvalConfig,
    UnionRuntimeConfig,
    filter_stage2_supported_items,
    list_processed_datasets,
    load_processed_split,
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
    stage2_turns: float = 0.0
    stage2_memory_records: float = 0.0
    structure_reward: float = 0.0
    coverage: float = 0.0
    complementarity: float = 0.0
    redundancy_quality: float = 0.0

    def add(self, row: Dict[str, Any]) -> None:
        self.count += 1
        self.reward += float(row["reward"])
        self.task_score += float(row["task_score"])
        self.success += float(row["success"])
        self.latency += float(row["latency"])
        self.token_cost += float(row["token_cost"])
        self.safety_penalty += float(row["safety_penalty"])
        self.stage2_turns += float(row.get("stage2_turns", 0.0))
        self.stage2_memory_records += float(row.get("stage2_memory_records", 0.0))
        self.structure_reward += float(row.get("structure_reward", 0.0))
        self.coverage += float(row.get("coverage", 0.0))
        self.complementarity += float(row.get("complementarity", 0.0))
        self.redundancy_quality += float(row.get("redundancy_quality", 0.0))

    def mean_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "reward": 0.0,
                "task_score": 0.0,
                "success": 0.0,
                "latency": 0.0,
                "token_cost": 0.0,
                "safety_penalty": 0.0,
                "stage2_turns": 0.0,
                "stage2_memory_records": 0.0,
                "structure_reward": 0.0,
                "coverage": 0.0,
                "complementarity": 0.0,
                "redundancy_quality": 0.0,
            }
        return {
            "reward": self.reward / self.count,
            "task_score": self.task_score / self.count,
            "success": self.success / self.count,
            "latency": self.latency / self.count,
            "token_cost": self.token_cost / self.count,
            "safety_penalty": self.safety_penalty / self.count,
            "stage2_turns": self.stage2_turns / self.count,
            "stage2_memory_records": self.stage2_memory_records / self.count,
            "structure_reward": self.structure_reward / self.count,
            "coverage": self.coverage / self.count,
            "complementarity": self.complementarity / self.count,
            "redundancy_quality": self.redundancy_quality / self.count,
        }

    def state_dict(self) -> Dict[str, float]:
        return {
            "count": float(self.count),
            "reward": float(self.reward),
            "task_score": float(self.task_score),
            "success": float(self.success),
            "latency": float(self.latency),
            "token_cost": float(self.token_cost),
            "safety_penalty": float(self.safety_penalty),
            "stage2_turns": float(self.stage2_turns),
            "stage2_memory_records": float(self.stage2_memory_records),
            "structure_reward": float(self.structure_reward),
            "coverage": float(self.coverage),
            "complementarity": float(self.complementarity),
            "redundancy_quality": float(self.redundancy_quality),
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
            stage2_turns=float(payload.get("stage2_turns", 0.0)),
            stage2_memory_records=float(payload.get("stage2_memory_records", 0.0)),
            structure_reward=float(payload.get("structure_reward", 0.0)),
            coverage=float(payload.get("coverage", 0.0)),
            complementarity=float(payload.get("complementarity", 0.0)),
            redundancy_quality=float(payload.get("redundancy_quality", 0.0)),
        )


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_json_dict(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _sample(items: List[Dict[str, Any]], limit: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    picked = list(items)
    if shuffle:
        random.Random(seed).shuffle(picked)
    if limit >= 0:
        return picked[:limit]
    return picked


def _load_stage2_supported_split(data_root: str, dataset_name: str, split: str) -> List[Dict[str, Any]]:
    items = load_processed_split(data_root, dataset_name, split)
    return filter_stage2_supported_items(dataset_name, items)


def _sampled_split_ids(items: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("id", "")) for item in items]


def _safe_item_key(value: Any) -> str:
    text = str(value if value is not None else "sample")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _item_dataset_name(dataset_name: str, item: Dict[str, Any]) -> str:
    value = item.get("source_dataset")
    return str(value or dataset_name)


def _item_metadata(dataset_name: str, item: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    metadata = item.get("metadata")
    resolved = dict(metadata) if isinstance(metadata, dict) else {}
    resolved.setdefault("mas_dataset_name", _item_dataset_name(dataset_name, item))
    if item.get("id") is not None:
        resolved.setdefault("id", item.get("id"))
    return resolved or None


def _structure_artifact_path(root: Path, dataset_name: str, split: str, item: Dict[str, Any]) -> Path:
    return root / dataset_name / split / f"{_safe_item_key(item.get('id', 'sample'))}.json"


def _validate_resume_ids(current_items: List[Dict[str, Any]], saved_ids: List[str], *, dataset_name: str, split: str) -> None:
    current_ids = _sampled_split_ids(current_items)
    if saved_ids and current_ids != list(saved_ids):
        raise ValueError(
            f"Resume split mismatch for dataset={dataset_name} split={split}: current sampled ids differ from checkpoint."
        )


def _print_row(prefix: str, idx: int, total: int, row: Dict[str, Any], *, elapsed_s: float) -> None:
    turn_token_costs = row.get("stage2_turn_token_costs", [])
    turn_token_text = json.dumps(turn_token_costs, ensure_ascii=False, separators=(",", ":")) if isinstance(turn_token_costs, list) else "[]"
    parts = [
        f"{prefix} {idx}/{total}",
        f"id={row['id']}",
        f"reward={row['reward']:.4f}",
        f"task={row['task_score']:.4f}",
        f"success={row['success']:.4f}",
        f"latency={row['latency']:.2f}",
        f"token={row['token_cost']:.4f}",
        f"s1_token={row.get('stage1_token_cost', 0.0):.4f}",
        f"s2_token={row.get('stage2_token_cost', 0.0):.4f}",
        f"s2_turn_token={turn_token_text}",
        f"s2_turns={row.get('stage2_turns', 0):.1f}",
        f"s2_mem={row.get('stage2_memory_records', 0):.1f}",
        f"select={row.get('selection_decision', '')}",
        f"elapsed_s={elapsed_s:.1f}",
        f"signature={row['signature']}",
    ]
    print(" ".join(parts), flush=True)


def _print_summary(prefix: str, stats: RunningStats, *, elapsed_s: float) -> None:
    metrics = stats.mean_dict()
    parts = [
        prefix,
        f"count={stats.count}",
        f"avg_reward={metrics['reward']:.4f}",
        f"avg_task={metrics['task_score']:.4f}",
        f"avg_success={metrics['success']:.4f}",
        f"avg_latency={metrics['latency']:.2f}",
        f"avg_token={metrics['token_cost']:.4f}",
        f"avg_s2_turns={metrics['stage2_turns']:.2f}",
        f"avg_s2_mem={metrics['stage2_memory_records']:.2f}",
        f"avg_struct_reward={metrics['structure_reward']:.4f}",
        f"avg_coverage={metrics['coverage']:.4f}",
        f"avg_complementarity={metrics['complementarity']:.4f}",
        f"avg_redundancy={metrics['redundancy_quality']:.4f}",
        f"elapsed_s={elapsed_s:.1f}",
    ]
    print(" ".join(parts), flush=True)


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


def _stage2_config_dict(stage2_config: Any) -> Dict[str, Any]:
    return asdict(stage2_config)


def _checkpoint_metadata(
    *,
    dataset_name: str,
    seed: int,
    plan: Dict[str, int],
    effective_runtime: Dict[str, Any],
    effective_stage2_config: Dict[str, Any],
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
        "effective_stage2_config": effective_stage2_config,
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


def _run_stage2_search(
    pipeline: Stage2MASPipeline,
    dataset_name: str,
    item: Dict[str, Any],
    *,
    split: str,
    replay_dir: Optional[str],
    structure_cache_root: Optional[Path],
    learn: bool,
) -> Any:
    resolved_dataset = _item_dataset_name(dataset_name, item)
    metadata = _item_metadata(dataset_name, item)
    allow_stage1_fallback = bool(learn and split == "train")
    artifact_path = None
    if structure_cache_root is not None:
        artifact_path = _structure_artifact_path(structure_cache_root, resolved_dataset, split, item)
        if artifact_path.exists():
            prepared = load_prepared_stage1_artifact(str(artifact_path))
            if prepared.question_text != item["question"]:
                prepared = pipeline.prepare_stage1_structure(
                    item["question"],
                    reference_answer=item.get("answer"),
                    metadata=metadata,
                    dataset_name=resolved_dataset,
                    learn=False,
                )
                prepared.metadata["source"] = "live_stage1_search_cached"
                save_prepared_stage1_artifact(prepared, str(artifact_path))
            else:
                prepared.metadata["source"] = "stage1_structure_cache"
            result = pipeline.search_prepared(
                item["question"],
                prepared_structure=prepared,
                reference_answer=item.get("answer"),
                metadata=metadata,
                dataset_name=resolved_dataset,
                replay_dir=replay_dir,
                learn=learn,
                allow_stage1_fallback=allow_stage1_fallback,
            )
        else:
            prepared = pipeline.prepare_stage1_structure(
                item["question"],
                reference_answer=item.get("answer"),
                metadata=metadata,
                dataset_name=resolved_dataset,
                learn=False,
            )
            prepared.metadata["source"] = "live_stage1_search_cached"
            save_prepared_stage1_artifact(prepared, str(artifact_path))
            result = pipeline.search_prepared(
                item["question"],
                prepared_structure=prepared,
                reference_answer=item.get("answer"),
                metadata=metadata,
                dataset_name=resolved_dataset,
                replay_dir=replay_dir,
                learn=learn,
                allow_stage1_fallback=allow_stage1_fallback,
            )
    else:
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=metadata,
            dataset_name=resolved_dataset,
            learn=learn,
            replay_dir=replay_dir,
            allow_stage1_fallback=allow_stage1_fallback,
        )
    if artifact_path is not None:
        result.stage2_result.metadata["structure_artifact_path"] = str(artifact_path)
    return result


def _row_from_result(dataset_name: str, split: str, item: Dict[str, Any], result: Any) -> Dict[str, Any]:
    summary = result.final_summary
    structure = result.stage1_result.structure_summary if result.stage1_result is not None else result.stage1_artifact.structure_summary
    row = {
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
        "signature": result.final_signature,
        "output": result.final_output,
        "stage1_signature": result.stage1_artifact.stage1_signature,
        "stage2_signature": result.stage2_result.signature,
        "selection_decision": str(result.stage2_result.metadata.get("selection_decision", "")),
        "selection_reason": str(result.stage2_result.metadata.get("selection_reason", "")),
        "final_selection_decision": str(result.stage2_result.metadata.get("final_selection_decision", "")),
        "final_selection_reason": str(result.stage2_result.metadata.get("final_selection_reason", "")),
        "fallback_applied": float(bool(result.stage2_result.metadata.get("fallback_applied", False))),
        "finalizer_strategy": str(result.stage2_result.metadata.get("finalizer_strategy", "")),
        "structure_source": str(result.stage2_result.metadata.get("structure_source", result.stage1_artifact.metadata.get("source", ""))),
        "stage2_turns": float(result.stage2_result.metadata.get("turn_count", 0)),
        "stage2_memory_records": float(sum(result.stage2_result.memory_record_counts.values())),
        "stage1_reward": float(result.stage2_result.metadata.get("stage1_reward", 0.0)),
        "stage2_reward": float(result.stage2_result.metadata.get("stage2_reward", 0.0)),
        "stage1_task_score": float(result.stage2_result.metadata.get("stage1_task_score", 0.0)),
        "stage2_task_score": float(result.stage2_result.metadata.get("stage2_task_score", 0.0)),
        "stage1_success": float(result.stage2_result.metadata.get("stage1_success", 0.0)),
        "stage2_success": float(result.stage2_result.metadata.get("stage2_success", 0.0)),
        "stage1_latency": float(result.stage2_result.metadata.get("stage1_latency", 0.0)),
        "stage2_latency": float(result.stage2_result.metadata.get("stage2_latency", 0.0)),
        "stage1_token_cost": float(result.stage2_result.metadata.get("stage1_token_cost", 0.0)),
        "stage2_token_cost": float(result.stage2_result.metadata.get("stage2_token_cost", 0.0)),
        "stage2_turn_token_costs": list(result.stage2_result.metadata.get("turn_token_costs", [])),
        "stage2_turn_token_estimates": list(result.stage2_result.metadata.get("turn_token_estimates", [])),
    }
    if structure is not None:
        row["structure_reward"] = float(structure.metrics.total_reward)
        row["coverage"] = float(structure.metrics.coverage)
        row["complementarity"] = float(structure.metrics.complementarity)
        row["redundancy_quality"] = float(structure.metrics.redundancy_quality)
    return row


def _run_eval_phase(
    pipeline: Stage2MASPipeline,
    dataset_name: str,
    items: Iterable[Dict[str, Any]],
    *,
    split: str,
    save_replays: bool,
    replay_root: Path,
    structure_cache_root: Optional[Path],
) -> tuple[List[Dict[str, Any]], RunningStats]:
    rows: List[Dict[str, Any]] = []
    stats = RunningStats()
    start_time = time.time()
    items_list = list(items)
    for idx, item in enumerate(items_list, start=1):
        replay_dir = str(replay_root / split / str(item.get("id", idx))) if save_replays else None
        result = _run_stage2_search(
            pipeline,
            dataset_name,
            item,
            split=split,
            replay_dir=replay_dir,
            structure_cache_root=structure_cache_root,
            learn=False,
        )
        row = _row_from_result(dataset_name, split, item, result)
        rows.append(row)
        stats.add(row)
        _print_row(f"[{split}][{dataset_name}]", idx, len(items_list), row, elapsed_s=time.time() - start_time)
    _print_summary(f"[{split}-summary][{dataset_name}]", stats, elapsed_s=time.time() - start_time)
    return rows, stats


def _resolve_dataset_plan(
    data_root: str,
    dataset_name: str,
    *,
    periodic_every: int,
    periodic_size: int,
    max_train: int,
    max_validation: int,
    max_test: int,
) -> Dict[str, int]:
    train_count = len(_load_stage2_supported_split(data_root, dataset_name, "train"))
    validation_count = len(_load_stage2_supported_split(data_root, dataset_name, "validation"))
    test_count = len(_load_stage2_supported_split(data_root, dataset_name, "test"))
    return {
        "max_train": train_count if max_train < 0 else min(max_train, train_count),
        "max_validation": validation_count if max_validation < 0 else min(max_validation, validation_count),
        "max_test": test_count if max_test < 0 else min(max_test, test_count),
        "periodic_every": periodic_every,
        "periodic_size": periodic_size,
    }


def _run_dataset(
    dataset_name: str,
    *,
    data_root: str,
    output_root: Path,
    search_config: SearchConfig,
    runtime_config: TieredEvalConfig,
    union_config: UnionRuntimeConfig,
    stage2_config: Any,
    plan: Dict[str, int],
    seed: int,
    resume: bool,
    checkpoint_every: int,
    save_replays: bool,
    stage1_checkpoint_root: str,
    structure_cache_enabled: bool,
) -> Dict[str, Any]:
    train_items = _sample(
        _load_stage2_supported_split(data_root, dataset_name, "train"),
        plan["max_train"],
        seed,
        True,
    )
    validation_items = _sample(
        _load_stage2_supported_split(data_root, dataset_name, "validation"),
        plan["max_validation"],
        seed + 1,
        True,
    )
    test_items = _sample(
        _load_stage2_supported_split(data_root, dataset_name, "test"),
        plan["max_test"],
        seed + 2,
        True,
    )
    if not train_items:
        raise ValueError(f"No train items found for dataset={dataset_name}")

    dataset_output_root = output_root / dataset_name
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    replay_root = dataset_output_root / "replays"
    checkpoint_path = dataset_output_root / "checkpoint.json"
    structure_cache_root = dataset_output_root / "stage1_structures" if structure_cache_enabled else None

    pipeline = Stage2MASPipeline(
        search_config=search_config,
        runtime_config=runtime_config,
        union_config=union_config,
        stage2_config=stage2_config,
    )
    if stage1_checkpoint_root:
        stage1_checkpoint = Path(stage1_checkpoint_root) / dataset_name / "checkpoint.json"
        if stage1_checkpoint.exists():
            pipeline._stage1.load_checkpoint(str(stage1_checkpoint))
            print(f"[stage1-checkpoint][{dataset_name}] {stage1_checkpoint}", flush=True)
    effective_runtime = _runtime_config_dict(pipeline.runtime_config)
    effective_stage2_config = _stage2_config_dict(stage2_config)

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
        _validate_resume_ids(validation_items, list(metadata.get("validation_ids", [])), dataset_name=dataset_name, split="validation")
        _validate_resume_ids(test_items, list(metadata.get("test_ids", [])), dataset_name=dataset_name, split="test")
        train_rows = list(metadata.get("train_rows", []))
        periodic_validation_rows = list(metadata.get("periodic_validation_rows", []))
        train_stats = RunningStats.from_state_dict(metadata.get("train_stats"))
        periodic_validation_stats = RunningStats.from_state_dict(metadata.get("periodic_validation_stats"))
        train_index_completed = int(metadata.get("train_index_completed", 0))
        validation_round = int(metadata.get("validation_round", 0))
        train_phase_complete = bool(metadata.get("train_phase_complete", False))
        print(
            f"[resume][{dataset_name}] checkpoint={checkpoint_path} train_index_completed={train_index_completed} "
            f"validation_round={validation_round} train_phase_complete={train_phase_complete}",
            flush=True,
        )

    print(
        f"\n=== STAGE2 DATASET {dataset_name} train={len(train_items)} validation={len(validation_items)} test={len(test_items)} ===",
        flush=True,
    )
    print(f"[plan][{dataset_name}] {json.dumps(plan, ensure_ascii=False)}", flush=True)
    print(f"[runtime][{dataset_name}] {json.dumps(effective_runtime, ensure_ascii=False)}", flush=True)
    print(f"[stage2-config][{dataset_name}] {json.dumps(effective_stage2_config, ensure_ascii=False)}", flush=True)

    if not checkpoint_path.exists():
        metadata = _checkpoint_metadata(
            dataset_name=dataset_name,
            seed=seed,
            plan=plan,
            effective_runtime=effective_runtime,
            effective_stage2_config=effective_stage2_config,
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
        pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)

    train_start = time.time()
    if not train_phase_complete:
        train_slice = train_items[train_index_completed:]
        base_train_index = train_index_completed
        for offset, item in enumerate(train_slice, start=1):
            idx = base_train_index + offset
            replay_dir = str(replay_root / "train" / str(item.get("id", idx))) if save_replays else None
            result = _run_stage2_search(
                pipeline,
                dataset_name,
                item,
                split="train",
                replay_dir=replay_dir,
                structure_cache_root=structure_cache_root,
                learn=True,
            )
            row = _row_from_result(dataset_name, "train", item, result)
            train_rows.append(row)
            train_stats.add(row)
            train_index_completed = idx
            _print_row(f"[train][{dataset_name}]", idx, len(train_items), row, elapsed_s=time.time() - train_start)

            should_run_periodic = validation_items and (idx % plan["periodic_every"] == 0 or idx == len(train_items))
            if should_run_periodic:
                window = _periodic_window(validation_items, round_idx=validation_round, window=plan["periodic_size"])
                validation_round += 1
                for eval_idx, eval_item in enumerate(window, start=1):
                    eval_replay_dir = str(replay_root / "periodic_validation" / str(eval_item.get("id", eval_idx))) if save_replays else None
                    eval_result = _run_stage2_search(
                        pipeline,
                        dataset_name,
                        eval_item,
                        split="periodic_validation",
                        replay_dir=eval_replay_dir,
                        structure_cache_root=structure_cache_root,
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
                    effective_stage2_config=effective_stage2_config,
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
                pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
        train_phase_complete = True

    _print_summary(f"[train-summary][{dataset_name}]", train_stats, elapsed_s=time.time() - train_start)

    metadata = _checkpoint_metadata(
        dataset_name=dataset_name,
        seed=seed,
        plan=plan,
        effective_runtime=effective_runtime,
        effective_stage2_config=effective_stage2_config,
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
    pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)

    validation_rows, validation_stats = _run_eval_phase(
        pipeline,
        dataset_name,
        validation_items,
        split="validation",
        save_replays=save_replays,
        replay_root=replay_root,
        structure_cache_root=structure_cache_root,
    )
    test_rows, test_stats = _run_eval_phase(
        pipeline,
        dataset_name,
        test_items,
        split="test",
        save_replays=save_replays,
        replay_root=replay_root,
        structure_cache_root=structure_cache_root,
    )

    report = {
        "dataset": dataset_name,
        "saved_at": _utc_now(),
        "plan": plan,
        "effective_runtime_config": effective_runtime,
        "effective_stage2_config": effective_stage2_config,
        "train_summary": train_stats.mean_dict() | {"count": train_stats.count},
        "periodic_validation_summary": periodic_validation_stats.mean_dict() | {"count": periodic_validation_stats.count},
        "validation_summary": validation_stats.mean_dict() | {"count": validation_stats.count},
        "test_summary": test_stats.mean_dict() | {"count": test_stats.count},
        "checkpoint_path": str(checkpoint_path),
        "structure_cache_root": str(structure_cache_root) if structure_cache_root is not None else "",
        "train_examples": train_rows[: min(5, len(train_rows))],
        "periodic_validation_examples": periodic_validation_rows[: min(5, len(periodic_validation_rows))],
        "validation_examples": validation_rows[: min(5, len(validation_rows))],
        "test_examples": test_rows[: min(5, len(test_rows))],
    }
    report_path = dataset_output_root / "report.json"
    _write_json(report_path, report)
    _write_jsonl(dataset_output_root / "train_rows.jsonl", train_rows)
    _write_jsonl(dataset_output_root / "periodic_validation_rows.jsonl", periodic_validation_rows)
    _write_jsonl(dataset_output_root / "validation_rows.jsonl", validation_rows)
    _write_jsonl(dataset_output_root / "test_rows.jsonl", test_rows)
    print(f"[report][{dataset_name}] {report_path}", flush=True)

    metadata = _checkpoint_metadata(
        dataset_name=dataset_name,
        seed=seed,
        plan=plan,
        effective_runtime=effective_runtime,
        effective_stage2_config=effective_stage2_config,
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
    pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
    return report


def _suite_progress_path(output_root: Path) -> Path:
    return output_root / "suite_progress.json"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_stage2_target_suite_from_stage1_test",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/outputs/mas_stage2_target_suite_train",
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
    parser.add_argument("--stage2-turn-count", type=int, default=5)
    parser.add_argument("--stage2-version", choices=("v1", "v2"), default="v2")
    parser.add_argument("--memory-top-k", type=int, default=4)
    parser.add_argument("--soft-prune-top-k", type=int, default=3)
    parser.add_argument("--soft-prune-threshold", type=float, default=0.38)
    parser.add_argument("--hard-prune-after-turn", type=int, default=4)
    parser.add_argument("--v2-latent-length", type=int, default=8)
    parser.add_argument("--v2-gnn-layers", type=int, default=2)
    parser.add_argument("--disable-v2-global-node", action="store_true")
    parser.add_argument("--disable-v2-lmpo", action="store_true")
    parser.add_argument("--periodic-every", type=int, default=50)
    parser.add_argument("--periodic-size", type=int, default=20)
    parser.add_argument("--max-train", type=int, default=-1)
    parser.add_argument("--max-validation", type=int, default=-1)
    parser.add_argument("--max-test", type=int, default=-1)
    parser.add_argument("--save-replays", action="store_true")
    parser.add_argument("--stage1-checkpoint-root", default="")
    parser.add_argument("--disable-structure-cache", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    datasets = list(args.dataset) if args.dataset else list_processed_datasets(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = _suite_progress_path(output_root)

    search_config = SearchConfig(
        search_iterations=args.search_iterations,
        candidate_core_k=args.candidate_core_k,
        candidate_explore_k=args.candidate_explore_k,
        candidate_max_k=args.candidate_max_k,
    )
    runtime_config = TieredEvalConfig()
    runtime_config.tier1.max_tokens = args.tier1_max_tokens
    runtime_config.tier2.max_tokens = args.tier2_max_tokens
    runtime_config.tier1.repeats = args.tier1_repeats
    runtime_config.tier2.repeats = args.tier2_repeats
    union_config = UnionRuntimeConfig()
    if args.stage2_version == "v2":
        stage2_config = Stage2V2Config()
        stage2_config.composer_latent_length = args.v2_latent_length
        stage2_config.gnn_num_layers = args.v2_gnn_layers
        stage2_config.global_node_enabled = not args.disable_v2_global_node
        stage2_config.lmpo_enabled = not args.disable_v2_lmpo
    else:
        stage2_config = Stage2RuntimeConfig()
    stage2_config.graph.turn_count = args.stage2_turn_count
    stage2_config.memory.max_selected_records = args.memory_top_k
    stage2_config.graph.soft_prune_top_k = args.soft_prune_top_k
    stage2_config.graph.soft_prune_threshold = args.soft_prune_threshold
    stage2_config.graph.hard_prune_after_turn = args.hard_prune_after_turn

    suite_report = _load_json_dict(output_root / "suite_report.json") if args.resume else None
    if suite_report is None:
        suite_report = {
            "started_at": _utc_now(),
            "data_root": args.data_root,
            "output_root": str(output_root),
            "datasets": {},
            "search_config": asdict(search_config),
            "runtime_config": asdict(runtime_config),
            "union_config": asdict(union_config),
            "stage2_config": asdict(stage2_config),
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
    else:
        suite_progress.pop("finished_at", None)

    for index, dataset_name in enumerate(datasets):
        existing = suite_progress["datasets"].get(dataset_name, {})
        if existing.get("status") == "completed":
            print(f"[resume][{dataset_name}] already completed, skipping", flush=True)
            continue
        plan = _resolve_dataset_plan(
            args.data_root,
            dataset_name,
            periodic_every=args.periodic_every,
            periodic_size=args.periodic_size,
            max_train=args.max_train,
            max_validation=args.max_validation,
            max_test=args.max_test,
        )
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
            union_config=union_config,
            stage2_config=stage2_config,
            plan=plan,
            seed=args.seed + index * 100,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
            save_replays=args.save_replays,
            stage1_checkpoint_root=args.stage1_checkpoint_root,
            structure_cache_enabled=not args.disable_structure_cache,
        )
        suite_report["datasets"][dataset_name] = report
        _write_json(output_root / "suite_report.json", suite_report)
        suite_progress["datasets"][dataset_name] = {
            "status": "completed",
            "seed": args.seed + index * 100,
            "checkpoint_path": report.get("checkpoint_path", ""),
            "updated_at": _utc_now(),
        }
        _write_json(progress_path, suite_progress)

    suite_report["finished_at"] = _utc_now()
    _write_json(output_root / "suite_report.json", suite_report)
    suite_progress["finished_at"] = _utc_now()
    _write_json(progress_path, suite_progress)
    print(json.dumps(suite_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
