from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
VENDOR_ROOT = PACKAGE_ROOT / "vendor"
for candidate in (PACKAGE_ROOT, VENDOR_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from mas_stage2 import load_prepared_stage1_artifact, save_prepared_stage1_artifact
from mas_treesearch import SearchConfig, TieredEvalConfig, UnionRuntimeConfig, list_processed_datasets, load_processed_split
from train_mas_stage2_target_suite import (
    RunningStats,
    _checkpoint_metadata,
    _item_dataset_name,
    _item_metadata,
    _load_json_dict,
    _periodic_window,
    _resolve_dataset_plan,
    _runtime_config_dict,
    _safe_item_key,
    _sample,
    _suite_progress_path,
    _utc_now,
    _validate_resume_ids,
    _write_json,
    _write_jsonl,
)

from stage2_phase3a_unified import Phase3aUnifiedConfig, Phase3aUnifiedPipeline


def _stage2_config_dict(stage2_config: Phase3aUnifiedConfig) -> Dict[str, Any]:
    payload = asdict(stage2_config)
    learning = payload.get("learning", {})
    if isinstance(learning, dict):
        learning.pop("fallback_penalty", None)
    return payload


def _structure_artifact_path(root: Path, dataset_name: str, split: str, item: Dict[str, Any]) -> Path:
    return root / dataset_name / split / f"{_safe_item_key(item.get('id', 'sample'))}.json"


def _selection_decision(row: Dict[str, Any]) -> str:
    return "use_stage1_anchor" if bool(row.get("stage1_anchor_used", False)) else "use_stage2_protocol"


def _compare_stage2_vs_stage1(row: Dict[str, Any], *, eps: float = 1e-9) -> str:
    stage1_success = float(row.get("stage1_success", 0.0))
    stage2_success = float(row.get("stage2_success", 0.0))
    if stage2_success > stage1_success + eps:
        return "better"
    if stage2_success < stage1_success - eps:
        return "worse"
    stage1_task = float(row.get("stage1_task_score", 0.0))
    stage2_task = float(row.get("stage2_task_score", 0.0))
    if stage2_task > stage1_task + eps:
        return "better"
    if stage2_task < stage1_task - eps:
        return "worse"
    return "same"


def _value_distribution(rows: Iterable[Dict[str, Any]], key: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, ""))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _mean(values: Iterable[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))


def _mean_ratio_by_turn(rows: Iterable[Dict[str, Any]], key: str) -> List[float]:
    rows_list = list(rows)
    max_len = max((len(row.get(key, [])) for row in rows_list), default=0)
    means: List[float] = []
    for turn_index in range(max_len):
        bucket = []
        for row in rows_list:
            values = row.get(key, [])
            if isinstance(values, list) and turn_index < len(values):
                bucket.append(float(values[turn_index]))
        means.append(_mean(bucket))
    return means


def _selection_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    reason_counts: Dict[str, int] = {}
    source_counts: Dict[str, int] = {}
    anchor_used = 0
    correction_selected = 0
    for row in rows_list:
        reason = str(row.get("selection_reason", ""))
        source = str(row.get("selected_candidate_source", ""))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        source_counts[source] = source_counts.get(source, 0) + 1
        if row.get("stage1_anchor_used", False):
            anchor_used += 1
        if row.get("selected_is_correction", False):
            correction_selected += 1
    return {
        "count": len(rows_list),
        "anchor_used_count": anchor_used,
        "anchor_used_rate": (anchor_used / len(rows_list)) if rows_list else 0.0,
        "correction_selected_count": correction_selected,
        "correction_selected_rate": (correction_selected / len(rows_list)) if rows_list else 0.0,
        "selection_reason_counts": dict(sorted(reason_counts.items())),
        "selected_source_counts": dict(sorted(source_counts.items())),
    }


def _phase_analysis_summary(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows_list = list(rows)
    outcome_counts = {"better": 0, "worse": 0, "same": 0}
    for row in rows_list:
        outcome = str(row.get("stage2_vs_stage1_outcome", "same"))
        outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
    return {
        "count": len(rows_list),
        "better": int(outcome_counts.get("better", 0)),
        "worse": int(outcome_counts.get("worse", 0)),
        "same": int(outcome_counts.get("same", 0)),
        "stage1_success_avg": _mean(row.get("stage1_success", 0.0) for row in rows_list),
        "stage2_success_avg": _mean(row.get("stage2_success", 0.0) for row in rows_list),
        "stage1_task_avg": _mean(row.get("stage1_task_score", 0.0) for row in rows_list),
        "stage2_task_avg": _mean(row.get("stage2_task_score", 0.0) for row in rows_list),
        "selection_reason_distribution": _value_distribution(rows_list, "selection_reason"),
        "active_edge_ratio_by_turn": _mean_ratio_by_turn(rows_list, "active_edge_ratio_by_turn"),
        "active_node_ratio_by_turn": _mean_ratio_by_turn(rows_list, "active_node_ratio_by_turn"),
        "candidate_provenance_coverage": _mean(row.get("candidate_provenance_coverage", 0.0) for row in rows_list),
        "recovery_subgraph_size": _mean(row.get("recovery_subgraph_size", 0.0) for row in rows_list),
        "final_answer_source_type": _value_distribution(rows_list, "final_answer_source_type"),
    }


def _print_row(prefix: str, idx: int, total: int, row: Dict[str, Any], *, elapsed_s: float) -> None:
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
        f"s2_turns={row.get('stage2_turns', 0):.1f}",
        f"decision={row.get('selection_decision', '')}",
        f"reason={row.get('selection_reason', '')}",
        f"anchor={int(bool(row.get('stage1_anchor_used', False)))}",
        f"cand={row.get('candidate_count', 0)}",
        f"class={row.get('soft_class_count', 0)}",
        f"corr_try={row.get('correction_attempt_count', 0)}",
        f"corr_acc={row.get('correction_accept_count', 0)}",
        f"halt={row.get('phase3a_halt_mass', 0.0):.3f}",
        f"sel_src={row.get('selected_candidate_source', '')}",
        f"sel_corr={int(bool(row.get('selected_is_correction', False)))}",
        f"elapsed_s={elapsed_s:.1f}",
        f"signature={row['signature']}",
    ]
    print(" ".join(parts), flush=True)


def _print_summary(prefix: str, stats: RunningStats, *, rows: Optional[Iterable[Dict[str, Any]]] = None, elapsed_s: float) -> None:
    metrics = stats.mean_dict()
    rows_list = list(rows or [])
    anchor_rate = 0.0
    correction_rate = 0.0
    analysis = _phase_analysis_summary(rows_list)
    if rows_list:
        anchor_rate = sum(1.0 for row in rows_list if row.get("stage1_anchor_used", False)) / len(rows_list)
        correction_rate = sum(1.0 for row in rows_list if row.get("selected_is_correction", False)) / len(rows_list)
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
        f"anchor_rate={anchor_rate:.4f}",
        f"correction_rate={correction_rate:.4f}",
        f"better={analysis['better']}",
        f"worse={analysis['worse']}",
        f"same={analysis['same']}",
        f"elapsed_s={elapsed_s:.1f}",
    ]
    print(" ".join(parts), flush=True)


def _run_stage2_search(
    pipeline: Phase3aUnifiedPipeline,
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
            )
    else:
        result = pipeline.search(
            item["question"],
            reference_answer=item.get("answer"),
            metadata=metadata,
            dataset_name=resolved_dataset,
            learn=learn,
            replay_dir=replay_dir,
        )
    if artifact_path is not None:
        result.stage2_result.metadata["structure_artifact_path"] = str(artifact_path)
    return result


def _row_from_result(dataset_name: str, split: str, item: Dict[str, Any], result: Any) -> Dict[str, Any]:
    summary = result.final_summary
    structure = result.stage1_result.structure_summary if result.stage1_result is not None else result.stage1_artifact.structure_summary
    meta = result.stage2_result.metadata
    learning_stats = meta.get("learning_stats", {}) if isinstance(meta.get("learning_stats"), dict) else {}
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
        "selection_decision": _selection_decision(meta),
        "selection_reason": str(meta.get("selection_reason", "")),
        "final_selection_decision": _selection_decision(meta),
        "final_selection_reason": str(meta.get("selection_reason", "")),
        "fallback_applied": float(bool(meta.get("stage1_anchor_used", False))),
        "finalizer_strategy": str(meta.get("finalizer_strategy", "")),
        "structure_source": str(meta.get("structure_source", result.stage1_artifact.metadata.get("source", ""))),
        "stage2_version": str(meta.get("stage2_version", "phase3a_unified_v1")),
        "stage2_turns": float(meta.get("turn_count", 0)),
        "stage2_memory_records": float(sum(result.stage2_result.memory_record_counts.values())),
        "stage1_reward": float(meta.get("stage1_reward", 0.0)),
        "stage2_reward": float(meta.get("stage2_reward", 0.0)),
        "stage1_task_score": float(meta.get("stage1_task_score", 0.0)),
        "stage2_task_score": float(meta.get("stage2_task_score", 0.0)),
        "stage1_success": float(meta.get("stage1_success", 0.0)),
        "stage2_success": float(meta.get("stage2_success", 0.0)),
        "stage1_latency": float(meta.get("stage1_latency", 0.0)),
        "stage2_latency": float(meta.get("stage2_latency", 0.0)),
        "stage1_token_cost": float(meta.get("stage1_token_cost", 0.0)),
        "stage2_token_cost": float(meta.get("stage2_token_cost", 0.0)),
        "stage2_turn_token_costs": list(meta.get("turn_token_costs", [])),
        "stage2_turn_token_estimates": list(meta.get("turn_token_estimates", [])),
        "active_edge_ratio_by_turn": list(meta.get("graph_faithfulness_active_edge_ratio_by_turn", [])),
        "active_node_ratio_by_turn": list(meta.get("graph_faithfulness_active_node_ratio_by_turn", [])),
        "candidate_provenance_coverage": float(meta.get("graph_faithfulness_candidate_provenance_coverage", 0.0)),
        "recovery_subgraph_size": float(meta.get("graph_faithfulness_recovery_subgraph_size", 0.0)),
        "final_answer_source_type": str(meta.get("graph_faithfulness_final_answer_source_type", "")),
        "stage1_anchor_present": bool(meta.get("stage1_anchor_present", False)),
        "stage1_anchor_used": bool(meta.get("stage1_anchor_used", False)),
        "candidate_count": int(meta.get("candidate_count", 0)),
        "soft_class_count": int(meta.get("soft_class_count", 0)),
        "selected_candidate_digest": str(meta.get("selected_candidate_digest", "")),
        "selected_candidate_source": str(meta.get("selected_candidate_source", "")),
        "selected_is_correction": bool(meta.get("selected_is_correction", False)),
        "correction_attempt_count": int(meta.get("correction_attempt_count", 0)),
        "correction_accept_count": int(meta.get("correction_accept_count", 0)),
        "correction_improvement_count": int(meta.get("correction_improvement_count", 0)),
        "phase3a_halt_mass": float(meta.get("phase3a_halt_mass", 0.0)),
        "phase3a_top_candidates": list(meta.get("phase3a_top_candidates", [])),
        "phase3a_top_classes": list(meta.get("phase3a_top_classes", [])),
        "phase3a_utility_updates": float(learning_stats.get("utility_updates", 0.0)),
        "phase3a_delta_updates": float(learning_stats.get("delta_updates", 0.0)),
        "phase3a_halt_updates": float(learning_stats.get("halt_updates", 0.0)),
        "phase3a_utility_steps_after": float(learning_stats.get("utility_steps", 0.0)),
        "phase3a_delta_steps_after": float(learning_stats.get("delta_steps", 0.0)),
        "phase3a_halt_steps_after": float(learning_stats.get("halt_steps", 0.0)),
    }
    row["stage2_vs_stage1_outcome"] = _compare_stage2_vs_stage1(row)
    if structure is not None:
        row["structure_reward"] = float(structure.metrics.total_reward)
        row["coverage"] = float(structure.metrics.coverage)
        row["complementarity"] = float(structure.metrics.complementarity)
        row["redundancy_quality"] = float(structure.metrics.redundancy_quality)
    return row


def _run_eval_phase(
    pipeline: Phase3aUnifiedPipeline,
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
    _print_summary(f"[{split}-summary][{dataset_name}]", stats, rows=rows, elapsed_s=time.time() - start_time)
    return rows, stats


def _run_dataset(
    dataset_name: str,
    *,
    data_root: str,
    output_root: Path,
    search_config: SearchConfig,
    runtime_config: TieredEvalConfig,
    union_config: UnionRuntimeConfig,
    stage2_config: Phase3aUnifiedConfig,
    plan: Dict[str, int],
    seed: int,
    resume: bool,
    checkpoint_every: int,
    save_replays: bool,
    stage1_checkpoint_root: str,
    structure_cache_enabled: bool,
) -> Dict[str, Any]:
    train_items = _sample(load_processed_split(data_root, dataset_name, "train"), plan["max_train"], seed, True)
    validation_items = _sample(load_processed_split(data_root, dataset_name, "validation"), plan["max_validation"], seed + 1, True)
    test_items = _sample(load_processed_split(data_root, dataset_name, "test"), plan["max_test"], seed + 2, True)
    if not train_items:
        raise ValueError(f"No train items found for dataset={dataset_name}")

    dataset_output_root = output_root / dataset_name
    dataset_output_root.mkdir(parents=True, exist_ok=True)
    replay_root = dataset_output_root / "replays"
    checkpoint_path = dataset_output_root / "checkpoint.json"
    structure_cache_root = dataset_output_root / "stage1_structures" if structure_cache_enabled else None

    pipeline = Phase3aUnifiedPipeline(
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
        f"\n=== PHASE3A_UNIFIED DATASET {dataset_name} train={len(train_items)} validation={len(validation_items)} test={len(test_items)} ===",
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
        metadata["train_phase_analysis"] = _phase_analysis_summary(train_rows)
        metadata["periodic_validation_phase_analysis"] = _phase_analysis_summary(periodic_validation_rows)
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
                    rows=periodic_validation_rows,
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
                metadata["train_phase_analysis"] = _phase_analysis_summary(train_rows)
                metadata["periodic_validation_phase_analysis"] = _phase_analysis_summary(periodic_validation_rows)
                pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
        train_phase_complete = True

    _print_summary(f"[train-summary][{dataset_name}]", train_stats, rows=train_rows, elapsed_s=time.time() - train_start)

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
    metadata["train_phase_analysis"] = _phase_analysis_summary(train_rows)
    metadata["periodic_validation_phase_analysis"] = _phase_analysis_summary(periodic_validation_rows)
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
        "train_selection_summary": _selection_summary(train_rows),
        "train_phase_analysis": _phase_analysis_summary(train_rows),
        "periodic_validation_summary": periodic_validation_stats.mean_dict() | {"count": periodic_validation_stats.count},
        "periodic_validation_selection_summary": _selection_summary(periodic_validation_rows),
        "periodic_validation_phase_analysis": _phase_analysis_summary(periodic_validation_rows),
        "validation_summary": validation_stats.mean_dict() | {"count": validation_stats.count},
        "validation_selection_summary": _selection_summary(validation_rows),
        "validation_phase_analysis": _phase_analysis_summary(validation_rows),
        "test_summary": test_stats.mean_dict() | {"count": test_stats.count},
        "test_selection_summary": _selection_summary(test_rows),
        "test_phase_analysis": _phase_analysis_summary(test_rows),
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
    metadata["train_phase_analysis"] = _phase_analysis_summary(train_rows)
    metadata["periodic_validation_phase_analysis"] = _phase_analysis_summary(periodic_validation_rows)
    metadata["validation_phase_analysis"] = _phase_analysis_summary(validation_rows)
    metadata["test_phase_analysis"] = _phase_analysis_summary(test_rows)
    pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="/mnt/nvme/projects/R-HAN/dataset/mas_stage2_target_suite_from_stage1_test")
    parser.add_argument("--output-root", default="/mnt/nvme/projects/R-HAN/outputs/mas_stage2_phase3a_unified_train")
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
    parser.add_argument("--memory-top-k", type=int, default=4)
    parser.add_argument("--soft-prune-top-k", type=int, default=3)
    parser.add_argument("--soft-prune-threshold", type=float, default=0.38)
    parser.add_argument("--hard-prune-after-turn", type=int, default=4)
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
    parser.add_argument("--correction-max-rounds", type=int, default=2)
    parser.add_argument("--frontier-top-k", type=int, default=4)
    parser.add_argument("--max-prompt-chars", type=int, default=16000)
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

    stage2_config = Phase3aUnifiedConfig()
    stage2_config.graph.turn_count = args.stage2_turn_count
    stage2_config.memory.max_selected_records = args.memory_top_k
    stage2_config.graph.soft_prune_top_k = args.soft_prune_top_k
    stage2_config.graph.soft_prune_threshold = args.soft_prune_threshold
    stage2_config.graph.hard_prune_after_turn = args.hard_prune_after_turn
    stage2_config.correction_max_rounds = args.correction_max_rounds
    stage2_config.frontier_top_k = args.frontier_top_k
    stage2_config.replay.max_prompt_chars = args.max_prompt_chars

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
            "stage2_config": _stage2_config_dict(stage2_config),
            "stage2_version": stage2_config.stage2_version,
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
            "stage2_version": stage2_config.stage2_version,
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
