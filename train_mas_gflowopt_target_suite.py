from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

from demo_train import build_llm_evaluator
from mas_gflowopt import MASConfig, MASGFlowPipeline
from mas_gflowopt.types import DAGState, GFlowNetTrainingStats, RewardBreakdown, TaskEvaluation
from mas_treesearch import load_processed_split


DEFAULT_DATASET_PLAN: Dict[str, Dict[str, int]] = {
    "multiarith": {"max_train": 493, "max_validation": 57, "max_test": 50, "periodic_every": 25, "periodic_size": 10},
    "humaneval": {"max_train": 125, "max_validation": 23, "max_test": 16, "periodic_every": 20, "periodic_size": 8},
    "mbpp": {"max_train": 500, "max_validation": 100, "max_test": 93, "periodic_every": 50, "periodic_size": 20},
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
    db_loss: float = 0.0
    contrastive_loss: float = 0.0
    total_loss: float = 0.0

    def add(self, row: Dict[str, Any]) -> None:
        self.count += 1
        self.reward += float(row["reward"])
        self.task_score += float(row["task_score"])
        self.success += float(row["success"])
        self.latency += float(row["latency"])
        self.token_cost += float(row["token_cost"])
        self.safety_penalty += float(row["safety_penalty"])
        self.db_loss += float(row.get("db_loss", 0.0))
        self.contrastive_loss += float(row.get("contrastive_loss", 0.0))
        self.total_loss += float(row.get("total_loss", 0.0))

    def mean_dict(self) -> Dict[str, float]:
        if self.count == 0:
            return {
                "reward": 0.0,
                "task_score": 0.0,
                "success": 0.0,
                "latency": 0.0,
                "token_cost": 0.0,
                "safety_penalty": 0.0,
                "db_loss": 0.0,
                "contrastive_loss": 0.0,
                "total_loss": 0.0,
            }
        return {
            "reward": self.reward / self.count,
            "task_score": self.task_score / self.count,
            "success": self.success / self.count,
            "latency": self.latency / self.count,
            "token_cost": self.token_cost / self.count,
            "safety_penalty": self.safety_penalty / self.count,
            "db_loss": self.db_loss / self.count,
            "contrastive_loss": self.contrastive_loss / self.count,
            "total_loss": self.total_loss / self.count,
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
            "db_loss": float(self.db_loss),
            "contrastive_loss": float(self.contrastive_loss),
            "total_loss": float(self.total_loss),
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
            db_loss=float(payload.get("db_loss", 0.0)),
            contrastive_loss=float(payload.get("contrastive_loss", 0.0)),
            total_loss=float(payload.get("total_loss", 0.0)),
        )


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sample(items: List[Dict[str, Any]], limit: int, seed: int, shuffle: bool) -> List[Dict[str, Any]]:
    picked = list(items)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(picked)
    if limit >= 0:
        return picked[:limit]
    return picked


def _periodic_window(items: List[Dict[str, Any]], *, round_idx: int, window: int) -> List[Dict[str, Any]]:
    if not items:
        return []
    size = min(window, len(items))
    start = (round_idx * size) % len(items)
    if start + size <= len(items):
        return items[start : start + size]
    overflow = start + size - len(items)
    return items[start:] + items[:overflow]


def _dag_signature(dag: DAGState) -> str:
    nodes = [str(node) for node in dag.nodes]
    named_edges: List[str] = []
    for src, dst in sorted(dag.edges):
        if 0 <= src < len(nodes) and 0 <= dst < len(nodes):
            named_edges.append(f"{nodes[src]}>{nodes[dst]}")
        else:
            named_edges.append(f"{src}>{dst}")
    return f"nodes={','.join(nodes)}|edges={';'.join(named_edges) if named_edges else '-'}"


def _build_config(args: argparse.Namespace) -> MASConfig:
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


def _runtime_dict(cfg: MASConfig, evaluator: Any) -> Dict[str, Any]:
    return {
        "llm": {
            "api_base": os.getenv("LLM_API_BASE", "http://localhost:8039"),
            "model": getattr(evaluator.config, "model", ""),
            "judge_model": getattr(evaluator.config, "judge_model", ""),
            "timeout_s": getattr(evaluator.config, "timeout_s", 0.0),
            "temperature": getattr(evaluator.config, "temperature", 0.0),
            "max_tokens": getattr(evaluator.config, "max_tokens", 0),
            "judge_temperature": getattr(evaluator.config, "judge_temperature", 0.0),
            "judge_max_tokens": getattr(evaluator.config, "judge_max_tokens", 0),
            "batch_eval": True,
            "batch_max_workers": getattr(evaluator.config, "batch_max_workers", 0),
        },
        "embedding": {
            "api_base": cfg.embedding_api_base,
            "model": cfg.embedding_model,
            "timeout_s": cfg.embedding_timeout_s,
            "dim": cfg.embedding_dim,
        },
        "gflow": asdict(cfg),
    }


def _evaluate_task(
    evaluator: Any,
    dag: DAGState,
    *,
    question_text: str,
    question_vector: List[float] | None,
) -> TaskEvaluation:
    return evaluator.evaluate(
        dag,
        active_agent_ids=list(dag.nodes),
        question_text=question_text,
        question_vector=question_vector,
    )


def _evaluate_reward(
    pipeline: MASGFlowPipeline,
    evaluator: Any,
    dag: DAGState,
    *,
    question_text: str,
    question_vector: List[float] | None,
    agent_vectors: Dict[str, List[float]],
    task_eval: TaskEvaluation,
) -> RewardBreakdown:
    reward_model = pipeline.reward_model
    cfg = reward_model.config
    task_utility = reward_model.utility(task_eval)
    if cfg.contribution_mode.lower() in {"none", "off", "disabled"}:
        contributions: Dict[str, float] = {}
        contribution_term = 0.0
    else:
        contributions = reward_model.estimate_agent_contributions(
            dag,
            evaluator,
            task_utility,
            question_text=question_text,
            question_vector=question_vector,
        )
        positive_contrib = [max(0.0, value) for value in contributions.values()]
        contribution_term = (sum(positive_contrib) / len(positive_contrib)) if positive_contrib else 0.0

    if cfg.use_task_score_as_bic:
        bic_score = task_utility
    else:
        bic_score = reward_model.scorer.score(dag)
    bic_term = reward_model._bic_term(bic_score)
    question_alignment_term = reward_model._question_alignment(dag, question_vector, agent_vectors)
    size_penalty_term = reward_model._size_penalty(dag)
    task_weight = 0.0 if cfg.use_task_score_as_bic else cfg.reward_task_weight
    total_score = (
        task_weight * task_utility
        + cfg.reward_bic_weight * bic_term
        + cfg.reward_contrib_weight * contribution_term
        + cfg.reward_question_weight * question_alignment_term
        + cfg.reward_size_penalty_weight * size_penalty_term
    )
    reward = math.exp(
        min(max(cfg.reward_temperature * total_score, cfg.reward_clip_min), cfg.reward_clip_max)
    )
    return RewardBreakdown(
        bic_score=bic_score,
        bic_term=bic_term,
        task_utility=task_utility,
        task_score=float(task_eval.task_score),
        task_success=float(task_eval.success),
        task_safety_penalty=float(task_eval.safety_penalty),
        contribution_term=contribution_term,
        total_score=total_score,
        reward=reward,
        question_alignment_term=question_alignment_term,
        size_penalty_term=size_penalty_term,
        agent_contributions=contributions,
        component_terms={
            "task": task_weight * task_utility,
            "bic": cfg.reward_bic_weight * bic_term,
            "contrib": cfg.reward_contrib_weight * contribution_term,
            "question": cfg.reward_question_weight * question_alignment_term,
            "size_penalty": cfg.reward_size_penalty_weight * size_penalty_term,
        },
    )


def _row_from_breakdown(
    dataset_name: str,
    split: str,
    item: Dict[str, Any],
    dag: DAGState,
    breakdown: RewardBreakdown,
    task_eval: TaskEvaluation,
    *,
    history: List[GFlowNetTrainingStats] | None = None,
) -> Dict[str, Any]:
    last = history[-1] if history else None
    return {
        "id": item.get("id", ""),
        "split": split,
        "dataset": dataset_name,
        "category": item.get("category", ""),
        "reward": float(breakdown.reward),
        "task_score": float(breakdown.task_score),
        "success": float(breakdown.task_success),
        "latency": float(task_eval.latency),
        "token_cost": float(task_eval.token_cost),
        "safety_penalty": float(breakdown.task_safety_penalty),
        "db_loss": float(last.db_loss) if last is not None else 0.0,
        "contrastive_loss": float(last.contrastive_loss) if last is not None else 0.0,
        "total_loss": float(last.total_loss) if last is not None else 0.0,
        "signature": _dag_signature(dag),
    }


def _print_train_row(prefix: str, idx: int, total: int, row: Dict[str, Any], *, elapsed_s: float) -> None:
    print(
        f"{prefix} {idx}/{total} id={row['id']} reward={row['reward']:.4f} "
        f"task={row['task_score']:.4f} success={row['success']:.4f} "
        f"latency={row['latency']:.2f} token={row['token_cost']:.4f} "
        f"db={row['db_loss']:.4f} cl={row['contrastive_loss']:.4f} total={row['total_loss']:.4f} "
        f"safety={row['safety_penalty']:.4f} elapsed_s={elapsed_s:.1f} signature={row['signature']}",
        flush=True,
    )


def _print_eval_row(prefix: str, idx: int, total: int, row: Dict[str, Any], *, elapsed_s: float) -> None:
    print(
        f"{prefix} {idx}/{total} id={row['id']} reward={row['reward']:.4f} "
        f"task={row['task_score']:.4f} success={row['success']:.4f} "
        f"safety={row['safety_penalty']:.4f} elapsed_s={elapsed_s:.1f} signature={row['signature']}",
        flush=True,
    )


def _print_summary(prefix: str, stats: RunningStats, *, elapsed_s: float) -> None:
    metrics = stats.mean_dict()
    print(
        f"{prefix} count={stats.count} avg_reward={metrics['reward']:.4f} "
        f"avg_task={metrics['task_score']:.4f} avg_success={metrics['success']:.4f} "
        f"avg_latency={metrics['latency']:.2f} avg_token={metrics['token_cost']:.4f} "
        f"avg_db={metrics['db_loss']:.4f} avg_cl={metrics['contrastive_loss']:.4f} "
        f"avg_total={metrics['total_loss']:.4f} avg_safety={metrics['safety_penalty']:.4f} "
        f"elapsed_s={elapsed_s:.1f}",
        flush=True,
    )


def _run_eval_phase(
    pipeline: MASGFlowPipeline,
    evaluator: Any,
    dataset_name: str,
    items: Iterable[Dict[str, Any]],
    *,
    split: str,
    agent_top_k: int,
) -> tuple[List[Dict[str, Any]], RunningStats]:
    rows: List[Dict[str, Any]] = []
    stats = RunningStats()
    start_time = time.time()
    items_list = list(items)
    for idx, item in enumerate(items_list, start=1):
        prepared = pipeline._prepare_context(question_text=item["question"], agent_top_k=agent_top_k)
        out = pipeline._run_prepared(
            evaluator=evaluator,
            prepared=prepared,
            task_tag=item.get("id") or None,
        )
        task_eval = _evaluate_task(
            evaluator,
            out.refined_best_dag,
            question_text=prepared.cond.question_text or item["question"],
            question_vector=prepared.cond.question_vector,
        )
        breakdown = _evaluate_reward(
            pipeline,
            evaluator,
            out.refined_best_dag,
            question_text=prepared.cond.question_text or item["question"],
            question_vector=prepared.cond.question_vector,
            agent_vectors=prepared.agent_vectors,
            task_eval=task_eval,
        )
        row = _row_from_breakdown(dataset_name, split, item, out.refined_best_dag, breakdown, task_eval)
        rows.append(row)
        stats.add(row)
        _print_eval_row(f"[{split}][{dataset_name}]", idx, len(items_list), row, elapsed_s=time.time() - start_time)
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
        "train_ids": [str(item.get("id", "")) for item in train_items],
        "validation_ids": [str(item.get("id", "")) for item in validation_items],
        "test_ids": [str(item.get("id", "")) for item in test_items],
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
    pipeline: MASGFlowPipeline,
    checkpoint_path: Path,
    *,
    metadata: Dict[str, Any],
) -> None:
    pipeline.save_checkpoint(str(checkpoint_path), metadata=metadata)
    print(f"[checkpoint][{metadata['dataset']}] {checkpoint_path}", flush=True)


def _sampled_split_ids(items: List[Dict[str, Any]]) -> List[str]:
    return [str(item.get("id", "")) for item in items]


def _validate_resume_ids(current_items: List[Dict[str, Any]], saved_ids: List[str], *, dataset_name: str, split: str) -> None:
    current_ids = _sampled_split_ids(current_items)
    if saved_ids and current_ids != list(saved_ids):
        raise ValueError(
            f"Resume split mismatch for dataset={dataset_name} split={split}: "
            f"current sampled ids differ from checkpoint."
        )


def _suite_progress_path(output_root: Path) -> Path:
    return output_root / "suite_progress.json"


def _load_json_dict(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _run_dataset(
    dataset_name: str,
    *,
    data_root: str,
    output_root: Path,
    cfg: MASConfig,
    plan: Dict[str, int],
    seed: int,
    agent_top_k: int,
    batch_eval_workers: int,
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
    checkpoint_path = dataset_output_root / "checkpoint.pt"

    pipeline = MASGFlowPipeline(config=cfg)
    evaluator = build_llm_evaluator(True, batch_eval_workers)
    effective_runtime = _runtime_dict(cfg, evaluator)

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
        effective_runtime = metadata.get("effective_runtime_config", effective_runtime)
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
        initial_meta = _checkpoint_metadata(
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
            train_phase_complete=False,
            phase="init",
        )
        _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=initial_meta)

    if not train_phase_complete:
        train_slice = train_items[train_index_completed:]
        base_train_index = train_index_completed
        for offset, item in enumerate(train_slice, start=1):
            idx = base_train_index + offset
            prepared = pipeline._prepare_context(question_text=item["question"], agent_top_k=agent_top_k)
            history = pipeline._train_prepared(
                evaluator=evaluator,
                prepared=prepared,
                task_tag=item.get("id") or None,
            )
            out = pipeline._run_prepared(
                evaluator=evaluator,
                prepared=prepared,
                task_tag=item.get("id") or None,
            )
            task_eval = _evaluate_task(
                evaluator,
                out.refined_best_dag,
                question_text=prepared.cond.question_text or item["question"],
                question_vector=prepared.cond.question_vector,
            )
            breakdown = _evaluate_reward(
                pipeline,
                evaluator,
                out.refined_best_dag,
                question_text=prepared.cond.question_text or item["question"],
                question_vector=prepared.cond.question_vector,
                agent_vectors=prepared.agent_vectors,
                task_eval=task_eval,
            )
            row = _row_from_breakdown(
                dataset_name,
                "train",
                item,
                out.refined_best_dag,
                breakdown,
                task_eval,
                history=history,
            )
            train_rows.append(row)
            train_stats.add(row)
            train_index_completed = idx
            _print_train_row(
                f"[train][{dataset_name}]",
                idx,
                len(train_items),
                row,
                elapsed_s=time.time() - train_start,
            )

            checkpoint_due = checkpoint_every > 0 and idx % checkpoint_every == 0
            if validation_items and (idx % plan["periodic_every"] == 0 or idx == len(train_items)):
                window = _periodic_window(validation_items, round_idx=validation_round, window=plan["periodic_size"])
                validation_round += 1
                for eval_idx, eval_item in enumerate(window, start=1):
                    prepared_eval = pipeline._prepare_context(
                        question_text=eval_item["question"],
                        agent_top_k=agent_top_k,
                    )
                    eval_out = pipeline._run_prepared(
                        evaluator=evaluator,
                        prepared=prepared_eval,
                        task_tag=eval_item.get("id") or None,
                    )
                    eval_task = _evaluate_task(
                        evaluator,
                        eval_out.refined_best_dag,
                        question_text=prepared_eval.cond.question_text or eval_item["question"],
                        question_vector=prepared_eval.cond.question_vector,
                    )
                    eval_breakdown = _evaluate_reward(
                        pipeline,
                        evaluator,
                        eval_out.refined_best_dag,
                        question_text=prepared_eval.cond.question_text or eval_item["question"],
                        question_vector=prepared_eval.cond.question_vector,
                        agent_vectors=prepared_eval.agent_vectors,
                        task_eval=eval_task,
                    )
                    eval_row = _row_from_breakdown(
                        dataset_name,
                        "periodic_validation",
                        eval_item,
                        eval_out.refined_best_dag,
                        eval_breakdown,
                        eval_task,
                    )
                    periodic_validation_rows.append(eval_row)
                    periodic_validation_stats.add(eval_row)
                    _print_eval_row(
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
                checkpoint_due = True

            if checkpoint_due or idx == len(train_items):
                checkpoint_meta = _checkpoint_metadata(
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
                    train_phase_complete=False,
                    phase="train",
                )
                _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=checkpoint_meta)

        train_phase_complete = True

    _print_summary(f"[train-summary][{dataset_name}]", train_stats, elapsed_s=time.time() - train_start)
    checkpoint_meta = _checkpoint_metadata(
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
        phase="post-train",
    )
    _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=checkpoint_meta)

    validation_rows, validation_stats = _run_eval_phase(
        pipeline,
        evaluator,
        dataset_name,
        validation_items,
        split="validation",
        agent_top_k=agent_top_k,
    )
    test_rows, test_stats = _run_eval_phase(
        pipeline,
        evaluator,
        dataset_name,
        test_items,
        split="test",
        agent_top_k=agent_top_k,
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
    final_checkpoint_meta = _checkpoint_metadata(
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
    _save_dataset_checkpoint(pipeline, checkpoint_path, metadata=final_checkpoint_meta)
    print(f"[report][{dataset_name}] {report_path}", flush=True)
    return report
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_target_suite_20260314",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/outputs/mas_gflowopt_target_suite_train",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to run. Repeatable.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--agent-top-k", type=int, default=6)
    parser.add_argument("--gflownet-train-epochs", type=int, default=1)
    parser.add_argument("--gflownet-batch-size", type=int, default=1)
    parser.add_argument("--num-sampled-dags", type=int, default=1)
    parser.add_argument("--contribution-mode", default="none")
    parser.add_argument("--true-eval-interval", type=int, default=6)
    parser.add_argument("--true-eval-budget", type=int, default=4)
    parser.add_argument("--one-eval-per-trajectory", action="store_true")
    parser.add_argument("--disable-refine", action="store_true")
    parser.add_argument("--batch-eval-workers", type=int, default=12)
    parser.add_argument("--early-stop-metric", default="total_loss")
    parser.add_argument("--early-stop-patience", type=int, default=3)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0001)
    parser.add_argument("--early-stop-warmup-epochs", type=int, default=1)
    parser.add_argument("--embedding-api-base", default="http://127.0.0.1:8018")
    parser.add_argument("--embedding-model", default="/mnt/nvme/Qwen3-Embedding-8B")
    parser.add_argument("--embedding-api-key", default="")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = list(args.dataset) if args.dataset else list(DEFAULT_DATASET_PLAN.keys())
    unknown = [name for name in datasets if name not in DEFAULT_DATASET_PLAN]
    if unknown:
        raise ValueError(f"Unsupported dataset(s): {unknown}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    progress_path = _suite_progress_path(output_root)

    cfg = _build_config(args)
    suite_report = _load_json_dict(output_root / "suite_report.json") if args.resume else None
    if suite_report is None:
        suite_report = {
            "started_at": _utc_now(),
            "data_root": args.data_root,
            "output_root": str(output_root),
            "datasets": {},
            "requested_gflow_config": asdict(cfg),
        }
    suite_progress = _load_json_dict(progress_path) if args.resume else None
    if suite_progress is None:
        suite_progress = {
            "started_at": _utc_now(),
            "data_root": args.data_root,
            "output_root": str(output_root),
            "datasets": {},
        }

    for index, dataset_name in enumerate(datasets):
        existing = suite_progress["datasets"].get(dataset_name, {})
        if existing.get("status") == "completed":
            print(f"[resume][{dataset_name}] already completed, skipping", flush=True)
            continue
        plan = dict(DEFAULT_DATASET_PLAN[dataset_name])
        suite_progress["datasets"][dataset_name] = {
            "status": "running",
            "seed": args.seed + index * 100,
            "checkpoint_path": str(output_root / dataset_name / "checkpoint.pt"),
            "updated_at": _utc_now(),
        }
        _write_json(progress_path, suite_progress)
        report = _run_dataset(
            dataset_name,
            data_root=args.data_root,
            output_root=output_root,
            cfg=cfg,
            plan=plan,
            seed=args.seed + index * 100,
            agent_top_k=args.agent_top_k,
            batch_eval_workers=args.batch_eval_workers,
            resume=args.resume,
            checkpoint_every=args.checkpoint_every,
        )
        suite_report["datasets"][dataset_name] = report
        suite_progress["datasets"][dataset_name] = {
            "status": "completed",
            "seed": args.seed + index * 100,
            "checkpoint_path": report.get("checkpoint_path", ""),
            "updated_at": _utc_now(),
        }
        _write_json(output_root / "suite_report.json", suite_report)
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
