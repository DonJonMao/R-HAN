from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request
from typing import Any, Callable, Optional

from mas_gflowopt import MASConfig, MASGFlowPipeline
from mas_gflowopt.evaluators import (
    BatchLLMExecutionMASTaskEvaluator,
    LLMExecutionConfig,
    LLMExecutionMASTaskEvaluator,
)
from mas_gflowopt.types import DAGState, OptimizationOutput


def load_questions(path: str, max_samples: int | None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = str(obj.get("question", "")).strip()
            if not question:
                continue
            items.append(
                {
                    "id": str(obj.get("id", "")).strip(),
                    "question": question,
                    "category": str(obj.get("category", "")).strip(),
                    "source_dataset": str(obj.get("source_dataset", "")).strip(),
                }
            )
            if max_samples is not None and len(items) >= max_samples:
                break
    return items


def resolve_model_id(api_base: str, fallback: str) -> str:
    model_env = os.getenv("LLM_MODEL")
    if model_env:
        return model_env
    url = api_base.rstrip("/") + "/v1/models"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = data.get("data") if isinstance(data, dict) else []
        if models:
            model_id = models[0].get("id")
            if model_id:
                return str(model_id)
    except Exception:
        return fallback
    return fallback


def build_llm_evaluator(use_batch_eval: bool, batch_eval_workers: int) -> LLMExecutionMASTaskEvaluator:
    api_base = os.getenv("LLM_API_BASE", "http://localhost:8039")
    model_id = resolve_model_id(api_base, "qwen3-8b")
    cfg = LLMExecutionConfig(
        api_base=api_base,
        model=model_id,
        api_key=os.getenv("LLM_API_KEY") or None,
        timeout_s=float(os.getenv("LLM_TIMEOUT_S", "60")),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
        max_tokens=int(os.getenv("LLM_MAX_TOKENS", "768")),
        max_retries=int(os.getenv("LLM_MAX_RETRIES", "2")),
        judge_model=os.getenv("LLM_JUDGE_MODEL") or model_id,
        judge_temperature=float(os.getenv("LLM_JUDGE_TEMPERATURE", "0.2")),
        judge_max_tokens=int(os.getenv("LLM_JUDGE_MAX_TOKENS", "256")),
        token_cost_per_word=float(os.getenv("LLM_TOKEN_COST_PER_WORD", "0.00001")),
        success_threshold=float(os.getenv("LLM_SUCCESS_THRESHOLD", "0.6")),
        verbose=(os.getenv("LLM_VERBOSE", "1").strip() not in {"0", "false", "False"}),
        batch_max_workers=max(1, int(batch_eval_workers)),
    )
    if not cfg.model:
        raise ValueError("LLM_MODEL is required.")
    if use_batch_eval:
        return BatchLLMExecutionMASTaskEvaluator(config=cfg)
    return LLMExecutionMASTaskEvaluator(config=cfg)


def update_stats(stats: dict[str, dict[str, float]], key: str, score: float, success: float) -> None:
    entry = stats.get(key)
    if entry is None:
        entry = {"count": 0.0, "score": 0.0, "success": 0.0}
        stats[key] = entry
    entry["count"] += 1.0
    entry["score"] += score
    entry["success"] += success


def format_stats(stats: dict[str, dict[str, float]]) -> list[str]:
    lines: list[str] = []
    for key in sorted(stats.keys()):
        entry = stats[key]
        count = max(1.0, entry["count"])
        avg_score = entry["score"] / count
        avg_success = entry["success"] / count
        lines.append(f"{key} n={int(count)} acc={avg_success:.4f} score={avg_score:.4f}")
    return lines


def format_progress(prefix: str, idx: int, total: int, start_ts: float) -> str:
    elapsed = max(0.0, time.time() - start_ts)
    rate = idx / elapsed if elapsed > 0 else 0.0
    remaining = (total - idx) / rate if rate > 0 else 0.0
    pct = (idx / total * 100.0) if total > 0 else 0.0
    avg_s = elapsed / max(1, idx)
    start_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    now_ts = time.time()
    now_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now_ts))
    return (
        f"{prefix} {idx}/{total} {pct:.2f}% elapsed={elapsed:.1f}s eta={remaining:.1f}s "
        f"avg_s={avg_s:.2f} start={start_label} now={now_label}"
    )


def graph_signature(dag: DAGState) -> tuple[tuple[str, ...], tuple[tuple[int, int], ...]]:
    return tuple(str(node) for node in dag.nodes), tuple(sorted((int(src), int(dst)) for src, dst in dag.edges))


def cached_eval_from_output(out: OptimizationOutput) -> Optional[dict[str, float]]:
    target = graph_signature(out.refined_best_dag)
    for sampled in out.sampled_dags:
        if graph_signature(sampled.graph) != target:
            continue
        rb = sampled.reward_breakdown
        if rb is None:
            continue
        return {
            "task_score": float(rb.task_score),
            "success": float(rb.task_success),
            "safety_penalty": float(rb.task_safety_penalty),
        }
    return None


def run_train(
    pipeline: MASGFlowPipeline,
    evaluator: LLMExecutionMASTaskEvaluator,
    samples: list[dict[str, Any]],
    agent_top_k: int,
    log_every: int,
    batch_eval_size: int,
    eval_every: int,
    eval_hook: Callable[[int], None] | None,
) -> list[float]:
    rewards: list[float] = []
    stats_window: list[dict[str, float]] = []
    by_dataset: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, float]] = {}
    start_ts = time.time()
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        uncached = [p for p in pending if p.get("cached_eval") is None]
        if uncached:
            dags = [p["dag"] for p in uncached]
            questions = [p["question"] for p in uncached]
            if hasattr(evaluator, "evaluate_batch"):
                fresh = evaluator.evaluate_batch(dags, question_texts=questions)
            else:
                fresh = [evaluator.evaluate(dag, question_text=q) for dag, q in zip(dags, questions)]
            fresh_rows = [
                {
                    "task_score": float(item.task_score),
                    "success": float(item.success),
                    "safety_penalty": float(item.safety_penalty),
                }
                for item in fresh
            ]
        else:
            fresh_rows = []

        fresh_iter = iter(fresh_rows)
        resolved: list[dict[str, float]] = []
        for payload in pending:
            if payload.get("cached_eval") is not None:
                resolved.append(dict(payload["cached_eval"]))
            else:
                resolved.append(next(fresh_iter))

        for payload, eval_result in zip(pending, resolved):
            dataset = payload["dataset"]
            category = payload["category"]
            idx = payload["idx"]
            elapsed_s = time.perf_counter() - float(payload["start_ts"])
            score = float(eval_result["task_score"])
            success = float(eval_result["success"])
            safety_penalty = float(eval_result["safety_penalty"])
            update_stats(by_dataset, dataset, score, success)
            update_stats(by_category, category, score, success)
            latest_reward = float(payload["latest_reward"])
            print(
                f"[train] item {idx}/{len(samples)} dataset={dataset} category={category} "
                f"task_score={score:.4f} success={success:.4f} "
                f"safety={safety_penalty:.4f} reward={latest_reward:.4f} "
                f"elapsed_s={elapsed_s:.2f}",
                flush=True,
            )
            if log_every > 0 and idx % log_every == 0:
                avg_reward = sum(rewards) / max(1, len(rewards))
                window = stats_window[-log_every:] if log_every > 0 else stats_window
                if window:
                    count = float(len(window))
                    avg_db = sum(s["db_loss"] for s in window) / count
                    avg_cl = sum(s["contrastive_loss"] for s in window) / count
                    avg_total = sum(s["total_loss"] for s in window) / count
                    avg_task = sum(s["task_score"] for s in window) / count
                    avg_success = sum(s["success"] for s in window) / count
                    print(
                        f"[train] stats last_n={int(count)} db_loss={avg_db:.4f} cl_loss={avg_cl:.4f} "
                        f"total_loss={avg_total:.4f} task_score={avg_task:.4f} acc={avg_success:.4f}",
                        flush=True,
                    )
                print(
                    f"[train] {format_progress('progress', idx, len(samples), start_ts)} avg_reward={avg_reward:.4f}",
                    flush=True,
                )
                for line in format_stats(by_dataset):
                    print(f"[train] dataset {line}", flush=True)
                for line in format_stats(by_category):
                    print(f"[train] category {line}", flush=True)
        pending = []

    for idx, sample in enumerate(samples, start=1):
        dataset = sample.get("source_dataset") or "unknown"
        category = sample.get("category") or "unknown"
        print(
            f"[train] start {idx}/{len(samples)} dataset={dataset} category={category}",
            flush=True,
        )
        item_start = time.perf_counter()
        history, out = pipeline.train_and_run(
            evaluator=evaluator,
            question_text=sample["question"],
            agent_top_k=agent_top_k,
            task_tag=sample.get("id") or None,
        )
        if history:
            rewards.append(float(history[-1].mean_terminal_reward))
            last = history[-1]
            stats_window.append(
                {
                    "db_loss": float(last.db_loss),
                    "contrastive_loss": float(last.contrastive_loss),
                    "total_loss": float(last.total_loss),
                    "task_score": float(last.mean_task_score),
                    "success": float(last.mean_success),
                }
            )
        latest_reward = float(history[-1].mean_terminal_reward) if history else 0.0
        pending.append(
            {
                "idx": idx,
                "dataset": dataset,
                "category": category,
                "dag": out.refined_best_dag,
                "question": sample["question"],
                "latest_reward": latest_reward,
                "start_ts": item_start,
                "cached_eval": cached_eval_from_output(out),
            }
        )
        if batch_eval_size > 0 and len(pending) >= batch_eval_size:
            flush_pending()
        if eval_every > 0 and idx % eval_every == 0 and eval_hook is not None:
            flush_pending()
            eval_hook(idx)
    flush_pending()
    return rewards


def run_test(
    pipeline: MASGFlowPipeline,
    evaluator: LLMExecutionMASTaskEvaluator,
    samples: list[dict[str, Any]],
    agent_top_k: int,
    log_every: int,
    batch_eval_size: int,
    report_summary: bool,
    accum_metrics: list[dict[str, float]] | None,
    accum_by_dataset: dict[str, dict[str, float]] | None,
    accum_by_category: dict[str, dict[str, float]] | None,
    accum_label: str,
) -> list[dict[str, float]]:
    metrics: list[dict[str, float]] = []
    by_dataset: dict[str, dict[str, float]] = {}
    by_category: dict[str, dict[str, float]] = {}
    start_ts = time.time()
    pending: list[dict[str, Any]] = []

    def flush_pending() -> None:
        nonlocal pending
        if not pending:
            return
        uncached = [p for p in pending if p.get("cached_eval") is None]
        if uncached:
            dags = [p["dag"] for p in uncached]
            questions = [p["question"] for p in uncached]
            if hasattr(evaluator, "evaluate_batch"):
                fresh = evaluator.evaluate_batch(dags, question_texts=questions)
            else:
                fresh = [evaluator.evaluate(dag, question_text=q) for dag, q in zip(dags, questions)]
            fresh_rows = [
                {
                    "task_score": float(item.task_score),
                    "success": float(item.success),
                    "safety_penalty": float(item.safety_penalty),
                }
                for item in fresh
            ]
        else:
            fresh_rows = []

        fresh_iter = iter(fresh_rows)
        resolved: list[dict[str, float]] = []
        for payload in pending:
            if payload.get("cached_eval") is not None:
                resolved.append(dict(payload["cached_eval"]))
            else:
                resolved.append(next(fresh_iter))

        for payload, eval_result in zip(pending, resolved):
            dataset = payload["dataset"]
            category = payload["category"]
            idx = payload["idx"]
            elapsed_s = time.perf_counter() - float(payload["start_ts"])
            score = float(eval_result["task_score"])
            success = float(eval_result["success"])
            safety_penalty = float(eval_result["safety_penalty"])
            update_stats(by_dataset, dataset, score, success)
            update_stats(by_category, category, score, success)
            if accum_by_dataset is not None:
                update_stats(accum_by_dataset, dataset, score, success)
            if accum_by_category is not None:
                update_stats(accum_by_category, category, score, success)
            metrics.append(
                {
                    "task_score": score,
                    "success": success,
                    "safety_penalty": safety_penalty,
                }
            )
            if accum_metrics is not None:
                accum_metrics.append(
                    {
                        "task_score": score,
                        "success": success,
                        "safety_penalty": safety_penalty,
                    }
                )
            print(
                f"[test] item {idx}/{len(samples)} dataset={dataset} category={category} "
                f"task_score={score:.4f} success={success:.4f} "
                f"safety={safety_penalty:.4f} elapsed_s={elapsed_s:.2f}",
                flush=True,
            )
            if log_every > 0 and idx % log_every == 0:
                avg_score = sum(m["task_score"] for m in metrics) / max(1, len(metrics))
                avg_success = sum(m["success"] for m in metrics) / max(1, len(metrics))
                avg_safety = sum(m["safety_penalty"] for m in metrics) / max(1, len(metrics))
                print(
                    f"[test] {format_progress('progress', idx, len(samples), start_ts)} avg_score={avg_score:.4f} avg_success={avg_success:.4f} avg_safety={avg_safety:.4f}",
                    flush=True,
                )
                for line in format_stats(by_dataset):
                    print(f"[test] dataset {line}", flush=True)
                for line in format_stats(by_category):
                    print(f"[test] category {line}", flush=True)
        pending = []

    for idx, sample in enumerate(samples, start=1):
        dataset = sample.get("source_dataset") or "unknown"
        category = sample.get("category") or "unknown"
        print(
            f"[test] start {idx}/{len(samples)} dataset={dataset} category={category}",
            flush=True,
        )
        item_start = time.perf_counter()
        out = pipeline.run(
            evaluator=evaluator,
            question_text=sample["question"],
            agent_top_k=agent_top_k,
            task_tag=sample.get("id") or None,
        )
        pending.append(
            {
                "idx": idx,
                "dataset": dataset,
                "category": category,
                "dag": out.refined_best_dag,
                "question": sample["question"],
                "start_ts": item_start,
                "cached_eval": cached_eval_from_output(out),
            }
        )
        if batch_eval_size > 0 and len(pending) >= batch_eval_size:
            flush_pending()
    flush_pending()
    if report_summary:
        elapsed = time.time() - start_ts
        avg_score = sum(m["task_score"] for m in metrics) / max(1, len(metrics))
        avg_success = sum(m["success"] for m in metrics) / max(1, len(metrics))
        avg_safety = sum(m["safety_penalty"] for m in metrics) / max(1, len(metrics))
        avg_s = elapsed / max(1, len(metrics))
        print(
            f"[test] summary items={len(metrics)} avg_score={avg_score:.4f} acc={avg_success:.4f} "
            f"safety={avg_safety:.4f} seconds={elapsed:.1f} avg_s={avg_s:.2f}",
            flush=True,
        )
        for line in format_stats(by_dataset):
            print(f"[test] dataset {line}", flush=True)
        for line in format_stats(by_category):
            print(f"[test] category {line}", flush=True)
        if accum_metrics is not None and accum_by_dataset is not None and accum_by_category is not None:
            total_avg_score = sum(m["task_score"] for m in accum_metrics) / max(1, len(accum_metrics))
            total_avg_success = sum(m["success"] for m in accum_metrics) / max(1, len(accum_metrics))
            total_avg_safety = sum(m["safety_penalty"] for m in accum_metrics) / max(1, len(accum_metrics))
            print(
                f"[test] {accum_label} items={len(accum_metrics)} avg_score={total_avg_score:.4f} "
                f"acc={total_avg_success:.4f} safety={total_avg_safety:.4f}",
                flush=True,
            )
            for line in format_stats(accum_by_dataset):
                print(f"[test] {accum_label} dataset {line}", flush=True)
            for line in format_stats(accum_by_category):
                print(f"[test] {accum_label} category {line}", flush=True)
    return metrics


def summarize_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {"task_score": 0.0, "success": 0.0, "safety_penalty": 0.0}
    return {
        "task_score": sum(m["task_score"] for m in metrics) / len(metrics),
        "success": sum(m["success"] for m in metrics) / len(metrics),
        "safety_penalty": sum(m["safety_penalty"] for m in metrics) / len(metrics),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", default="/mnt/nvme/projects/R-HAN/dataset/unified_mixed/train.jsonl")
    parser.add_argument("--test-path", default="/mnt/nvme/projects/R-HAN/dataset/unified_mixed/test.jsonl")
    parser.add_argument("--val-path", default="")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-test", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
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
    args = parser.parse_args()

    true_eval_interval = args.true_eval_interval
    true_eval_budget = args.true_eval_budget
    true_eval_terminal_always = True
    if args.one_eval_per_trajectory:
        true_eval_interval = 0
        true_eval_budget = 0
        true_eval_terminal_always = True

    cfg = MASConfig(
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
    pipeline = MASGFlowPipeline(config=cfg)
    evaluator = build_llm_evaluator(args.batch_eval, args.batch_eval_workers)

    train_samples = load_questions(args.train_path, args.max_train or None)
    test_samples = load_questions(args.test_path, args.max_test or None)
    val_samples = load_questions(args.val_path, args.max_val or None) if args.val_path else []

    print(f"train_samples={len(train_samples)} test_samples={len(test_samples)} val_samples={len(val_samples)}")
    start = time.time()
    start_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start))
    periodic_test_samples = test_samples
    if args.periodic_test_size > 0:
        periodic_test_samples = test_samples[: args.periodic_test_size]
    cumulative_metrics: list[dict[str, float]] = []
    cumulative_by_dataset: dict[str, dict[str, float]] = {}
    cumulative_by_category: dict[str, dict[str, float]] = {}

    def run_periodic_test(step_idx: int) -> None:
        if not periodic_test_samples:
            return
        print(f"[test] start_eval train_items={step_idx}", flush=True)
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
    train_end = time.time()
    end_label = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(train_end))
    train_time = train_end - start
    avg_train_s = train_time / max(1, len(train_samples))
    avg_train_reward = sum(train_rewards) / max(1, len(train_rewards))
    print(
        f"train_done avg_reward={avg_train_reward:.4f} seconds={train_time:.1f} "
        f"avg_s={avg_train_s:.2f} start={start_label} end={end_label}"
    )

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
    if val_samples:
        val_metrics = run_test(
            pipeline,
            evaluator,
            val_samples,
            args.agent_top_k,
            0,
            args.batch_eval_size,
            True,
            None,
            None,
            None,
            "cumulative",
        )


if __name__ == "__main__":
    main()
