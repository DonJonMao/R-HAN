from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from mas_treesearch import load_processed_split

from stage2_rollback.online import (
    build_online_candidate_state,
    build_online_evaluator,
    build_online_samples,
    generate_online_candidate,
    load_online_models,
    resolve_profile,
)


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(items: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    picked = list(items)
    random.Random(seed).shuffle(picked)
    if limit >= 0:
        picked = picked[:limit]
    return picked


def main() -> None:
    parser = argparse.ArgumentParser(description="Run stage2-rollback full online evaluation with frozen stage1 artifacts.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--trained-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--max-train", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--chat-api-base", default="http://127.0.0.1:8088")
    args = parser.parse_args()

    dataset_name = str(args.dataset)
    output_root = Path(args.output_root) / dataset_name
    output_root.mkdir(parents=True, exist_ok=True)
    profile = resolve_profile(dataset_name)
    evaluator = build_online_evaluator(chat_api_base=args.chat_api_base)
    runtime = load_online_models(trained_root=args.trained_root, dataset_name=dataset_name)

    items = _sample(load_processed_split(args.data_root, dataset_name, "train"), args.max_train, args.seed)
    if not items:
        raise ValueError(f"No train items found for dataset={dataset_name}")
    samples = build_online_samples(dataset_name=dataset_name, items=items)
    rows: List[Dict[str, Any]] = []
    start_ts = time.time()

    metrics = {
        "dataset": dataset_name,
        "count": 0,
        "short_circuit_count": 0,
        "null_emission_count": 0,
        "generated_candidate_count": 0,
        "selector_override_count": 0,
        "improved_count": 0,
        "degraded_count": 0,
        "equal_count": 0,
        "stage1_wrong_stage2_right": 0,
        "stage1_right_stage2_wrong": 0,
    }

    for index, sample in enumerate(samples):
        analysis = runtime.analyze(sample, train_mode=False)
        anchor_summary = evaluator.evaluate_output(
            sample.question,
            sample.anchor_candidate.output,
            tier="tier2",
            reference_answer=sample.reference_answer,
            metadata=sample.metadata,
            dataset_profile=profile,
        )
        candidate_texts: List[str] = []
        candidate_states = []
        raw_requests = list(analysis.requests)
        null_emission = len(raw_requests) == 0
        if analysis.boundary_output.predicted_boundary == 0:
            metrics["short_circuit_count"] += 1
        elif null_emission:
            metrics["null_emission_count"] += 1

        for request in raw_requests:
            generated = generate_online_candidate(
                sample=sample,
                request=request,
                evaluator=evaluator,
                profile=profile,
                pool=evaluator.agent_pool,
            )
            if not generated:
                continue
            candidate_texts.append(generated)
            candidate_states.append(build_online_candidate_state(sample=sample, candidate_text=generated))
        if candidate_texts:
            metrics["generated_candidate_count"] += 1

        selector_output = runtime.selector_model(analysis.anchor_state, candidate_states)
        if selector_output.winner_index > 0 and 0 <= selector_output.winner_index - 1 < len(candidate_texts):
            final_output = candidate_texts[selector_output.winner_index - 1]
            metrics["selector_override_count"] += 1
        else:
            final_output = sample.anchor_candidate.output

        final_summary = evaluator.evaluate_output(
            sample.question,
            final_output,
            tier="tier2",
            reference_answer=sample.reference_answer,
            metadata=sample.metadata,
            dataset_profile=profile,
        )

        anchor_tuple = (float(anchor_summary.mean_success), float(anchor_summary.mean_task_score))
        final_tuple = (float(final_summary.mean_success), float(final_summary.mean_task_score))
        if final_tuple > anchor_tuple:
            metrics["improved_count"] += 1
        elif final_tuple < anchor_tuple:
            metrics["degraded_count"] += 1
        else:
            metrics["equal_count"] += 1
        if anchor_tuple[0] < 1.0 and final_tuple[0] >= 1.0:
            metrics["stage1_wrong_stage2_right"] += 1
        if anchor_tuple[0] >= 1.0 and final_tuple[0] < 1.0:
            metrics["stage1_right_stage2_wrong"] += 1
        metrics["count"] += 1

        row = {
            "id": sample.id,
            "boundary_index": int(analysis.boundary_output.predicted_boundary),
            "null_emission": bool(null_emission),
            "request_count": len(raw_requests),
            "requests": raw_requests,
            "generated_candidates": candidate_texts,
            "winner_index": int(selector_output.winner_index),
            "anchor_success": float(anchor_summary.mean_success),
            "anchor_task_score": float(anchor_summary.mean_task_score),
            "final_success": float(final_summary.mean_success),
            "final_task_score": float(final_summary.mean_task_score),
            "improved": final_tuple > anchor_tuple,
            "degraded": final_tuple < anchor_tuple,
            "selected_final_output": final_output,
        }
        if candidate_texts and selector_output.candidate_views:
            row["candidate_views"] = [
                {
                    "candidate_id": view.candidate_id,
                    "probability": float(view.probability.item()),
                    "delta_type": float(view.delta_type),
                    "delta_ctr": float(view.delta_ctr),
                    "d_sig": float(view.d_sig),
                    "m_keep": float(view.m_keep),
                }
                for view in selector_output.candidate_views
            ]
        rows.append(row)
        if index < int(args.debug_limit):
            print(
                "[rollback-online]"
                f" id={sample.id}"
                f" boundary={analysis.boundary_output.predicted_boundary}"
                f" requests={len(raw_requests)}"
                f" generated={len(candidate_texts)}"
                f" winner={selector_output.winner_index}"
                f" anchor=({anchor_summary.mean_success:.2f},{anchor_summary.mean_task_score:.3f})"
                f" final=({final_summary.mean_success:.2f},{final_summary.mean_task_score:.3f})",
                flush=True,
            )

    metrics["elapsed_s"] = time.time() - start_ts
    _write_json(output_root / "metrics.json", metrics)
    _write_json(output_root / "suite_report.json", {"dataset": dataset_name, "saved_at": _utc_now(), "metrics": metrics})
    _write_jsonl(output_root / "online_rows.jsonl", rows)
    print(f"[rollback-online-done] dataset={dataset_name} metrics={json.dumps(metrics, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
