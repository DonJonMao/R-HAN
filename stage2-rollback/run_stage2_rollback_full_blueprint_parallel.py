from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

PACKAGE_ROOT = Path(__file__).resolve().parent
ROLLBACK_ROOT = PACKAGE_ROOT / "stage2_rollback"
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, ROLLBACK_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from mas_treesearch import load_processed_split


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
        return picked[:limit]
    return picked


def _shard_items(items: List[Dict[str, Any]], workers: int) -> List[List[Dict[str, Any]]]:
    shards = [[] for _ in range(workers)]
    for idx, item in enumerate(items):
        shards[idx % workers].append(item)
    return shards


def _emit_empty_splits(root: Path, dataset_name: str) -> None:
    _write_jsonl(root / dataset_name / "validation.jsonl", [])
    _write_jsonl(root / dataset_name / "test.jsonl", [])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full rollback blueprint training in parallel shards.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--gpu-ids", default="0,1,2")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-train", type=int, default=180)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--epochs-verifier", type=int, default=40)
    parser.add_argument("--epochs-boundary", type=int, default=40)
    parser.add_argument("--epochs-diffusion", type=int, default=40)
    parser.add_argument("--epochs-selector", type=int, default=40)
    args = parser.parse_args()

    gpu_ids = [gpu.strip() for gpu in str(args.gpu_ids).split(",") if gpu.strip()]
    if len(gpu_ids) < args.workers:
        raise ValueError(f"Need at least {args.workers} gpu ids, got {gpu_ids}")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    items = _sample(load_processed_split(args.data_root, args.dataset, "train"), args.max_train, args.seed)
    if not items:
        raise ValueError(f"No train items found for dataset={args.dataset}")
    shards = _shard_items(items, args.workers)
    sample_root = output_root / "sample_shards"
    worker_root = output_root / "workers"
    log_root = output_root / "logs"
    sample_root.mkdir(parents=True, exist_ok=True)
    worker_root.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    manifest = {"dataset": args.dataset, "total_items": len(items), "workers": args.workers, "gpu_ids": gpu_ids[: args.workers], "seed": args.seed, "shards": []}
    procs: List[subprocess.Popen[str]] = []
    handles = []
    for worker_idx, shard in enumerate(shards):
        shard_root = sample_root / f"shard_{worker_idx}"
        _write_jsonl(shard_root / args.dataset / "train.jsonl", shard)
        _emit_empty_splits(shard_root, args.dataset)
        worker_output = worker_root / f"worker_{worker_idx}"
        worker_output.mkdir(parents=True, exist_ok=True)
        log_path = log_root / f"worker_{worker_idx}.log"
        gpu_id = gpu_ids[worker_idx]
        manifest["shards"].append({"worker_idx": worker_idx, "count": len(shard), "data_root": str(shard_root), "gpu_id": gpu_id})
        env = os.environ.copy()
        env.setdefault("LLM_API_BASE", "http://127.0.0.1:8039")
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        cmd = [
            args.python_bin,
            str(PACKAGE_ROOT / "train_stage2_rollback_full_blueprint_target_suite.py"),
            "--data-root",
            str(shard_root),
            "--output-root",
            str(worker_output),
            "--dataset",
            args.dataset,
            "--max-train",
            str(args.max_train),
            "--seed",
            str(args.seed + worker_idx * 97),
            "--device",
            "cuda",
            "--debug-limit",
            str(args.debug_limit),
            "--epochs-verifier",
            str(args.epochs_verifier),
            "--epochs-boundary",
            str(args.epochs_boundary),
            "--epochs-diffusion",
            str(args.epochs_diffusion),
            "--epochs-selector",
            str(args.epochs_selector),
        ]
        handle = log_path.open("w", encoding="utf-8")
        handles.append(handle)
        proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT, text=True)
        procs.append(proc)
    _write_json(output_root / "shard_manifest.json", manifest)
    exit_codes = [proc.wait() for proc in procs]
    for handle in handles:
        handle.close()
    metrics: List[Dict[str, Any]] = []
    for worker_idx in range(args.workers):
        metrics_path = worker_root / f"worker_{worker_idx}" / args.dataset / "metrics.json"
        if metrics_path.exists():
            metrics.append(json.loads(metrics_path.read_text(encoding="utf-8")))
    _write_json(
        output_root / "parallel_runner_summary.json",
        {
            "dataset": args.dataset,
            "exit_codes": exit_codes,
            "metrics": metrics,
        },
    )
    if any(code != 0 for code in exit_codes):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
