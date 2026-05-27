from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from mas_treesearch import list_processed_datasets, load_processed_split

from stage2_rollback.train_bank import build_teacher_bank, teacher_bank_to_jsonl
from stage2_rollback.train_verifier import train_verifier_model


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


def _suite_progress_path(root: Path) -> Path:
    return root / "suite_progress.json"


def _device_from_arg(value: str) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def run_dataset(
    *,
    dataset_name: str,
    data_root: str,
    output_root: Path,
    max_train: int,
    seed: int,
    device: str,
) -> Dict[str, Any]:
    train_items = _sample(load_processed_split(data_root, dataset_name, "train"), max_train, seed)
    if not train_items:
        raise ValueError(f"No train items found for dataset={dataset_name}")
    dataset_root = output_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    progress_path = _suite_progress_path(output_root)
    progress = {
        "started_at": _utc_now(),
        "data_root": data_root,
        "output_root": str(output_root),
        "datasets": {
            dataset_name: {
                "status": "running",
                "seed": seed,
                "updated_at": _utc_now(),
            }
        },
        "stage2_version": "rollback_stage1_teacher_bank_verifier_v1",
    }
    _write_json(progress_path, progress)
    start_ts = time.time()
    teacher_bank = build_teacher_bank(dataset_name=dataset_name, items=train_items)
    _write_jsonl(dataset_root / "teacher_bank.jsonl", teacher_bank_to_jsonl(teacher_bank))
    verifier = train_verifier_model(teacher_bank, device=device, seed=seed)
    torch.save(
        {
            "feature_names": verifier.feature_names,
            "state_dict": verifier.model_state,
            "train_loss": verifier.train_loss,
            "train_accuracy": verifier.train_accuracy,
            "positive_rate": verifier.positive_rate,
            "count": verifier.count,
        },
        dataset_root / "verifier.pt",
    )
    metrics = {
        "dataset": dataset_name,
        "seed": seed,
        "device": device,
        "count": verifier.count,
        "train_loss": verifier.train_loss,
        "train_accuracy": verifier.train_accuracy,
        "positive_rate": verifier.positive_rate,
        "rerun_needed_count": int(sum(int(row.rerun_needed) for row in teacher_bank)),
        "anchor_contract_valid_rate": float(
            sum(1.0 for row in teacher_bank if row.anchor_contract_valid) / max(1, len(teacher_bank))
        ),
        "anchor_recoverable_valid_rate": float(
            sum(1.0 for row in teacher_bank if row.anchor_recoverable_valid) / max(1, len(teacher_bank))
        ),
        "elapsed_s": time.time() - start_ts,
    }
    _write_json(dataset_root / "metrics.json", metrics)
    progress["datasets"][dataset_name]["status"] = "completed"
    progress["datasets"][dataset_name]["updated_at"] = _utc_now()
    progress["datasets"][dataset_name]["metrics_path"] = str(dataset_root / "metrics.json")
    _write_json(progress_path, progress)
    print(json.dumps(metrics, ensure_ascii=False), flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train stage2-rollback stage1 teacher-bank verifier.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-train", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    os.environ.setdefault("LLM_API_BASE", "http://127.0.0.1:8039")
    datasets = list(args.dataset) if args.dataset else list_processed_datasets(args.data_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = _device_from_arg(args.device)
    summaries = []
    for dataset_name in datasets:
        summaries.append(
            run_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                output_root=output_root,
                max_train=args.max_train,
                seed=args.seed,
                device=device,
            )
        )
    _write_json(output_root / "suite_report.json", {"datasets": summaries, "saved_at": _utc_now()})


if __name__ == "__main__":
    main()
