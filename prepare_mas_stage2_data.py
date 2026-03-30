from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from mas_treesearch import list_processed_datasets, load_processed_split


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_target_suite_20260314",
        help="Stage-1 processed dataset root containing train/validation/test splits.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_stage2_target_suite_from_stage1_test",
        help="Output root for stage-2 train/validation/test view.",
    )
    parser.add_argument("--dataset", action="append", default=[], help="Dataset to prepare. Repeatable.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--train-multiplier",
        type=float,
        default=1.0,
        help="Multiplier applied to the stage-1 train count when sampling stage-2 train from stage-1 test.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = list(args.dataset) if args.dataset else list_processed_datasets(str(source_root))
    manifest: Dict[str, Any] = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "policy": {
            "train_source_split": "stage1_test",
            "train_size": "stage1_train_count * train_multiplier (clipped by stage1_test_count)",
            "train_multiplier": args.train_multiplier,
            "validation_source_split": "stage1_validation",
            "test_source_split": "remaining_stage1_test",
        },
        "datasets": {},
    }

    for index, dataset_name in enumerate(datasets):
        train_source = load_processed_split(str(source_root), dataset_name, "train")
        validation_source = load_processed_split(str(source_root), dataset_name, "validation")
        test_source = load_processed_split(str(source_root), dataset_name, "test")
        requested_train = int(round(len(train_source) * max(0.0, args.train_multiplier)))
        train_limit = min(requested_train, len(test_source))
        sampled_test = _sample(test_source, len(test_source), args.seed + index * 100)
        stage2_train = sampled_test[:train_limit]
        remaining_test = sampled_test[train_limit:]
        dataset_dir = output_root / dataset_name
        _write_jsonl(dataset_dir / "train.jsonl", stage2_train)
        _write_jsonl(dataset_dir / "validation.jsonl", list(validation_source))
        _write_jsonl(dataset_dir / "test.jsonl", remaining_test)
        manifest["datasets"][dataset_name] = {
            "stage1_counts": {
                "train": len(train_source),
                "validation": len(validation_source),
                "test": len(test_source),
            },
            "stage2_counts": {
                "train": len(stage2_train),
                "validation": len(validation_source),
                "test": len(remaining_test),
            },
            "stage2_train_ids": [str(item.get("id", "")) for item in stage2_train],
            "stage2_test_ids": [str(item.get("id", "")) for item in remaining_test],
        }
    _write_json(output_root / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
