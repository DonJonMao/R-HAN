from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Iterable

from prepare_mas_treesearch_target_suite import _collect_math


def _write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-root",
        default="/mnt/nvme/projects/math/MATH",
        help="Root directory of the raw MATH tree with train/ and test/ subdirectories.",
    )
    parser.add_argument(
        "--dataset-root",
        default="/mnt/nvme/projects/R-HAN/dataset/math",
        help="Destination root. train/validation/test JSONL files are written here.",
    )
    parser.add_argument(
        "--copy-raw-to-dataset",
        action="store_true",
        help="Copy the raw MATH tree into <dataset-root>/MATH before processing.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    dataset_root = Path(args.dataset_root)
    dataset_root.mkdir(parents=True, exist_ok=True)

    if args.copy_raw_to_dataset:
        copied_raw_root = dataset_root / "MATH"
        if copied_raw_root.resolve() != raw_root.resolve():
            shutil.copytree(raw_root, copied_raw_root, dirs_exist_ok=True)
        raw_root = copied_raw_root

    split_rows = _collect_math(raw_root)
    counts = {
        split: _write_jsonl(dataset_root / f"{split}.jsonl", split_rows.get(split, []))
        for split in ("train", "validation", "test")
    }

    manifest = {
        "dataset": "math",
        "raw_root": str(raw_root),
        "dataset_root": str(dataset_root),
        "counts": counts,
    }
    manifest_path = dataset_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
