from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mas_treesearch.data import standardize_record
from mas_treesearch.profiles import profile_summary


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows_list:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows_list)


def _hash_sort_key(key: str) -> tuple[int, str]:
    bucket = int(hashlib.sha1(f"gsm8k-official:{key}".encode("utf-8")).hexdigest()[:12], 16)
    return (bucket, key)


def _build_official_test_records(raw_rows: List[Dict[str, Any]], source_path: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for idx, row in enumerate(raw_rows):
        uid = f"gsm8k:official_test:{idx}"
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        records.append(
            {
                "id": uid,
                "source_dataset": "gsm8k",
                "category": "Knowledge",
                "question": question,
                "answer": answer,
                "metadata": {
                    "uid": uid,
                    "path": str(source_path),
                    "official_split": "test",
                    "official_index": idx,
                    "official_dataset": "openai_grade_school_math",
                    "official_subset": "test_1319",
                    "split_policy": "stable_hash_20_80_over_official_test_only",
                },
            }
        )
    return records


def _repartition_official_test(records: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    sorted_rows = sorted(records, key=lambda row: _hash_sort_key(str(row["id"])))
    if not sorted_rows:
        return split_rows
    train_count = max(1, int(round(len(sorted_rows) * 0.2)))
    train_rows = sorted_rows[:train_count]
    test_rows = sorted_rows[train_count:]
    split_rows["train"] = [standardize_record(row, "train") for row in train_rows]
    split_rows["test"] = [standardize_record(row, "test") for row in test_rows]
    return split_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare a MAS-ready GSM8K dataset by repartitioning the official 1319-item test set into a stable 20/80 split."
    )
    parser.add_argument(
        "--source-test",
        default="/mnt/nvme/projects/R-HAN/dataset/grade-school-math/grade_school_math/data/test.jsonl",
        help="Path to the official GSM8K test.jsonl file.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_gsm8k_official_test_20_80_20260323",
        help="Output root that will contain gsm8k/train.jsonl, validation.jsonl, and test.jsonl.",
    )
    args = parser.parse_args()

    source_test = Path(args.source_test)
    if not source_test.exists():
        raise FileNotFoundError(f"Official GSM8K test file not found: {source_test}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    raw_rows = _load_jsonl(source_test)
    records = _build_official_test_records(raw_rows, source_test)
    split_rows = _repartition_official_test(records)

    dataset_dir = output_root / "gsm8k"
    counts = {
        "train": _write_jsonl(dataset_dir / "train.jsonl", split_rows["train"]),
        "validation": _write_jsonl(dataset_dir / "validation.jsonl", split_rows["validation"]),
        "test": _write_jsonl(dataset_dir / "test.jsonl", split_rows["test"]),
    }

    manifest: Dict[str, Any] = {
        "output_root": str(output_root),
        "dataset": "gsm8k",
        "source": {
            "path": str(source_test),
            "dataset": "openai_grade_school_math",
            "split": "official_test",
            "count": len(raw_rows),
        },
        "split_protocol": {
            "train": "20% stable-hash split from official GSM8K test",
            "validation": "empty",
            "test": "80% held-out split from official GSM8K test",
        },
        "counts": counts,
        "profile": profile_summary(["gsm8k"])["gsm8k"],
        "notes": [
            "This root is derived only from the official GSM8K 1319-item test set.",
            "It is intended for internal AFlow-style search/evaluation, not for leaderboard reporting.",
            "Because 1319 is not divisible exactly by 5, the realized split is 264 train and 1055 test.",
        ],
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
