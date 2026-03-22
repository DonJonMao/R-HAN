from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mas_treesearch.data import standardize_record
from mas_treesearch.profiles import profile_summary


THREEWAY_DATASETS = ("mmlu_pro", "nlgraph", "knowledge_crosswords")

THREEWAY_PLAN: Dict[str, Dict[str, int]] = {
    "mmlu_pro": {"stage1_train": 1200, "stage2_train": 1200, "final_eval": 1500},
    "nlgraph": {"stage1_train": 500, "stage2_train": 500, "final_eval": 700},
    "knowledge_crosswords": {"stage1_train": 250, "stage2_train": 250, "final_eval": 350},
}

THREEWAY_SPLITS = ("stage1_train", "stage2_train", "final_eval")
SOURCE_SPLITS = ("train", "validation", "test")


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
    bucket = int(hashlib.sha1(f"threeway:{key}".encode("utf-8")).hexdigest()[:12], 16)
    return bucket, key


def _restandardize_processed_record(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    base = dict(row)
    base["question"] = row.get("original_question") or row.get("question") or ""
    base["answer"] = row.get("original_answer") or row.get("answer") or ""
    base["metadata"] = dict(row.get("metadata") or {})
    return standardize_record(base, split)


def _has_usable_reference(dataset_name: str, row: Dict[str, Any]) -> bool:
    if dataset_name != "knowledge_crosswords":
        return True
    metadata = row.get("metadata") or {}
    answer_all = metadata.get("answer_all")
    if isinstance(answer_all, list) and answer_all:
        return True
    answer = str(row.get("original_answer") or row.get("answer") or "").strip()
    if not answer or answer == "None":
        return False
    try:
        parsed = json.loads(answer)
    except Exception:
        return False
    return isinstance(parsed, list) and bool(parsed)


def _collect_processed_pool(processed_root: Path, dataset_name: str) -> List[Dict[str, Any]]:
    deduped: Dict[str, Dict[str, Any]] = {}
    for split in SOURCE_SPLITS:
        path = processed_root / dataset_name / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in _load_jsonl(path):
            row_id = str(row.get("id", "")).strip()
            if not row_id:
                continue
            standardized = _restandardize_processed_record(row, split)
            current = deduped.get(row_id)
            if current is None:
                deduped[row_id] = standardized
                continue
            current_usable = _has_usable_reference(dataset_name, current)
            new_usable = _has_usable_reference(dataset_name, standardized)
            if new_usable and not current_usable:
                deduped[row_id] = standardized
            elif current_usable and not new_usable:
                continue
            else:
                deduped[row_id] = standardized
    if dataset_name == "knowledge_crosswords":
        deduped = {
            row_id: row
            for row_id, row in deduped.items()
            if _has_usable_reference(dataset_name, row)
        }
    return sorted(deduped.values(), key=lambda row: _hash_sort_key(str(row["id"])))


def _threeway_partition(rows: List[Dict[str, Any]], plan: Dict[str, int]) -> Dict[str, List[Dict[str, Any]]]:
    required = sum(int(plan[split]) for split in THREEWAY_SPLITS)
    if len(rows) < required:
        raise ValueError(f"Not enough rows: have {len(rows)}, need {required}")
    stage1_end = int(plan["stage1_train"])
    stage2_end = stage1_end + int(plan["stage2_train"])
    final_end = stage2_end + int(plan["final_eval"])

    partition = {
        "stage1_train": rows[:stage1_end],
        "stage2_train": rows[stage1_end:stage2_end],
        "final_eval": rows[stage2_end:final_end],
    }
    result: Dict[str, List[Dict[str, Any]]] = {}
    for split, split_rows in partition.items():
        result[split] = [_restandardize_processed_record(row, split) for row in split_rows]
    return result


def _write_dataset(output_root: Path, dataset_name: str, split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split in THREEWAY_SPLITS:
        counts[split] = _write_jsonl(dataset_dir / f"{split}.jsonl", split_rows.get(split, []))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed",
        help="Processed dataset root containing train/validation/test jsonl files.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_threeway_general_20260319",
        help="Output root for the new three-way split datasets.",
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    for existing_dir in output_root.iterdir():
        if existing_dir.is_dir() and existing_dir.name not in THREEWAY_DATASETS:
            shutil.rmtree(existing_dir)

    manifest: Dict[str, Any] = {
        "output_root": str(output_root),
        "protocol": {
            "stage1_train": "Used only for union-graph / topology generation.",
            "stage2_train": "Reserved for later three-layer runtime training.",
            "final_eval": "Shared held-out evaluation split for both stages. Must not be used for tuning.",
        },
        "datasets": {},
    }

    for dataset_name in THREEWAY_DATASETS:
        pool = _collect_processed_pool(processed_root, dataset_name)
        plan = dict(THREEWAY_PLAN[dataset_name])
        split_rows = _threeway_partition(pool, plan)
        counts = _write_dataset(output_root, dataset_name, split_rows)
        manifest["datasets"][dataset_name] = {
            "plan": plan,
            "counts": counts,
            "available_pool": len(pool),
            "unused_pool": len(pool) - sum(counts.values()),
            "source_root": str(processed_root / dataset_name),
            "profile": profile_summary([dataset_name])[dataset_name],
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
