from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mas_treesearch.data import load_jsonl, standardize_record
from mas_treesearch.profiles import profile_summary


AFLOW_DATASETS = ("humaneval", "mbpp", "gsm8k", "math")
AFLOW_MATH_CATEGORIES = {
    "countingandprobability": "Counting & Probability",
    "countingprobability": "Counting & Probability",
    "numbertheory": "Number Theory",
    "prealgebra": "Pre-algebra",
    "precalculus": "Pre-calculus",
}
AFLOW_MATH_TARGET_SIZE = 617


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
    bucket = int(hashlib.sha1(f"sort:{key}".encode("utf-8")).hexdigest()[:12], 16)
    return (bucket, key)


def _write_dataset(output_root: Path, dataset_name: str, split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for split in ("train", "validation", "test"):
        counts[split] = _write_jsonl(dataset_dir / f"{split}.jsonl", split_rows.get(split, []))
    return counts


def _extract_mbpp_entry_point(code: str) -> str:
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return match.group(1) if match else ""


def _repartition_rows(raw_rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    rows_list = sorted(list(raw_rows), key=lambda row: _hash_sort_key(str(row["id"])))
    if not rows_list:
        return split_rows
    search_count = max(1, int(round(len(rows_list) * 0.2)))
    search_rows = rows_list[:search_count]
    test_rows = rows_list[search_count:]
    for row in search_rows:
        split_rows["train"].append(standardize_record(row, "train"))
    for row in test_rows:
        split_rows["test"].append(standardize_record(row, "test"))
    return split_rows


def _collect_humaneval(repo_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw_path = repo_root / "dataset/HumanEval.jsonl"
    rows: List[Dict[str, Any]] = []
    for row in _load_jsonl(raw_path):
        task_id = str(row.get("task_id", "")).strip()
        prompt = str(row.get("prompt", ""))
        canonical_solution = str(row.get("canonical_solution", ""))
        rows.append(
            {
                "id": task_id,
                "source_dataset": "humaneval",
                "category": "Code",
                "question": prompt.strip(),
                "answer": f"{prompt}{canonical_solution}".rstrip(),
                "metadata": {
                    "entry_point": str(row.get("entry_point", "")).strip(),
                    "test": str(row.get("test", "")),
                    "canonical_solution": canonical_solution,
                    "aflow_split_rule": "stable_hash_20_80",
                },
            }
        )
    return _repartition_rows(rows)


def _collect_mbpp(repo_root: Path) -> Dict[str, List[Dict[str, Any]]]:
    raw_path = repo_root / "dataset/mbpp/mbpp.jsonl"
    rows: List[Dict[str, Any]] = []
    for row in _load_jsonl(raw_path):
        task_id = int(row.get("task_id", -1))
        code = str(row.get("code", "")).replace("\r\n", "\n").strip()
        rows.append(
            {
                "id": f"mbpp:{task_id}",
                "source_dataset": "mbpp",
                "category": "Code",
                "question": str(row.get("text", "")).strip(),
                "answer": code,
                "metadata": {
                    "entry_point": _extract_mbpp_entry_point(code),
                    "test_setup_code": str(row.get("test_setup_code", "")).replace("\r\n", "\n"),
                    "test_list": list(row.get("test_list") or []),
                    "challenge_test_list": list(row.get("challenge_test_list") or []),
                    "aflow_public_test_rule": "first_test_case_as_public_test",
                    "aflow_split_rule": "stable_hash_20_80",
                },
            }
        )
    return _repartition_rows(rows)


def _collect_gsm8k(repo_root: Path, processed_root: Path) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    seen_ids: set[str] = set()
    source_dir = processed_root / "gsm8k"
    for split in ("train", "validation", "test"):
        path = source_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        for row in load_jsonl(str(path)):
            row_id = str(row.get("id", "")).strip()
            if not row_id or row_id in seen_ids:
                continue
            seen_ids.add(row_id)
            rows.append(
                {
                    "id": row_id,
                    "source_dataset": "gsm8k",
                    "category": str(row.get("category", "Reasoning")),
                    "question": str(row.get("original_question") or row.get("question") or "").strip(),
                    "answer": str(row.get("original_answer") or row.get("answer") or "").strip(),
                    "metadata": dict(row.get("metadata") or {}) | {
                        "aflow_split_rule": "stable_hash_20_80",
                        "aflow_source_note": "fallback_from_local_processed_gsm8k_pool",
                    },
                }
            )
    return _repartition_rows(rows), {
        "source": str(source_dir),
        "row_count": len(rows),
        "note": "Used local processed GSM8K pool because canonical raw GSM8K jsonl files were not present in the repository.",
    }


def _normalize_math_category(value: str) -> str:
    lowered = value.strip().lower()
    return re.sub(r"[^a-z]", "", lowered)


def _largest_remainder_alloc(sizes: Dict[str, int], total_target: int) -> Dict[str, int]:
    total_size = sum(sizes.values())
    if total_size <= 0:
        return {name: 0 for name in sizes}
    fractional: Dict[str, float] = {}
    alloc: Dict[str, int] = {}
    used = 0
    for name, size in sizes.items():
        raw = total_target * (size / total_size)
        base = int(raw)
        alloc[name] = base
        fractional[name] = raw - base
        used += base
    remaining = total_target - used
    for name, _ in sorted(fractional.items(), key=lambda item: (-item[1], item[0])):
        if remaining <= 0:
            break
        alloc[name] += 1
        remaining -= 1
    return alloc


def _collect_math(repo_root: Path) -> tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Any]]:
    math_root = repo_root / "dataset/math/MATH"
    by_category: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for source_split in ("train", "test"):
        split_root = math_root / source_split
        if not split_root.exists():
            continue
        for path in split_root.rglob("*.json"):
            row = json.loads(path.read_text(encoding="utf-8"))
            raw_category = str(row.get("type", "")).strip() or path.parent.name
            category_key = _normalize_math_category(raw_category)
            if category_key not in AFLOW_MATH_CATEGORIES:
                continue
            if str(row.get("level", "")).strip() != "Level 5":
                continue
            rel_path = path.relative_to(math_root).as_posix()
            by_category[category_key].append(
                {
                    "id": f"math_aflow:{rel_path}",
                    "source_dataset": "math",
                    "category": AFLOW_MATH_CATEGORIES[category_key],
                    "question": str(row.get("problem", "")).strip(),
                    "answer": str(row.get("solution", "")).strip(),
                    "metadata": {
                        "path": str(path),
                        "relative_path": rel_path,
                        "source_split": source_split,
                        "level": str(row.get("level", "")).strip(),
                        "type": raw_category,
                        "aflow_math_filter": "level5_four_categories",
                    },
                }
            )

    sizes = {name: len(rows) for name, rows in by_category.items()}
    alloc = _largest_remainder_alloc(sizes, AFLOW_MATH_TARGET_SIZE)
    selected_rows: List[Dict[str, Any]] = []
    for category_key, rows in by_category.items():
        picked = sorted(rows, key=lambda row: _hash_sort_key(str(row["id"])))[: alloc[category_key]]
        selected_rows.extend(picked)
    selected_rows = sorted(selected_rows, key=lambda row: _hash_sort_key(str(row["id"])))
    repartitioned = _repartition_rows(selected_rows)
    return repartitioned, {
        "source": str(math_root),
        "candidate_count": sum(sizes.values()),
        "selected_count": len(selected_rows),
        "per_category_candidates": {AFLOW_MATH_CATEGORIES[k]: v for k, v in sizes.items()},
        "per_category_selected": {AFLOW_MATH_CATEGORIES[k]: v for k, v in alloc.items()},
        "note": "Approximate AFlow MATH subset: level-5 four-category filter plus deterministic stratified sample to 617 items.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed",
        help="Processed dataset root used as fallback source for GSM8K.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_aflow_four_20260318",
        help="Directory where AFlow-style 20/80 splits will be written.",
    )
    args = parser.parse_args()

    repo_root = Path("/mnt/nvme/projects/R-HAN")
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "output_root": str(output_root),
        "datasets": list(AFLOW_DATASETS),
        "split_protocol": {
            "train": "20% search/optimization split",
            "validation": "empty",
            "test": "80% held-out evaluation split",
        },
        "notes": [
            "This data root is for AFlow-style offline structure search, not standard supervised train/validation/test.",
            "The train split corresponds to the paper's workflow-search subset.",
        ],
        "dataset_details": {},
    }

    humaneval_rows = _collect_humaneval(repo_root)
    manifest["dataset_details"]["humaneval"] = {
        "counts": _write_dataset(output_root, "humaneval", humaneval_rows),
        "profile": profile_summary(["humaneval"])["humaneval"],
        "source": str(repo_root / "dataset/HumanEval.jsonl"),
    }

    mbpp_rows = _collect_mbpp(repo_root)
    manifest["dataset_details"]["mbpp"] = {
        "counts": _write_dataset(output_root, "mbpp", mbpp_rows),
        "profile": profile_summary(["mbpp"])["mbpp"],
        "source": str(repo_root / "dataset/mbpp/mbpp.jsonl"),
    }

    gsm8k_rows, gsm8k_meta = _collect_gsm8k(repo_root, processed_root)
    manifest["dataset_details"]["gsm8k"] = {
        "counts": _write_dataset(output_root, "gsm8k", gsm8k_rows),
        "profile": profile_summary(["gsm8k"])["gsm8k"],
    } | gsm8k_meta

    math_rows, math_meta = _collect_math(repo_root)
    manifest["dataset_details"]["math"] = {
        "counts": _write_dataset(output_root, "math", math_rows),
        "profile": profile_summary(["math"])["math"],
    } | math_meta

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
