from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

from mas_treesearch.data import standardize_record
from mas_treesearch.profiles import profile_summary


TARGET_DATASETS = (
    "gsm8k",
    "math",
    "multiarith",
    "humaneval",
    "mbpp",
    "mmlu",
    "nlgraph",
    "knowledge_crosswords",
    "normad",
)

RESTANDARDIZE_FROM_PROCESSED = (
    "gsm8k",
    "mmlu",
    "nlgraph",
    "knowledge_crosswords",
    "normad",
)


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


def _stable_split(key: str) -> str:
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "validation"
    return "test"


def _restandardize_processed_record(row: Dict[str, Any], split: str) -> Dict[str, Any]:
    base = dict(row)
    base["question"] = row.get("original_question") or row.get("question") or ""
    base["answer"] = row.get("original_answer") or row.get("answer") or ""
    base["metadata"] = dict(row.get("metadata") or {})
    return standardize_record(base, split)


def _collect_processed_dataset(processed_root: Path, dataset_name: str) -> Dict[str, List[Dict[str, Any]]]:
    split_rows: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split in split_rows:
        path = processed_root / dataset_name / f"{split}.jsonl"
        if not path.exists():
            continue
        split_rows[split] = [
            _restandardize_processed_record(row, split)
            for row in _load_jsonl(path)
        ]
    return split_rows


def _extract_mbpp_entry_point(code: str) -> str:
    match = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", code)
    return match.group(1) if match else ""


def _collect_multiarith(raw_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(raw_path.read_text(encoding="utf-8"))
    grouped: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in data:
        idx = int(row["iIndex"])
        split = _stable_split(f"multiarith:{idx}")
        raw_record = {
            "id": f"multiarith:{idx}",
            "source_dataset": "multiarith",
            "category": "Reasoning",
            "question": str(row.get("sQuestion", "")).strip(),
            "answer": str((row.get("lSolutions") or [""])[0]),
            "metadata": {
                "equations": list(row.get("lEquations") or []),
                "solutions": list(row.get("lSolutions") or []),
                "alignments": list(row.get("lAlignments") or []),
            },
        }
        grouped[split].append(standardize_record(raw_record, split))
    return grouped


def _collect_humaneval(raw_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in _load_jsonl(raw_path):
        task_id = str(row.get("task_id", "")).strip()
        prompt = str(row.get("prompt", ""))
        canonical_solution = str(row.get("canonical_solution", ""))
        split = _stable_split(task_id)
        raw_record = {
            "id": task_id,
            "source_dataset": "humaneval",
            "category": "Code",
            "question": prompt.strip(),
            "answer": f"{prompt}{canonical_solution}".rstrip(),
            "metadata": {
                "entry_point": str(row.get("entry_point", "")).strip(),
                "test": str(row.get("test", "")),
                "canonical_solution": canonical_solution,
            },
        }
        grouped[split].append(standardize_record(raw_record, split))
    return grouped


def _collect_mbpp(raw_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for row in _load_jsonl(raw_path):
        task_id = int(row.get("task_id", -1))
        code = str(row.get("code", "")).replace("\r\n", "\n").strip()
        split = _stable_split(f"mbpp:{task_id}")
        raw_record = {
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
            },
        }
        grouped[split].append(standardize_record(raw_record, split))
    return grouped


def _write_dataset(output_root: Path, dataset_name: str, split_rows: Dict[str, List[Dict[str, Any]]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    dataset_dir = output_root / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation", "test"):
        counts[split] = _write_jsonl(dataset_dir / f"{split}.jsonl", split_rows.get(split, []))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--processed-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_processed",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_target_suite_20260314",
    )
    args = parser.parse_args()

    repo_root = Path("/mnt/nvme/projects/R-HAN")
    processed_root = Path(args.processed_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "target_datasets": list(TARGET_DATASETS),
        "output_root": str(output_root),
        "datasets": {},
        "missing_datasets": {},
    }

    for dataset_name in RESTANDARDIZE_FROM_PROCESSED:
        split_rows = _collect_processed_dataset(processed_root, dataset_name)
        counts = _write_dataset(output_root, dataset_name, split_rows)
        manifest["datasets"][dataset_name] = {
            "source": "processed_restandardized",
            "counts": counts,
            "profile": profile_summary([dataset_name])[dataset_name],
        }

    builders = {
        "multiarith": _collect_multiarith(repo_root / "dataset/MultiArith.json"),
        "humaneval": _collect_humaneval(repo_root / "dataset/HumanEval.jsonl"),
        "mbpp": _collect_mbpp(repo_root / "dataset/mbpp/mbpp.jsonl"),
    }
    for dataset_name, split_rows in builders.items():
        counts = _write_dataset(output_root, dataset_name, split_rows)
        manifest["datasets"][dataset_name] = {
            "source": "raw_local",
            "counts": counts,
            "profile": profile_summary([dataset_name])[dataset_name],
        }

    math_data_root = repo_root / "dataset/math/MATH"
    if math_data_root.exists():
        manifest["missing_datasets"]["math"] = {
            "reason": "builder not implemented for local MATH tree yet",
            "path": str(math_data_root),
        }
    else:
        manifest["missing_datasets"]["math"] = {
            "reason": "raw MATH problem files are not present locally",
            "expected_path": str(math_data_root),
        }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
