from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _checkpoint_paths(root: Path) -> List[Path]:
    return sorted(root.glob("workers/worker_*/*/checkpoint.json"))


def _load_rows(root: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in _checkpoint_paths(root):
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        metadata = payload.get("metadata", {})
        for row in metadata.get("train_rows", []) or []:
            if isinstance(row, dict):
                item = dict(row)
                item["_checkpoint"] = str(path)
                rows.append(item)
    return rows


def _sum_float(rows: Iterable[Dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        try:
            total += float(row.get(key, 0.0))
        except (TypeError, ValueError):
            continue
    return total


def _stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    stage1_correct = _sum_float(rows, "stage1_success")
    stage2_correct = _sum_float(rows, "stage2_success")
    stage1_task = _sum_float(rows, "stage1_task_score")
    stage2_task = _sum_float(rows, "stage2_task_score")
    wrong_to_right = [
        row for row in rows
        if float(row.get("stage1_success", 0.0)) < 1.0 and float(row.get("stage2_success", 0.0)) >= 1.0
    ]
    right_to_wrong = [
        row for row in rows
        if float(row.get("stage1_success", 0.0)) >= 1.0 and float(row.get("stage2_success", 0.0)) < 1.0
    ]
    return {
        "n": n,
        "unique_ids": len({str(row.get("id", "")) for row in rows}),
        "wrong_to_right": len(wrong_to_right),
        "right_to_wrong": len(right_to_wrong),
        "better": sum(1 for row in rows if row.get("stage2_vs_stage1_outcome") == "better"),
        "worse": sum(1 for row in rows if row.get("stage2_vs_stage1_outcome") == "worse"),
        "same": sum(1 for row in rows if row.get("stage2_vs_stage1_outcome") == "same"),
        "stage1_correct": stage1_correct,
        "stage2_correct": stage2_correct,
        "stage1_acc": stage1_correct / n if n else 0.0,
        "stage2_acc": stage2_correct / n if n else 0.0,
        "delta_correct": stage2_correct - stage1_correct,
        "task_delta": stage2_task - stage1_task,
        "wrong_to_right_ids": [str(row.get("id", "")) for row in wrong_to_right],
        "right_to_wrong_ids": [str(row.get("id", "")) for row in right_to_wrong],
    }


def _by_dataset(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    datasets = sorted({str(row.get("dataset", "")) for row in rows if row.get("dataset")})
    payload = {dataset: _stats([row for row in rows if str(row.get("dataset", "")) == dataset]) for dataset in datasets}
    payload["total"] = _stats(rows)
    return payload


def _markdown(report: Dict[str, Any]) -> str:
    lines = [
        "# TopK + Qwen3 Composer vs bdc1ae0",
        "",
        f"Baseline: `{report['baseline_root']}`",
        f"Candidate: `{report['candidate_root']}`",
        "",
        "| Run | Dataset | n | wrong->right | right->wrong | stage2 correct | delta correct | task delta |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run_name in ("baseline", "candidate"):
        for dataset in ("mbpp", "humaneval", "total"):
            stats = report[run_name].get(dataset, {})
            lines.append(
                "| {run} | {dataset} | {n} | {w2r} | {r2w} | {s2:.0f} | {dc:.0f} | {td:.3f} |".format(
                    run=run_name,
                    dataset=dataset,
                    n=stats.get("n", 0),
                    w2r=stats.get("wrong_to_right", 0),
                    r2w=stats.get("right_to_wrong", 0),
                    s2=float(stats.get("stage2_correct", 0.0)),
                    dc=float(stats.get("delta_correct", 0.0)),
                    td=float(stats.get("task_delta", 0.0)),
                )
            )
    lines.extend(["", "## Wrong-To-Right IDs", ""])
    for run_name in ("baseline", "candidate"):
        lines.append(f"### {run_name}")
        for dataset in ("mbpp", "humaneval"):
            stats = report[run_name].get(dataset, {})
            ids = stats.get("wrong_to_right_ids", [])
            lines.append(f"- {dataset}: {', '.join(ids) if ids else '(none)'}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--markdown", required=True)
    args = parser.parse_args()

    baseline_root = Path(args.baseline)
    candidate_root = Path(args.candidate)
    report = {
        "baseline_root": str(baseline_root),
        "candidate_root": str(candidate_root),
        "baseline": _by_dataset(_load_rows(baseline_root)),
        "candidate": _by_dataset(_load_rows(candidate_root)),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    Path(args.markdown).write_text(_markdown(report), encoding="utf-8")


if __name__ == "__main__":
    main()
