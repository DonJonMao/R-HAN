#!/usr/bin/env python3
"""Collect paper-ready R-HAN case-study artifacts from experiment outputs.

The script searches R-HAN outputs for final-design rows, scores candidate
examples using the case-study heuristics, and emits markdown/JSON/CSV artifacts
without relying on report.json preview examples.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable


DATASETS = {"mmlu_pro", "mbpp"}
ROW_FILES = {"train_rows.jsonl", "validation_rows.jsonl", "test_rows.jsonl"}

COMMON_FIELDS = [
    "id",
    "dataset",
    "split",
    "category",
    "question",
    "reference_answer",
    "gold_answer",
    "final_output",
    "stage1_signature",
    "stage2_signature",
    "signature",
    "reward",
    "task_score",
    "success",
    "stage1_reward",
    "stage2_reward",
    "stage1_task_score",
    "stage2_task_score",
    "stage1_success",
    "stage2_success",
    "stage2_vs_stage1_outcome",
    "selection_decision",
    "selection_reason",
    "final_selection_decision",
    "final_selection_reason",
    "fallback_applied",
    "finalizer_strategy",
    "structure_source",
    "stage2_version",
    "stage2_turns",
    "stage2_memory_records",
    "stage1_latency",
    "stage2_latency",
    "latency",
    "stage1_token_cost",
    "stage2_token_cost",
    "token_cost",
    "stage2_turn_token_costs",
    "stage2_turn_token_estimates",
    "active_edge_ratio_by_turn",
    "active_node_ratio_by_turn",
    "candidate_provenance_coverage",
    "recovery_subgraph_size",
    "final_answer_source_type",
    "v4_4_route_family",
    "v4_4_execution_mode",
    "v4_4_stage1_anchor_present",
    "v4_4_stage1_anchor_used",
    "v4_4_candidate_count",
    "v4_4_collapsed_class_count",
    "v4_4_selected_candidate_source",
    "v4_4_selected_candidate_digest",
    "v4_4_selected_quality_score",
    "v4_4_selected_model_uncertainty",
    "v4_4_top_candidates",
    "v4_4_top_classes",
]

MMLU_EXTRA_FIELDS = [
    "discrete_slot_count",
    "discrete_output_kind",
    "discrete_assignment",
    "discrete_assignment_signature",
    "discrete_fatal_count",
    "discrete_local_count",
    "discrete_fatal_kinds",
    "discrete_local_kinds",
    "discrete_invalid_slots",
    "discrete_unstable_slots",
    "discrete_probe_triggered",
    "discrete_probe_winner",
    "discrete_probe_confidence",
    "discrete_update_accepted",
    "discrete_update_slots",
    "discrete_challenger_proposal_triggered",
    "discrete_challenger_proposal_count",
    "discrete_challenger_source",
    "discrete_pairwise_reject_reason",
    "discrete_target_polarity",
    "discrete_probe_question_polarity",
    "discrete_probe_target_condition",
    "discrete_probe_inverse_condition",
    "discrete_probe_discriminator",
    "discrete_probe_anchor_satisfies_target",
    "discrete_probe_challenger_satisfies_target",
    "discrete_probe_challenger_satisfies_inverse",
    "discrete_audit_triggered",
    "discrete_audit_winner",
    "discrete_audit_confidence",
    "certificate_kind",
    "candidate_universe_size",
    "candidate_bank_size",
    "factor_count",
    "cert_bank_size",
    "option_matrix_yes_votes",
    "option_matrix_no_votes",
    "evidence_atom_count",
    "anchor_conflict_count",
    "challenger_support_count",
    "shared_discriminator_count",
    "score_margin",
    "vote_margin",
    "audit_agree",
    "calibrator_p_accept",
    "target_condition_consistency",
    "joint_update_slots",
    "accept_blocker",
    "rubric_parse_status",
    "rubric_target_condition_nonempty",
    "matrix_raw_count",
    "matrix_parse_status",
    "matrix_row_count",
    "matrix_option_covered_count",
    "matrix_factor_eval_count",
    "anchor_eval_status_hist",
    "candidate_eval_status_hist",
    "same_factor_flip_count",
    "factor_group_flip_count",
    "support_lift_count",
    "conflict_lift_count",
    "empty_cert_reason",
    "top_candidate_value",
    "top_candidate_soft_score",
    "top_candidate_support_count",
    "top_candidate_anchor_conflict_count",
    "rescue_triggered",
    "rescue_cert_count",
    "rescue_parse_status",
]

MBPP_EXTRA_FIELDS = [
    "function_signature",
    "visible_tests",
    "stage1_code",
    "final_selected_code",
    "challenger_or_repair_code",
    "code_diff",
    "v4_4_selected_is_repair_branch",
    "v4_4_selected_visible_tests_passed",
    "v4_4_selected_visible_tests_total",
    "v4_4_selected_failure_kind",
    "v4_4_stage1_anchor_visible_tests_passed",
    "v4_4_stage1_anchor_visible_tests_total",
    "v4_4_stage1_anchor_failure_kind",
    "v4_4_repair_rounds_run",
    "v4_4_repair_branch_count",
    "v4_4_repair_improvement_count",
    "v4_4_restart_count",
    "failing_examples",
    "stdout",
    "stderr",
    "exec_error",
]

INDEX_COLUMNS = [
    "case_name",
    "dataset",
    "split",
    "id",
    "category",
    "route_family",
    "phenomenon",
    "stage1_success",
    "stage2_success",
    "final_success",
    "stage2_vs_stage1_outcome",
    "stage1_answer_or_code_digest",
    "stage2_answer_or_code_digest",
    "gold",
    "selection_decision",
    "selection_reason",
    "stage1_token_cost",
    "stage2_token_cost",
    "total_token_cost",
    "latency",
    "active_edge_ratio_by_turn",
    "active_node_ratio_by_turn",
    "stage2_memory_records",
    "candidate_count",
    "collapsed_class_count",
    "key_verifier_signal",
    "replay_path",
    "raw_json_path",
    "markdown_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=Path("/mnt/nvme/projects/R-HAN/outputs"),
        help="R-HAN outputs directory to scan.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("/mnt/nvme/projects/R-HAN/outputs/case_studies/final_design_cases_20260517"),
        help="Directory for generated case-study artifacts.",
    )
    parser.add_argument(
        "--allow-fallback-final-design-runs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use other stage2_final_design runs if the primary run lacks a required phenomenon.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        number = float(value)
        if math.isnan(number):
            return default
        return number
    except (TypeError, ValueError):
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def nonempty(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, (list, dict, tuple, set)):
        return bool(value)
    return True


def sanitize_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def digest_text(text: Any, n: int = 12) -> str:
    raw = "" if text is None else str(text)
    return hashlib.sha1(raw.encode("utf-8", errors="replace")).hexdigest()[:n]


def truncate(text: Any, limit: int = 700) -> str:
    raw = "" if text is None else str(text)
    raw = raw.strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(0, limit - 20)].rstrip() + " ... [truncated]"


def compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def parse_jsonish(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return value
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def canonical_option(value: Any) -> str:
    if value is None:
        return "not found in rows/replay"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"OPTION - {int(value)}"
    raw = str(value).strip()
    if not raw:
        return "not found in rows/replay"
    match = re.search(r"OPTION\s*-\s*(\d+)", raw, flags=re.I)
    if match:
        return f"OPTION - {match.group(1)}"
    if raw.isdigit():
        return f"OPTION - {raw}"
    return raw


def option_number(value: Any) -> str:
    text = canonical_option(value)
    match = re.search(r"OPTION\s*-\s*(\d+)", text, flags=re.I)
    return match.group(1) if match else text


def infer_row_root(row_path: Path) -> Path:
    parts = list(row_path.parts)
    if "workers" not in parts:
        return row_path.parents[2]
    idx = parts.index("workers")
    return Path(*parts[:idx])


def find_primary_run(rows: list[dict[str, Any]]) -> Path | None:
    stats: dict[Path, dict[str, int]] = {}
    for row in rows:
        run_root = row["_run_root"]
        ds = row.get("dataset")
        split = row.get("split")
        if ds not in DATASETS:
            continue
        stats.setdefault(run_root, {"mmlu_pro": 0, "mbpp": 0, "test": 0})
        stats[run_root][ds] += 1
        if split == "test":
            stats[run_root]["test"] += 1
    candidates = []
    for run_root, item in stats.items():
        if item["mmlu_pro"] and item["mbpp"]:
            name = str(run_root)
            score = item["mmlu_pro"] + item["mbpp"] + item["test"]
            if "stage2_final_design" in name:
                score += 100_000
            if "best_graph_operator_ablation_8ds_matched" in name:
                score += 10_000
            candidates.append((score, run_root))
    if not candidates:
        return None
    return sorted(candidates, reverse=True)[0][1]


def load_rows(outputs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in outputs_root.rglob("*_rows.jsonl"):
        if path.name not in ROW_FILES:
            continue
        if path.parent.name not in DATASETS:
            continue
        if "sample_shards" in path.parts or path.parent.parent.name == "data":
            continue
        if "workers" not in path.parts and path.parent.parent.name != "outputs":
            continue
        try:
            for row in iter_jsonl(path):
                if row.get("dataset") not in DATASETS:
                    continue
                row["_row_path"] = str(path)
                row["_run_root"] = infer_row_root(path)
                row["_row_split_file"] = path.name.replace("_rows", "")
                row["_dataset_output_dir"] = path.parent
                rows.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[warn] skip unreadable rows {path}: {exc}")
    return rows


def is_final_design(row: dict[str, Any]) -> bool:
    return "stage2_final_design" in str(row.get("_run_root", ""))


def find_data_record(row: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    row_id = row["id"]
    dataset = row["dataset"]
    run_root = Path(row["_run_root"])
    split_file = row.get("_row_split_file") or f"{row.get('split', 'test')}.jsonl"
    candidates = []
    candidates.extend(run_root.glob(f"data/{dataset}/{split_file}"))
    candidates.extend(run_root.glob(f"sample_shards/*/{dataset}/{split_file}"))
    # Recovery sub-runs sometimes reuse the parent data directory.
    if run_root.name.startswith("worker2_"):
        parent = run_root.parent
        candidates.extend(parent.glob(f"data/{dataset}/{split_file}"))
        candidates.extend(parent.glob(f"sample_shards/*/{dataset}/{split_file}"))
    for path in candidates:
        if not path.exists():
            continue
        try:
            for item in iter_jsonl(path):
                if item.get("id") == row_id:
                    return sanitize_data_record(item), path
        except (OSError, json.JSONDecodeError):
            continue
    return {}, None


def sanitize_data_record(item: dict[str, Any]) -> dict[str, Any]:
    safe = {
        "id": item.get("id"),
        "source_dataset": item.get("source_dataset"),
        "category": item.get("category"),
        "split": item.get("split"),
        "question": item.get("question"),
        "answer": item.get("answer"),
        "original_question": item.get("original_question"),
        "original_answer": item.get("original_answer"),
        "task_type": item.get("task_type"),
        "answer_format": item.get("answer_format"),
    }
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    safe_meta: dict[str, Any] = {}
    for key in ["uid", "category", "src", "entry_point", "answer_index", "options"]:
        if key in metadata:
            safe_meta[key] = metadata[key]
    visible_tests = extract_visible_tests(item.get("question", ""))
    if visible_tests:
        safe_meta["visible_tests_from_prompt"] = visible_tests
        safe_meta["visible_test_count_from_prompt"] = len(visible_tests)
    elif item.get("source_dataset") == "mbpp":
        safe_meta["visible_tests_from_prompt"] = []
        safe_meta["visible_test_count_from_prompt"] = 0
    safe["metadata"] = safe_meta
    return safe


def extract_visible_tests(question: str) -> list[str]:
    if not question:
        return []
    lines = question.splitlines()
    tests: list[str] = []
    in_visible = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower().startswith("visible test"):
            in_visible = True
            continue
        if in_visible and stripped.lower().startswith("return only"):
            break
        if in_visible and stripped.startswith("assert "):
            tests.append(stripped)
    return tests


def find_structure_artifact(row: dict[str, Any]) -> tuple[dict[str, Any], Path | None]:
    explicit = row.get("structure_artifact_path")
    if explicit:
        path = Path(str(explicit))
        if path.exists():
            return summarize_structure(path), path
    dataset = row["dataset"]
    split = row.get("split") or "test"
    safe = sanitize_id(row["id"])
    base = Path(row["_dataset_output_dir"])
    candidates = [
        base / "stage1_structures" / dataset / split / f"{safe}.json",
        base / "stage1_structures" / dataset / split / f"{row['id'].replace(':', '_')}.json",
    ]
    if dataset == "mmlu_pro":
        candidates.append(base / "stage1_structures" / "mmlu_pro" / split / f"{safe}.json")
    if dataset == "mbpp":
        candidates.append(base / "stage1_structures" / "mbpp" / split / f"{safe}.json")
    for path in candidates:
        if path.exists():
            return summarize_structure(path), path
    return {}, None


def summarize_structure(path: Path) -> dict[str, Any]:
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "read_error": str(exc)}
    graph = data.get("union_graph") if isinstance(data.get("union_graph"), dict) else {}
    nodes = graph.get("nodes") if isinstance(graph.get("nodes"), dict) else {}
    edges = graph.get("edges") if isinstance(graph.get("edges"), list) else []
    node_summaries = []
    for node_id, node in list(nodes.items())[:12]:
        if not isinstance(node, dict):
            node = {}
        node_summaries.append(
            {
                "node_id": node_id,
                "role": node.get("role"),
                "agent_id": node.get("agent_id"),
                "node_type": node.get("node_type"),
            }
        )
    edge_summaries = []
    for edge in edges[:20]:
        if isinstance(edge, dict):
            edge_summaries.append(
                {
                    "source": edge.get("source") or edge.get("src"),
                    "target": edge.get("target") or edge.get("dst"),
                    "weight": edge.get("weight"),
                    "support": edge.get("support"),
                }
            )
    structure_summary = data.get("structure_summary")
    if isinstance(structure_summary, dict):
        structure_summary = {
            "mode": structure_summary.get("mode"),
            "signature": structure_summary.get("signature"),
            "selected_topology_signatures": structure_summary.get("selected_topology_signatures"),
            "selected_topology_k": len(structure_summary.get("selected_topology_signatures") or []),
        }
    return {
        "path": str(path),
        "dataset_name": data.get("dataset_name"),
        "metadata": data.get("metadata"),
        "question_text": data.get("question_text"),
        "stage1_output": data.get("stage1_output"),
        "stage1_signature": data.get("stage1_signature"),
        "structure_summary": structure_summary,
        "union_graph_summary": {
            "node_count": len(nodes),
            "edge_count": len(edges),
            "nodes": node_summaries,
            "edges": edge_summaries,
        },
    }


def find_replay_summary(row: dict[str, Any]) -> dict[str, Any]:
    dataset_dir = Path(row["_dataset_output_dir"])
    split = row.get("split") or "test"
    safe = sanitize_id(row["id"])
    candidates = [
        dataset_dir / "replays" / split / safe,
        dataset_dir / "replays" / split / row["id"],
        Path(row["_run_root"]) / "replays" / dataset_dir.name / split / safe,
        Path(row["_run_root"]) / "replays" / split / safe,
    ]
    for path in candidates:
        if path.exists():
            return scan_replay(path)
    return {
        "replay_missing": True,
        "replay_path": "",
        "selected_memory_records": [],
        "memory_note": "not found in rows/replay",
        "raw_option_matrix_found": False,
        "raw_graph_trace_found": False,
    }


def scan_replay(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "replay_missing": False,
        "replay_path": str(path),
        "selected_memory_records": [],
        "raw_option_matrix_found": False,
        "raw_graph_trace_found": False,
    }
    keys = {
        "memory",
        "selected_records",
        "memory_record_counts",
        "memory_brief",
        "exported_memory",
        "local_latent_memory",
        "slot_mask",
        "record_scores",
        "support_set",
    }
    memory_records = []
    for item in path.rglob("*.json"):
        try:
            payload = load_json(item)
        except (OSError, json.JSONDecodeError):
            continue
        found = find_keys(payload, keys)
        if found:
            for key, value in found[:5]:
                memory_records.append(
                    {
                        "file": str(item),
                        "key": key,
                        "summary": truncate(value, 260),
                    }
                )
        if find_keys(payload, {"option_matrix", "evidence_atoms"}):
            summary["raw_option_matrix_found"] = True
        if find_keys(payload, {"graph_trace", "active_edges", "active_nodes"}):
            summary["raw_graph_trace_found"] = True
    summary["selected_memory_records"] = memory_records[:3]
    if not memory_records:
        summary["memory_note"] = "not found in rows/replay"
    return summary


def find_keys(payload: Any, keys: set[str]) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            if key in keys:
                found.append((key, value))
            if len(found) >= 20:
                break
            found.extend(find_keys(value, keys))
            if len(found) >= 20:
                break
    elif isinstance(payload, list):
        for value in payload[:20]:
            found.extend(find_keys(value, keys))
            if len(found) >= 20:
                break
    return found


def mmlu_wrong_correct_score(row: dict[str, Any]) -> float:
    score = 0.0
    if safe_float(row.get("stage1_success")) < 1 and safe_float(row.get("stage2_success")) >= 1:
        score += 100
    if not safe_bool(row.get("v4_4_stage1_anchor_used")):
        score += 30
    if nonempty(row.get("certificate_kind")):
        score += 20
    if safe_bool(row.get("discrete_update_accepted")):
        score += 20
    if safe_bool(row.get("discrete_probe_triggered")):
        score += 15
    if safe_bool(row.get("discrete_audit_triggered")):
        score += 15
    if safe_float(row.get("challenger_support_count")) > 0:
        score += 10
    if safe_float(row.get("anchor_conflict_count")) > 0:
        score += 10
    if safe_float(row.get("evidence_atom_count")) > 0:
        score += 10
    score += min(20, 5 * abs(safe_float(row.get("score_margin"))))
    score += min(20, 2 * abs(safe_float(row.get("vote_margin"))))
    if "fail" in str(row.get("matrix_parse_status", "")).lower():
        score -= 30
    if safe_float(row.get("candidate_bank_size")) == 0:
        score -= 30
    if row.get("split") == "test":
        score += 12
    return score


def mmlu_preserve_score(row: dict[str, Any]) -> float:
    score = 0.0
    if safe_float(row.get("stage1_success")) >= 1 and safe_float(row.get("stage2_success")) >= 1:
        score += 100
    if safe_bool(row.get("v4_4_stage1_anchor_used")):
        score += 30
    if nonempty(row.get("accept_blocker")):
        score += 20
    if nonempty(row.get("empty_cert_reason")):
        score += 20
    if safe_bool(row.get("discrete_probe_triggered")) or safe_bool(row.get("discrete_audit_triggered")):
        score += 15
    if safe_float(row.get("challenger_support_count")) > 0:
        score += 10
    if safe_float(row.get("anchor_conflict_count")) > 0:
        score += 10
    if safe_float(row.get("evidence_atom_count")) > 0:
        score += 10
    if row.get("audit_agree") is False:
        score += 10
    if not (
        safe_float(row.get("challenger_support_count")) > 0
        or safe_float(row.get("anchor_conflict_count")) > 0
        or safe_float(row.get("candidate_bank_size")) > 0
    ):
        score -= 30
    if row.get("split") == "test":
        score += 12
    if str(row.get("matrix_parse_status")) == "json_repaired":
        score += 5
    return score


def mbpp_repair_success_score(row: dict[str, Any]) -> float:
    score = 0.0
    if safe_float(row.get("stage1_success")) < 1 and safe_float(row.get("stage2_success")) >= 1:
        score += 100
    if safe_bool(row.get("v4_4_selected_is_repair_branch")):
        score += 40
    if safe_float(row.get("v4_4_repair_improvement_count")) > 0:
        score += 30
    if safe_float(row.get("v4_4_repair_branch_count")) > 0:
        score += 20
    if safe_float(row.get("v4_4_selected_visible_tests_passed")) > safe_float(
        row.get("v4_4_stage1_anchor_visible_tests_passed")
    ):
        score += 20
    failure = str(row.get("v4_4_selected_failure_kind") or "").lower()
    if failure in {"", "none"}:
        score += 15
    if safe_float(row.get("recovery_subgraph_size")) > 0:
        score += 10
    if safe_float(row.get("candidate_provenance_coverage")) > 0:
        score += 10
    if row.get("split") == "test":
        score += 12
    return score


def mbpp_anchor_guard_score(row: dict[str, Any]) -> float:
    score = 0.0
    if safe_float(row.get("stage1_success")) >= 1 and safe_float(row.get("stage2_success")) >= 1:
        score += 100
    if safe_bool(row.get("v4_4_stage1_anchor_used")):
        score += 30
    if safe_float(row.get("v4_4_candidate_count")) > 1:
        score += 20
    if safe_float(row.get("v4_4_repair_branch_count")) > 0:
        score += 20
    if safe_float(row.get("v4_4_selected_visible_tests_passed")) <= safe_float(
        row.get("v4_4_stage1_anchor_visible_tests_passed")
    ):
        score += 20
    reason = str(row.get("selection_reason", "")).lower()
    if any(token in reason for token in ["preserve", "anchor", "guard", "no_dominance", "bypass_stable"]):
        score += 10
    if row.get("split") == "test":
        score += 12
    if not strongest_non_anchor_candidate(row):
        score -= 20
    return score


def select_cases(rows: list[dict[str, Any]], primary_run: Path | None, allow_fallback: bool) -> list[dict[str, Any]]:
    primary_rows = [r for r in rows if primary_run and r["_run_root"] == primary_run]
    final_design_rows = [r for r in rows if is_final_design(r)]
    pools = final_design_rows if allow_fallback else primary_rows

    def pick(pool: list[dict[str, Any]], predicate: Any, score_fn: Any, used_ids: set[str]) -> dict[str, Any]:
        candidates = [r for r in pool if predicate(r) and r["id"] not in used_ids]
        if not candidates:
            raise RuntimeError("No candidate found for requested case.")
        return sorted(candidates, key=score_fn, reverse=True)[0]

    used: set[str] = set()
    cases: list[dict[str, Any]] = []

    primary_mmlu_wc = [
        r
        for r in primary_rows
        if r.get("dataset") == "mmlu_pro"
        and safe_float(r.get("stage1_success")) < 1
        and safe_float(r.get("stage2_success")) >= 1
        and (
            not safe_bool(r.get("v4_4_stage1_anchor_used"))
            or nonempty(r.get("certificate_kind"))
            or safe_bool(r.get("discrete_update_accepted"))
            or safe_bool(r.get("discrete_probe_triggered"))
            or safe_bool(r.get("discrete_audit_triggered"))
        )
    ]
    mmlu_wc_pool = primary_mmlu_wc or [
        r
        for r in pools
        if r.get("dataset") == "mmlu_pro"
        and safe_float(r.get("stage1_success")) < 1
        and safe_float(r.get("stage2_success")) >= 1
    ]
    mmlu_wc = sorted(mmlu_wc_pool, key=mmlu_wrong_correct_score, reverse=True)[0]
    used.add(mmlu_wc["id"])
    cases.append(
        {
            "case_name": "mmlu_pro_case_1",
            "phenomenon": "wrong_to_correct_override",
            "row": mmlu_wc,
            "selection_note": (
                "Primary combined run had no mechanistically clear test wrong-to-correct MMLU-Pro override; "
                "selected the strongest final-design fallback row."
                if mmlu_wc not in primary_mmlu_wc
                else "Selected from primary combined run."
            ),
        }
    )

    mmlu_preserve = pick(
        primary_rows or pools,
        lambda r: r.get("dataset") == "mmlu_pro"
        and r.get("split") == "test"
        and safe_float(r.get("stage1_success")) >= 1
        and safe_float(r.get("stage2_success")) >= 1
        and safe_bool(r.get("v4_4_stage1_anchor_used"))
        and (
            safe_bool(r.get("discrete_probe_triggered"))
            or safe_bool(r.get("discrete_audit_triggered"))
            or safe_float(r.get("anchor_conflict_count")) > 0
            or safe_float(r.get("challenger_support_count")) > 0
        ),
        mmlu_preserve_score,
        used,
    )
    used.add(mmlu_preserve["id"])
    cases.append(
        {
            "case_name": "mmlu_pro_case_2",
            "phenomenon": "safe_preserve_rejected_challenger",
            "row": mmlu_preserve,
            "selection_note": "Selected test preserve case with challenger/conflict evidence.",
        }
    )

    mbpp_repair = pick(
        primary_rows or pools,
        lambda r: r.get("dataset") == "mbpp"
        and r.get("split") == "test"
        and safe_float(r.get("stage1_success")) < 1
        and safe_float(r.get("stage2_success")) >= 1
        and (
            safe_bool(r.get("v4_4_selected_is_repair_branch"))
            or safe_float(r.get("v4_4_repair_improvement_count")) > 0
        ),
        mbpp_repair_success_score,
        used,
    )
    used.add(mbpp_repair["id"])
    cases.append(
        {
            "case_name": "mbpp_case_1",
            "phenomenon": "repair_wrong_to_correct",
            "row": mbpp_repair,
            "selection_note": "Selected test repair case with visible-test improvement.",
        }
    )

    mbpp_guard_candidates = [
        r
        for r in (primary_rows or pools)
        if r.get("dataset") == "mbpp"
        and r.get("split") == "test"
        and safe_float(r.get("stage1_success")) >= 1
        and safe_float(r.get("stage2_success")) >= 1
        and safe_bool(r.get("v4_4_stage1_anchor_used"))
        and (safe_float(r.get("v4_4_repair_branch_count")) > 0 or safe_float(r.get("v4_4_candidate_count")) > 1)
        and r["id"] not in used
    ]
    if mbpp_guard_candidates:
        mbpp_guard = sorted(mbpp_guard_candidates, key=mbpp_anchor_guard_score, reverse=True)[0]
        phenomenon = "anchor_guard_no_regression"
        note = "Selected test no-regression case; anchor passed executable verifier while challenger did not dominate."
    else:
        first_failure = str(mbpp_repair.get("v4_4_stage1_anchor_failure_kind") or "")
        fallback_pool = [
            r
            for r in (primary_rows or pools)
            if r.get("dataset") == "mbpp"
            and r.get("split") == "test"
            and safe_float(r.get("stage1_success")) < 1
            and safe_float(r.get("stage2_success")) >= 1
            and r["id"] not in used
            and str(r.get("v4_4_stage1_anchor_failure_kind") or "") != first_failure
        ]
        mbpp_guard = sorted(fallback_pool, key=mbpp_repair_success_score, reverse=True)[0]
        phenomenon = "repair_wrong_to_correct_fallback"
        note = "No high-quality anchor-guard case found; selected second repair with a different failure kind."
    used.add(mbpp_guard["id"])
    cases.append(
        {
            "case_name": "mbpp_case_2",
            "phenomenon": phenomenon,
            "row": mbpp_guard,
            "selection_note": note,
        }
    )
    return cases


def strongest_non_anchor_candidate(row: dict[str, Any]) -> dict[str, Any] | None:
    candidates = row.get("v4_4_top_candidates")
    if not isinstance(candidates, list):
        return None
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        if safe_bool(candidate.get("stage1_anchor")):
            continue
        preview = str(candidate.get("text_preview") or "").strip()
        if preview.startswith("def ") or "\ndef " in preview or preview.startswith("class "):
            return candidate
    for candidate in candidates:
        if isinstance(candidate, dict) and not safe_bool(candidate.get("stage1_anchor")):
            return candidate
    return None


def extract_case_payload(case: dict[str, Any]) -> dict[str, Any]:
    row = case["row"]
    data_record, data_path = find_data_record(row)
    structure_summary, structure_path = find_structure_artifact(row)
    replay_summary = find_replay_summary(row)
    question = data_record.get("question") or structure_summary.get("question_text") or "not found in rows/replay"
    gold = data_record.get("answer") or "not found in rows/replay"
    stage1_output = structure_summary.get("stage1_output") or stage1_candidate_text(row) or "not found in rows/replay"
    final_output = row.get("output", "not found in rows/replay")
    payload: dict[str, Any] = {
        "case_name": case["case_name"],
        "phenomenon": case["phenomenon"],
        "selection_note": case["selection_note"],
        "source_run": str(row["_run_root"]),
        "row_path": row["_row_path"],
        "data_path": str(data_path) if data_path else "",
        "structure_artifact_path": str(structure_path) if structure_path else "",
        "replay_missing": replay_summary.get("replay_missing", True),
        "replay_summary": replay_summary,
        "data_record": data_record,
        "structure_artifact_summary": structure_summary,
        "common_fields": {},
        "dataset_specific_fields": {},
        "stage1_output": stage1_output,
        "final_output": final_output,
        "question": question,
        "gold_answer": gold,
        "answer_options": extract_answer_options(data_record, question),
        "timeline": build_timeline(row, replay_summary),
        "original_row": row_to_json_safe(row),
    }
    for field in COMMON_FIELDS:
        if field == "question":
            payload["common_fields"][field] = question
        elif field in {"reference_answer", "gold_answer"}:
            payload["common_fields"][field] = gold
        elif field == "final_output":
            payload["common_fields"][field] = final_output
        elif field == "v4_4_top_candidates":
            payload["common_fields"][field] = sanitize_top_candidates(row.get(field))
        else:
            payload["common_fields"][field] = row.get(field, "not found in rows/replay")
    if row["dataset"] == "mmlu_pro":
        for field in MMLU_EXTRA_FIELDS:
            payload["dataset_specific_fields"][field] = row.get(field, "not found in rows/replay")
        payload["stage1_anchor_option"] = canonical_option(stage1_output)
        payload["stage2_final_option"] = canonical_option(final_output)
        payload["gold_option"] = canonical_option(gold)
        payload["option_matrix_summary"] = build_option_matrix_summary(row)
    else:
        visible_tests = extract_visible_tests(question)
        function_signature = infer_function_signature(data_record, stage1_output, final_output)
        challenger = strongest_non_anchor_candidate(row)
        challenger_code = str((challenger or {}).get("text_preview") or "")
        stage1_code = str(stage1_output)
        final_code = str(final_output)
        if case["phenomenon"] == "anchor_guard_no_regression" and challenger_code:
            diff_text = make_diff(stage1_code, challenger_code, "stage1_anchor.py", "strongest_challenger.py")
        else:
            diff_text = make_diff(stage1_code, final_code, "stage1_anchor.py", "stage2_selected.py")
        payload["dataset_specific_fields"] = {
            "function_signature": function_signature,
            "visible_tests": visible_tests,
            "stage1_code": stage1_code,
            "final_selected_code": final_code,
            "challenger_or_repair_code": challenger_code or "not found in rows/replay",
            "code_diff": diff_text,
            "failing_examples": summarize_failing_examples(row, visible_tests),
            "stdout": row.get("stdout", "not found in rows/replay"),
            "stderr": row.get("stderr", "not found in rows/replay"),
            "exec_error": row.get("exec_error", "not found in rows/replay"),
        }
        for field in MBPP_EXTRA_FIELDS:
            payload["dataset_specific_fields"].setdefault(field, row.get(field, "not found in rows/replay"))
    return payload


def row_to_json_safe(row: dict[str, Any]) -> dict[str, Any]:
    safe = {}
    for key, value in row.items():
        if key.startswith("_"):
            safe[key] = str(value)
        elif key == "v4_4_top_candidates":
            safe[key] = sanitize_top_candidates(value)
        elif key in {"discrete_challenger_proposal_raw_preview"}:
            safe[key] = sanitize_reasoning_preview(value)
        else:
            safe[key] = value
    return safe


def sanitize_top_candidates(value: Any) -> Any:
    if not isinstance(value, list):
        return value
    cleaned = []
    for item in value:
        if not isinstance(item, dict):
            cleaned.append(item)
            continue
        candidate = dict(item)
        if "text_preview" in candidate:
            candidate["text_preview"] = sanitize_reasoning_preview(candidate.get("text_preview"))
        cleaned.append(candidate)
    return cleaned


def sanitize_reasoning_preview(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    code_or_answer = (
        text.startswith("def ")
        or text.startswith("class ")
        or text.startswith("import ")
        or text.upper().startswith("OPTION -")
        or text.startswith("FINAL:")
    )
    reasoning_markers = ["<think>", "好的", "我现在", "我需要", "首先", "Let's", "I need to"]
    looks_like_reasoning = any(marker in text for marker in reasoning_markers)
    if code_or_answer and len(text) <= 3000:
        return value
    if looks_like_reasoning or len(text) > 900:
        return f"[redacted model reasoning preview; sha1={digest_text(text, 16)}; chars={len(text)}]"
    return value


def stage1_candidate_text(row: dict[str, Any]) -> str:
    candidates = row.get("v4_4_top_candidates")
    if isinstance(candidates, list):
        for candidate in candidates:
            if isinstance(candidate, dict) and safe_bool(candidate.get("stage1_anchor")):
                return str(candidate.get("text_preview") or "")
    return ""


def extract_answer_options(data_record: dict[str, Any], question: str) -> list[str]:
    metadata = data_record.get("metadata") if isinstance(data_record.get("metadata"), dict) else {}
    options = metadata.get("options")
    if isinstance(options, list) and options:
        return [f"{idx}) {option}" for idx, option in enumerate(options, start=1)]
    parsed = []
    for line in question.splitlines():
        if re.match(r"^\s*\d+\)", line):
            parsed.append(line.strip())
    return parsed


def infer_function_signature(data_record: dict[str, Any], stage1_output: Any, final_output: Any) -> str:
    metadata = data_record.get("metadata") if isinstance(data_record.get("metadata"), dict) else {}
    entry = metadata.get("entry_point")
    code = f"{stage1_output}\n{final_output}"
    match = re.search(r"^def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(([^)]*)\)", code, flags=re.M)
    if match:
        return f"def {match.group(1)}({match.group(2)})"
    if entry:
        return f"{entry}(...)"
    return "not found in rows/replay"


def normalize_code(code: str) -> list[str]:
    stripped = code.strip("\n")
    if not stripped:
        return []
    return [line.rstrip() for line in stripped.splitlines()]


def make_diff(before: str, after: str, before_name: str, after_name: str) -> str:
    diff = difflib.unified_diff(
        normalize_code(before),
        normalize_code(after),
        fromfile=before_name,
        tofile=after_name,
        lineterm="",
    )
    return "\n".join(diff) + "\n"


def summarize_failing_examples(row: dict[str, Any], visible_tests: list[str]) -> list[dict[str, Any]]:
    examples = []
    visible_joined = "\n".join(visible_tests)
    classes = row.get("v4_4_top_classes")
    if not isinstance(classes, list):
        return examples
    for item in classes:
        if not isinstance(item, dict):
            continue
        class_key = item.get("class_key")
        if not isinstance(class_key, list) or len(class_key) < 6:
            continue
        failure = str(class_key[4] or "")
        locus = str(class_key[5] or "")
        if not failure:
            continue
        visible = any(test in locus for test in visible_tests)
        examples.append(
            {
                "failure_kind": failure,
                "passed": item.get("passed"),
                "total": item.get("total"),
                "contains_anchor": item.get("contains_anchor"),
                "representative_digest": item.get("representative_digest"),
                "locus": locus if visible and locus in visible_joined or visible else summarize_locus(locus),
                "visibility": "visible_prompt_test" if visible else "non-visible evaluator detail summarized",
            }
        )
    return examples[:5]


def summarize_locus(locus: str) -> str:
    if not locus:
        return ""
    if "NameError" in locus:
        return "execution error: NameError during verifier run"
    if "AssertionError" in locus:
        return "assertion failure during verifier run"
    if "syntax" in locus.lower() or "invalid character" in locus.lower():
        return "syntax error during verifier run"
    return truncate(locus, 120)


def build_option_matrix_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": "row-level counts only; raw matrix not found",
        "candidate_bank_size": row.get("candidate_bank_size"),
        "cert_bank_size": row.get("cert_bank_size"),
        "option_matrix_yes_votes": row.get("option_matrix_yes_votes"),
        "option_matrix_no_votes": row.get("option_matrix_no_votes"),
        "evidence_atom_count": row.get("evidence_atom_count"),
        "anchor_conflict_count": row.get("anchor_conflict_count"),
        "challenger_support_count": row.get("challenger_support_count"),
        "score_margin": row.get("score_margin"),
        "vote_margin": row.get("vote_margin"),
        "audit_agree": row.get("audit_agree"),
        "calibrator_p_accept": row.get("calibrator_p_accept"),
        "accept_blocker": row.get("accept_blocker"),
        "certificate_kind": row.get("certificate_kind"),
        "matrix_parse_status": row.get("matrix_parse_status"),
        "matrix_option_covered_count": row.get("matrix_option_covered_count"),
        "top_candidate_value": row.get("top_candidate_value"),
    }


def build_timeline(row: dict[str, Any], replay_summary: dict[str, Any]) -> list[dict[str, Any]]:
    edge = row.get("active_edge_ratio_by_turn")
    node = row.get("active_node_ratio_by_turn")
    costs = row.get("stage2_turn_token_costs")
    estimates = row.get("stage2_turn_token_estimates")
    edge = edge if isinstance(edge, list) else []
    node = node if isinstance(node, list) else []
    costs = costs if isinstance(costs, list) else []
    estimates = estimates if isinstance(estimates, list) else []
    turns = int(max(safe_float(row.get("stage2_turns")), len(edge), len(node), len(costs), len(estimates)))
    memory_records = replay_summary.get("selected_memory_records") or []
    timeline = []
    for idx in range(turns):
        timeline.append(
            {
                "turn_index": idx + 1,
                "active_edge_ratio": edge[idx] if idx < len(edge) else "not found in rows/replay",
                "active_node_ratio": node[idx] if idx < len(node) else "not found in rows/replay",
                "stage2_turn_token_cost": costs[idx] if idx < len(costs) else "not found in rows/replay",
                "stage2_turn_token_estimate": estimates[idx] if idx < len(estimates) else "not found in rows/replay",
                "active_nodes": "full graph replay not found; using row-level graph-faithfulness metrics",
                "active_edges": "full graph replay not found; using row-level graph-faithfulness metrics",
                "skipped_nodes": "not found in rows/replay",
                "selected_memory_records": memory_records if idx == 0 else [],
                "exported_memory_message": replay_summary.get("memory_note", "not found in rows/replay"),
                "candidate_count_after_turn": row.get("v4_4_candidate_count") if idx == turns - 1 else "not found in rows/replay",
                "key_feedback_or_residual": key_verifier_signal(row),
            }
        )
    return timeline


def key_verifier_signal(row: dict[str, Any]) -> str:
    if row.get("dataset") == "mmlu_pro":
        parts = [
            f"certificate={row.get('certificate_kind') or 'none'}",
            f"accept_blocker={row.get('accept_blocker') or row.get('empty_cert_reason') or 'none'}",
            f"support/conflict={row.get('challenger_support_count')}/{row.get('anchor_conflict_count')}",
            f"score_margin={row.get('score_margin')}",
            f"vote_margin={row.get('vote_margin')}",
        ]
        return "; ".join(parts)
    return (
        f"visible_tests {row.get('v4_4_stage1_anchor_visible_tests_passed')}/"
        f"{row.get('v4_4_stage1_anchor_visible_tests_total')} -> "
        f"{row.get('v4_4_selected_visible_tests_passed')}/"
        f"{row.get('v4_4_selected_visible_tests_total')}; "
        f"failure {row.get('v4_4_stage1_anchor_failure_kind') or 'none'} -> "
        f"{row.get('v4_4_selected_failure_kind') or 'none'}"
    )


def render_timeline_md(payload: dict[str, Any]) -> str:
    lines = [
        f"# {payload['case_name']} graph/memory timeline",
        "",
        "Full graph replay not found; using row-level graph-faithfulness metrics.",
        "",
        "| turn | active_edge_ratio | active_node_ratio | token_cost | token_estimate | candidate_count_after_turn | feedback/residual |",
        "|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in payload["timeline"]:
        lines.append(
            "| {turn_index} | {active_edge_ratio} | {active_node_ratio} | {stage2_turn_token_cost} | "
            "{stage2_turn_token_estimate} | {candidate_count_after_turn} | {key_feedback_or_residual} |".format(
                **item
            )
        )
    lines.extend(["", "## Selected Memory Records", ""])
    records = payload["replay_summary"].get("selected_memory_records") or []
    if records:
        for record in records[:3]:
            lines.append(f"- `{record.get('key')}` from `{record.get('file')}`: {record.get('summary')}")
    else:
        lines.append("- not found in rows/replay")
    lines.extend(["", "## Graph Details", ""])
    graph = payload["structure_artifact_summary"].get("union_graph_summary") or {}
    lines.append(f"- Stage1 union graph nodes: {graph.get('node_count', 'not found in rows/replay')}")
    lines.append(f"- Stage1 union graph edges: {graph.get('edge_count', 'not found in rows/replay')}")
    for node in (graph.get("nodes") or [])[:6]:
        lines.append(f"- node `{node.get('node_id')}`: role={node.get('role')}, agent={node.get('agent_id')}")
    return "\n".join(lines) + "\n"


def render_option_matrix_md(payload: dict[str, Any]) -> str:
    row = payload["original_row"]
    summary = payload["option_matrix_summary"]
    lines = [
        f"# {payload['case_name']} option/evidence matrix",
        "",
        "row-level counts only; raw matrix not found",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in summary.items():
        lines.append(f"| {key} | {compact_json(value) if isinstance(value, (dict, list)) else value} |")
    lines.extend(["", "## Option Status Summary", ""])
    lines.extend(render_option_status_table(payload))
    return "\n".join(lines) + "\n"


def render_option_status_table(payload: dict[str, Any]) -> list[str]:
    row = payload["original_row"]
    lines = [
        "| option | support_count | conflict_count | yes_votes | no_votes | evidence_atoms | final_status |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    final_option = payload["stage2_final_option"]
    anchor_option = payload["stage1_anchor_option"]
    gold_option = payload["gold_option"]
    options = payload["answer_options"]
    if options:
        for option in options:
            status = []
            number = option.split(")", 1)[0]
            if number == option_number(final_option):
                status.append("final")
            if number == option_number(anchor_option):
                status.append("stage1_anchor")
            if number == option_number(gold_option):
                status.append("gold")
            lines.append(
                f"| {option} | row-level | row-level | {row.get('option_matrix_yes_votes')} | "
                f"{row.get('option_matrix_no_votes')} | {row.get('evidence_atom_count')} | "
                f"{', '.join(status) if status else 'not selected'} |"
            )
    else:
        lines.append(
            "| not found in rows/replay | "
            f"{row.get('challenger_support_count')} | {row.get('anchor_conflict_count')} | "
            f"{row.get('option_matrix_yes_votes')} | {row.get('option_matrix_no_votes')} | "
            f"{row.get('evidence_atom_count')} | row-level only |"
        )
    return lines


def render_case_md(payload: dict[str, Any]) -> str:
    if payload["common_fields"]["dataset"] == "mmlu_pro":
        return render_mmlu_case_md(payload)
    return render_mbpp_case_md(payload)


def render_mmlu_case_md(payload: dict[str, Any]) -> str:
    row = payload["original_row"]
    options = "\n".join(f"- {item}" for item in payload["answer_options"]) or "- not found in rows/replay"
    matrix = payload["option_matrix_summary"]
    memory_records = payload["replay_summary"].get("selected_memory_records") or []
    memory_text = "\n".join(
        f"- {item.get('key')}: {item.get('summary')}" for item in memory_records[:3]
    ) or "- not found in rows/replay"
    if payload["phenomenon"] == "wrong_to_correct_override":
        takeaway = (
            "Stage2 did not simply majority-vote options; it accepted a challenger only after the "
            "discrete slot update, probe/audit signals, and contrastive certificate supported replacing "
            "the incorrect Stage1 anchor."
        )
    else:
        takeaway = (
            "Stage2 explored an alternative answer but preserved the stage1 anchor because the "
            "option-level certificate was incomplete, the audit/probe did not confidently support the "
            "challenger, or the conflict/support margin was insufficient."
        )
    lines = [
        f"# mmlu_pro {row['id']}: {payload['phenomenon']}",
        "",
        "## 1. Problem",
        f"- Dataset / split / id / category: `mmlu_pro` / `{row.get('split')}` / `{row['id']}` / `{row.get('category')}`",
        f"- Question: {payload['question']}",
        f"- Gold answer: {payload['gold_option']}",
        "- Answer options:",
        options,
        "",
        "## 2. Stage1 Anchor / Before / After",
        f"- Stage1 output: `{payload['stage1_anchor_option']}`",
        f"- Stage2 final output: `{payload['stage2_final_option']}`",
        f"- Gold option: `{payload['gold_option']}`",
        f"- stage1_success -> stage2_success: `{row.get('stage1_success')}` -> `{row.get('stage2_success')}`",
        f"- Stage1 success: `{row.get('stage1_success')}`",
        f"- Stage1 signature: `{truncate(row.get('stage1_signature'), 420)}`",
        f"- Anchor verifier status: anchor_conflict_count={row.get('anchor_conflict_count')}; "
        f"anchor_eval_status_hist={compact_json(row.get('anchor_eval_status_hist'))}",
        "",
        "## 3. Stage2 Candidate / Repair / Probe / Why Stage2 Changed Or Preserved",
        f"- Stage2 route family: `{row.get('v4_4_route_family')}`",
        f"- Candidate count / collapsed class count: `{row.get('v4_4_candidate_count')}` / `{row.get('v4_4_collapsed_class_count')}`",
        f"- Selected candidate source: `{row.get('v4_4_selected_candidate_source')}`",
        f"- Certificate kind: `{row.get('certificate_kind') or 'not found in rows/replay'}`",
        f"- Top candidate value: `{row.get('top_candidate_value') or row.get('discrete_assignment') or 'not found in rows/replay'}`",
        f"- score_margin / vote_margin: `{row.get('score_margin')}` / `{row.get('vote_margin')}`",
        f"- challenger_support_count vs anchor_conflict_count: `{row.get('challenger_support_count')}` vs `{row.get('anchor_conflict_count')}`",
        f"- Probe: triggered={row.get('discrete_probe_triggered')}, winner={row.get('discrete_probe_winner') or 'not found in rows/replay'}, confidence={row.get('discrete_probe_confidence') or 'not found in rows/replay'}",
        f"- Audit: triggered={row.get('discrete_audit_triggered')}, winner={row.get('discrete_audit_winner') or 'not found in rows/replay'}, confidence={row.get('discrete_audit_confidence') or 'not found in rows/replay'}, audit_agree={row.get('audit_agree')}",
        "",
        "## 4. Final Selection",
        f"- Final output: `{payload['stage2_final_option']}`",
        f"- Final success: `{row.get('success')}`",
        f"- Selection decision: `{row.get('selection_decision')}`",
        f"- Selection reason: `{row.get('selection_reason')}`",
        f"- Why override or preserve: {mmlu_why(payload)}",
        "",
        "## 5. Graph and Memory Behavior",
        f"- active_edge_ratio_by_turn: `{row.get('active_edge_ratio_by_turn')}`",
        f"- active_node_ratio_by_turn: `{row.get('active_node_ratio_by_turn')}`",
        f"- stage2_turns: `{row.get('stage2_turns')}`",
        f"- stage2_memory_records: `{row.get('stage2_memory_records')}`",
        f"- recovery_subgraph_size: `{row.get('recovery_subgraph_size')}`",
        f"- candidate_provenance_coverage: `{row.get('candidate_provenance_coverage')}`",
        "- Selected memory records:",
        memory_text,
        "",
        "## 6. Cost",
        f"- stage1 token cost: `{row.get('stage1_token_cost')}`",
        f"- stage2 token cost: `{row.get('stage2_token_cost')}`",
        f"- total token cost: `{row.get('token_cost')}`",
        f"- latency: `{row.get('latency')}`",
        "",
        "## 7. Paper-ready takeaway",
        takeaway,
        "",
        "## Option Matrix",
        "row-level counts only; raw matrix not found",
        "",
        "| metric | value |",
        "|---|---:|",
    ]
    for key, value in matrix.items():
        lines.append(f"| {key} | {compact_json(value) if isinstance(value, (dict, list)) else value} |")
    lines.extend(["", "### Option Status Rows"])
    lines.extend(render_option_status_table(payload))
    return "\n".join(lines) + "\n"


def mmlu_why(payload: dict[str, Any]) -> str:
    row = payload["original_row"]
    if payload["phenomenon"] == "wrong_to_correct_override":
        return (
            f"Stage2 replaced the Stage1 anchor `{payload['stage1_anchor_option']}` with "
            f"`{payload['stage2_final_option']}` via `{row.get('selection_reason')}`; "
            f"certificate={row.get('certificate_kind')}, update_accepted={row.get('discrete_update_accepted')}, "
            f"probe={row.get('discrete_probe_triggered')}, audit={row.get('discrete_audit_triggered')}."
        )
    return (
        f"Stage2 kept `{payload['stage1_anchor_option']}` despite challenger `{row.get('top_candidate_value')}` "
        f"because accept_blocker=`{row.get('accept_blocker') or row.get('empty_cert_reason') or 'not found in rows/replay'}` "
        f"and selection_reason=`{row.get('selection_reason')}`."
    )


def render_mbpp_case_md(payload: dict[str, Any]) -> str:
    row = payload["original_row"]
    fields = payload["dataset_specific_fields"]
    visible = "\n".join(f"- `{test}`" for test in fields.get("visible_tests") or []) or "- not found in rows/replay"
    memory_records = payload["replay_summary"].get("selected_memory_records") or []
    memory_text = "\n".join(
        f"- {item.get('key')}: {item.get('summary')}" for item in memory_records[:3]
    ) or "- not found in rows/replay"
    diff_short = truncate(fields.get("code_diff"), 1600)
    if payload["phenomenon"] == "anchor_guard_no_regression":
        takeaway = (
            "Stage2 considered challenger branches but the final anchor guard preserved the original "
            "stage1 answer because no candidate strictly dominated the anchor under executable verifier evidence."
        )
    else:
        takeaway = (
            "The code route turned a failed solution into an executable residual, generated a local repair, "
            "and accepted it only after the repaired branch dominated the anchor on visible tests."
        )
    lines = [
        f"# mbpp {row['id']}: {payload['phenomenon']}",
        "",
        "## 1. Problem",
        f"- Dataset / split / id / category: `mbpp` / `{row.get('split')}` / `{row['id']}` / `{row.get('category')}`",
        f"- Function signature: `{fields.get('function_signature')}`",
        f"- Short task description: {first_line(payload['question'])}",
        "- Visible tests:",
        visible,
        "- Gold answer: reference code is stored in the sanitized raw JSON; hidden tests are not shown.",
        "",
        "## 2. Stage1 Anchor / Stage1 Failure",
        f"- Stage1 success: `{row.get('stage1_success')}`",
        f"- Stage1 signature: `{truncate(row.get('stage1_signature'), 420)}`",
        f"- Anchor visible tests passed / total: `{row.get('v4_4_stage1_anchor_visible_tests_passed')}` / `{row.get('v4_4_stage1_anchor_visible_tests_total')}`",
        f"- Anchor failure_kind: `{row.get('v4_4_stage1_anchor_failure_kind') or 'none'}`",
        "- Stage1 code:",
        "```python",
        truncate(fields.get("stage1_code"), 1800),
        "```",
        "",
        "## 3. Stage2 Candidate / Repair / Probe / Stage2 Repair",
        f"- Stage2 route family: `{row.get('v4_4_route_family')}`",
        f"- Candidate count / collapsed class count: `{row.get('v4_4_candidate_count')}` / `{row.get('v4_4_collapsed_class_count')}`",
        f"- Selected candidate source: `{row.get('v4_4_selected_candidate_source')}`",
        f"- repair branch count: `{row.get('v4_4_repair_branch_count')}`",
        f"- selected_is_repair_branch: `{row.get('v4_4_selected_is_repair_branch')}`",
        f"- selected visible tests passed / total: `{row.get('v4_4_selected_visible_tests_passed')}` / `{row.get('v4_4_selected_visible_tests_total')}`",
        f"- selected failure_kind: `{row.get('v4_4_selected_failure_kind') or 'none'}`",
        "- Code diff:",
        "```diff",
        diff_short,
        "```",
        "",
        "## 4. Final Selection / Why Accepted",
        f"- Final success: `{row.get('success')}`",
        f"- Selection decision: `{row.get('selection_decision')}`",
        f"- Selection reason: `{row.get('selection_reason')}`",
        f"- Why accepted or preserved: {mbpp_why(payload)}",
        "",
        "## 5. Graph and Memory Behavior",
        f"- active_edge_ratio_by_turn: `{row.get('active_edge_ratio_by_turn')}`",
        f"- active_node_ratio_by_turn: `{row.get('active_node_ratio_by_turn')}`",
        f"- stage2_turns: `{row.get('stage2_turns')}`",
        f"- stage2_memory_records: `{row.get('stage2_memory_records')}`",
        f"- recovery_subgraph_size: `{row.get('recovery_subgraph_size')}`",
        f"- candidate_provenance_coverage: `{row.get('candidate_provenance_coverage')}`",
        "- Selected memory records:",
        memory_text,
        "",
        "## 6. Cost",
        f"- stage1 token cost: `{row.get('stage1_token_cost')}`",
        f"- stage2 token cost: `{row.get('stage2_token_cost')}`",
        f"- total token cost: `{row.get('token_cost')}`",
        f"- latency: `{row.get('latency')}`",
        "",
        "## 7. Paper-ready takeaway",
        takeaway,
    ]
    return "\n".join(lines) + "\n"


def first_line(text: str) -> str:
    return next((line.strip() for line in str(text).splitlines() if line.strip()), "not found in rows/replay")


def mbpp_why(payload: dict[str, Any]) -> str:
    row = payload["original_row"]
    if payload["phenomenon"] == "anchor_guard_no_regression":
        return (
            f"Anchor passed `{row.get('v4_4_stage1_anchor_visible_tests_passed')}/"
            f"{row.get('v4_4_stage1_anchor_visible_tests_total')}` visible verifier tests; "
            f"the strongest challenger did not dominate, so `{row.get('selection_reason')}` kept the anchor."
        )
    return (
        f"Repair improved visible tests from `{row.get('v4_4_stage1_anchor_visible_tests_passed')}/"
        f"{row.get('v4_4_stage1_anchor_visible_tests_total')}` to "
        f"`{row.get('v4_4_selected_visible_tests_passed')}/{row.get('v4_4_selected_visible_tests_total')}` "
        f"and failure kind from `{row.get('v4_4_stage1_anchor_failure_kind') or 'none'}` to "
        f"`{row.get('v4_4_selected_failure_kind') or 'none'}`."
    )


def render_index_row(payload: dict[str, Any], raw_path: Path, md_path: Path) -> dict[str, Any]:
    row = payload["original_row"]
    if row.get("dataset") == "mbpp":
        gold = f"reference_code_sha1={digest_text(payload.get('gold_answer'), 16)}; see raw JSON"
    else:
        gold = payload.get("gold_answer")
    return {
        "case_name": payload["case_name"],
        "dataset": row.get("dataset"),
        "split": row.get("split"),
        "id": row.get("id"),
        "category": row.get("category"),
        "route_family": row.get("v4_4_route_family"),
        "phenomenon": payload["phenomenon"],
        "stage1_success": row.get("stage1_success"),
        "stage2_success": row.get("stage2_success"),
        "final_success": row.get("success"),
        "stage2_vs_stage1_outcome": row.get("stage2_vs_stage1_outcome"),
        "stage1_answer_or_code_digest": digest_text(payload.get("stage1_output")),
        "stage2_answer_or_code_digest": digest_text(payload.get("final_output")),
        "gold": gold,
        "selection_decision": row.get("selection_decision"),
        "selection_reason": row.get("selection_reason"),
        "stage1_token_cost": row.get("stage1_token_cost"),
        "stage2_token_cost": row.get("stage2_token_cost"),
        "total_token_cost": row.get("token_cost"),
        "latency": row.get("latency"),
        "active_edge_ratio_by_turn": compact_json(row.get("active_edge_ratio_by_turn")),
        "active_node_ratio_by_turn": compact_json(row.get("active_node_ratio_by_turn")),
        "stage2_memory_records": row.get("stage2_memory_records"),
        "candidate_count": row.get("v4_4_candidate_count"),
        "collapsed_class_count": row.get("v4_4_collapsed_class_count"),
        "key_verifier_signal": key_verifier_signal(row),
        "replay_path": payload["replay_summary"].get("replay_path", ""),
        "raw_json_path": str(raw_path),
        "markdown_path": str(md_path),
    }


def validate_outputs(out_dir: Path, index_rows: list[dict[str, Any]], payloads: list[dict[str, Any]]) -> None:
    if len(index_rows) != 4:
        raise AssertionError(f"case_study_index.csv must have exactly 4 rows, got {len(index_rows)}")
    dataset_counts = {}
    for row in index_rows:
        dataset_counts[row["dataset"]] = dataset_counts.get(row["dataset"], 0) + 1
    if dataset_counts.get("mmlu_pro") != 2 or dataset_counts.get("mbpp") != 2:
        raise AssertionError(f"Expected 2 MMLU-Pro and 2 MBPP rows, got {dataset_counts}")
    for required in ["case_study_index.csv", "case_study_index.json"]:
        if not (out_dir / required).exists():
            raise AssertionError(f"Missing {required}")
    for name in ["mmlu_pro_case_1.md", "mmlu_pro_case_2.md", "mbpp_case_1.md", "mbpp_case_2.md"]:
        text = (out_dir / name).read_text(encoding="utf-8")
        for marker in ["## 1. Problem", "## 2. Stage1 Anchor", "## 4. Final Selection", "## 6. Cost"]:
            if marker not in text:
                raise AssertionError(f"{name} missing marker {marker}")
        if "not found in rows/replay" in text:
            print(f"[warn] {name} contains some explicitly missing row/replay details")
    for payload in payloads:
        row = payload["original_row"]
        safe = sanitize_id(row["id"])
        base = f"case_{row['dataset']}_{safe}"
        for suffix in ["_raw.json", "_timeline.md"]:
            if not (out_dir / f"{base}{suffix}").exists():
                raise AssertionError(f"Missing {base}{suffix}")
        if row["dataset"] == "mmlu_pro" and not (out_dir / f"{base}_option_matrix.md").exists():
            raise AssertionError(f"Missing {base}_option_matrix.md")
        if row["dataset"] == "mbpp" and not (out_dir / f"{base}_patch.diff").exists():
            raise AssertionError(f"Missing {base}_patch.diff")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.outputs_root)
    if not rows:
        raise RuntimeError(f"No rows found under {args.outputs_root}")
    primary_run = find_primary_run(rows)
    if primary_run is None:
        raise RuntimeError("No run directory containing both mmlu_pro and mbpp rows was found.")
    print(f"[primary-run] {primary_run}")
    cases = select_cases(rows, primary_run, args.allow_fallback_final_design_runs)
    payloads = [extract_case_payload(case) for case in cases]

    index_rows = []
    for payload in payloads:
        row = payload["original_row"]
        safe = sanitize_id(row["id"])
        base = f"case_{row['dataset']}_{safe}"
        raw_path = args.out_dir / f"{base}_raw.json"
        timeline_path = args.out_dir / f"{base}_timeline.md"
        write_json(raw_path, payload)
        write_text(timeline_path, render_timeline_md(payload))
        if row["dataset"] == "mmlu_pro":
            write_text(args.out_dir / f"{base}_option_matrix.md", render_option_matrix_md(payload))
        else:
            write_text(args.out_dir / f"{base}_patch.diff", payload["dataset_specific_fields"]["code_diff"])
        md_path = args.out_dir / f"{payload['case_name']}.md"
        write_text(md_path, render_case_md(payload))
        index_rows.append(render_index_row(payload, raw_path, md_path))

    csv_path = args.out_dir / "case_study_index.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(index_rows)
    write_json(args.out_dir / "case_study_index.json", index_rows)
    validate_outputs(args.out_dir, index_rows, payloads)
    print(f"[done] wrote {len(index_rows)} cases to {args.out_dir}")
    for row in index_rows:
        print(
            f"[case] {row['case_name']} {row['dataset']} {row['split']} {row['id']} "
            f"{row['phenomenon']} s1={row['stage1_success']} s2={row['stage2_success']} "
            f"success={row['final_success']}"
        )


if __name__ == "__main__":
    main()
