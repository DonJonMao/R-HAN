from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mas_treesearch.evaluator import MultiFidelityEvaluator


@dataclass(frozen=True)
class AnswerObject:
    kind: str
    value: Any
    valid: bool
    fields: Dict[str, Any] = field(default_factory=dict)
    signature: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stable_hash(value: Any) -> str:
    if isinstance(value, str):
        payload = value
    else:
        payload = _compact_json(value)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def _extract_option_value(text: str, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
    cleaned = str(text or "").strip()
    max_option = 0
    options = metadata.get("options") if isinstance(metadata, dict) else None
    if isinstance(options, list):
        max_option = len(options)

    def _normalize_token(token: str) -> Optional[int]:
        raw = token.strip().upper().rstrip(".")
        if not raw:
            return None
        if raw.isdigit():
            value = int(raw)
        elif len(raw) == 1 and raw.isalpha():
            value = ord(raw) - ord("A") + 1
        else:
            return None
        if max_option > 0 and not (1 <= value <= max_option):
            return None
        return value

    patterns = (
        r"(?:^|\b)(?:OPTION|ANSWER)\s*[:：-]?\s*([A-D]|\d{1,2})(?:\b|\)|\.)",
        r"(?:^|\b)([A-D]|\d{1,2})(?:\b|\)|\.)",
    )
    for pattern in patterns:
        match = re.search(pattern, cleaned, flags=re.IGNORECASE)
        if match:
            value = _normalize_token(match.group(1))
            if value is not None:
                return value
    return None


def _extract_graph_json(text: str) -> Optional[dict]:
    if not isinstance(text, str):
        return None
    return MultiFidelityEvaluator._safe_json(text)


def _sequence_from_text(text: str) -> List[int]:
    return MultiFidelityEvaluator._extract_sequence_numbers(str(text or ""))


def _graph_bool_value(text: str, parsed: Optional[dict]) -> str:
    if isinstance(parsed, dict) and "answer" in parsed:
        return MultiFidelityEvaluator._normalize_yes_no_output(str(parsed.get("answer", "")))
    return MultiFidelityEvaluator._normalize_yes_no_output(str(text or ""))


def _graph_scalar_value(text: str, parsed: Optional[dict], field: str) -> Optional[float]:
    if isinstance(parsed, dict) and isinstance(parsed.get(field), (int, float)):
        return float(parsed[field])
    return MultiFidelityEvaluator._extract_last_number(str(text or ""))


def _graph_sequence_value(text: str, parsed: Optional[dict], field: str) -> List[int]:
    if isinstance(parsed, dict) and isinstance(parsed.get(field), list):
        return _sequence_from_text(_compact_json(parsed[field]))
    return _sequence_from_text(str(text or ""))


def _graph_matching_value(text: str, parsed: Optional[dict]) -> Tuple[List[Tuple[int, int]], Optional[int]]:
    matches: List[Tuple[int, int]] = []
    count: Optional[int] = None
    if isinstance(parsed, dict):
        raw_matches = parsed.get("matches")
        if isinstance(raw_matches, list):
            for item in raw_matches:
                if isinstance(item, list) and len(item) >= 2:
                    try:
                        matches.append((int(item[0]), int(item[1])))
                    except Exception:
                        continue
        raw_count = parsed.get("count")
        if isinstance(raw_count, (int, float)):
            count = int(raw_count)
    if not matches:
        for left, right in re.findall(r"applicant\s+(\d+)\s*:\s*job\s+(\d+)", str(text or ""), flags=re.IGNORECASE):
            matches.append((int(left), int(right)))
    if count is None:
        match = re.search(r"(\d+)\s+applicants can find", str(text or ""), flags=re.IGNORECASE)
        if match:
            count = int(match.group(1))
    return matches, count


def _graph_embeddings(text: str, parsed: Optional[dict]) -> Dict[str, List[int]]:
    embeddings: Dict[str, List[int]] = {}
    if isinstance(parsed, dict) and isinstance(parsed.get("node_embeddings"), dict):
        for key, value in parsed["node_embeddings"].items():
            if isinstance(value, list):
                try:
                    embeddings[str(key)] = [int(item) for item in value]
                except Exception:
                    continue
        return embeddings
    for node, vec in re.findall(r"node\s+(\d+)\s*:\s*\[([^\]]+)\]", str(text or ""), flags=re.IGNORECASE):
        try:
            embeddings[str(node)] = [int(token) for token in re.findall(r"-?\d+", vec)]
        except Exception:
            continue
    return embeddings


def _parse_code_answer(text: str, metadata: Optional[Dict[str, Any]]) -> AnswerObject:
    code = MultiFidelityEvaluator._extract_python_code(str(text or ""))
    entry_point = str((metadata or {}).get("entry_point") or "").strip()
    syntax_ok = False
    tree_dump = ""
    try:
        tree = ast.parse(code) if code.strip() else None
        if tree is not None:
            syntax_ok = True
            tree_dump = ast.dump(tree, annotate_fields=False, include_attributes=False)
    except Exception:
        syntax_ok = False
    entry_point_present = bool(entry_point and re.search(rf"\bdef\s+{re.escape(entry_point)}\s*\(", code))
    signature_key = tree_dump if tree_dump else code.strip()
    if entry_point:
        signature = f"code::{entry_point}::{_stable_hash(signature_key)}"
    else:
        signature = f"code::<anon>::{_stable_hash(signature_key)}"
    return AnswerObject(
        kind="code",
        value=code,
        valid=bool(code.strip()),
        fields={
            "entry_point": entry_point,
            "syntax_ok": syntax_ok,
            "entry_point_present": entry_point_present or not entry_point,
            "ast_hash": _stable_hash(tree_dump) if tree_dump else "",
            "schema_valid": True,
        },
        signature=signature,
    )


def _parse_graph_answer(text: str, task_subtype: str) -> AnswerObject:
    cleaned = str(text or "").strip()
    parsed = _extract_graph_json(cleaned)
    schema_valid = isinstance(parsed, dict)
    subtype = str(task_subtype or "").strip()

    if subtype in {"connectivity", "cycle"}:
        value = _graph_bool_value(cleaned, parsed)
        valid = value in {"yes", "no"}
        field_name = "answer"
        signature = f"graph_bool::{field_name}::{value}" if valid else "graph_bool::invalid"
        return AnswerObject(
            kind="graph_bool",
            value=value,
            valid=valid,
            fields={"field": field_name, "schema_valid": schema_valid},
            signature=signature,
        )

    if subtype == "flow":
        value = _graph_scalar_value(cleaned, parsed, "max_flow")
        valid = value is not None
        signature = f"graph_scalar::max_flow::{int(value) if value is not None and float(value).is_integer() else value}" if valid else "graph_scalar::invalid"
        return AnswerObject(
            kind="graph_scalar",
            value=value,
            valid=valid,
            fields={"field": "max_flow", "schema_valid": schema_valid},
            signature=signature,
        )

    if subtype == "topology":
        order = _graph_sequence_value(cleaned, parsed, "order")
        return AnswerObject(
            kind="graph_sequence",
            value=order,
            valid=bool(order),
            fields={"field": "order", "schema_valid": schema_valid},
            signature=f"graph_sequence::order::{_stable_hash(order)}" if order else "graph_sequence::invalid",
        )

    if subtype == "hamilton":
        path = _graph_sequence_value(cleaned, parsed, "path")
        return AnswerObject(
            kind="graph_sequence",
            value=path,
            valid=bool(path),
            fields={"field": "path", "schema_valid": schema_valid},
            signature=f"graph_sequence::path::{_stable_hash(path)}" if path else "graph_sequence::invalid",
        )

    if subtype == "shortest_path":
        path = _graph_sequence_value(cleaned, parsed, "path")
        total_weight = _graph_scalar_value(cleaned, parsed, "total_weight")
        valid = bool(path) or total_weight is not None
        value = {"path": path, "total_weight": total_weight}
        return AnswerObject(
            kind="graph_path",
            value=value,
            valid=valid,
            fields={"field": "path", "schema_valid": schema_valid},
            signature=f"graph_path::{_stable_hash(value)}" if valid else "graph_path::invalid",
        )

    if subtype == "matching":
        matches, count = _graph_matching_value(cleaned, parsed)
        valid = bool(matches) or count is not None
        value = {"matches": matches, "count": count}
        return AnswerObject(
            kind="graph_matching",
            value=value,
            valid=valid,
            fields={"field": "matches", "schema_valid": schema_valid},
            signature=f"graph_matching::{_stable_hash(value)}" if valid else "graph_matching::invalid",
        )

    if subtype == "GNN":
        embeddings = _graph_embeddings(cleaned, parsed)
        return AnswerObject(
            kind="node_embeddings",
            value=embeddings,
            valid=bool(embeddings),
            fields={"field": "node_embeddings", "schema_valid": schema_valid},
            signature=f"node_embeddings::{_stable_hash(embeddings)}" if embeddings else "node_embeddings::invalid",
        )

    return AnswerObject(
        kind="graph_text",
        value=cleaned,
        valid=bool(cleaned),
        fields={"schema_valid": schema_valid, "task_subtype": subtype},
        signature=f"graph_text::{_stable_hash(cleaned)}" if cleaned else "graph_text::invalid",
    )


def parse_answer_object(
    text: str,
    *,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[dict],
) -> AnswerObject:
    cleaned = str(text or "").strip()
    dataset = str(dataset_name or "").strip()
    fmt = str(answer_format or "").strip()

    if fmt == "option" or dataset in {"mmlu", "mmlu_pro", "popqa", "cqa"}:
        option = _extract_option_value(cleaned, metadata)
        return AnswerObject(
            kind="option",
            value=option,
            valid=option is not None,
            fields={"schema_valid": True},
            signature=f"option::{option}" if option is not None else "option::invalid",
        )

    if fmt == "graph_json" or dataset == "nlgraph":
        return _parse_graph_answer(cleaned, task_subtype)

    if fmt == "python_code" or dataset in {"mbpp", "humaneval"}:
        return _parse_code_answer(cleaned, metadata)

    if fmt == "yes_no":
        value = MultiFidelityEvaluator._normalize_yes_no_output(cleaned)
        valid = value in {"yes", "no"}
        return AnswerObject(
            kind="bool",
            value=value,
            valid=valid,
            fields={"schema_valid": True},
            signature=f"bool::{value}" if valid else "bool::invalid",
        )

    numeric = MultiFidelityEvaluator._extract_last_number(cleaned)
    if numeric is not None:
        return AnswerObject(
            kind="numeric",
            value=numeric,
            valid=True,
            fields={"schema_valid": True},
            signature=f"numeric::{numeric}",
        )

    return AnswerObject(
        kind="text",
        value=cleaned,
        valid=bool(cleaned),
        fields={"schema_valid": True},
        signature=f"text::{_stable_hash(cleaned)}" if cleaned else "text::invalid",
    )


def _sequence_constraint_set(value: Sequence[int], field: str) -> set[Tuple[int, int]]:
    items = [int(item) for item in value]
    if not items:
        return set()
    if field == "order":
        pairs: set[Tuple[int, int]] = set()
        for idx, left in enumerate(items):
            for right in items[idx + 1 :]:
                pairs.add((left, right))
        return pairs
    return {(left, right) for left, right in zip(items, items[1:])}


def typed_answer_distance(left: AnswerObject, right: AnswerObject) -> float:
    if not left.valid and not right.valid:
        return 0.0
    if not left.valid or not right.valid:
        return 1.0
    if left.kind != right.kind:
        return 1.0

    if left.kind in {"option", "bool"}:
        return 0.0 if left.value == right.value else 1.0

    if left.kind in {"numeric", "graph_scalar"}:
        try:
            left_value = float(left.value)
            right_value = float(right.value)
        except Exception:
            return 1.0
        scale = max(1.0, abs(left_value), abs(right_value))
        return min(1.0, abs(left_value - right_value) / scale)

    if left.kind == "graph_sequence":
        field = str(left.fields.get("field") or right.fields.get("field") or "path")
        left_set = _sequence_constraint_set(left.value or [], field)
        right_set = _sequence_constraint_set(right.value or [], field)
        if not left_set and not right_set:
            return 0.0 if list(left.value or []) == list(right.value or []) else 1.0
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        return 1.0 - (float(overlap) / float(max(1, union)))

    if left.kind == "graph_path":
        left_path = list((left.value or {}).get("path") or [])
        right_path = list((right.value or {}).get("path") or [])
        left_weight = (left.value or {}).get("total_weight")
        right_weight = (right.value or {}).get("total_weight")
        path_distance = typed_answer_distance(
            AnswerObject(kind="graph_sequence", value=left_path, valid=True, fields={"field": "path"}, signature=""),
            AnswerObject(kind="graph_sequence", value=right_path, valid=True, fields={"field": "path"}, signature=""),
        )
        if left_weight is None or right_weight is None:
            return path_distance
        return min(1.0, 0.6 * path_distance + 0.4 * typed_answer_distance(
            AnswerObject(kind="graph_scalar", value=left_weight, valid=True, fields={}, signature=""),
            AnswerObject(kind="graph_scalar", value=right_weight, valid=True, fields={}, signature=""),
        ))

    if left.kind == "graph_matching":
        left_matches = {tuple(item) for item in ((left.value or {}).get("matches") or [])}
        right_matches = {tuple(item) for item in ((right.value or {}).get("matches") or [])}
        left_count = (left.value or {}).get("count")
        right_count = (right.value or {}).get("count")
        if not left_matches and not right_matches and left_count is not None and right_count is not None:
            return 0.0 if int(left_count) == int(right_count) else 1.0
        overlap = len(left_matches & right_matches)
        union = len(left_matches | right_matches)
        count_penalty = 0.0
        if left_count is not None and right_count is not None:
            count_penalty = typed_answer_distance(
                AnswerObject(kind="graph_scalar", value=left_count, valid=True, fields={}, signature=""),
                AnswerObject(kind="graph_scalar", value=right_count, valid=True, fields={}, signature=""),
            )
        return min(1.0, 0.75 * (1.0 - float(overlap) / float(max(1, union))) + 0.25 * count_penalty)

    if left.kind == "node_embeddings":
        left_emb = dict(left.value or {})
        right_emb = dict(right.value or {})
        all_nodes = set(left_emb) | set(right_emb)
        if not all_nodes:
            return 0.0
        matches = 0
        for node in all_nodes:
            if list(left_emb.get(node, [])) == list(right_emb.get(node, [])):
                matches += 1
        return 1.0 - (float(matches) / float(len(all_nodes)))

    if left.kind == "code":
        if left.signature == right.signature:
            return 0.0
        same_entry = str(left.fields.get("entry_point", "")) == str(right.fields.get("entry_point", ""))
        syntax_ok = bool(left.fields.get("syntax_ok")) and bool(right.fields.get("syntax_ok"))
        if same_entry and syntax_ok:
            return 0.4
        if same_entry:
            return 0.7
        return 1.0

    return 0.0 if left.signature == right.signature else 1.0


def typed_answer_similarity(left: AnswerObject, right: AnswerObject) -> float:
    return max(0.0, 1.0 - typed_answer_distance(left, right))
