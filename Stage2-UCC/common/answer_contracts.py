from __future__ import annotations

import ast
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from common.parsing_utils import (
    extract_last_number,
    extract_python_code,
    extract_sequence_numbers,
    normalize_yes_no_output,
    safe_json,
)


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


def _normalize_option_token(token: str, metadata: Optional[Dict[str, Any]]) -> Optional[int]:
    max_option = 0
    options = metadata.get("options") if isinstance(metadata, dict) else None
    if isinstance(options, list):
        max_option = len(options)
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


def _extract_option_surfaces(text: str, answer_surfaces: Optional[Sequence[str]]) -> List[str]:
    surfaces = [str(surface).strip() for surface in (answer_surfaces or ()) if str(surface).strip()]
    if surfaces:
        return surfaces
    cleaned = str(text or "").strip()
    if not cleaned:
        return []
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        return lines[-2:]
    return [cleaned]


def _extract_option_value(
    text: str,
    metadata: Optional[Dict[str, Any]],
    answer_surfaces: Optional[Sequence[str]] = None,
) -> Optional[int]:
    surfaces = _extract_option_surfaces(text, answer_surfaces)
    patterns = (
        r"(?i)(?:final answer|answer|option)\s*[:：-]?\s*([A-Z]|\d{1,2})\b",
        r"^\s*([A-Z]|\d{1,2})\s*$",
    )
    for surface in surfaces:
        for pattern in patterns:
            match = re.search(pattern, surface)
            if not match:
                continue
            value = _normalize_option_token(match.group(1), metadata)
            if value is not None:
                return value
    return None


def _extract_graph_json(text: str) -> Optional[dict]:
    return safe_json(text) if isinstance(text, str) else None


def _sequence_from_text(text: str) -> List[int]:
    return extract_sequence_numbers(str(text or ""))


def _graph_bool_value(text: str, parsed: Optional[dict]) -> str:
    if isinstance(parsed, dict) and "answer" in parsed:
        return normalize_yes_no_output(str(parsed.get("answer", "")))
    return normalize_yes_no_output(str(text or ""))


def _graph_scalar_value(text: str, parsed: Optional[dict], field: str) -> Optional[float]:
    if isinstance(parsed, dict) and isinstance(parsed.get(field), (int, float)):
        return float(parsed[field])
    return extract_last_number(str(text or ""))


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
    code = extract_python_code(str(text or ""))
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
            "contract_valid": True,
            "recoverable_valid": bool(code.strip()),
            "recoverable_value": code,
            "recoverable_object": code,
            "task_subtype": str((metadata or {}).get("task", "")),
        },
        signature=signature,
    )


def _graph_contract_fields(
    *,
    field_name: str,
    task_subtype: str,
    contract_valid: bool,
    recoverable_valid: bool,
    recoverable_value: Any,
) -> Dict[str, Any]:
    return {
        "field": field_name,
        "task_subtype": task_subtype,
        "schema_valid": contract_valid,
        "contract_valid": contract_valid,
        "recoverable_valid": recoverable_valid,
        "recoverable_value": recoverable_value,
        "recoverable_object": recoverable_value,
    }


def _resolved_answer_format(dataset_name: str, answer_format: str) -> str:
    fmt = str(answer_format or "").strip()
    if fmt:
        return fmt
    dataset = str(dataset_name or "").strip().lower()
    if dataset in {"mmlu", "mmlu_pro", "cqa"}:
        return "option"
    if dataset == "nlgraph":
        return "graph_json"
    if dataset in {"mbpp", "humaneval"}:
        return "python_code"
    return ""


def _parse_graph_answer(text: str, task_subtype: str) -> AnswerObject:
    cleaned = str(text or "").strip()
    parsed = _extract_graph_json(cleaned)
    schema_valid = isinstance(parsed, dict)
    subtype = str(task_subtype or "").strip()

    if subtype in {"connectivity", "cycle"}:
        value = _graph_bool_value(cleaned, parsed)
        recoverable_valid = value in {"yes", "no"}
        field_name = "answer"
        contract_valid = schema_valid and recoverable_valid and "answer" in parsed
        signature = f"graph_bool::{field_name}::{value}" if recoverable_valid else "graph_bool::invalid"
        return AnswerObject(
            kind="graph_bool",
            value=value,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name=field_name,
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=value,
            ),
            signature=signature,
        )

    if subtype == "flow":
        value = _graph_scalar_value(cleaned, parsed, "max_flow")
        recoverable_valid = value is not None
        contract_valid = schema_valid and isinstance(parsed, dict) and isinstance(parsed.get("max_flow"), (int, float))
        signature = (
            f"graph_scalar::max_flow::{int(value) if value is not None and float(value).is_integer() else value}"
            if recoverable_valid
            else "graph_scalar::invalid"
        )
        return AnswerObject(
            kind="graph_scalar",
            value=value,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="max_flow",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=value,
            ),
            signature=signature,
        )

    if subtype == "topology":
        order = _graph_sequence_value(cleaned, parsed, "order")
        recoverable_valid = bool(order)
        contract_valid = schema_valid and isinstance(parsed, dict) and isinstance(parsed.get("order"), list) and bool(order)
        return AnswerObject(
            kind="graph_sequence",
            value=order,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="order",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=order,
            ),
            signature=f"graph_sequence::order::{_stable_hash(order)}" if recoverable_valid else "graph_sequence::invalid",
        )

    if subtype == "hamilton":
        path = _graph_sequence_value(cleaned, parsed, "path")
        recoverable_valid = bool(path)
        contract_valid = schema_valid and isinstance(parsed, dict) and isinstance(parsed.get("path"), list) and bool(path)
        return AnswerObject(
            kind="graph_sequence",
            value=path,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="path",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=path,
            ),
            signature=f"graph_sequence::path::{_stable_hash(path)}" if recoverable_valid else "graph_sequence::invalid",
        )

    if subtype == "shortest_path":
        path = _graph_sequence_value(cleaned, parsed, "path")
        total_weight = _graph_scalar_value(cleaned, parsed, "total_weight")
        recoverable_valid = bool(path) or total_weight is not None
        contract_valid = (
            schema_valid
            and isinstance(parsed, dict)
            and isinstance(parsed.get("path"), list)
            and isinstance(parsed.get("total_weight"), (int, float))
        )
        value = {"path": path, "total_weight": total_weight}
        return AnswerObject(
            kind="graph_path",
            value=value,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="path",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=value,
            ),
            signature=f"graph_path::{_stable_hash(value)}" if recoverable_valid else "graph_path::invalid",
        )

    if subtype == "matching":
        matches, count = _graph_matching_value(cleaned, parsed)
        recoverable_valid = bool(matches) or count is not None
        contract_valid = schema_valid and isinstance(parsed, dict) and ("matches" in parsed or "count" in parsed)
        value = {"matches": matches, "count": count}
        return AnswerObject(
            kind="graph_matching",
            value=value,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="matches",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=value,
            ),
            signature=f"graph_matching::{_stable_hash(value)}" if recoverable_valid else "graph_matching::invalid",
        )

    if subtype == "GNN":
        embeddings = _graph_embeddings(cleaned, parsed)
        recoverable_valid = bool(embeddings)
        contract_valid = schema_valid and isinstance(parsed, dict) and isinstance(parsed.get("node_embeddings"), dict) and bool(embeddings)
        return AnswerObject(
            kind="node_embeddings",
            value=embeddings,
            valid=contract_valid,
            fields=_graph_contract_fields(
                field_name="node_embeddings",
                task_subtype=subtype,
                contract_valid=contract_valid,
                recoverable_valid=recoverable_valid,
                recoverable_value=embeddings,
            ),
            signature=f"node_embeddings::{_stable_hash(embeddings)}" if recoverable_valid else "node_embeddings::invalid",
        )

    return AnswerObject(
        kind="graph_text",
        value=cleaned,
        valid=schema_valid,
        fields={
            "schema_valid": schema_valid,
            "contract_valid": schema_valid,
            "recoverable_valid": bool(cleaned),
            "recoverable_value": cleaned,
            "recoverable_object": cleaned,
            "task_subtype": subtype,
        },
        signature=f"graph_text::{_stable_hash(cleaned)}" if cleaned else "graph_text::invalid",
    )


def parse_answer_object(
    text: str,
    *,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[dict],
    answer_surfaces: Optional[Sequence[str]] = None,
) -> AnswerObject:
    cleaned = str(text or "").strip()
    fmt = _resolved_answer_format(dataset_name, answer_format)

    if fmt == "option":
        option = _extract_option_value(cleaned, metadata, answer_surfaces=answer_surfaces)
        return AnswerObject(
            kind="option",
            value=option,
            valid=option is not None,
            fields={
                "schema_valid": True,
                "contract_valid": option is not None,
                "recoverable_valid": option is not None,
                "recoverable_value": option,
                "recoverable_object": option,
                "task_subtype": str(task_subtype or ""),
            },
            signature=f"option::{option}" if option is not None else "option::invalid",
        )

    if fmt == "graph_json":
        return _parse_graph_answer(cleaned, task_subtype)

    if fmt == "python_code":
        return _parse_code_answer(cleaned, metadata)

    if fmt == "yes_no":
        value = normalize_yes_no_output(cleaned)
        valid = value in {"yes", "no"}
        return AnswerObject(
            kind="bool",
            value=value,
            valid=valid,
            fields={
                "schema_valid": True,
                "contract_valid": valid,
                "recoverable_valid": valid,
                "recoverable_value": value,
                "recoverable_object": value,
                "task_subtype": str(task_subtype or ""),
            },
            signature=f"bool::{value}" if valid else "bool::invalid",
        )

    numeric = extract_last_number(cleaned)
    if numeric is not None:
        return AnswerObject(
            kind="numeric",
            value=numeric,
            valid=True,
            fields={
                "schema_valid": True,
                "contract_valid": True,
                "recoverable_valid": True,
                "recoverable_value": numeric,
                "recoverable_object": numeric,
                "task_subtype": str(task_subtype or ""),
            },
            signature=f"numeric::{numeric}",
        )

    return AnswerObject(
        kind="text",
        value=cleaned,
        valid=bool(cleaned),
        fields={
            "schema_valid": True,
            "contract_valid": bool(cleaned),
            "recoverable_valid": bool(cleaned),
            "recoverable_value": cleaned,
            "recoverable_object": cleaned,
            "task_subtype": str(task_subtype or ""),
        },
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


def _recoverable_valid(answer_object: AnswerObject) -> bool:
    return bool(answer_object.valid or answer_object.fields.get("recoverable_valid", False))


def _semantic_value(answer_object: AnswerObject) -> Any:
    if "recoverable_value" in answer_object.fields:
        return answer_object.fields.get("recoverable_value")
    if "recoverable_object" in answer_object.fields:
        return answer_object.fields.get("recoverable_object")
    return answer_object.value


def typed_answer_distance(left: AnswerObject, right: AnswerObject) -> float:
    left_valid = _recoverable_valid(left)
    right_valid = _recoverable_valid(right)
    if not left_valid and not right_valid:
        return 0.0
    if not left_valid or not right_valid:
        return 1.0
    if left.kind != right.kind:
        return 1.0

    left_value = _semantic_value(left)
    right_value = _semantic_value(right)

    if left.kind in {"option", "bool"}:
        return 0.0 if left_value == right_value else 1.0

    if left.kind in {"numeric", "graph_scalar"}:
        try:
            left_scalar = float(left_value)
            right_scalar = float(right_value)
        except Exception:
            return 1.0
        scale = max(1.0, abs(left_scalar), abs(right_scalar))
        return min(1.0, abs(left_scalar - right_scalar) / scale)

    if left.kind == "graph_sequence":
        field = str(left.fields.get("field") or right.fields.get("field") or "path")
        left_set = _sequence_constraint_set(left_value or [], field)
        right_set = _sequence_constraint_set(right_value or [], field)
        if not left_set and not right_set:
            return 0.0 if list(left_value or []) == list(right_value or []) else 1.0
        overlap = len(left_set & right_set)
        union = len(left_set | right_set)
        return 1.0 - (float(overlap) / float(max(1, union)))

    if left.kind == "graph_path":
        left_path = list((left_value or {}).get("path") or [])
        right_path = list((right_value or {}).get("path") or [])
        left_weight = (left_value or {}).get("total_weight")
        right_weight = (right_value or {}).get("total_weight")
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
        left_matches = {tuple(item) for item in ((left_value or {}).get("matches") or [])}
        right_matches = {tuple(item) for item in ((right_value or {}).get("matches") or [])}
        left_count = (left_value or {}).get("count")
        right_count = (right_value or {}).get("count")
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
        left_emb = dict(left_value or {})
        right_emb = dict(right_value or {})
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
