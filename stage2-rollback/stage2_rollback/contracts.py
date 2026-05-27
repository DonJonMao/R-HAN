from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Dict, Optional, Sequence

from .parsing_utils import (
    extract_boxed,
    extract_last_number,
    extract_python_code,
    extract_sequence_numbers,
    normalize_yes_no,
    safe_float,
    safe_json_snippet,
    strip_hidden_reasoning,
)


@dataclass
class AnswerObject:
    kind: str
    value: Any
    valid: bool
    signature: str
    fields: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContractResidual:
    parse: float
    completeness: float
    constraint: float
    execution: float

    @property
    def pooled(self) -> float:
        return float(self.parse + self.completeness + self.constraint + self.execution) / 4.0


def _normalize_math_text(text: str) -> str:
    cleaned = strip_hidden_reasoning(text)
    boxed = extract_boxed(cleaned)
    target = boxed if boxed is not None else cleaned
    target = target.strip()
    target = re.sub(r"\s+", "", target)
    return target


def _safe_sha(text: str) -> str:
    return sha1(text.encode("utf-8")).hexdigest()[:16]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _parse_option(text: str, metadata: Optional[dict]) -> AnswerObject:
    cleaned = strip_hidden_reasoning(text)
    patterns = [
        r"(?i)(?:final answer|answer|option)\s*[:：-]?\s*([A-Z]|\d{1,2})\b",
        r"^\s*([A-Z]|\d{1,2})\s*$",
    ]
    options = []
    if isinstance(metadata, dict):
        raw_options = metadata.get("options")
        if isinstance(raw_options, list):
            options = raw_options
    option_count = len(options)
    for pat in patterns:
        match = re.search(pat, cleaned)
        if not match:
            continue
        token = match.group(1).strip().upper()
        value: Optional[int]
        if token.isdigit():
            value = int(token)
        elif "A" <= token <= "Z":
            value = ord(token) - ord("A") + 1
        else:
            value = None
        if value is None:
            continue
        if option_count > 0 and not (1 <= value <= option_count):
            continue
        return AnswerObject(
            kind="option",
            value=value,
            valid=True,
            signature=f"option::{value}",
            fields={
                "contract_valid": True,
                "recoverable_valid": True,
                "recoverable_value": value,
                "task_subtype": str((metadata or {}).get("task", "")),
            },
        )
    return AnswerObject(
        kind="option",
        value=None,
        valid=False,
        signature="option::invalid",
        fields={
            "contract_valid": False,
            "recoverable_valid": False,
            "recoverable_value": None,
            "task_subtype": str((metadata or {}).get("task", "")),
        },
    )


def _parse_graph(text: str, task_subtype: str) -> AnswerObject:
    cleaned = strip_hidden_reasoning(text)
    snippet = safe_json_snippet(cleaned)
    parsed = None
    if snippet:
        try:
            parsed = json.loads(snippet)
        except json.JSONDecodeError:
            parsed = None
    subtype = str(task_subtype or "")
    contract_valid = isinstance(parsed, dict)
    recoverable_valid = False
    value: Any = None
    signature = f"graph_json::{subtype or 'unknown'}::invalid"
    if subtype in {"connectivity", "cycle"}:
        if isinstance(parsed, dict) and normalize_yes_no(str(parsed.get("answer", ""))) is not None:
            value = normalize_yes_no(str(parsed.get("answer", "")))
            contract_valid = True
            recoverable_valid = True
        else:
            recovered = normalize_yes_no(cleaned)
            if recovered is not None:
                value = recovered
                contract_valid = False
                recoverable_valid = True
        if value is not None:
            signature = f"graph_bool::answer::{value}"
    elif subtype == "flow":
        if isinstance(parsed, dict):
            recovered = parsed.get("max_flow", parsed.get("answer"))
        else:
            recovered = extract_last_number(cleaned)
        if recovered is not None:
            value = str(recovered)
            recoverable_valid = True
            contract_valid = isinstance(parsed, dict) and "max_flow" in parsed
            signature = f"graph_scalar::max_flow::{value}"
    elif subtype in {"topology", "hamilton", "shortest_path"}:
        if isinstance(parsed, dict):
            recovered = parsed.get("order", parsed.get("path", parsed.get("answer")))
            if isinstance(recovered, list):
                seq = [int(x) for x in recovered]
            else:
                seq = extract_sequence_numbers(str(recovered))
        else:
            seq = extract_sequence_numbers(cleaned)
        if seq:
            value = seq
            recoverable_valid = True
            contract_valid = isinstance(parsed, dict)
            signature = f"graph_sequence::{subtype}::{','.join(map(str, seq))}"
    elif subtype.lower() == "gnn":
        recovered = parsed.get("node_embeddings") if isinstance(parsed, dict) else None
        if isinstance(recovered, dict) and recovered:
            value = recovered
            contract_valid = True
            recoverable_valid = True
            signature = f"node_embeddings::{_safe_sha(json.dumps(recovered, sort_keys=True, ensure_ascii=False))}"
    else:
        if isinstance(parsed, dict):
            value = parsed
            contract_valid = True
            recoverable_valid = True
            signature = f"graph_json::{subtype or 'unknown'}::{_safe_sha(json.dumps(parsed, sort_keys=True, ensure_ascii=False))}"
    return AnswerObject(
        kind="graph_json",
        value=value,
        valid=bool(contract_valid),
        signature=signature,
        fields={
            "contract_valid": bool(contract_valid),
            "recoverable_valid": bool(recoverable_valid),
            "recoverable_value": value,
            "task_subtype": subtype,
            "raw_json": parsed if isinstance(parsed, dict) else None,
        },
    )


def _parse_python_code(text: str, metadata: Optional[dict]) -> AnswerObject:
    code = extract_python_code(text)
    syntax_ok = False
    try:
        module = ast.parse(code or "")
        syntax_ok = True
        ast_hash = _safe_sha(ast.dump(module, annotate_fields=False, include_attributes=False))
    except SyntaxError:
        ast_hash = _safe_sha(code)
    entry_point = ""
    if isinstance(metadata, dict):
        entry_point = str(metadata.get("entry_point", "") or "")
    signature = f"code::{entry_point or 'unknown'}::{ast_hash}"
    return AnswerObject(
        kind="code",
        value=code,
        valid=syntax_ok,
        signature=signature,
        fields={
            "contract_valid": syntax_ok,
            "recoverable_valid": bool(code.strip()),
            "recoverable_value": code,
            "task_subtype": str((metadata or {}).get("task", "")),
            "entry_point": entry_point,
            "syntax_ok": syntax_ok,
        },
    )


def _parse_math_expression(text: str, metadata: Optional[dict]) -> AnswerObject:
    expr = _normalize_math_text(text)
    recovered = expr or extract_last_number(text) or ""
    contract_valid = bool(expr)
    recoverable_valid = bool(recovered)
    signature = f"math_expression::{recovered or 'invalid'}"
    return AnswerObject(
        kind="math_expression",
        value=recovered or None,
        valid=contract_valid,
        signature=signature,
        fields={
            "contract_valid": contract_valid,
            "recoverable_valid": recoverable_valid,
            "recoverable_value": recovered or None,
            "task_subtype": str((metadata or {}).get("task", "")),
        },
    )


def parse_answer_object(
    text: str,
    *,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[dict],
) -> AnswerObject:
    del dataset_name
    fmt = str(answer_format or "").strip()
    if fmt == "option":
        return _parse_option(text, metadata)
    if fmt == "graph_json":
        return _parse_graph(text, task_subtype)
    if fmt == "python_code":
        return _parse_python_code(text, metadata)
    if fmt in {"math_expression", "numeric"}:
        return _parse_math_expression(text, metadata)
    cleaned = strip_hidden_reasoning(text)
    recovered = extract_last_number(cleaned)
    if recovered is None:
        recovered = cleaned
    numeric = safe_float(recovered)
    kind = "numeric" if numeric is not None else "text"
    value = numeric if numeric is not None else recovered
    signature = f"{kind}::{value}"
    return AnswerObject(
        kind=kind,
        value=value,
        valid=bool(recovered),
        signature=signature,
        fields={
            "contract_valid": bool(recovered),
            "recoverable_valid": bool(recovered),
            "recoverable_value": value,
            "task_subtype": str(task_subtype or ""),
        },
    )


def materialize_answer_object(
    answer_object: AnswerObject,
    *,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[dict],
) -> str:
    del metadata
    fmt = str(answer_format or "").strip()
    subtype = str(task_subtype or "")
    if fmt == "option":
        value = answer_object.fields.get("recoverable_value", answer_object.value)
        return f"OPTION - {value}" if value is not None else "OPTION - 1"
    if fmt == "graph_json":
        recovered = answer_object.fields.get("recoverable_value", answer_object.value)
        if subtype in {"connectivity", "cycle"}:
            value = str(recovered or "yes").lower()
            return json.dumps({"answer": value}, ensure_ascii=False)
        if subtype == "flow":
            value = recovered if recovered is not None else 0
            return json.dumps({"max_flow": value}, ensure_ascii=False)
        if subtype in {"topology", "hamilton", "shortest_path"}:
            key = "order" if subtype == "topology" else "path"
            seq = recovered if isinstance(recovered, list) else []
            return json.dumps({key: seq}, ensure_ascii=False)
        if subtype.lower() == "gnn":
            payload = recovered if isinstance(recovered, dict) else {}
            return json.dumps({"node_embeddings": payload}, ensure_ascii=False)
        payload = recovered if isinstance(recovered, dict) else {}
        return json.dumps(payload, ensure_ascii=False)
    if fmt == "python_code":
        code = str(answer_object.fields.get("recoverable_value", answer_object.value) or "")
        return code
    if fmt in {"math_expression", "numeric"}:
        value = answer_object.fields.get("recoverable_value", answer_object.value)
        return str(value if value is not None else "0")
    recovered = answer_object.fields.get("recoverable_value", answer_object.value)
    return str(recovered if recovered is not None else "")


def contract_residual(
    answer_object: AnswerObject,
    *,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[dict],
) -> ContractResidual:
    del metadata
    fmt = str(answer_format or "").strip()
    subtype = str(task_subtype or "")
    contract_valid = bool(answer_object.fields.get("contract_valid", answer_object.valid))
    recoverable_valid = bool(answer_object.fields.get("recoverable_valid", answer_object.valid))
    parse_residual = 0.0 if contract_valid else (0.3 if recoverable_valid else 1.0)
    completeness_residual = 0.0
    constraint_residual = 0.0
    execution_residual = 0.0

    if fmt == "option":
        value = answer_object.fields.get("recoverable_value", answer_object.value)
        completeness_residual = 0.0 if value is not None else 1.0
        constraint_residual = 0.0 if isinstance(value, int) and value >= 1 else 1.0
        execution_residual = 0.0 if contract_valid else 0.5
    elif fmt == "graph_json":
        raw_json = answer_object.fields.get("raw_json")
        if subtype in {"connectivity", "cycle"}:
            completeness_residual = 0.0 if isinstance(raw_json, dict) and "answer" in raw_json else (0.4 if recoverable_valid else 1.0)
        elif subtype == "flow":
            completeness_residual = 0.0 if isinstance(raw_json, dict) and "max_flow" in raw_json else (0.4 if recoverable_valid else 1.0)
        elif subtype in {"topology", "hamilton", "shortest_path"}:
            keys = {"topology": "order", "hamilton": "path", "shortest_path": "path"}
            want_key = keys[subtype]
            completeness_residual = 0.0 if isinstance(raw_json, dict) and want_key in raw_json else (0.4 if recoverable_valid else 1.0)
            seq = answer_object.fields.get("recoverable_value")
            constraint_residual = 0.0 if isinstance(seq, list) and len(seq) > 0 else (0.5 if recoverable_valid else 1.0)
        elif subtype.lower() == "gnn":
            payload = answer_object.fields.get("recoverable_value")
            completeness_residual = 0.0 if isinstance(raw_json, dict) and "node_embeddings" in raw_json else (0.5 if recoverable_valid else 1.0)
            constraint_residual = 0.0 if isinstance(payload, dict) and len(payload) > 0 else (0.5 if recoverable_valid else 1.0)
        else:
            completeness_residual = 0.0 if contract_valid else (0.5 if recoverable_valid else 1.0)
        execution_residual = 0.0 if recoverable_valid else 1.0
    elif fmt == "python_code":
        syntax_ok = bool(answer_object.fields.get("syntax_ok", False))
        entry_point = str(answer_object.fields.get("entry_point", "") or "")
        completeness_residual = 0.0 if entry_point else 0.3
        constraint_residual = 0.0 if syntax_ok else 1.0
        execution_residual = 0.0 if syntax_ok else 1.0
    elif fmt in {"math_expression", "numeric"}:
        recovered = answer_object.fields.get("recoverable_value", answer_object.value)
        completeness_residual = 0.0 if recovered not in {None, ""} else 1.0
        constraint_residual = 0.0 if recoverable_valid else 1.0
        execution_residual = 0.0 if contract_valid else (0.35 if recoverable_valid else 1.0)
    else:
        completeness_residual = 0.0 if recoverable_valid else 1.0
        constraint_residual = 0.0 if recoverable_valid else 1.0
        execution_residual = 0.0 if recoverable_valid else 1.0

    return ContractResidual(
        parse=_clamp01(parse_residual),
        completeness=_clamp01(completeness_residual),
        constraint=_clamp01(constraint_residual),
        execution=_clamp01(execution_residual),
    )


def typed_distance(left: AnswerObject, right: AnswerObject) -> float:
    if left.kind != right.kind:
        return 1.0
    if left.signature == right.signature:
        return 0.0
    if left.kind == "option":
        return 1.0 if left.value != right.value else 0.0
    if left.kind in {"math_expression", "numeric"}:
        left_value = safe_float(left.value)
        right_value = safe_float(right.value)
        if left_value is None or right_value is None:
            return 1.0
        denom = max(1.0, abs(right_value))
        return _clamp01(abs(left_value - right_value) / denom)
    if left.kind == "graph_json":
        left_value = left.fields.get("recoverable_value", left.value)
        right_value = right.fields.get("recoverable_value", right.value)
        if isinstance(left_value, list) and isinstance(right_value, list):
            left_set = set(map(str, left_value))
            right_set = set(map(str, right_value))
            union = left_set | right_set
            if not union:
                return 0.0
            return 1.0 - float(len(left_set & right_set)) / float(len(union))
        if isinstance(left_value, dict) and isinstance(right_value, dict):
            left_keys = set(map(str, left_value.keys()))
            right_keys = set(map(str, right_value.keys()))
            union = left_keys | right_keys
            if not union:
                return 0.0
            return 1.0 - float(len(left_keys & right_keys)) / float(len(union))
        return 1.0 if left_value != right_value else 0.0
    if left.kind == "code":
        left_entry = str(left.fields.get("entry_point", "") or "")
        right_entry = str(right.fields.get("entry_point", "") or "")
        if left_entry and right_entry and left_entry != right_entry:
            return 1.0
        left_code = str(left.fields.get("recoverable_value", left.value) or "")
        right_code = str(right.fields.get("recoverable_value", right.value) or "")
        return 0.0 if _safe_sha(left_code) == _safe_sha(right_code) else 1.0
    return 1.0


def signature_embedding(answer_object: AnswerObject) -> Sequence[float]:
    signature = answer_object.signature or ""
    digest = sha1(signature.encode("utf-8")).digest()
    return [float(b) / 255.0 for b in digest[:16]]
