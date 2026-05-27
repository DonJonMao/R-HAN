from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .contracts import AnswerObject, ContractResidual, contract_residual, parse_answer_object
from .parsing_utils import extract_python_code, strip_hidden_reasoning


def _normalized(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).lower()


def _candidate_lines(text: str) -> List[str]:
    cleaned = strip_hidden_reasoning(text or "")
    return [line.strip() for line in cleaned.splitlines() if line.strip()]


def _answer_surface(answer_object: AnswerObject) -> str:
    value = answer_object.fields.get("recoverable_value", answer_object.value)
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


@dataclass
class ArtifactUnit:
    unit_id: str
    text: str
    position: int
    kind: str
    role_probs: Dict[str, float]
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class ArtifactIR:
    dataset_name: str
    answer_format: str
    task_subtype: str
    question_text: str
    candidate_text: str
    answer_object: AnswerObject
    answer_signature: str
    units: List[ArtifactUnit]
    answer_unit_ids: List[str]
    evidence_unit_ids: List[str]
    mixed_unit_ids: List[str]
    schema_features: Dict[str, Any]
    contract_residual: ContractResidual


def _build_graph_units(text: str, answer_object: AnswerObject, task_subtype: str) -> List[str]:
    raw_json = answer_object.fields.get("raw_json")
    if isinstance(raw_json, dict) and raw_json:
        units = []
        for key, value in raw_json.items():
            rendered = json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value)
            units.append(f"{key}: {rendered}")
        if units:
            return units
    recovered = answer_object.fields.get("recoverable_value", answer_object.value)
    if isinstance(recovered, list):
        return [f"{task_subtype or 'path'}[{idx}] = {value}" for idx, value in enumerate(recovered)]
    if isinstance(recovered, dict):
        return [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in recovered.items()]
    lines = _candidate_lines(text)
    return lines if lines else [str(recovered or text or "{}")]


def _build_code_units(text: str) -> List[str]:
    code = extract_python_code(text)
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return lines if lines else ["pass"]


def _build_generic_units(text: str) -> List[str]:
    lines = _candidate_lines(text)
    if lines:
        return lines
    chunks = [chunk.strip() for chunk in re.split(r"(?<=[.!?])\s+", strip_hidden_reasoning(text or "")) if chunk.strip()]
    return chunks if chunks else [str(text or "").strip()]


def _base_units(
    *,
    text: str,
    answer_format: str,
    answer_object: AnswerObject,
    task_subtype: str,
) -> List[str]:
    if answer_format == "graph_json":
        return _build_graph_units(text, answer_object, task_subtype)
    if answer_format == "python_code":
        return _build_code_units(text)
    return _build_generic_units(text)


def _role_probs(
    *,
    unit_text: str,
    position: int,
    total: int,
    answer_object: AnswerObject,
    answer_format: str,
) -> Dict[str, float]:
    normalized_unit = _normalized(unit_text)
    answer_surface = _normalized(_answer_surface(answer_object))
    answer_score = 0.1
    evidence_score = 0.2
    mixed_score = 0.2
    if answer_surface and answer_surface in normalized_unit:
        answer_score += 0.55
    if re.search(r"\b(final answer|answer|option)\b", normalized_unit):
        answer_score += 0.45
    if answer_format == "python_code":
        if re.match(r"\s*def\s+\w+", unit_text):
            answer_score += 0.35
            mixed_score += 0.2
        if "return" in normalized_unit:
            answer_score += 0.2
            evidence_score += 0.15
    if any(token in normalized_unit for token in ("because", "therefore", "thus", "so ", "=")):
        evidence_score += 0.35
    if answer_format == "graph_json" and ":" in unit_text:
        answer_score += 0.25
        evidence_score += 0.1
    if position >= max(0, total - 2):
        answer_score += 0.2
    if position == 0:
        mixed_score += 0.1
    total_score = answer_score + evidence_score + mixed_score
    return {
        "answer": answer_score / total_score,
        "evidence": evidence_score / total_score,
        "mixed": mixed_score / total_score,
    }


def canonicalize_candidate(
    *,
    question_text: str,
    candidate_text: str,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[Dict[str, Any]] = None,
    answer_role_threshold: float = 0.58,
    evidence_role_threshold: float = 0.52,
) -> ArtifactIR:
    answer_object = parse_answer_object(
        candidate_text,
        dataset_name=dataset_name,
        answer_format=answer_format,
        task_subtype=task_subtype,
        metadata=metadata,
    )
    residual = contract_residual(
        answer_object,
        answer_format=answer_format,
        task_subtype=task_subtype,
        metadata=metadata,
    )
    raw_units = _base_units(
        text=candidate_text,
        answer_format=answer_format,
        answer_object=answer_object,
        task_subtype=task_subtype,
    )
    units: List[ArtifactUnit] = []
    answer_ids: List[str] = []
    evidence_ids: List[str] = []
    mixed_ids: List[str] = []
    for index, unit_text in enumerate(raw_units):
        role_probs = _role_probs(
            unit_text=unit_text,
            position=index,
            total=len(raw_units),
            answer_object=answer_object,
            answer_format=answer_format,
        )
        unit_id = f"u{index}"
        features = {
            "length": float(len(unit_text)),
            "digit_ratio": float(sum(ch.isdigit() for ch in unit_text)) / max(1.0, float(len(unit_text))),
            "position_ratio": float(index + 1) / max(1.0, float(len(raw_units))),
            "contains_answer_surface": 1.0 if _answer_surface(answer_object) and _normalized(_answer_surface(answer_object)) in _normalized(unit_text) else 0.0,
        }
        kind = "answer" if role_probs["answer"] >= max(role_probs["evidence"], role_probs["mixed"]) else "evidence"
        unit = ArtifactUnit(unit_id=unit_id, text=unit_text, position=index, kind=kind, role_probs=role_probs, features=features)
        units.append(unit)
        if role_probs["answer"] >= answer_role_threshold:
            answer_ids.append(unit_id)
        elif role_probs["evidence"] >= evidence_role_threshold:
            evidence_ids.append(unit_id)
        else:
            mixed_ids.append(unit_id)
    if not answer_ids and units:
        answer_ids = [units[-1].unit_id]
    if not evidence_ids and len(units) > 1:
        evidence_ids = [unit.unit_id for unit in units[:-1]]
    if not mixed_ids:
        mixed_ids = [unit.unit_id for unit in units if unit.unit_id not in set(answer_ids) | set(evidence_ids)]
    schema_features = {
        "contract_valid": bool(answer_object.fields.get("contract_valid", answer_object.valid)),
        "recoverable_valid": bool(answer_object.fields.get("recoverable_valid", answer_object.valid)),
        "answer_kind": answer_object.kind,
        "task_subtype": task_subtype,
        "unit_count": len(units),
        "text_len": len(candidate_text or ""),
        "question_len": len(question_text or ""),
    }
    return ArtifactIR(
        dataset_name=dataset_name,
        answer_format=answer_format,
        task_subtype=task_subtype,
        question_text=question_text,
        candidate_text=candidate_text,
        answer_object=answer_object,
        answer_signature=answer_object.signature,
        units=units,
        answer_unit_ids=answer_ids,
        evidence_unit_ids=evidence_ids,
        mixed_unit_ids=mixed_ids,
        schema_features=schema_features,
        contract_residual=residual,
    )


def unit_text_overlap(left: Sequence[ArtifactUnit], right: Sequence[ArtifactUnit]) -> float:
    left_set = {_normalized(unit.text) for unit in left if unit.text.strip()}
    right_set = {_normalized(unit.text) for unit in right if unit.text.strip()}
    union = left_set | right_set
    if not union:
        return 0.0
    return float(len(left_set & right_set)) / float(len(union))
