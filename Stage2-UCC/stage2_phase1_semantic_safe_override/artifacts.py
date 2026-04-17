from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def sparsemax(values: Sequence[float]) -> List[float]:
    scores = [float(value) for value in values]
    if not scores:
        return []
    sorted_scores = sorted(scores, reverse=True)
    cumulative = 0.0
    support = 0
    tau = 0.0
    for index, score in enumerate(sorted_scores, start=1):
        cumulative += score
        candidate_tau = (cumulative - 1.0) / float(index)
        if score > candidate_tau:
            support = index
            tau = candidate_tau
    if support <= 0:
        return [0.0 for _ in scores]
    return [max(0.0, score - tau) for score in scores]


def softmax(values: Sequence[float], *, temperature: float = 1.0) -> List[float]:
    scores = [float(value) for value in values]
    if not scores:
        return []
    temperature = max(1e-6, float(temperature))
    max_score = max(scores)
    exps = [math.exp((score - max_score) / temperature) for score in scores]
    total = sum(exps)
    if total <= 0.0:
        return [1.0 / float(len(scores)) for _ in scores]
    return [value / total for value in exps]


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(left, right))
    left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
    right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
    if left_norm <= 1e-9 or right_norm <= 1e-9:
        return 0.0
    return dot / (left_norm * right_norm)


def lexical_feature_vector(text: str, *, dim: int = 32) -> List[float]:
    vector = [0.0 for _ in range(dim)]
    tokens = re.findall(r"[A-Za-z_]\w+|\d+|[^\s]", text.lower())
    if not tokens:
        return vector
    for token in tokens:
        index = hash(token) % dim
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm <= 1e-9:
        return vector
    return [value / norm for value in vector]


@dataclass
class ArtifactUnit:
    unit_id: str
    text: str
    unit_type: str
    position: int
    role_scores: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactView:
    name: str
    rendered: str
    unit_ids: List[str]
    confidence_clue: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactIR:
    units: List[ArtifactUnit]
    edges: List[Dict[str, str]]
    views: Dict[str, ArtifactView]
    rendered_answer: str
    provenance: List[Dict[str, Any]]
    parse_confidence: Dict[str, float]
    answer_unit_ids: List[str]
    evidence_unit_ids: List[str]
    schema_features: Dict[str, Any]
    answer_signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [asdict(unit) for unit in self.units],
            "edges": [dict(edge) for edge in self.edges],
            "views": {name: asdict(view) for name, view in self.views.items()},
            "rendered_answer": self.rendered_answer,
            "provenance": [dict(item) for item in self.provenance],
            "parse_confidence": dict(self.parse_confidence),
            "answer_unit_ids": list(self.answer_unit_ids),
            "evidence_unit_ids": list(self.evidence_unit_ids),
            "schema_features": dict(self.schema_features),
            "answer_signature": self.answer_signature,
            "metadata": dict(self.metadata),
        }


def _split_code_units(text: str) -> List[str]:
    lines = [line.rstrip() for line in text.splitlines()]
    meaningful = [line for line in lines if line.strip()]
    return meaningful or [text.strip()]


def _split_sentence_units(text: str) -> List[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if "\n" in stripped:
        lines = [line.strip() for line in stripped.splitlines() if line.strip()]
        if lines:
            return lines
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    normalized = [part.strip() for part in parts if part.strip()]
    return normalized or [stripped]


def _split_struct_units(text: str) -> tuple[List[str], Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return [], {"mode": "empty"}
    try:
        parsed = json.loads(stripped)
    except Exception:
        parsed = None
    if isinstance(parsed, dict):
        units = [f"{key}: {json.dumps(value, ensure_ascii=False)}" for key, value in parsed.items()]
        return units, {"mode": "json_dict", "field_count": len(units)}
    if isinstance(parsed, list):
        units = [json.dumps(item, ensure_ascii=False) for item in parsed]
        return units, {"mode": "json_list", "field_count": len(units)}
    bullet_lines = [
        line.strip()
        for line in stripped.splitlines()
        if re.match(r"^(\d+[\).\s]|[-*•]\s+)", line.strip())
    ]
    if bullet_lines:
        return bullet_lines, {"mode": "bullet_list", "field_count": len(bullet_lines)}
    return [], {"mode": "unavailable"}


def _split_exec_units(text: str, *, is_code: bool) -> tuple[List[str], Dict[str, Any]]:
    stripped = text.strip()
    if not stripped:
        return [], {"mode": "empty", "parse_ok": False}
    if is_code:
        try:
            tree = ast.parse(stripped)
        except Exception:
            return [], {"mode": "python", "parse_ok": False}
        exec_units: List[str] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                exec_units.append(node.name)
        if not exec_units:
            exec_units = _split_code_units(stripped)[:4]
        return exec_units, {"mode": "python", "parse_ok": True, "symbol_count": len(exec_units)}
    if re.search(r"[=+\-*/()]", stripped):
        return [part.strip() for part in stripped.splitlines() if part.strip()] or [stripped], {
            "mode": "expression",
            "parse_ok": True,
        }
    return [], {"mode": "unavailable", "parse_ok": False}


def _render_units(units: Sequence[ArtifactUnit], *, separator: str) -> str:
    return separator.join(unit.text for unit in units).strip()


def _provenance_coverage(units: Sequence[ArtifactUnit], provenance: Sequence[Dict[str, Any]]) -> float:
    if not units or not provenance:
        return 0.0
    return clamp01(float(len(provenance)) / float(len(units)))


def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _extract_option_label(text: str) -> str:
    match = re.search(r"(?:^|\b)(?:answer\s*[:：]\s*|option\s*)([A-D])(?:\b|\)|\.)", text.strip(), re.IGNORECASE)
    if match:
        return match.group(1).upper()
    stripped = text.strip().upper()
    if re.fullmatch(r"[A-D]", stripped):
        return stripped
    return ""


def _extract_numeric_answer(text: str) -> str:
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:/\d+)?", text.replace(",", ""))
    return match.group(0) if match else ""


def _looks_like_answer_line(text: str, *, task_type: str, position: int, total_units: int) -> bool:
    stripped = text.strip()
    lower = stripped.lower()
    if not stripped:
        return False
    if task_type == "code_generation":
        return True
    if lower.startswith(("answer:", "final answer", "final:", "therefore", "thus")):
        return True
    if _extract_option_label(stripped):
        return True
    if position >= max(0, total_units - 2) and len(stripped.split()) <= 8:
        return True
    if len(stripped.split()) <= 4 and _extract_numeric_answer(stripped):
        return True
    if stripped.startswith("{") or stripped.startswith("["):
        return True
    return False


def _role_scores_for_unit(
    text: str,
    *,
    task_type: str,
    position: int,
    total_units: int,
    answer_role_threshold: float,
    evidence_role_threshold: float,
) -> Dict[str, float]:
    stripped = text.strip()
    lower = stripped.lower()
    answer_score = 0.15
    evidence_score = 0.20
    mixed_score = 0.10
    if task_type == "code_generation":
        answer_score = 0.72
        evidence_score = 0.58 if any(token in lower for token in ("def ", "class ", "return", "assert ")) else 0.42
        mixed_score = 0.30
    else:
        if _looks_like_answer_line(stripped, task_type=task_type, position=position, total_units=total_units):
            answer_score += 0.62
        if lower.startswith(("because", "since", "step", "first", "second", "therefore", "so ")):
            evidence_score += 0.45
        if any(token in lower for token in ("because", "proof", "reason", "constraint", "path", "derive", "compute")):
            evidence_score += 0.20
        if len(stripped.split()) > 10:
            evidence_score += 0.12
        if position == total_units - 1:
            answer_score += 0.12
        if position == 0 and total_units > 1:
            evidence_score += 0.08
        if stripped.startswith("{") or stripped.startswith("["):
            answer_score += 0.18
            mixed_score += 0.12
    answer_score = clamp01(answer_score)
    evidence_score = clamp01(evidence_score)
    mixed_score = clamp01(mixed_score + 0.25 * min(answer_score, evidence_score))
    boosted_answer = max(answer_score, answer_role_threshold if answer_score >= answer_role_threshold else answer_score)
    boosted_evidence = max(evidence_score, evidence_role_threshold if evidence_score >= evidence_role_threshold else evidence_score)
    total = boosted_answer + boosted_evidence + mixed_score
    if total <= 1e-9:
        return {"answer": 0.0, "evidence": 0.0, "mixed": 1.0}
    return {
        "answer": boosted_answer / total,
        "evidence": boosted_evidence / total,
        "mixed": mixed_score / total,
    }


def _schema_features(
    text: str,
    *,
    task_type: str,
    units: Sequence[str],
    struct_units: Sequence[str],
    exec_meta: Dict[str, Any],
) -> Dict[str, Any]:
    option = _extract_option_label(text)
    numeric = _extract_numeric_answer(text)
    answer_kind = "text"
    if task_type == "code_generation":
        answer_kind = "code"
    elif option:
        answer_kind = "multiple_choice"
    elif numeric:
        answer_kind = "numeric"
    elif struct_units:
        answer_kind = "structured"
    return {
        "task_type": task_type,
        "answer_kind": answer_kind,
        "unit_count": int(len(units)),
        "has_json_shape": bool(struct_units),
        "has_code_shape": bool(task_type == "code_generation"),
        "has_option_label": bool(option),
        "has_numeric_answer": bool(numeric),
        "exec_parse_ok": bool(exec_meta.get("parse_ok", False)),
        "answer_token_count": int(len(_normalize_text(text).split())),
    }


def _answer_signature(
    answer_text: str,
    *,
    schema_features: Dict[str, Any],
    metadata: Dict[str, Any],
) -> str:
    normalized = _normalize_text(answer_text)
    answer_kind = str(schema_features.get("answer_kind", "text"))
    if answer_kind == "code":
        entry_point = str(metadata.get("entry_point", "")).strip()
        return f"code::{entry_point}::{hash(normalized) & 0xfffffff:x}"
    option = _extract_option_label(answer_text)
    if option:
        return f"option::{option}"
    numeric = _extract_numeric_answer(answer_text)
    if numeric:
        return f"numeric::{numeric}"
    if answer_kind == "structured":
        try:
            parsed = json.loads(answer_text)
            return "structured::" + json.dumps(parsed, ensure_ascii=False, sort_keys=True)
        except Exception:
            pass
    compact = normalized[:96]
    return f"text::{compact}"


def canonicalize_candidate(
    *,
    candidate_text: str,
    provenance: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_type: str = "",
    answer_role_threshold: float = 0.58,
    evidence_role_threshold: float = 0.52,
) -> ArtifactIR:
    text = str(candidate_text or "").strip()
    metadata = dict(metadata or {})
    provenance_list = [dict(item) for item in (provenance or []) if isinstance(item, dict)]
    is_code = task_type == "code_generation"

    if is_code:
        raw_units = _split_code_units(text)
        unit_type = "code_line"
        separator = "\n"
    else:
        raw_units = _split_sentence_units(text)
        unit_type = "text_span"
        separator = "\n" if "\n" in text else " "

    units: List[ArtifactUnit] = []
    total_units = max(1, len(raw_units))
    for index, raw_unit in enumerate(raw_units):
        role_scores = _role_scores_for_unit(
            raw_unit,
            task_type=task_type,
            position=index,
            total_units=total_units,
            answer_role_threshold=answer_role_threshold,
            evidence_role_threshold=evidence_role_threshold,
        )
        unit_role = max(role_scores.items(), key=lambda item: item[1])[0]
        units.append(
            ArtifactUnit(
                unit_id=f"u{index + 1}",
                text=raw_unit,
                unit_type=unit_type,
                position=index,
                role_scores=role_scores,
                metadata={"unit_role": unit_role},
            )
        )

    edges = [
        {"src": units[index - 1].unit_id, "dst": unit.unit_id, "type": "sequential"}
        for index, unit in enumerate(units)
        if index > 0
    ]

    step_units = _split_sentence_units(text)
    struct_units, struct_meta = _split_struct_units(text)
    exec_units, exec_meta = _split_exec_units(text, is_code=is_code)

    provenance_coverage = _provenance_coverage(units, provenance_list)
    surface_conf = 1.0 if text else 0.0
    step_conf = clamp01(0.35 + 0.15 * min(len(step_units), 4) + 0.15 * provenance_coverage)
    struct_conf = 0.1
    if struct_units:
        struct_conf = clamp01(0.55 + 0.05 * min(len(struct_units), 4) + 0.15 * provenance_coverage)
    exec_conf = 0.1
    if exec_meta.get("parse_ok"):
        exec_conf = clamp01(0.60 + 0.15 * provenance_coverage)

    views = {
        "surface_view": ArtifactView(
            name="surface_view",
            rendered=text,
            unit_ids=[unit.unit_id for unit in units],
            confidence_clue=surface_conf,
            metadata={"mode": "raw_text"},
        ),
        "step_view": ArtifactView(
            name="step_view",
            rendered="\n".join(step_units),
            unit_ids=[f"u{index + 1}" for index in range(min(len(step_units), len(units)))],
            confidence_clue=step_conf,
            metadata={"step_count": len(step_units)},
        ),
        "struct_view": ArtifactView(
            name="struct_view",
            rendered="\n".join(struct_units),
            unit_ids=[f"u{index + 1}" for index in range(min(len(struct_units), len(units)))],
            confidence_clue=struct_conf,
            metadata=struct_meta,
        ),
        "exec_view": ArtifactView(
            name="exec_view",
            rendered="\n".join(exec_units),
            unit_ids=[f"u{index + 1}" for index in range(min(len(exec_units), len(units)))],
            confidence_clue=exec_conf,
            metadata=exec_meta,
        ),
    }
    parse_confidence = {name: clamp01(view.confidence_clue) for name, view in views.items()}

    answer_unit_ids = [
        unit.unit_id
        for unit in units
        if float(unit.role_scores.get("answer", 0.0)) >= float(answer_role_threshold)
    ]
    evidence_unit_ids = [
        unit.unit_id
        for unit in units
        if float(unit.role_scores.get("evidence", 0.0)) >= float(evidence_role_threshold)
    ]
    if not answer_unit_ids and units:
        answer_unit_ids = [units[-1].unit_id]
    if not evidence_unit_ids and len(units) > 1:
        evidence_unit_ids = [unit.unit_id for unit in units[:-1]]
    if not evidence_unit_ids and units:
        evidence_unit_ids = [units[0].unit_id]

    answer_text_value = "\n".join(unit.text for unit in units if unit.unit_id in answer_unit_ids).strip() or text
    schema_features = _schema_features(
        answer_text_value,
        task_type=task_type,
        units=raw_units,
        struct_units=struct_units,
        exec_meta=exec_meta,
    )
    answer_signature = _answer_signature(
        answer_text_value,
        schema_features=schema_features,
        metadata=metadata,
    )

    return ArtifactIR(
        units=units,
        edges=edges,
        views=views,
        rendered_answer=_render_units(units, separator=separator),
        provenance=provenance_list,
        parse_confidence=parse_confidence,
        answer_unit_ids=answer_unit_ids,
        evidence_unit_ids=evidence_unit_ids,
        schema_features=schema_features,
        answer_signature=answer_signature,
        metadata={
            "task_type": task_type,
            "provenance_coverage": provenance_coverage,
            "render_separator": separator,
            **metadata,
        },
    )


def pooled_artifact_vector(artifact: ArtifactIR) -> List[float]:
    if not artifact.units:
        return [0.0 for _ in range(32)]
    vectors = []
    for unit in artifact.units:
        weight = 1.0 + 0.35 * float(unit.role_scores.get("answer", 0.0)) + 0.15 * float(unit.role_scores.get("evidence", 0.0))
        base = lexical_feature_vector(unit.text)
        vectors.append([weight * value for value in base])
    dim = len(vectors[0])
    pooled = [0.0 for _ in range(dim)]
    for vector in vectors:
        for index, value in enumerate(vector):
            pooled[index] += value
    pooled = [value / float(len(vectors)) for value in pooled]
    norm = math.sqrt(sum(value * value for value in pooled))
    if norm <= 1e-9:
        return pooled
    return [value / norm for value in pooled]


def unit_texts(artifact: ArtifactIR) -> List[str]:
    return [unit.text for unit in artifact.units]


def answer_units(artifact: ArtifactIR) -> List[ArtifactUnit]:
    answer_ids = set(artifact.answer_unit_ids)
    return [unit for unit in artifact.units if unit.unit_id in answer_ids]


def evidence_units(artifact: ArtifactIR) -> List[ArtifactUnit]:
    evidence_ids = set(artifact.evidence_unit_ids)
    return [unit for unit in artifact.units if unit.unit_id in evidence_ids]


def answer_text(artifact: ArtifactIR) -> str:
    units = answer_units(artifact)
    if not units:
        return artifact.rendered_answer
    separator = str(artifact.metadata.get("render_separator", " "))
    return _render_units(units, separator=separator)
