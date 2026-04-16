from __future__ import annotations

import ast
import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


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
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "units": [asdict(unit) for unit in self.units],
            "edges": [dict(edge) for edge in self.edges],
            "views": {name: asdict(view) for name, view in self.views.items()},
            "rendered_answer": self.rendered_answer,
            "provenance": [dict(item) for item in self.provenance],
            "parse_confidence": dict(self.parse_confidence),
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
    if not units:
        return 0.0
    if not provenance:
        return 0.0
    return clamp01(float(len(provenance)) / float(len(units)))


def canonicalize_candidate(
    *,
    candidate_text: str,
    provenance: Optional[Sequence[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
    task_type: str = "",
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

    units = [
        ArtifactUnit(unit_id=f"u{index + 1}", text=raw_unit, unit_type=unit_type, position=index)
        for index, raw_unit in enumerate(raw_units)
    ]
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
    return ArtifactIR(
        units=units,
        edges=edges,
        views=views,
        rendered_answer=_render_units(units, separator=separator),
        provenance=provenance_list,
        parse_confidence=parse_confidence,
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
    vectors = [lexical_feature_vector(unit.text) for unit in artifact.units]
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

