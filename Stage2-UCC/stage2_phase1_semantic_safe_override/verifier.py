from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stage2_gcr_plus.code_repair import evaluate_code_candidate

from .artifacts import (
    ArtifactIR,
    answer_text,
    answer_units,
    clamp01,
    cosine_similarity,
    evidence_units,
    lexical_feature_vector,
    pooled_artifact_vector,
    sparsemax,
)


RESIDUAL_KEYS: tuple[str, ...] = (
    "r_parse",
    "r_consistency",
    "r_completeness",
    "r_execution",
    "r_constraint",
    "r_support",
    "r_preserve",
)


@dataclass
class VerifierState:
    residual_vector: Dict[str, float]
    unit_error_map: Dict[str, float]
    support_map: Dict[str, float]
    completeness_score: float
    consistency_score: float
    answer_consistency_score: float
    executability_score: float
    constraint_score: float
    confidence_score: float
    progress_score: float
    preserve_risk: float
    anchor_similarity: float
    answer_similarity: float
    answer_delta: float
    overturn_risk: float
    channel_weights: Dict[str, float] = field(default_factory=dict)
    meta_summary: str = ""
    issues: List[str] = field(default_factory=list)
    missing_requirements: List[str] = field(default_factory=list)
    executor_feedback: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _mean(values: Sequence[float]) -> float:
    values_list = [float(value) for value in values]
    if not values_list:
        return 0.0
    return sum(values_list) / float(len(values_list))


def mean_residual(state: VerifierState) -> float:
    return _mean(list(state.residual_vector.values()))


def _extract_question_keywords(question_text: str) -> List[str]:
    words = re.findall(r"[A-Za-z_]\w+", question_text.lower())
    stop = {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "your",
        "return",
        "write",
        "given",
        "input",
        "output",
        "following",
    }
    seen: List[str] = []
    for word in words:
        if len(word) < 4 or word in stop or word in seen:
            continue
        seen.append(word)
    return seen[:8]


def _answer_similarity(candidate: ArtifactIR, anchor_artifact: Optional[ArtifactIR]) -> float:
    if anchor_artifact is None:
        return 0.0
    if candidate.answer_signature == anchor_artifact.answer_signature and candidate.answer_signature:
        return 1.0
    candidate_vec = lexical_feature_vector(answer_text(candidate))
    anchor_vec = lexical_feature_vector(answer_text(anchor_artifact))
    return clamp01((cosine_similarity(candidate_vec, anchor_vec) + 1.0) * 0.5)


def _answer_delta(candidate: ArtifactIR, anchor_artifact: Optional[ArtifactIR]) -> float:
    if anchor_artifact is None:
        return 0.0
    similarity = _answer_similarity(candidate, anchor_artifact)
    return clamp01(1.0 - similarity)


def _executor_channel(
    artifact: ArtifactIR,
    *,
    metadata: Optional[Dict[str, Any]],
    task_type: str,
    timeout_s: float,
    max_failed_examples: int,
) -> Tuple[Dict[str, float], float, Dict[str, Any]]:
    if task_type != "code_generation":
        exec_conf = float(artifact.parse_confidence.get("exec_view", 0.0))
        return (
            {
                "r_execution": 1.0 - exec_conf,
                "r_constraint": 1.0 - min(1.0, exec_conf + 0.15),
            },
            exec_conf,
            {"available": False, "task_type": task_type},
        )
    feedback = evaluate_code_candidate(
        artifact.rendered_answer,
        metadata,
        timeout_s=timeout_s,
        max_failed_examples=max_failed_examples,
    )
    accuracy = clamp01(feedback.accuracy if feedback.total > 0 else 0.0)
    constraint_ok = 1.0 if feedback.entry_point_ok else 0.0
    parse_ok = 1.0 if feedback.syntax_ok else 0.0
    quality = clamp01(0.25 * parse_ok + 0.25 * constraint_ok + 0.50 * accuracy)
    return (
        {
            "r_execution": clamp01(1.0 - accuracy if feedback.total > 0 else 1.0 - parse_ok),
            "r_constraint": clamp01(1.0 - constraint_ok),
            "r_parse": clamp01(1.0 - parse_ok),
        },
        quality,
        {
            "available": True,
            "syntax_ok": feedback.syntax_ok,
            "entry_point_ok": feedback.entry_point_ok,
            "passed": feedback.passed,
            "total": feedback.total,
            "failure_kind": feedback.failure_kind,
            "failing_examples": list(feedback.failing_examples),
            "stderr": feedback.stderr,
            "exec_error": feedback.exec_error,
            "syntax_error_line": feedback.syntax_error_line,
            "syntax_error_offset": feedback.syntax_error_offset,
            "syntax_error_text": feedback.syntax_error_text,
        },
    )


def _support_map(artifact: ArtifactIR, candidate_entry: Optional[Dict[str, Any]]) -> Dict[str, float]:
    provenance = artifact.provenance
    base_coverage = clamp01(float(artifact.metadata.get("provenance_coverage", 0.0)))
    stage1_anchor = bool((candidate_entry or {}).get("stage1_anchor", False))
    reviewer_bonus = clamp01(float((candidate_entry or {}).get("reviewer_mean_trust", 0.5)))
    supports: Dict[str, float] = {}
    answer_ids = set(artifact.answer_unit_ids)
    evidence_ids = set(artifact.evidence_unit_ids)
    for unit in artifact.units:
        score = 0.20 * base_coverage + 0.30 * reviewer_bonus
        if unit.unit_id in evidence_ids:
            score += 0.18
        if unit.unit_id in answer_ids:
            score += 0.12
        if stage1_anchor:
            score += 0.22
        if provenance:
            score += 0.08
        if len(unit.text.strip()) >= 4:
            score += 0.05
        supports[unit.unit_id] = clamp01(score)
    return supports


def _consistency_issues(artifact: ArtifactIR) -> Tuple[List[str], Dict[str, float], float]:
    issues: List[str] = []
    unit_error_map = {unit.unit_id: 0.0 for unit in artifact.units}
    normalized: Dict[str, List[str]] = {}
    for unit in artifact.units:
        text = unit.text.strip().lower()
        if not text:
            unit_error_map[unit.unit_id] = max(unit_error_map[unit.unit_id], 0.4)
            issues.append(f"{unit.unit_id}: empty unit")
            continue
        normalized.setdefault(text, []).append(unit.unit_id)
    duplicates = [unit_ids for unit_ids in normalized.values() if len(unit_ids) > 1]
    for unit_ids in duplicates:
        for unit_id in unit_ids:
            unit_error_map[unit_id] = max(unit_error_map[unit_id], 0.25)
        issues.append(f"duplicate units: {', '.join(unit_ids)}")
    contradiction_score = clamp01(0.25 * len(duplicates))
    answer_ids = set(artifact.answer_unit_ids)
    answer_signatures = {
        re.sub(r"\s+", " ", unit.text.strip().lower())
        for unit in artifact.units
        if unit.unit_id in answer_ids and unit.text.strip()
    }
    if len(answer_signatures) > 1:
        contradiction_score = clamp01(contradiction_score + 0.20)
        issues.append("answer units disagree")
        for unit in artifact.units:
            if unit.unit_id in answer_ids:
                unit_error_map[unit.unit_id] = max(unit_error_map[unit.unit_id], 0.35)
    return issues, unit_error_map, contradiction_score


def _completeness_signal(artifact: ArtifactIR, question_text: str) -> Tuple[float, List[str]]:
    keywords = _extract_question_keywords(question_text)
    answer_lower = artifact.rendered_answer.lower()
    missing = [keyword for keyword in keywords if keyword not in answer_lower]
    if not artifact.rendered_answer.strip():
        return 1.0, ["empty answer"]
    residual = clamp01(float(len(missing)) / float(max(1, len(keywords))))
    return residual, missing


def _answer_consistency(artifact: ArtifactIR, support_map: Dict[str, float], contradiction_score: float) -> float:
    answer_text_value = answer_text(artifact).strip()
    if not answer_text_value:
        return 0.0
    answer_ids = set(artifact.answer_unit_ids)
    evidence_ids = set(artifact.evidence_unit_ids)
    answer_support = _mean([support_map.get(unit_id, 0.0) for unit_id in answer_ids]) if answer_ids else 0.0
    evidence_support = _mean([support_map.get(unit_id, 0.0) for unit_id in evidence_ids]) if evidence_ids else 0.0
    answer_vec = lexical_feature_vector(answer_text_value)
    evidence_text = "\n".join(
        unit.text for unit in artifact.units if unit.unit_id in evidence_ids and unit.text.strip()
    )
    lexical = 0.5
    if evidence_text.strip():
        lexical = clamp01((cosine_similarity(answer_vec, lexical_feature_vector(evidence_text)) + 1.0) * 0.5)
    score = 0.30 + 0.30 * answer_support + 0.20 * evidence_support + 0.20 * lexical
    if artifact.answer_signature.startswith("option::") or artifact.answer_signature.startswith("numeric::"):
        score += 0.08
    score -= 0.35 * contradiction_score
    return clamp01(score)


def _preserve_risk(
    artifact: ArtifactIR,
    *,
    anchor_artifact: Optional[ArtifactIR],
    support_map: Dict[str, float],
) -> Tuple[float, float]:
    if anchor_artifact is None or not anchor_artifact.rendered_answer.strip():
        return 0.0, 0.0
    candidate_vec = pooled_artifact_vector(artifact)
    anchor_vec = pooled_artifact_vector(anchor_artifact)
    similarity = clamp01((cosine_similarity(candidate_vec, anchor_vec) + 1.0) * 0.5)
    stable_support = _mean(list(support_map.values()))
    risk = clamp01((1.0 - similarity) * (0.35 + 0.65 * stable_support))
    return risk, similarity


def verify_artifact(
    artifact: ArtifactIR,
    *,
    question_text: str,
    metadata: Optional[Dict[str, Any]],
    task_type: str,
    candidate_entry: Optional[Dict[str, Any]] = None,
    anchor_artifact: Optional[ArtifactIR] = None,
    timeout_s: float = 8.0,
    max_failed_examples: int = 3,
) -> VerifierState:
    parser_quality = _mean(list(artifact.parse_confidence.values()))
    support_map = _support_map(artifact, candidate_entry)
    issues, unit_error_map, contradiction_score = _consistency_issues(artifact)
    completeness_residual, missing_requirements = _completeness_signal(artifact, question_text)
    preserve_risk, anchor_similarity = _preserve_risk(
        artifact,
        anchor_artifact=anchor_artifact,
        support_map=support_map,
    )
    answer_similarity_score = _answer_similarity(artifact, anchor_artifact)
    answer_delta_value = _answer_delta(artifact, anchor_artifact)
    answer_consistency_score = _answer_consistency(artifact, support_map, contradiction_score)
    executor_residuals, executor_quality, executor_feedback = _executor_channel(
        artifact,
        metadata=metadata,
        task_type=task_type,
        timeout_s=timeout_s,
        max_failed_examples=max_failed_examples,
    )
    search_quality = _mean(list(support_map.values()))
    symbolic_quality = clamp01(1.0 - contradiction_score)
    channel_names = (
        "parser_channel",
        "executor_channel",
        "constraint_channel",
        "search_channel",
        "symbolic_channel",
    )
    channel_quality_values = (
        parser_quality,
        executor_quality,
        clamp01(1.0 - executor_residuals.get("r_constraint", 0.0)),
        search_quality,
        symbolic_quality,
    )
    channel_weights = {
        name: weight
        for name, weight in zip(channel_names, sparsemax(channel_quality_values))
    }

    residual_vector = {
        "r_parse": clamp01(1.0 - parser_quality),
        "r_consistency": contradiction_score,
        "r_completeness": completeness_residual,
        "r_execution": clamp01(executor_residuals.get("r_execution", 0.0)),
        "r_constraint": clamp01(executor_residuals.get("r_constraint", 0.0)),
        "r_support": clamp01(1.0 - search_quality),
        "r_preserve": preserve_risk,
    }

    if executor_feedback.get("syntax_error_line"):
        line_index = int(executor_feedback["syntax_error_line"]) - 1
        if 0 <= line_index < len(artifact.units):
            unit_id = artifact.units[line_index].unit_id
            unit_error_map[unit_id] = max(unit_error_map.get(unit_id, 0.0), 0.9)
    if executor_feedback.get("failure_kind") == "entry_point_missing" and artifact.units:
        unit_error_map[artifact.units[0].unit_id] = max(unit_error_map.get(artifact.units[0].unit_id, 0.0), 0.85)
    for unit_id, support in support_map.items():
        if support < 0.45:
            unit_error_map[unit_id] = max(unit_error_map.get(unit_id, 0.0), clamp01(0.55 - support))

    confidence_score = clamp01(
        0.55 * (1.0 - _mean(list(residual_vector.values())))
        + 0.20 * answer_consistency_score
        + 0.10 * search_quality
        + 0.15 * parser_quality
    )
    progress_score = clamp01(
        0.40 * confidence_score
        + 0.20 * answer_consistency_score
        + 0.20 * (1.0 - residual_vector["r_support"])
        + 0.20 * (1.0 - preserve_risk)
    )
    meta_summary = (
        f"parse={1.0 - residual_vector['r_parse']:.2f} "
        f"consistency={1.0 - residual_vector['r_consistency']:.2f} "
        f"answer_consistency={answer_consistency_score:.2f} "
        f"support={1.0 - residual_vector['r_support']:.2f} "
        f"preserve={1.0 - residual_vector['r_preserve']:.2f}"
    )
    return VerifierState(
        residual_vector={key: clamp01(residual_vector.get(key, 0.0)) for key in RESIDUAL_KEYS},
        unit_error_map={unit_id: clamp01(score) for unit_id, score in unit_error_map.items()},
        support_map={unit_id: clamp01(score) for unit_id, score in support_map.items()},
        completeness_score=clamp01(1.0 - residual_vector["r_completeness"]),
        consistency_score=clamp01(1.0 - residual_vector["r_consistency"]),
        answer_consistency_score=answer_consistency_score,
        executability_score=clamp01(1.0 - residual_vector["r_execution"]),
        constraint_score=clamp01(1.0 - residual_vector["r_constraint"]),
        confidence_score=confidence_score,
        progress_score=progress_score,
        preserve_risk=preserve_risk,
        anchor_similarity=anchor_similarity,
        answer_similarity=answer_similarity_score,
        answer_delta=answer_delta_value,
        overturn_risk=0.0,
        channel_weights=channel_weights,
        meta_summary=meta_summary,
        issues=issues,
        missing_requirements=list(missing_requirements),
        executor_feedback=executor_feedback,
    )


def compute_overturn_risk(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    anchor_artifact: Optional[ArtifactIR],
    anchor_state: Optional[VerifierState],
) -> float:
    if anchor_artifact is None or anchor_state is None:
        return 0.0
    candidate_residual = mean_residual(verifier_state)
    anchor_residual = mean_residual(anchor_state)
    residual_gap = candidate_residual - anchor_residual
    confidence_gap = float(verifier_state.confidence_score) - float(anchor_state.confidence_score)
    risk = (
        0.32 * float(verifier_state.answer_delta)
        + 0.20 * clamp01(residual_gap + 0.5)
        + 0.14 * clamp01(-confidence_gap + 0.5)
        + 0.16 * float(verifier_state.preserve_risk)
        + 0.18 * clamp01(1.0 - verifier_state.answer_consistency_score)
    )
    if artifact.answer_signature == anchor_artifact.answer_signature and artifact.answer_signature:
        risk *= 0.35
    return clamp01(risk)
