from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from stage2_gcr_plus.code_repair import CodeRepairEval, evaluate_code_candidate

from .artifacts import ArtifactIR, clamp01, cosine_similarity, lexical_feature_vector, pooled_artifact_vector, sparsemax


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
    executability_score: float
    constraint_score: float
    confidence_score: float
    progress_score: float
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
    return _mean(state.residual_vector.values())


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
    }
    seen: List[str] = []
    for word in words:
        if len(word) < 4 or word in stop or word in seen:
            continue
        seen.append(word)
    return seen[:8]


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
    for unit in artifact.units:
        score = 0.25 * base_coverage + 0.35 * reviewer_bonus
        if stage1_anchor:
            score += 0.30
        if provenance:
            score += 0.10
        if len(unit.text.strip()) >= 4:
            score += 0.05
        supports[unit.unit_id] = clamp01(score)
    return supports


def _consistency_issues(artifact: ArtifactIR) -> Tuple[List[str], Dict[str, float], float]:
    issues: List[str] = []
    unit_error_map = {unit.unit_id: 0.0 for unit in artifact.units}
    normalized = {}
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
    return issues, unit_error_map, contradiction_score


def _completeness_signal(artifact: ArtifactIR, question_text: str) -> Tuple[float, List[str]]:
    keywords = _extract_question_keywords(question_text)
    answer_text = artifact.rendered_answer.lower()
    missing = [keyword for keyword in keywords if keyword not in answer_text]
    if not artifact.rendered_answer.strip():
        return 1.0, ["empty answer"]
    residual = clamp01(float(len(missing)) / float(max(1, len(keywords))))
    return residual, missing


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
    stable_support = _mean(support_map.values())
    risk = clamp01((1.0 - similarity) * (0.4 + 0.6 * stable_support))
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
    parser_quality = _mean(artifact.parse_confidence.values())
    support_map = _support_map(artifact, candidate_entry)
    issues, unit_error_map, contradiction_score = _consistency_issues(artifact)
    completeness_residual, missing_requirements = _completeness_signal(artifact, question_text)
    preserve_risk, anchor_similarity = _preserve_risk(
        artifact,
        anchor_artifact=anchor_artifact,
        support_map=support_map,
    )
    executor_residuals, executor_quality, executor_feedback = _executor_channel(
        artifact,
        metadata=metadata,
        task_type=task_type,
        timeout_s=timeout_s,
        max_failed_examples=max_failed_examples,
    )
    search_quality = _mean(support_map.values())
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

    confidence_score = clamp01(1.0 - _mean(residual_vector.values()))
    progress_score = clamp01((anchor_similarity + confidence_score) * 0.5)
    meta_summary = (
        f"parse={1.0 - residual_vector['r_parse']:.2f} "
        f"consistency={1.0 - residual_vector['r_consistency']:.2f} "
        f"support={1.0 - residual_vector['r_support']:.2f} "
        f"preserve={1.0 - residual_vector['r_preserve']:.2f}"
    )
    return VerifierState(
        residual_vector={key: clamp01(residual_vector.get(key, 0.0)) for key in RESIDUAL_KEYS},
        unit_error_map={unit_id: clamp01(score) for unit_id, score in unit_error_map.items()},
        support_map={unit_id: clamp01(score) for unit_id, score in support_map.items()},
        completeness_score=clamp01(1.0 - residual_vector["r_completeness"]),
        consistency_score=clamp01(1.0 - residual_vector["r_consistency"]),
        executability_score=clamp01(1.0 - residual_vector["r_execution"]),
        constraint_score=clamp01(1.0 - residual_vector["r_constraint"]),
        confidence_score=confidence_score,
        progress_score=progress_score,
        channel_weights=channel_weights,
        meta_summary=meta_summary,
        issues=issues,
        missing_requirements=list(missing_requirements),
        executor_feedback=executor_feedback,
    )

