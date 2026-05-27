from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence

from common.answer_contracts import AnswerObject
from stage2_phase1_semantic_safe_override.artifacts import (
    ArtifactIR,
    ArtifactUnit,
    answer_text,
    answer_units,
    canonicalize_candidate,
    clamp01,
    evidence_units,
)
from stage2_phase1_semantic_safe_override.verifier import VerifierState, mean_residual


ALLOWED_OPERATIONS = {"replace", "insert_before", "insert_after", "delete", "reorder"}


@dataclass
class CorrectionArtifact:
    target_units: List[str]
    operation: str
    new_units: List[str]
    preserve_units: List[str]
    expected_delta: Dict[str, Any]
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["operation"] = str(self.operation)
        return payload


@dataclass
class DeltaPrediction:
    expected_residual_drop: Dict[str, float]
    expected_confidence_gain: float
    expected_progress_gain: float
    expected_preserve_risk: float
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _render_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _render_typed_answer(answer_object: AnswerObject) -> str:
    semantic_object = answer_object.fields.get("recoverable_object")
    semantic_value = answer_object.fields.get("recoverable_value", answer_object.value)
    if answer_object.kind == "option":
        return f"OPTION - {semantic_value}"
    if answer_object.kind == "bool":
        return str(semantic_value).lower()
    if answer_object.kind == "numeric":
        return str(semantic_value)
    if answer_object.kind == "graph_bool":
        field_name = str(answer_object.fields.get("field", "answer"))
        return _render_json({field_name: semantic_value})
    if answer_object.kind == "graph_scalar":
        field_name = str(answer_object.fields.get("field", "value"))
        return _render_json({field_name: semantic_value})
    if answer_object.kind == "graph_sequence":
        field_name = str(answer_object.fields.get("field", "order"))
        return _render_json({field_name: list(semantic_value or [])})
    if answer_object.kind == "graph_path":
        payload = dict(semantic_object or {})
        if "path" not in payload and semantic_value:
            payload["path"] = list(semantic_value)
        return _render_json(payload)
    if answer_object.kind == "graph_matching":
        payload = dict(semantic_object or {})
        return _render_json(payload)
    if answer_object.kind == "node_embeddings":
        return _render_json({"node_embeddings": dict(semantic_object or semantic_value or {})})
    if answer_object.kind == "code":
        return str(semantic_value or "")
    return str(semantic_value or "")


def answer_first_gate(artifact: ArtifactIR, verifier_state: VerifierState, *, safe_utility: float) -> float:
    answer_mass = sum(float(unit.role_scores.get("answer", 0.0)) for unit in answer_units(artifact))
    evidence_mass = sum(float(unit.role_scores.get("evidence", 0.0)) for unit in evidence_units(artifact))
    ratio = answer_mass / float(max(1e-6, answer_mass + evidence_mass))
    score = (
        0.18
        + 0.30 * float(verifier_state.answer_delta)
        + 0.18 * float(verifier_state.residual_vector.get("r_parse", 0.0))
        + 0.12 * float(verifier_state.residual_vector.get("r_completeness", 0.0))
        + 0.10 * ratio
        + 0.06 * clamp01(0.5 - safe_utility)
        - 0.08 * float(verifier_state.typed_support_score)
    )
    return clamp01(score)


def preserve_heatmap(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    anchor_artifact: Optional[ArtifactIR] = None,
) -> Dict[str, float]:
    anchor_texts = {unit.text.strip() for unit in (anchor_artifact.units if anchor_artifact is not None else []) if unit.text.strip()}
    answer_ids = set(artifact.answer_unit_ids)
    mask: Dict[str, float] = {}
    for unit in artifact.units:
        support = float(verifier_state.support_map.get(unit.unit_id, 0.0))
        anchor_agreement = 1.0 if unit.text.strip() in anchor_texts else 0.0
        answer_bonus = 1.0 if unit.unit_id in answer_ids else 0.0
        score = clamp01(
            0.48 * support
            + 0.22 * answer_bonus
            + 0.22 * anchor_agreement
            + 0.08 * float(unit.role_scores.get("evidence", 0.0))
        )
        mask[unit.unit_id] = score
    return mask


def localize_units(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    preserve_map: Dict[str, float],
    answer_first_score: float,
    previous_delta: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    previous_delta = dict(previous_delta or {})
    heatmap: Dict[str, float] = {}
    for unit in artifact.units:
        unit_error = float(verifier_state.unit_error_map.get(unit.unit_id, 0.0))
        support_penalty = 1.0 - float(verifier_state.support_map.get(unit.unit_id, 0.0))
        preserve_penalty = float(preserve_map.get(unit.unit_id, 0.0))
        role_focus = (
            answer_first_score * float(unit.role_scores.get("answer", 0.0))
            + (1.0 - answer_first_score) * float(unit.role_scores.get("evidence", 0.0))
            + 0.20 * float(unit.role_scores.get("mixed", 0.0))
        )
        recent_effect = float(previous_delta.get(unit.unit_id, 0.0))
        score = clamp01(
            0.42 * unit_error
            + 0.20 * support_penalty
            + 0.24 * role_focus
            + 0.08 * recent_effect
            - 0.28 * preserve_penalty
        )
        heatmap[unit.unit_id] = score
    return heatmap


def critique_summary(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    localization_map: Dict[str, float],
    preserve_map: Dict[str, float],
    answer_first_score: float,
    top_k: int,
) -> Dict[str, Any]:
    ranked_targets = sorted(localization_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    target_units = [unit_id for unit_id, score in ranked_targets[:top_k] if score > 0.0]
    preserve_units = [unit_id for unit_id, score in preserve_map.items() if score >= 0.68]
    if not target_units and artifact.answer_unit_ids:
        target_units = list(artifact.answer_unit_ids[:top_k])
    main_residuals = [
        residual_name
        for residual_name, residual_value in verifier_state.residual_vector.items()
        if residual_value >= 0.30 and residual_name != "r_preserve"
    ]
    if not main_residuals:
        main_residuals = ["r_support"]
    return {
        "target_units": target_units,
        "preserve_units": preserve_units,
        "answer_first_score": float(answer_first_score),
        "main_residuals": main_residuals,
        "issues": list(verifier_state.issues[:3]),
        "typed_support_score": float(verifier_state.typed_support_score),
        "answer_signature": artifact.answer_signature,
    }


def _suggest_typing_import(error_text: str) -> List[str]:
    missing = []
    for symbol in ("List", "Tuple", "Dict", "Set", "Optional"):
        if re.search(rf"name '{symbol}' is not defined", error_text):
            missing.append(symbol)
    if not missing:
        return []
    return [f"from typing import {', '.join(sorted(set(missing)))}"]


def propose_correction(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    critique: Dict[str, Any],
    anchor_artifact: Optional[ArtifactIR] = None,
    anchor_state: Optional[VerifierState] = None,
) -> Optional[CorrectionArtifact]:
    target_units = [str(unit_id) for unit_id in critique.get("target_units", [])]
    preserve_units = [str(unit_id) for unit_id in critique.get("preserve_units", [])]
    answer_object = artifact.answer_object
    if not target_units and artifact.answer_unit_ids:
        target_units = list(artifact.answer_unit_ids)
    if not target_units:
        return None

    if answer_object.kind == "code":
        feedback = verifier_state.executor_feedback
        error_text = " ".join(
            str(value)
            for value in (
                feedback.get("exec_error", ""),
                feedback.get("stderr", ""),
                " ".join(feedback.get("failing_examples", [])),
            )
        )
        typing_import = _suggest_typing_import(error_text)
        if typing_import:
            return CorrectionArtifact(
                target_units=[artifact.units[0].unit_id] if artifact.units else target_units,
                operation="insert_before",
                new_units=typing_import,
                preserve_units=preserve_units,
                expected_delta={"delta_exec": 0.20, "delta_contract": 0.05},
                rationale="restore missing typing symbols required by execution",
            )

    contract_valid = bool(answer_object.fields.get("contract_valid", answer_object.valid))
    recoverable_valid = bool(answer_object.fields.get("recoverable_valid", answer_object.valid))
    if recoverable_valid and not contract_valid:
        canonical_answer = _render_typed_answer(answer_object)
        if canonical_answer.strip():
            return CorrectionArtifact(
                target_units=target_units,
                operation="replace",
                new_units=[canonical_answer],
                preserve_units=preserve_units,
                expected_delta={"delta_contract": 0.45, "delta_parse": 0.45, "delta_answer": 0.10},
                rationale="re-render recoverable typed answer into the dataset contract surface",
            )

    if (
        anchor_artifact is not None
        and anchor_state is not None
        and verifier_state.answer_delta > 0.0
        and anchor_state.typed_support_score > verifier_state.typed_support_score + 0.10
    ):
        anchor_answer = answer_text(anchor_artifact).strip()
        if anchor_answer:
            anchor_target_units = target_units or list(artifact.answer_unit_ids) or [artifact.units[0].unit_id]
            return CorrectionArtifact(
                target_units=anchor_target_units,
                operation="replace",
                new_units=[anchor_answer],
                preserve_units=preserve_units,
                expected_delta={"delta_answer": 0.35, "delta_support": 0.20},
                rationale="restore a higher-support answer unit when typed support strongly favors the anchor",
            )

    if len(set(target_units)) == 1 and mean_residual(verifier_state) >= 0.40:
        return CorrectionArtifact(
            target_units=target_units,
            operation="delete",
            new_units=[],
            preserve_units=preserve_units,
            expected_delta={"delta_support": 0.08},
            rationale="drop the highest-residual local unit when no typed replacement is available",
        )
    return None


def apply_correction_artifact(artifact: ArtifactIR, correction: CorrectionArtifact) -> ArtifactIR:
    if correction.operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported correction operation: {correction.operation}")
    target_set = set(correction.target_units)
    original_units = list(artifact.units)
    new_units: List[ArtifactUnit] = []
    inserted = False
    for unit in original_units:
        if unit.unit_id not in target_set:
            new_units.append(
                ArtifactUnit(
                    unit_id=unit.unit_id,
                    text=unit.text,
                    unit_type=unit.unit_type,
                    position=len(new_units),
                    role_scores=dict(unit.role_scores),
                    metadata=dict(unit.metadata),
                )
            )
            continue
        if correction.operation == "insert_before" and not inserted:
            for text in correction.new_units:
                new_units.append(
                    ArtifactUnit(
                        unit_id=f"u{len(new_units) + 1}",
                        text=text,
                        unit_type=unit.unit_type,
                        position=len(new_units),
                    )
                )
            inserted = True
        if correction.operation == "replace":
            for text in correction.new_units:
                new_units.append(
                    ArtifactUnit(
                        unit_id=f"u{len(new_units) + 1}",
                        text=text,
                        unit_type=unit.unit_type,
                        position=len(new_units),
                    )
                )
        elif correction.operation == "insert_after":
            new_units.append(
                ArtifactUnit(
                    unit_id=f"u{len(new_units) + 1}",
                    text=unit.text,
                    unit_type=unit.unit_type,
                    position=len(new_units),
                )
            )
            for text in correction.new_units:
                new_units.append(
                    ArtifactUnit(
                        unit_id=f"u{len(new_units) + 1}",
                        text=text,
                        unit_type=unit.unit_type,
                        position=len(new_units),
                    )
                )
        elif correction.operation == "delete":
            continue
        elif correction.operation == "insert_before":
            new_units.append(
                ArtifactUnit(
                    unit_id=f"u{len(new_units) + 1}",
                    text=unit.text,
                    unit_type=unit.unit_type,
                    position=len(new_units),
                )
            )
        elif correction.operation == "reorder":
            continue
    if correction.operation == "reorder":
        kept = [unit for unit in original_units if unit.unit_id not in target_set]
        moved = [unit for unit in original_units if unit.unit_id in target_set]
        combined = kept + moved
        new_units = [
            ArtifactUnit(
                unit_id=f"u{index + 1}",
                text=unit.text,
                unit_type=unit.unit_type,
                position=index,
                role_scores=dict(unit.role_scores),
                metadata=dict(unit.metadata),
            )
            for index, unit in enumerate(combined)
        ]
    if correction.operation == "insert_before" and not inserted:
        for text in reversed(correction.new_units):
            new_units.insert(
                0,
                ArtifactUnit(
                    unit_id=f"u{len(new_units) + 1}",
                    text=text,
                    unit_type="text_span",
                    position=0,
                ),
            )
    new_text = "\n".join(unit.text for unit in new_units).strip()
    metadata = dict(artifact.metadata)
    return canonicalize_candidate(
        candidate_text=new_text,
        provenance=artifact.provenance,
        metadata=metadata,
        dataset_name=str(metadata.get("dataset_name", "")),
        task_type=str(metadata.get("task_type", "")),
        answer_format=str(metadata.get("answer_format", "")),
        task_subtype=str(metadata.get("task_subtype", "")),
        answer_role_threshold=float(metadata.get("answer_role_threshold", 0.58)),
        evidence_role_threshold=float(metadata.get("evidence_role_threshold", 0.52)),
    )


def predict_delta(before: VerifierState, *, correction: Optional[CorrectionArtifact]) -> DeltaPrediction:
    target_drop = {key: clamp01(value * 0.55) for key, value in before.residual_vector.items()}
    if correction is None:
        return DeltaPrediction(
            expected_residual_drop=target_drop,
            expected_confidence_gain=0.0,
            expected_progress_gain=0.0,
            expected_preserve_risk=float(before.preserve_risk),
            rationale="no correction proposed",
        )
    if "delta_contract" in correction.expected_delta:
        target_drop["r_parse"] = clamp01(max(target_drop.get("r_parse", 0.0), float(correction.expected_delta["delta_contract"])))
        target_drop["r_completeness"] = clamp01(max(target_drop.get("r_completeness", 0.0), 0.35))
    if "delta_exec" in correction.expected_delta:
        target_drop["r_execution"] = clamp01(max(target_drop.get("r_execution", 0.0), float(correction.expected_delta["delta_exec"])))
    if "delta_support" in correction.expected_delta:
        target_drop["r_support"] = clamp01(max(target_drop.get("r_support", 0.0), float(correction.expected_delta["delta_support"])))
    if "delta_answer" in correction.expected_delta:
        target_drop["r_consistency"] = clamp01(max(target_drop.get("r_consistency", 0.0), float(correction.expected_delta["delta_answer"]) * 0.6))
    target_drop["r_preserve"] = 0.0
    target_count = float(max(1, len(correction.target_units)))
    preserve_count = float(len(correction.preserve_units))
    preserve_risk = clamp01(float(before.preserve_risk) + 0.08 * (target_count / float(max(1.0, target_count + preserve_count))))
    confidence_gain = clamp01(0.16 + 0.18 * (len(correction.new_units) > 0))
    progress_gain = clamp01(0.14 + 0.18 * (len(correction.target_units) > 0))
    return DeltaPrediction(
        expected_residual_drop=target_drop,
        expected_confidence_gain=confidence_gain,
        expected_progress_gain=progress_gain,
        expected_preserve_risk=preserve_risk,
        rationale=correction.rationale,
    )


def actual_delta(before: VerifierState, after: VerifierState) -> Dict[str, float]:
    return {
        key: clamp01(float(before.residual_vector.get(key, 0.0)) - float(after.residual_vector.get(key, 0.0)))
        for key in before.residual_vector
    }


def mean_expected_drop(prediction: DeltaPrediction) -> float:
    values = [float(value) for key, value in prediction.expected_residual_drop.items() if key != "r_preserve"]
    if not values:
        return 0.0
    return clamp01(sum(values) / float(len(values)))
