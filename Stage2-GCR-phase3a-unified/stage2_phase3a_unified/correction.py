from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .artifacts import ArtifactIR, ArtifactUnit, canonicalize_candidate, clamp01
from .verifier import VerifierState, mean_residual


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
    expected_preserve_risk: float
    expected_frontier_shift: float
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def preserve_heatmap(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    anchor_artifact: Optional[ArtifactIR] = None,
) -> Dict[str, float]:
    anchor_texts = {unit.text.strip() for unit in (anchor_artifact.units if anchor_artifact is not None else [])}
    mask: Dict[str, float] = {}
    for unit in artifact.units:
        support = float(verifier_state.support_map.get(unit.unit_id, 0.0))
        anchor_agreement = 1.0 if unit.text.strip() in anchor_texts and unit.text.strip() else 0.0
        score = clamp01(0.55 * support + 0.35 * anchor_agreement + 0.10 * (1.0 if unit.position < 2 else 0.0))
        mask[unit.unit_id] = score
    return mask


def localize_units(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    preserve_map: Dict[str, float],
    previous_delta: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    previous_delta = dict(previous_delta or {})
    heatmap: Dict[str, float] = {}
    for unit in artifact.units:
        unit_error = float(verifier_state.unit_error_map.get(unit.unit_id, 0.0))
        support_penalty = 1.0 - float(verifier_state.support_map.get(unit.unit_id, 0.0))
        preserve_penalty = float(preserve_map.get(unit.unit_id, 0.0))
        recent_effect = float(previous_delta.get(unit.unit_id, 0.0))
        score = clamp01(0.60 * unit_error + 0.25 * support_penalty + 0.15 * recent_effect - 0.30 * preserve_penalty)
        heatmap[unit.unit_id] = score
    return heatmap


def deterministic_critique(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    localization_map: Dict[str, float],
    preserve_map: Dict[str, float],
    top_k: int,
) -> Dict[str, Any]:
    ranked_targets = sorted(localization_map.items(), key=lambda item: (item[1], item[0]), reverse=True)
    target_units = [unit_id for unit_id, score in ranked_targets[:top_k] if score > 0.0]
    preserve_units = [unit_id for unit_id, score in preserve_map.items() if score >= 0.65]
    expected_drop = [
        residual_name
        for residual_name, residual_value in verifier_state.residual_vector.items()
        if residual_value >= 0.35 and residual_name != "r_preserve"
    ]
    if not expected_drop:
        expected_drop = ["r_support"]
    return {
        "target_units": target_units,
        "preserve_units": preserve_units,
        "main_residual_causes": list(verifier_state.issues[:3]) or ["low confidence region"],
        "expected_delta": {
            "should_drop": expected_drop,
            "must_not_increase": ["r_preserve"],
        },
        "edit_rationale": verifier_state.meta_summary,
    }


def _suggest_typing_import(error_text: str) -> List[str]:
    missing = []
    for symbol in ("List", "Tuple", "Dict", "Set", "Optional"):
        if re.search(rf"name '{symbol}' is not defined", error_text):
            missing.append(symbol)
    if not missing:
        return []
    return [f"from typing import {', '.join(sorted(set(missing)))}"]


def deterministic_proposal(
    artifact: ArtifactIR,
    verifier_state: VerifierState,
    *,
    critique: Dict[str, Any],
    anchor_artifact: Optional[ArtifactIR] = None,
    task_type: str = "",
) -> Optional[CorrectionArtifact]:
    target_units = [str(unit_id) for unit_id in critique.get("target_units", [])]
    preserve_units = [str(unit_id) for unit_id in critique.get("preserve_units", [])]
    if not target_units:
        return None

    exec_feedback = verifier_state.executor_feedback
    if task_type == "code_generation":
        error_text = " ".join(
            str(value)
            for value in (
                exec_feedback.get("exec_error", ""),
                exec_feedback.get("stderr", ""),
                " ".join(exec_feedback.get("failing_examples", [])),
            )
        )
        typing_import = _suggest_typing_import(error_text)
        if typing_import:
            return CorrectionArtifact(
                target_units=[artifact.units[0].unit_id] if artifact.units else target_units,
                operation="insert_before",
                new_units=typing_import,
                preserve_units=preserve_units,
                expected_delta=critique.get("expected_delta", {}),
                rationale="restore missing typing symbols required by execution",
            )

    if anchor_artifact is not None:
        anchor_by_position = {unit.position: unit.text for unit in anchor_artifact.units}
        replacement_units = []
        for unit in artifact.units:
            if unit.unit_id not in target_units:
                continue
            anchor_text = anchor_by_position.get(unit.position)
            if anchor_text and anchor_text != unit.text:
                replacement_units.append(anchor_text)
        if replacement_units:
            return CorrectionArtifact(
                target_units=target_units,
                operation="replace",
                new_units=replacement_units,
                preserve_units=preserve_units,
                expected_delta=critique.get("expected_delta", {}),
                rationale="restore a higher-support anchor fragment in the highest-residual region",
            )

    unique_targets = {unit_id for unit_id in target_units}
    if len(unique_targets) == 1:
        return CorrectionArtifact(
            target_units=target_units,
            operation="delete",
            new_units=[],
            preserve_units=preserve_units,
            expected_delta=critique.get("expected_delta", {}),
            rationale="remove a low-support unit when no safe replacement is available",
        )
    return None


def apply_correction_artifact(
    artifact: ArtifactIR,
    correction: CorrectionArtifact,
) -> ArtifactIR:
    if correction.operation not in ALLOWED_OPERATIONS:
        raise ValueError(f"unsupported correction operation: {correction.operation}")
    target_set = set(correction.target_units)
    original_units = list(artifact.units)
    new_units: List[ArtifactUnit] = []
    inserted = False
    for unit in original_units:
        if unit.unit_id not in target_set:
            new_units.append(ArtifactUnit(unit_id=unit.unit_id, text=unit.text, unit_type=unit.unit_type, position=len(new_units)))
            continue
        if correction.operation == "insert_before" and not inserted:
            for text in correction.new_units:
                new_units.append(ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=text, unit_type=unit.unit_type, position=len(new_units)))
            inserted = True
        if correction.operation == "replace":
            for text in correction.new_units:
                new_units.append(ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=text, unit_type=unit.unit_type, position=len(new_units)))
        elif correction.operation == "insert_after":
            new_units.append(ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=unit.text, unit_type=unit.unit_type, position=len(new_units)))
            for text in correction.new_units:
                new_units.append(ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=text, unit_type=unit.unit_type, position=len(new_units)))
        elif correction.operation == "reorder":
            continue
        elif correction.operation == "delete":
            continue
        elif correction.operation == "insert_before":
            new_units.append(ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=unit.text, unit_type=unit.unit_type, position=len(new_units)))
    if correction.operation == "reorder":
        kept = [unit for unit in original_units if unit.unit_id not in target_set]
        moved = [unit for unit in original_units if unit.unit_id in target_set]
        combined = kept + moved
        new_units = [
            ArtifactUnit(unit_id=f"u{index + 1}", text=unit.text, unit_type=unit.unit_type, position=index)
            for index, unit in enumerate(combined)
        ]
    if correction.operation == "insert_before" and not inserted:
        for text in correction.new_units:
            new_units.insert(
                0,
                ArtifactUnit(unit_id=f"u{len(new_units) + 1}", text=text, unit_type="text_span", position=0),
            )
    new_text = "\n".join(unit.text for unit in new_units).strip()
    return canonicalize_candidate(
        candidate_text=new_text,
        provenance=artifact.provenance,
        metadata=artifact.metadata,
        task_type=str(artifact.metadata.get("task_type", "")),
    )


def predict_delta(
    before: VerifierState,
    *,
    correction: Optional[CorrectionArtifact],
) -> DeltaPrediction:
    target_drop = {
        key: clamp01(value * 0.60)
        for key, value in before.residual_vector.items()
    }
    if correction is None:
        return DeltaPrediction(
            expected_residual_drop=target_drop,
            expected_confidence_gain=0.0,
            expected_preserve_risk=before.residual_vector.get("r_preserve", 0.0),
            expected_frontier_shift=0.0,
            rationale="no correction proposed",
        )
    protected_count = float(len(correction.preserve_units))
    target_count = float(max(1, len(correction.target_units)))
    preserve_risk = clamp01(before.residual_vector.get("r_preserve", 0.0) + 0.08 * (target_count / (protected_count + target_count)))
    confidence_gain = clamp01(0.25 + 0.20 * (len(correction.new_units) > 0))
    frontier_shift = clamp01(0.20 + 0.10 * (len(correction.target_units) > 0))
    if "r_preserve" in target_drop:
        target_drop["r_preserve"] = 0.0
    return DeltaPrediction(
        expected_residual_drop=target_drop,
        expected_confidence_gain=confidence_gain,
        expected_preserve_risk=preserve_risk,
        expected_frontier_shift=frontier_shift,
        rationale=correction.rationale,
    )


def actual_delta(before: VerifierState, after: VerifierState) -> Dict[str, float]:
    return {
        key: clamp01(float(before.residual_vector.get(key, 0.0)) - float(after.residual_vector.get(key, 0.0)))
        for key in before.residual_vector
    }


def delta_match_score(prediction: DeltaPrediction, before: VerifierState, after: VerifierState) -> float:
    actual = actual_delta(before, after)
    errors = []
    for key, expected in prediction.expected_residual_drop.items():
        observed = float(actual.get(key, 0.0))
        errors.append(abs(float(expected) - observed))
    return clamp01(1.0 - (sum(errors) / float(max(1, len(errors)))))

