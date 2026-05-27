from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stage2_phase1_semantic_safe_override.artifacts import canonicalize_candidate
from stage2_phase1_semantic_safe_override.verifier import verify_artifact
from stage2_phase2_unified_local_correction.correction import apply_correction_artifact, critique_summary, preserve_heatmap, predict_delta, propose_correction
from stage2_phase2_unified_local_correction.runtime import Phase2UnifiedLocalCorrectionRuntime


def _runtime_stub() -> Phase2UnifiedLocalCorrectionRuntime:
    runtime = object.__new__(Phase2UnifiedLocalCorrectionRuntime)
    runtime.config = SimpleNamespace(
        stage2_version="phase2_unified_local_correction_v1",
        candidate_max_k=8,
        similarity_threshold=0.82,
        redundancy_temperature=0.7,
        redundancy_gamma=0.18,
        safe_margin=0.03,
        safe_override_threshold=0.52,
        overturn_threshold=0.58,
        catastrophic_answer_delta_threshold=0.75,
        catastrophic_consistency_threshold=0.48,
        correction_risk_threshold=0.64,
        correction_value_weight=0.22,
    )
    runtime._phase1_last_residual_mean = 0.5
    return runtime


def test_propose_correction_re_renders_recoverable_graph_answer_into_contract_json():
    artifact = canonicalize_candidate(
        candidate_text="The max flow is 31.",
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="flow",
    )
    state = verify_artifact(
        artifact,
        question_text="Edges: 0->1 (capacity=16), 0->2 (capacity=13), 1->2 (capacity=10), 1->3 (capacity=12), 2->1 (capacity=4), 2->4 (capacity=14), 3->2 (capacity=9), 3->5 (capacity=20), 4->3 (capacity=7), 4->5 (capacity=4). What is the max flow from 0 to 5?",
        metadata={"task": "flow"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="flow",
        candidate_entry={},
    )
    preserve_map = preserve_heatmap(artifact, state)
    critique = critique_summary(
        artifact,
        state,
        localization_map={unit.unit_id: 0.9 for unit in artifact.units},
        preserve_map=preserve_map,
        answer_first_score=0.8,
        top_k=1,
    )

    correction = propose_correction(artifact, state, critique=critique)

    assert correction is not None
    edited = apply_correction_artifact(artifact, correction)
    assert edited.answer_object.fields.get("contract_valid") is True
    assert "\"max_flow\": 31" in edited.rendered_answer


def test_phase2_selection_prefers_low_risk_correction_candidate():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.80,
        "phase2_final_utility": 0.80,
        "phase1_confidence_score": 0.80,
        "phase1_residual_mean": 0.20,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.90,
    }
    corrected = {
        "digest": "corrected",
        "phase1_safe_utility": 0.84,
        "phase2_final_utility": 0.91,
        "phase2_correction_value": 0.32,
        "phase2_correction_risk": 0.22,
        "phase2_correction_artifact": {"operation": "replace"},
        "phase1_overturn_risk": 0.24,
        "phase1_safe_override_score": 0.76,
        "phase1_confidence_score": 0.82,
        "phase1_residual_mean": 0.16,
        "phase1_answer_delta": 0.14,
        "phase1_answer_consistency_score": 0.84,
    }

    selected, reason = runtime._select_final_candidate([corrected], anchor)

    assert selected == corrected
    assert reason == "phase2_unified_local_correction_override_with_correction"


def test_phase2_selection_blocks_high_correction_risk():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.80,
        "phase2_final_utility": 0.80,
        "phase1_confidence_score": 0.80,
        "phase1_residual_mean": 0.20,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.90,
    }
    corrected = {
        "digest": "corrected",
        "phase1_safe_utility": 0.84,
        "phase2_final_utility": 0.91,
        "phase2_correction_value": 0.32,
        "phase2_correction_risk": 0.86,
        "phase2_correction_artifact": {"operation": "replace"},
        "phase1_overturn_risk": 0.24,
        "phase1_safe_override_score": 0.76,
        "phase1_confidence_score": 0.82,
        "phase1_residual_mean": 0.16,
        "phase1_answer_delta": 0.14,
        "phase1_answer_consistency_score": 0.84,
    }

    selected, reason = runtime._select_final_candidate([corrected], anchor)

    assert selected == anchor
    assert reason == "phase2_unified_local_correction_preserve_anchor_guard"


def test_predict_delta_exposes_progress_gain():
    artifact = canonicalize_candidate(
        candidate_text="Answer: B",
        metadata={"options": ["A", "B", "C", "D"]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
    )
    state = verify_artifact(
        artifact,
        question_text="1) A\n2) B\n3) C\n4) D\nReturn only the final answer.",
        metadata={"options": ["A", "B", "C", "D"]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
        task_subtype="",
        candidate_entry={},
    )
    delta = predict_delta(
        state,
        correction=SimpleNamespace(
            target_units=["u1"],
            new_units=["OPTION - 2"],
            preserve_units=[],
            expected_delta={"delta_answer": 0.2},
            rationale="normalize answer surface",
        ),
    )

    assert delta.expected_progress_gain > 0.0
