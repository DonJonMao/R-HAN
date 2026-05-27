from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from types import SimpleNamespace

from stage2_phase1_semantic_safe_override.artifacts import canonicalize_candidate
from stage2_phase1_semantic_safe_override.config import Phase1SemanticSafeOverrideConfig
from stage2_phase1_semantic_safe_override.runtime import Phase1SemanticSafeOverrideRuntime
from stage2_phase1_semantic_safe_override.verifier import verify_artifact


def _runtime_stub() -> Phase1SemanticSafeOverrideRuntime:
    runtime = object.__new__(Phase1SemanticSafeOverrideRuntime)
    runtime.config = SimpleNamespace(
        stage2_version="phase1_semantic_safe_override_v1",
        candidate_max_k=8,
        similarity_threshold=0.82,
        redundancy_temperature=0.7,
        redundancy_gamma=0.18,
        safe_margin=0.03,
        safe_override_threshold=0.52,
        overturn_threshold=0.58,
        catastrophic_answer_delta_threshold=0.75,
        catastrophic_consistency_threshold=0.48,
    )
    runtime._phase1_last_residual_mean = 0.5
    return runtime


def test_soft_cluster_groups_near_duplicate_candidates():
    runtime = _runtime_stub()
    candidates = [
        {
            "digest": "a",
            "phase1_artifact": canonicalize_candidate(candidate_text="answer 42", task_type="reasoning"),
            "phase1_safe_utility": 0.9,
            "stage1_anchor": False,
        },
        {
            "digest": "b",
            "phase1_artifact": canonicalize_candidate(candidate_text="answer 42", task_type="reasoning"),
            "phase1_safe_utility": 0.8,
            "stage1_anchor": False,
        },
    ]

    classes = runtime._soft_cluster_candidates(candidates)

    assert len(classes) == 1
    assert classes[0]["size"] == 2


def test_select_final_candidate_preserves_anchor_on_high_overturn_risk():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.80,
        "phase1_confidence_score": 0.80,
        "phase1_residual_mean": 0.20,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.90,
    }
    challenger = {
        "digest": "challenger",
        "phase1_safe_utility": 0.88,
        "phase1_safe_override_score": 0.81,
        "phase1_overturn_risk": 0.72,
        "phase1_confidence_score": 0.84,
        "phase1_residual_mean": 0.16,
        "phase1_answer_delta": 0.55,
        "phase1_answer_consistency_score": 0.82,
    }

    selected, reason = runtime._select_final_candidate([challenger], anchor)

    assert selected == anchor
    assert reason == "phase1_semantic_safe_override_preserve_anchor_guard"


def test_select_final_candidate_overrides_when_safe_score_is_good():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.76,
        "phase1_confidence_score": 0.72,
        "phase1_residual_mean": 0.28,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.90,
    }
    challenger = {
        "digest": "challenger",
        "phase1_safe_utility": 0.86,
        "phase1_safe_override_score": 0.78,
        "phase1_overturn_risk": 0.22,
        "phase1_confidence_score": 0.79,
        "phase1_residual_mean": 0.19,
        "phase1_answer_delta": 0.22,
        "phase1_answer_consistency_score": 0.81,
    }

    selected, reason = runtime._select_final_candidate([challenger], anchor)

    assert selected == challenger
    assert reason == "phase1_semantic_safe_override_override_frontier"


def test_select_final_candidate_compares_pairwise_against_anchor_for_all_challengers():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.80,
        "phase1_confidence_score": 0.80,
        "phase1_residual_mean": 0.20,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.92,
    }
    weaker = {
        "digest": "weaker",
        "phase1_safe_utility": 0.79,
        "phase1_safe_override_score": 0.90,
        "phase1_overturn_risk": 0.12,
        "phase1_confidence_score": 0.84,
        "phase1_residual_mean": 0.17,
        "phase1_answer_delta": 0.10,
        "phase1_answer_consistency_score": 0.83,
    }
    stronger = {
        "digest": "stronger",
        "phase1_safe_utility": 0.88,
        "phase1_safe_override_score": 0.79,
        "phase1_overturn_risk": 0.20,
        "phase1_confidence_score": 0.82,
        "phase1_residual_mean": 0.15,
        "phase1_answer_delta": 0.12,
        "phase1_answer_consistency_score": 0.85,
    }

    selected, reason = runtime._select_final_candidate([weaker, stronger], anchor)

    assert selected == stronger
    assert reason == "phase1_semantic_safe_override_override_frontier"


def test_select_final_candidate_blocks_catastrophic_answer_rewrite():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.70,
        "phase1_confidence_score": 0.75,
        "phase1_residual_mean": 0.25,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.90,
    }
    challenger = {
        "digest": "challenger",
        "phase1_safe_utility": 0.92,
        "phase1_safe_override_score": 0.91,
        "phase1_overturn_risk": 0.20,
        "phase1_confidence_score": 0.88,
        "phase1_residual_mean": 0.16,
        "phase1_answer_delta": 0.90,
        "phase1_answer_consistency_score": 0.22,
    }

    selected, reason = runtime._select_final_candidate([challenger], anchor)

    assert selected == anchor
    assert reason == "phase1_semantic_safe_override_preserve_anchor_guard"


def test_select_final_candidate_scans_all_admissible_challengers_not_only_top1():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase1_safe_utility": 0.80,
        "phase1_confidence_score": 0.80,
        "phase1_residual_mean": 0.20,
        "phase1_answer_delta": 0.0,
        "phase1_answer_consistency_score": 0.92,
    }
    blocked_top1 = {
        "digest": "blocked_top1",
        "phase1_safe_utility": 0.93,
        "phase1_safe_override_score": 0.90,
        "phase1_overturn_risk": 0.71,
        "phase1_confidence_score": 0.88,
        "phase1_residual_mean": 0.15,
        "phase1_answer_delta": 0.18,
        "phase1_answer_consistency_score": 0.87,
    }
    admissible_second = {
        "digest": "admissible_second",
        "phase1_safe_utility": 0.87,
        "phase1_safe_override_score": 0.76,
        "phase1_overturn_risk": 0.24,
        "phase1_confidence_score": 0.82,
        "phase1_residual_mean": 0.17,
        "phase1_answer_delta": 0.12,
        "phase1_answer_consistency_score": 0.84,
    }

    selected, reason = runtime._select_final_candidate([blocked_top1, admissible_second], anchor)

    assert selected == admissible_second
    assert reason == "phase1_semantic_safe_override_override_frontier"


def test_utility_features_do_not_encode_stage1_anchor_identity():
    runtime = _runtime_stub()
    artifact = canonicalize_candidate(
        candidate_text="Answer: B",
        metadata={"options": ["alpha", "beta", "gamma", "delta"]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
    )
    state = verify_artifact(
        artifact,
        question_text="1) A\n2) B\n3) C\n4) D\nReturn only the final answer.",
        metadata={"options": ["alpha", "beta", "gamma", "delta"]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
        task_subtype="",
        candidate_entry={"stage1_anchor": True},
    )

    features = runtime._utility_features(
        entry={"stage1_anchor": True, "occurrence_count": 1.0, "sink_support": 0.0, "reviewer_mean_trust": 0.5},
        verifier_state=state,
        artifact=artifact,
        total_turns=1,
        dataset_profile=SimpleNamespace(name="mmlu_pro", task_type="mcq", answer_format="option"),
    )

    assert "stage1_anchor" not in features


def test_phase1_config_exposes_safe_override_fields():
    config = Phase1SemanticSafeOverrideConfig()

    assert config.code_require_entry_point is True
    assert config.max_logged_candidates == 8
    assert config.safe_override_threshold == 0.52
    assert config.overturn_threshold == 0.58
    assert config.replay.max_prompt_chars == 16000


def test_evaluate_candidate_summary_uses_runtime_evaluator_attribute():
    runtime = _runtime_stub()
    runtime.evaluator = SimpleNamespace(
        evaluate_output=lambda *args, **kwargs: SimpleNamespace(
            mean_success=1.0,
            mean_task_score=0.75,
            mean_safety_penalty=0.1,
        )
    )

    summary = runtime._evaluate_candidate_summary(
        question_text="What is 2+2?",
        candidate_text="Answer: 4",
        reference_answer="4",
        metadata={},
        dataset_profile=SimpleNamespace(name="gsm8k", task_type="reasoning", answer_format="numeric"),
    )

    assert summary == {
        "success": 1.0,
        "task_score": 0.75,
        "safety_penalty": 0.1,
    }
