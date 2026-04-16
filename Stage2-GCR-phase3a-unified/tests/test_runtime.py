from __future__ import annotations

from types import SimpleNamespace

from stage2_phase3a_unified.runtime import Phase3aUnifiedRuntime


def _runtime_stub() -> Phase3aUnifiedRuntime:
    runtime = object.__new__(Phase3aUnifiedRuntime)
    runtime.config = SimpleNamespace(
        stage2_version="phase3a_unified_v1",
        candidate_max_k=8,
        similarity_threshold=0.82,
        redundancy_temperature=0.7,
        redundancy_gamma=0.18,
        preserve_threshold=0.72,
        guard_margin=0.04,
    )
    runtime._phase3a_last_residual_mean = 0.5
    return runtime


def test_soft_cluster_groups_near_duplicate_candidates():
    runtime = _runtime_stub()
    candidates = [
        {
            "digest": "a",
            "phase3a_artifact": SimpleNamespace(),
            "phase3a_adjusted_utility": 0.9,
            "stage1_anchor": False,
        },
        {
            "digest": "b",
            "phase3a_artifact": SimpleNamespace(),
            "phase3a_adjusted_utility": 0.8,
            "stage1_anchor": False,
        },
    ]
    candidates[0]["phase3a_artifact"] = SimpleNamespace(units=[SimpleNamespace(text="answer 42")])
    candidates[1]["phase3a_artifact"] = SimpleNamespace(units=[SimpleNamespace(text="answer 42")])

    classes = runtime._soft_cluster_candidates(candidates)

    assert len(classes) == 1
    assert classes[0]["size"] == 2


def test_select_final_candidate_preserves_anchor_without_margin():
    runtime = _runtime_stub()
    anchor = {
        "digest": "anchor",
        "phase3a_adjusted_utility": 0.80,
        "phase3a_confidence_score": 0.80,
        "phase3a_residual_mean": 0.20,
        "phase3a_verifier_state": SimpleNamespace(residual_vector={"r_preserve": 0.10}),
    }
    challenger = {
        "digest": "challenger",
        "phase3a_adjusted_utility": 0.81,
        "phase3a_confidence_score": 0.79,
        "phase3a_residual_mean": 0.19,
        "phase3a_verifier_state": SimpleNamespace(residual_vector={"r_preserve": 0.90}),
    }

    selected, reason = runtime._select_final_candidate([challenger], anchor)

    assert selected == anchor
    assert reason == "phase3a_unified_preserve_anchor_guard"
