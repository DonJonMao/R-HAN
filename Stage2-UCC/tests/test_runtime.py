from __future__ import annotations

from types import SimpleNamespace

from stage2_phase3a_unified.config import Phase3aUnifiedConfig
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


def test_phase3a_config_exposes_runtime_compatibility_fields():
    config = Phase3aUnifiedConfig()

    assert config.code_require_entry_point is True
    assert config.max_logged_candidates == 8
    assert config.explicit_challenger_agents == ("skeptic", "debater_b")
    assert config.override_inspector_agent_id == "verifier"
    assert config.replay.max_prompt_chars == 16000


def test_ensure_phase3a_entry_fields_adds_repair_metadata_defaults():
    runtime = _runtime_stub()
    entry = {}

    runtime._ensure_phase3a_entry_fields(entry)

    assert entry["repair_branch"] is False
    assert entry["repair_agent_id"] == ""
    assert entry["repair_round"] == -1
    assert entry["repair_parent_digest"] == ""
    assert entry["code_repair_level"] == 0
    assert entry["code_repair_passed"] == 0
    assert entry["code_repair_total"] == 0
    assert entry["code_repair_failure_kind"] == ""
    assert entry["code_repair_failing_examples"] == []


def test_prompt_json_returns_none_when_prompt_too_long(monkeypatch):
    runtime = object.__new__(Phase3aUnifiedRuntime)
    runtime.config = SimpleNamespace(replay=SimpleNamespace(max_prompt_chars=16))
    runtime._by_id = {"critic": SimpleNamespace(system_prompt="system")}
    runtime._json_slots = lambda: None
    runtime.evaluator = SimpleNamespace(
        _cached_chat=lambda *args, **kwargs: '{"ok": true}',
        _resolve_runtime=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("stage2_phase3a_unified.runtime.build_system_prompt", lambda *args, **kwargs: "system")

    result = runtime._prompt_json(
        agent_id="critic",
        instruction="instruction",
        question_text="question text",
        metadata=None,
        payload={"artifact": {"text": "x" * 100}},
        dataset_profile=SimpleNamespace(),
        extra_role_hint="phase3a_critic",
    )

    assert result is None


def test_prompt_json_returns_none_on_chat_runtime_error(monkeypatch):
    runtime = object.__new__(Phase3aUnifiedRuntime)
    runtime.config = SimpleNamespace(replay=SimpleNamespace(max_prompt_chars=100000))
    runtime._by_id = {"critic": SimpleNamespace(system_prompt="system")}
    runtime._json_slots = lambda: None
    runtime.evaluator = SimpleNamespace(
        _cached_chat=lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("HTTP Error 400: Bad Request")),
        _resolve_runtime=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr("stage2_phase3a_unified.runtime.build_system_prompt", lambda *args, **kwargs: "system")

    result = runtime._prompt_json(
        agent_id="critic",
        instruction="instruction",
        question_text="short question",
        metadata=None,
        payload={"artifact": {"text": "short"}},
        dataset_profile=SimpleNamespace(),
        extra_role_hint="phase3a_critic",
    )

    assert result is None
