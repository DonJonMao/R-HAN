from __future__ import annotations

from types import SimpleNamespace

from mas_stage2.types import ExportedMemoryMessage
from mas_stage2_v4_2.runtime import Stage2RuntimeV42


def _message(node_id: str, vector: list[float], confidence: float) -> ExportedMemoryMessage:
    return ExportedMemoryMessage(
        node_id=node_id,
        turn_index=1,
        summary=node_id,
        latent_vector=vector,
        provenance_record_ids=[node_id],
        confidence=confidence,
    )


class _FakeEmbedder:
    def embed(self, text: str):
        text = text.strip()
        if text == "42":
            return [1.0, 0.0]
        if text == "43":
            return [0.0, 1.0]
        return [0.5, 0.5]


def _runtime_stub() -> Stage2RuntimeV42:
    runtime = object.__new__(Stage2RuntimeV42)
    runtime.config = SimpleNamespace(
        max_logged_candidates=4,
        calibration_rounds=3,
        auditor_agent_ids=("verifier", "skeptic"),
        adjudicator_agent_id="verifier",
        calibration_agent_id="verifier",
    )
    runtime.embedder = _FakeEmbedder()
    runtime._last_candidate_bundle = {
        "question_text": "What is 40 + 2?",
        "metadata": {"id": "smoke-v4-2"},
    }
    return runtime


def _comparison(*, digest: str, explicit: bool, probability: float = 0.9) -> dict:
    candidate = {
        "digest": digest,
        "text": "43" if digest == "challenger-digest" else "41",
        "explicit_challenger": explicit,
        "candidate_model_score": 0.85,
        "candidate_model_uncertainty": 0.15,
        "support_score": 0.85,
        "sink_support": 1,
        "occurrence_count": 1,
        "source_node_ids": {"solver"},
        "source_roles": {"solver"},
        "turn_indices": {1},
        "feedback_pass_calibrated": 0.7,
        "feedback_challenge_calibrated": 0.1,
        "feedback_uncertain_calibrated": 0.1,
        "reviewer_event_count": 1.0,
        "reviewer_mean_trust": 0.7,
        "parse_ok": True,
        "entry_point_ok": True,
    }
    return {
        "candidate": candidate,
        "digest": str(candidate["digest"]),
        "source": "explicit_challenger" if explicit else "sink_consensus",
        "pairwise_probability": probability,
        "pairwise_uncertainty": 0.1,
        "quality_score": 0.9,
        "review_consensus": 0.8,
        "sink_ratio": 0.7,
        "source_diversity": 1.2,
        "is_valid_code": True,
    }


def test_dar_retention_keeps_consensus_and_diversity():
    runtime = _runtime_stub()
    messages = [
        _message("consensus", [1.0, 0.0], 0.90),
        _message("nearby", [0.95, 0.05], 0.80),
        _message("diverse", [-1.0, 0.0], 0.70),
    ]

    selected = runtime._dar_retain_messages(messages, max_items=2)
    selected_ids = {item.node_id for item in selected}

    assert len(selected) == 2
    assert selected_ids.intersection({"consensus", "nearby"})
    assert "diverse" in selected_ids


def test_soft_promotion_can_create_provisional_challenger():
    runtime = _runtime_stub()
    runtime._audit_candidate_for_promotion = lambda **kwargs: {  # type: ignore[attr-defined]
        "promote": True,
        "support_count": 1,
        "support_agents": {"verifier"},
        "local_consistency": "consistent",
        "findings": [{"agent_id": "verifier", "support": "anomaly", "consistency": "consistent"}],
        "promotion_rationale": "candidate fixes the arithmetic step",
        "promotion_evidence": "different final computation",
    }

    anchor = {
        "digest": "anchor-digest",
        "text": "42",
        "stage1_anchor": True,
        "candidate_model_score": 0.9,
        "support_score": 0.9,
        "candidate_model_uncertainty": 0.1,
        "sink_support": 1,
        "occurrence_count": 1,
        "source_node_ids": {"anchor"},
        "source_roles": {"solver"},
        "turn_indices": {1},
        "feedback_pass_calibrated": 0.8,
        "feedback_challenge_calibrated": 0.0,
        "feedback_uncertain_calibrated": 0.0,
        "reviewer_event_count": 1.0,
        "reviewer_mean_trust": 0.8,
    }
    candidate = {
        "digest": "challenger-digest",
        "text": "43",
        "explicit_challenger": False,
        "candidate_model_score": 0.8,
        "support_score": 0.8,
        "candidate_model_uncertainty": 0.2,
        "sink_support": 1,
        "occurrence_count": 1,
        "source_node_ids": {"solver"},
        "source_roles": {"solver"},
        "turn_indices": {1},
        "feedback_pass_calibrated": 0.6,
        "feedback_challenge_calibrated": 0.1,
        "feedback_uncertain_calibrated": 0.1,
        "reviewer_event_count": 1.0,
        "reviewer_mean_trust": 0.7,
        "parse_ok": True,
        "entry_point_ok": True,
    }
    comparisons = [_comparison(digest="challenger-digest", explicit=False, probability=0.72)]

    provisional = runtime._promote_provisional_challengers(
        question_text="What is 40 + 2?",
        metadata={"id": "soft-promotion"},
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
        anchor=anchor,
        candidates=[candidate, anchor],
        comparisons=comparisons,
    )

    assert len(provisional) == 1
    assert provisional[0]["digest"] == "challenger-digest"
    assert provisional[0]["provisional_challenger"] is True
    assert provisional[0]["promotion_source"] == "auditor_soft_promotion"


def test_preserve_without_provisional_challenger():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(digest="challenger-digest", explicit=False)]  # type: ignore[attr-defined]
    runtime._promote_provisional_challengers = lambda **kwargs: []  # type: ignore[attr-defined]

    anchor = {"digest": "anchor-digest", "text": "42", "stage1_anchor": True}
    challenger = {"digest": "challenger-digest", "text": "43", "explicit_challenger": False}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[challenger, anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected == anchor
    assert reason == "v4_2_preserve_no_provisional_challenger"
    assert extra["v4_2_provisional_challenger_count"] == 0


def test_override_requires_calibrated_majority():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(digest="challenger-digest", explicit=True)]  # type: ignore[attr-defined]
    runtime._promote_provisional_challengers = lambda **kwargs: [  # type: ignore[attr-defined]
        {"digest": "challenger-digest", "text": "43", "explicit_challenger": True, "provisional_challenger": True}
    ]
    runtime._adjudicate_hypotheses = lambda **kwargs: (  # type: ignore[attr-defined]
        {"digest": "challenger-digest", "text": "43", "explicit_challenger": True, "provisional_challenger": True},
        {"winner": "H1", "winner_digest": "challenger-digest", "decision": "challenger", "pressure": "", "rationale": "", "hypothesis_count": 2},
    )
    runtime._calibrate_challenger_against_anchor = lambda **kwargs: {  # type: ignore[attr-defined]
        "rounds": 3,
        "challenger_votes": 2,
        "anchor_votes": 1,
        "uncertain_votes": 0,
        "probability": 2.0 / 3.0,
        "override": True,
        "traces": [],
    }

    anchor = {"digest": "anchor-digest", "text": "42", "stage1_anchor": True}
    challenger = {"digest": "challenger-digest", "text": "43", "explicit_challenger": True}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[challenger, anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected["digest"] == "challenger-digest"
    assert reason == "v4_2_override_calibrated_challenger"
    assert extra["v4_2_calibration_override"] is True


def test_preserve_when_calibration_rejects_override():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(digest="challenger-digest", explicit=True)]  # type: ignore[attr-defined]
    runtime._promote_provisional_challengers = lambda **kwargs: [  # type: ignore[attr-defined]
        {"digest": "challenger-digest", "text": "43", "explicit_challenger": True, "provisional_challenger": True}
    ]
    runtime._adjudicate_hypotheses = lambda **kwargs: (  # type: ignore[attr-defined]
        {"digest": "challenger-digest", "text": "43", "explicit_challenger": True, "provisional_challenger": True},
        {"winner": "H1", "winner_digest": "challenger-digest", "decision": "challenger", "pressure": "", "rationale": "", "hypothesis_count": 2},
    )
    runtime._calibrate_challenger_against_anchor = lambda **kwargs: {  # type: ignore[attr-defined]
        "rounds": 3,
        "challenger_votes": 1,
        "anchor_votes": 2,
        "uncertain_votes": 0,
        "probability": 1.0 / 3.0,
        "override": False,
        "traces": [],
    }

    anchor = {"digest": "anchor-digest", "text": "42", "stage1_anchor": True}
    challenger = {"digest": "challenger-digest", "text": "43", "explicit_challenger": True}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[challenger, anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected == anchor
    assert reason == "v4_2_preserve_calibration"
    assert extra["v4_2_calibration_override"] is False
