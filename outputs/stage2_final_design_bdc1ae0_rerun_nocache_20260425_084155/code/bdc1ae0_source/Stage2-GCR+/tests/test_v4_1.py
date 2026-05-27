from __future__ import annotations

from types import SimpleNamespace

from mas_stage2.types import ControllerState, ExportedMemoryMessage, FeedbackEvent, NodeTurnTrace, TurnTrace
from stage2_gcr_plus.runtime_v41 import Stage2RuntimeV41


def _message(node_id: str, vector: list[float], confidence: float) -> ExportedMemoryMessage:
    return ExportedMemoryMessage(
        node_id=node_id,
        turn_index=1,
        summary=node_id,
        latent_vector=vector,
        provenance_record_ids=[node_id],
        confidence=confidence,
    )


def _runtime_stub() -> Stage2RuntimeV41:
    runtime = object.__new__(Stage2RuntimeV41)
    runtime.config = SimpleNamespace(max_logged_candidates=4)
    runtime.evaluator = SimpleNamespace(
        _sanitize_final_output=lambda question_text, raw_text, reference_answer=None, metadata=None: raw_text.strip()
    )
    runtime._last_candidate_bundle = {
        "question_text": "What is 40 + 2?",
        "metadata": {"id": "smoke-v4-1"},
    }
    runtime._score_bank = lambda bank, **kwargs: {  # type: ignore[attr-defined]
        "candidates": list(bank.values()),
        "candidates_serialized": [dict(item) for item in bank.values()],
        "anchor": next((dict(item) for item in bank.values() if item.get("stage1_anchor")), None),
        "anchor_serialized": next((dict(item) for item in bank.values() if item.get("stage1_anchor")), None),
    }
    runtime._generate_explicit_challengers = lambda **kwargs: []  # type: ignore[attr-defined]
    runtime._aggregate_occurrence_feedback = lambda events, **kwargs: {  # type: ignore[attr-defined]
        "pass_count": 0.0,
        "challenge_count": 0.0,
        "uncertain_count": 0.0,
        "pass_weight": 0.0,
        "challenge_weight": 0.0,
        "uncertain_weight": 0.0,
        "event_count": float(len(events)),
        "trust_sum": 0.0,
    }
    return runtime


def _comparison(*, explicit: bool, probability: float = 0.9) -> dict:
    candidate = {
        "digest": "challenger-digest",
        "text": "43" if explicit else "41",
        "explicit_challenger": explicit,
        "candidate_model_uncertainty": 0.15,
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


def test_override_requires_explicit_challenger():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(explicit=False)]  # type: ignore[attr-defined]

    anchor = {"digest": "anchor-digest", "text": "42"}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected == anchor
    assert reason == "v4_1_preserve_no_explicit_challenger"
    assert extra["v4_1_explicit_challenger_count"] == 0


def test_override_requires_inspector_agreement():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(explicit=True)]  # type: ignore[attr-defined]
    runtime._inspector_compare = lambda **kwargs: ("anchor", "anchor remains more reliable")  # type: ignore[attr-defined]

    anchor = {"digest": "anchor-digest", "text": "42"}
    challenger = {"digest": "challenger-digest", "text": "43", "explicit_challenger": True}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[challenger, anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected == anchor
    assert reason == "v4_1_preserve_inspector"
    assert extra["v4_1_override_eligible"] is False
    assert extra["v4_1_inspector_decision"] == "anchor"


def test_override_succeeds_with_explicit_challenger_and_inspector():
    runtime = _runtime_stub()
    runtime._pairwise_candidate_comparisons = lambda candidates, anchor, dataset_profile: [_comparison(explicit=True)]  # type: ignore[attr-defined]
    runtime._inspector_compare = lambda **kwargs: ("challenger", "challenger fixes the anchor flaw")  # type: ignore[attr-defined]

    anchor = {"digest": "anchor-digest", "text": "42"}
    challenger = {"digest": "challenger-digest", "text": "43", "explicit_challenger": True}
    selected, reason, extra = runtime._select_against_anchor(
        candidates=[challenger, anchor],
        anchor=anchor,
        dataset_profile=SimpleNamespace(task_type="math_reasoning"),
    )

    assert selected["digest"] == "challenger-digest"
    assert selected["explicit_challenger"] is True
    assert reason == "v4_1_override_explicit_challenger"
    assert extra["v4_1_override_eligible"] is True
    assert extra["v4_1_inspector_decision"] == "challenger"


def test_candidate_bank_admission_accepts_checker_positive_aggregator_without_veto():
    runtime = _runtime_stub()

    admitted, source = runtime._candidate_bank_admission(
        role="aggregator",
        is_sink=False,
        checker_snapshot={"positive_without_veto": True},
        is_recovery_output=False,
    )

    assert admitted is True
    assert source == "checker_positive_aggregator"


def test_candidate_bank_admission_rejects_vetoed_aggregator_even_with_positive_signal():
    runtime = _runtime_stub()

    admitted, source = runtime._candidate_bank_admission(
        role="aggregator",
        is_sink=False,
        checker_snapshot={"positive_without_veto": False, "positive": True, "hard_veto": True},
        is_recovery_output=False,
    )

    assert admitted is False
    assert source == "filtered"


def test_candidate_bank_only_admits_sink_and_checker_positive_entries():
    runtime = _runtime_stub()
    controller_state = ControllerState(
        turn_index=0,
        mode="lean",
        focus="",
        uncertainty=0.1,
        role_weights={},
        summary="state",
    )
    turn_trace = TurnTrace(
        turn_index=0,
        controller_state=controller_state,
        active_edges=[],
        node_traces=[
            NodeTurnTrace(
                node_id="solver_pass",
                agent_id="solver",
                role="solver",
                active_incoming_edge_ids=[],
                selected_records=[],
                neighbour_sources=[],
                memory_brief="",
                output="proposal-pass",
                local_latent_summary="",
                exported_summary="",
                prompt_excerpt="",
            ),
            NodeTurnTrace(
                node_id="solver_fail",
                agent_id="solver",
                role="solver",
                active_incoming_edge_ids=[],
                selected_records=[],
                neighbour_sources=[],
                memory_brief="",
                output="proposal-fail",
                local_latent_summary="",
                exported_summary="",
                prompt_excerpt="",
            ),
            NodeTurnTrace(
                node_id="agg_conflict",
                agent_id="agg",
                role="aggregator",
                active_incoming_edge_ids=[],
                selected_records=[],
                neighbour_sources=[],
                memory_brief="",
                output="agg-conflict",
                local_latent_summary="",
                exported_summary="",
                prompt_excerpt="",
            ),
            NodeTurnTrace(
                node_id="agg_pass",
                agent_id="agg",
                role="aggregator",
                active_incoming_edge_ids=[],
                selected_records=[],
                neighbour_sources=[],
                memory_brief="",
                output="agg-pass",
                local_latent_summary="",
                exported_summary="",
                prompt_excerpt="",
            ),
            NodeTurnTrace(
                node_id="sink",
                agent_id="sink",
                role="aggregator",
                active_incoming_edge_ids=[],
                selected_records=[],
                neighbour_sources=[],
                memory_brief="",
                output="final-answer",
                local_latent_summary="",
                exported_summary="",
                prompt_excerpt="",
            ),
        ],
        feedback_events=[
            FeedbackEvent(
                event_id="evt-pass",
                turn_index=0,
                source_node_id="verifier",
                target_node_id="solver_pass",
                source_kind="verifier",
                event_type="pass",
                confidence=0.9,
                detail="looks correct",
            ),
            FeedbackEvent(
                event_id="evt-challenge",
                turn_index=0,
                source_node_id="verifier",
                target_node_id="solver_fail",
                source_kind="verifier",
                event_type="challenge",
                confidence=0.8,
                detail="needs work",
            ),
            FeedbackEvent(
                event_id="evt-conflict",
                turn_index=0,
                source_node_id="verifier",
                target_node_id="agg_conflict",
                source_kind="verifier",
                event_type="conflict",
                confidence=0.9,
                detail="conflicting merge",
            ),
            FeedbackEvent(
                event_id="evt-agg-pass",
                turn_index=0,
                source_node_id="verifier",
                target_node_id="agg_pass",
                source_kind="verifier",
                event_type="pass",
                confidence=0.9,
                detail="supported merge",
            ),
        ],
        sink_outputs={"sink": "final-answer"},
    )

    bundle = runtime._candidate_bank_bundle(
        question_text="What is 40 + 2?",
        turn_traces=[turn_trace],
        metadata={"stage1_anchor_output": ""},
        dataset_profile=SimpleNamespace(task_type="numeric"),
    )

    candidate_texts = {item["text"] for item in bundle["candidates"]}
    occurrence_map = bundle["occurrences"]

    assert candidate_texts == {"proposal-pass", "agg-pass", "final-answer"}
    assert occurrence_map[(0, "solver_pass")]["candidate_bank_source"] == "checker_approved_proposal"
    assert occurrence_map[(0, "solver_fail")]["admitted_to_candidate_bank"] is False
    assert occurrence_map[(0, "agg_conflict")]["admitted_to_candidate_bank"] is False
    assert occurrence_map[(0, "agg_pass")]["candidate_bank_source"] == "checker_positive_aggregator"
    assert occurrence_map[(0, "sink")]["candidate_bank_source"] == "sink_output"
    solver_entry = next(item for item in bundle["candidates"] if item["text"] == "proposal-pass")
    assert solver_entry["origin_node_id"] == "solver_pass"
    assert solver_entry["origin_turn_index"] == 0
    assert solver_entry["origin_role"] == "solver"
    assert solver_entry["candidate_bank_source"] == "checker_approved_proposal"
    assert solver_entry["verifier_snapshot"]["positive"] is True
    assert solver_entry["provenance"] == [
        {
            "node_id": "solver_pass",
            "turn_index": 0,
            "role": "solver",
            "candidate_bank_source": "checker_approved_proposal",
        }
    ]
