from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from stage2_gcr_plus.discrete_slot_calibration import (
    PairwiseSlotProbeResult,
    SlotChallenger,
    accepts_certified_slot_update,
    accepts_pairwise_slot_update,
    accepts_kc_certificate,
    build_kc_constraints,
    build_mmlu_option_matrix_certificates,
    build_slot_challenger_proposal_prompt,
    challenger_from_certificate,
    pairwise_probe_from_certificate,
    parse_kc_factor_certificate_result,
    make_slot_eval,
    mine_multislot_mentions_safely,
    mine_slot_challengers_from_mentions,
    parse_slot_challenger_proposal,
    parse_slot_artifact,
    parse_slot_problem,
    preserves_frozen_slots,
    slot_assignment_final_answer,
    slot_update_dominates,
)
from stage2_gcr_plus.runtime_v44 import Stage2RuntimeV44


def test_mcq_options_parse_as_single_slot():
    q = """
Question: Which is correct?
A. alpha
B. beta
C. gamma
"""
    problem = parse_slot_problem(q)

    assert len(problem.slots) == 1
    assert problem.output_kind == "single_label"
    assert problem.slots[0].option_labels == ("A", "B", "C")


def test_blank_options_parse_as_multi_slot():
    meta = {
        "options_by_blank": {
            "blank_1": ["A", "B"],
            "blank_2": ["C", "D"],
        }
    }
    problem = parse_slot_problem("Fill blanks.", metadata=meta)

    assert len(problem.slots) == 2
    assert problem.output_kind == "json_list"


def test_membership_repair_preserves_frozen_slots():
    anchor = {"blank_1": "bad", "blank_2": "Christopher_Young", "blank_3": "Kim_Coates"}
    good = {"blank_1": "Blink_(film)", "blank_2": "Christopher_Young", "blank_3": "Kim_Coates"}
    bad = {"blank_1": "Blink_(film)", "blank_2": "Sol_Kaplan", "blank_3": "Supriya_Pathak"}
    mutable = {"blank_1"}

    assert preserves_frozen_slots(anchor, good, mutable)
    assert not preserves_frozen_slots(anchor, bad, mutable)


def test_membership_clean_is_not_semantic_certificate():
    q = """
Question: pick one.
A. alpha
B. beta
C. gamma
D. delta
"""
    problem = parse_slot_problem(q)
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: B", problem))
    challenger_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: D", problem))

    assert challenger_eval.residual.residual_kind == "membership_clean"
    assert not slot_update_dominates(challenger_eval, anchor_eval)


def test_pairwise_probe_accepts_challenger():
    problem = parse_slot_problem(
        """
Question: pick one.
A. alpha
B. beta
C. gamma
D. delta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: B", problem))
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("contradicts discriminator",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("matches discriminator",),
        discriminator="key relation",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key relation",
        inverse_condition="does not match key relation",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )
    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="beta",
        challenger_value="delta",
        occurrence_count=2,
        sink_support=2,
        source_count=2,
        best_entry_digest="x",
        best_entry={},
    )

    assert accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
    )


def test_negative_polarity_rejects_inverse_challenger():
    q = (
        "Which of the following is not an abnormal breathing pattern seen in head injury "
        "and altered conscious level?\n"
        "1. Hyperventilation.\n"
        "2. Anaerobic respiration.\n"
    )

    problem = parse_slot_problem(q)
    anchor_artifact = parse_slot_artifact("FINAL: 2", problem)
    anchor_eval = make_slot_eval(problem, anchor_artifact)

    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="Anaerobic respiration.",
        challenger_value="Hyperventilation.",
        occurrence_count=2,
        sink_support=2,
        source_count=2,
        best_entry_digest="x",
        best_entry={},
    )

    probe = PairwiseSlotProbeResult(
        winner="challenger",
        confidence="high",
        anchor_conflict=("claimed anaerobic respiration is abnormal",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("hyperventilation is abnormal breathing pattern",),
        discriminator="not abnormal breathing pattern",
        raw="{}",
        question_polarity="negative",
        target_condition="is not an abnormal breathing pattern",
        inverse_condition="is an abnormal breathing pattern",
        anchor_satisfies_target="yes",
        challenger_satisfies_target="no",
        anchor_satisfies_inverse="no",
        challenger_satisfies_inverse="yes",
    )

    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
    )


def test_pairwise_accept_requires_challenger_support_and_discriminator():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="alpha",
        challenger_value="beta",
        occurrence_count=2,
        sink_support=2,
        source_count=2,
        best_entry_digest="x",
        best_entry={},
    )
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("anchor conflicts",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("challenger supports",),
        discriminator="key fact",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key fact",
        inverse_condition="does not match key fact",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )

    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=replace(probe, challenger_support=()),
    )
    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=replace(probe, discriminator=""),
    )


def test_pairwise_accept_requires_target_satisfaction():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="alpha",
        challenger_value="beta",
        occurrence_count=2,
        sink_support=2,
        source_count=2,
        best_entry_digest="x",
        best_entry={},
    )
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("anchor conflicts",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("challenger supports",),
        discriminator="key fact",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key fact",
        inverse_condition="does not match key fact",
        anchor_satisfies_target="no",
        challenger_satisfies_target="uncertain",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )

    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
    )


def test_audit_disagreement_rejects_update():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="alpha",
        challenger_value="beta",
        occurrence_count=2,
        sink_support=2,
        source_count=2,
        best_entry_digest="x",
        best_entry={},
    )
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("anchor conflicts",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("challenger supports",),
        discriminator="key fact",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key fact",
        inverse_condition="does not match key fact",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )
    audit = replace(probe, winner="anchor")

    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
        audit=audit,
    )


def test_proposal_challenger_cannot_update_without_double_probe():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
"""
    )
    _, proposals = parse_slot_challenger_proposal(
        '{"target_condition":"matches key fact","challengers":[{"slot_id":"answer","value":"beta","reason":"candidate"}]}',
        problem,
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    challenger = SlotChallenger(
        slot_id=proposals[0]["slot_id"],
        anchor_value="alpha",
        challenger_value=proposals[0]["value"],
        occurrence_count=0,
        sink_support=0,
        source_count=0,
        best_entry_digest="proposal",
        best_entry={},
    )
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("anchor conflicts",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("challenger supports",),
        discriminator="key fact",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key fact",
        inverse_condition="does not match key fact",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )

    assert not accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
    )


def test_proposal_challenger_can_update_with_audit_confirmation():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    challenger = SlotChallenger(
        slot_id="answer",
        anchor_value="alpha",
        challenger_value="beta",
        occurrence_count=0,
        sink_support=0,
        source_count=0,
        best_entry_digest="proposal_0",
        best_entry={"candidate_bank_source": "slot_challenger_proposal"},
    )
    probe = PairwiseSlotProbeResult(
        winner="challenger",
        anchor_conflict=("anchor conflicts",),
        challenger_conflict=(),
        anchor_support=(),
        challenger_support=("challenger supports",),
        discriminator="key fact",
        confidence="high",
        raw="{}",
        question_polarity="positive",
        target_condition="matches key fact",
        inverse_condition="does not match key fact",
        anchor_satisfies_target="no",
        challenger_satisfies_target="yes",
        anchor_satisfies_inverse="yes",
        challenger_satisfies_inverse="no",
    )

    assert accepts_pairwise_slot_update(
        problem=problem,
        anchor_eval=anchor_eval,
        challenger=challenger,
        probe=probe,
        audit=probe,
        anchor_support=(1, 1, 0),
        challenger_support=(0, 0, 0),
    )


def test_slot_challenger_proposal_prompt_uses_max_challengers():
    problem = parse_slot_problem(
        """
Question: pick one.
1. alpha
2. beta
3. gamma
"""
    )
    artifact = parse_slot_artifact("FINAL: 1", problem)

    prompt = build_slot_challenger_proposal_prompt(
        problem=problem,
        artifact=artifact,
        max_challengers=2,
    )

    assert "propose at most 2 challengers" in prompt


def test_proposal_parser_single_slot_raw_option_label():
    problem = parse_slot_problem(
        """
Question: What happens to money supply?
1. no change in the money supply
8. a reduction of the money supply by $30,000
"""
    )

    _, proposals = parse_slot_challenger_proposal("OPTION - 8", problem)

    assert proposals
    assert proposals[0]["slot_id"] == "answer"
    assert proposals[0]["value"] == "a reduction of the money supply by $30,000"


def test_proposal_parser_single_slot_final_option_label():
    problem = parse_slot_problem(
        """
Question: What happens to money supply?
1. no change in the money supply
8. a reduction of the money supply by $30,000
"""
    )

    _, proposals = parse_slot_challenger_proposal("FINAL: OPTION - 8", problem)

    assert proposals[0]["value"] == "a reduction of the money supply by $30,000"


def test_proposal_parser_accepts_slot_aliases():
    problem = parse_slot_problem(
        """
Question: What happens?
1. no change
8. reduction by 30000
"""
    )

    raw = '{"challengers":[{"slot":"answer","option":"8","reason":"multiplier"}]}'
    _, proposals = parse_slot_challenger_proposal(raw, problem)

    assert proposals[0]["slot_id"] == "answer"
    assert proposals[0]["value"] == "reduction by 30000"


def test_proposal_parser_drops_value_outside_allowed_options():
    problem = parse_slot_problem(
        """
Question: What happens?
1. no change
8. reduction by 30000
"""
    )

    raw = '{"challengers":[{"slot_id":"answer","value":"not in options"}]}'
    _, proposals = parse_slot_challenger_proposal(raw, problem)

    assert proposals == []


def test_discrete_json_probe_slots_request_json_output():
    slots = Stage2RuntimeV44._discrete_json_probe_slots()

    assert slots.output_style == "json"
    assert slots.reasoning_mode == "direct"


def test_single_slot_mention_mining_extracts_option_text():
    problem = parse_slot_problem(
        """
Question: untreated dental caries prevalence?
3. 70%
9. 40%
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 3", problem))
    entries = [{"text": "The correct prevalence is 40%.", "digest": "x"}]

    challengers = mine_slot_challengers_from_mentions(
        problem=problem,
        anchor_eval=anchor_eval,
        candidate_entries=entries,
    )

    assert any(ch.challenger_value == "40%" for ch in challengers)


def test_mmlu_option_matrix_builds_certified_challenger():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
3. gamma
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    rubric = {"target_condition": "satisfies the beta target"}
    matrix = """
{
  "target_condition": "satisfies the beta target",
  "options": [
    {"label": "1", "option_value": "alpha", "satisfies_target": "no",
     "support": [], "conflict": ["alpha lacks the beta property"], "decisive_relation": "beta property"},
    {"label": "2", "option_value": "beta", "satisfies_target": "yes",
     "support": ["beta has the beta property"], "conflict": [], "decisive_relation": "beta property"},
    {"label": "3", "option_value": "gamma", "satisfies_target": "no",
     "support": [], "conflict": ["gamma lacks the beta property"], "decisive_relation": "beta property"}
  ]
}
"""

    certs = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric=rubric,
        matrix_texts=[matrix, matrix, matrix],
    )

    assert certs
    assert certs[0].certificate_kind == "mmlu_option_matrix"
    assert certs[0].challenger_value == "beta"
    assert certs[0].discriminator == "beta property"
    assert certs[0].challenger_support
    assert certs[0].anchor_conflict


def test_mmlu_option_matrix_blocks_negative_inverse_challenger():
    problem = parse_slot_problem(
        """
Question: Which of the following is not an abnormal breathing pattern?
1. Hyperventilation
2. Anaerobic respiration
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 2", problem))
    rubric = {"target_condition": "is not an abnormal breathing pattern"}
    matrix = """
{
  "target_condition": "is not an abnormal breathing pattern",
  "options": [
    {"label": "1", "option_value": "Hyperventilation", "satisfies_target": "no",
     "support": [], "conflict": ["hyperventilation is an abnormal breathing pattern"], "decisive_relation": "abnormal breathing pattern"},
    {"label": "2", "option_value": "Anaerobic respiration", "satisfies_target": "yes",
     "support": ["anaerobic respiration is not an abnormal breathing pattern in this list"], "conflict": [], "decisive_relation": "not abnormal breathing pattern"}
  ]
}
"""

    certs = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric=rubric,
        matrix_texts=[matrix, matrix, matrix],
    )

    assert all(cert.challenger_value != "Hyperventilation" for cert in certs)


def test_mmlu_option_matrix_requires_support_and_discriminator():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    matrix = """
{
  "target_condition": "satisfies the beta target",
  "options": [
    {"label": "1", "option_value": "alpha", "satisfies_target": "no",
     "support": [], "conflict": ["alpha fails"], "decisive_relation": "beta property"},
    {"label": "2", "option_value": "beta", "satisfies_target": "yes",
     "support": [], "conflict": [], "decisive_relation": ""}
  ]
}
"""

    certs = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric={"target_condition": "satisfies the beta target"},
        matrix_texts=[matrix, matrix, matrix],
    )

    assert certs == []


def test_certified_update_requires_audit_confirmation():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    matrix = """
{
  "target_condition": "satisfies the beta target",
  "options": [
    {"label": "1", "option_value": "alpha", "satisfies_target": "no",
     "support": [], "conflict": ["alpha fails"], "decisive_relation": "beta property"},
    {"label": "2", "option_value": "beta", "satisfies_target": "yes",
     "support": ["beta passes"], "conflict": [], "decisive_relation": "beta property"}
  ]
}
"""
    cert = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric={"target_condition": "satisfies the beta target"},
        matrix_texts=[matrix, matrix, matrix],
    )[0]
    probe = pairwise_probe_from_certificate(cert, problem)

    assert not accepts_certified_slot_update(
        problem=problem,
        cert=cert,
        probe=probe,
        audit=replace(probe, winner="anchor"),
    )


def test_kc_constraints_extracted_from_question():
    problem = parse_slot_problem(
        "Fill blanks in triples. Sources: ['blank 1', 'Actor_A']. Relations: ['actedIn', 'actedIn']. "
        "Targets: ['Film_X', 'blank 2']. Blanks: ['blank 1', 'blank 2']. "
        "Options: {'blank 1': ['Actor_A', 'Actor_B'], 'blank 2': ['Film_X', 'Film_Y']}.",
    )

    constraints = build_kc_constraints(problem)

    assert len(constraints) == 2
    assert constraints[0].subject == "blank 1"
    assert constraints[0].predicate == "actedIn"
    assert constraints[0].object == "Film_X"


def test_kc_factor_certificate_parser_accepts_joint_update():
    problem = parse_slot_problem(
        "Fill blanks in triples. Sources: ['blank 1', 'Actor_A']. Relations: ['actedIn', 'actedIn']. "
        "Targets: ['Film_X', 'blank 2']. Blanks: ['blank 1', 'blank 2']. "
        "Options: {'blank 1': ['Actor_A', 'Actor_B'], 'blank 2': ['Film_X', 'Film_Y']}.",
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact('["Actor_B","Film_Y"]', problem))
    raw = """
{
  "certificates": [
    {
      "changed_slots": ["blank 1", "blank 2"],
      "challenger_assignment": {"blank 1": "Actor_A", "blank 2": "Film_X"},
      "target_condition": "satisfies the supplied KG constraints",
      "discriminator": "actedIn constraints",
      "support_edges": ["Actor_A actedIn Film_X"],
      "anchor_conflict_edges": ["Actor_B does not satisfy the actedIn constraint"],
      "new_conflict_edges": [],
      "score_margin": 2.0
    }
  ]
}
"""

    certs = parse_kc_factor_certificate_result(raw, problem, anchor_eval)

    assert certs
    assert certs[0].certificate_kind == "kc_factor_graph"
    assert set(certs[0].changed_slots) == {"blank 1", "blank 2"}
    assert accepts_kc_certificate(problem=problem, cert=certs[0])


def test_safe_multislot_mention_mining_extracts_unique_slot_option():
    problem = parse_slot_problem(
        "Fill blanks.",
        metadata={
            "options_by_blank": {
                "blank 1": ["Actor_A", "Actor_B"],
                "blank 2": ["Film_X", "Film_Y"],
            }
        },
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact('["Actor_B","Film_Y"]', problem))
    entries = [{"text": "The evidence points to Actor_A for the actor slot.", "digest": "x"}]

    challengers = mine_multislot_mentions_safely(
        problem=problem,
        anchor_eval=anchor_eval,
        candidate_entries=entries,
    )

    assert any(ch.slot_id == "blank 1" and ch.challenger_value == "Actor_A" for ch in challengers)


def test_single_label_final_answer_renders_label():
    q = """
Question: Which is correct?
A. alpha
B. beta
C. gamma
D. delta
"""
    problem = parse_slot_problem(q)
    artifact = parse_slot_artifact("FINAL: D", problem)
    eval_obj = make_slot_eval(problem, artifact)

    assert slot_assignment_final_answer(eval_obj) == "D"


def test_multi_slot_final_answer_renders_json_list():
    problem = parse_slot_problem(
        "Fill blanks.",
        metadata={
            "options_by_blank": {
                "blank_1": ["Blink_(film)", "Unforgettable_(film)"],
                "blank_2": ["Christopher_Young", "Sol_Kaplan"],
                "blank_3": ["Kim_Coates", "Supriya_Pathak"],
            }
        },
    )
    artifact = parse_slot_artifact('["Blink_(film)","Christopher_Young","Kim_Coates"]', problem)
    eval_obj = make_slot_eval(problem, artifact)

    assert slot_assignment_final_answer(eval_obj) == '["Blink_(film)","Christopher_Young","Kim_Coates"]'


def test_route_family_prefers_discrete_slot_calibration_over_structural():
    runtime = object.__new__(Stage2RuntimeV44)
    profile = SimpleNamespace(name="knowledge_crosswords", task_type="structured_list", answer_format="json_list")
    metadata = {
        "mas_dataset_name": "knowledge_crosswords",
        "blanks": ["blank 1", "blank 2"],
        "options": {"blank 1": ["A", "B"], "blank 2": ["C", "D"]},
    }

    route = runtime._route_family(profile, metadata, "Fill blanks.")

    assert route == "kc_factor_slot_calibration"


def test_route_family_uses_discrete_slot_calibration_for_mmlu_pro():
    runtime = object.__new__(Stage2RuntimeV44)
    profile = SimpleNamespace(name="mmlu_pro", task_type="mcq", answer_format="option")
    metadata = {"mas_dataset_name": "mmlu_pro", "options": ["alpha", "beta", "gamma", "delta"]}
    question = """
Question: Which is correct?
1) alpha
2) beta
3) gamma
4) delta
"""

    route = runtime._route_family(profile, metadata, question)

    assert route == "mmlu_option_matrix_calibration"


def test_route_family_keeps_nlgraph_in_structural_route():
    runtime = object.__new__(Stage2RuntimeV44)
    profile = SimpleNamespace(name="nlgraph", task_type="graph_reasoning", answer_format="graph_json")
    metadata = {"mas_dataset_name": "nlgraph"}
    question = "In an undirected graph, (0,1) (1,2). Is there a path between node 0 and node 2?"

    route = runtime._route_family(profile, metadata, question)

    assert route == "graph_constrained"
