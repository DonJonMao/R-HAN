from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

from stage2_gcr_plus.discrete_slot_calibration import (
    PairwiseSlotProbeResult,
    SlotChallenger,
    accepts_contrast_certificate,
    accepts_certified_slot_update,
    accepts_pairwise_slot_update,
    accepts_kc_certificate,
    aggregate_fd_ccs_factor_evals,
    build_contrast_certificate,
    build_fd_ccs_candidate_bank,
    build_factor_ir,
    build_kc_fd_ccs_certificates,
    build_kc_constraints,
    build_mmlu_factor_evals_from_matrix,
    build_mmlu_option_matrix_certificates,
    build_contrastive_rescue_certificate_prompt,
    build_slot_challenger_proposal_prompt,
    challenger_from_certificate,
    fd_ccs_policy_for_dataset,
    pairwise_probe_from_certificate,
    parse_kc_factor_certificate_result,
    parse_contrastive_rescue_certificate_result,
    parse_contrastive_rescue_hint_result,
    parse_fd_ccs_factor_eval_rows,
    make_slot_eval,
    mine_multislot_mentions_safely,
    mine_slot_challengers_from_mentions,
    parse_slot_challenger_proposal,
    parse_slot_artifact,
    parse_slot_problem,
    preserves_frozen_slots,
    slot_assignment_final_answer,
    slot_update_dominates,
    summarize_fd_ccs_generation,
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
    assert certs[0].certificate_kind == "fd_ccs_factor_group_contrast"
    assert certs[0].challenger_value == "beta"
    assert certs[0].discriminator == "satisfies the beta target"
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


def test_mmlu_option_matrix_lifts_stable_empty_support_to_auditable_certificate():
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

    assert certs
    assert certs[0].challenger_support
    assert any(atom.source == "llm_vote_summary" for atom in certs[0].challenger_support)


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
    assert certs[0].certificate_kind == "fd_ccs_factor_group_contrast"
    assert set(certs[0].changed_slots) == {"blank 1", "blank 2"}
    assert not accepts_kc_certificate(problem=problem, cert=certs[0])


def test_fd_ccs_mmlu_factor_ir_enumerates_all_non_anchor_options():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
3. gamma
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    ir = build_factor_ir(
        problem=problem,
        anchor_eval=anchor_eval,
        dataset_name="mmlu_pro",
        rubric={"target_condition": "satisfies the beta target", "must_have": ["beta property"]},
    )
    candidates = build_fd_ccs_candidate_bank(
        problem=problem,
        anchor_eval=anchor_eval,
        ir=ir,
        policy=fd_ccs_policy_for_dataset("mmlu_pro"),
    )

    assert ir.variables == ("answer",)
    assert ir.domains["answer"] == ("alpha", "beta", "gamma")
    assert any(factor.factor_id == "mmlu_target_condition" for factor in ir.factors)
    assert all(factor.group_id == "mmlu_correctness" for factor in ir.factors)
    assert {item["answer"] for item in candidates} == {"beta", "gamma"}


def test_mmlu_factor_matrix_evaluates_all_rubric_factors():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    rubric = {
        "target_condition": "selects the beta-compatible option",
        "must_have": ["has beta property"],
        "disqualifiers": ["uses alpha distractor"],
    }
    ir = build_factor_ir(problem=problem, anchor_eval=anchor_eval, dataset_name="mmlu_pro", rubric=rubric)
    matrix = """
{
  "target_condition": "selects the beta-compatible option",
  "options": [
    {
      "label": "1",
      "option_value": "alpha",
      "overall_status": "violated",
      "factor_evals": [
        {"factor_id": "mmlu_disqualifier_1", "status": "violated",
         "support": [], "conflict": ["alpha triggers the distractor disqualifier"],
         "conflict_key": "alpha_disqualifier", "source_kind": "definition", "confidence": 0.8}
      ]
    },
    {
      "label": "2",
      "option_value": "beta",
      "overall_status": "satisfied",
      "factor_evals": [
        {"factor_id": "mmlu_must_have_1", "status": "satisfied",
         "support": ["beta has the required beta property"], "conflict": [],
         "support_key": "beta_property", "source_kind": "definition", "confidence": 0.8}
      ]
    }
  ]
}
"""
    parsed = parse_fd_ccs_factor_eval_rows(matrix, problem, ir)
    evals = build_mmlu_factor_evals_from_matrix(ir=ir, problem=problem, matrix_texts=[matrix, matrix, matrix])
    cert = build_contrast_certificate(
        problem=problem,
        ir=ir,
        anchor_assignment=dict(anchor_eval.artifact.assignment),
        candidate_assignment={"answer": "beta"},
        factor_evals=evals,
        policy=fd_ccs_policy_for_dataset("mmlu_pro"),
        candidate_bank_size=1,
    )

    assert any(key[1] == "mmlu_must_have_1" for key in parsed)
    assert any(key[1] == "mmlu_disqualifier_1" for key in parsed)
    assert cert is not None
    assert cert.certificate_kind == "fd_ccs_factor_group_contrast"
    assert cert.anchor_conflict
    assert cert.challenger_support


def test_evidence_lifting_keeps_non_repeated_support_when_votes_stable():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    rubric = {"target_condition": "satisfies the beta target"}
    matrices = []
    for idx in range(3):
        matrices.append(
            f"""
{{
  "target_condition": "satisfies the beta target",
  "options": [
    {{"label": "1", "option_value": "alpha", "satisfies_target": "no",
      "support": [], "conflict": ["alpha fails beta in wording {idx}"], "decisive_relation": ""}},
    {{"label": "2", "option_value": "beta", "satisfies_target": "yes",
      "support": ["beta support phrasing {idx}"], "conflict": [], "decisive_relation": ""}}
  ]
}}
"""
        )
    certs = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric=rubric,
        matrix_texts=matrices,
    )

    assert certs
    assert any(atom.source == "llm_vote_summary" for atom in certs[0].challenger_support)
    assert any(atom.source == "llm_vote_summary" for atom in certs[0].anchor_conflict)


def test_fd_ccs_contrast_certificate_builds_factor_group_flip():
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
     "support": [], "conflict": ["alpha lacks beta"], "decisive_relation": "beta property"},
    {"label": "2", "option_value": "beta", "satisfies_target": "yes",
     "support": ["beta has beta"], "conflict": [], "decisive_relation": "beta property"}
  ]
}
"""
    certs = build_mmlu_option_matrix_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        rubric={"target_condition": "satisfies the beta target"},
        matrix_texts=[matrix, matrix, matrix],
    )

    assert certs
    assert certs[0].shared_discriminator_count == 1
    assert certs[0].factor_count >= 1
    assert certs[0].candidate_bank_size == 1
    assert certs[0].score_margin > 0


def test_contrastive_rescue_parser_builds_auditable_certificate():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    ir = build_factor_ir(
        problem=problem,
        anchor_eval=anchor_eval,
        dataset_name="mmlu_pro",
        rubric={"target_condition": "satisfies the beta target"},
    )
    evals = build_mmlu_factor_evals_from_matrix(ir=ir, problem=problem, matrix_texts=[])
    prompt = build_contrastive_rescue_certificate_prompt(
        problem=problem,
        ir=ir,
        anchor_assignment=dict(anchor_eval.artifact.assignment),
        candidate_assignment={"answer": "beta"},
        factor_evals=evals,
    )
    raw = """
{
  "valid_certificate": true,
  "target_condition": "satisfies the beta target",
  "discriminator": "beta property where challenger passes and anchor fails",
  "anchor_conflict": ["alpha lacks beta property"],
  "challenger_support": ["beta has beta property"],
  "challenger_conflict": [],
  "changed_slots": ["answer"],
  "score_margin_explanation": "medium confidence contrast",
  "confidence": "medium"
}
"""
    cert = parse_contrastive_rescue_certificate_result(
        raw,
        problem=problem,
        ir=ir,
        anchor_assignment=dict(anchor_eval.artifact.assignment),
        candidate_assignment={"answer": "beta"},
        factor_evals=evals,
        candidate_bank_size=1,
    )

    assert "constructing a contrastive certificate" in prompt
    assert cert is not None
    assert cert.certificate_kind == "fd_ccs_contrastive_rescue"
    assert cert.score_margin >= 1.0
    assert cert.challenger_support
    assert parse_contrastive_rescue_hint_result(
        raw,
        problem=problem,
        candidate_assignment={"answer": "beta"},
    ) == {"answer": "beta"}
    decision = accepts_contrast_certificate(
        problem=problem,
        cert=cert,
        policy=replace(fd_ccs_policy_for_dataset("mmlu_pro"), require_audit=False),
        audit=None,
    )
    assert not decision.accepted
    assert decision.reason == "reject_rescue_cert_untrusted"


def test_mmlu_matrix_parser_preserves_dict_option_labels():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    ir = build_factor_ir(
        problem=problem,
        anchor_eval=anchor_eval,
        dataset_name="mmlu_pro",
        rubric={"target_condition": "satisfies the beta target"},
    )
    matrix = """
{
  "schema": "fd_ccs_mmlu_matrix_v2",
  "target_condition": "satisfies the beta target",
  "option_evals": {
    "1": {"status": "V", "conflict": "alpha lacks beta", "decisive_factor": "mmlu_target_condition"},
    "2": {"status": "S", "support": "beta has beta", "decisive_factor": "mmlu_target_condition"}
  }
}
"""
    parsed = parse_fd_ccs_factor_eval_rows(matrix, problem, ir)
    evals = build_mmlu_factor_evals_from_matrix(ir=ir, problem=problem, matrix_texts=[matrix, matrix, matrix])
    cert = build_contrast_certificate(
        problem=problem,
        ir=ir,
        anchor_assignment=dict(anchor_eval.artifact.assignment),
        candidate_assignment={"answer": "beta"},
        factor_evals=evals,
        policy=fd_ccs_policy_for_dataset("mmlu_pro"),
        candidate_bank_size=1,
    )

    assert len({key[0] for key in parsed}) == 2
    assert cert is not None
    assert cert.native_corroborated


def test_fd_ccs_diagnostics_reports_no_certificate_reason():
    problem = parse_slot_problem(
        """
Question: Which option satisfies the beta target?
1. alpha
2. beta
"""
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact("FINAL: 1", problem))
    ir = build_factor_ir(
        problem=problem,
        anchor_eval=anchor_eval,
        dataset_name="mmlu_pro",
        rubric={"target_condition": "satisfies the beta target"},
    )
    candidate = {"answer": "beta"}
    evals = build_mmlu_factor_evals_from_matrix(ir=ir, problem=problem, matrix_texts=[])
    diagnostics = summarize_fd_ccs_generation(
        problem=problem,
        ir=ir,
        candidate_assignments=[candidate],
        factor_evals=evals,
        certs=[],
        raw_texts=[],
        policy=fd_ccs_policy_for_dataset("mmlu_pro"),
        rubric={"target_condition": "satisfies the beta target"},
    )

    assert diagnostics["candidate_bank_size"] == 1
    assert diagnostics["empty_cert_reason"] == "no_matrix_raw"
    assert "anchor_eval_status_hist" in diagnostics


def test_fd_ccs_kc_duplicate_factor_builds_contrast_certificate():
    problem = parse_slot_problem(
        "Fill blanks.",
        metadata={
            "options_by_blank": {
                "blank 1": ["Film_X", "Film_Y"],
                "blank 2": ["Film_X", "Film_Y"],
            }
        },
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact('["Film_X","Film_X"]', problem))
    candidate = {"blank 1": "Film_X", "blank 2": "Film_Y"}
    ir = build_factor_ir(problem=problem, anchor_eval=anchor_eval, dataset_name="knowledge_crosswords")
    evals = aggregate_fd_ccs_factor_evals(
        problem=problem,
        ir=ir,
        assignments=[dict(anchor_eval.artifact.assignment), candidate],
        eval_texts=[],
        include_deterministic=True,
    )
    cert = build_contrast_certificate(
        problem=problem,
        ir=ir,
        anchor_assignment=dict(anchor_eval.artifact.assignment),
        candidate_assignment=candidate,
        factor_evals=evals,
        policy=fd_ccs_policy_for_dataset("knowledge_crosswords"),
        candidate_bank_size=1,
    )

    assert cert is not None
    assert cert.certificate_kind == "fd_ccs_factor_group_contrast"
    assert cert.anchor_conflict
    assert cert.challenger_support
    assert cert.discriminator


def test_unified_contrast_accept_rejects_challenger_conflict():
    problem = parse_slot_problem(
        "Fill blanks.",
        metadata={
            "options_by_blank": {
                "blank 1": ["Film_X", "Film_Y"],
                "blank 2": ["Film_X", "Film_Y"],
            }
        },
    )
    anchor_eval = make_slot_eval(problem, parse_slot_artifact('["Film_X","Film_X"]', problem))
    cert = build_kc_fd_ccs_certificates(
        problem=problem,
        anchor_eval=anchor_eval,
        candidate_assignments=[{"blank 1": "Film_X", "blank 2": "Film_Y"}],
        eval_texts=[],
        policy=fd_ccs_policy_for_dataset("knowledge_crosswords"),
    )[0]
    policy = replace(fd_ccs_policy_for_dataset("knowledge_crosswords"), require_audit=False)
    bad = replace(cert, challenger_conflict=cert.anchor_conflict)

    assert accepts_contrast_certificate(problem=problem, cert=cert, policy=policy).accepted
    decision = accepts_contrast_certificate(problem=problem, cert=bad, policy=policy)
    assert not decision.accepted
    assert decision.reason == "reject_challenger_conflict"


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
