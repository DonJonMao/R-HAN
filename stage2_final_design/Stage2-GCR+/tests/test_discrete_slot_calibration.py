from __future__ import annotations

from types import SimpleNamespace

from stage2_gcr_plus.discrete_slot_calibration import (
    PairwiseSlotProbeResult,
    SlotChallenger,
    accepts_pairwise_slot_update,
    make_slot_eval,
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

    assert route == "discrete_slot_calibration"


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

    assert route == "discrete_slot_calibration"


def test_route_family_keeps_nlgraph_in_structural_route():
    runtime = object.__new__(Stage2RuntimeV44)
    profile = SimpleNamespace(name="nlgraph", task_type="graph_reasoning", answer_format="graph_json")
    metadata = {"mas_dataset_name": "nlgraph"}
    question = "In an undirected graph, (0,1) (1,2). Is there a path between node 0 and node 2?"

    route = runtime._route_family(profile, metadata, question)

    assert route == "graph_constrained"
