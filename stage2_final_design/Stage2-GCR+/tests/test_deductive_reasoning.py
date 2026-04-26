from __future__ import annotations

from types import SimpleNamespace

from mas_treesearch.evaluator import MultiFidelityEvaluator
from stage2_gcr_plus.deductive_reasoning import (
    answer_only_from_artifact,
    deductive_dominates,
    make_deductive_eval,
    parse_deductive_artifact,
    verify_deductive_artifact,
)
from stage2_gcr_plus.runtime_v44 import Stage2RuntimeV44


def _runtime_stub() -> Stage2RuntimeV44:
    runtime = object.__new__(Stage2RuntimeV44)
    runtime.evaluator = object.__new__(MultiFidelityEvaluator)
    runtime.config = SimpleNamespace(max_logged_candidates=4)
    runtime._v4_4_route_family = "deductive_reasoning"
    return runtime


def _eval(text: str, *, dataset_name: str = "gsm8k"):
    profile = SimpleNamespace(task_type="numeric" if dataset_name == "gsm8k" else "math_expression", name=dataset_name)
    artifact = parse_deductive_artifact(text, dataset_name)
    residual = verify_deductive_artifact("A test problem.", artifact, profile, {"dataset_name": dataset_name})
    return make_deductive_eval(artifact, residual, {"text": text})


def test_deductive_parser_extracts_steps_final_and_equation():
    artifact = parse_deductive_artifact(
        "SOLUTION:\n"
        "1. There are 17 groups.\n"
        "2. 17 * 6 = 112.\n"
        "FINAL: 112\n",
        "gsm8k",
    )

    assert len(artifact.steps) == 2
    assert artifact.final_answer == "112"
    assert artifact.normalized_final_answer == "112"
    assert artifact.steps[1].equations == ("17 * 6 = 112",)
    assert artifact.final_source == "final_line"
    assert artifact.step_source == "solution_numbered"
    assert artifact.contract_ok is True


def test_deductive_verifier_flags_arithmetic_mismatch():
    artifact = parse_deductive_artifact(
        "SOLUTION:\n"
        "1. There are 17 groups.\n"
        "2. 17 * 6 = 112.\n"
        "FINAL: 112\n",
        "gsm8k",
    )
    profile = SimpleNamespace(task_type="numeric", name="gsm8k")

    residual = verify_deductive_artifact("How many items are in 17 groups of 6?", artifact, profile, {"dataset_name": "gsm8k"})

    assert residual.residual_kind == "arithmetic_mismatch"
    assert residual.first_bad_step == 2
    assert residual.verified_prefix_len == 1
    assert residual.checked_equation_count == 1


def test_deductive_verifier_flags_final_inconsistent_with_derivation():
    artifact = parse_deductive_artifact(
        "SOLUTION:\n"
        "1. There are 17 groups.\n"
        "2. 17 * 6 = 102.\n"
        "FINAL: 112\n",
        "gsm8k",
    )
    profile = SimpleNamespace(task_type="numeric", name="gsm8k")

    residual = verify_deductive_artifact("How many items are in 17 groups of 6?", artifact, profile, {"dataset_name": "gsm8k"})

    assert residual.residual_kind == "final_inconsistent_with_derivation"
    assert residual.final_consistent is False


def test_deductive_class_collapse_ignores_verified_prefix_len_for_same_residual():
    runtime = _runtime_stub()
    profile = SimpleNamespace(task_type="numeric", name="gsm8k")
    entries = [
        {
            "digest": "a",
            "text": "SOLUTION:\n1. 5 + 5 = 10.\nFINAL: 10",
            "sink_support": 0,
            "stage1_anchor": False,
        },
        {
            "digest": "b",
            "text": "SOLUTION:\n1. 5 + 5 = 10.\n2. 6 + 4 = 10.\nFINAL: 10",
            "sink_support": 0,
            "stage1_anchor": False,
        },
    ]

    verified = runtime._verify_deductive_pool("How many items?", None, entries, profile, {"dataset_name": "gsm8k"})
    collapsed = runtime._collapse_deductive_classes(verified)

    assert len(collapsed) == 1
    assert collapsed[0]["v4_4_class_size"] == 2


def test_deductive_dominance_accepts_residual_reduction():
    old = _eval("SOLUTION:\n1. There are 17 groups.\n2. 17 * 6 = 112.\nFINAL: 112")
    new = _eval("SOLUTION:\n1. There are 17 groups.\n2. 17 * 6 = 102.\n3. Therefore the answer is 102.\nFINAL: 102")

    assert deductive_dominates(new, old) is True


def test_deductive_dominance_rejects_earlier_error():
    old = _eval("SOLUTION:\n1. 1 + 1 = 2.\n2. 2 + 2 = 4.\n3. 4 + 4 = 8.\n4. 8 + 1 = 10.\nFINAL: 10")
    new = _eval("SOLUTION:\n1. 1 + 1 = 2.\n2. 2 + 2 = 5.\nFINAL: 10")

    assert deductive_dominates(new, old) is False


def test_gsm8k_long_text_without_final_has_no_final_answer():
    artifact = parse_deductive_artifact(
        "He studies 1/16 of the weekday time on each weekend day.",
        "gsm8k",
    )

    assert artifact.final_answer is None
    assert artifact.final_source == "none"
    assert artifact.parser_confidence == "low"


def test_answer_only_anchor_not_dominated_by_long_wrong_derivation():
    old = _eval("30")
    new = _eval(
        "SOLUTION:\n"
        "1. Weekday minutes = (6+2) * 40 * 5 = 1600\n"
        "2. Weekend ratio is 1/16.\n"
        "FINAL: 16\n"
    )

    assert deductive_dominates(new, old) is False


def test_same_answer_derivation_can_enrich_anchor():
    old = _eval("30")
    new = _eval(
        "SOLUTION:\n"
        "1. Weekday minutes = (6 + 2) * 40 * 5 = 1600\n"
        "2. Weekend minutes = 1600 / 16 * 2 = 200\n"
        "3. Total minutes = 1600 + 200 = 1800\n"
        "4. Hours = 1800 / 60 = 30\n"
        "\n"
        "FINAL: 30\n"
    )

    assert deductive_dominates(new, old) is True


def test_arithmetic_repair_can_change_answer_when_old_has_fatal():
    old = _eval("SOLUTION:\n1. Total = 17 * 6 = 112\n\nFINAL: 112")
    new = _eval("SOLUTION:\n1. Total = 17 * 6 = 102\n\nFINAL: 102")

    assert deductive_dominates(new, old) is True


def test_same_answer_same_residual_collapses_despite_prefix_len():
    a = _eval("SOLUTION:\n1. 5 + 5 = 10\nFINAL: 10")
    b = _eval("SOLUTION:\n1. 5 + 5 = 10\n2. 6 + 4 = 10\nFINAL: 10")

    assert a.class_key == b.class_key


def test_runtime_routes_gsm8k_and_math_to_deductive_reasoning():
    runtime = _runtime_stub()

    assert runtime._route_family(SimpleNamespace(task_type="numeric", name="gsm8k"), {}) == "deductive_reasoning"
    assert runtime._route_family(SimpleNamespace(task_type="math_expression", name="math"), {}) == "deductive_reasoning"
    assert runtime._route_family(SimpleNamespace(task_type="numeric", name=""), {"mas_dataset_name": "gsm8k"}) == "deductive_reasoning"


def test_deductive_internal_solution_returns_answer_only_artifact():
    artifact = parse_deductive_artifact(
        "SOLUTION:\n1. There are 17 groups.\n2. 17 * 6 = 102.\nFINAL: 102",
        "gsm8k",
    )

    assert answer_only_from_artifact(artifact) == "102"
