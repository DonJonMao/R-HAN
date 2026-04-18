from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stage2_phase1_semantic_safe_override.artifacts import answer_text, canonicalize_candidate, evidence_units


def test_canonicalize_multiple_choice_extracts_answer_and_evidence_units():
    artifact = canonicalize_candidate(
        candidate_text="We compare the options and eliminate A, C, and D.\nAnswer: B",
        provenance=[{"node_id": "aggregator", "turn_index": 1, "role": "aggregator"}],
        metadata={"options": ["alpha", "beta", "gamma", "delta"]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
    )

    assert artifact.answer_signature == "option::2"
    assert artifact.answer_object.kind == "option"
    assert artifact.answer_object.value == 2
    assert artifact.answer_unit_ids
    assert artifact.evidence_unit_ids
    assert answer_text(artifact).endswith("Answer: B")
    assert any(unit.text.startswith("We compare") for unit in evidence_units(artifact))


def test_canonicalize_nlgraph_builds_typed_graph_answer_object():
    artifact = canonicalize_candidate(
        candidate_text='{"answer":"no"}',
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    assert artifact.answer_signature == "graph_bool::answer::no"
    assert artifact.answer_object.kind == "graph_bool"
    assert artifact.answer_object.fields["schema_valid"] is True
    assert artifact.schema_features["answer_format"] == "graph_json"


def test_canonicalize_option_ignores_stray_reasoning_numbers_and_uses_final_answer_surface():
    artifact = canonicalize_candidate(
        candidate_text="There are 8 candidate explanations and 3 tempting distractors.\nFinal answer: B",
        provenance=[{"node_id": "aggregator", "turn_index": 1, "role": "aggregator"}],
        metadata={"options": [f"opt_{idx}" for idx in range(1, 11)]},
        dataset_name="mmlu_pro",
        task_type="mcq",
        answer_format="option",
    )

    assert artifact.answer_object.kind == "option"
    assert artifact.answer_object.value == 2
    assert artifact.answer_signature == "option::2"


def test_canonicalize_nlgraph_recoverable_prose_is_contract_invalid_but_semantically_recoverable():
    artifact = canonicalize_candidate(
        candidate_text="After checking the graph, the answer is yes.",
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    assert artifact.answer_object.kind == "graph_bool"
    assert artifact.answer_object.valid is False
    assert artifact.answer_object.fields["contract_valid"] is False
    assert artifact.answer_object.fields["recoverable_valid"] is True
    assert artifact.answer_object.value == "yes"
    assert artifact.answer_signature == "graph_bool::answer::yes"


def test_canonicalize_code_builds_answer_signature_and_exec_view():
    artifact = canonicalize_candidate(
        candidate_text="from typing import List\n\ndef solve(xs: List[int]) -> int:\n    return xs[0]\n",
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata={"entry_point": "solve"},
        dataset_name="mbpp",
        task_type="code_generation",
        answer_format="python_code",
    )

    assert artifact.answer_signature.startswith("code::solve::")
    assert artifact.views["exec_view"].confidence_clue > 0.5
    assert artifact.answer_unit_ids
    assert artifact.schema_features["answer_kind"] == "code"
    assert len(artifact.answer_unit_ids) < len(artifact.units)
    assert artifact.answer_object.fields["syntax_ok"] is True
