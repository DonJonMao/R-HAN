from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stage2_phase1_semantic_safe_override.artifacts import canonicalize_candidate
from stage2_phase1_semantic_safe_override.verifier import _support_map, verify_artifact


QUESTION_TEXT = (
    "Determine if there is a path between two nodes in the graph. "
    "Note that (i,j) means that node i and node j are connected with an undirected edge.\n"
    "Graph: (0,1) (1,2)\n"
    "Q: Is there a path between node 0 and node 2?\n"
    'A:\n\nReturn only a JSON object using this schema: {"answer":"yes"}. Do not output explanations.'
)


def test_verify_artifact_nlgraph_connectivity_marks_correct_json_as_low_execution_residual():
    artifact = canonicalize_candidate(
        candidate_text='{"answer":"yes"}',
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    state = verify_artifact(
        artifact,
        question_text=QUESTION_TEXT,
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    assert state.residual_vector["r_parse"] == 0.0
    assert state.residual_vector["r_execution"] == 0.0
    assert state.residual_vector["r_constraint"] == 0.0
    assert state.answer_consistency_score > 0.7


def test_verify_artifact_nlgraph_connectivity_penalizes_wrong_answer_even_if_json_is_valid():
    artifact = canonicalize_candidate(
        candidate_text='{"answer":"no"}',
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    state = verify_artifact(
        artifact,
        question_text=QUESTION_TEXT,
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    assert state.residual_vector["r_parse"] == 0.0
    assert state.residual_vector["r_execution"] > 0.9
    assert state.answer_consistency_score < 0.75


def test_verify_artifact_nlgraph_prose_recoverable_answer_gets_high_parse_penalty_but_executes():
    artifact = canonicalize_candidate(
        candidate_text="I checked the graph carefully. The answer is yes.",
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    state = verify_artifact(
        artifact,
        question_text=QUESTION_TEXT,
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    assert artifact.answer_object.valid is False
    assert artifact.answer_object.fields["recoverable_valid"] is True
    assert state.residual_vector["r_parse"] >= 0.85
    assert state.residual_vector["r_execution"] == 0.0
    assert state.residual_vector["r_completeness"] >= 0.85


def test_support_map_is_not_boosted_by_stage1_anchor_identity():
    artifact = canonicalize_candidate(
        candidate_text='{"answer":"yes"}',
        metadata={"task": "connectivity"},
        dataset_name="nlgraph",
        task_type="graph_reasoning",
        answer_format="graph_json",
        task_subtype="connectivity",
    )

    anchor_support = _support_map(artifact, {"stage1_anchor": True, "reviewer_mean_trust": 0.5})
    non_anchor_support = _support_map(artifact, {"stage1_anchor": False, "reviewer_mean_trust": 0.5})

    assert anchor_support == non_anchor_support
