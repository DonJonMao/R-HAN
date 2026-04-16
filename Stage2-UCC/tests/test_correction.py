from __future__ import annotations

from stage2_phase3a_unified.artifacts import canonicalize_candidate
from stage2_phase3a_unified.correction import apply_correction_artifact, deterministic_proposal
from stage2_phase3a_unified.verifier import verify_artifact


def test_deterministic_proposal_recovers_missing_typing_import():
    metadata = {
        "entry_point": "solve",
        "test": "def check(candidate):\n    assert candidate([1]) == 1\n",
    }
    artifact = canonicalize_candidate(
        candidate_text="def solve(xs: List[int]) -> int:\n    return xs[0]\n",
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata=metadata,
        task_type="code_generation",
    )
    state = verify_artifact(
        artifact,
        question_text="Write a function solve that returns the first element.",
        metadata=metadata,
        task_type="code_generation",
        candidate_entry={"stage1_anchor": False},
        anchor_artifact=None,
    )

    correction = deterministic_proposal(
        artifact,
        state,
        critique={"target_units": ["u1"], "preserve_units": [], "expected_delta": {}},
        anchor_artifact=None,
        task_type="code_generation",
    )

    assert correction is not None
    assert correction.operation == "insert_before"
    assert "from typing import List" in correction.new_units[0]


def test_apply_correction_artifact_inserts_before_first_unit():
    artifact = canonicalize_candidate(
        candidate_text="def solve(x):\n    return x\n",
        provenance=[],
        metadata={},
        task_type="code_generation",
    )
    patched = apply_correction_artifact(
        artifact,
        deterministic_proposal(
            canonicalize_candidate(
                candidate_text="def solve(xs: List[int]) -> int:\n    return xs[0]\n",
                provenance=[],
                metadata={"entry_point": "solve", "test": "def check(candidate):\n    assert candidate([1]) == 1\n"},
                task_type="code_generation",
            ),
            verify_artifact(
                canonicalize_candidate(
                    candidate_text="def solve(xs: List[int]) -> int:\n    return xs[0]\n",
                    provenance=[],
                    metadata={"entry_point": "solve", "test": "def check(candidate):\n    assert candidate([1]) == 1\n"},
                    task_type="code_generation",
                ),
                question_text="Write a function solve that returns the first element.",
                metadata={"entry_point": "solve", "test": "def check(candidate):\n    assert candidate([1]) == 1\n"},
                task_type="code_generation",
                candidate_entry={"stage1_anchor": False},
                anchor_artifact=None,
            ),
            critique={"target_units": ["u1"], "preserve_units": [], "expected_delta": {}},
            anchor_artifact=None,
            task_type="code_generation",
        ),
    )

    assert patched.rendered_answer.startswith("from typing import List")
