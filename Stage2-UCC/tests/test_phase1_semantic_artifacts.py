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
        metadata={},
        task_type="multiple_choice_qa",
    )

    assert artifact.answer_signature == "option::B"
    assert artifact.answer_unit_ids
    assert artifact.evidence_unit_ids
    assert answer_text(artifact).endswith("Answer: B")
    assert any(unit.text.startswith("We compare") for unit in evidence_units(artifact))


def test_canonicalize_code_builds_answer_signature_and_exec_view():
    artifact = canonicalize_candidate(
        candidate_text="from typing import List\n\ndef solve(xs: List[int]) -> int:\n    return xs[0]\n",
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata={"entry_point": "solve"},
        task_type="code_generation",
    )

    assert artifact.answer_signature.startswith("code::solve::")
    assert artifact.views["exec_view"].confidence_clue > 0.5
    assert artifact.answer_unit_ids
    assert artifact.schema_features["answer_kind"] == "code"
