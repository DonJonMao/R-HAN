from __future__ import annotations

from stage2_phase3a_unified.artifacts import canonicalize_candidate


def test_canonicalize_code_builds_all_views():
    artifact = canonicalize_candidate(
        candidate_text="from typing import List\n\ndef solve(xs: List[int]) -> int:\n    return xs[0]\n",
        provenance=[{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        metadata={"entry_point": "solve"},
        task_type="code_generation",
    )

    assert len(artifact.units) >= 2
    assert artifact.views["surface_view"].confidence_clue == 1.0
    assert artifact.views["exec_view"].confidence_clue > 0.5
    assert artifact.parse_confidence["struct_view"] >= 0.0


def test_canonicalize_json_prefers_struct_view():
    artifact = canonicalize_candidate(
        candidate_text='{"answer": 42, "reason": "ok"}',
        provenance=[{"node_id": "aggregator", "turn_index": 1, "role": "aggregator"}],
        metadata={},
        task_type="structured_list",
    )

    assert artifact.views["struct_view"].confidence_clue > 0.5
    assert "answer" in artifact.views["struct_view"].rendered
