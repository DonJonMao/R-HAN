from __future__ import annotations

from types import SimpleNamespace

from mas_stage2_v4_3.code_repair import CodeRepairEval, evaluate_code_candidate
from mas_stage2_v4_3.runtime import Stage2RuntimeV43


def _eval(
    *,
    passed: int,
    total: int,
    syntax_ok: bool = True,
    entry_point_ok: bool = True,
    failure_kind: str = "",
) -> CodeRepairEval:
    return CodeRepairEval(
        code_text="def solve(x):\n    return x\n",
        syntax_ok=syntax_ok,
        entry_point_ok=entry_point_ok,
        passed=passed,
        total=total,
        failure_kind=failure_kind,
        failing_examples=(),
    )


def test_code_repair_eval_dominance_respects_verifier_order():
    syntax_error = _eval(passed=0, total=0, syntax_ok=False, entry_point_ok=False, failure_kind="syntax_error")
    partial = _eval(passed=1, total=2, failure_kind="visible_test_failure")
    full = _eval(passed=2, total=2)

    assert partial.dominates(syntax_error) is True
    assert full.dominates(partial) is True
    assert syntax_error.dominates(partial) is False


def test_mbpp_visible_test_evaluator_ignores_challenge_tests():
    candidate = """
```python
def inc(x):
    return x + 1
```
"""
    metadata = {
        "entry_point": "inc",
        "test_list": ["assert inc(1) == 2"],
        "challenge_test_list": ["assert inc(2) == 999"],
    }

    result = evaluate_code_candidate(candidate, metadata, timeout_s=2.0)

    assert result.syntax_ok is True
    assert result.entry_point_ok is True
    assert result.passed == 1
    assert result.total == 1
    assert result.fully_passed is True


def test_verified_seed_pool_keeps_anchor_even_if_not_in_top_k():
    runtime = object.__new__(Stage2RuntimeV43)
    runtime.config = SimpleNamespace(repair_seed_top_k=1)
    runtime._verified_rank_key = lambda entry, feedback: feedback.rank_key  # type: ignore[attr-defined]

    def prepare(text, *, metadata, parent=None, repair_agent_id="", repair_round=-1):
        del text, metadata, repair_agent_id, repair_round
        assert parent is not None
        return dict(parent), parent["feedback"]

    runtime._prepare_verified_entry = prepare  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "anchor", "feedback": _eval(passed=1, total=2)}
    better = {"digest": "better", "text": "better", "feedback": _eval(passed=2, total=2)}

    pool, anchor_pair = runtime._verified_seed_pool(anchor=anchor, candidates=[better], metadata={})

    assert [entry["digest"] for entry, _ in pool] == ["better", "anchor"]
    assert anchor_pair is not None
    assert anchor_pair[0]["digest"] == "anchor"


def test_preserve_anchor_when_restart_seed_does_not_beat_anchor():
    runtime = object.__new__(Stage2RuntimeV43)
    runtime.config = SimpleNamespace(repair_rounds=2)
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]
    runtime._verified_rank_key = lambda entry, feedback: feedback.rank_key  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "A", "score": 0.9, "review": 0.9, "source": "anchor"}
    seed = {"digest": "seed", "text": "B", "score": 0.7, "review": 0.7, "source": "stage2"}
    anchor_pair = (anchor, _eval(passed=1, total=2, failure_kind="visible_test_failure"))
    seed_pair = (seed, _eval(passed=0, total=2, failure_kind="visible_test_failure"))

    runtime._verified_seed_pool = lambda **kwargs: ([anchor_pair, seed_pair], anchor_pair)  # type: ignore[attr-defined]
    runtime._generate_repair_branches = lambda **kwargs: []  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor(
        question_text="q",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="code_generation"),
        candidates=[seed, anchor],
        anchor=anchor,
    )

    assert selected == anchor
    assert reason == "v4_3_code_preserve_anchor_superior"
    assert extra["v4_3_stage1_anchor_used"] is True


def test_override_when_repair_branch_strictly_improves_anchor():
    runtime = object.__new__(Stage2RuntimeV43)
    runtime.config = SimpleNamespace(repair_rounds=2)
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]
    runtime._verified_rank_key = lambda entry, feedback: feedback.rank_key  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "A", "score": 0.8, "review": 0.8, "source": "anchor"}
    repaired = {
        "digest": "repair1",
        "text": "patched",
        "score": 0.9,
        "review": 0.9,
        "source": "code_repair_branch",
        "repair_branch": True,
    }
    anchor_pair = (anchor, _eval(passed=1, total=2, failure_kind="visible_test_failure"))
    repaired_pair = (repaired, _eval(passed=2, total=2))

    runtime._verified_seed_pool = lambda **kwargs: ([anchor_pair], anchor_pair)  # type: ignore[attr-defined]
    runtime._generate_repair_branches = lambda **kwargs: [repaired_pair]  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor(
        question_text="q",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="code_generation"),
        candidates=[anchor],
        anchor=anchor,
    )

    assert selected == repaired
    assert reason == "v4_3_code_override_repair_improvement"
    assert extra["v4_3_selected_is_repair_branch"] is True
    assert extra["v4_3_stage1_anchor_used"] is False
