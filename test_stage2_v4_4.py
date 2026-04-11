from __future__ import annotations

from types import SimpleNamespace

from mas_stage2.types import ControllerState, EdgeActivation, TurnTrace, Stage2RunResult
from mas_stage2_v4_4.code_repair import CodeRepairEval
from mas_stage2_v4_4.runtime import GraphConstraintEval, ReasoningEval, Stage2RuntimeV44
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.types import UnionGraph, UnionNode


def _code_eval(
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


def _runtime_stub() -> Stage2RuntimeV44:
    runtime = object.__new__(Stage2RuntimeV44)
    runtime.evaluator = object.__new__(MultiFidelityEvaluator)
    runtime._by_id = {}
    runtime.config = SimpleNamespace(
        max_logged_candidates=4,
        repair_seed_top_k=3,
        repair_rounds=1,
        repair_max_failed_examples=3,
        graph_seed_top_k=2,
        graph_repair_rounds=1,
        graph_full_branch_cap=2,
        adversarial_lean_hypothesis_cap=2,
        adversarial_full_hypothesis_cap=3,
        default_budget_bucket="normal",
        graph_repair_agent_ids=("coder",),
        inspector_agent_id="verifier",
    )
    runtime._v4_4_route_family = "adversarial"
    runtime._v4_4_execution_mode_hint = "lean"
    runtime._v4_4_focus_node_ids = set()
    runtime._v4_4_sink_guard_ids = set()
    runtime._v4_4_protected_ids = set()
    runtime._v4_4_champion_provenance_ids = set()
    return runtime


def _recovery_branch_entry(digest: str = "repair") -> dict:
    return {
        "digest": digest,
        "text": "patched",
        "score": 0.9,
        "review": 0.9,
        "source": "code_repair_branch",
        "repair_branch": True,
        "origin_node_id": "solver",
        "origin_turn_index": 0,
        "origin_role": "solver",
        "parent_candidate_digest": "seed",
        "repair_operator_type": "code_repair_patch",
        "recovery_subgraph_node_ids": ["solver", "sink"],
        "recovery_subgraph_edge_ids": ["e1"],
        "trigger_verifier_snapshot": {"labels": ["challenge"]},
        "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
    }


def test_collapse_code_classes_keeps_only_representatives():
    runtime = _runtime_stub()
    runtime._verified_rank_key = lambda entry, feedback: feedback.rank_key  # type: ignore[attr-defined]
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]

    a = {"digest": "a", "text": "A", "score": 0.7, "review": 0.7}
    b = {"digest": "b", "text": "B", "score": 0.8, "review": 0.8}
    c = {"digest": "c", "text": "C", "score": 0.9, "review": 0.9}
    pool = [
        (a, _code_eval(passed=1, total=2, failure_kind="visible_test_failure")),
        (b, _code_eval(passed=1, total=2, failure_kind="visible_test_failure")),
        (c, _code_eval(passed=2, total=2)),
    ]

    classes = runtime._collapse_code_classes(pool, anchor_digest="a")

    assert len(classes) == 2
    assert classes[0]["representative"]["digest"] == "c"
    assert classes[1]["size"] == 2
    assert classes[1]["contains_anchor"] is True


def test_collapse_code_classes_does_not_prefer_larger_duplicate_class():
    runtime = _runtime_stub()
    runtime._verified_rank_key = lambda entry, feedback: feedback.rank_key  # type: ignore[attr-defined]
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]

    pool = [
        ({"digest": "a1", "text": "A1", "score": 0.3, "review": 0.3}, _code_eval(passed=1, total=2, failure_kind="kind_a")),
        ({"digest": "a2", "text": "A2", "score": 0.2, "review": 0.2}, _code_eval(passed=1, total=2, failure_kind="kind_a")),
        ({"digest": "b1", "text": "B1", "score": 0.9, "review": 0.9}, _code_eval(passed=1, total=2, failure_kind="kind_b")),
    ]

    classes = runtime._collapse_code_classes(pool, anchor_digest="")

    assert classes[0]["representative"]["digest"] == "b1"


def test_code_route_mode_is_discrete_and_becomes_full_under_two_survivors():
    runtime = _runtime_stub()
    anchor_pair = ({"digest": "anchor"}, _code_eval(passed=1, total=3, failure_kind="visible_test_failure"))
    challenger_classes = [
        {"feedback": _code_eval(passed=2, total=3, failure_kind="visible_test_failure")},
        {"feedback": _code_eval(passed=1, total=3, failure_kind="timeout")},
    ]

    mode = runtime._code_mode(anchor_pair=anchor_pair, challenger_classes=challenger_classes, budget_bucket="normal")

    assert mode == "full"


def test_code_route_mode_keeps_anchor_recovery_when_no_survivors_exist():
    runtime = _runtime_stub()
    anchor_pair = ({"digest": "anchor"}, _code_eval(passed=1, total=3, failure_kind="visible_test_failure"))

    mode = runtime._code_mode(anchor_pair=anchor_pair, challenger_classes=[], budget_bucket="normal")

    assert mode == "lean"


def test_sparse_activate_filters_roles_and_keeps_focus_neighbourhood():
    runtime = _runtime_stub()
    runtime._v4_4_route_family = "code_repair"
    runtime._v4_4_execution_mode_hint = "lean"
    runtime._v4_4_focus_node_ids = {"verifier_node"}

    nodes = {
        "root": UnionNode("root", "planner", "router", "task", [], 1, 1.0),
        "solver": UnionNode("solver", "coder", "solver", "task", [], 1, 1.0),
        "solver_b": UnionNode("solver_b", "coder2", "solver_b", "task", [], 1, 1.0),
        "verifier_node": UnionNode("verifier_node", "verifier", "verifier", "task", [], 1, 1.0),
        "sink": UnionNode("sink", "summarizer", "aggregator", "task", [], 1, 1.0),
    }
    graph = UnionGraph(
        nodes=nodes,
        edges=[],
        source_topology_signatures=[],
        root_node_ids=["root"],
        sink_node_ids=["sink"],
    )
    task_nodes = list(nodes.values())
    active_edges = [
        EdgeActivation("e1", "solver", "verifier_node", 0.9, True, "test"),
        EdgeActivation("e2", "solver_b", "verifier_node", 0.9, True, "test"),
        EdgeActivation("e3", "verifier_node", "sink", 0.9, True, "test"),
    ]

    active = runtime._active_task_nodes_v2(graph, task_nodes, active_edges)
    active_ids = {node.node_id for node in active}

    assert "solver" in active_ids
    assert "verifier_node" in active_ids
    assert "sink" in active_ids
    assert "solver_b" not in active_ids


def test_sink_guards_and_protected_ids_are_explicit_sets():
    runtime = _runtime_stub()
    runtime._v4_4_route_family = "code_repair"
    runtime._v4_4_execution_mode_hint = "lean"
    runtime._v4_4_focus_node_ids = {"solver"}
    runtime._v4_4_champion_provenance_ids = {"solver"}

    nodes = {
        "solver": UnionNode("solver", "coder", "solver", "task", [], 1, 1.0),
        "verifier_node": UnionNode("verifier_node", "verifier", "verifier", "task", [], 1, 1.0),
        "guard": UnionNode("guard", "agg", "aggregator", "task", [], 1, 1.0),
        "sink": UnionNode("sink", "summarizer", "aggregator", "task", [], 1, 1.0),
    }
    graph = UnionGraph(
        nodes=nodes,
        edges=[],
        source_topology_signatures=[],
        root_node_ids=[],
        sink_node_ids=["sink"],
    )
    task_nodes = list(nodes.values())
    active_edges = [
        EdgeActivation("e1", "solver", "verifier_node", 0.9, True, "test"),
        EdgeActivation("e2", "guard", "sink", 0.9, True, "test"),
    ]

    active = runtime._active_task_nodes_v2(graph, task_nodes, active_edges)
    active_ids = {node.node_id for node in active}

    assert runtime._v4_4_sink_guard_ids == {"sink", "guard"}
    assert runtime._v4_4_protected_ids == {"sink", "solver", "verifier_node"}
    assert {"sink", "guard", "solver", "verifier_node"}.issubset(active_ids)


def test_reasoning_ach_lexicographic_prefers_non_fatal_candidate():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "stage2"))  # type: ignore[attr-defined]
    runtime._calibrate_challenger_against_anchor = lambda **kwargs: {  # type: ignore[attr-defined]
        "rounds": 3,
        "challenger_votes": 0,
        "probability": 0.0,
        "override": False,
    }

    anchor = {"digest": "anchor", "text": "I am not sure", "score": 0.4, "source": "anchor"}
    challenger = {"digest": "c1", "text": "Answer: 42", "score": 0.7, "source": "stage2", "sink_support": 1}
    profile = SimpleNamespace(task_type="numeric", name="gsm8k")

    selected, reason, extra = runtime._select_reasoning_against_anchor_v44(
        question_text="What is 40 + 2?",
        metadata={},
        dataset_profile=profile,
        candidates=[anchor, challenger],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected == challenger
    assert reason == "v4_4_reasoning_override_ach_lexicographic"
    assert extra["v4_4_stage1_anchor_used"] is False
    assert extra["v4_4_selected_fatal_count"] == 0


def test_collapse_reasoning_classes_does_not_use_occurrence_or_size_bias():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._evaluate_reasoning_candidate = lambda **kwargs: ReasoningEval(  # type: ignore[attr-defined]
        normalized_answer=str(kwargs["entry"]["digest"]),
        fatal_contradictions=(),
        major_contradictions=(),
        critical_support=1,
        contradiction_cluster="stable",
    )

    classes = runtime._collapse_reasoning_classes(
        question_text="q",
        candidates=[
            {"digest": "majority_a1", "text": "A1", "score": 0.2, "review": 0.2, "occurrence_count": 9},
            {"digest": "majority_a2", "text": "A2", "score": 0.1, "review": 0.1, "occurrence_count": 9},
            {"digest": "minority_b", "text": "B", "score": 0.9, "review": 0.9, "occurrence_count": 1},
        ],
        anchor_digest="",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="numeric"),
    )

    assert classes[0]["representative"]["digest"] == "minority_b"


def test_collapse_graph_classes_does_not_prefer_larger_duplicate_class():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._evaluate_graph_candidate = lambda **kwargs: GraphConstraintEval(  # type: ignore[attr-defined]
        normalized_structure=str(kwargs["entry"]["digest"]),
        broken_blocks=(),
        fatal_blocks=(),
        repair_locus="stable",
        verified_blocks=1,
    )

    classes = runtime._collapse_graph_classes(
        question_text="q",
        candidates=[
            {"digest": "majority_a1", "text": "A1", "score": 0.2, "review": 0.2},
            {"digest": "majority_a2", "text": "A2", "score": 0.1, "review": 0.1},
            {"digest": "minority_b", "text": "B", "score": 0.9, "review": 0.9},
        ],
        anchor_digest="",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="graph_reasoning"),
    )

    assert classes[0]["representative"]["digest"] == "minority_b"


def test_best_code_recovery_target_prefers_recoverable_champion():
    runtime = _runtime_stub()
    champion = {"digest": "champion", "text": "champion"}
    champion_feedback = _code_eval(passed=1, total=3, failure_kind="visible_test_failure")
    anchor_pair = (
        {"digest": "anchor", "text": "anchor"},
        _code_eval(passed=1, total=4, failure_kind="visible_test_failure"),
    )

    target = runtime._best_code_recovery_target(
        champion_entry=champion,
        champion_feedback=champion_feedback,
        anchor_pair=anchor_pair,
    )

    assert target is not None
    assert target[0] == champion
    assert target[2] == "champion"


def test_best_code_recovery_target_falls_back_to_anchor():
    runtime = _runtime_stub()
    champion = {"digest": "champion", "text": "champion"}
    champion_feedback = _code_eval(passed=0, total=0, failure_kind="no_dataset_tests")
    anchor_pair = (
        {"digest": "anchor", "text": "anchor"},
        _code_eval(passed=1, total=4, failure_kind="visible_test_failure"),
    )

    target = runtime._best_code_recovery_target(
        champion_entry=champion,
        champion_feedback=champion_feedback,
        anchor_pair=anchor_pair,
    )

    assert target is not None
    assert target[0]["digest"] == "anchor"
    assert target[2] == "anchor"


def test_best_code_recovery_target_returns_none_without_recoverable_failure():
    runtime = _runtime_stub()
    champion = {"digest": "champion", "text": "champion"}
    champion_feedback = _code_eval(passed=2, total=2)
    anchor_pair = (
        {"digest": "anchor", "text": "anchor"},
        _code_eval(passed=4, total=4),
    )

    target = runtime._best_code_recovery_target(
        champion_entry=champion,
        champion_feedback=champion_feedback,
        anchor_pair=anchor_pair,
    )

    assert target is None


def test_recovery_entry_invariant_requires_reinsert_and_provenance():
    runtime = _runtime_stub()
    branch = _recovery_branch_entry()

    try:
        runtime._assert_recovery_entry_invariants(branch)
    except ValueError as exc:
        assert "reinserted" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected recovery invariant failure")


def test_inspector_rejects_partial_code_promotion():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]
    runtime._generate_repair_branches = lambda **kwargs: [  # type: ignore[attr-defined]
        (
            {
                "digest": "repair",
                "text": "patched",
                "score": 0.9,
                "review": 0.9,
                "source": "code_repair_branch",
                "repair_branch": True,
            },
            _code_eval(passed=2, total=4, failure_kind="visible_test_failure"),
        )
    ]
    runtime._inspect_code_promotion = lambda **kwargs: (False, "not fixed enough")  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "A", "score": 0.7, "review": 0.7, "source": "anchor"}
    seed = {"digest": "seed", "text": "B", "score": 0.6, "review": 0.6, "source": "stage2"}
    anchor_pair = (anchor, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    seed_pair = (seed, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair, seed_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, _, extra = runtime._select_code_repair_against_anchor_v44(
        question_text="q",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="code_generation", name="mbpp"),
        candidates=[seed],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected == anchor
    assert extra["v4_4_stage1_anchor_used"] is True
    assert extra["v4_4_inspector_approval_count"] == 0


def test_inspector_allows_partial_code_promotion():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]
    runtime._generate_repair_branches = lambda **kwargs: [  # type: ignore[attr-defined]
        (
            _recovery_branch_entry(),
            _code_eval(passed=2, total=4, failure_kind="visible_test_failure"),
        )
    ]
    runtime._inspect_code_promotion = lambda **kwargs: (True, "primary failure repaired")  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "A", "score": 0.7, "review": 0.7, "source": "anchor"}
    seed = {"digest": "seed", "text": "B", "score": 0.6, "review": 0.6, "source": "stage2"}
    anchor_pair = (anchor, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    seed_pair = (seed, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair, seed_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor_v44(
        question_text="q",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="code_generation", name="mbpp"),
        candidates=[seed],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected["digest"] == "repair"
    assert reason == "v4_4_code_reinsert_recollapse"
    assert extra["v4_4_selected_is_repair_branch"] is True
    assert extra["v4_4_inspector_approval_count"] == 1
    assert extra["v4_4_reinserted_recovery_count"] == 1


def test_graph_faithfulness_metrics_are_logged():
    runtime = _runtime_stub()
    runtime._last_candidate_bundle = {
        "candidates_serialized": [
            {
                "digest": "repair",
                "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
                "candidate_bank_source": "recovery_output",
                "recovery_subgraph_node_ids": ["solver", "sink"],
            }
        ]
    }
    runtime._last_v4_4_selection = {"v4_4_selected_candidate_digest": "repair", "v4_4_selected_candidate_source": "code_repair_branch"}
    result = Stage2RunResult(
        final_answer="patched",
        final_controller_state=ControllerState(
            turn_index=0,
            mode="lean",
            focus="",
            uncertainty=0.1,
            role_weights={},
            summary="state",
        ),
        turn_traces=[
            TurnTrace(
                turn_index=0,
                controller_state=ControllerState(
                    turn_index=0,
                    mode="lean",
                    focus="",
                    uncertainty=0.1,
                    role_weights={},
                    summary="state",
                ),
                active_edges=[
                    EdgeActivation("e1", "solver", "sink", 0.9, True, "test"),
                    EdgeActivation("e2", "other", "sink", 0.1, False, "test"),
                ],
                node_traces=[],
                feedback_events=[],
                sink_outputs={},
                metadata={"active_node_ids": ["solver", "sink"], "skipped_node_ids": ["other"]},
            )
        ],
        memory_record_counts={},
        signature="sig",
        metadata={},
    )

    metrics = runtime._graph_faithfulness_metrics(result)

    assert metrics["graph_faithfulness_candidate_provenance_coverage"] == 1.0
    assert metrics["graph_faithfulness_final_answer_source_type"] == "recovery_output"
    assert metrics["graph_faithfulness_recovery_subgraph_size"] == 2
