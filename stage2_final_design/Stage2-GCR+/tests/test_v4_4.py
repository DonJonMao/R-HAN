from __future__ import annotations

from types import SimpleNamespace

import torch

from mas_stage2.composer import MemoryComposerConfig, SimpleMemoryComposer
from mas_stage2.lmpo import LMPOConfig, LMPOTrainer
from mas_stage2.types import ControllerState, EdgeActivation, FeedbackEvent, TurnTrace, Stage2RunResult
from stage2_gcr_plus.code_repair import CodeRepairEval
from stage2_gcr_plus.runtime_v44 import GraphConstraintEval, ReasoningEval, Stage2RuntimeV44
from mas_treesearch.evaluator import MultiFidelityEvaluator
from mas_treesearch.types import UnionEdge, UnionGraph, UnionNode


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
    runtime.config.graph = SimpleNamespace(
        turn_count=5,
        min_incoming_edges=1,
        soft_prune_top_k=3,
        hard_prune_after_turn=4,
        soft_prune_threshold=0.38,
        support_set_mode="sparsemax",
        edge_support_logit_temperature=2.0,
        edge_support_probability_epsilon=1e-4,
    )
    runtime._v4_4_route_family = "adversarial"
    runtime._v4_4_execution_mode_hint = "lean"
    runtime._v4_4_focus_node_ids = set()
    runtime._v4_4_sink_guard_ids = set()
    runtime._v4_4_protected_ids = set()
    runtime._v4_4_champion_provenance_ids = set()
    runtime.global_node = None
    runtime.gnn = None
    runtime._current_learn = False
    runtime._pending_policy_log_probs = []
    runtime._pending_policy_entropies = []
    runtime._pending_edge_gate_records = []
    runtime.v2_config = SimpleNamespace(edge_gate_aux_enabled=True, edge_gate_aux_loss_weight=1.0)
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


def test_simple_memory_composer_accepts_dense_qwen_embeddings():
    composer = SimpleMemoryComposer(
        MemoryComposerConfig(hidden_dim=8, latent_length=2, encoder_layers=1, dropout=0.0, max_input_length=4),
        vocab_size=32,
    )
    input_embeddings = torch.randn(1, 3, 8)
    attention_mask = torch.ones(1, 3, dtype=torch.bool)

    latent = composer.forward_embeddings(input_embeddings, attention_mask)

    assert latent.shape == (1, 2, 8)


def test_lmpo_updates_from_auxiliary_loss_without_policy_logprob():
    module = torch.nn.Linear(1, 1)
    trainer = LMPOTrainer(module, LMPOConfig(learning_rate=0.01, enabled=True))
    aux_loss = module(torch.ones(1, 1)).sum()

    stats = trainer.update_from_policy([], 0.0, auxiliary_losses=[aux_loss])

    assert stats["policy_updates"] == 1.0
    assert "last_loss" in stats


def test_edge_gate_auxiliary_loss_uses_active_downstream_feedback():
    runtime = _runtime_stub()
    gate = torch.tensor(0.2, requires_grad=True)
    runtime._pending_edge_gate_records = [
        {
            "turn_index": 0,
            "edge_id": "solver->sink",
            "src": "solver",
            "dst": "sink",
            "active": True,
            "gate": gate,
        }
    ]
    state = ControllerState(0, "lean", "focus", 0.1, {}, "summary")
    event = FeedbackEvent(
        event_id="f1",
        turn_index=0,
        source_node_id="verifier",
        target_node_id="sink",
        source_kind="verifier",
        event_type="pass",
        confidence=0.9,
        detail="sink output preserved",
    )
    result = Stage2RunResult(
        final_answer="",
        final_controller_state=state,
        turn_traces=[TurnTrace(0, state, [], [], [event], {})],
        memory_record_counts={},
        signature="sig",
    )

    loss, stats = runtime._edge_gate_auxiliary_loss(result)

    assert loss is not None
    assert stats["edge_aux_terms"] == 1.0
    assert stats["edge_aux_positive_labels"] == 1.0
    loss.backward()
    assert gate.grad is not None


def _code_recovery_graph() -> UnionGraph:
    nodes = {
        "solver": UnionNode("solver", "coder", "solver", "task", ["g_high"], 3, 1.0),
        "checker": UnionNode("checker", "verifier", "verifier", "task", ["g_high"], 2, 1.0),
        "relay": UnionNode("relay", "router", "router", "task", ["g_high"], 2, 1.0),
        "sink": UnionNode("sink", "summarizer", "aggregator", "task", ["g_high"], 3, 1.0),
        "other": UnionNode("other", "coder2", "solver", "task", ["g_low"], 1, 0.4),
    }
    return UnionGraph(
        nodes=nodes,
        edges=[
            UnionEdge("solver", "checker", "task", ["g_high"], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("checker", "relay", "task", ["g_high"], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("relay", "sink", "task", ["g_high"], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("other", "sink", "task", ["g_low"], 1, 1.0, 0.3, 0.3, 0.5, 0.5),
        ],
        source_topology_signatures=["g_low", "g_high"],
        root_node_ids=["solver"],
        sink_node_ids=["sink"],
    )


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


def test_verified_rank_key_ignores_quality_and_review_scores():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]

    shorter = {"digest": "short", "text": "A", "score": 0.1, "review": 0.1, "line_count": 4}
    longer = {"digest": "long", "text": "B", "score": 0.9, "review": 0.9, "line_count": 20}
    feedback = _code_eval(passed=1, total=2, failure_kind="visible_test_failure")

    assert runtime._verified_rank_key(shorter, feedback) > runtime._verified_rank_key(longer, feedback)


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


def test_runtime_node_type_mapping_uses_sink_distance_and_support():
    runtime = _runtime_stub()
    nodes = {
        "proposal": UnionNode("proposal", "solver", "solver", "task", [], 1, 1.0),
        "aggregator": UnionNode("aggregator", "agg", "router", "task", [], 4, 1.0),
        "checker": UnionNode("checker", "verifier", "verifier", "task", [], 2, 1.0),
        "sink": UnionNode("sink", "sink_agent", "aggregator", "task", [], 5, 1.0),
    }
    graph = UnionGraph(
        nodes=nodes,
        edges=[
            UnionEdge("proposal", "aggregator", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("aggregator", "sink", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("checker", "sink", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
        ],
        source_topology_signatures=[],
        root_node_ids=[],
        sink_node_ids=["sink"],
    )

    assert runtime._runtime_node_type(graph, nodes["sink"]) == "sink"
    assert runtime._runtime_node_type(graph, nodes["checker"]) == "checker"
    assert runtime._runtime_node_type(graph, nodes["aggregator"]) == "aggregator"
    assert runtime._runtime_node_type(graph, nodes["proposal"]) == "proposal"


def test_runtime_node_type_promotes_near_sink_router_with_support():
    runtime = _runtime_stub()
    nodes = {
        "proposal": UnionNode("proposal", "solver", "solver", "task", [], 2, 1.0),
        "router": UnionNode("router", "router_agent", "router", "task", [], 4, 1.0),
        "relay": UnionNode("relay", "relay_agent", "solver", "task", [], 5, 1.0),
        "sink": UnionNode("sink", "sink_agent", "aggregator", "task", [], 5, 1.0),
    }
    graph = UnionGraph(
        nodes=nodes,
        edges=[
            UnionEdge("proposal", "router", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("router", "relay", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
            UnionEdge("relay", "sink", "task", [], 1, 1.0, 1.0, 1.0, 0.5, 0.5),
        ],
        source_topology_signatures=[],
        root_node_ids=[],
        sink_node_ids=["sink"],
    )

    assert runtime._runtime_node_type(graph, nodes["relay"]) == "aggregator"
    assert runtime._runtime_node_type(graph, nodes["router"]) == "aggregator"
    assert runtime._runtime_node_type(graph, nodes["proposal"]) == "proposal"


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


def test_reasoning_stabilization_preserves_nonfatal_anchor():
    runtime = _runtime_stub()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "stage2"))  # type: ignore[attr-defined]

    anchor = {"digest": "anchor", "text": "42", "score": 0.2, "source": "anchor"}
    challenger = {"digest": "challenger", "text": "42", "score": 0.9, "source": "stage2", "sink_support": 1}
    profile = SimpleNamespace(task_type="numeric", name="gsm8k")

    selected, reason, extra = runtime._select_reasoning_against_anchor_v44(
        question_text="What is 40 + 2?",
        metadata={},
        dataset_profile=profile,
        candidates=[anchor, challenger],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected == anchor
    assert reason == "v4_4_reasoning_stabilize_preserve_anchor"
    assert extra["v4_4_stage1_anchor_used"] is True


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


def test_collapse_reasoning_classes_prefers_anchor_on_typed_tie():
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
            {"digest": "anchor", "text": "A", "score": 0.1, "review": 0.1, "stage1_anchor": True},
            {"digest": "challenger", "text": "B", "score": 0.9, "review": 0.9},
        ],
        anchor_digest="anchor",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="numeric"),
    )

    assert classes[0]["representative"]["digest"] == "anchor"
    assert classes[0]["contains_anchor"] is True


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


def test_collapse_graph_classes_prefers_anchor_on_typed_tie():
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
            {"digest": "anchor", "text": "A", "score": 0.1, "review": 0.1, "stage1_anchor": True},
            {"digest": "challenger", "text": "B", "score": 0.9, "review": 0.9},
        ],
        anchor_digest="anchor",
        metadata={},
        dataset_profile=SimpleNamespace(task_type="graph_reasoning"),
    )

    assert classes[0]["representative"]["digest"] == "anchor"
    assert classes[0]["contains_anchor"] is True


def test_sparsemax_support_set_prunes_large_margin_edges():
    runtime = _runtime_stub()
    runtime.config.graph.support_set_mode = "sparsemax"
    score_by_src = {"src_a": 0.9, "src_b": 0.8, "src_c": 0.1}
    runtime.gnn = SimpleNamespace(
        edge_gate=lambda src_latent, dst_latent, edge_features, global_state=None: torch.tensor(edge_features[0])
    )
    runtime._edge_feature_vector = lambda edge, turn_index: [score_by_src[edge.src]]  # type: ignore[attr-defined]
    graph = UnionGraph(
        nodes={
            "src_a": UnionNode("src_a", "a", "solver", "task", [], 1, 1.0),
            "src_b": UnionNode("src_b", "b", "solver", "task", [], 1, 1.0),
            "src_c": UnionNode("src_c", "c", "solver", "task", [], 1, 1.0),
            "dst": UnionNode("dst", "d", "verifier", "task", [], 1, 1.0),
        },
        edges=[
            UnionEdge("src_a", "dst", "task", [], 1, 1.0, 0.9, 0.9, 0.5, 0.5),
            UnionEdge("src_b", "dst", "task", [], 1, 1.0, 0.8, 0.8, 0.5, 0.5),
            UnionEdge("src_c", "dst", "task", [], 1, 1.0, 0.1, 0.1, 0.5, 0.5),
        ],
        source_topology_signatures=[],
        root_node_ids=[],
        sink_node_ids=["dst"],
    )
    prepared_states = {node_id: {"local_latent": torch.zeros(1, 1)} for node_id in graph.nodes}

    activations = runtime._activate_edges_v2(graph, prepared_states, turn_index=0)
    active_ids = {item.edge_id for item in activations if item.active}
    metadata = {item.edge_id: item.metadata for item in activations}

    assert active_ids == {"src_a->dst"}
    assert metadata["src_a->dst"]["support_set_mode"] == "sparsemax"
    assert metadata["src_a->dst"]["support_set_weight"] > 0.0
    assert metadata["src_b->dst"]["support_set_weight"] == 0.0
    assert metadata["src_c->dst"]["support_set_weight"] == 0.0


def test_sparsemax_support_uses_logit_margin_not_probability_scale():
    runtime = _runtime_stub()
    runtime.config.graph.support_set_mode = "sparsemax"
    runtime.config.graph.edge_support_logit_temperature = 2.0
    score_by_src = {"src_a": 0.72, "src_b": 0.68, "src_c": 0.61, "src_d": 0.55}
    runtime.gnn = SimpleNamespace(
        edge_gate=lambda src_latent, dst_latent, edge_features, global_state=None: torch.tensor(edge_features[0])
    )
    runtime._edge_feature_vector = lambda edge, turn_index: [score_by_src[edge.src]]  # type: ignore[attr-defined]
    graph = UnionGraph(
        nodes={
            "src_a": UnionNode("src_a", "a", "solver", "task", [], 1, 1.0),
            "src_b": UnionNode("src_b", "b", "solver", "task", [], 1, 1.0),
            "src_c": UnionNode("src_c", "c", "solver", "task", [], 1, 1.0),
            "src_d": UnionNode("src_d", "d", "solver", "task", [], 1, 1.0),
            "dst": UnionNode("dst", "sink", "verifier", "task", [], 1, 1.0),
        },
        edges=[
            UnionEdge(src, "dst", "task", [], 1, 1.0, score_by_src[src], score_by_src[src], 0.5, 0.5)
            for src in ("src_a", "src_b", "src_c", "src_d")
        ],
        source_topology_signatures=[],
        root_node_ids=[],
        sink_node_ids=["dst"],
    )
    prepared_states = {node_id: {"local_latent": torch.zeros(1, 1)} for node_id in graph.nodes}

    activations = runtime._activate_edges_v2(graph, prepared_states, turn_index=0)
    active_ids = {item.edge_id for item in activations if item.active}
    metadata = {item.edge_id: item.metadata for item in activations}

    assert active_ids == {"src_a->dst", "src_b->dst"}
    assert metadata["src_a->dst"]["support_set_logit"] > metadata["src_c->dst"]["support_set_logit"]
    assert metadata["src_c->dst"]["support_set_weight"] == 0.0
    assert metadata["src_d->dst"]["support_set_weight"] == 0.0


def test_best_code_recovery_target_prefers_recoverable_champion():
    runtime = _runtime_stub()
    champion = {
        "digest": "champion",
        "text": "champion",
        "origin_node_id": "solver",
        "origin_turn_index": 0,
        "origin_role": "solver",
        "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
    }
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
        {
            "digest": "anchor",
            "text": "anchor",
            "origin_node_id": "solver",
            "origin_turn_index": 0,
            "origin_role": "solver",
            "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
        },
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


def test_best_code_recovery_target_skips_anchor_without_seed_provenance():
    runtime = _runtime_stub()
    champion = {"digest": "champion", "text": "champion"}
    champion_feedback = _code_eval(passed=0, total=0, failure_kind="no_dataset_tests")
    anchor_pair = (
        {"digest": "anchor", "text": "anchor", "stage1_anchor": True},
        _code_eval(passed=1, total=4, failure_kind="visible_test_failure"),
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
    graph = _code_recovery_graph()
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

    anchor = {
        "digest": "anchor",
        "text": "A",
        "score": 0.7,
        "review": 0.7,
        "source": "anchor",
        "origin_node_id": "solver",
        "origin_turn_index": 0,
        "origin_role": "solver",
        "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
    }
    seed = {"digest": "seed", "text": "B", "score": 0.6, "review": 0.6, "source": "stage2"}
    anchor_pair = (anchor, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    seed_pair = (seed, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair, seed_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, _, extra = runtime._select_code_repair_against_anchor_v44(
        graph=graph,
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
    graph = _code_recovery_graph()
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

    anchor = {
        "digest": "anchor",
        "text": "A",
        "score": 0.7,
        "review": 0.7,
        "source": "anchor",
        "origin_node_id": "solver",
        "origin_turn_index": 0,
        "origin_role": "solver",
        "provenance": [{"node_id": "solver", "turn_index": 0, "role": "solver"}],
    }
    seed = {"digest": "seed", "text": "B", "score": 0.6, "review": 0.6, "source": "stage2"}
    anchor_pair = (anchor, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    seed_pair = (seed, _code_eval(passed=1, total=4, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair, seed_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor_v44(
        graph=graph,
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


def test_anchor_recovery_seed_is_legalized_from_best_original_graph_prior():
    runtime = _runtime_stub()
    graph = _code_recovery_graph()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]
    runtime._generate_repair_branches = lambda **kwargs: []  # type: ignore[attr-defined]

    anchor = {
        "digest": "anchor",
        "text": "def solve(x):\n    return x\n",
        "score": 0.8,
        "review": 0.8,
        "source": "anchor",
        "stage1_anchor": True,
    }
    anchor_pair = (anchor, _code_eval(passed=1, total=3, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor_v44(
        graph=graph,
        question_text="q",
        metadata={
            "stage1_selected_topology_signatures": ["g_low", "g_high"],
            "stage1_selected_topology_scores": [0.2, 0.9],
        },
        dataset_profile=SimpleNamespace(task_type="code_generation", name="mbpp"),
        candidates=[],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected == anchor
    assert reason in {"v4_4_code_preserve_anchor_after_class_collapse", "v4_4_code_anchor_guard_preserve"}
    assert extra["v4_4_anchor_recovery_seed_legalized"] is True
    assert extra["v4_4_anchor_recovery_seed_binding_kind"] == "best_original_graph_prior"
    assert extra["v4_4_anchor_recovery_seed_source_graph"] == "g_high"
    assert extra["v4_4_anchor_recoverable_but_blocked_no_provenance_count"] == 0
    assert anchor["origin_node_id"] == "solver"
    assert anchor["stage1_anchor_binding_legalized"] is True
    assert anchor["anchor_bound_node_ids"] == ["checker", "relay", "sink", "solver"]
    assert anchor["anchor_bound_edge_ids"] == ["checker->relay", "relay->sink", "solver->checker"]
    assert [item["source_graph_id"] for item in anchor["provenance"]] == ["g_high", "g_high", "g_high", "g_high"]


def test_anchor_recovery_seed_blocked_count_is_recorded_when_prior_binding_fails():
    runtime = _runtime_stub()
    graph = _code_recovery_graph()
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]
    runtime._review_consensus = lambda entry: float(entry.get("review", 0.0))  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", "seed"))  # type: ignore[attr-defined]

    anchor = {
        "digest": "anchor",
        "text": "def solve(x):\n    return x\n",
        "score": 0.8,
        "review": 0.8,
        "source": "anchor",
        "stage1_anchor": True,
    }
    anchor_pair = (anchor, _code_eval(passed=1, total=3, failure_kind="visible_test_failure"))
    runtime._verify_code_pool = lambda **kwargs: ([anchor_pair], anchor_pair)  # type: ignore[attr-defined]

    selected, reason, extra = runtime._select_code_repair_against_anchor_v44(
        graph=graph,
        question_text="q",
        metadata={"stage1_selected_topology_signatures": ["missing_graph"], "stage1_selected_topology_scores": [1.0]},
        dataset_profile=SimpleNamespace(task_type="code_generation", name="mbpp"),
        candidates=[],
        anchor=anchor,
        budget_bucket="normal",
    )

    assert selected == anchor
    assert reason in {"v4_4_code_preserve_anchor_after_class_collapse", "v4_4_code_anchor_guard_preserve"}
    assert extra["v4_4_anchor_recovery_seed_legalized"] is False
    assert extra["v4_4_anchor_recoverable_but_blocked_no_provenance_count"] == 1


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


def test_finalize_answer_code_route_uses_graph_from_run_context():
    runtime = _runtime_stub()
    graph = _code_recovery_graph()
    runtime._candidate_bank_bundle = lambda **kwargs: {  # type: ignore[attr-defined]
        "candidates": [],
        "candidates_serialized": [],
        "anchor": {"digest": "anchor"},
        "anchor_serialized": {"digest": "anchor"},
    }
    runtime._route_family = lambda dataset_profile, metadata: "code_repair"  # type: ignore[attr-defined]
    runtime._budget_bucket = lambda metadata: "normal"  # type: ignore[attr-defined]
    runtime._select_code_repair_against_anchor_v44 = lambda **kwargs: (  # type: ignore[attr-defined]
        {"digest": "anchor", "text": "patched"},
        "v4_4_code_anchor_guard_preserve",
        {"v4_4_selected_candidate_digest": "anchor"},
    )
    runtime._record_selection_metadata = lambda **kwargs: None  # type: ignore[attr-defined]
    runtime._assert_recovery_entry_invariants = lambda entry: None  # type: ignore[attr-defined]
    runtime._serialize_candidate_entry = lambda entry: dict(entry)  # type: ignore[attr-defined]
    runtime._candidate_source_label = lambda entry: str(entry.get("source", ""))  # type: ignore[attr-defined]
    runtime._quality_score = lambda entry: float(entry.get("score", 0.0))  # type: ignore[attr-defined]

    runtime._v4_4_current_graph = graph
    try:
        final_answer, strategy = runtime._finalize_answer(
            question_text="q",
            controller_state=SimpleNamespace(),
            sink_outputs={},
            turn_traces=[],
            metadata={},
            reference_answer=None,
            dataset_profile=SimpleNamespace(task_type="code_generation", name="mbpp"),
        )
    finally:
        runtime._v4_4_current_graph = None

    assert final_answer == "patched"
    assert strategy == "v4_4_code_anchor_guard_preserve"
