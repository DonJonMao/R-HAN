from __future__ import annotations

from types import SimpleNamespace

import pytest

from mas_stage2.config import Stage2MemoryConfig
from mas_stage2.learning import OnlineLinearModel
from stage2_gcr_plus.memory import LocalMemoryComposer, PrivateEpisodeMemoryStore, RoleAwareMemorySelector
from mas_stage2.types import MemoryRecord, SelectedMemoryItem
from mas_treesearch.types import UnionNode


def _record(
    *,
    record_id: str,
    owner_node_id: str = "node-1",
    record_type: str,
    feedback_type: str = "unresolved",
) -> MemoryRecord:
    return MemoryRecord(
        record_id=record_id,
        episode_id="ep-1",
        turn_index=0,
        owner_node_id=owner_node_id,
        agent_id="agent",
        role="solver",
        record_type=record_type,
        text=record_id,
        embedding=[0.0, 1.0],
        token_estimate=2,
        feedback_type=feedback_type,
    )


def test_private_memory_store_exposes_bucket_and_view_helpers():
    store = PrivateEpisodeMemoryStore(Stage2MemoryConfig(max_private_records_per_agent=8))
    store.add(_record(record_id="self", record_type="self_output"))
    store.add(_record(record_id="pass", record_type="feedback", feedback_type="pass"))
    store.add(_record(record_id="reject", record_type="feedback", feedback_type="reject"))
    store.add(_record(record_id="repair", record_type="repair_trace"))

    assert [item.record_id for item in store.get_bucket("node-1", "self_output")] == ["self"]
    assert [item.record_id for item in store.get_feedback_view("node-1", "stable_view")] == ["pass"]
    assert [item.record_id for item in store.get_feedback_view("node-1", "failure_view")] == ["reject"]
    checker_summary = store.get_view("node-1", "checker_verdict_summary")
    recovery_summary = store.get_view("node-1", "recovery_summary")

    assert [item.record_id for item in checker_summary] == ["node-1::checker_verdict_summary"]
    assert checker_summary[0].metadata["summary_view"] is True
    assert [item.record_id for item in recovery_summary] == ["node-1::recovery_summary"]
    assert recovery_summary[0].metadata["summary_view"] is True


def test_private_memory_store_rejects_unknown_physical_bucket():
    store = PrivateEpisodeMemoryStore(Stage2MemoryConfig(max_private_records_per_agent=8))

    with pytest.raises(ValueError):
        store.add(_record(record_id="bad", record_type="ad_hoc_bucket"))


class _Embedder:
    def embed(self, text: str) -> list[float]:
        if "slot=class_summary" in text:
            return [1.0, 0.0]
        if "slot=checker_verdict_summary" in text:
            return [0.0, 1.0]
        if "slot=recovery_summary" in text:
            return [1.0, 1.0]
        return [1.0, 0.0]


def test_role_aware_selector_uses_typed_slot_plan():
    selector = RoleAwareMemorySelector(Stage2MemoryConfig(max_selected_records=4), _Embedder())
    node = UnionNode("sink", "sink_agent", "aggregator", "task", [], 1, 1.0, metadata={"runtime_node_type": "sink"})
    records = [
        _record(record_id="class", owner_node_id="sink", record_type="class_summary"),
        _record(record_id="pass", owner_node_id="sink", record_type="feedback", feedback_type="pass"),
        _record(record_id="repair", owner_node_id="sink", record_type="repair_trace"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        records,
        current_turn=1,
    )

    rationales = {item.rationale for item in selected}
    record_ids = {item.record_id for item in selected}

    assert "slot:class_summary" in rationales
    assert "slot:checker_verdict_summary" in rationales
    assert "slot:recovery_summary" in rationales
    assert "sink::checker_verdict_summary" in record_ids
    assert "sink::recovery_summary" in record_ids


def test_runtime_node_type_metadata_overrides_role_only_slot_plan():
    selector = RoleAwareMemorySelector(Stage2MemoryConfig(max_selected_records=4), _Embedder())
    node = UnionNode("solver", "solver_agent", "aggregator", "task", [], 1, 1.0, metadata={"runtime_node_type": "proposal"})
    records = [
        _record(record_id="self", owner_node_id="solver", record_type="self_output"),
        _record(record_id="pass", owner_node_id="solver", record_type="feedback", feedback_type="pass"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        records,
        current_turn=1,
    )

    rationales = {item.rationale for item in selected}
    assert "slot:self_output" in rationales
    assert "slot:feedback" in rationales
    assert "slot:checker_verdict_summary" not in rationales
    assert node.metadata["runtime_node_type_source"] == "runtime_annotation"


def test_role_aware_selector_uses_slot_sparsemax_support_set_not_slot_cap_topr():
    selector = RoleAwareMemorySelector(Stage2MemoryConfig(max_selected_records=4), _Embedder())
    node = UnionNode("solver", "solver_agent", "solver", "task", [], 1, 1.0, metadata={"runtime_node_type": "proposal"})
    records = [
        _record(record_id="feedback-a", owner_node_id="solver", record_type="feedback", feedback_type="neutral"),
        _record(record_id="feedback-b", owner_node_id="solver", record_type="feedback", feedback_type="neutral"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        records,
        current_turn=1,
    )

    feedback_items = [item for item in selected if item.rationale == "slot:feedback"]
    feedback_ids = {item.record_id for item in feedback_items}

    assert feedback_ids == {"feedback-a", "feedback-b"}
    assert all(item.metadata["slot_name"] == "feedback" for item in feedback_items)
    assert all(item.metadata["slot_schema_cap"] == 1 for item in feedback_items)
    assert all(item.metadata["slot_sparsemax_weight"] > 0.0 for item in feedback_items)
    assert all(item.metadata["features"]["slot=feedback"] == 1.0 for item in feedback_items)
    assert all("selection_order" in item.metadata for item in feedback_items)


def test_role_aware_selector_round_robins_across_slots_before_global_budget():
    selector = RoleAwareMemorySelector(Stage2MemoryConfig(max_selected_records=2), _Embedder())
    node = UnionNode("solver", "solver_agent", "solver", "task", [], 1, 1.0, metadata={"runtime_node_type": "proposal"})
    records = [
        _record(record_id="self-a", owner_node_id="solver", record_type="self_output"),
        _record(record_id="self-b", owner_node_id="solver", record_type="self_output"),
        _record(record_id="pass", owner_node_id="solver", record_type="feedback", feedback_type="pass"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        records,
        current_turn=1,
    )

    selected_ids = [item.record_id for item in selected]
    rationales = [item.rationale for item in selected]

    assert len(selected) == 2
    assert rationales == ["slot:self_output", "slot:feedback"]
    assert selected_ids[1] == "pass"
    assert [item.metadata["selection_order"] for item in selected] == [0, 1]


def test_role_aware_selector_applies_soft_slot_gate_without_hard_pruning_optional_slot():
    gate_model = OnlineLinearModel()
    gate_model.bias = 0.25
    selector = RoleAwareMemorySelector(
        Stage2MemoryConfig(max_selected_records=4, slot_mask_mode="soft"),
        _Embedder(),
        slot_gate_model=gate_model,
        slot_gate_weight=1.0,
    )
    node = UnionNode("sink", "sink_agent", "aggregator", "task", [], 1, 1.0, metadata={"runtime_node_type": "sink"})
    records = [
        _record(record_id="class", owner_node_id="sink", record_type="class_summary"),
        _record(record_id="pass", owner_node_id="sink", record_type="feedback", feedback_type="pass"),
        _record(record_id="repair", owner_node_id="sink", record_type="repair_trace"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.8, mode="lean", role_weights={}, metadata={"challenge_count": 2}),
        records,
        current_turn=2,
    )

    by_slot = {item.metadata["slot_name"]: item for item in selected}

    assert by_slot["class_summary"].metadata["slot_core"] is True
    assert by_slot["class_summary"].metadata["slot_gate_weight"] == 1.0
    assert by_slot["recovery_summary"].metadata["slot_core"] is False
    assert 0.0 < by_slot["recovery_summary"].metadata["slot_gate_weight"] < 1.0
    assert by_slot["recovery_summary"].metadata["effective_weight"] == pytest.approx(
        by_slot["recovery_summary"].metadata["slot_gate_weight"]
        * by_slot["recovery_summary"].metadata["slot_support_weight"]
    )
    assert "slot_gate_features" in by_slot["recovery_summary"].metadata
    assert "mode::lean" not in by_slot["recovery_summary"].metadata["slot_gate_features"]


def test_fixed_slot_mask_mode_keeps_optional_slots_at_full_gate():
    gate_model = OnlineLinearModel()
    gate_model.bias = 0.1
    selector = RoleAwareMemorySelector(
        Stage2MemoryConfig(max_selected_records=4, slot_mask_mode="fixed"),
        _Embedder(),
        slot_gate_model=gate_model,
        slot_gate_weight=1.0,
    )
    node = UnionNode("sink", "sink_agent", "aggregator", "task", [], 1, 1.0, metadata={"runtime_node_type": "sink"})
    records = [_record(record_id="repair", owner_node_id="sink", record_type="repair_trace")]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.8, mode="lean", role_weights={}, metadata={}),
        records,
        current_turn=2,
    )

    recovery_item = next(item for item in selected if item.metadata["slot_name"] == "recovery_summary")
    assert recovery_item.metadata["slot_gate_weight"] == 1.0


def test_local_memory_composer_preserves_selection_order_instead_of_global_score_sort():
    composer = LocalMemoryComposer(Stage2MemoryConfig(max_selected_records=4), _Embedder())
    node = UnionNode("solver", "solver_agent", "solver", "task", [], 1, 1.0, metadata={})
    records_by_id = {
        "low-score-first": _record(record_id="low-score-first", record_type="feedback", feedback_type="pass"),
        "high-score-second": _record(record_id="high-score-second", record_type="feedback", feedback_type="reject"),
    }
    selected = [
        SelectedMemoryItem(
            record_id="low-score-first",
            score=0.1,
            rationale="slot:feedback",
            metadata={"slot_name": "feedback", "support_weight": 0.1, "selection_order": 0},
        ),
        SelectedMemoryItem(
            record_id="high-score-second",
            score=0.9,
            rationale="slot:failure_view",
            metadata={"slot_name": "failure_view", "support_weight": 0.9, "selection_order": 1},
        ),
    ]

    latent = composer.compose(
        node,
        records_by_id,
        selected,
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        current_turn=1,
    )

    assert [item.record_id for item in latent.selected_items] == ["low-score-first", "high-score-second"]
    assert latent.metadata["selected_record_ids"] == ["low-score-first", "high-score-second"]
    assert "[feedback | alpha=0.100]" in latent.summary


def test_role_aware_selector_marks_compat_fallback_when_runtime_type_missing():
    selector = RoleAwareMemorySelector(Stage2MemoryConfig(max_selected_records=4), _Embedder())
    node = UnionNode("router", "router_agent", "router", "task", [], 1, 1.0, metadata={})
    records = [
        _record(record_id="class", owner_node_id="router", record_type="class_summary"),
        _record(record_id="pass", owner_node_id="router", record_type="feedback", feedback_type="pass"),
        _record(record_id="repair", owner_node_id="router", record_type="repair_trace"),
    ]

    selected = selector.select(
        node,
        "question",
        SimpleNamespace(summary="global", uncertainty=0.1, mode="lean", role_weights={}),
        records,
        current_turn=1,
    )

    rationales = {item.rationale for item in selected}
    assert "slot:class_summary" in rationales
    assert "slot:checker_verdict_summary" in rationales
    assert "slot:recovery_summary" in rationales
    assert node.metadata["runtime_node_type"] == "aggregator"
    assert node.metadata["runtime_node_type_source"] == "compat_fallback"
    assert node.metadata["runtime_node_type_fallback_role"] == "router"
