from __future__ import annotations

from types import SimpleNamespace

import pytest

from mas_stage2.config import Stage2MemoryConfig
from mas_stage2.memory import PrivateEpisodeMemoryStore, RoleAwareMemorySelector
from mas_stage2.types import MemoryRecord
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
