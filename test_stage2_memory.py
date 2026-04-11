from __future__ import annotations

import pytest

from mas_stage2.config import Stage2MemoryConfig
from mas_stage2.memory import PrivateEpisodeMemoryStore
from mas_stage2.types import MemoryRecord


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
    assert [item.record_id for item in store.get_view("node-1", "checker_verdict_summary")] == ["pass", "reject"]
    assert [item.record_id for item in store.get_view("node-1", "recovery_summary")] == ["repair"]


def test_private_memory_store_rejects_unknown_physical_bucket():
    store = PrivateEpisodeMemoryStore(Stage2MemoryConfig(max_private_records_per_agent=8))

    with pytest.raises(ValueError):
        store.add(_record(record_id="bad", record_type="ad_hoc_bucket"))
