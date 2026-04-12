from __future__ import annotations

import json
from pathlib import Path

import pytest

from stage2_gcr_plus.orchestration.gate import check_mbpp_completion
from stage2_gcr_plus.orchestration.routing import BackendRouter, BackendTarget, parse_backend_specs
from stage2_gcr_plus.orchestration.runner import resolve_execution_mode
from stage2_gcr_plus.orchestration.sharding import build_sample_shards


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_parse_backend_specs_supports_name_url_weight():
    parsed = parse_backend_specs([
        "chat_a=http://127.0.0.1:8101@1.5",
        "http://127.0.0.1:8102",
    ])

    assert len(parsed) == 2
    assert parsed[0].name == "chat_a"
    assert parsed[0].base_url == "http://127.0.0.1:8101"
    assert parsed[0].weight == 1.5
    assert parsed[1].name == "backend_1"


def test_backend_router_prefers_lower_active_load():
    router = BackendRouter(
        [
            BackendTarget(name="a", base_url="http://127.0.0.1:8101", weight=1.0),
            BackendTarget(name="b", base_url="http://127.0.0.1:8102", weight=1.0),
        ]
    )
    router.start_request("a")

    chosen = router.choose_backend()
    assert chosen.name == "b"


def test_backend_router_uses_unhealthy_pool_when_all_down():
    router = BackendRouter(
        [
            BackendTarget(name="a", base_url="http://127.0.0.1:8101", weight=1.0),
            BackendTarget(name="b", base_url="http://127.0.0.1:8102", weight=1.0),
        ]
    )
    router.mark_backend_health("a", False, reason="boom")
    router.mark_backend_health("b", False, reason="boom")

    chosen = router.choose_backend(allow_unhealthy=True)
    assert chosen.name in {"a", "b"}


def test_mbpp_completion_gate_reads_suite_progress(tmp_path: Path):
    progress = {
        "datasets": {
            "mbpp": {
                "status": "completed",
            }
        }
    }
    progress_path = tmp_path / "suite_progress.json"
    progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    status = check_mbpp_completion(progress_path)
    assert status.ready is True
    assert status.status == "completed"


def test_resolve_execution_mode_auto_switches_to_3x_when_ready():
    assert resolve_execution_mode("auto", mbpp_ready=True) == "3x"
    assert resolve_execution_mode("auto", mbpp_ready=False) == "tp4"


def test_resolve_execution_mode_rejects_unknown_mode():
    with pytest.raises(ValueError):
        resolve_execution_mode("4x", mbpp_ready=True)


def test_build_sample_shards_preserves_ids_and_balances(tmp_path: Path):
    data_root = tmp_path / "data"
    dataset_name = "mbpp"
    train_rows = [{"id": f"t{i}", "question": "q", "answer": "a"} for i in range(10)]
    validation_rows = [{"id": f"v{i}", "question": "q", "answer": "a"} for i in range(6)]
    test_rows = [{"id": f"x{i}", "question": "q", "answer": "a"} for i in range(5)]

    _write_jsonl(data_root / dataset_name / "train.jsonl", train_rows)
    _write_jsonl(data_root / dataset_name / "validation.jsonl", validation_rows)
    _write_jsonl(data_root / dataset_name / "test.jsonl", test_rows)

    shard_root = tmp_path / "shards"
    shards = build_sample_shards(
        data_root=data_root,
        datasets=[dataset_name],
        shard_root=shard_root,
        shard_count=4,
        seed=7,
    )

    seen_train_ids: list[str] = []
    shard_train_sizes: list[int] = []
    for spec in shards:
        train_path = spec.data_root / dataset_name / "train.jsonl"
        rows = [json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        seen_train_ids.extend(str(row["id"]) for row in rows)
        shard_train_sizes.append(len(rows))

    assert sorted(seen_train_ids) == sorted(str(row["id"]) for row in train_rows)
    assert max(shard_train_sizes) - min(shard_train_sizes) <= 1
