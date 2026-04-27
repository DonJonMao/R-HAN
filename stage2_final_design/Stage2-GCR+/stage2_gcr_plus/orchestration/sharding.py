from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from mas_treesearch.data import filter_stage2_supported_items


@dataclass(frozen=True)
class ShardSpec:
    shard_index: int
    data_root: Path
    counts: Dict[str, Dict[str, int]]


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if isinstance(payload, dict):
                rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _round_robin_assign(items: Sequence[Dict[str, Any]], shard_count: int, seed: int) -> list[list[Dict[str, Any]]]:
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    indices = list(range(len(items)))
    random.Random(seed).shuffle(indices)
    shards: list[list[Dict[str, Any]]] = [[] for _ in range(shard_count)]
    for rank, idx in enumerate(indices):
        shard_id = rank % shard_count
        shards[shard_id].append(items[idx])
    return shards


def build_sample_shards(
    *,
    data_root: str | Path,
    datasets: Sequence[str],
    shard_root: str | Path,
    shard_count: int,
    seed: int = 7,
    splits: Sequence[str] = ("train", "validation", "test"),
) -> list[ShardSpec]:
    source_root = Path(data_root)
    out_root = Path(shard_root)
    if shard_count <= 0:
        raise ValueError("shard_count must be > 0")
    if not datasets:
        raise ValueError("datasets must not be empty")

    counts_by_shard: list[dict[str, dict[str, int]]] = [dict() for _ in range(shard_count)]
    filter_summary: dict[str, dict[str, dict[str, int | str]]] = {}

    for dataset_index, dataset_name in enumerate(datasets):
        dataset_dir = source_root / dataset_name
        if not dataset_dir.exists():
            raise FileNotFoundError(f"dataset directory not found: {dataset_dir}")
        for split_index, split in enumerate(splits):
            source_file = dataset_dir / f"{split}.jsonl"
            source_rows = _load_jsonl(source_file)
            rows = filter_stage2_supported_items(dataset_name, source_rows)
            dropped = len(source_rows) - len(rows)
            if dropped:
                filter_summary.setdefault(dataset_name, {})[split] = {
                    "filter": "stage2_supported_reference_answer",
                    "source": len(source_rows),
                    "kept": len(rows),
                    "dropped": dropped,
                }
            shard_rows = _round_robin_assign(rows, shard_count=shard_count, seed=seed + dataset_index * 100 + split_index)
            for shard_id, picked in enumerate(shard_rows):
                target_file = out_root / f"shard_{shard_id}" / dataset_name / f"{split}.jsonl"
                _write_jsonl(target_file, picked)
                dataset_counts = counts_by_shard[shard_id].setdefault(dataset_name, {})
                dataset_counts[split] = len(picked)

    shard_specs = [
        ShardSpec(
            shard_index=shard_id,
            data_root=out_root / f"shard_{shard_id}",
            counts=counts_by_shard[shard_id],
        )
        for shard_id in range(shard_count)
    ]

    manifest = {
        "source_root": str(source_root),
        "shard_root": str(out_root),
        "datasets": list(datasets),
        "splits": list(splits),
        "seed": int(seed),
        "shard_count": int(shard_count),
        "filter_summary": filter_summary,
        "shards": [
            {
                "shard_index": spec.shard_index,
                "data_root": str(spec.data_root),
                "counts": spec.counts,
            }
            for spec in shard_specs
        ],
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "shard_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return shard_specs
