from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict


DATASETS = ("mmlu_pro", "nlgraph", "knowledge_crosswords")
SPLIT_MAP: Dict[str, str] = {
    "train": "stage1_train",
    "validation": "stage2_train",
    "test": "final_eval",
}


def _replace_with_symlink(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    try:
        os.symlink(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_threeway_general_20260319",
        help="Three-way split root containing stage1_train/stage2_train/final_eval files.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_treesearch_threeway_general_trainview_20260319",
        help="Train-compatible view root exposing train/validation/test jsonl files.",
    )
    args = parser.parse_args()

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "datasets": {},
        "split_map": dict(SPLIT_MAP),
    }

    for dataset in DATASETS:
        src_dir = source_root / dataset
        dst_dir = output_root / dataset
        dst_dir.mkdir(parents=True, exist_ok=True)
        manifest["datasets"][dataset] = {}
        for split, src_split in SPLIT_MAP.items():
            src_path = src_dir / f"{src_split}.jsonl"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing source split for dataset={dataset}: {src_path}")
            dst_path = dst_dir / f"{split}.jsonl"
            _replace_with_symlink(src_path, dst_path)
            manifest["datasets"][dataset][split] = {
                "source_split": src_split,
                "path": str(dst_path),
            }

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
