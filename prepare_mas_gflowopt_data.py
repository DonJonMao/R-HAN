from __future__ import annotations

import argparse
import json

from mas_treesearch import build_processed_datasets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--unified-root",
        default="/mnt/nvme/projects/R-HAN/dataset/unified_mixed",
        help="Root directory containing mixed train/validation/test JSONL files.",
    )
    parser.add_argument(
        "--output-root",
        default="/mnt/nvme/projects/R-HAN/dataset/mas_gflowopt_processed",
        help="Directory where per-dataset processed files will be written.",
    )
    args = parser.parse_args()

    manifest = build_processed_datasets(
        unified_root=args.unified_root,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
