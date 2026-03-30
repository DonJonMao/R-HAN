from __future__ import annotations

import argparse

from mas_stage2 import Stage2RuntimeConfig, build_default_stage2_runtime
from mas_treesearch import TreeSearchMASPipeline, resolve_dataset_profile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--replay-dir", default="")
    args = parser.parse_args()

    pipeline = TreeSearchMASPipeline(pipeline_mode="structure_only")
    result = pipeline.search(args.question, dataset_name=args.dataset or None, learn=False)
    if result.union_graph is None:
        raise RuntimeError("Stage-1 search did not produce a union graph.")

    stage2 = build_default_stage2_runtime(config=Stage2RuntimeConfig())
    run = stage2.run(
        result.union_graph,
        question_text=args.question,
        metadata={"mas_dataset_name": args.dataset} if args.dataset else None,
        dataset_profile=resolve_dataset_profile(args.dataset or None),
        replay_dir=args.replay_dir or None,
    )
    print(run.final_answer)


if __name__ == "__main__":
    main()
