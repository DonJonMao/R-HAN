from __future__ import annotations

import argparse
import json
import random
from typing import Any

from mas_treesearch import (
    SearchConfig,
    TieredEvalConfig,
    TreeSearchMASPipeline,
    has_structure_output,
    is_union_result,
    resolve_primary_reward,
    resolve_result_output,
    resolve_result_signature,
    resolve_result_summary,
    resolve_structure_summary,
)


def load_question_from_jsonl(path: str, index: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = str(obj.get("question", "")).strip()
            if not question:
                continue
            items.append(obj)
    if not items:
        raise ValueError(f"No usable questions found in {path}")
    if index < 0 or index >= len(items):
        raise IndexError(f"Question index {index} out of range for {len(items)} items")
    return items[index]


def summarize_node(node, debug_judge: bool = False) -> str:
    tier2 = node.tier2
    tier1 = node.tier1
    precheck = node.precheck
    active_agents = node.state.active_agents()
    lines = [
        f"signature={node.compiled.signature()}",
        f"template={node.state.template.value}",
        f"active_agents={active_agents}",
        f"role_to_agent={node.state.role_to_agent}",
        f"role_to_prompt={node.state.role_to_prompt}",
        f"proxy_score={node.proxy_score}",
        f"proxy_uncertainty={node.proxy_uncertainty}",
        f"q_mean={node.stats.q_mean:.4f} visits={node.stats.visits}",
    ]
    if tier1 is not None:
        lines.append(
            "tier1="
            f"reward={tier1.mean_reward:.4f} "
            f"task={tier1.mean_task_score:.4f} "
            f"success={tier1.mean_success:.4f} "
            f"latency={tier1.mean_latency:.2f} "
            f"token_cost={tier1.mean_token_cost:.4f}"
        )
    if precheck is not None:
        lines.append(
            "precheck="
            f"reward={precheck.mean_reward:.4f} "
            f"task={precheck.mean_task_score:.4f} "
            f"success={precheck.mean_success:.4f} "
            f"latency={precheck.mean_latency:.2f} "
            f"token_cost={precheck.mean_token_cost:.4f}"
        )
    if tier2 is not None:
        lines.append(
            "tier2="
            f"reward={tier2.mean_reward:.4f}±{tier2.reward_std:.4f} "
            f"task={tier2.mean_task_score:.4f} "
            f"success={tier2.mean_success:.4f} "
            f"latency={tier2.mean_latency:.2f} "
            f"token_cost={tier2.mean_token_cost:.4f}"
        )
        if tier2.evaluations:
            lines.append(f"final_output={tier2.evaluations[0].raw_output}")
            if debug_judge and tier2.evaluations[0].debug_info:
                lines.append(f"debug_info={json.dumps(tier2.evaluations[0].debug_info, ensure_ascii=False)}")
    return "\n".join(lines)


def load_all_questions(path: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            question = str(obj.get("question", "")).strip()
            if not question:
                continue
            items.append(obj)
    if not items:
        raise ValueError(f"No usable questions found in {path}")
    return items


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default="", help="Direct question text.")
    parser.add_argument(
        "--question-file",
        default="",
        help="Optional JSONL file containing a 'question' field.",
    )
    parser.add_argument(
        "--question-index",
        type=int,
        default=0,
        help="Index used with --question-file.",
    )
    parser.add_argument("--search-iterations", type=int, default=12)
    parser.add_argument("--candidate-core-k", type=int, default=4)
    parser.add_argument("--candidate-explore-k", type=int, default=2)
    parser.add_argument("--candidate-max-k", type=int, default=6)
    parser.add_argument("--tier1-max-tokens", type=int, default=192)
    parser.add_argument("--tier2-max-tokens", type=int, default=512)
    parser.add_argument("--tier1-repeats", type=int, default=1)
    parser.add_argument("--tier2-repeats", type=int, default=2)
    parser.add_argument("--disable-learned-prior", action="store_true")
    parser.add_argument("--disable-learned-value", action="store_true")
    parser.add_argument("--show-records", type=int, default=10, help="How many search records to print.")
    parser.add_argument("--debug-judge", action="store_true")
    parser.add_argument("--random-count", type=int, default=1, help="Randomly evaluate N items from --question-file.")
    parser.add_argument("--random-seed", type=int, default=7)
    parser.add_argument(
        "--pipeline-mode",
        choices=("structure_only", "structure_plus_union_runtime"),
        default="structure_only",
    )
    args = parser.parse_args()

    search_config = SearchConfig(
        search_iterations=args.search_iterations,
        candidate_core_k=args.candidate_core_k,
        candidate_explore_k=args.candidate_explore_k,
        candidate_max_k=args.candidate_max_k,
        enable_learned_edit_prior=not args.disable_learned_prior,
        enable_learned_value_model=not args.disable_learned_value,
    )
    runtime_config = TieredEvalConfig()
    runtime_config.tier1.max_tokens = args.tier1_max_tokens
    runtime_config.tier2.max_tokens = args.tier2_max_tokens
    runtime_config.tier1.repeats = args.tier1_repeats
    runtime_config.tier2.repeats = args.tier2_repeats
    runtime_config.debug_judge = args.debug_judge

    pipeline = TreeSearchMASPipeline(
        search_config=search_config,
        runtime_config=runtime_config,
        pipeline_mode=args.pipeline_mode,
    )

    items: list[tuple[str, dict[str, Any], Optional[str], Optional[dict]]] = []
    if args.question:
        question_text = args.question.strip()
        items.append((question_text, {"source": "cli", "index": -1}, None, None))
    elif args.question_file:
        all_items = load_all_questions(args.question_file)
        if args.random_count > 1:
            rng = random.Random(args.random_seed)
            indices = rng.sample(range(len(all_items)), k=min(args.random_count, len(all_items)))
            for idx in indices:
                item = all_items[idx]
                items.append(
                    (
                        str(item.get("question", "")).strip(),
                        {
                            "source": args.question_file,
                            "index": idx,
                            "id": item.get("id", ""),
                            "category": item.get("category", ""),
                            "source_dataset": item.get("source_dataset", ""),
                        },
                        str(item.get("answer", "")).strip() or None,
                        item.get("metadata"),
                    )
                )
        else:
            item = all_items[args.question_index]
            items.append(
                (
                    str(item.get("question", "")).strip(),
                    {
                        "source": args.question_file,
                        "index": args.question_index,
                        "id": item.get("id", ""),
                        "category": item.get("category", ""),
                        "source_dataset": item.get("source_dataset", ""),
                    },
                    str(item.get("answer", "")).strip() or None,
                    item.get("metadata"),
                )
            )
    else:
        raise ValueError("Provide either --question or --question-file.")

    aggregate = []
    for run_idx, (question_text, question_meta, reference_answer, metadata) in enumerate(items, start=1):
        result = pipeline.search(
            question_text,
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_name=str(question_meta.get("source_dataset", "")).strip() or None,
        )
        print(f"=== Run {run_idx}/{len(items)} ===", flush=True)
        print("=== Question ===", flush=True)
        print(question_text, flush=True)
        print("=== Meta ===", flush=True)
        print(question_meta, flush=True)
        if reference_answer:
            print("=== Reference Answer ===", flush=True)
            print(reference_answer, flush=True)
        print("=== Candidate Agents ===", flush=True)
        print(result.selected_agents, flush=True)
        print("=== Root Signatures ===", flush=True)
        for sig in result.root_signatures:
            print(sig, flush=True)
        print("=== Best Node ===", flush=True)
        print(summarize_node(result.best_node, debug_judge=args.debug_judge), flush=True)
        if has_structure_output(result):
            structure = resolve_structure_summary(result)
            if structure is not None:
                metrics = structure.metrics
                print("=== Structure Output ===", flush=True)
                print(f"mode={result.pipeline_mode}", flush=True)
                print(f"structure_signature={structure.signature}", flush=True)
                print(
                    "selected_topologies="
                    + json.dumps(structure.selected_topology_signatures, ensure_ascii=False),
                    flush=True,
                )
                print(
                    "selected_topology_scores="
                    + json.dumps([round(score, 4) for score in structure.selected_topology_scores], ensure_ascii=False),
                    flush=True,
                )
                print(
                    "structure_metrics="
                    f"coverage={metrics.coverage:.4f} "
                    f"complementarity={metrics.complementarity:.4f} "
                    f"redundancy={metrics.redundancy_quality:.4f} "
                    f"faithfulness={metrics.structural_faithfulness:.4f} "
                    f"affordability={metrics.runtime_affordability:.4f} "
                    f"probe={metrics.execution_probe:.4f} "
                    f"total={metrics.total_reward:.4f}",
                    flush=True,
                )
        if is_union_result(result):
            final_summary = resolve_result_summary(result)
            print("=== Union Runtime ===", flush=True)
            print(f"final_signature={resolve_result_signature(result)}", flush=True)
            if result.selected_topology_nodes:
                print(
                    "selected_topologies="
                    + json.dumps([node.compiled.signature() for node in result.selected_topology_nodes], ensure_ascii=False),
                    flush=True,
                )
            if final_summary is not None:
                print(
                    "final_summary="
                    f"reward={final_summary.mean_reward:.4f} "
                    f"task={final_summary.mean_task_score:.4f} "
                    f"success={final_summary.mean_success:.4f} "
                    f"latency={final_summary.mean_latency:.2f} "
                    f"token_cost={final_summary.mean_token_cost:.4f}",
                    flush=True,
                )
                print(f"final_output={resolve_result_output(result)}", flush=True)
            print(f"turns={len(result.turn_traces)}", flush=True)
        print("=== Top Nodes ===", flush=True)
        for idx, node in enumerate(result.top_nodes, start=1):
            print(f"[top {idx}]", flush=True)
            print(summarize_node(node, debug_judge=args.debug_judge), flush=True)
            print("---", flush=True)
        print("=== Search Records ===", flush=True)
        for record in result.records[: max(0, args.show_records)]:
            print(
                f"state={record.state_signature} parent={record.parent_signature} "
                f"action={record.action} proxy={record.proxy_score:.4f} "
                f"tier1={record.tier1_score} tier2={record.tier2_score}",
                flush=True,
            )
        best = resolve_result_summary(result)
        primary_reward = resolve_primary_reward(result)
        if primary_reward is not None:
            aggregate.append(
                {
                    "reward": primary_reward,
                    "task_score": best.mean_task_score if best is not None else (result.structure_summary.metrics.execution_probe if result.structure_summary is not None else 0.0),
                    "success": best.mean_success if best is not None else (result.structure_summary.metrics.execution_probe if result.structure_summary is not None else 0.0),
                }
            )
    if aggregate:
        avg_reward = sum(x["reward"] for x in aggregate) / len(aggregate)
        avg_task = sum(x["task_score"] for x in aggregate) / len(aggregate)
        avg_success = sum(x["success"] for x in aggregate) / len(aggregate)
        print("=== Aggregate Summary ===", flush=True)
        print(
            f"runs={len(aggregate)} avg_reward={avg_reward:.4f} "
            f"avg_task_score={avg_task:.4f} avg_success={avg_success:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
