from __future__ import annotations

import argparse
import json
import random
from typing import Any

from mas_treesearch import SearchConfig, TieredEvalConfig, TreeSearchMASPipeline


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
        best = result.best_node.tier2
        if best is not None:
            aggregate.append(
                {
                    "reward": best.mean_reward,
                    "task_score": best.mean_task_score,
                    "success": best.mean_success,
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
