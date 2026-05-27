from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent
for candidate in (PACKAGE_ROOT, REPO_ROOT):
    text = str(candidate)
    if text not in sys.path:
        sys.path.insert(0, text)

from mas_treesearch import load_processed_split

from stage2_rollback.boundary import RollbackBoundaryModel
from stage2_rollback.diffusion import RollbackDiffusionModel
from stage2_rollback.emitter import RollbackEmitterModel
from stage2_rollback.runtime import RollbackRuntime
from stage2_rollback.selector import RollbackSelectorModel
from stage2_rollback.train_bank import build_blueprint_teacher_samples, blueprint_teacher_samples_to_jsonl
from stage2_rollback.train_boundary import train_boundary_model
from stage2_rollback.train_diffusion import train_diffusion_model
from stage2_rollback.train_selector import train_selector_model
from stage2_rollback.train_verifier import train_blueprint_verifier_model


def _utc_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _sample(items: List[Dict[str, Any]], limit: int, seed: int) -> List[Dict[str, Any]]:
    picked = list(items)
    random.Random(seed).shuffle(picked)
    if limit >= 0:
        picked = picked[:limit]
    return picked


def _device_from_arg(value: str) -> str:
    if value:
        return value
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _print_header(title: str) -> None:
    print(f"\n[rollback-blueprint] {title}", flush=True)


def _debug_runtime(
    *,
    samples,
    dataset_root: Path,
    phase_name: str,
    boundary_state: Dict[str, torch.Tensor] | None = None,
    diffusion_state: Dict[str, torch.Tensor] | None = None,
    emitter_state: Dict[str, torch.Tensor] | None = None,
    selector_state: Dict[str, torch.Tensor] | None = None,
    debug_limit: int = 3,
) -> None:
    runtime = RollbackRuntime(
        boundary_model=RollbackBoundaryModel(),
        diffusion_model=RollbackDiffusionModel(),
        emitter_model=RollbackEmitterModel(),
        selector_model=RollbackSelectorModel(),
    )
    if boundary_state:
        runtime.boundary_model.load_state_dict(boundary_state, strict=False)
    if diffusion_state:
        runtime.diffusion_model.load_state_dict(diffusion_state, strict=False)
    if emitter_state:
        runtime.emitter_model.load_state_dict(emitter_state, strict=False)
    if selector_state:
        runtime.selector_model.load_state_dict(selector_state, strict=False)
    rows = []
    for sample in list(samples)[:debug_limit]:
        result = runtime.analyze(sample, train_mode=True)
        support_pairs = list(zip(result.diffusion_output.node_ids, result.diffusion_output.support.alpha_tilde.tolist()))
        answer_pairs = list(zip(result.diffusion_output.node_ids, result.diffusion_output.answer.alpha_tilde.tolist()))
        support_pairs.sort(key=lambda item: item[1], reverse=True)
        answer_pairs.sort(key=lambda item: item[1], reverse=True)
        row = {
            "phase": phase_name,
            "id": sample.id,
            "teacher_boundary": sample.teacher_boundary,
            "predicted_boundary": result.boundary_output.predicted_boundary,
            "support_top_nodes": support_pairs[:3],
            "answer_top_nodes": answer_pairs[:3],
            "null_emit_prob": float(result.emitter_output.gamma_tilde_plus[-1].item()),
            "selected_emit_nodes": result.emitter_output.selected_node_ids,
            "selector_winner_index": result.selector_output.winner_index,
            "selector_probs": [float(x) for x in result.selector_output.probabilities.tolist()],
            "requests": result.requests,
        }
        print(
            "[rollback-debug]"
            f" phase={phase_name}"
            f" id={sample.id}"
            f" teacher_j={sample.teacher_boundary}"
            f" pred_j={result.boundary_output.predicted_boundary}"
            f" support_top={support_pairs[:2]}"
            f" answer_top={answer_pairs[:2]}"
            f" null_emit={row['null_emit_prob']:.4f}"
            f" winner={result.selector_output.winner_index}",
            flush=True,
        )
        rows.append(row)
    _write_jsonl(dataset_root / f"debug_{phase_name}.jsonl", rows)


def run_dataset(
    *,
    dataset_name: str,
    data_root: str,
    output_root: Path,
    max_train: int,
    seed: int,
    device: str,
    debug_limit: int,
    epochs_verifier: int,
    epochs_boundary: int,
    epochs_diffusion: int,
    epochs_selector: int,
) -> Dict[str, Any]:
    train_items = _sample(load_processed_split(data_root, dataset_name, "train"), max_train, seed)
    if not train_items:
        raise ValueError(f"No train items found for dataset={dataset_name}")
    dataset_root = output_root / dataset_name
    dataset_root.mkdir(parents=True, exist_ok=True)
    start_ts = time.time()

    _print_header(f"{dataset_name}: phase A teacher bank")
    samples = build_blueprint_teacher_samples(dataset_name=dataset_name, items=train_items)
    _write_jsonl(dataset_root / "teacher_bank_blueprint.jsonl", blueprint_teacher_samples_to_jsonl(samples))
    rerun_needed_count = int(sum(sample.rerun_needed for sample in samples))
    avg_trace_len = float(sum(len(sample.trace.steps) for sample in samples) / max(1, len(samples)))
    avg_candidate_count = float(sum(len(sample.candidates) for sample in samples) / max(1, len(samples)))
    print(
        f"[rollback-phaseA] dataset={dataset_name} count={len(samples)} rerun_needed={rerun_needed_count} "
        f"avg_trace_len={avg_trace_len:.2f} avg_candidate_count={avg_candidate_count:.2f}",
        flush=True,
    )

    _print_header(f"{dataset_name}: phase S1 verifier")
    verifier = train_blueprint_verifier_model(samples, device=device, seed=seed, epochs=epochs_verifier)
    torch.save(
        {
            "feature_names": verifier.feature_names,
            "state_dict": verifier.model_state,
            "train_loss": verifier.train_loss,
            "train_accuracy": verifier.train_accuracy,
            "positive_rate": verifier.positive_rate,
            "count": verifier.count,
            "avg_anchor_typed_support": verifier.avg_anchor_typed_support,
        },
        dataset_root / "phase_s1_verifier.pt",
    )
    print(
        f"[rollback-phaseS1] loss={verifier.train_loss:.4f} acc={verifier.train_accuracy:.4f} "
        f"positive_rate={verifier.positive_rate:.4f} avg_anchor_typed_support={verifier.avg_anchor_typed_support:.4f}",
        flush=True,
    )
    _debug_runtime(samples=samples, dataset_root=dataset_root, phase_name="s1_bootstrap", debug_limit=debug_limit)

    _print_header(f"{dataset_name}: phase S2 boundary")
    boundary = train_boundary_model(samples, device=device, seed=seed + 11, epochs=epochs_boundary)
    torch.save(
        {
            "state_dict": boundary.model_state,
            "train_loss": boundary.train_loss,
            "train_accuracy": boundary.train_accuracy,
            "no_rerun_rate": boundary.no_rerun_rate,
            "avg_predicted_boundary": boundary.avg_predicted_boundary,
            "count": boundary.count,
        },
        dataset_root / "phase_s2_boundary.pt",
    )
    print(
        f"[rollback-phaseS2] loss={boundary.train_loss:.4f} acc={boundary.train_accuracy:.4f} "
        f"no_rerun_rate={boundary.no_rerun_rate:.4f} avg_pred_boundary={boundary.avg_predicted_boundary:.2f}",
        flush=True,
    )
    _debug_runtime(
        samples=samples,
        dataset_root=dataset_root,
        phase_name="s2_boundary",
        boundary_state=boundary.model_state,
        debug_limit=debug_limit,
    )

    _print_header(f"{dataset_name}: phase S3/S4 diffusion")
    diffusion = train_diffusion_model(samples, device=device, seed=seed + 29, epochs=epochs_diffusion)
    torch.save(
        {
            "state_dict": diffusion.model_state,
            "train_loss_sup": diffusion.train_loss_sup,
            "train_loss_ans": diffusion.train_loss_ans,
            "avg_alpha_sup_entropy": diffusion.avg_alpha_sup_entropy,
            "avg_alpha_ans_entropy": diffusion.avg_alpha_ans_entropy,
            "count": diffusion.count,
        },
        dataset_root / "phase_s34_diffusion.pt",
    )
    print(
        f"[rollback-phaseS34] loss_sup={diffusion.train_loss_sup:.4f} loss_ans={diffusion.train_loss_ans:.4f} "
        f"entropy_sup={diffusion.avg_alpha_sup_entropy:.4f} entropy_ans={diffusion.avg_alpha_ans_entropy:.4f}",
        flush=True,
    )
    _debug_runtime(
        samples=samples,
        dataset_root=dataset_root,
        phase_name="s34_diffusion",
        boundary_state=boundary.model_state,
        diffusion_state=diffusion.model_state,
        debug_limit=debug_limit,
    )

    _print_header(f"{dataset_name}: phase S4/S5 emitter+selector")
    selector = train_selector_model(
        samples,
        device=device,
        seed=seed + 47,
        diffusion_state=diffusion.model_state,
        epochs=epochs_selector,
    )
    torch.save(
        {
            "emitter_state": selector.emitter_state,
            "selector_state": selector.selector_state,
            "train_loss_emit": selector.train_loss_emit,
            "train_loss_sel": selector.train_loss_sel,
            "null_emit_rate": selector.null_emit_rate,
            "anchor_select_rate": selector.anchor_select_rate,
            "count": selector.count,
        },
        dataset_root / "phase_s45_selector.pt",
    )
    print(
        f"[rollback-phaseS45] loss_emit={selector.train_loss_emit:.4f} loss_sel={selector.train_loss_sel:.4f} "
        f"null_emit_rate={selector.null_emit_rate:.4f} anchor_select_rate={selector.anchor_select_rate:.4f}",
        flush=True,
    )
    _debug_runtime(
        samples=samples,
        dataset_root=dataset_root,
        phase_name="s45_selector",
        boundary_state=boundary.model_state,
        diffusion_state=diffusion.model_state,
        emitter_state=selector.emitter_state,
        selector_state=selector.selector_state,
        debug_limit=debug_limit,
    )

    metrics = {
        "dataset": dataset_name,
        "seed": seed,
        "device": device,
        "count": len(samples),
        "rerun_needed_count": rerun_needed_count,
        "avg_trace_len": avg_trace_len,
        "avg_candidate_count": avg_candidate_count,
        "verifier_loss": verifier.train_loss,
        "verifier_accuracy": verifier.train_accuracy,
        "boundary_loss": boundary.train_loss,
        "boundary_accuracy": boundary.train_accuracy,
        "boundary_no_rerun_rate": boundary.no_rerun_rate,
        "diffusion_loss_sup": diffusion.train_loss_sup,
        "diffusion_loss_ans": diffusion.train_loss_ans,
        "selector_loss_emit": selector.train_loss_emit,
        "selector_loss_sel": selector.train_loss_sel,
        "selector_null_emit_rate": selector.null_emit_rate,
        "selector_anchor_select_rate": selector.anchor_select_rate,
        "elapsed_s": time.time() - start_ts,
    }
    _write_json(dataset_root / "metrics.json", metrics)
    print(f"[rollback-done] dataset={dataset_name} metrics={json.dumps(metrics, ensure_ascii=False)}", flush=True)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train full stage2-rollback blueprint stack.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--max-train", type=int, default=180)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="")
    parser.add_argument("--debug-limit", type=int, default=3)
    parser.add_argument("--epochs-verifier", type=int, default=40)
    parser.add_argument("--epochs-boundary", type=int, default=40)
    parser.add_argument("--epochs-diffusion", type=int, default=40)
    parser.add_argument("--epochs-selector", type=int, default=40)
    args = parser.parse_args()

    os.environ.setdefault("LLM_API_BASE", "http://127.0.0.1:8039")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    datasets = list(args.dataset)
    if not datasets:
        raise ValueError("At least one --dataset is required for rollback full blueprint training.")
    device = _device_from_arg(args.device)
    summaries = []
    for dataset_name in datasets:
        summaries.append(
            run_dataset(
                dataset_name=dataset_name,
                data_root=args.data_root,
                output_root=output_root,
                max_train=args.max_train,
                seed=args.seed,
                device=device,
                debug_limit=args.debug_limit,
                epochs_verifier=args.epochs_verifier,
                epochs_boundary=args.epochs_boundary,
                epochs_diffusion=args.epochs_diffusion,
                epochs_selector=args.epochs_selector,
            )
        )
    _write_json(output_root / "suite_report.json", {"datasets": summaries, "saved_at": _utc_now()})


if __name__ == "__main__":
    main()
