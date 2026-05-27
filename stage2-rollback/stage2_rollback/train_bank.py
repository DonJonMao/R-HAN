from __future__ import annotations

import glob
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from mas_treesearch import DatasetProfile, resolve_dataset_profile
from mas_treesearch.agents import default_agent_pool
from mas_treesearch.config import TieredEvalConfig
from mas_treesearch.evaluator import MultiFidelityEvaluator

from .artifacts import ArtifactIR, canonicalize_candidate, unit_text_overlap
from .contracts import (
    AnswerObject,
    contract_residual,
    materialize_answer_object,
    parse_answer_object,
    typed_distance,
)
from .pipeline import RollbackPreparedStage1Artifact, load_optional_prepared_artifact
from .trace_ir import RollbackTrace, build_rollback_trace


def _safe_item_key(value: Any) -> str:
    text = str(value if value is not None else "sample")
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _lexicographic_better(success_a: float, task_a: float, success_b: float, task_b: float) -> bool:
    if float(success_a) != float(success_b):
        return float(success_a) > float(success_b)
    return float(task_a) > float(task_b)


@dataclass
class AnchorCandidate:
    item_id: str
    output: str
    stage1_signature: str = ""
    stage1_success: float = 0.0
    stage1_task_score: float = 0.0
    stage1_reward: float = 0.0
    source_kind: str = ""
    source_path: str = ""
    structure_artifact_path: str = ""


@dataclass
class TeacherBankRecord:
    id: str
    dataset: str
    answer_format: str
    task_subtype: str
    question: str
    reference_answer: str
    anchor_output: str
    oracle_output: str
    anchor_source_kind: str
    anchor_source_path: str
    structure_artifact_path: str
    anchor_contract_valid: bool
    anchor_recoverable_valid: bool
    anchor_signature: str
    oracle_signature: str
    anchor_kind: str
    oracle_kind: str
    stage1_success: float
    stage1_task_score: float
    stage1_reward: float
    anchor_eval_success: float
    anchor_eval_task_score: float
    oracle_eval_success: float
    oracle_eval_task_score: float
    rerun_needed: int
    teacher_margin: float
    feature_map: Dict[str, float] = field(default_factory=dict)
    rollback_prepared: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RollbackTeacherCandidate:
    candidate_id: str
    boundary_index: int
    origin_kind: str
    emitted_node_id: str
    output: str
    artifact: ArtifactIR
    eval_success: float
    eval_task_score: float
    teacher_rank: int = 0
    bank_probability: float = 0.0
    typed_support_teacher: float = 0.0
    contract_residual_teacher: float = 0.0
    keep_teacher: float = 0.0
    signature_distance_teacher: float = 0.0
    rerun_subgraph_node_ids: List[str] = field(default_factory=list)
    q_sup_target: Dict[str, float] = field(default_factory=dict)
    q_ans_target: Dict[str, float] = field(default_factory=dict)


@dataclass
class RollbackTeacherSample:
    id: str
    dataset: str
    answer_format: str
    task_subtype: str
    question: str
    reference_answer: str
    metadata: Dict[str, Any]
    prepared: RollbackPreparedStage1Artifact
    trace: RollbackTrace
    anchor_candidate: RollbackTeacherCandidate
    candidates: List[RollbackTeacherCandidate]
    teacher_boundary: int
    phat_bank: Dict[str, float]
    rerun_needed: int
    summary_features: Dict[str, float] = field(default_factory=dict)


def _load_jsonl_rows(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                yield payload


def _discover_anchor_row_sources(dataset_name: str) -> List[Tuple[str, str]]:
    patterns: Dict[str, Sequence[Tuple[str, str]]] = {
        "mbpp": (
            ("checkpoint", "/mnt/nvme/projects/R-HAN/outputs/stage1_selected_ckpts_v2_20260330/mbpp/checkpoint.json"),
            ("rows", "/mnt/nvme/projects/R-HAN/outputs/stage2_ucc_phase3a_unified_mbpp_3x_20260416_181035/workers/worker_*/mbpp/train_rows.jsonl"),
        ),
        "mmlu_pro": (
            ("rows", "/mnt/nvme/projects/R-HAN/outputs/stage2-ucc- phase1_mmlu-pro_3x_20260417_215446/workers/worker_*/mmlu_pro/train_rows.jsonl"),
        ),
        "nlgraph": (
            ("rows", "/mnt/nvme/projects/R-HAN/outputs/stage2-ucc- phase1_nlgraph_3x_20260418_210939/workers/worker_*/nlgraph/train_rows.jsonl"),
        ),
        "math": (
            ("checkpoint", "/mnt/nvme/projects/R-HAN/outputs/mas_treesearch_target_suite_train_20260314_212033/math/checkpoint.json"),
            ("rows", "/mnt/nvme/projects/R-HAN/outputs/mas_treesearch_aflow_four_bg_20260318_182959/math/math/train_rows.jsonl"),
        ),
    }
    discovered: List[Tuple[str, str]] = []
    for kind, pattern in patterns.get(dataset_name, ()):
        matches = sorted(glob.glob(pattern))
        if matches:
            discovered.extend((kind, match) for match in matches)
    return discovered


def _discover_structure_sources(dataset_name: str) -> List[str]:
    patterns = {
        "mbpp": (
            "/mnt/nvme/projects/R-HAN/outputs/mas_stage2_mixed_gsm8k_official_mbpp_humaneval_trainx2_20260327_193544/mbpp/stage1_structures/mbpp/train",
        ),
        "mmlu_pro": (
            "/mnt/nvme/projects/R-HAN/outputs/stage2-ucc- phase1_mmlu-pro_3x_20260417_215446/workers/worker_*/mmlu_pro/stage1_structures/mmlu_pro/train",
        ),
        "nlgraph": (
            "/mnt/nvme/projects/R-HAN/outputs/stage2-ucc- phase1_nlgraph_3x_20260418_210939/workers/worker_*/nlgraph/stage1_structures/nlgraph/train",
        ),
        "math": (),
    }
    dirs: List[str] = []
    for pattern in patterns.get(dataset_name, ()):
        dirs.extend(sorted(glob.glob(pattern)))
    return dirs


def _load_anchor_candidates(dataset_name: str) -> Dict[str, AnchorCandidate]:
    anchors: Dict[str, AnchorCandidate] = {}
    for kind, source_path in _discover_anchor_row_sources(dataset_name):
        path = Path(source_path)
        if kind == "checkpoint":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows = payload.get("metadata", {}).get("train_rows", [])
            for row in rows:
                if not isinstance(row, dict):
                    continue
                item_id = str(row.get("id", ""))
                if not item_id or item_id in anchors:
                    continue
                anchors[item_id] = AnchorCandidate(
                    item_id=item_id,
                    output=str(row.get("output", "")),
                    stage1_signature=str(row.get("signature", "")),
                    stage1_success=float(row.get("success", row.get("stage1_success", 0.0))),
                    stage1_task_score=float(row.get("task_score", row.get("stage1_task_score", 0.0))),
                    stage1_reward=float(row.get("reward", row.get("stage1_reward", 0.0))),
                    source_kind="checkpoint",
                    source_path=str(path),
                )
            continue
        for row in _load_jsonl_rows(path):
            item_id = str(row.get("id", ""))
            if not item_id or item_id in anchors:
                continue
            anchors[item_id] = AnchorCandidate(
                item_id=item_id,
                output=str(row.get("output", "")),
                stage1_signature=str(row.get("stage1_signature", row.get("signature", ""))),
                stage1_success=float(row.get("stage1_success", row.get("success", 0.0))),
                stage1_task_score=float(row.get("stage1_task_score", row.get("task_score", 0.0))),
                stage1_reward=float(row.get("stage1_reward", row.get("reward", 0.0))),
                source_kind="rows",
                source_path=str(path),
                structure_artifact_path=str(row.get("structure_artifact_path", "")),
            )
    structure_dirs = _discover_structure_sources(dataset_name)
    for item_id, candidate in anchors.items():
        if candidate.structure_artifact_path:
            continue
        key = _safe_item_key(item_id)
        for directory in structure_dirs:
            candidate_path = Path(directory) / f"{key}.json"
            if candidate_path.exists():
                candidate.structure_artifact_path = str(candidate_path)
                break
    return anchors


def _build_output_evaluator() -> MultiFidelityEvaluator:
    return MultiFidelityEvaluator(TieredEvalConfig(), default_agent_pool())


def _fallback_anchor_output(item: Dict[str, Any], profile: DatasetProfile) -> str:
    answer = str(item.get("answer", "") or "")
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    if profile.answer_format == "option":
        return "OPTION - 1"
    if profile.answer_format == "graph_json":
        subtype = str(metadata.get("task", "") or "")
        if subtype in {"connectivity", "cycle"}:
            return '{"answer":"yes"}'
        if subtype == "flow":
            return '{"max_flow":0}'
        if subtype in {"topology", "hamilton", "shortest_path"}:
            return '{"path":[]}'
        if subtype.lower() == "gnn":
            return '{"node_embeddings":{}}'
        return "{}"
    if profile.answer_format == "python_code":
        entry = str(metadata.get("entry_point", "solution") or "solution")
        return f"def {entry}(*args, **kwargs):\n    return None\n"
    if profile.answer_format in {"math_expression", "numeric"}:
        return "0" if answer.strip() != "0" else "1"
    return ""


def _build_feature_map(
    *,
    profile: DatasetProfile,
    task_subtype: str,
    anchor_obj: AnswerObject,
    anchor: AnchorCandidate,
    prepared: Optional[RollbackPreparedStage1Artifact],
    anchor_eval_success: float,
    anchor_eval_task_score: float,
) -> Dict[str, float]:
    feature_map: Dict[str, float] = {
        f"answer_format::{profile.answer_format}": 1.0,
        f"task_type::{profile.task_type}": 1.0,
        f"task_subtype::{task_subtype or 'unknown'}": 1.0,
        f"answer_kind::{anchor_obj.kind}": 1.0,
        "anchor_contract_valid": 1.0 if bool(anchor_obj.fields.get("contract_valid")) else 0.0,
        "anchor_recoverable_valid": 1.0 if bool(anchor_obj.fields.get("recoverable_valid")) else 0.0,
        "anchor_text_len": float(len(anchor.output)),
        "anchor_newline_count": float(anchor.output.count("\n")),
        "stage1_success": float(anchor.stage1_success),
        "stage1_task_score": float(anchor.stage1_task_score),
        "stage1_reward": float(anchor.stage1_reward),
        "anchor_eval_success": float(anchor_eval_success),
        "anchor_eval_task_score": float(anchor_eval_task_score),
        "has_structure_artifact": 1.0 if prepared and prepared.base_prepared is not None else 0.0,
    }
    recovered = anchor_obj.fields.get("recoverable_value")
    if isinstance(recovered, (str, int, float)):
        feature_map["recoverable_value_len"] = float(len(str(recovered)))
    if prepared and prepared.base_prepared and prepared.base_prepared.structure_summary is not None:
        metrics = prepared.base_prepared.structure_summary.metrics
        feature_map["structure_coverage"] = float(metrics.coverage)
        feature_map["structure_complementarity"] = float(metrics.complementarity)
        feature_map["structure_redundancy_quality"] = float(metrics.redundancy_quality)
        feature_map["structure_total_reward"] = float(metrics.total_reward)
    return feature_map


def _build_local_adapter(
    *,
    anchor: AnchorCandidate,
    anchor_obj: AnswerObject,
) -> RollbackPreparedStage1Artifact:
    base_prepared = load_optional_prepared_artifact(anchor.structure_artifact_path or None)
    return RollbackPreparedStage1Artifact(
        base_prepared=base_prepared,
        anchor_artifact={
            "stage1_output": anchor.output,
            "stage1_signature": anchor.stage1_signature,
            "answer_signature": anchor_obj.signature,
            "answer_kind": anchor_obj.kind,
            "fields": dict(anchor_obj.fields),
        },
        rollback_trace={
            "available": False,
            "steps": [],
        },
        anchor_replay_cache={
            "available": False,
            "source_kind": anchor.source_kind,
        },
    )


def _candidate_output_for_boundary(
    *,
    boundary_index: int,
    max_boundary: int,
    profile: DatasetProfile,
    task_subtype: str,
    metadata: Dict[str, Any],
    anchor_output: str,
    anchor_obj: AnswerObject,
    reference_answer: str,
    oracle_obj: AnswerObject,
) -> Tuple[str, str]:
    if boundary_index == 0:
        return anchor_output, "anchor"
    depth = float(boundary_index) / max(1.0, float(max_boundary))
    anchor_contract = materialize_answer_object(
        anchor_obj,
        answer_format=profile.answer_format,
        task_subtype=task_subtype,
        metadata=metadata,
    )
    oracle_contract = materialize_answer_object(
        oracle_obj,
        answer_format=profile.answer_format,
        task_subtype=task_subtype,
        metadata=metadata,
    )
    if profile.answer_format == "option":
        if anchor_obj.signature == oracle_obj.signature:
            return (anchor_contract or anchor_output), "contract_anchor"
        if depth <= 0.34:
            return oracle_contract, "oracle_shallow"
        if depth <= 0.67:
            return f"Updated evidence.\n{oracle_contract}", "oracle_mid"
        return (anchor_contract or anchor_output), "anchor_recover"
    if profile.answer_format == "graph_json":
        if depth <= 0.40:
            return oracle_contract, "oracle_struct"
        if not anchor_obj.fields.get("contract_valid", False) and anchor_obj.fields.get("recoverable_valid", False):
            return anchor_contract, "recover_struct"
        return (oracle_contract if depth <= 0.70 else anchor_output), "mixed_struct"
    if profile.answer_format == "python_code":
        if depth <= 0.50:
            return reference_answer, "oracle_code"
        recovered = anchor_contract.strip() if anchor_contract.strip() else anchor_output
        return recovered, "recover_code"
    if profile.answer_format in {"math_expression", "numeric"}:
        if depth <= 0.40:
            return oracle_contract or reference_answer, "oracle_math"
        if anchor_obj.fields.get("recoverable_valid", False):
            return anchor_contract, "recover_math"
        return anchor_output, "anchor_math"
    return (oracle_contract if depth <= 0.50 else anchor_output), "generic"


def _teacher_typed_support(
    artifact: ArtifactIR,
    *,
    reference_signature: str,
) -> float:
    residual = contract_residual(
        artifact.answer_object,
        answer_format=artifact.answer_format,
        task_subtype=artifact.task_subtype,
        metadata=None,
    )
    support_hit = 1.0 if artifact.answer_signature == reference_signature else 0.0
    answer_overlap = unit_text_overlap(
        [unit for unit in artifact.units if unit.unit_id in artifact.answer_unit_ids],
        [unit for unit in artifact.units if unit.unit_id in artifact.evidence_unit_ids],
    )
    raw = 0.40 * (1.0 - residual.pooled) + 0.40 * support_hit + 0.20 * answer_overlap
    return max(0.0, min(1.0, raw))


def _teacher_keep(anchor_artifact: ArtifactIR, candidate_artifact: ArtifactIR) -> float:
    return unit_text_overlap(
        [unit for unit in anchor_artifact.units if unit.unit_id in set(anchor_artifact.answer_unit_ids) | set(anchor_artifact.evidence_unit_ids)],
        [unit for unit in candidate_artifact.units],
    )


def _graph_support(node_ids: Sequence[str], adjacency: Sequence[Sequence[float]], seed: List[float], steps: int = 8, alpha: float = 0.85) -> List[float]:
    if not node_ids:
        return [1.0]
    length = len(node_ids)
    current = list(seed)
    total = sum(current)
    if total <= 0:
        current = [1.0 / float(length)] * length
    else:
        current = [value / total for value in current]
    row_norm = []
    for row in adjacency:
        denom = sum(row)
        if denom <= 0:
            row_norm.append([1.0 / float(length)] * length)
        else:
            row_norm.append([float(value) / float(denom) for value in row])
    for _ in range(max(1, int(steps))):
        nxt = [(1.0 - alpha) * seed[idx] for idx in range(length)]
        for src in range(length):
            for dst in range(length):
                nxt[dst] += alpha * current[src] * row_norm[src][dst]
        total = sum(nxt)
        current = [value / total for value in nxt] if total > 0 else [1.0 / float(length)] * length
    return current


def _teacher_subgraph(
    prepared: RollbackPreparedStage1Artifact,
    trace: RollbackTrace,
    *,
    boundary_index: int,
) -> Tuple[str, List[str], Dict[str, float], Dict[str, float]]:
    if prepared.base_prepared is None:
        return "fallback", ["fallback"], {"fallback": 1.0}, {"fallback": 1.0}
    graph = prepared.base_prepared.union_graph
    node_ids = list(graph.nodes.keys())
    if not node_ids:
        return "fallback", ["fallback"], {"fallback": 1.0}, {"fallback": 1.0}
    node_index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = [[0.0 for _ in node_ids] for _ in node_ids]
    for edge in graph.edges:
        if edge.src in node_index and edge.dst in node_index:
            adjacency[node_index[edge.src]][node_index[edge.dst]] = 1.0 + float(edge.support_ratio)
    seed = [0.0 for _ in node_ids]
    suffix_nodes: List[str] = []
    for step in trace.steps:
        if step.index > boundary_index:
            suffix_nodes.extend(step.provenance_node_ids)
    suffix_nodes = [node for node in suffix_nodes if node in node_index]
    if suffix_nodes:
        mass = 1.0 / float(len(suffix_nodes))
        for node_id in suffix_nodes:
            seed[node_index[node_id]] += mass
    else:
        anchor_nodes = [node_id for node_id in graph.sink_node_ids if node_id in node_index] or node_ids
        mass = 1.0 / float(len(anchor_nodes))
        for node_id in anchor_nodes:
            seed[node_index[node_id]] += mass
    ppr = _graph_support(node_ids, adjacency, seed)
    support = sorted(zip(node_ids, ppr), key=lambda item: item[1], reverse=True)
    top_nodes = [node_id for node_id, score in support if score >= support[0][1] * 0.5][:4]
    emitted_node = top_nodes[0]
    target = {node_id: float(score) for node_id, score in support if node_id in top_nodes}
    total = sum(target.values()) or 1.0
    q_target = {node_id: value / total for node_id, value in target.items()}
    return emitted_node, top_nodes, q_target, q_target


def _lexicographic_order_key(candidate: RollbackTeacherCandidate) -> Tuple[float, float, float, float, float]:
    return (
        float(candidate.eval_success),
        float(candidate.eval_task_score),
        float(candidate.typed_support_teacher),
        -float(candidate.contract_residual_teacher),
        float(candidate.keep_teacher) - 0.01 * float(candidate.signature_distance_teacher),
    )


def _candidate_probabilities(candidates: Sequence[RollbackTeacherCandidate]) -> Dict[str, float]:
    ranked = sorted(candidates, key=_lexicographic_order_key, reverse=True)
    for rank, candidate in enumerate(ranked):
        candidate.teacher_rank = rank
    weights = [pow(2.718281828, -float(candidate.teacher_rank)) for candidate in ranked]
    total = sum(weights) or 1.0
    probs = {}
    for candidate, weight in zip(ranked, weights):
        candidate.bank_probability = float(weight / total)
        probs[candidate.candidate_id] = candidate.bank_probability
    return probs


def build_blueprint_teacher_samples(
    *,
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
    evaluator: Optional[MultiFidelityEvaluator] = None,
) -> List[RollbackTeacherSample]:
    profile = resolve_dataset_profile(dataset_name)
    anchor_map = _load_anchor_candidates(dataset_name)
    output_evaluator = evaluator or _build_output_evaluator()
    samples: List[RollbackTeacherSample] = []
    for item in items:
        item_id = str(item.get("id", ""))
        question = str(item.get("question", ""))
        reference_answer = str(item.get("answer", ""))
        metadata = dict(item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {})
        task_subtype = str(metadata.get("task", ""))
        anchor = anchor_map.get(item_id)
        if anchor is None:
            anchor = AnchorCandidate(
                item_id=item_id,
                output=_fallback_anchor_output(item, profile),
                source_kind="fallback",
                source_path="fallback",
            )
        anchor_obj = parse_answer_object(
            anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        oracle_obj = parse_answer_object(
            reference_answer,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        prepared = _build_local_adapter(anchor=anchor, anchor_obj=anchor_obj)
        trace = build_rollback_trace(
            prepared,
            question_text=question,
            candidate_text=anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        anchor_artifact = canonicalize_candidate(
            question_text=question,
            candidate_text=anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        candidates: List[RollbackTeacherCandidate] = []
        max_boundary = max(1, len(trace.steps))
        for boundary_index in range(0, max_boundary + 1):
            candidate_output, origin_kind = _candidate_output_for_boundary(
                boundary_index=boundary_index,
                max_boundary=max_boundary,
                profile=profile,
                task_subtype=task_subtype,
                metadata=metadata,
                anchor_output=anchor.output,
                anchor_obj=anchor_obj,
                reference_answer=reference_answer,
                oracle_obj=oracle_obj,
            )
            artifact = canonicalize_candidate(
                question_text=question,
                candidate_text=candidate_output,
                dataset_name=dataset_name,
                answer_format=profile.answer_format,
                task_subtype=task_subtype,
                metadata=metadata,
            )
            summary = output_evaluator.evaluate_output(
                question,
                candidate_output,
                tier="tier2",
                reference_answer=reference_answer,
                metadata=metadata,
                dataset_profile=profile,
            )
            emitted_node, rerun_nodes, q_sup_target, q_ans_target = _teacher_subgraph(
                prepared,
                trace,
                boundary_index=boundary_index,
            )
            candidates.append(
                RollbackTeacherCandidate(
                    candidate_id=f"{item_id}::b{boundary_index}",
                    boundary_index=boundary_index,
                    origin_kind=origin_kind,
                    emitted_node_id="__null__" if boundary_index == 0 else emitted_node,
                    output=candidate_output,
                    artifact=artifact,
                    eval_success=float(summary.mean_success),
                    eval_task_score=float(summary.mean_task_score),
                    typed_support_teacher=_teacher_typed_support(artifact, reference_signature=oracle_obj.signature),
                    contract_residual_teacher=float(artifact.contract_residual.pooled),
                    keep_teacher=_teacher_keep(anchor_artifact, artifact),
                    signature_distance_teacher=float(typed_distance(artifact.answer_object, anchor_artifact.answer_object)),
                    rerun_subgraph_node_ids=[] if boundary_index == 0 else rerun_nodes,
                    q_sup_target={} if boundary_index == 0 else q_sup_target,
                    q_ans_target={} if boundary_index == 0 else q_ans_target,
                )
            )
        phat_bank = _candidate_probabilities(candidates)
        best_candidate = max(candidates, key=_lexicographic_order_key)
        rerun_needed = int(best_candidate.boundary_index > 0)
        anchor_candidate = next(candidate for candidate in candidates if candidate.boundary_index == 0)
        samples.append(
            RollbackTeacherSample(
                id=item_id,
                dataset=dataset_name,
                answer_format=profile.answer_format,
                task_subtype=task_subtype,
                question=question,
                reference_answer=reference_answer,
                metadata=metadata,
                prepared=prepared,
                trace=trace,
                anchor_candidate=anchor_candidate,
                candidates=candidates,
                teacher_boundary=int(best_candidate.boundary_index),
                phat_bank=phat_bank,
                rerun_needed=rerun_needed,
                summary_features={
                    "trace_length": float(len(trace.steps)),
                    "candidate_count": float(len(candidates)),
                    "teacher_boundary": float(best_candidate.boundary_index),
                    "anchor_contract_valid": 1.0 if anchor_artifact.schema_features.get("contract_valid", False) else 0.0,
                    "anchor_typed_support": float(anchor_candidate.typed_support_teacher),
                    "oracle_signature_match": 1.0 if anchor_artifact.answer_signature == oracle_obj.signature else 0.0,
                },
            )
        )
    return samples


def build_online_rollout_samples(
    *,
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
) -> List[RollbackTeacherSample]:
    profile = resolve_dataset_profile(dataset_name)
    anchor_map = _load_anchor_candidates(dataset_name)
    samples: List[RollbackTeacherSample] = []
    for item in items:
        item_id = str(item.get("id", ""))
        question = str(item.get("question", ""))
        reference_answer = str(item.get("answer", ""))
        metadata = dict(item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {})
        metadata.setdefault("mas_answer_format", profile.answer_format)
        task_subtype = str(metadata.get("task", ""))
        anchor = anchor_map.get(item_id)
        if anchor is None:
            anchor = AnchorCandidate(
                item_id=item_id,
                output=_fallback_anchor_output(item, profile),
                source_kind="fallback",
                source_path="fallback",
            )
        anchor_obj = parse_answer_object(
            anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        prepared = _build_local_adapter(anchor=anchor, anchor_obj=anchor_obj)
        trace = build_rollback_trace(
            prepared,
            question_text=question,
            candidate_text=anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        anchor_artifact = canonicalize_candidate(
            question_text=question,
            candidate_text=anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        anchor_candidate = RollbackTeacherCandidate(
            candidate_id=f"{item_id}::b0",
            boundary_index=0,
            origin_kind="anchor",
            emitted_node_id="__null__",
            output=anchor.output,
            artifact=anchor_artifact,
            eval_success=float(anchor.stage1_success),
            eval_task_score=float(anchor.stage1_task_score),
        )
        samples.append(
            RollbackTeacherSample(
                id=item_id,
                dataset=dataset_name,
                answer_format=profile.answer_format,
                task_subtype=task_subtype,
                question=question,
                reference_answer=reference_answer,
                metadata=metadata,
                prepared=prepared,
                trace=trace,
                anchor_candidate=anchor_candidate,
                candidates=[anchor_candidate],
                teacher_boundary=0,
                phat_bank={anchor_candidate.candidate_id: 1.0},
                rerun_needed=0,
                summary_features={
                    "trace_length": float(len(trace.steps)),
                    "anchor_contract_valid": 1.0 if anchor_artifact.schema_features.get("contract_valid", False) else 0.0,
                    "anchor_typed_support": float(_teacher_typed_support(anchor_artifact, reference_signature=anchor_artifact.answer_signature)),
                },
            )
        )
    return samples


def blueprint_teacher_samples_to_jsonl(samples: Sequence[RollbackTeacherSample]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for sample in samples:
        rows.append(
            {
                "id": sample.id,
                "dataset": sample.dataset,
                "answer_format": sample.answer_format,
                "task_subtype": sample.task_subtype,
                "teacher_boundary": sample.teacher_boundary,
                "rerun_needed": sample.rerun_needed,
                "summary_features": sample.summary_features,
                "phat_bank": sample.phat_bank,
                "candidates": [
                    {
                        "candidate_id": candidate.candidate_id,
                        "boundary_index": candidate.boundary_index,
                        "origin_kind": candidate.origin_kind,
                        "emitted_node_id": candidate.emitted_node_id,
                        "eval_success": candidate.eval_success,
                        "eval_task_score": candidate.eval_task_score,
                        "teacher_rank": candidate.teacher_rank,
                        "bank_probability": candidate.bank_probability,
                        "typed_support_teacher": candidate.typed_support_teacher,
                        "contract_residual_teacher": candidate.contract_residual_teacher,
                        "keep_teacher": candidate.keep_teacher,
                        "signature_distance_teacher": candidate.signature_distance_teacher,
                        "answer_signature": candidate.artifact.answer_signature,
                    }
                    for candidate in sample.candidates
                ],
            }
        )
    return rows


def build_teacher_bank(
    *,
    dataset_name: str,
    items: Sequence[Dict[str, Any]],
    evaluator: Optional[MultiFidelityEvaluator] = None,
) -> List[TeacherBankRecord]:
    profile = resolve_dataset_profile(dataset_name)
    anchor_map = _load_anchor_candidates(dataset_name)
    output_evaluator = evaluator or _build_output_evaluator()
    records: List[TeacherBankRecord] = []
    for item in items:
        item_id = str(item.get("id", ""))
        question = str(item.get("question", ""))
        reference_answer = str(item.get("answer", ""))
        metadata = dict(item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {})
        task_subtype = str(metadata.get("task", ""))
        anchor = anchor_map.get(item_id)
        if anchor is None:
            anchor = AnchorCandidate(
                item_id=item_id,
                output=_fallback_anchor_output(item, profile),
                source_kind="fallback",
                source_path="fallback",
            )
        anchor_obj = parse_answer_object(
            anchor.output,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        oracle_obj = parse_answer_object(
            reference_answer,
            dataset_name=dataset_name,
            answer_format=profile.answer_format,
            task_subtype=task_subtype,
            metadata=metadata,
        )
        prepared = _build_local_adapter(anchor=anchor, anchor_obj=anchor_obj)
        anchor_summary = output_evaluator.evaluate_output(
            question,
            anchor.output,
            tier="tier2",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=profile,
        )
        oracle_summary = output_evaluator.evaluate_output(
            question,
            reference_answer,
            tier="tier2",
            reference_answer=reference_answer,
            metadata=metadata,
            dataset_profile=profile,
        )
        anchor_eval_success = float(anchor_summary.mean_success)
        anchor_eval_task_score = float(anchor_summary.mean_task_score)
        oracle_eval_success = float(oracle_summary.mean_success)
        oracle_eval_task_score = float(oracle_summary.mean_task_score)
        rerun_needed = int(
            _lexicographic_better(
                oracle_eval_success,
                oracle_eval_task_score,
                anchor_eval_success,
                anchor_eval_task_score,
            )
        )
        teacher_margin = (oracle_eval_success - anchor_eval_success) + 0.5 * (
            oracle_eval_task_score - anchor_eval_task_score
        )
        feature_map = _build_feature_map(
            profile=profile,
            task_subtype=task_subtype,
            anchor_obj=anchor_obj,
            anchor=anchor,
            prepared=prepared,
            anchor_eval_success=anchor_eval_success,
            anchor_eval_task_score=anchor_eval_task_score,
        )
        records.append(
            TeacherBankRecord(
                id=item_id,
                dataset=dataset_name,
                answer_format=profile.answer_format,
                task_subtype=task_subtype,
                question=question,
                reference_answer=reference_answer,
                anchor_output=anchor.output,
                oracle_output=reference_answer,
                anchor_source_kind=anchor.source_kind,
                anchor_source_path=anchor.source_path,
                structure_artifact_path=anchor.structure_artifact_path,
                anchor_contract_valid=bool(anchor_obj.fields.get("contract_valid")),
                anchor_recoverable_valid=bool(anchor_obj.fields.get("recoverable_valid")),
                anchor_signature=anchor_obj.signature,
                oracle_signature=oracle_obj.signature,
                anchor_kind=anchor_obj.kind,
                oracle_kind=oracle_obj.kind,
                stage1_success=float(anchor.stage1_success),
                stage1_task_score=float(anchor.stage1_task_score),
                stage1_reward=float(anchor.stage1_reward),
                anchor_eval_success=anchor_eval_success,
                anchor_eval_task_score=anchor_eval_task_score,
                oracle_eval_success=oracle_eval_success,
                oracle_eval_task_score=oracle_eval_task_score,
                rerun_needed=rerun_needed,
                teacher_margin=float(teacher_margin),
                feature_map=feature_map,
                rollback_prepared={
                    "has_base_prepared": prepared.base_prepared is not None,
                    "stage1_signature": prepared.anchor_artifact.get("stage1_signature", ""),
                    "answer_signature": prepared.anchor_artifact.get("answer_signature", ""),
                    "source_kind": prepared.anchor_replay_cache.get("source_kind", ""),
                },
            )
        )
    return records


def teacher_bank_to_jsonl(records: Sequence[TeacherBankRecord]) -> List[Dict[str, Any]]:
    return [asdict(record) for record in records]
