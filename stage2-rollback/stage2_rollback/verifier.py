from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import torch

from .artifacts import ArtifactIR, ArtifactUnit, canonicalize_candidate
from .contracts import ContractResidual, contract_residual, signature_embedding
from .trace_ir import RollbackTrace, TraceStep


def _safe_mean(values: Sequence[float]) -> float:
    return float(sum(values)) / float(max(1, len(values)))


def _normalize(text: str) -> str:
    return " ".join(str(text or "").lower().split())


def _text_features(text: str) -> List[float]:
    raw = str(text or "")
    stripped = raw.strip()
    if not stripped:
        return [0.0] * 8
    digits = sum(ch.isdigit() for ch in stripped)
    alphas = sum(ch.isalpha() for ch in stripped)
    braces = sum(ch in "{}[]()" for ch in stripped)
    colons = stripped.count(":")
    underscores = stripped.count("_")
    return [
        float(len(stripped)),
        float(digits) / float(len(stripped)),
        float(alphas) / float(len(stripped)),
        float(braces) / float(len(stripped)),
        float(colons) / float(len(stripped)),
        float(underscores) / float(len(stripped)),
        1.0 if "return" in stripped else 0.0,
        1.0 if any(token in stripped.lower() for token in ("answer", "option", "boxed", "max_flow")) else 0.0,
    ]


def _unit_vector(unit: ArtifactUnit) -> torch.Tensor:
    return torch.tensor(
        _text_features(unit.text)
        + [
            float(unit.position),
            float(unit.role_probs.get("answer", 0.0)),
            float(unit.role_probs.get("evidence", 0.0)),
            float(unit.role_probs.get("mixed", 0.0)),
            float(unit.features.get("contains_answer_surface", 0.0)),
            float(unit.features.get("position_ratio", 0.0)),
        ],
        dtype=torch.float32,
    )


def _step_vector(step: TraceStep) -> torch.Tensor:
    state = list(step.state_vector)[:8]
    if len(state) < 8:
        state = state + [0.0] * (8 - len(state))
    residual = step.contract_residual
    return torch.tensor(
        _text_features(step.output_summary)
        + state
        + [
            float(step.local_signal.get("support_count", 0.0)),
            float(step.local_signal.get("avg_graph_score", 0.0)),
            float(step.local_signal.get("root_frequency", 0.0)),
            float(step.local_signal.get("sink_frequency", 0.0)),
            float(residual.get("parse", 0.0)),
            float(residual.get("completeness", 0.0)),
            float(residual.get("constraint", 0.0)),
            float(residual.get("execution", 0.0)),
        ],
        dtype=torch.float32,
    )


def _node_vector(node_state: Sequence[float], node_text: str = "") -> torch.Tensor:
    state = list(node_state)[:8]
    if len(state) < 8:
        state = state + [0.0] * (8 - len(state))
    return torch.tensor(_text_features(node_text) + state, dtype=torch.float32)


def _softmax_scores(query: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
    if keys.numel() == 0:
        return torch.zeros((query.shape[0], 0), dtype=torch.float32)
    logits = query @ keys.T
    return torch.softmax(logits, dim=-1)


def _pooled(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    if not vectors:
        return torch.zeros(1, dtype=torch.float32)
    return torch.stack(list(vectors), dim=0).mean(dim=0)


def _pad_last_dim(tensor: torch.Tensor, width: int) -> torch.Tensor:
    if tensor.shape[-1] >= width:
        return tensor[..., :width]
    pad = torch.zeros((*tensor.shape[:-1], width - tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=-1)


def _support_mentions_answer(artifact: ArtifactIR) -> float:
    answer_surface = _normalize(str(artifact.answer_object.fields.get("recoverable_value", artifact.answer_object.value) or ""))
    if not answer_surface:
        return 0.0
    evidence_text = " ".join(
        unit.text
        for unit in artifact.units
        if unit.unit_id in set(artifact.evidence_unit_ids) | set(artifact.mixed_unit_ids)
    )
    normalized = _normalize(evidence_text)
    return 1.0 if answer_surface and answer_surface in normalized else 0.0


def _typed_support_bootstrap(
    artifact: ArtifactIR,
    *,
    residual: ContractResidual,
) -> float:
    contract_valid = float(artifact.schema_features.get("contract_valid", False))
    recoverable_valid = float(artifact.schema_features.get("recoverable_valid", False))
    support_hit = _support_mentions_answer(artifact)
    answer_mass = _safe_mean(
        [unit.role_probs.get("answer", 0.0) for unit in artifact.units if unit.unit_id in artifact.answer_unit_ids]
    )
    evidence_mass = _safe_mean(
        [unit.role_probs.get("evidence", 0.0) for unit in artifact.units if unit.unit_id in artifact.evidence_unit_ids]
    )
    raw = 0.30 * contract_valid + 0.20 * recoverable_valid + 0.25 * support_hit + 0.15 * answer_mass + 0.10 * evidence_mass
    raw += 0.25 * (1.0 - residual.pooled)
    return max(0.0, min(1.0, raw))


@dataclass
class UnitLatents:
    unit_vectors: torch.Tensor
    trace_vectors: torch.Tensor
    node_vectors: torch.Tensor
    unit_to_trace: torch.Tensor
    unit_to_nodes: torch.Tensor


@dataclass
class VerifierState:
    artifact: ArtifactIR
    unit_latents: UnitLatents
    correctness: torch.Tensor
    support: torch.Tensor
    keep: torch.Tensor
    epsilon: torch.Tensor
    mu: torch.Tensor
    kappa: torch.Tensor
    role_ans: torch.Tensor
    role_evd: torch.Tensor
    role_mix: torch.Tensor
    typed_support_score: float
    contract_residual: ContractResidual
    summary_vector: torch.Tensor
    answer_pool: torch.Tensor
    evidence_pool: torch.Tensor
    role_pool: torch.Tensor
    extra_features: Dict[str, float] = field(default_factory=dict)


def build_verifier_state(
    artifact: ArtifactIR,
    trace: RollbackTrace,
    *,
    union_graph: Optional[Any] = None,
) -> VerifierState:
    unit_vectors = torch.stack([_unit_vector(unit) for unit in artifact.units], dim=0) if artifact.units else torch.zeros((1, 14))
    trace_vectors = torch.stack(
        [torch.zeros(24, dtype=torch.float32)] + [_step_vector(step) for step in trace.steps],
        dim=0,
    )
    if union_graph is not None and getattr(union_graph, "nodes", None):
        node_vectors = torch.stack(
            [_node_vector(node.state_vector, f"{node.role}:{node.agent_id}") for node in union_graph.nodes.values()],
            dim=0,
        )
    else:
        node_vectors = torch.zeros((1, 16), dtype=torch.float32)
    # Match vector widths for soft alignment.
    unit_for_align = _pad_last_dim(unit_vectors, 16)
    trace_for_align = _pad_last_dim(trace_vectors, 16)
    node_for_align = _pad_last_dim(node_vectors, 16)
    unit_to_trace = _softmax_scores(unit_for_align, trace_for_align)
    unit_to_nodes = _softmax_scores(unit_for_align, node_for_align)

    answer_ids = set(artifact.answer_unit_ids)
    evidence_ids = set(artifact.evidence_unit_ids)
    correctness: List[float] = []
    support: List[float] = []
    keep: List[float] = []
    for unit in artifact.units:
        role_answer = float(unit.role_probs.get("answer", 0.0))
        role_evidence = float(unit.role_probs.get("evidence", 0.0))
        parse_bonus = 1.0 - artifact.contract_residual.parse
        correct = 0.45 + 0.25 * parse_bonus + 0.20 * role_answer + 0.10 * unit.features.get("contains_answer_surface", 0.0)
        if unit.unit_id in answer_ids and not artifact.schema_features.get("contract_valid", False):
            correct -= 0.25
        evidence_score = 0.25 + 0.35 * role_evidence + 0.20 * (1.0 - artifact.contract_residual.completeness)
        if unit.unit_id in evidence_ids:
            evidence_score += 0.15
        keep_score = 0.20 + 0.30 * (1.0 - float(unit.position) / max(1.0, float(len(artifact.units)))) + 0.25 * unit.role_probs.get("mixed", 0.0)
        if unit.unit_id in answer_ids:
            keep_score += 0.15
        correctness.append(max(0.0, min(1.0, correct)))
        support.append(max(0.0, min(1.0, evidence_score)))
        keep.append(max(0.0, min(1.0, keep_score)))
    correctness_t = torch.tensor(correctness, dtype=torch.float32)
    support_t = torch.tensor(support, dtype=torch.float32)
    keep_t = torch.tensor(keep, dtype=torch.float32)
    epsilon = 1.0 - correctness_t
    mu = 1.0 - support_t
    kappa = keep_t
    role_ans = torch.tensor([unit.role_probs.get("answer", 0.0) for unit in artifact.units], dtype=torch.float32)
    role_evd = torch.tensor([unit.role_probs.get("evidence", 0.0) for unit in artifact.units], dtype=torch.float32)
    role_mix = torch.tensor([unit.role_probs.get("mixed", 0.0) for unit in artifact.units], dtype=torch.float32)
    answer_vectors = [unit_vectors[idx] for idx, unit in enumerate(artifact.units) if unit.unit_id in answer_ids]
    evidence_vectors = [unit_vectors[idx] for idx, unit in enumerate(artifact.units) if unit.unit_id in evidence_ids]
    answer_pool = _pooled(answer_vectors)
    evidence_pool = _pooled(evidence_vectors)
    role_pool = torch.tensor(
        [
            float(role_ans.mean().item()) if role_ans.numel() else 0.0,
            float(role_evd.mean().item()) if role_evd.numel() else 0.0,
            float(role_mix.mean().item()) if role_mix.numel() else 0.0,
        ],
        dtype=torch.float32,
    )
    sig_embed = torch.tensor(signature_embedding(artifact.answer_object), dtype=torch.float32)
    pooled_residual = torch.tensor(
        [
            artifact.contract_residual.parse,
            artifact.contract_residual.completeness,
            artifact.contract_residual.constraint,
            artifact.contract_residual.execution,
        ],
        dtype=torch.float32,
    )
    typed_support = _typed_support_bootstrap(artifact, residual=artifact.contract_residual)
    summary_vector = torch.cat(
        [
            answer_pool[:8],
            evidence_pool[:8],
            role_pool,
            sig_embed[:8],
            pooled_residual,
            torch.tensor([typed_support], dtype=torch.float32),
        ],
        dim=0,
    )
    return VerifierState(
        artifact=artifact,
        unit_latents=UnitLatents(
            unit_vectors=unit_vectors,
            trace_vectors=trace_vectors,
            node_vectors=node_vectors,
            unit_to_trace=unit_to_trace,
            unit_to_nodes=unit_to_nodes,
        ),
        correctness=correctness_t,
        support=support_t,
        keep=keep_t,
        epsilon=epsilon,
        mu=mu,
        kappa=kappa,
        role_ans=role_ans,
        role_evd=role_evd,
        role_mix=role_mix,
        typed_support_score=typed_support,
        contract_residual=artifact.contract_residual,
        summary_vector=summary_vector,
        answer_pool=answer_pool,
        evidence_pool=evidence_pool,
        role_pool=role_pool,
        extra_features={
            "typed_support_score": typed_support,
            "contract_residual": artifact.contract_residual.pooled,
            "answer_unit_count": float(len(artifact.answer_unit_ids)),
            "evidence_unit_count": float(len(artifact.evidence_unit_ids)),
        },
    )


def build_candidate_artifact_and_state(
    *,
    question_text: str,
    candidate_text: str,
    dataset_name: str,
    answer_format: str,
    task_subtype: str,
    metadata: Optional[Dict[str, Any]],
    trace: RollbackTrace,
    union_graph: Optional[Any],
) -> VerifierState:
    artifact = canonicalize_candidate(
        question_text=question_text,
        candidate_text=candidate_text,
        dataset_name=dataset_name,
        answer_format=answer_format,
        task_subtype=task_subtype,
        metadata=metadata,
    )
    return build_verifier_state(artifact, trace, union_graph=union_graph)
