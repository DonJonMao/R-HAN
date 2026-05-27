from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .boundary import BoundaryOutput
from .diffusion import RollbackDiffusionModel, build_graph_matrices
from .verifier import build_candidate_artifact_and_state
from .train_bank import RollbackTeacherSample


@dataclass
class DiffusionTrainResult:
    model_state: Dict[str, torch.Tensor]
    train_loss_sup: float
    train_loss_ans: float
    count: int
    avg_alpha_sup_entropy: float
    avg_alpha_ans_entropy: float


def _teacher_boundary_output(sample: RollbackTeacherSample) -> BoundaryOutput:
    logits = torch.full((len(sample.trace.steps) + 1,), -12.0, dtype=torch.float32)
    logits[int(sample.teacher_boundary)] = 0.0
    probs = torch.softmax(logits, dim=0)
    return BoundaryOutput(
        step_features=torch.zeros((max(1, len(sample.trace.steps)), 32), dtype=torch.float32),
        macro_logits=torch.zeros(1, dtype=torch.float32),
        final_logits=logits,
        pi_tilde_rb=probs,
        predicted_boundary=int(sample.teacher_boundary),
    )


def _target_vector(node_ids: Sequence[str], target_map: Dict[str, float], *, device: torch.device) -> torch.Tensor:
    if not node_ids:
        return torch.ones(1, dtype=torch.float32, device=device)
    values = torch.tensor([float(target_map.get(node_id, 0.0)) for node_id in node_ids], dtype=torch.float32, device=device)
    if torch.sum(values) <= 0:
        values = torch.full_like(values, 1.0 / max(1, values.shape[0]))
    else:
        values = values / torch.sum(values)
    return values


def train_diffusion_model(
    samples: Sequence[RollbackTeacherSample],
    *,
    device: str,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> DiffusionTrainResult:
    if not samples:
        raise ValueError("Cannot train diffusion model with empty samples.")
    torch.manual_seed(int(seed))
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RollbackDiffusionModel().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    prepared = []
    for sample in samples:
        union_graph = sample.prepared.base_prepared.union_graph if sample.prepared.base_prepared is not None else None
        anchor_state = build_candidate_artifact_and_state(
            question_text=sample.question,
            candidate_text=sample.anchor_candidate.output,
            dataset_name=sample.dataset,
            answer_format=sample.answer_format,
            task_subtype=sample.task_subtype,
            metadata=sample.metadata,
            trace=sample.trace,
            union_graph=union_graph,
        )
        teacher_candidate = next((candidate for candidate in sample.candidates if candidate.boundary_index == sample.teacher_boundary), sample.anchor_candidate)
        prepared.append((sample, union_graph, anchor_state, teacher_candidate, _teacher_boundary_output(sample)))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        loss_terms = []
        for sample, union_graph, anchor_state, teacher_candidate, teacher_boundary in prepared:
            output = model(
                verifier_state=anchor_state,
                boundary_output=teacher_boundary,
                union_graph=union_graph,
                train_mode=True,
            )
            q_sup_target = _target_vector(output.node_ids, teacher_candidate.q_sup_target, device=target_device)
            q_ans_target = _target_vector(output.node_ids, teacher_candidate.q_ans_target, device=target_device)
            loss_sup = torch.sum(q_sup_target * (torch.log(q_sup_target + 1e-8) - torch.log(output.support.alpha_tilde + 1e-8)))
            loss_ans = torch.sum(q_ans_target * (torch.log(q_ans_target + 1e-8) - torch.log(output.answer.alpha_tilde + 1e-8)))
            loss_terms.append(loss_sup + loss_ans)
        loss = torch.stack(loss_terms).mean()
        loss.backward()
        optimizer.step()
    sup_losses = []
    ans_losses = []
    sup_entropy = 0.0
    ans_entropy = 0.0
    with torch.no_grad():
        for sample, union_graph, anchor_state, teacher_candidate, teacher_boundary in prepared:
            output = model(
                verifier_state=anchor_state,
                boundary_output=teacher_boundary,
                union_graph=union_graph,
                train_mode=True,
            )
            q_sup_target = _target_vector(output.node_ids, teacher_candidate.q_sup_target, device=target_device)
            q_ans_target = _target_vector(output.node_ids, teacher_candidate.q_ans_target, device=target_device)
            sup_losses.append(float(torch.sum(q_sup_target * (torch.log(q_sup_target + 1e-8) - torch.log(output.support.alpha_tilde + 1e-8))).item()))
            ans_losses.append(float(torch.sum(q_ans_target * (torch.log(q_ans_target + 1e-8) - torch.log(output.answer.alpha_tilde + 1e-8))).item()))
            sup_entropy += float((-torch.sum(output.support.alpha_tilde * torch.log(output.support.alpha_tilde + 1e-8))).item())
            ans_entropy += float((-torch.sum(output.answer.alpha_tilde * torch.log(output.answer.alpha_tilde + 1e-8))).item())
    return DiffusionTrainResult(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        train_loss_sup=float(sum(sup_losses) / max(1, len(sup_losses))),
        train_loss_ans=float(sum(ans_losses) / max(1, len(ans_losses))),
        count=len(samples),
        avg_alpha_sup_entropy=sup_entropy / max(1, len(samples)),
        avg_alpha_ans_entropy=ans_entropy / max(1, len(samples)),
    )
