from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import torch

from .boundary import BoundaryOutput
from .diffusion import RollbackDiffusionModel
from .emitter import RollbackEmitterModel
from .selector import RollbackSelectorModel
from .verifier import build_candidate_artifact_and_state
from .train_bank import RollbackTeacherSample


@dataclass
class SelectorTrainResult:
    emitter_state: Dict[str, torch.Tensor]
    selector_state: Dict[str, torch.Tensor]
    train_loss_emit: float
    train_loss_sel: float
    count: int
    null_emit_rate: float
    anchor_select_rate: float


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


def train_selector_model(
    samples: Sequence[RollbackTeacherSample],
    *,
    device: str,
    seed: int,
    diffusion_state: Dict[str, torch.Tensor] | None = None,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> SelectorTrainResult:
    if not samples:
        raise ValueError("Cannot train selector model with empty samples.")
    torch.manual_seed(int(seed))
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    diffusion_model = RollbackDiffusionModel().to(target_device)
    if diffusion_state:
        diffusion_model.load_state_dict(diffusion_state, strict=False)
    diffusion_model.eval()
    emitter_model = RollbackEmitterModel().to(target_device)
    selector_model = RollbackSelectorModel().to(target_device)
    optimizer = torch.optim.Adam(list(emitter_model.parameters()) + list(selector_model.parameters()), lr=float(learning_rate))
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
        candidate_states = [
            build_candidate_artifact_and_state(
                question_text=sample.question,
                candidate_text=candidate.output,
                dataset_name=sample.dataset,
                answer_format=sample.answer_format,
                task_subtype=sample.task_subtype,
                metadata=sample.metadata,
                trace=sample.trace,
                union_graph=union_graph,
            )
            for candidate in sample.candidates
            if candidate.boundary_index > 0
        ]
        prepared.append((sample, union_graph, anchor_state, candidate_states, _teacher_boundary_output(sample)))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for sample, union_graph, anchor_state, candidate_states, teacher_boundary in prepared:
            with torch.no_grad():
                diffusion_output = diffusion_model(
                    verifier_state=anchor_state,
                    boundary_output=teacher_boundary,
                    union_graph=union_graph,
                    train_mode=True,
                )
            emitter_output = emitter_model(
                diffusion_output=diffusion_output,
                boundary_index=sample.teacher_boundary,
                anchor_typed_support=anchor_state.typed_support_score,
                train_mode=True,
            )
            selector_output = selector_model(anchor_state, candidate_states)
            emit_target = torch.zeros_like(emitter_output.gamma_tilde_plus)
            for candidate in sample.candidates:
                prob = float(sample.phat_bank.get(candidate.candidate_id, 0.0))
                if candidate.boundary_index == 0 or candidate.emitted_node_id == "__null__":
                    emit_target[-1] += prob
                    continue
                if candidate.emitted_node_id in diffusion_output.node_ids:
                    emit_target[diffusion_output.node_ids.index(candidate.emitted_node_id)] += prob
                else:
                    emit_target[-1] += prob
            emit_target = emit_target / (torch.sum(emit_target) + 1e-8)
            sel_target = torch.tensor(
                [
                    float(sample.phat_bank.get(sample.anchor_candidate.candidate_id, 0.0))
                ]
                + [
                    float(sample.phat_bank.get(candidate.candidate_id, 0.0))
                    for candidate in sample.candidates
                    if candidate.boundary_index > 0
                ],
                dtype=torch.float32,
                device=target_device,
            )
            sel_target = sel_target / (torch.sum(sel_target) + 1e-8)
            loss_emit = torch.sum(emit_target * (torch.log(emit_target + 1e-8) - torch.log(emitter_output.gamma_tilde_plus + 1e-8)))
            loss_sel = torch.sum(sel_target * (torch.log(sel_target + 1e-8) - torch.log(selector_output.probabilities + 1e-8)))
            losses.append(loss_emit + loss_sel)
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    emit_losses = []
    sel_losses = []
    null_emit = 0
    anchor_win = 0
    with torch.no_grad():
        for sample, union_graph, anchor_state, candidate_states, teacher_boundary in prepared:
            diffusion_output = diffusion_model(
                verifier_state=anchor_state,
                boundary_output=teacher_boundary,
                union_graph=union_graph,
                train_mode=True,
            )
            emitter_output = emitter_model(
                diffusion_output=diffusion_output,
                boundary_index=sample.teacher_boundary,
                anchor_typed_support=anchor_state.typed_support_score,
                train_mode=True,
            )
            selector_output = selector_model(anchor_state, candidate_states)
            emit_target = torch.zeros_like(emitter_output.gamma_tilde_plus)
            for candidate in sample.candidates:
                prob = float(sample.phat_bank.get(candidate.candidate_id, 0.0))
                if candidate.boundary_index == 0 or candidate.emitted_node_id == "__null__":
                    emit_target[-1] += prob
                    continue
                if candidate.emitted_node_id in diffusion_output.node_ids:
                    emit_target[diffusion_output.node_ids.index(candidate.emitted_node_id)] += prob
                else:
                    emit_target[-1] += prob
            emit_target = emit_target / (torch.sum(emit_target) + 1e-8)
            sel_target = torch.tensor(
                [
                    float(sample.phat_bank.get(sample.anchor_candidate.candidate_id, 0.0))
                ]
                + [
                    float(sample.phat_bank.get(candidate.candidate_id, 0.0))
                    for candidate in sample.candidates
                    if candidate.boundary_index > 0
                ],
                dtype=torch.float32,
                device=target_device,
            )
            sel_target = sel_target / (torch.sum(sel_target) + 1e-8)
            emit_losses.append(float(torch.sum(emit_target * (torch.log(emit_target + 1e-8) - torch.log(emitter_output.gamma_tilde_plus + 1e-8))).item()))
            sel_losses.append(float(torch.sum(sel_target * (torch.log(sel_target + 1e-8) - torch.log(selector_output.probabilities + 1e-8))).item()))
            null_emit += int(int(torch.argmax(emitter_output.gamma_plus).item()) == len(emitter_output.gamma_plus) - 1)
            anchor_win += int(selector_output.winner_index == 0)
    return SelectorTrainResult(
        emitter_state={key: value.detach().cpu() for key, value in emitter_model.state_dict().items()},
        selector_state={key: value.detach().cpu() for key, value in selector_model.state_dict().items()},
        train_loss_emit=float(sum(emit_losses) / max(1, len(emit_losses))),
        train_loss_sel=float(sum(sel_losses) / max(1, len(sel_losses))),
        count=len(samples),
        null_emit_rate=float(null_emit) / float(max(1, len(samples))),
        anchor_select_rate=float(anchor_win) / float(max(1, len(samples))),
    )
