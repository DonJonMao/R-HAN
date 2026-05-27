from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch

from .boundary import RollbackBoundaryModel, build_boundary_features
from .verifier import build_candidate_artifact_and_state
from .train_bank import RollbackTeacherSample


@dataclass
class BoundaryTrainResult:
    model_state: Dict[str, torch.Tensor]
    train_loss: float
    train_accuracy: float
    count: int
    no_rerun_rate: float
    avg_predicted_boundary: float


def train_boundary_model(
    samples: Sequence[RollbackTeacherSample],
    *,
    device: str,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> BoundaryTrainResult:
    if not samples:
        raise ValueError("Cannot train boundary model with empty samples.")
    torch.manual_seed(int(seed))
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = RollbackBoundaryModel().to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    losses: List[torch.Tensor] = []
    train_pairs = []
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
        train_pairs.append((sample, build_boundary_features(anchor_state, sample.trace)))
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        losses.clear()
        for sample, features in train_pairs:
            output = model(features)
            target = torch.tensor(int(sample.teacher_boundary), dtype=torch.long, device=target_device)
            losses.append(-torch.log(output.pi_tilde_rb[target] + 1e-8))
        loss = torch.stack(losses).mean()
        loss.backward()
        optimizer.step()
    correct = 0
    no_rerun = 0
    predicted_sum = 0.0
    with torch.no_grad():
        eval_losses = []
        for sample, features in train_pairs:
            output = model(features)
            target = int(sample.teacher_boundary)
            eval_losses.append(float((-torch.log(output.pi_tilde_rb[target] + 1e-8)).item()))
            predicted = int(output.predicted_boundary)
            correct += int(predicted == target)
            no_rerun += int(predicted == 0)
            predicted_sum += float(predicted)
    return BoundaryTrainResult(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        train_loss=float(sum(eval_losses) / max(1, len(eval_losses))),
        train_accuracy=float(correct) / float(max(1, len(samples))),
        count=len(samples),
        no_rerun_rate=float(no_rerun) / float(max(1, len(samples))),
        avg_predicted_boundary=predicted_sum / float(max(1, len(samples))),
    )
