from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import torch
from torch import nn

from .runtime import RollbackRuntime
from .train_bank import RollbackTeacherSample, TeacherBankRecord


@dataclass
class VerifierTrainResult:
    model_state: Dict[str, torch.Tensor]
    feature_names: List[str]
    train_loss: float
    train_accuracy: float
    positive_rate: float
    count: int


@dataclass
class BlueprintVerifierTrainResult:
    model_state: Dict[str, torch.Tensor]
    feature_names: List[str]
    train_loss: float
    train_accuracy: float
    positive_rate: float
    count: int
    avg_anchor_typed_support: float


def _feature_schema(records: Sequence[TeacherBankRecord]) -> List[str]:
    keys = set()
    for record in records:
        keys.update(record.feature_map.keys())
    return sorted(keys)


def _vectorize(records: Sequence[TeacherBankRecord], feature_names: Sequence[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[float]] = []
    labels: List[float] = []
    for record in records:
        rows.append([float(record.feature_map.get(name, 0.0)) for name in feature_names])
        labels.append(float(record.rerun_needed))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


def _vectorize_blueprint(
    samples: Sequence[RollbackTeacherSample],
    feature_names: Sequence[str],
    runtime: RollbackRuntime,
) -> Tuple[torch.Tensor, torch.Tensor]:
    rows: List[List[float]] = []
    labels: List[float] = []
    for sample in samples:
        result = runtime.analyze(sample, train_mode=True)
        feature_map: Dict[str, float] = dict(sample.summary_features)
        feature_map["anchor_typed_support_runtime"] = float(result.anchor_state.typed_support_score)
        feature_map["anchor_contract_runtime"] = float(result.anchor_state.contract_residual.pooled)
        feature_map["trace_length_runtime"] = float(len(sample.trace.steps))
        feature_map["boundary_teacher"] = float(sample.teacher_boundary)
        feature_map["candidate_count_runtime"] = float(len(sample.candidates))
        rows.append([float(feature_map.get(name, 0.0)) for name in feature_names])
        labels.append(float(sample.rerun_needed))
    return torch.tensor(rows, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)


class RollbackVerifierNet(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        hidden = max(16, min(128, input_dim * 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs).squeeze(-1)


def train_verifier_model(
    records: Sequence[TeacherBankRecord],
    *,
    device: str,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> VerifierTrainResult:
    if not records:
        raise ValueError("Cannot train verifier with empty teacher bank.")
    torch.manual_seed(int(seed))
    feature_names = _feature_schema(records)
    inputs_cpu, labels_cpu = _vectorize(records, feature_names)
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = inputs_cpu.to(target_device)
    labels = labels_cpu.to(target_device)
    model = RollbackVerifierNet(inputs.shape[1]).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits, labels).item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        accuracy = float((preds == labels).float().mean().item())
        positive_rate = float(labels.mean().item())
    return VerifierTrainResult(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        feature_names=feature_names,
        train_loss=float(loss),
        train_accuracy=accuracy,
        positive_rate=positive_rate,
        count=len(records),
    )


def train_blueprint_verifier_model(
    samples: Sequence[RollbackTeacherSample],
    *,
    device: str,
    seed: int,
    epochs: int = 80,
    learning_rate: float = 1e-3,
) -> BlueprintVerifierTrainResult:
    if not samples:
        raise ValueError("Cannot train blueprint verifier with empty samples.")
    torch.manual_seed(int(seed))
    runtime = RollbackRuntime()
    feature_names = sorted(
        {
            *{key for sample in samples for key in sample.summary_features.keys()},
            "anchor_typed_support_runtime",
            "anchor_contract_runtime",
            "trace_length_runtime",
            "boundary_teacher",
            "candidate_count_runtime",
        }
    )
    inputs_cpu, labels_cpu = _vectorize_blueprint(samples, feature_names, runtime)
    target_device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    inputs = inputs_cpu.to(target_device)
    labels = labels_cpu.to(target_device)
    model = RollbackVerifierNet(inputs.shape[1]).to(target_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    criterion = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        loss = criterion(logits, labels).item()
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        accuracy = float((preds == labels).float().mean().item())
        positive_rate = float(labels.mean().item())
    avg_anchor_typed_support = float(
        sum(RollbackRuntime().analyze(sample, train_mode=True).anchor_state.typed_support_score for sample in samples) / max(1, len(samples))
    )
    return BlueprintVerifierTrainResult(
        model_state={key: value.detach().cpu() for key, value in model.state_dict().items()},
        feature_names=feature_names,
        train_loss=float(loss),
        train_accuracy=accuracy,
        positive_rate=positive_rate,
        count=len(samples),
        avg_anchor_typed_support=avg_anchor_typed_support,
    )
