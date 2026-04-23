from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from mas_treesearch.profiles import DatasetProfile
from mas_treesearch.types import UnionEdge, UnionNode

from .types import ControllerState, MemoryRecord


def _dot(weights: Dict[str, float], features: Dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


@dataclass
class OnlineLinearModel:
    learning_rate: float = 0.03
    init_uncertainty: float = 0.20
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.5
    steps: int = 0
    residual_ema: float = 0.20

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        mean = max(0.0, min(1.0, self.bias + _dot(self.weights, features)))
        uncertainty = max(0.03, self.residual_ema / math.sqrt(max(1, self.steps)))
        return mean, max(self.init_uncertainty if self.steps == 0 else uncertainty, 0.03)

    def update(self, features: Dict[str, float], target: float) -> None:
        pred, _ = self.predict(features)
        error = pred - target
        self.bias -= self.learning_rate * error
        for name, value in features.items():
            self.weights[name] = self.weights.get(name, 0.0) - self.learning_rate * error * value
        self.residual_ema = 0.9 * self.residual_ema + 0.1 * abs(error)
        self.steps += 1

    def state_dict(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "init_uncertainty": self.init_uncertainty,
            "weights": dict(self.weights),
            "bias": self.bias,
            "steps": self.steps,
            "residual_ema": self.residual_ema,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.init_uncertainty = float(state.get("init_uncertainty", self.init_uncertainty))
        raw_weights = state.get("weights", {})
        self.weights = {str(name): float(value) for name, value in dict(raw_weights).items()}
        self.bias = float(state.get("bias", self.bias))
        self.steps = int(state.get("steps", self.steps))
        self.residual_ema = float(state.get("residual_ema", self.residual_ema))


def selector_features(
    node: UnionNode,
    record: MemoryRecord,
    controller_state: ControllerState,
    *,
    current_turn: int,
    query_similarity: float,
) -> Dict[str, float]:
    recency = 1.0 / max(1.0, 1.0 + (current_turn - record.turn_index))
    return {
        "bias": 1.0,
        f"node_role::{node.role}": 1.0,
        f"record_role::{record.role}": 1.0,
        f"record_type::{record.record_type}": 1.0,
        f"feedback::{record.feedback_type}": 1.0,
        f"mode::{controller_state.mode}": 1.0,
        "query_similarity": query_similarity,
        "recency": recency,
        "role_match": 1.0 if record.role == node.role else 0.0,
        "confidence": float(record.confidence),
        "uncertainty": float(controller_state.uncertainty),
        "token_penalty": min(1.0, record.token_estimate / 120.0),
    }


def slot_features(
    node: UnionNode,
    controller_state: ControllerState,
    *,
    current_turn: int,
    slot_name: str,
    runtime_node_type: str,
    candidate_count: int,
    stable_count: int,
    failure_count: int,
    core_slot: bool,
    learnable_slot: bool,
) -> Dict[str, float]:
    """Slot-level features for deciding whether a typed memory slot should speak."""
    metadata = dict(node.metadata or {})
    controller_metadata = dict(getattr(controller_state, "metadata", {}) or {})
    return {
        "bias": 1.0,
        f"runtime_node_type::{runtime_node_type}": 1.0,
        f"slot::{slot_name}": 1.0,
        f"node_role::{node.role}": 1.0,
        "turn_index": float(current_turn),
        "uncertainty": float(controller_state.uncertainty),
        "role_weight": float(controller_state.role_weights.get(str(node.role), 1.0)),
        "support_count": float(controller_metadata.get("support_count", 0.0)),
        "challenge_count": float(controller_metadata.get("challenge_count", 0.0)),
        "uncertain_count": float(controller_metadata.get("uncertain_count", 0.0)),
        "active_edge_ratio": float(controller_metadata.get("active_edge_ratio", 0.0)),
        "candidate_count": float(candidate_count),
        "has_candidates": 1.0 if candidate_count > 0 else 0.0,
        "recent_stable_evidence": float(stable_count),
        "recent_failure_evidence": float(failure_count),
        "core_slot": 1.0 if core_slot else 0.0,
        "learnable_slot": 1.0 if learnable_slot else 0.0,
        "is_sink_runtime": 1.0 if bool(metadata.get("is_sink_runtime", False)) else 0.0,
        "in_recovery_chain": 1.0 if bool(metadata.get("in_recovery_chain", False)) else 0.0,
        "sink_distance": float(metadata.get("sink_distance", 0.0) or 0.0),
    }


def edge_features(
    edge: UnionEdge,
    *,
    src_role: str,
    dst_role: str,
    controller_state: ControllerState,
    support_count: int,
    challenge_count: int,
) -> Dict[str, float]:
    return {
        "bias": 1.0,
        f"src_role::{src_role}": 1.0,
        f"dst_role::{dst_role}": 1.0,
        f"mode::{controller_state.mode}": 1.0,
        "initial_keep_logit": float(edge.initial_keep_logit),
        "support_ratio": float(edge.support_ratio),
        "avg_parent_score": float(edge.avg_parent_score),
        "best_parent_score": float(edge.best_parent_score),
        "src_role_weight": float(controller_state.role_weights.get(src_role, 1.0)),
        "dst_role_weight": float(controller_state.role_weights.get(dst_role, 1.0)),
        "support_count": float(support_count),
        "challenge_count": float(challenge_count),
        "uncertainty": float(controller_state.uncertainty),
    }


def controller_features(
    role: str,
    dataset_profile: DatasetProfile,
    *,
    controller_state: ControllerState,
    turn_index: int,
    total_turns: int,
    support_count: int,
    challenge_count: int,
    uncertain_count: int,
) -> Dict[str, float]:
    progress = float(turn_index + 1) / max(1.0, float(total_turns))
    return {
        "bias": 1.0,
        f"role::{role}": 1.0,
        f"task::{dataset_profile.task_type}": 1.0,
        f"mode::{controller_state.mode}": 1.0,
        "progress": progress,
        "uncertainty": float(controller_state.uncertainty),
        "support_count": float(support_count),
        "challenge_count": float(challenge_count),
        "uncertain_count": float(uncertain_count),
        "current_role_weight": float(controller_state.role_weights.get(role, 1.0)),
    }
