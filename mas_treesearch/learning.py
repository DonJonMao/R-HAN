from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence, Tuple

from .agents import AgentPool
from .gating import cosine
from .types import ArchitectureState, CompiledArchitecture, EditAction, Vector, WorkflowTemplate


def _dot(weights: Dict[str, float], features: Dict[str, float]) -> float:
    return sum(weights.get(name, 0.0) * value for name, value in features.items())


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


@dataclass
class FeatureBuilder:
    agent_pool: AgentPool
    agent_vectors: Dict[str, Vector]

    def __post_init__(self) -> None:
        self._by_id = self.agent_pool.by_id()

    def _alignment(self, question_vector: Sequence[float], agent_ids: Sequence[str]) -> float:
        if not agent_ids:
            return 0.0
        vals = [cosine(question_vector, self.agent_vectors[aid]) for aid in agent_ids if aid in self.agent_vectors]
        return sum(vals) / max(1, len(vals))

    def _diversity(self, agent_ids: Sequence[str]) -> float:
        if len(agent_ids) <= 1:
            return 0.0
        pairs: List[float] = []
        for idx, aid in enumerate(agent_ids):
            for bid in agent_ids[idx + 1 :]:
                pairs.append(1.0 - cosine(self.agent_vectors[aid], self.agent_vectors[bid]))
        return sum(pairs) / max(1, len(pairs))

    def state_features(self, compiled: CompiledArchitecture, question_vector: Sequence[float]) -> Dict[str, float]:
        state = compiled.state
        active = state.active_agents()
        template = state.template
        prompt_complexity = sum(slot.complexity() for slot in state.role_to_prompt.values()) / max(
            1, len(state.role_to_prompt)
        )
        features: Dict[str, float] = {
            "bias": 1.0,
            "num_agents": float(len(active)),
            "num_edges": float(len(compiled.edges)),
            "alignment": self._alignment(question_vector, active),
            "diversity": self._diversity(active),
            "prompt_complexity": prompt_complexity,
            "has_verifier": 1.0 if any("verification" in self._by_id[aid].capabilities for aid in active) else 0.0,
        }
        for name in WorkflowTemplate:
            features[f"template::{name.value}"] = 1.0 if template == name else 0.0
        for role, aid in state.role_to_agent.items():
            if aid in self._by_id:
                for cap in self._by_id[aid].capabilities:
                    features[f"rolecap::{role}::{cap}"] = 1.0
        return features

    def action_features(
        self,
        state: ArchitectureState,
        action: EditAction,
        question_vector: Sequence[float],
    ) -> Dict[str, float]:
        role_to_agent = dict(state.role_to_agent)
        prompt_complexity_before = sum(slot.complexity() for slot in state.role_to_prompt.values()) / max(
            1, len(state.role_to_prompt)
        )
        features: Dict[str, float] = {
            "bias": 1.0,
            "action::stop": 1.0 if action.kind == "stop" else 0.0,
            "action::change_template": 1.0 if action.kind == "change_template" else 0.0,
            "action::swap_agent": 1.0 if action.kind == "swap_agent" else 0.0,
            "action::set_prompt_slot": 1.0 if action.kind == "set_prompt_slot" else 0.0,
            "prompt_complexity_before": prompt_complexity_before,
        }
        if action.kind == "swap_agent":
            role = action.payload["role"]
            agent_id = action.payload["agent_id"]
            features[f"swap_role::{role}"] = 1.0
            features["swap_alignment"] = cosine(question_vector, self.agent_vectors.get(agent_id, []))
            if agent_id in self._by_id:
                for cap in self._by_id[agent_id].capabilities:
                    features[f"swap_cap::{cap}"] = 1.0
        elif action.kind == "change_template":
            features[f"template_to::{action.payload['template']}"] = 1.0
        elif action.kind == "set_prompt_slot":
            features[f"slot::{action.payload['slot']}"] = 1.0
            features[f"slot_value::{action.payload['value']}"] = 1.0
        active = list(dict.fromkeys(role_to_agent.values()))
        features["state_alignment"] = self._alignment(question_vector, active)
        features["state_diversity"] = self._diversity(active)
        features["state_num_agents"] = float(len(active))
        return features


@dataclass
class LearnableEditPrior:
    learning_rate: float = 0.05
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    steps: int = 0

    def score(self, features: Dict[str, float]) -> float:
        raw = self.bias + _dot(self.weights, features)
        return _sigmoid(raw)

    def update(self, features: Dict[str, float], target: float) -> None:
        pred = self.score(features)
        error = pred - target
        self.bias -= self.learning_rate * error
        for name, value in features.items():
            self.weights[name] = self.weights.get(name, 0.0) - self.learning_rate * error * value
        self.steps += 1

    def state_dict(self) -> Dict[str, object]:
        return {
            "learning_rate": self.learning_rate,
            "weights": dict(self.weights),
            "bias": self.bias,
            "steps": self.steps,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        raw_weights = state.get("weights", {})
        self.weights = {str(name): float(value) for name, value in dict(raw_weights).items()}
        self.bias = float(state.get("bias", self.bias))
        self.steps = int(state.get("steps", self.steps))


@dataclass
class LearnableValueModel:
    learning_rate: float = 0.03
    init_uncertainty: float = 0.20
    weights: Dict[str, float] = field(default_factory=dict)
    bias: float = 0.0
    steps: int = 0
    residual_ema: float = 0.20

    def predict(self, features: Dict[str, float]) -> Tuple[float, float]:
        mean = self.bias + _dot(self.weights, features)
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
