from __future__ import annotations

from dataclasses import dataclass

from .types import EvalSummary, TaskEvaluation


@dataclass(frozen=True)
class RewardWeights:
    task_score: float = 1.0
    success: float = 0.4
    token_cost: float = 0.02
    latency: float = 0.03
    safety_penalty: float = 0.35
    size_penalty: float = 0.10
    prompt_penalty: float = 0.08


def reward_from_evaluation(
    evaluation: TaskEvaluation,
    *,
    size_penalty: float,
    prompt_penalty: float,
    weights: RewardWeights = RewardWeights(),
) -> float:
    return (
        weights.task_score * evaluation.task_score
        + weights.success * evaluation.success
        - weights.token_cost * evaluation.token_cost
        - weights.latency * evaluation.latency
        - weights.safety_penalty * evaluation.safety_penalty
        - weights.size_penalty * size_penalty
        - weights.prompt_penalty * prompt_penalty
    )


def risk_adjusted_score(summary: EvalSummary, std_penalty: float) -> float:
    return summary.mean_reward - std_penalty * summary.reward_std
