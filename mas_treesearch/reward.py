from __future__ import annotations

from dataclasses import dataclass
from math import tanh
from typing import Optional

from .profiles import DEFAULT_PROFILE, DatasetProfile
from .types import EvalSummary, StructureMetrics, TaskEvaluation


@dataclass(frozen=True)
class RewardWeights:
    task_score: float = 1.0
    success: float = 0.4
    token_cost: float = 0.02
    latency: float = 0.03
    safety_penalty: float = 0.35
    size_penalty: float = 0.10
    prompt_penalty: float = 0.08


def reward_weights_for_profile(
    profile: DatasetProfile = DEFAULT_PROFILE,
    metadata: Optional[dict] = None,
) -> RewardWeights:
    del metadata
    task_type = profile.task_type
    answer_format = profile.answer_format

    if profile.name == "mmlu_pro":
        return RewardWeights(
            task_score=1.00,
            success=0.62,
            token_cost=0.025,
            latency=0.025,
            safety_penalty=0.42,
            size_penalty=0.11,
            prompt_penalty=0.10,
        )
    if profile.name == "nlgraph":
        return RewardWeights(
            task_score=1.10,
            success=0.62,
            token_cost=0.012,
            latency=0.015,
            safety_penalty=0.18,
            size_penalty=0.07,
            prompt_penalty=0.05,
        )
    if profile.name == "knowledge_crosswords":
        return RewardWeights(
            task_score=1.12,
            success=0.58,
            token_cost=0.012,
            latency=0.015,
            safety_penalty=0.20,
            size_penalty=0.08,
            prompt_penalty=0.05,
        )
    if task_type == "code_generation" or answer_format == "python_code":
        return RewardWeights(
            task_score=1.15,
            success=0.65,
            token_cost=0.01,
            latency=0.02,
            safety_penalty=0.18,
            size_penalty=0.05,
            prompt_penalty=0.04,
        )
    if task_type == "mcq":
        return RewardWeights(
            task_score=1.0,
            success=0.50,
            token_cost=0.02,
            latency=0.02,
            safety_penalty=0.45,
            size_penalty=0.08,
            prompt_penalty=0.09,
        )
    if task_type in {"numeric", "math_expression"} or answer_format in {"number", "math_expression"}:
        return RewardWeights(
            task_score=1.10,
            success=0.55,
            token_cost=0.015,
            latency=0.02,
            safety_penalty=0.25,
            size_penalty=0.07,
            prompt_penalty=0.06,
        )
    if task_type in {"graph_reasoning", "structured_list"} or answer_format in {"graph_json", "json_list"}:
        return RewardWeights(
            task_score=1.05,
            success=0.50,
            token_cost=0.015,
            latency=0.02,
            safety_penalty=0.24,
            size_penalty=0.06,
            prompt_penalty=0.05,
        )
    if task_type == "boolean" or answer_format == "yes_no":
        return RewardWeights(
            task_score=1.0,
            success=0.45,
            token_cost=0.015,
            latency=0.02,
            safety_penalty=0.40,
            size_penalty=0.08,
            prompt_penalty=0.07,
        )
    if answer_format in {"short_span", "question_defined"}:
        return RewardWeights(
            task_score=1.0,
            success=0.45,
            token_cost=0.015,
            latency=0.02,
            safety_penalty=0.28,
            size_penalty=0.07,
            prompt_penalty=0.06,
        )
    return RewardWeights()


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


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def squash_probe_score(value: float) -> float:
    return clamp01(0.5 + 0.5 * tanh(value))


def structure_reward_from_components(
    *,
    coverage: float,
    complementarity: float,
    redundancy_quality: float,
    structural_faithfulness: float,
    runtime_affordability: float,
    execution_probe: float,
    topology_quality_weight: float = 0.40,
    diversity_quality_weight: float = 0.35,
    deployability_weight: float = 0.25,
    execution_probe_weight: float = 0.20,
    coverage_mix: float = 0.55,
    complementarity_mix: float = 0.60,
    metadata: Optional[dict] = None,
) -> StructureMetrics:
    coverage = clamp01(coverage)
    complementarity = clamp01(complementarity)
    redundancy_quality = clamp01(redundancy_quality)
    structural_faithfulness = clamp01(structural_faithfulness)
    runtime_affordability = clamp01(runtime_affordability)
    execution_probe = clamp01(execution_probe)

    coverage_mix = clamp01(coverage_mix)
    complementarity_mix = clamp01(complementarity_mix)
    topology_quality = clamp01(coverage_mix * coverage + (1.0 - coverage_mix) * structural_faithfulness)
    diversity_quality = clamp01(complementarity_mix * complementarity + (1.0 - complementarity_mix) * redundancy_quality)
    deployability = runtime_affordability
    total = max(1e-6, topology_quality_weight + diversity_quality_weight + deployability_weight)
    topology_quality_weight /= total
    diversity_quality_weight /= total
    deployability_weight /= total
    structure_reward = (
        topology_quality_weight * topology_quality
        + diversity_quality_weight * diversity_quality
        + deployability_weight * deployability
    )
    execution_probe_weight = clamp01(execution_probe_weight)
    total_reward = clamp01((1.0 - execution_probe_weight) * structure_reward + execution_probe_weight * execution_probe)

    return StructureMetrics(
        coverage=coverage,
        complementarity=complementarity,
        redundancy_quality=redundancy_quality,
        structural_faithfulness=structural_faithfulness,
        runtime_affordability=runtime_affordability,
        topology_quality=topology_quality,
        diversity_quality=diversity_quality,
        deployability=deployability,
        execution_probe=execution_probe,
        total_reward=total_reward,
        metadata=dict(metadata or {}),
    )
