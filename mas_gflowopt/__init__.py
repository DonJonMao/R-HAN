"""MAS-oriented GFlowOpt framework skeleton.

This package provides an end-to-end pipeline with replaceable interfaces:
- agent pool + vectorization
- GFlowNet-style DAG sampling
- proxy model over continuous graph representations
- clustering + multi-start gradient ascent
- discrete hill-climbing refinement by BIC objective
"""

from .agent_pool import AgentPool, default_agent_pool
from .conditioning import TaskConditioner
from .evaluators import (
    BatchLLMExecutionMASTaskEvaluator,
    HeuristicEvaluatorConfig,
    HeuristicMASTaskEvaluator,
    LLMExecutionConfig,
    LLMExecutionMASTaskEvaluator,
    TemplateWeightedEvaluatorConfig,
    TemplateWeightedMASTaskEvaluator,
)
from .pipeline import MASGFlowPipeline
from .reward import MASTaskEvaluator, MASRewardModel
from .scoring import BICScorer, DiscreteDataBICScorer
from .types import GFlowNetTrainingStats, MASConfig, OptimizationOutput, TaskEvaluation

__all__ = [
    "AgentPool",
    "BICScorer",
    "DiscreteDataBICScorer",
    "GFlowNetTrainingStats",
    "BatchLLMExecutionMASTaskEvaluator",
    "HeuristicEvaluatorConfig",
    "HeuristicMASTaskEvaluator",
    "LLMExecutionConfig",
    "LLMExecutionMASTaskEvaluator",
    "TemplateWeightedEvaluatorConfig",
    "TemplateWeightedMASTaskEvaluator",
    "MASRewardModel",
    "MASConfig",
    "MASGFlowPipeline",
    "MASTaskEvaluator",
    "OptimizationOutput",
    "TaskConditioner",
    "TaskEvaluation",
    "default_agent_pool",
]
