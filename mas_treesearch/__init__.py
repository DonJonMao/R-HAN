"""Standalone tree-search MAS framework.

This package is intentionally independent from ``mas_gflowopt``.
It provides a minimal viable architecture search stack with:

- real OpenAI-compatible chat / embedding clients
- cached agent and question embeddings
- task-conditioned agent subset selection
- template-based architecture + lightweight prompt-slot search
- tree search with multi-fidelity evaluation
"""

from .agents import AgentPool, AgentSpec, default_agent_pool
from .config import EmbeddingConfig, SearchConfig, TieredEvalConfig
from .data import build_processed_datasets, list_processed_datasets, load_processed_split, standardize_record
from .learning import FeatureBuilder, LearnableEditPrior, LearnableValueModel
from .pipeline import TreeSearchMASPipeline
from .profiles import DatasetProfile, list_supported_datasets, resolve_dataset_profile
from .types import SearchResult, TaskEvaluation

__all__ = [
    "AgentPool",
    "AgentSpec",
    "DatasetProfile",
    "EmbeddingConfig",
    "FeatureBuilder",
    "LearnableEditPrior",
    "LearnableValueModel",
    "SearchConfig",
    "SearchResult",
    "TaskEvaluation",
    "TieredEvalConfig",
    "TreeSearchMASPipeline",
    "build_processed_datasets",
    "default_agent_pool",
    "list_processed_datasets",
    "list_supported_datasets",
    "load_processed_split",
    "resolve_dataset_profile",
    "standardize_record",
]
