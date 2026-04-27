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
from .config import EmbeddingConfig, SearchConfig, TieredEvalConfig, UnionRuntimeConfig
from .data import (
    build_processed_datasets,
    filter_stage2_supported_items,
    has_stage2_reference_answer,
    list_processed_datasets,
    load_processed_split,
    standardize_record,
)
from .learning import FeatureBuilder, LearnableEditPrior, LearnableValueModel
from .pipeline import TreeSearchMASPipeline
from .profiles import DatasetProfile, StructurePrior, list_supported_datasets, resolve_dataset_profile
from .result_utils import (
    has_structure_output,
    is_union_result,
    resolve_result_output,
    resolve_primary_reward,
    resolve_result_signature,
    resolve_result_summary,
    resolve_structure_signature,
    resolve_structure_summary,
)
from .types import SearchResult, StructureMetrics, StructureSummary, TaskEvaluation

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
    "UnionRuntimeConfig",
    "build_processed_datasets",
    "default_agent_pool",
    "is_union_result",
    "filter_stage2_supported_items",
    "has_stage2_reference_answer",
    "list_processed_datasets",
    "list_supported_datasets",
    "load_processed_split",
    "resolve_result_output",
    "resolve_result_signature",
    "resolve_result_summary",
    "resolve_dataset_profile",
    "standardize_record",
]
