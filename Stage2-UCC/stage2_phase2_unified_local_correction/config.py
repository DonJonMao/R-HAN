from __future__ import annotations

from dataclasses import dataclass

from stage2_phase1_semantic_safe_override.config import Phase1SemanticSafeOverrideConfig


@dataclass
class Phase2UnifiedLocalCorrectionConfig(Phase1SemanticSafeOverrideConfig):
    stage2_version: str = "phase2_unified_local_correction_v1"

    correction_value_learning_rate: float = 0.03

    correction_frontier_k: int = 3
    correction_max_rounds: int = 2
    localizer_top_k: int = 2

    correction_min_trigger_residual: float = 0.18
    correction_min_value: float = 0.12
    correction_risk_threshold: float = 0.64
    correction_value_weight: float = 0.22

    phase2_answer_gate_weight: float = 0.55
    phase2_preserve_threshold: float = 0.68

