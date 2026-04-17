from __future__ import annotations

from dataclasses import dataclass, field

from mas_stage2.config import Stage2ReplayConfig
from mas_stage2.config_v2 import Stage2V2Config


@dataclass
class Phase1SemanticSafeOverrideConfig(Stage2V2Config):
    stage2_version: str = "phase1_semantic_safe_override_v1"
    replay: Stage2ReplayConfig = field(default_factory=lambda: Stage2ReplayConfig(max_prompt_chars=16000))

    # Compatibility surface for the inherited Stage2-GCR+ runtime chain.
    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    override_inspector_agent_id: str = "verifier"

    view_names: tuple[str, ...] = ("surface_view", "step_view", "struct_view", "exec_view")
    memory_view_names: tuple[str, ...] = ("stable", "failure", "provenance", "global")

    utility_learning_rate: float = 0.03
    overturn_learning_rate: float = 0.03
    safe_override_learning_rate: float = 0.03

    candidate_core_k: int = 4
    candidate_max_k: int = 8
    frontier_top_k: int = 4

    redundancy_gamma: float = 0.18
    redundancy_temperature: float = 0.7
    similarity_threshold: float = 0.82

    answer_role_threshold: float = 0.58
    evidence_role_threshold: float = 0.52

    safe_margin: float = 0.03
    safe_override_threshold: float = 0.52
    overturn_threshold: float = 0.58
    catastrophic_answer_delta_threshold: float = 0.75
    catastrophic_consistency_threshold: float = 0.48

    overturn_penalty_weight: float = 0.32
    preserve_penalty_weight: float = 0.18
    answer_consistency_bonus: float = 0.14

    code_timeout_s: float = 8.0
    code_max_failed_examples: int = 3

    controller_node_top_k: int = 6
    controller_edge_threshold: float = 0.35
    controller_role_temperature: float = 0.8
    node_participation_focus_bonus: float = 0.25
    node_participation_sink_bonus: float = 0.20
    node_participation_root_bonus: float = 0.10
