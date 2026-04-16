from __future__ import annotations

from dataclasses import dataclass, field

from mas_stage2.config import Stage2ReplayConfig
from mas_stage2.config_v2 import Stage2V2Config


@dataclass
class Phase3aUnifiedConfig(Stage2V2Config):
    stage2_version: str = "phase3a_unified_v1"
    replay: Stage2ReplayConfig = field(default_factory=lambda: Stage2ReplayConfig(max_prompt_chars=16000))

    # Compatibility surface for the inherited Stage2-GCR+ runtime chain.
    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    override_inspector_agent_id: str = "verifier"

    view_names: tuple[str, ...] = ("surface_view", "step_view", "struct_view", "exec_view")
    view_encoder_hidden_dim: int = 256
    view_encoder_layers: int = 2

    utility_learning_rate: float = 0.03
    delta_learning_rate: float = 0.03
    halt_learning_rate: float = 0.03

    candidate_core_k: int = 4
    candidate_max_k: int = 8
    frontier_top_k: int = 4
    correction_frontier_k: int = 2
    correction_branch_budget: int = 2
    correction_max_rounds: int = 2
    localizer_top_k: int = 2

    redundancy_gamma: float = 0.18
    redundancy_temperature: float = 0.7
    similarity_threshold: float = 0.82
    preserve_threshold: float = 0.72
    guard_margin: float = 0.04

    correction_accept_delta: float = 0.02
    halt_threshold: float = 0.72
    halt_plateau_rounds: int = 2

    use_prompt_canonicalizer: bool = False
    use_prompt_meta_verifier: bool = False
    use_prompt_correction: bool = True
    use_prompt_delta: bool = False

    canonicalizer_agent_id: str = "aggregator"
    verifier_agent_id: str = "verifier"
    critic_agent_id: str = "critic"
    editor_agent_id: str = "reviser"
    delta_agent_id: str = "judge"

    code_timeout_s: float = 8.0
    code_max_failed_examples: int = 3

    controller_node_top_k: int = 6
    controller_edge_threshold: float = 0.35
    controller_role_temperature: float = 0.8
    node_participation_focus_bonus: float = 0.25
    node_participation_sink_bonus: float = 0.20
    node_participation_root_bonus: float = 0.10

    memory_view_names: tuple[str, ...] = ("stable", "failure", "provenance", "global")
