from __future__ import annotations

from dataclasses import dataclass

from mas_stage2.config_v2 import Stage2V2Config


@dataclass
class Stage2V3Config(Stage2V2Config):
    numeric_stage1_anchor_prior: float = 2.0
    numeric_override_margin: float = 1.5
    numeric_min_support_to_override: float = 3.0

    code_stage1_anchor_prior: float = 1.5
    code_support_margin: float = 1.0
    code_min_review_advantage: int = 1
    code_require_entry_point: bool = True

    generic_stage1_anchor_prior: float = 1.0
    generic_override_margin: float = 1.0

    max_logged_candidates: int = 8


@dataclass
class Stage2V31Config(Stage2V3Config):
    pass


@dataclass
class Stage2V41Config(Stage2V2Config):
    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    override_inspector_agent_id: str = "verifier"


@dataclass
class Stage2V42Config(Stage2V2Config):
    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    auditor_agent_ids: tuple[str, ...] = ("verifier", "skeptic")
    adjudicator_agent_id: str = "verifier"
    calibration_agent_id: str = "verifier"
    calibration_rounds: int = 3


@dataclass
class Stage2V43Config(Stage2V42Config):
    protocol_family: str = "code_repair"
    repair_agent_ids: tuple[str, ...] = ("coder", "planner", "debater_a")
    repair_rounds: int = 2
    repair_seed_top_k: int = 4
    repair_max_failed_examples: int = 3
    repair_timeout_s: float = 8.0


@dataclass
class Stage2V44Config(Stage2V43Config):
    protocol_family: str = "auto"
    default_budget_bucket: str = "normal"
    lean_hypothesis_cap: int = 2
    full_hypothesis_cap: int = 3
    graph_seed_top_k: int = 3
    graph_repair_rounds: int = 1
    graph_full_branch_cap: int = 3
    graph_repair_agent_ids: tuple[str, ...] = ("coder", "reasoner", "planner")
    adversarial_lean_hypothesis_cap: int = 2
    adversarial_full_hypothesis_cap: int = 3
    inspector_agent_id: str = "verifier"
    repair_self_check_enabled: bool = True
