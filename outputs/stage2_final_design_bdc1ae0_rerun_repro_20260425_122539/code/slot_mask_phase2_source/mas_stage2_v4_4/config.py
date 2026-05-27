from __future__ import annotations

from dataclasses import dataclass

from mas_stage2_v4_3.config import Stage2V43Config


@dataclass
class Stage2V44Config(Stage2V43Config):
    """Stage2 V4.4 配置。

    V4.4 将 Stage2 硬化为离散协议链：
    1. `CollapseClasses`
    2. `AffordanceRoute`
    3. `SparseActivate`
    4. route-specific `HardDominance / ACH-Lexicographic`
    5. `AnchorGuard`

    本版只保留布尔 gate、偏序支配、词典序淘汰与少量整数预算。
    """

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
