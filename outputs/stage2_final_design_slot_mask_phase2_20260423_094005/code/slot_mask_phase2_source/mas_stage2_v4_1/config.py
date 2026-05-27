from __future__ import annotations

from dataclasses import dataclass

from mas_stage2.config_v2 import Stage2V2Config


@dataclass
class Stage2V41Config(Stage2V2Config):
    """Stage2 V4.1 配置。

    V4.1 的新增点刻意保持最小化：
    1. 显式 challenger 通道由固定 agent 集合产生；
    2. override 资格必须经过 inspector；
    3. DAR 式消息保留直接复用现有 `memory.max_neighbour_exports` 作为保留规模。
    """

    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    override_inspector_agent_id: str = "verifier"
