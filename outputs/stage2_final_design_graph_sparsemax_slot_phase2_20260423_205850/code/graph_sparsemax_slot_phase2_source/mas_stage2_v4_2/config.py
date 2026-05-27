from __future__ import annotations

from dataclasses import dataclass

from mas_stage2.config_v2 import Stage2V2Config


@dataclass
class Stage2V42Config(Stage2V2Config):
    """Stage2 V4.2 配置。

    V4.2 在 V4.1 之上继续修改候选选择链：
    1. 保留显式 challenger 通道；
    2. 增加 auditor 支持的 provisional challenger 晋升；
    3. 用结构化 adjudication 替代简单 inspector 二选一；
    4. 在最终 override 前执行置换式校准。
    """

    code_require_entry_point: bool = True
    max_logged_candidates: int = 8
    explicit_challenger_agents: tuple[str, ...] = ("skeptic", "debater_b")
    auditor_agent_ids: tuple[str, ...] = ("verifier", "skeptic")
    adjudicator_agent_id: str = "verifier"
    calibration_agent_id: str = "verifier"
    calibration_rounds: int = 3
