from __future__ import annotations

from dataclasses import dataclass

from mas_stage2_v4_2.config import Stage2V42Config


@dataclass
class Stage2V43Config(Stage2V42Config):
    """Stage2 V4.3 配置。

    当前先落最小闭环的 code-repair 版本：
    1. 先用可执行 verifier 重新评估 anchor 与 stage2 候选；
    2. 若 anchor 未通过可见约束，则围绕 failure event 生成 patch branches；
    3. 仅当分支在 verifier 关系下严格优于当前 checkpoint 时才允许晋升。
    """

    protocol_family: str = "code_repair"
    repair_agent_ids: tuple[str, ...] = ("coder", "planner", "debater_a")
    repair_rounds: int = 2
    repair_seed_top_k: int = 4
    repair_max_failed_examples: int = 3
    repair_timeout_s: float = 8.0
