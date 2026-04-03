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
