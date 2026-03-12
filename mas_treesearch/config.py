from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class ChatConfig:
    api_base: str = "http://127.0.0.1:8039"
    model: Optional[str] = None
    api_key: Optional[str] = None
    timeout_s: float = 60.0
    temperature: float = 0.2
    max_tokens: int = 512
    max_retries: int = 2


@dataclass
class JudgeConfig:
    model: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 128


@dataclass
class EmbeddingConfig:
    api_base: str = "http://127.0.0.1:8018"
    model: str = "/mnt/nvme/Qwen3-Embedding-8B"
    api_key: Optional[str] = None
    timeout_s: float = 30.0
    dim: int = 256


@dataclass
class TierRuntimeConfig:
    max_tokens: int
    judge_max_tokens: int
    repeats: int
    temperature: float


@dataclass
class TieredEvalConfig:
    chat: ChatConfig = field(default_factory=ChatConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    tier1: TierRuntimeConfig = field(
        default_factory=lambda: TierRuntimeConfig(
            max_tokens=192,
            judge_max_tokens=96,
            repeats=1,
            temperature=0.1,
        )
    )
    tier2: TierRuntimeConfig = field(
        default_factory=lambda: TierRuntimeConfig(
            max_tokens=512,
            judge_max_tokens=128,
            repeats=2,
            temperature=0.2,
        )
    )
    success_threshold: float = 0.6
    token_cost_per_word: float = 0.00001
    enable_chat_cache: bool = True
    debug_judge: bool = False


@dataclass
class SearchConfig:
    random_seed: int = 7
    candidate_core_k: int = 4
    candidate_explore_k: int = 2
    candidate_max_k: int = 6
    min_active_agents: int = 2
    max_active_agents: int = 6
    search_iterations: int = 24
    top_k_selection: int = 5
    root_restart_prob: float = 0.2
    puct_c: float = 1.25
    novelty_bonus: float = 0.05
    uncertainty_penalty: float = 0.10
    progressive_widening_alpha: float = 1.5
    progressive_widening_base: int = 2
    tier1_top_fraction: float = 0.25
    tier2_top_fraction: float = 0.20
    final_top_k: int = 3
    prompt_edit_cooldown: int = 2
    max_prompt_edits_per_state: int = 1
    risk_std_penalty: float = 0.30
    score_improvement_epsilon: float = 1e-6
    enable_learned_edit_prior: bool = True
    enable_learned_value_model: bool = True
    learned_prior_weight: float = 0.45
    learned_value_weight: float = 0.35
    learned_prior_lr: float = 0.05
    learned_value_lr: float = 0.03
    learned_value_uncertainty_init: float = 0.20
    root_templates: Tuple[str, ...] = (
        "direct",
        "solve_verify",
        "parallel_vote",
        "critique_revise",
    )
    reserve_roles_for_specialists: Tuple[str, ...] = (
        "reasoning",
        "verification",
        "synthesis",
    )
