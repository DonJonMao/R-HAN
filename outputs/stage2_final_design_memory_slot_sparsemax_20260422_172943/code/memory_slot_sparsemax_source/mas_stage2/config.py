from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Stage2MemoryConfig:
    max_private_records_per_agent: int = 32
    max_selected_records: int = 4
    max_neighbour_exports: int = 3
    max_record_chars: int = 600
    max_export_chars: int = 320
    max_brief_chars: int = 900
    query_max_chars: int = 1200
    keep_latest_self_output: bool = True
    keep_latest_feedback: bool = True
    include_failure_memory: bool = True
    include_success_memory: bool = True


@dataclass
class Stage2GraphConfig:
    turn_count: int = 5
    soft_prune_top_k: int = 3
    soft_prune_threshold: float = 0.38
    hard_prune_after_turn: int = 4
    min_incoming_edges: int = 1
    controller_role_boost: float = 0.18
    challenge_penalty: float = 0.22
    support_bonus: float = 0.16


@dataclass
class Stage2ReplayConfig:
    save_prompts: bool = True
    save_exports: bool = True
    max_prompt_chars: int = 4000


@dataclass
class Stage2LearningConfig:
    enabled: bool = True
    selector_model_weight: float = 0.18
    edge_model_weight: float = 0.22
    controller_model_weight: float = 0.16
    selector_learning_rate: float = 0.03
    edge_learning_rate: float = 0.03
    controller_learning_rate: float = 0.025
    positive_feedback_bonus: float = 0.14
    negative_feedback_penalty: float = 0.18
    uncertain_penalty: float = 0.06
    sink_bonus: float = 0.05
    fallback_penalty: float = 0.12


@dataclass
class Stage2RuntimeConfig:
    memory: Stage2MemoryConfig = field(default_factory=Stage2MemoryConfig)
    graph: Stage2GraphConfig = field(default_factory=Stage2GraphConfig)
    replay: Stage2ReplayConfig = field(default_factory=Stage2ReplayConfig)
    learning: Stage2LearningConfig = field(default_factory=Stage2LearningConfig)
    controller_max_chars: int = 1200
    finalizer_max_chars: int = 1200
    allow_cross_agent_raw_memory: bool = False
