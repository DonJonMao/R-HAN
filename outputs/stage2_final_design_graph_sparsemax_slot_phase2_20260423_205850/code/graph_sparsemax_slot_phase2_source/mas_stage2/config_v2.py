"""Stage2 V2 统一配置。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field

from .config import Stage2GraphConfig, Stage2LearningConfig, Stage2MemoryConfig, Stage2ReplayConfig, Stage2RuntimeConfig


@dataclass
class Stage2V2Config:
    """二阶段 V2 完整配置。

    V2 仍然复用 V1 已经成熟的执行壳，因此保留 `memory/graph/replay/learning`
    四个运行时配置块；在此基础上额外增加 latent memory / GNN / global node /
    LMPO 等 V2 特有配置。
    """

    memory: Stage2MemoryConfig = field(default_factory=Stage2MemoryConfig)
    graph: Stage2GraphConfig = field(default_factory=Stage2GraphConfig)
    replay: Stage2ReplayConfig = field(default_factory=Stage2ReplayConfig)
    learning: Stage2LearningConfig = field(default_factory=Stage2LearningConfig)
    controller_max_chars: int = 1200
    finalizer_max_chars: int = 1200
    allow_cross_agent_raw_memory: bool = False

    # === Memory Composer ===
    composer_enabled: bool = True
    composer_hidden_dim: int = 0  # 0 表示自动对齐 embedder 维度
    composer_latent_length: int = 8
    composer_encoder_layers: int = 2
    composer_dropout: float = 0.1
    composer_vocab_size: int = 50000
    composer_max_input_length: int = 512

    # === GNN ===
    gnn_enabled: bool = True
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.1

    # === Global Context Node ===
    global_node_enabled: bool = True
    global_node_update_method: str = "attention"  # attention | mean | none
    global_node_learnable: bool = True

    # === LMPO Training ===
    lmpo_enabled: bool = True
    lmpo_learning_rate: float = 3e-4
    lmpo_baseline_momentum: float = 0.9
    lmpo_sampling_temperature: float = 0.8

    # === Latent-to-Prompt Bridge ===
    latent_prompt_max_local: int = 2
    latent_prompt_max_neighbor: int = 2
    global_summary_max_nodes: int = 2
    latent_bridge_trainable: bool = True

    # === Runtime ===
    fallback_to_v1: bool = False

    def resolved_hidden_dim(self, embed_dim: int | None = None) -> int:
        if self.composer_hidden_dim > 0:
            return int(self.composer_hidden_dim)
        if embed_dim is not None and embed_dim > 0:
            return int(embed_dim)
        return 4096

    def get_composer_config(self, embed_dim: int | None = None):
        """获取 Composer 配置。"""
        from .composer import MemoryComposerConfig

        return MemoryComposerConfig(
            hidden_dim=self.resolved_hidden_dim(embed_dim),
            latent_length=self.composer_latent_length,
            encoder_layers=self.composer_encoder_layers,
            dropout=self.composer_dropout,
            max_input_length=self.composer_max_input_length,
        )

    def get_gnn_config(self, embed_dim: int | None = None):
        """获取 GNN 配置。"""
        from .gnn import GNNConfig

        return GNNConfig(
            hidden_dim=self.resolved_hidden_dim(embed_dim),
            num_layers=self.gnn_num_layers,
            dropout=self.gnn_dropout,
        )

    def get_global_node_config(self, embed_dim: int | None = None):
        """获取 Global Node 配置。"""
        from .global_node import GlobalContextConfig

        return GlobalContextConfig(
            hidden_dim=self.resolved_hidden_dim(embed_dim),
            enabled=self.global_node_enabled,
            update_method=self.global_node_update_method,
            learnable=self.global_node_learnable,
        )

    def get_lmpo_config(self):
        """获取 LMPO 配置。"""
        from .lmpo import LMPOConfig

        return LMPOConfig(
            learning_rate=self.lmpo_learning_rate,
            baseline_momentum=self.lmpo_baseline_momentum,
            enabled=self.lmpo_enabled,
        )

    def to_runtime_config(self) -> Stage2RuntimeConfig:
        """导出兼容 V1 执行壳的基础配置。"""
        return Stage2RuntimeConfig(
            memory=deepcopy(self.memory),
            graph=deepcopy(self.graph),
            replay=deepcopy(self.replay),
            learning=deepcopy(self.learning),
            controller_max_chars=self.controller_max_chars,
            finalizer_max_chars=self.finalizer_max_chars,
            allow_cross_agent_raw_memory=self.allow_cross_agent_raw_memory,
        )


# 预定义配置
FULL_V2_CONFIG = Stage2V2Config(
    composer_enabled=True,
    gnn_enabled=True,
    global_node_enabled=True,
    lmpo_enabled=True
)

ABLATION_NO_GNN = Stage2V2Config(
    composer_enabled=True,
    gnn_enabled=False,
    global_node_enabled=True,
    lmpo_enabled=True
)

ABLATION_NO_GLOBAL = Stage2V2Config(
    composer_enabled=True,
    gnn_enabled=True,
    global_node_enabled=False,
    lmpo_enabled=True
)

ABLATION_NO_LEARNING = Stage2V2Config(
    composer_enabled=True,
    gnn_enabled=True,
    global_node_enabled=True,
    lmpo_enabled=False
)

V1_COMPATIBLE = Stage2V2Config(
    composer_enabled=False,
    gnn_enabled=False,
    global_node_enabled=False,
    lmpo_enabled=False,
    fallback_to_v1=True
)
