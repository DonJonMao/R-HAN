"""Stage2 V2 统一配置

模块化设计，支持消融实验
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Stage2V2Config:
    """二阶段 V2 完整配置"""

    # === Memory Composer ===
    composer_enabled: bool = True
    composer_hidden_dim: int = 4096
    composer_latent_length: int = 8
    composer_encoder_layers: int = 2
    composer_dropout: float = 0.1

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

    # === Runtime ===
    turn_count: int = 5
    max_selected_memories: int = 4

    # === Fallback to V1 ===
    # 如果某个模块禁用，自动回退到 V1 的实现
    fallback_to_v1: bool = True

    def get_composer_config(self):
        """获取 Composer 配置"""
        from .composer import MemoryComposerConfig
        return MemoryComposerConfig(
            hidden_dim=self.composer_hidden_dim,
            latent_length=self.composer_latent_length,
            encoder_layers=self.composer_encoder_layers,
            dropout=self.composer_dropout
        )

    def get_gnn_config(self):
        """获取 GNN 配置"""
        from .gnn import GNNConfig
        return GNNConfig(
            hidden_dim=self.composer_hidden_dim,
            num_layers=self.gnn_num_layers,
            dropout=self.gnn_dropout
        )

    def get_global_node_config(self):
        """获取 Global Node 配置"""
        from .global_node import GlobalContextConfig
        return GlobalContextConfig(
            hidden_dim=self.composer_hidden_dim,
            enabled=self.global_node_enabled,
            update_method=self.global_node_update_method,
            learnable=self.global_node_learnable
        )

    def get_lmpo_config(self):
        """获取 LMPO 配置"""
        from .lmpo import LMPOConfig
        return LMPOConfig(
            learning_rate=self.lmpo_learning_rate,
            baseline_momentum=self.lmpo_baseline_momentum,
            enabled=self.lmpo_enabled
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
