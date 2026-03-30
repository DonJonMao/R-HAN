"""Global Context Node: 可学习的全局节点"""

import torch
import torch.nn as nn
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class GlobalContextConfig:
    """全局上下文配置"""
    hidden_dim: int = 4096
    enabled: bool = True  # 是否启用全局节点（消融开关）
    update_method: str = "attention"  # attention | mean | none
    learnable: bool = True  # 是否可学习


class GlobalContextNode(nn.Module):
    """可学习的全局上下文节点

    所有智能体都连接到它，用于存储任务级的全局偏好
    """

    def __init__(self, config: GlobalContextConfig):
        super().__init__()
        self.config = config

        if config.enabled:
            # 全局状态向量
            if config.learnable:
                self.global_state = nn.Parameter(
                    torch.randn(1, config.hidden_dim) * 0.02
                )
            else:
                self.register_buffer(
                    'global_state',
                    torch.zeros(1, config.hidden_dim)
                )

            # Attention pooling for updates
            if config.update_method == "attention":
                self.update_attn = nn.MultiheadAttention(
                    embed_dim=config.hidden_dim,
                    num_heads=8,
                    batch_first=True
                )
                self.update_norm = nn.LayerNorm(config.hidden_dim)

    def get_context(self, batch_size: int = 1) -> torch.Tensor:
        """获取全局上下文

        Returns:
            (B, D) 全局上下文向量
        """
        if not self.config.enabled:
            return torch.zeros(batch_size, self.config.hidden_dim)

        return self.global_state.expand(batch_size, -1)

    def update(
        self,
        agent_updates: List[Tuple[str, torch.Tensor]],
        agent_outputs: List[str]
    ) -> torch.Tensor:
        """更新全局节点

        Args:
            agent_updates: List of (node_id, latent_vector)
            agent_outputs: List of agent output texts (for scoring)

        Returns:
            updated_state: (1, D)
        """
        if not self.config.enabled or not agent_updates:
            return self.global_state if hasattr(self, 'global_state') else torch.zeros(1, self.config.hidden_dim)

        # Stack all updates
        updates = torch.stack([u[1] for u in agent_updates])  # (N, D)

        if self.config.update_method == "attention":
            # Attention pooling
            query = self.global_state.unsqueeze(0)  # (1, 1, D)
            key_value = updates.unsqueeze(0)  # (1, N, D)

            aggregated, attn_weights = self.update_attn(
                query=query,
                key=key_value,
                value=key_value
            )  # (1, 1, D)

            # Update with residual
            new_state = self.update_norm(
                self.global_state + aggregated.squeeze(0)
            )

        elif self.config.update_method == "mean":
            # Simple mean pooling
            aggregated = updates.mean(dim=0, keepdim=True)  # (1, D)
            new_state = 0.9 * self.global_state + 0.1 * aggregated

        else:  # none
            new_state = self.global_state

        # Update state
        if self.config.learnable:
            self.global_state.data = new_state.data
        else:
            self.global_state = new_state

        return new_state

    def reset(self):
        """重置全局状态（用于新任务）"""
        if self.config.enabled:
            if self.config.learnable:
                nn.init.normal_(self.global_state, std=0.02)
            else:
                self.global_state.zero_()
