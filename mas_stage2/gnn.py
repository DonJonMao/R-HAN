"""轻量 GNN: 用于聚合邻居的 latent memory"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class GNNConfig:
    """GNN 配置"""
    hidden_dim: int = 4096
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_weights: bool = True


class LightweightGNN(nn.Module):
    """轻量 GNN 用于邻居聚合

    不冻结，动态适应不同图结构
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config

        # Message passing layers
        self.layers = nn.ModuleList([
            GNNLayer(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        self_latent: torch.Tensor,
        neighbor_latents: List[torch.Tensor],
        edge_weights: List[float]
    ) -> torch.Tensor:
        """聚合邻居信息

        Args:
            self_latent: (L', D) 自己的 latent memory
            neighbor_latents: List of (L', D) 邻居的 latent memory
            edge_weights: List of float 边权重

        Returns:
            aggregated: (L', D) 聚合后的 latent memory
        """
        if not neighbor_latents:
            return self_latent

        x = self_latent.unsqueeze(0)  # (1, L', D)

        for layer in self.layers:
            x = layer(x, neighbor_latents, edge_weights)

        return x.squeeze(0)  # (L', D)


class GNNLayer(nn.Module):
    """单层 GNN"""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()

        # Message MLP
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Update MLP
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        neighbor_latents: List[torch.Tensor],
        edge_weights: List[float]
    ) -> torch.Tensor:
        """
        Args:
            x: (1, L', D)
            neighbor_latents: List of (L', D)
            edge_weights: List of float
        """
        # 1. 邻居消息聚合
        neighbor_msgs = []
        for neighbor, weight in zip(neighbor_latents, edge_weights):
            msg = self.message_mlp(neighbor.unsqueeze(0))  # (1, L', D)
            neighbor_msgs.append(weight * msg)

        if neighbor_msgs:
            aggregated_msg = torch.stack(neighbor_msgs).sum(dim=0)  # (1, L', D)
        else:
            aggregated_msg = torch.zeros_like(x)

        # 2. 自身更新
        combined = torch.cat([x, aggregated_msg], dim=-1)  # (1, L', 2D)
        updated = self.update_mlp(combined)  # (1, L', D)

        # 3. Residual + Norm
        output = self.norm(x + updated)

        return output
