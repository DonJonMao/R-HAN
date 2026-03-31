"""轻量 GNN: 统一负责 edge gate、edge weight 和邻居聚合。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GNNConfig:
    """GNN 配置"""

    hidden_dim: int = 4096
    num_layers: int = 2
    dropout: float = 0.1
    use_edge_weights: bool = True
    gate_dim: int = 256
    edge_feature_dim: int = 6


class LightweightGNN(nn.Module):
    """轻量 GNN。

    当前不只负责聚合邻居 latent，也负责：
    - 计算 edge gate
    - 给出用于消息传播的 edge weight
    - 为逐轮 pruning 提供统一打分
    """

    def __init__(self, config: GNNConfig):
        super().__init__()
        self.config = config
        self.state_proj = nn.Linear(config.hidden_dim, config.gate_dim)
        self.global_proj = nn.Linear(config.hidden_dim, config.gate_dim)
        self.edge_gate_mlp = nn.Sequential(
            nn.Linear(config.gate_dim * 5 + config.edge_feature_dim, config.gate_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.gate_dim, 1),
        )

        self.layers = nn.ModuleList([GNNLayer(config.hidden_dim, config.dropout) for _ in range(config.num_layers)])

    @staticmethod
    def _pool_latent(latent: torch.Tensor) -> torch.Tensor:
        if latent.dim() == 1:
            return latent
        if latent.dim() == 2:
            return latent.mean(dim=0)
        if latent.dim() == 3:
            return latent.mean(dim=(0, 1))
        raise ValueError(f"Unsupported latent rank: {latent.dim()}")

    def edge_gate(
        self,
        src_latent: torch.Tensor,
        dst_latent: torch.Tensor,
        edge_features: Sequence[float],
        global_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        src_state = self.state_proj(self._pool_latent(src_latent))
        dst_state = self.state_proj(self._pool_latent(dst_latent))
        if global_state is None:
            global_repr = torch.zeros_like(src_state)
        else:
            global_repr = self.global_proj(self._pool_latent(global_state))
        feature_tensor = torch.tensor(list(edge_features), dtype=src_state.dtype, device=src_state.device)
        if feature_tensor.numel() < self.config.edge_feature_dim:
            feature_tensor = F.pad(feature_tensor, (0, self.config.edge_feature_dim - feature_tensor.numel()))
        elif feature_tensor.numel() > self.config.edge_feature_dim:
            feature_tensor = feature_tensor[: self.config.edge_feature_dim]
        gate_input = torch.cat(
            [
                src_state,
                dst_state,
                src_state * dst_state,
                torch.abs(src_state - dst_state),
                global_repr,
                feature_tensor,
            ],
            dim=0,
        )
        return torch.sigmoid(self.edge_gate_mlp(gate_input)).reshape(())

    @staticmethod
    def normalize_edge_weights(edge_gates: Sequence[torch.Tensor | float]) -> List[float]:
        if not edge_gates:
            return []
        tensor = torch.stack(
            [gate.reshape(()) if isinstance(gate, torch.Tensor) else torch.tensor(float(gate), dtype=torch.float32) for gate in edge_gates]
        )
        weights = torch.softmax(tensor, dim=0)
        return [float(weight.detach().cpu().item()) for weight in weights]

    def forward(
        self,
        self_latent: torch.Tensor,
        neighbor_latents: List[torch.Tensor],
        edge_weights: List[float],
    ) -> torch.Tensor:
        if not neighbor_latents:
            return self_latent

        x = self_latent.unsqueeze(0)

        for layer in self.layers:
            x = layer(x, neighbor_latents, edge_weights)

        return x.squeeze(0)


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
