"""LMPO: Latent Memory Policy Optimization

用 RL 训练 Memory Composer，LLM 保持冻结
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Dict, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class LMPOConfig:
    """LMPO 训练配置"""
    learning_rate: float = 3e-4
    gamma: float = 0.99  # discount factor
    baseline_momentum: float = 0.9
    clip_grad_norm: float = 1.0
    entropy_coef: float = 0.01
    value_loss_coef: float = 0.5
    enabled: bool = True  # 消融开关


class LMPOTrainer:
    """LMPO 训练器"""

    def __init__(
        self,
        composer: nn.Module,
        config: LMPOConfig
    ):
        self.composer = composer
        self.config = config

        if config.enabled:
            self.optimizer = optim.Adam(
                composer.parameters(),
                lr=config.learning_rate
            )

            # Moving average baseline
            self.baseline = 0.0

            # Training history
            self.reward_history = deque(maxlen=100)

    def compute_loss(
        self,
        trajectory: Dict,
        reward: float,
        latent_memories: List[torch.Tensor]
    ) -> torch.Tensor:
        """计算 policy gradient loss

        Args:
            trajectory: 轨迹信息
            reward: 最终 reward
            latent_memories: 各节点的 latent memory

        Returns:
            loss: scalar
        """
        if not self.config.enabled:
            return torch.tensor(0.0)

        # Update baseline
        self.baseline = (
            self.config.baseline_momentum * self.baseline +
            (1 - self.config.baseline_momentum) * reward
        )

        # Compute advantage
        advantage = reward - self.baseline

        # Policy gradient loss
        # 这里简化处理：假设 latent_memories 的生成概率可以通过重新 forward 得到
        loss = -advantage * self._compute_log_prob(latent_memories)

        return loss

    def _compute_log_prob(self, latent_memories: List[torch.Tensor]) -> torch.Tensor:
        """计算 log probability（简化版）"""
        # 实际实现中需要记录 forward 时的分布参数
        # 这里用 L2 norm 作为代理
        log_prob = sum(
            -0.5 * (mem ** 2).sum()
            for mem in latent_memories
        )
        return log_prob / len(latent_memories)

    def update(
        self,
        trajectory: Dict,
        reward: float,
        latent_memories: List[torch.Tensor]
    ):
        """更新 Memory Composer"""
        if not self.config.enabled:
            return

        loss = self.compute_loss(trajectory, reward, latent_memories)

        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.composer.parameters(),
            self.config.clip_grad_norm
        )

        self.optimizer.step()

        # Record
        self.reward_history.append(reward)

    def get_stats(self) -> Dict:
        """获取训练统计"""
        if not self.reward_history:
            return {}

        return {
            "mean_reward": sum(self.reward_history) / len(self.reward_history),
            "baseline": self.baseline,
            "num_updates": len(self.reward_history)
        }
