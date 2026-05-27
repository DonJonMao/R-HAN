"""LMPO: Latent Memory Policy Optimization."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class LMPOConfig:
    """LMPO 训练配置。"""

    learning_rate: float = 3e-4
    gamma: float = 0.99
    baseline_momentum: float = 0.9
    clip_grad_norm: float = 1.0
    entropy_coef: float = 0.01
    enabled: bool = True


class LMPOTrainer:
    """针对黑盒 LLM 执行链路的轻量策略梯度训练器。

    这里不试图穿过 LLM 反传，而是对“latent memory 如何选择/ verbalize 成 prompt”
    的离散策略做 REINFORCE 更新。
    """

    def __init__(self, modules: nn.Module | Sequence[nn.Module], config: LMPOConfig):
        self.modules = self._normalize_modules(modules)
        self.config = config
        self.baseline = 0.0
        self.reward_history = deque(maxlen=100)
        self.loss_history = deque(maxlen=100)
        if config.enabled:
            params: List[nn.Parameter] = []
            for module in self.modules:
                params.extend(list(module.parameters()))
            self.optimizer = optim.Adam(params, lr=config.learning_rate) if params else None
        else:
            self.optimizer = None

    @staticmethod
    def _normalize_modules(modules: nn.Module | Sequence[nn.Module]) -> List[nn.Module]:
        if isinstance(modules, nn.Module):
            return [modules]
        return [module for module in modules if isinstance(module, nn.Module)]

    def compute_loss(
        self,
        log_probs: Sequence[torch.Tensor],
        reward: float,
        *,
        entropy_terms: Sequence[torch.Tensor] | None = None,
        auxiliary_losses: Sequence[torch.Tensor] | None = None,
    ) -> torch.Tensor | None:
        if not self.config.enabled or not log_probs:
            return None
        self.baseline = self.config.baseline_momentum * self.baseline + (1 - self.config.baseline_momentum) * float(reward)
        advantage = float(reward) - self.baseline
        stacked_log_probs = torch.stack([term.reshape(()) for term in log_probs]).sum()
        loss = -advantage * stacked_log_probs
        if entropy_terms:
            entropy = torch.stack([term.reshape(()) for term in entropy_terms]).sum()
            loss = loss - self.config.entropy_coef * entropy
        if auxiliary_losses:
            aux = torch.stack([term.reshape(()) for term in auxiliary_losses]).sum()
            loss = loss + aux
        return loss

    def update_from_policy(
        self,
        log_probs: Sequence[torch.Tensor],
        reward: float,
        *,
        entropy_terms: Sequence[torch.Tensor] | None = None,
        auxiliary_losses: Sequence[torch.Tensor] | None = None,
    ) -> Dict[str, float]:
        if not self.config.enabled:
            return {"enabled": 0.0, "policy_updates": 0.0}
        loss = self.compute_loss(log_probs, reward, entropy_terms=entropy_terms, auxiliary_losses=auxiliary_losses)
        if loss is None or self.optimizer is None:
            return {"enabled": 1.0, "policy_updates": 0.0}
        self.optimizer.zero_grad()
        loss.backward()
        for module in self.modules:
            torch.nn.utils.clip_grad_norm_(module.parameters(), self.config.clip_grad_norm)
        self.optimizer.step()
        self.reward_history.append(float(reward))
        self.loss_history.append(float(loss.detach().cpu().item()))
        return {
            "enabled": 1.0,
            "policy_updates": 1.0,
            "baseline": float(self.baseline),
            "last_loss": float(self.loss_history[-1]),
            "mean_reward": float(sum(self.reward_history) / len(self.reward_history)),
        }

    def get_stats(self) -> Dict[str, float]:
        if not self.reward_history:
            return {}
        return {
            "mean_reward": float(sum(self.reward_history) / len(self.reward_history)),
            "baseline": float(self.baseline),
            "num_updates": float(len(self.reward_history)),
            "mean_loss": float(sum(self.loss_history) / max(1, len(self.loss_history))),
        }
