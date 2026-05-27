from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
from torch import nn

from .diffusion import DualDiffusionOutput


@dataclass
class EmissionRequest:
    question_text: str
    boundary_index: int
    origin_node_id: str
    rerun_subgraph_node_ids: List[str]
    trace_suffix_briefs: List[str]
    anchor_replay_cache: Dict[str, Any]


@dataclass
class EmitterOutput:
    eta_nodes: torch.Tensor
    eta_null: torch.Tensor
    gamma_tilde_plus: torch.Tensor
    gamma_plus: torch.Tensor
    selected_node_ids: List[str]


class RollbackEmitterModel(nn.Module):
    def __init__(self, hidden_dim: int = 24) -> None:
        super().__init__()
        self.node_head = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.null_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        *,
        diffusion_output: DualDiffusionOutput,
        boundary_index: int,
        anchor_typed_support: float,
        train_mode: bool,
    ) -> EmitterOutput:
        device = next(self.parameters()).device
        alpha_bar_ans = diffusion_output.answer.alpha_bar.to(device)
        alpha_bar_sup = diffusion_output.support.alpha_bar.to(device)
        support_memory = diffusion_output.support_memory.to(device)
        node_inputs = torch.stack(
            [
                alpha_bar_ans,
                alpha_bar_sup,
                support_memory,
            ],
            dim=-1,
        )
        eta_nodes = self.node_head(node_inputs).squeeze(-1)
        eta_null = self.null_head(
            torch.tensor(
                [
                    float(alpha_bar_ans.mean().item()),
                    float(alpha_bar_sup.mean().item()),
                    float(anchor_typed_support),
                    1.0 if int(boundary_index) == 0 else 0.0,
                ],
                dtype=eta_nodes.dtype,
                device=eta_nodes.device,
            )
        ).squeeze(-1)
        full_logits = torch.cat([eta_nodes, eta_null.unsqueeze(0)], dim=0)
        gamma_tilde_plus = torch.softmax(full_logits, dim=0)
        gamma_plus = torch.zeros_like(gamma_tilde_plus)
        gamma_plus[int(torch.argmax(gamma_tilde_plus).item())] = 1.0
        selected_node_ids: List[str] = []
        for node_idx, score in enumerate(gamma_plus[:-1]):
            if float(score.item()) > 0.0:
                selected_node_ids.append(diffusion_output.node_ids[node_idx])
        return EmitterOutput(
            eta_nodes=eta_nodes,
            eta_null=eta_null,
            gamma_tilde_plus=gamma_tilde_plus if train_mode else gamma_plus,
            gamma_plus=gamma_plus,
            selected_node_ids=selected_node_ids,
        )


def build_rerun_request(
    *,
    question_text: str,
    boundary_index: int,
    origin_node_id: str,
    rerun_subgraph_node_ids: List[str],
    trace_suffix_briefs: List[str],
    anchor_replay_cache: Dict[str, Any],
) -> EmissionRequest:
    return EmissionRequest(
        question_text=question_text,
        boundary_index=int(boundary_index),
        origin_node_id=origin_node_id,
        rerun_subgraph_node_ids=list(rerun_subgraph_node_ids),
        trace_suffix_briefs=list(trace_suffix_briefs),
        anchor_replay_cache=dict(anchor_replay_cache),
    )
