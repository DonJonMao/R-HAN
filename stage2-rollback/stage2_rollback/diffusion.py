from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch import nn

from .boundary import BoundaryOutput
from .verifier import VerifierState


def build_graph_matrices(union_graph: Any) -> Tuple[List[str], torch.Tensor]:
    if union_graph is None or not getattr(union_graph, "nodes", None):
        return ["fallback"], torch.ones((1, 1), dtype=torch.float32)
    node_ids = list(union_graph.nodes.keys())
    index = {node_id: idx for idx, node_id in enumerate(node_ids)}
    adjacency = torch.zeros((len(node_ids), len(node_ids)), dtype=torch.float32)
    for edge in union_graph.edges:
        if edge.src in index and edge.dst in index:
            adjacency[index[edge.src], index[edge.dst]] = 1.0 + float(edge.support_ratio)
    if torch.sum(adjacency) <= 0:
        adjacency += torch.eye(len(node_ids), dtype=torch.float32)
    return node_ids, adjacency


def DiffTilde(q: torch.Tensor, P_tilde: torch.Tensor, T_diff: int) -> torch.Tensor:
    if q.numel() == 0:
        return q
    if torch.sum(q) <= 0:
        q = torch.full_like(q, 1.0 / max(1, q.shape[0]))
    else:
        q = q / torch.sum(q)
    p = q
    for _ in range(max(1, int(T_diff))):
        p = P_tilde.T @ p
        denom = torch.sum(p)
        p = p / (denom + 1e-8)
    return p


def Diff(q: torch.Tensor, P_tilde: torch.Tensor, T_diff: int) -> torch.Tensor:
    dense = DiffTilde(q, P_tilde, T_diff)
    if dense.numel() == 0:
        return dense
    top_index = int(torch.argmax(dense).item())
    sparse = torch.zeros_like(dense)
    sparse[top_index] = 1.0
    return sparse


@dataclass
class DiffusionStageOutput:
    q: torch.Tensor
    alpha_tilde: torch.Tensor
    alpha: torch.Tensor
    alpha_bar: torch.Tensor
    P_tilde: torch.Tensor
    P_rr: torch.Tensor
    absorb: torch.Tensor


@dataclass
class DualDiffusionOutput:
    node_ids: List[str]
    support: DiffusionStageOutput
    answer: DiffusionStageOutput
    support_memory: torch.Tensor


class RollbackDiffusionModel(nn.Module):
    def __init__(self, hidden_dim: int = 24, T_diff: int = 3) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.T_diff = int(T_diff)
        self.seed_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.support_absorb = nn.Sequential(nn.Linear(2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.answer_absorb = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.edge_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.support_memory = nn.Sequential(nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.answer_couple = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _stage_transition(
        self,
        adjacency: torch.Tensor,
        q_seed: torch.Tensor,
        absorb_logits: torch.Tensor,
        *,
        train_mode: bool,
    ) -> DiffusionStageOutput:
        absorb = torch.sigmoid(absorb_logits)
        norm = adjacency / (torch.sum(adjacency, dim=1, keepdim=True) + 1e-8)
        P_tilde = torch.diag(absorb) + (torch.eye(norm.shape[0], dtype=norm.dtype, device=norm.device) - torch.diag(absorb)) @ norm
        alpha_tilde = DiffTilde(q_seed, P_tilde, self.T_diff)
        alpha_sparse = Diff(q_seed, P_tilde, self.T_diff)
        alpha_bar = alpha_tilde if train_mode else alpha_sparse
        denom = torch.sum(alpha_bar[:, None] * P_tilde, dim=0, keepdim=True) + 1e-8
        P_rr = (alpha_bar[:, None] * P_tilde) / denom
        return DiffusionStageOutput(
            q=q_seed,
            alpha_tilde=alpha_tilde,
            alpha=alpha_sparse,
            alpha_bar=alpha_bar,
            P_tilde=P_tilde,
            P_rr=P_rr,
            absorb=absorb,
        )

    def forward(
        self,
        *,
        verifier_state: VerifierState,
        boundary_output: BoundaryOutput,
        union_graph: Any,
        train_mode: bool,
    ) -> DualDiffusionOutput:
        device = next(self.parameters()).device
        node_ids, adjacency = build_graph_matrices(union_graph)
        adjacency = adjacency.to(device)
        num_nodes = len(node_ids)
        unit_to_nodes = verifier_state.unit_latents.unit_to_nodes.to(device)
        if unit_to_nodes.numel() == 0:
            unit_to_nodes = torch.full((1, num_nodes), 1.0 / max(1, num_nodes), dtype=torch.float32, device=device)
        suffix_mass = 1.0 - float(boundary_output.pi_tilde_rb[0].item()) if boundary_output.pi_tilde_rb.numel() > 0 else 1.0
        w_ans = (verifier_state.role_ans * verifier_state.epsilon).to(device)
        w_evd = (verifier_state.role_evd * verifier_state.mu).to(device)
        w_mix_eps = (verifier_state.role_mix * verifier_state.epsilon).to(device)
        w_mix_mu = (verifier_state.role_mix * verifier_state.mu).to(device)

        q_sup = torch.sum(unit_to_nodes * (w_evd + w_mix_mu)[:, None], dim=0) * suffix_mass
        p_safe = torch.sum(unit_to_nodes * verifier_state.kappa.to(device)[:, None], dim=0) * max(0.0, 1.0 - suffix_mass)
        support_absorb_logits = self.support_absorb(torch.stack([q_sup, p_safe], dim=-1)).squeeze(-1)
        support = self._stage_transition(adjacency, q_sup + 1e-6, support_absorb_logits, train_mode=train_mode)

        support_memory = torch.sigmoid(
            self.support_memory(
                torch.stack(
                    [
                        support.alpha_bar,
                        p_safe,
                        torch.full_like(support.alpha_bar, suffix_mass),
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        )
        q_ans_raw = torch.sum(unit_to_nodes * (w_ans + w_mix_eps)[:, None], dim=0) * suffix_mass
        q_ans_coupled = torch.nn.functional.softplus(
            self.answer_couple(
                torch.stack(
                    [
                        q_ans_raw,
                        support.alpha_bar,
                        support_memory,
                        p_safe,
                    ],
                    dim=-1,
                )
            ).squeeze(-1)
        )
        answer_absorb_logits = self.answer_absorb(
            torch.stack([q_ans_coupled, p_safe, support.alpha_bar], dim=-1)
        ).squeeze(-1)
        answer = self._stage_transition(adjacency, q_ans_coupled + 1e-6, answer_absorb_logits, train_mode=train_mode)
        return DualDiffusionOutput(
            node_ids=node_ids,
            support=support,
            answer=answer,
            support_memory=support_memory,
        )
