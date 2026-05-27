from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import torch
from torch import nn

from .artifacts import ArtifactIR, unit_text_overlap
from .contracts import typed_distance
from .verifier import VerifierState


@dataclass
class SelectorCandidateView:
    candidate_id: str
    score: torch.Tensor
    probability: torch.Tensor
    delta_type: float
    delta_ctr: float
    d_sig: float
    m_keep: float


@dataclass
class SelectorOutput:
    logits: torch.Tensor
    probabilities: torch.Tensor
    winner_index: int
    candidate_views: List[SelectorCandidateView]


class RollbackSelectorModel(nn.Module):
    def __init__(self, hidden_dim: int = 32) -> None:
        super().__init__()
        self.keep_head = nn.Sequential(nn.Linear(4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.base_head = nn.Sequential(nn.Linear(10, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def _alignment(self, anchor_state: VerifierState, cand_state: VerifierState) -> torch.Tensor:
        device = next(self.parameters()).device
        left = anchor_state.unit_latents.unit_vectors[:, :8].to(device)
        right = cand_state.unit_latents.unit_vectors[:, :8].to(device)
        if left.numel() == 0 or right.numel() == 0:
            return torch.ones((1, 1), dtype=torch.float32, device=device)
        logits = left @ right.T
        return torch.softmax(logits, dim=-1)

    def _m_keep(self, anchor_state: VerifierState, cand_state: VerifierState, omega: torch.Tensor) -> torch.Tensor:
        anchor_units = anchor_state.artifact.units
        cand_units = cand_state.artifact.units
        rows = []
        for i, left in enumerate(anchor_units):
            row = []
            for j, right in enumerate(cand_units):
                overlap = unit_text_overlap([left], [right])
                delta_pos = abs(float(left.position) - float(right.position)) / max(1.0, float(max(len(anchor_units), len(cand_units))))
                features = torch.tensor(
                    [
                        overlap,
                        1.0 - delta_pos,
                        float(left.role_probs.get("mixed", 0.0)),
                        float(right.role_probs.get("mixed", 0.0)),
                    ],
                    dtype=torch.float32,
                    device=omega.device,
                )
                row.append(torch.sigmoid(self.keep_head(features)).squeeze(-1))
            rows.append(torch.stack(row, dim=0))
        keep_matrix = torch.stack(rows, dim=0)
        anchor_kappa = anchor_state.kappa.to(omega.device)[:, None]
        return torch.sum(omega * anchor_kappa * keep_matrix)

    def pairwise_score(self, anchor_state: VerifierState, cand_state: VerifierState) -> SelectorCandidateView:
        omega = self._alignment(anchor_state, cand_state)
        m_keep = self._m_keep(anchor_state, cand_state, omega)
        anchor_role_ans = anchor_state.role_ans.to(omega.device)
        anchor_role_evd = anchor_state.role_evd.to(omega.device)
        anchor_role_mix = anchor_state.role_mix.to(omega.device)
        anchor_eps = anchor_state.epsilon.to(omega.device)
        anchor_mu = anchor_state.mu.to(omega.device)
        cand_eps = cand_state.epsilon.to(omega.device)
        cand_mu = cand_state.mu.to(omega.device)
        delta_err = torch.sum(omega * anchor_role_ans[:, None] * (anchor_eps[:, None] - cand_eps[None, :]))
        delta_sup = torch.sum(omega * anchor_role_evd[:, None] * (anchor_mu[:, None] - cand_mu[None, :]))
        delta_mix_err = torch.sum(omega * anchor_role_mix[:, None] * (anchor_eps[:, None] - cand_eps[None, :]))
        delta_mix_sup = torch.sum(omega * anchor_role_mix[:, None] * (anchor_mu[:, None] - cand_mu[None, :]))
        delta_type = float(cand_state.typed_support_score - anchor_state.typed_support_score)
        delta_ctr = float(anchor_state.contract_residual.pooled - cand_state.contract_residual.pooled)
        d_sig = float(typed_distance(cand_state.artifact.answer_object, anchor_state.artifact.answer_object))
        base_score = self.base_head(
            torch.tensor(
                [
                    float(delta_err.item()),
                    float(delta_sup.item()),
                    float(delta_mix_err.item()),
                    float(delta_mix_sup.item()),
                    float(m_keep.item()),
                    delta_type,
                    delta_ctr,
                    d_sig,
                    float(anchor_state.summary_vector.mean().item()),
                    float(cand_state.summary_vector.mean().item()),
                ],
                dtype=torch.float32,
                device=omega.device,
            )
        ).squeeze(-1)
        flip_barrier = torch.nn.functional.softplus(torch.tensor(d_sig - delta_type, dtype=torch.float32, device=omega.device))
        safe_score = base_score - flip_barrier
        return SelectorCandidateView(
            candidate_id="candidate",
            score=safe_score,
            probability=torch.tensor(0.0),
            delta_type=delta_type,
            delta_ctr=delta_ctr,
            d_sig=d_sig,
            m_keep=float(m_keep.item()),
        )

    def forward(self, anchor_state: VerifierState, candidate_states: Sequence[VerifierState]) -> SelectorOutput:
        device = next(self.parameters()).device
        logits = [torch.tensor(0.0, dtype=torch.float32, device=device)]
        views: List[SelectorCandidateView] = []
        for idx, cand_state in enumerate(candidate_states):
            view = self.pairwise_score(anchor_state, cand_state)
            view.candidate_id = f"candidate_{idx}"
            views.append(view)
            logits.append(view.score)
        logits_tensor = torch.stack(logits, dim=0)
        probs = torch.softmax(logits_tensor, dim=0)
        for index, view in enumerate(views, start=1):
            view.probability = probs[index]
        return SelectorOutput(
            logits=logits_tensor,
            probabilities=probs,
            winner_index=int(torch.argmax(probs).item()),
            candidate_views=views,
        )
