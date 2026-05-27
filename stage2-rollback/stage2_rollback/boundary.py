from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
from torch import nn

from .trace_ir import RollbackTrace
from .verifier import VerifierState


@dataclass
class BoundaryFeatures:
    step_features: torch.Tensor
    macro_blocks: Dict[int, List[int]]
    cumulative_prefix: torch.Tensor
    no_rerun_index: int = 0


@dataclass
class BoundaryOutput:
    step_features: torch.Tensor
    macro_logits: torch.Tensor
    final_logits: torch.Tensor
    pi_tilde_rb: torch.Tensor
    predicted_boundary: int


def build_boundary_features(state: VerifierState, trace: RollbackTrace) -> BoundaryFeatures:
    B = state.unit_latents.unit_to_trace
    if B.ndim == 1:
        B = B.unsqueeze(0)
    if B.shape[1] <= 1:
        cumulative = torch.zeros((B.shape[0], 1), dtype=torch.float32)
    else:
        cumulative = torch.cumsum(B[:, 1:], dim=1)
    step_rows: List[torch.Tensor] = []
    for step in trace.steps:
        j = step.index
        c_ij = cumulative[:, j - 1] if cumulative.shape[1] >= j else torch.zeros(B.shape[0], dtype=torch.float32)
        m_pre = torch.sum(c_ij)
        m_suf = torch.sum(1.0 - c_ij)
        sbar = torch.sum(c_ij * state.kappa) / (m_pre + 1e-6)
        rbar_ans = torch.sum((1.0 - c_ij) * state.role_ans * state.epsilon) / (m_suf + 1e-6)
        rbar_evd = torch.sum((1.0 - c_ij) * state.role_evd * state.mu) / (m_suf + 1e-6)
        rbar_mix_eps = torch.sum((1.0 - c_ij) * state.role_mix * state.epsilon) / (m_suf + 1e-6)
        rbar_mix_mu = torch.sum((1.0 - c_ij) * state.role_mix * state.mu) / (m_suf + 1e-6)
        residual = step.contract_residual
        step_rows.append(
            torch.tensor(
                [
                    float(sbar.item()),
                    float(m_pre.item()),
                    float(rbar_ans.item()),
                    float(rbar_evd.item()),
                    float(rbar_mix_eps.item()),
                    float(rbar_mix_mu.item()),
                    float(m_suf.item()),
                    float(step.local_signal.get("support_count", 0.0)),
                    float(step.local_signal.get("avg_graph_score", 0.0)),
                    float(step.local_signal.get("root_frequency", 0.0)),
                    float(step.local_signal.get("sink_frequency", 0.0)),
                    float(residual.get("parse", 0.0)),
                    float(residual.get("completeness", 0.0)),
                    float(residual.get("constraint", 0.0)),
                    float(residual.get("execution", 0.0)),
                    float(step.index) / max(1.0, float(len(trace.steps))),
                ],
                dtype=torch.float32,
            )
        )
    return BoundaryFeatures(
        step_features=torch.stack(step_rows, dim=0) if step_rows else torch.zeros((1, 16), dtype=torch.float32),
        macro_blocks=dict(trace.macro_blocks),
        cumulative_prefix=cumulative,
    )


class RollbackBoundaryModel(nn.Module):
    def __init__(self, input_dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.step_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.no_rerun_head = nn.Sequential(nn.Linear(hidden_dim + 3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.macro_head = nn.Sequential(nn.Linear(hidden_dim + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        self.micro_head = nn.Sequential(nn.Linear(hidden_dim * 2 + 4, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))

    def forward(self, features: BoundaryFeatures) -> BoundaryOutput:
        device = next(self.parameters()).device
        step_encoded = self.step_encoder(features.step_features.to(device))
        macro_logits = []
        final_logits = [torch.tensor(0.0, dtype=step_encoded.dtype, device=step_encoded.device)]
        if step_encoded.shape[0] == 0:
            return BoundaryOutput(step_features=step_encoded, macro_logits=torch.zeros(1), final_logits=torch.zeros(1), pi_tilde_rb=torch.ones(1), predicted_boundary=0)
        global_step_pool = step_encoded.mean(dim=0)
        no_rerun_input = torch.cat(
            [global_step_pool, torch.tensor([1.0, 0.0, 0.0], dtype=step_encoded.dtype, device=step_encoded.device)],
            dim=0,
        )
        no_rerun_logit = self.no_rerun_head(no_rerun_input).squeeze(-1)
        macro_logits.append(no_rerun_logit)
        for block_id in sorted(features.macro_blocks):
            indexes = [idx - 1 for idx in features.macro_blocks[block_id] if 0 < idx <= step_encoded.shape[0]]
            if not indexes:
                continue
            block_tensor = step_encoded[indexes]
            pooled = block_tensor.mean(dim=0)
            first_index = min(indexes)
            last_index = max(indexes)
            suffix_mass = float(features.step_features[last_index, 6].item())
            macro_input = torch.cat(
                [
                    pooled,
                    torch.tensor(
                        [
                            float(features.step_features[first_index, 0].item()),
                            suffix_mass,
                            float(len(indexes)) / max(1.0, float(step_encoded.shape[0])),
                            float(features.step_features[first_index, 11:15].mean().item()),
                        ],
                        dtype=step_encoded.dtype,
                        device=step_encoded.device,
                    ),
                ],
                dim=0,
            )
            macro_logits.append(self.macro_head(macro_input).squeeze(-1))
        macro_logits_tensor = torch.stack(macro_logits, dim=0)
        macro_probs = torch.softmax(macro_logits_tensor, dim=0)
        for block_offset, block_id in enumerate(sorted(features.macro_blocks), start=1):
            indexes = [idx - 1 for idx in features.macro_blocks[block_id] if 0 < idx <= step_encoded.shape[0]]
            if not indexes:
                continue
            block_pool = step_encoded[indexes].mean(dim=0)
            micro_logits = []
            for idx in indexes:
                micro_input = torch.cat(
                    [
                        step_encoded[idx],
                        block_pool,
                        torch.tensor(
                            [
                                float(features.step_features[idx, 11].item()),
                                float(features.step_features[idx, 12].item()),
                                float(features.step_features[idx, 13].item()),
                                float(features.step_features[idx, 14].item()),
                            ],
                            dtype=step_encoded.dtype,
                            device=step_encoded.device,
                        ),
                    ],
                    dim=0,
                )
                micro_logits.append(self.micro_head(micro_input).squeeze(-1))
            micro_logits_tensor = torch.stack(micro_logits, dim=0)
            micro_probs = torch.softmax(micro_logits_tensor, dim=0)
            for local_idx, idx in enumerate(indexes):
                final_logits.append(macro_probs[block_offset] + torch.log(micro_probs[local_idx] + 1e-8))
        final_logits_tensor = torch.stack(final_logits, dim=0)
        pi_tilde_rb = torch.softmax(final_logits_tensor, dim=0)
        predicted_boundary = int(torch.argmax(pi_tilde_rb).item())
        return BoundaryOutput(
            step_features=step_encoded,
            macro_logits=macro_logits_tensor,
            final_logits=final_logits_tensor,
            pi_tilde_rb=pi_tilde_rb,
            predicted_boundary=predicted_boundary,
        )
