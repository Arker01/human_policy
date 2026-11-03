from __future__ import annotations

import torch
from torch import nn


class GatedObjectFusion(nn.Module):
    """Residual object conditioning that leaves the original HAT path intact."""

    def __init__(
        self,
        object_dim: int,
        hidden_dim: int,
        gate_init: float = 0.0,
        use_presence_gate: bool = False,
    ) -> None:
        super().__init__()
        self.use_presence_gate = bool(use_presence_gate)
        self.context = nn.Sequential(
            nn.LayerNorm(object_dim + 3),
            nn.Linear(object_dim + 3, hidden_dim),
            nn.GELU(),
        )
        self.delta = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate_logit = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, hidden: torch.Tensor, object_output: dict[str, torch.Tensor]) -> torch.Tensor:
        present = object_output["present_logit"].sigmoid()
        present_context = present if self.use_presence_gate else torch.zeros_like(present)
        raw_context = torch.cat(
            (
                object_output["object_token"],
                object_output["uv"],
                present_context[:, None],
            ),
            dim=-1,
        )
        context = self.context(raw_context).unsqueeze(1).expand(-1, hidden.shape[1], -1)
        delta = self.delta(torch.cat((hidden, context), dim=-1))
        gate = self.gate_logit.sigmoid()
        if self.use_presence_gate:
            gate = gate * present[:, None, None]
        return hidden + gate * delta

