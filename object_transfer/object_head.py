from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ObjectHeatmapHead(nn.Module):
    """Small spatial head over the frozen V-JEPA patch grid.

    The input is the unpooled V-JEPA feature map ``[B, C, H, W]``.  The head
    predicts a categorical distribution over patches and a separate presence
    probability; the latter is deliberately not derived from softmax confidence.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, 1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, 1),
        )
        self.present = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 4:
            raise ValueError(f"expected BCHW features, got {tuple(features.shape)}")
        batch, channels, grid_h, grid_w = features.shape
        logits = self.score(features).flatten(1)
        probability = logits.softmax(dim=-1)
        tokens = features.flatten(2).transpose(1, 2)
        object_token = torch.bmm(probability.unsqueeze(1), tokens).squeeze(1)

        ys = (torch.arange(grid_h, device=features.device, dtype=features.dtype) + 0.5) / grid_h
        xs = (torch.arange(grid_w, device=features.device, dtype=features.dtype) + 0.5) / grid_w
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        coordinates = torch.stack((xx, yy), dim=-1).reshape(1, grid_h * grid_w, 2)
        uv = (probability.unsqueeze(-1) * coordinates).sum(dim=1)
        present_logit = self.present(features.mean(dim=(2, 3))).squeeze(-1)
        return {
            "heatmap_logits": logits,
            "heatmap_probability": probability,
            "object_token": object_token,
            "uv": uv,
            "present_logit": present_logit,
            "grid_hw": torch.tensor((grid_h, grid_w), device=features.device),
        }


def soft_point_target(
    uv: torch.Tensor,
    grid_h: int,
    grid_w: int,
    sigma_patches: float,
) -> torch.Tensor:
    """Return normalized Gaussian patch targets for normalized ``(u, v)``."""
    ys = torch.arange(grid_h, device=uv.device, dtype=uv.dtype) + 0.5
    xs = torch.arange(grid_w, device=uv.device, dtype=uv.dtype) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    target_x = uv[:, 0, None, None] * grid_w
    target_y = uv[:, 1, None, None] * grid_h
    squared = (xx[None] - target_x).square() + (yy[None] - target_y).square()
    target = torch.exp(-0.5 * squared / max(float(sigma_patches), 1e-3) ** 2)
    return target.flatten(1) / target.flatten(1).sum(dim=1, keepdim=True).clamp_min(1e-8)

