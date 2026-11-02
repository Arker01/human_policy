from __future__ import annotations

import torch
from torch.nn import functional as F

from object_transfer.object_head import soft_point_target


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1.0)


def domain_weighted_mean(
    per_sample: torch.Tensor,
    is_human: torch.Tensor,
    human_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Keep the robot mean intact and add a smaller human-domain mean."""
    is_human = is_human.to(dtype=torch.bool, device=per_sample.device)
    robot = ~is_human
    zero = per_sample.sum() * 0.0
    robot_mean = per_sample[robot].mean() if robot.any() else zero
    human_mean = per_sample[is_human].mean() if is_human.any() else zero
    return robot_mean + float(human_weight) * human_mean, robot_mean, human_mean


def object_localization_loss(
    output: dict[str, torch.Tensor],
    target_uv: torch.Tensor,
    target_valid: torch.Tensor,
    target_quality: torch.Tensor,
    present_target: torch.Tensor,
    present_valid: torch.Tensor,
    sigma_patches: float = 1.0,
    uv_weight: float = 1.0,
    present_weight: float = 0.2,
) -> dict[str, torch.Tensor]:
    grid_h, grid_w = (int(v) for v in output["grid_hw"].tolist())
    soft_target = soft_point_target(target_uv, grid_h, grid_w, sigma_patches)
    point_ce = -(soft_target * output["heatmap_logits"].log_softmax(dim=-1)).sum(dim=-1)
    point_l1 = F.smooth_l1_loss(output["uv"], target_uv, reduction="none").sum(dim=-1)
    valid_weight = target_valid.to(point_ce.dtype) * target_quality.to(point_ce.dtype)
    point_loss = _masked_mean(point_ce + float(uv_weight) * point_l1, valid_weight)
    present_bce = F.binary_cross_entropy_with_logits(
        output["present_logit"], present_target.to(output["present_logit"].dtype), reduction="none"
    )
    present_loss = _masked_mean(present_bce, present_valid)
    pixel_error_normalized = (output["uv"] - target_uv).norm(dim=-1)
    return {
        "loss": point_loss + float(present_weight) * present_loss,
        "point_loss": point_loss,
        "present_loss": present_loss,
        "uv_error_norm": _masked_mean(pixel_error_normalized, target_valid),
        "valid_fraction": target_valid.float().mean(),
        "present_probability": output["present_logit"].sigmoid().mean(),
    }


def metric_relative_wrist_loss(
    predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    qpos: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    qpos_mean: torch.Tensor,
    qpos_std: torch.Tensor,
    is_pad: torch.Tensor,
    wrist_slices: tuple[slice, slice] = (slice(0, 3), slice(9, 12)),
) -> torch.Tensor:
    """Supervise future wrist displacement, independent of a fixed world origin."""
    predicted_metric = predicted_action * action_std[:, None] + action_mean[:, None]
    target_metric = target_action * action_std[:, None] + action_mean[:, None]
    current_metric = qpos * qpos_std + qpos_mean
    terms = []
    valid = (~is_pad).to(predicted_action.dtype)
    for wrist in wrist_slices:
        pred_delta = predicted_metric[:, :, wrist] - current_metric[:, None, wrist]
        target_delta = target_metric[:, :, wrist] - current_metric[:, None, wrist]
        per_step = F.smooth_l1_loss(pred_delta, target_delta, reduction="none").mean(dim=-1)
        terms.append((per_step * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0))
    return torch.stack(terms, dim=0).mean(dim=0)

