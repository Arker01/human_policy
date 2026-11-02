from __future__ import annotations

import torch

from object_transfer.fusion import GatedObjectFusion
from object_transfer.losses import domain_weighted_mean, object_localization_loss
from object_transfer.object_head import ObjectHeatmapHead


def _output(batch: int = 3):
    torch.manual_seed(7)
    return ObjectHeatmapHead(32, 16)(torch.randn(batch, 32, 14, 19))


def test_vjepa_grid_shape_and_normalized_uv():
    output = _output()
    assert output["heatmap_logits"].shape == (3, 14 * 19)
    assert torch.all((output["uv"] >= 0) & (output["uv"] <= 1))


def test_invalid_object_labels_contribute_no_loss():
    output = _output()
    zero = torch.zeros(3)
    losses = object_localization_loss(
        output, torch.zeros(3, 2), zero, zero, zero, zero)
    assert losses["loss"].item() == 0.0


def test_human_action_supervision_is_smaller_but_not_discarded():
    per_sample = torch.tensor([2.0, 2.0, 5.0, 5.0])
    is_human = torch.tensor([0, 0, 1, 1], dtype=torch.bool)
    total, robot, human = domain_weighted_mean(per_sample, is_human, 0.2)
    assert robot.item() == 2.0
    assert human.item() == 5.0
    assert total.item() == 3.0  # robot + 0.2 * human


def test_disabled_gate_preserves_action_hidden():
    fusion = GatedObjectFusion(32, 16, gate_init=-100.0)
    hidden = torch.randn(2, 4, 16)
    output = ObjectHeatmapHead(32, 8)(torch.randn(2, 32, 3, 5))
    fused = fusion(hidden, output)
    assert torch.equal(hidden, fused)


def test_no_action_translation_constant_in_object_modules():
    # 20.6 cm belongs to camera extrinsics.  Object supervision carries only
    # normalized image coordinates and has no path that mutates state/action.
    output = _output(1)
    assert set(output) == {
        "heatmap_logits", "heatmap_probability", "object_token", "uv",
        "present_logit", "grid_hw",
    }
