"""Exercise the actual snapshot state mask and ACT loss on left-hand dimensions."""
import json
from pathlib import Path

import torch
from torch import nn

import hdt.constants as C
import hdt.policy as policy_module
from hdt.detr.models.detr_vae import DETRVAE


class Backbone(nn.Module):
    num_channels = 3


class Transformer(nn.Module):
    d_model = 32


class Prediction(nn.Module):
    num_queries = 3

    def __init__(self):
        super().__init__()
        self.prediction = nn.Parameter(torch.ones(2, 3, 128))

    def forward(self, qpos, *args, **kwargs):
        self.observed_qpos = qpos.detach().clone()
        zeros = self.prediction.new_zeros(2, 32)
        return self.prediction, None, (zeros, zeros), None


def main():
    root = Path(__file__).resolve().parent
    assert Path(policy_module.__file__).resolve().is_relative_to(root / 'source')
    model = DETRVAE([Backbone()], Transformer(), nn.Identity(), 128, 128, 3,
                    ['top'], 'ACT_linear', False, zero_state_dims=(98, 128))
    left = list(C.OUTPUT_LEFT_KEYPOINTS) + list(C.OUTPUT_LEFT_EEF)
    assert torch.all(model.state_keep_mask[left] == 1)
    assert torch.all(model.state_keep_mask[98:] == 0)
    policy = policy_module.ACTPolicy.__new__(policy_module.ACTPolicy)
    nn.Module.__init__(policy)
    policy.model = Prediction()
    policy.transform = nn.Identity()
    policy.kl_weight = 10
    policy.query0_extra_weight = 0
    policy.USE_EEF_LOSS = True
    qpos = torch.arange(128, dtype=torch.float32).repeat(2, 1)
    losses = policy(torch.zeros(2, 1, 3, 2, 2), qpos,
                    torch.zeros(2, 3, 128), torch.zeros(2, 3, dtype=torch.bool), {})
    losses['loss'].backward()
    gradient = policy.model.prediction.grad
    assert torch.all(gradient[:, :, left] > 0), 'Left hand must contribute to loss'
    assert torch.all(gradient[:, :, C.OUTPUT_NECK] == 0)
    torch.testing.assert_close(policy.model.observed_qpos[:, left], qpos[:, left])
    report = dict(status='pass', policy_source=policy_module.__file__,
                  left_hand_state_mask_all_one=True, left_hand_state_passed_to_model=True,
                  left_eef_loss_gradient_min=float(gradient[:, :, C.OUTPUT_LEFT_EEF].min()),
                  left_keypoint_loss_gradient_min=float(gradient[:, :, C.OUTPUT_LEFT_KEYPOINTS].min()),
                  only_normalized_state_tail_zeroed='98:128', existing_neck_loss_mask_preserved=True)
    (root / 'left_hand_verification.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
