import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import runtime
import unittest
import torch
from torch import nn
from hat_adapter import SCDPHAT
from test_spatial import camera
from calibrated_data import RecordedCrop, collate
from torchvision.transforms import v2


class Patches(nn.Module):
    num_channels = 4
    def forward(self, image):
        return {'0': image}


class FakeHAT(nn.Module):
    """Synthetic fixture only, not a substitute for real-checkpoint smoke.py."""
    camera_names = ['top']
    num_queries = 10
    def __init__(self):
        super().__init__()
        self.backbones = nn.ModuleList([nn.Sequential(Patches())])
        self.action_head = nn.Linear(8, 128)
        self.hidden = nn.Parameter(torch.randn(1, 10, 8))
        with torch.no_grad():
            self.action_head.weight[[32, 82]] = 0
            self.action_head.bias[[32, 82]] = 1
    def forward(self, qpos, image, env, actions=None, pad=None, conditioning_dict=None, future_image=None):
        self.backbones[0](image)
        return self.action_head(self.hidden), None, [None, None], None


class AdapterTests(unittest.TestCase):
    def test_disabled_fallback_and_only_wrist_changes(self):
        torch.manual_seed(11)
        base = FakeHAT().eval()
        image = torch.randn(1, 4, 14, 19)
        original = base(None, image, None)[0].detach()
        wrapped = SCDPHAT(base, dict(enabled=False))
        self.assertTrue(torch.equal(original, wrapped(None, image, None)[0]))
        wrapped.conditioner.enabled = True
        wrapped.base.requires_grad_(False)
        cond = dict(scdp_camera=camera(), scdp_action_mean=torch.zeros(1, 128), scdp_action_std=torch.ones(1, 128))
        zero = wrapped(None, image, None, conditioning_dict=cond)[0]
        self.assertTrue(torch.equal(original, zero))
        wrapped.conditioner.film[-1].bias.data.fill_(.2)
        changed = wrapped(None, image, None, conditioning_dict=cond)[0]
        keep = torch.ones(128, dtype=torch.bool)
        keep[[80, 81, 82, 30, 31, 32]] = False
        self.assertTrue(torch.equal(original[..., keep], changed[..., keep]))
        self.assertGreater((original-changed).abs().sum().item(), 0.)
        changed.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in base.parameters()))
        self.assertGreater(wrapped.conditioner.film[-1].weight.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.equal(original, wrapped(None, image, None)[0]))
        self.assertEqual(len(base.action_head._forward_hooks), 0)
        self.assertEqual(len(base.backbones[0][0]._forward_hooks), 0)

    def test_recorded_crop_matches_original_rng_and_image(self):
        crop = v2.RandomResizedCrop((240, 320), scale=(.75, 1.), ratio=(1.2, 1.4815))
        recorder = RecordedCrop(crop)
        images = torch.rand(2, 3, 240, 320)
        torch.manual_seed(19); expected = crop(images)
        torch.manual_seed(19); actual = recorder(images)
        self.assertTrue(torch.equal(expected, actual))
        self.assertIsNotNone(recorder.last)

    def test_collate_preserves_geometry(self):
        def item():
            return (torch.zeros(1, 3, 240, 320), torch.zeros(128), torch.zeros(10, 128),
                    torch.zeros(10, dtype=torch.bool), dict(language_embeddings=None, plain_text='',
                    scdp_action_mean=torch.zeros(128), scdp_action_std=torch.ones(128),
                    scdp_camera={k:v[0] for k,v in camera().items()}))
        output = collate([item(), item()])
        self.assertEqual(output[4]['scdp_camera']['K'].shape, (2, 3, 3))
        self.assertEqual(output[4]['scdp_action_mean'].shape, (2, 128))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
