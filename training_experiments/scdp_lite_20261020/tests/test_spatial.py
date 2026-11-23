import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import unittest
import torch
from spatial_conditioner import SpatialTrajectoryConditioner, project_wrist, resize_crop_affine, sample_local


def camera(batch=1, hw=(224, 304)):
    return dict(K=torch.tensor([[100., 0., 151.5], [0., 100., 111.5], [0., 0., 1.]]).repeat(batch, 1, 1),
                T_cam_from_kp=torch.eye(4).repeat(batch, 1, 1),
                pixel_from_raw=torch.eye(3).repeat(batch, 1, 1),
                raw_hw=torch.tensor(hw).repeat(batch, 1), valid=torch.ones(batch, dtype=torch.bool))


class SpatialTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_projection_and_moving_camera(self):
        xyz = torch.tensor([[[[0., 0., 1.], [1., 0., 1.]]]])
        cam = camera()
        grid, valid = project_wrist(xyz, cam)
        self.assertTrue(valid.all())
        torch.testing.assert_close(grid[0, 0, 0], torch.zeros(2))
        torch.testing.assert_close(grid[0, 0, 1, 0], torch.tensor(200 / 304))
        cam['T_cam_from_kp'][0, 0, 3] = -1
        moved, _ = project_wrist(xyz, cam)
        torch.testing.assert_close(moved[0, 0, 1], torch.zeros(2))

    def test_masks_and_no_calibration(self):
        xyz = torch.tensor([[[[0., 0., -1.], [10., 0., 1.]],
                             [[float('nan'), 0., 1.], [0., 0., 0.]]]])
        for cam in (None, {}, camera()):
            grid, valid = project_wrist(xyz, cam)
            self.assertFalse(valid.any())
            self.assertTrue(torch.isfinite(grid).all())

    def test_resize_crop_pixel_centres(self):
        affine = resize_crop_affine((480, 640), (240, 320), (20, 30, 200, 250))
        raw = torch.tensor([200., 100., 1.])
        expected = torch.tensor([((200 + .5) * .5 - 30) * 304 / 250 - .5,
                                 ((100 + .5) * .5 - 20) * 224 / 200 - .5, 1.])
        torch.testing.assert_close(affine @ raw, expected)

    def test_native_grid_known_values(self):
        fmap = torch.arange(6.).reshape(1, 1, 2, 3)
        grid = torch.tensor([[[[-2 / 3, -.5], [2 / 3, .5]]]])
        local = sample_local(fmap, grid, torch.ones(1, 1, 2, dtype=torch.bool))
        torch.testing.assert_close(local.flatten(), torch.tensor([0., 5.]))

    def test_indices_zero_init_fallback_and_gradients(self):
        model = SpatialTrajectoryConditioner(4, 8, 12, 8, enabled=True)
        hidden = torch.randn(2, 100, 8)
        coarse = torch.zeros(2, 100, 128, requires_grad=True)
        with torch.no_grad():
            coarse[..., 82] = 1
            coarse[..., 32] = 2
        feature = torch.randn(2, 4, 14, 19)
        mean, std = torch.zeros(2, 128), torch.ones(2, 128) - 1e-6
        fused, debug = model(feature, coarse, hidden, camera(2), mean, std)
        self.assertEqual(tuple(debug['valid'].shape), (2, 8, 2))
        self.assertTrue(debug['valid'].all())
        torch.testing.assert_close(debug['wrist_xyz_m'][..., 0, 2], torch.ones(2, 8))
        torch.testing.assert_close(debug['wrist_xyz_m'][..., 1, 2], torch.full((2, 8), 2.))
        self.assertTrue(torch.equal(hidden, fused))
        fused.square().mean().backward()
        self.assertGreater(model.film[-1].weight.grad.abs().sum().item(), 0)
        self.assertIsNone(coarse.grad)
        with torch.no_grad():
            model.film[-1].bias.fill_(1.)
        invalid, _ = model(feature, coarse, hidden, None, mean, std)
        self.assertTrue(torch.equal(hidden, invalid))
        model.enabled = False
        disabled, _ = model(feature, coarse, hidden, camera(2), mean, std)
        self.assertTrue(torch.equal(hidden, disabled))

    def test_mixed_validity_and_nonfinite_camera(self):
        model = SpatialTrajectoryConditioner(4, 8, 12, enabled=True)
        model.film[-1].bias.data.fill_(.1)
        cam = camera(2)
        cam['K'][1, 0, 0] = float('nan')
        coarse = torch.zeros(2, 12, 128)
        coarse[..., [32, 82]] = 1.
        hidden = torch.randn(2, 12, 8)
        output, debug = model(torch.randn(2, 4, 14, 19), coarse, hidden, cam,
                              torch.zeros(2, 128), torch.ones(2, 128))
        self.assertTrue(debug['valid'][0].all())
        self.assertFalse(debug['valid'][1].any())
        self.assertTrue(torch.equal(output[1], hidden[1]))
        self.assertTrue(torch.isfinite(output).all())

    def test_synthetic_minibatch_overfit(self):
        # Geometry here is synthetic ONLY; it is never used for real training.
        model = SpatialTrajectoryConditioner(4, 8, 16, enabled=True)
        head = torch.nn.Linear(8, 3).requires_grad_(False)
        optimizer = torch.optim.Adam(model.parameters(), lr=.02)
        hidden = torch.randn(1, 10, 8)
        coarse = torch.zeros(1, 10, 128)
        coarse[..., [32, 82]] = 1.
        feature = torch.randn(1, 4, 14, 19)
        target = head(hidden).detach() + .2
        losses = []
        for _ in range(60):
            fused, _ = model(feature, coarse, hidden, camera(), torch.zeros(1, 128), torch.ones(1, 128))
            loss = (head(fused) - target).square().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * .05)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
