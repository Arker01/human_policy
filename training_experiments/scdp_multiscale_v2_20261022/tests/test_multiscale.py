import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch.nn import functional as F
from multiscale_conditioner import FeaturePyramid, MultiScaleTrajectoryConditioner, grid_at_stride
from test_spatial import camera


class MultiscaleTests(unittest.TestCase):
    def test_real_scales_and_all_gradient_paths(self):
        torch.manual_seed(8)
        module = MultiScaleTrajectoryConditioner(feature_dim=8, pyramid_channels=8,
            hidden_dim=8, film_dim=16, enabled=True)
        sources = [torch.randn(2,8,14,19,requires_grad=True) for _ in range(5)]
        coarse = torch.zeros(2,10,128); coarse[...,32] = 1; coarse[...,82] = 1
        hidden = torch.randn(2,10,8)
        cam = {k:v.expand(2,*v.shape[1:]).clone() for k,v in camera().items()}
        mean = torch.zeros(2,128); std = torch.ones(2,128)
        output, _ = module(sources,coarse,hidden,cam,mean,std)
        self.assertTrue(torch.equal(output,hidden), 'Zero FiLM must preserve the baseline')
        self.assertEqual([x[-2:] for x in module.pyramid_debug['pyramid_shapes']],
                         [(14,19),(7,10),(4,5),(2,3),(1,2)])
        output.square().mean().backward()
        self.assertGreater(module.film[-1].weight.grad.abs().sum().item(),0)
        module.zero_grad(set_to_none=True)
        with torch.no_grad():module.film[-1].weight.normal_(0,.01)
        output,_ = module(sources,coarse,hidden,cam,mean,std)
        output.square().mean().backward()
        for level in module.pyramid.levels:
            for parameter in level.parameters():
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(sum(p.grad.abs().sum().item() for p in level.parameters()),0)
        self.assertTrue(all(x.grad is None for x in sources))
        out,_ = module(sources,coarse,hidden,None,mean,std)
        self.assertTrue(torch.equal(out,hidden), 'Missing calibration fallback must remain exact')

    def test_stride_grid_centres_match_convolution(self):
        # Native centre (8, 0) stays at level index (0.5, 0) for stride 16.
        x,y = 8.,0.
        grid = torch.tensor([[[[2*(x+.5)/19-1,2*(y+.5)/14-1]]]])
        mapped = grid_at_stride(grid,(14,19),(1,2),16)
        values = torch.tensor([[[[0.,16.]]]])
        actual = F.grid_sample(values,mapped,align_corners=False)
        self.assertAlmostEqual(actual.item(),8.,places=5)
        # Reusing the original normalized grid gives a different, wrong location.
        wrong = F.grid_sample(values,grid,align_corners=False)
        self.assertGreater(abs(wrong.item()-actual.item()),1.)

    def test_coarse_scales_integrate_neighbours(self):
        pyramid=FeaturePyramid(input_dim=8,channels=8)
        self.assertEqual(len(pyramid.levels),5)
        for index,level in enumerate(pyramid.levels):
            downs=[m for m in level if isinstance(m,torch.nn.Conv2d) and m.stride==(2,2)]
            self.assertEqual(len(downs),index)
        self.assertFalse(any(isinstance(m,torch.nn.Upsample) for m in pyramid.modules()))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
