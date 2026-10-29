"""Five real convolutional scales from frozen intermediate V-JEPA patch tokens.

No image/feature upsampling. Stride-2, kernel-3, padding-1 convolutions keep
feature centre origin at native index zero. Sampling explicitly accounts for
their stride, rather than reusing normalized grids on differently padded maps.
"""
import torch
from torch import nn
from spatial_conditioner import SpatialTrajectoryConditioner, sample_local


def grid_at_stride(grid, native_hw, output_hw, stride):
    native_wh = grid.new_tensor([native_hw[1], native_hw[0]])
    output_wh = grid.new_tensor([output_hw[1], output_hw[0]])
    native_index = ((grid + 1) * native_wh - 1) / 2
    return 2 * (native_index / stride + .5) / output_wh - 1


class FeaturePyramid(nn.Module):
    source_layers = (3, 6, 9, 12, 12)
    strides = (1, 2, 4, 8, 16)

    def __init__(self, input_dim=768, channels=128):
        super().__init__()
        self.levels = nn.ModuleList()
        for level in range(5):
            layers = [nn.Conv2d(input_dim, channels, 1), nn.GroupNorm(8, channels), nn.GELU()]
            for step in range(max(1, level)):
                stride = 1 if level == 0 else 2
                layers += [nn.Conv2d(channels, channels, 3, stride=stride, padding=1,
                                     groups=channels), nn.GELU()]
            self.levels.append(nn.Sequential(*layers))

    def forward(self, sources):
        if len(sources) != 5 or any(tuple(x.shape[-2:]) != (14, 19) for x in sources):
            raise ValueError('Expected five intermediate sources on the native 14x19 patch grid')
        return [module(source.detach()) for module, source in zip(self.levels, sources)]


class MultiScaleTrajectoryConditioner(SpatialTrajectoryConditioner):
    def __init__(self, feature_dim=768, pyramid_channels=128, **kwargs):
        super().__init__(feature_dim=5 * pyramid_channels, **kwargs)
        self.pyramid = FeaturePyramid(feature_dim, pyramid_channels)
        self.pyramid_debug = {}

    def aggregate(self, sources, grid, valid):
        maps = self.pyramid(sources)
        contexts = []
        for feature, stride in zip(maps, self.pyramid.strides):
            level_grid = grid_at_stride(grid, (14, 19), feature.shape[-2:], stride)
            local = sample_local(feature, level_grid, valid)
            contexts.append(local.sum(1) / valid.sum(1).clamp_min(1)[..., None])
        self.pyramid_debug = dict(pyramid_shapes=[tuple(x.shape) for x in maps],
            pyramid_source_layers=list(self.pyramid.source_layers),
            pyramid_strides=list(self.pyramid.strides))
        # Per-hand per-scale concatenation, then the same zero-initialized FiLM.
        return torch.cat(contexts, dim=-1)
