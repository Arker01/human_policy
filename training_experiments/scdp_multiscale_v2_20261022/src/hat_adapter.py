"""Isolated wrapper: original HAT code and its return contract stay untouched."""
import torch
from torch import nn
from torch.nn import functional as F
from spatial_conditioner import SpatialTrajectoryConditioner


class SCDPHAT(nn.Module):
    def __init__(self, baseline, config=None):
        super().__init__()
        config = dict(config or {})
        if getattr(baseline, 'use_history', False) or len(baseline.camera_names) != 1:
            raise ValueError('First version requires the specified one-camera, no-history HAT')
        self.base = baseline
        self.multiscale = config.get('spatial_mode', 'single') == 'five_scale_v2'
        if config.get('enabled', False):
            self.base.requires_grad_(False)
        from hdt.constants import OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF
        indices = [OUTPUT_LEFT_EEF[:3].tolist(), OUTPUT_RIGHT_EEF[:3].tolist()]
        conditioner_class = SpatialTrajectoryConditioner
        extra = {}
        if self.multiscale:
            from multiscale_conditioner import MultiScaleTrajectoryConditioner
            conditioner_class = MultiScaleTrajectoryConditioner
            extra['pyramid_channels'] = config.get('pyramid_channels', 128)
        self.conditioner = conditioner_class(
            feature_dim=baseline.backbones[0][0].num_channels,
            hidden_dim=baseline.action_head.in_features,
            film_dim=config.get('film_dim', 256), anchors=config.get('anchors', 8),
            wrist_indices=indices, enabled=config.get('enabled', False), **extra)
        self.debug = None

    # Attributes accessed by the original ACTPolicy.
    @property
    def num_queries(self):
        return self.base.num_queries

    @property
    def experiment_variant(self):
        return getattr(self.base,'experiment_variant','baseline')

    @property
    def readiness_logits(self):
        return getattr(self.base,'readiness_logits',None)

    def train(self, mode=True):
        super().train(mode)
        if self.conditioner.enabled:
            self.base.eval()  # Frozen trunk, including dropout and CVAE encoder.
        return self

    def forward(self, qpos, image, env_state, actions=None, is_pad=None,
                conditioning_dict=None, future_image=None):
        self.debug = None
        if not self.conditioner.enabled:
            return self.base(qpos, image, env_state, actions, is_pad,
                             conditioning_dict, future_image=future_image)
        cond = conditioning_dict or {}
        cache = {}

        def capture(_module, _inputs, output):
            cache['map'] = output['0']

        def refine(head, inputs, coarse):
            features = cache['map']
            if self.multiscale:
                features = [cache[layer] for layer in (3, 6, 9)] + [features, features]
            hidden, debug = self.conditioner(features, coarse, inputs[0],
                cond.get('scdp_camera'), cond.get('scdp_action_mean'), cond.get('scdp_action_std'))
            candidate = F.linear(hidden, head.weight, head.bias)
            # Preserve the coarse output for EVERY other channel, bit for bit.
            channel_mask = torch.zeros(coarse.shape[0],coarse.shape[-1], dtype=torch.bool, device=coarse.device)
            for hand,indices in enumerate(self.conditioner.wrist_indices):
                channel_mask[:,indices] = debug['valid'][:,:,hand].any(1)[:,None]
            channel_mask = channel_mask[:,None,:]
            refined = torch.where(channel_mask, candidate, coarse)
            # No recomputed linear GEMM on unavailable samples: preserve the exact
            # coarse values even if autograd selects a different numerical kernel.
            refined = torch.where(debug['valid'].any((1, 2))[:, None, None], refined, coarse)
            self.debug = dict(debug, coarse=coarse, refined=refined,
                              feature_shape=tuple(cache['map'].shape), hidden_shape=tuple(inputs[0].shape))
            if self.multiscale:
                self.debug.update(self.conditioner.pyramid_debug)
            return refined

        handles = [self.base.backbones[0][0].register_forward_hook(capture),
                   self.base.action_head.register_forward_hook(refine)]
        if self.multiscale:
            body = self.base.backbones[0][0].encoder.body
            if len(body.blocks) != 12:
                for handle in handles: handle.remove()
                raise ValueError('Five-scale source layers require the specified 12-block V-JEPA ViT-B')
            def capture_layer(layer):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        output = output[0]  # V-JEPA 2.1 Block returns (tokens, attention).
                    if output.ndim != 3 or tuple(output.shape[1:]) != (266, 768):
                        raise ValueError('Intermediate V-JEPA tokens must be one temporal slice, 14x19x768')
                    normalizer = body.norms_block[body.hierarchical_layers.index(layer-1)]
                    cache[layer] = normalizer(output.detach()).reshape(-1,14,19,768).permute(0,3,1,2)
                return hook
            handles += [body.blocks[layer-1].register_forward_hook(capture_layer(layer))
                        for layer in (3, 6, 9)]
        try:
            return self.base(qpos, image, env_state, actions, is_pad,
                             conditioning_dict, future_image=future_image)
        finally:
            for handle in handles:
                handle.remove()
