"""Exact training model + live current-head ZED geometry, with original service I/O."""
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn


def camera_for_state(raw_state, parameters, adapter_dir, device):
    sys.path.insert(0, str(adapter_dir)) if str(adapter_dir) not in sys.path else None
    from camera_geometry import poses_from_state
    from spatial_conditioner import resize_crop_affine
    kp_from_head = poses_from_state(np.asarray(raw_state)[None, :9])[0]
    head_from_camera = np.array(parameters['T_head_link_from_left_optical'], dtype=np.float64)
    head_from_camera[:3, 3] = parameters['t_virtual_head_from_left_optical']
    camera_from_kp = np.linalg.inv(head_from_camera) @ np.linalg.inv(kp_from_head)
    values = dict(K=parameters['K'], D=parameters['D'], T_cam_from_kp=camera_from_kp,
                  raw_hw=parameters['image_hw'], valid=True, hand_valid=[True, True])
    camera = {k: torch.as_tensor(v, device=device).unsqueeze(0) for k, v in values.items()}
    camera['pixel_from_raw'] = resize_crop_affine((240, 320), (240, 320)).to(device).unsqueeze(0)
    return camera


class ConditionedPolicy(nn.Module):
    def __init__(self, core, stats, device):
        super().__init__()
        self.core = core
        self.camera = None
        self.register_buffer('action_mean', torch.as_tensor(stats['action_mean'], device=device).float()[None])
        self.register_buffer('action_std', torch.as_tensor(stats['action_std'], device=device).float()[None])

    def forward(self, image, qpos, conditioning_dict=None):
        if self.camera is None:
            raise RuntimeError('SCDP live camera conditioning was not prepared')
        cond = dict(conditioning_dict or {})
        cond.update(scdp_camera=self.camera, scdp_action_mean=self.action_mean, scdp_action_std=self.action_std)
        return self.core(image, qpos, conditioning_dict=cond)


class UnusedDinoTarget(nn.Module):
    """Training-only, unregistered teacher metadata; never used by online inference."""
    def __init__(self, name):
        super().__init__()
        if name != 'dinov2_vits14':
            raise ValueError('Unexpected training teacher in this inference bundle')
        self.name, self.embed_dim, self.patch_size = name, 384, 14
        self.expects_clip, self.tubelet_size = False, 1

    def forward(self, *args, **kwargs):
        raise RuntimeError('Online-only SCDP loader cannot compute training teacher targets')


def make_online_class(original):
    class SCDPOnlineHATPolicy(original.OnlineHATPolicy):
        def _load_policy(self, checkpoint):
            bundle = checkpoint.resolve().parent
            section = self.config.get('scdp')
            if not section:
                return super()._load_policy(checkpoint)
            if self.policy_class != 'ACT' or self.history_offsets or self.action_representation != 'original':
                raise ValueError('This SCDP bundle requires ACT, no history, original action representation')
            if self.camera_names != ('top',):
                raise ValueError('This SCDP bundle uses one raw left-eye ZED stream mapped to top')
            adapter = torch.load(bundle / section['adapter'], map_location='cpu', weights_only=False)
            digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest != adapter['base_weights_sha256'] or adapter['base_variant'] != section['base_variant']:
                raise ValueError('SCDP adapter and base checkpoint do not match')
            sys.path[:0] = [str(bundle / 'source/hdt'), str(bundle / 'source/hdt/detr'),
                            str(bundle / 'source'), str(bundle / 'adapter')]
            os.environ['HDT_EXPERIMENT'] = 'geometry'
            from policy import ACTPolicy
            from hat_adapter import SCDPHAT
            saved = json.loads((bundle / 'training_config.json').read_text())
            args = saved['config']['policy_config'].copy()
            args['experiment_variant'] = saved['variant']
            from detr.models import detr_vae
            target_class = detr_vae.FrozenPatchTargetEncoder

            def inference_target(name='dinov2_vits14', **kwargs):
                # V-JEPA trunk stays real. DINO teacher is outside state_dict and
                # gated by future_image != None, which this online path never sends.
                if name.startswith('dinov2'):
                    return UnusedDinoTarget(name)
                return target_class(name, **kwargs)

            detr_vae.FrozenPatchTargetEncoder = inference_target
            try:
                policy = ACTPolicy(args)
            finally:
                detr_vae.FrozenPatchTargetEncoder = target_class
            policy.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True), strict=True)
            expected = torch.ones_like(policy.model.state_keep_mask)
            for lo, hi in adapter['input_mask_ranges']:
                expected[lo:hi] = 0
            if not torch.equal(policy.model.state_keep_mask, expected):
                raise ValueError('Input mask differs from training')
            wrapper = SCDPHAT(policy.model, adapter['conditioner_config'])
            wrapper.conditioner.load_state_dict(adapter['conditioner'], strict=True)
            policy.model = wrapper
            self.scdp_wrapper = wrapper
            self.scdp_bundle = bundle
            self.scdp_camera_parameters = json.loads((bundle / section['camera']).read_text())['robot_zed_left']
            self.scdp_step = adapter['step']
            _, preprocessor = original.make_visual_encoder('ACT', None)
            print(f"[SCDP] loaded {adapter['base_variant']} step={adapter['step']} anchors=8 five-scale-v2; frozen base SHA256={digest}", flush=True)
            return ConditionedPolicy(policy, self.stats, self.device).eval().to(self.device), preprocessor, False

        @torch.inference_mode()
        def infer(self, images, robot_state, origin=None, normalized_state=None,
                  heading_quat_wxyz=None, history_images=None,
                  normalized_history_states=None, action_anchor=None):
            if hasattr(self, 'scdp_wrapper'):
                if np.asarray(images).shape != (1, 3, 240, 320):
                    raise ValueError('SCDP calibration requires raw left-eye ZED images resized to (1,3,240,320)')
                # Use measured current head, never a masked/normalized/scheduled state.
                raw, _ = self.prepare_policy_state(robot_state, origin, heading_quat_wxyz)
                self.policy.camera = camera_for_state(raw, self.scdp_camera_parameters,
                                                     self.scdp_bundle / 'adapter', self.device)
            try:
                return super().infer(images, robot_state, origin, normalized_state,
                                     heading_quat_wxyz, history_images,
                                     normalized_history_states, action_anchor)
            finally:
                if hasattr(self, 'scdp_wrapper'):
                    self.policy.camera = None

    return SCDPOnlineHATPolicy
