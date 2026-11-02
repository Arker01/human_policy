# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

import IPython
e = IPython.embed

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other policy_models than torchvision.policy_models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        # for name, parameter in backbone.named_parameters(): # only train later layers # TODO do we want this?
        #     if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
        #         parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        xs = self.body(tensor)
        return xs
        # out: Dict[str, NestedTensor] = {}
        # for name, x in xs.items():
        #     m = tensor_list.mask
        #     assert m is not None
        #     mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        #     out[name] = NestedTensor(x, mask)
        # return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d) # pretrained # TODO do we want frozen batch_norm??
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)

class DINOv2BackBone(nn.Module):
    def __init__(self, model_name: str='dinov2_vits14', image_feature_strategy: str='ACT_linear') -> None:
        super().__init__()
        self.body = torch.hub.load('facebookresearch/dinov2', model_name)
        self.body.eval()
        # This follows depth estimation indices used in
        # https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_nyu_linear4_config.py
        # These layers are used for both linear4 (similar to FPN) and DPT.
        self.image_feature_strategy = image_feature_strategy
        self.FEATURE_EXTRACT_INDICES = [2, 5, 8, 11]
        if model_name == 'dinov2_vits14':
            self.num_channels = 384
            self.model_stride = 14
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    @torch.no_grad()
    def forward(self, tensor):
        # The tensor from self.body.forward_features(tensor)["x_norm_patchtokens"] is equivalent to
        # self.body.get_intermediate_layers(tensor, n=[11], reshape=False, return_class_token=False, norm=True)[0]
        B, C, H, W = tensor.shape
        output_H, output_W = H // self.model_stride, W // self.model_stride
        od = OrderedDict()
        xs_cls_pair = self.body.get_intermediate_layers(tensor,
                                                   n=self.FEATURE_EXTRACT_INDICES,
                                                   reshape=False,
                                                   return_class_token=True,
                                                   norm=True)
        if self.image_feature_strategy == 'ACT_linear':
            xs = xs_cls_pair[-1][0]
            od["0"] = xs.reshape(xs.shape[0], output_H, output_W, self.num_channels).permute(0, 3, 1, 2)
        elif self.image_feature_strategy == 'linear':
            xs = xs_cls_pair[-1][0]
            xs = xs.reshape(xs.shape[0], output_H, output_W, self.num_channels).permute(0, 3, 1, 2)
            cls_token = xs_cls_pair[-1][1]
            cls_token = cls_token.reshape(cls_token.shape[0], self.num_channels, 1, 1).expand_as(xs)
            od["0"] = torch.cat([cls_token, xs], dim=1)
        elif self.image_feature_strategy == 'linear4':
            tensor_list = []
            for i, (xs, cls_token) in enumerate(xs_cls_pair):
                xs = xs.reshape(xs.shape[0], output_H, output_W, self.num_channels).permute(0, 3, 1, 2)
                cls_token = cls_token.reshape(cls_token.shape[0], self.num_channels, 1, 1).expand_as(xs)
                tensor_list.append(torch.cat([cls_token, xs], dim=1))
            od["0"] = torch.cat(tensor_list, dim=1)
        else:
            raise ValueError(f"Strategy {self.image_feature_strategy} not supported")
        return od
    
class VJEPABackBone(nn.Module):
    """V-JEPA 2.1 as the trunk's INPUT encoder, mirroring DINOv2BackBone above.

    This is the experiment the literature actually supports. omega-0 uses a frozen
    V-JEPA 2.1 encoder on the CURRENT observation (its future regression target is a
    frozen Wan VAE latent, not V-JEPA), and swapping that input encoder to Wan costs it
    15.5 SR points. Note what is NOT established anywhere: V-JEPA beating DINOv2 or a
    trained resnet18 in this slot. DexHand's encoder sweep (DINOv2 / DINOv3 / Web-SSL /
    V-JEPA 2 / SigLIP 2) put DINOv2 first. So this class exists to measure that, not
    because the answer is known.

    Two things to keep in mind when reading the comparison:
      * A video encoder needs `tubelet` (=2) frames. The trunk only ever has the current
        frame, so the frame is repeated to fill the tubelet -- the tubelet conv sees zero
        motion, and the temporal half of V-JEPA's pretraining is idle here.
        Two things make that acceptable, and one thing does not follow from it:
          - There is no alternative with these weights. The released 2.1 ViT-B is built
            as a video model (num_frames=64, PatchEmbed3D); upstream's 4-D image branch
            needs num_frames=1 and a 2D PatchEmbed, which the pretrained Conv3d weights
            cannot load into. Feeding [B,3,H,W] directly raises ValueError.
          - Repeating is exactly the standard 3D->2D kernel deflation: the same frame
            twice through the Conv3d equals a Conv2d whose kernel is the temporal sum
            (measured max abs diff 1.9e-05). So nothing is lost versus a "proper"
            image mode.
          - It does NOT follow that omega-0 does this. Their paper says only that the
            policy "receives only a single current image" and calls it a frozen V-JEPA
            2.1 "image encoder"; the tubelet/temporal handling is never stated. They do
            apply 2D RoPE to the current-observation tokens (3D RoPE is reserved for the
            future-video queries), whereas this class keeps 3D RoPE with a single
            temporal group -- a real implementation difference, not a match.
      * Like DINOv2BackBone, this runs under no_grad and is frozen, whereas the resnet18
        baseline is trained end-to-end with lr_backbone. That confound is inherited from
        the existing design, not introduced here.
    """
    def __init__(self, model_name: str = 'vjepa2_1_vitb',
                 image_feature_strategy: str = 'ACT_linear') -> None:
        super().__init__()
        # import here so the resnet/dinov2 paths never pay for it
        from .detr_vae import FrozenPatchTargetEncoder
        assert image_feature_strategy == 'ACT_linear', (
            f"VJEPABackBone only implements ACT_linear, got {image_feature_strategy}")
        self.image_feature_strategy = image_feature_strategy
        # target_resolution_hw snaps 224x308 -> 224x304 so patch 16 divides the grid
        self.encoder = FrozenPatchTargetEncoder(model_name, target_resolution_hw=[224, 304])
        self.num_channels = self.encoder.embed_dim
        self.model_stride = self.encoder.patch_size
        self.grid_hw = (224 // self.model_stride, 304 // self.model_stride)   # 14 x 19

    @torch.no_grad()
    def forward(self, tensor):
        B = tensor.shape[0]
        # [B,3,H,W] -> [B,K,3,H,W] by repeating the frame across the tubelet
        clip = tensor.unsqueeze(1).expand(-1, self.encoder.tubelet_size, -1, -1, -1)
        tokens = self.encoder(clip)                       # [B, 1*14*19, D]
        gh, gw = self.grid_hw
        assert tokens.shape[1] == gh * gw, (
            f"expected {gh * gw} tokens, got {tokens.shape[1]}")
        od = OrderedDict()
        od["0"] = tokens.reshape(B, gh, gw, self.num_channels).permute(0, 3, 1, 2)
        return od


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    # def forward (self, tensor):
    #     xs = self[0](tensor)
    #     pos = self[1](xs)
    #     return xs, pos
    
    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.dtype))

        return out, pos


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    if args.backbone.startswith('dinov2'):
        backbone = DINOv2BackBone(args.backbone, args.image_feature_strategy)
    elif args.backbone.startswith('vjepa'):
        backbone = VJEPABackBone(args.backbone, args.image_feature_strategy)
    else:
        assert args.backbone in ['resnet18', 'resnet34']
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, dilation=False)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
