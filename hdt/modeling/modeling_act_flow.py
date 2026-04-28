"""
ACT + Flow Matching Policy.

Architecture:
  - Context encoder: shared backbone (ResNet/DINOv2) + TransformerEncoder
    encodes image(s) + qpos into context tokens
  - FlowMatchingHead: TransformerDecoder cross-attending to context tokens
    predicts velocity field v(xt, t, context)
  - Training: Conditional Flow Matching with linear interpolation
    xt = (1-t)*x0 + t*x1,  v_target = x1 - x0
  - Inference: Euler ODE  x_{i+1} = x_i + v * dt
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

# detr imports (same working-dir convention as policy.py)
from detr.models.backbone import build_backbone
from detr.models.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
    TransformerDecoder,
    TransformerDecoderLayer,
)
from detr.main import get_args_parser
import argparse

import hdt.constants


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding + 2-layer MLP."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) float in [0, 1]
        half = self.hidden_dim // 2
        freqs = torch.exp(
            -np.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        x = t[:, None].float() * freqs[None]          # (B, half)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)  # (B, hidden_dim)
        return self.mlp(emb)


# ---------------------------------------------------------------------------
# Flow Matching Head
# ---------------------------------------------------------------------------

class FlowMatchingHead(nn.Module):
    """
    Predicts velocity field v(xt, t, context).

    Args:
        action_dim:       dimension of action space
        hidden_dim:       transformer hidden size
        num_layers:       number of TransformerDecoder layers
        nheads:           number of attention heads
        dim_feedforward:  feedforward dimension inside each layer
    """

    def __init__(
        self,
        action_dim: int,
        hidden_dim: int,
        num_layers: int,
        nheads: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.action_proj = nn.Linear(action_dim, hidden_dim)
        self.time_emb = TimestepEmbedding(hidden_dim)

        decoder_layer = TransformerDecoderLayer(
            hidden_dim, nheads, dim_feedforward, dropout,
            activation="relu", normalize_before=False,
        )
        self.decoder = TransformerDecoder(
            decoder_layer, num_layers, norm=nn.LayerNorm(hidden_dim),
            return_intermediate=False,
        )
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        xt: torch.Tensor,       # (B, T, action_dim)
        t: torch.Tensor,        # (B,)
        context: torch.Tensor,  # (B, N_ctx, hidden_dim)
    ) -> torch.Tensor:
        B, T, _ = xt.shape

        # project action tokens and inject timestep
        x = self.action_proj(xt)               # (B, T, hidden_dim)
        x = x + self.time_emb(t).unsqueeze(1)  # broadcast over T

        # TransformerDecoder convention: (seq, batch, dim)
        x   = x.permute(1, 0, 2)              # (T, B, hidden_dim)
        ctx = context.permute(1, 0, 2)        # (N_ctx, B, hidden_dim)

        out = self.decoder(x, ctx)             # (1, T, B, hidden_dim) with return_intermediate=False
        out = out[0].permute(1, 0, 2)         # (B, T, hidden_dim)

        return self.action_head(out)           # (B, T, action_dim)


# ---------------------------------------------------------------------------
# Context Encoder (backbone + TransformerEncoder)
# ---------------------------------------------------------------------------

class ACTFlowMatching(nn.Module):
    """
    Core model: context encoder + flow matching head.
    Follows the same forward-pass convention as DETRVAE.
    """

    def __init__(
        self,
        backbone,
        state_dim: int,
        action_dim: int,
        hidden_dim: int,
        enc_layers: int,
        fm_layers: int,
        nheads: int,
        dim_feedforward: int,
        camera_names: list,
        image_feature_strategy: str,
        chunk_size: int,
        num_flow_steps: int = 10,
        use_language_conditioning: bool = False,
    ):
        super().__init__()
        self.camera_names = camera_names
        self.image_feature_strategy = image_feature_strategy
        self.num_flow_steps = num_flow_steps
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.use_language_conditioning = use_language_conditioning

        # -- backbone (shared across cameras, same as DETRVAE) --
        self.backbone = backbone

        # -- image feature projection --
        if image_feature_strategy == "ACT_linear":
            self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        elif image_feature_strategy == "linear":
            self.input_proj = nn.Conv2d(backbone.num_channels * 2, hidden_dim, kernel_size=1)
        elif image_feature_strategy == "linear4":
            self.input_proj = nn.Conv2d(backbone.num_channels * 2 * 4, hidden_dim, kernel_size=1)
        else:
            raise ValueError(f"Unsupported image_feature_strategy: {image_feature_strategy}")

        # -- qpos projection --
        state_input_dim = state_dim + 4096 * 2 if use_language_conditioning else state_dim
        self.input_proj_robot_state = nn.Linear(state_input_dim, hidden_dim)

        # learned positional embedding for the single proprio token
        self.proprio_pos_embed = nn.Embedding(1, hidden_dim)

        # -- context transformer encoder --
        enc_layer = TransformerEncoderLayer(
            hidden_dim, nheads, dim_feedforward,
            dropout=0.1, activation="relu", normalize_before=False,
        )
        self.context_encoder = TransformerEncoder(enc_layer, enc_layers, norm=None)

        # -- flow matching head --
        self.flow_head = FlowMatchingHead(
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            num_layers=fm_layers,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
        )

    # ------------------------------------------------------------------
    def _get_context(self, image, qpos, conditioning_dict=None):
        """
        image:  (B, num_cam, C, H, W)
        qpos:   (B, state_dim)
        returns context: (B, 1 + H*W*num_cam, hidden_dim)
        """
        B, num_cam, C, H, W = image.shape

        # extract backbone features (same logic as DETRVAE.get_features_and_pos)
        featuress, poss = self.backbone(image.reshape(B * num_cam, C, H, W))
        featuress = featuress[-1]   # (B*num_cam, C_feat, Hf, Wf)
        pos = poss[-1]              # (1, hidden_dim, Hf, Wf) — batch-invariant

        _, output_C, output_H, output_W = featuress.shape
        featuress = featuress.view(B, num_cam, output_C, output_H, output_W)

        all_cam_features, all_cam_pos = [], []
        for cam_id in range(len(self.camera_names)):
            feat = featuress[:, cam_id]                       # (B, C_feat, Hf, Wf)
            all_cam_features.append(self.input_proj(feat))   # (B, hidden_dim, Hf, Wf)
            all_cam_pos.append(pos / 2 + cam_id - 0.5)      # (1, hidden_dim, Hf, Wf)

        # fold cameras into width dimension
        src = torch.cat(all_cam_features, dim=3)      # (B, hidden_dim, Hf, Wf*nc)
        pos_embed = torch.cat(all_cam_pos, dim=3)     # (1, hidden_dim, Hf, Wf*nc)

        # flatten spatial → sequence
        src_flat = src.flatten(2).permute(2, 0, 1)                           # (Hf*Wf*nc, B, hidden_dim)
        pos_flat = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, B, 1)    # (Hf*Wf*nc, B, hidden_dim)

        # proprio token
        if self.use_language_conditioning and conditioning_dict is not None:
            lang_emb = conditioning_dict["language_embeddings"].flatten(start_dim=1)
            state_input = torch.cat([qpos, lang_emb], dim=1)
        else:
            state_input = qpos
        proprio_token = self.input_proj_robot_state(state_input)             # (B, hidden_dim)
        proprio_token = proprio_token.unsqueeze(0)                           # (1, B, hidden_dim)
        proprio_pos = self.proprio_pos_embed.weight.unsqueeze(1).repeat(1, B, 1)  # (1, B, hidden_dim)

        # concatenate [proprio | image_tokens]
        src_seq = torch.cat([proprio_token, src_flat], dim=0)  # (1+Hf*Wf*nc, B, hidden_dim)
        pos_seq = torch.cat([proprio_pos, pos_flat], dim=0)    # (1+Hf*Wf*nc, B, hidden_dim)

        context = self.context_encoder(src_seq, pos=pos_seq)   # (1+Hf*Wf*nc, B, hidden_dim)
        return context.permute(1, 0, 2)                        # (B, 1+Hf*Wf*nc, hidden_dim)

    # ------------------------------------------------------------------
    def forward(self, qpos, image, actions=None, is_pad=None, conditioning_dict=None):
        context = self._get_context(image, qpos, conditioning_dict)

        if actions is not None:  # ---- training ----
            B, T, _ = actions.shape
            t = torch.rand(B, device=actions.device)          # t ~ U[0,1]
            x0 = torch.randn_like(actions)                    # x0 ~ N(0,I)
            t_exp = t[:, None, None]
            xt = (1 - t_exp) * x0 + t_exp * actions          # linear interpolation
            v_target = actions - x0                           # target velocity
            v_pred = self.flow_head(xt, t, context)
            return v_pred, v_target, is_pad

        else:  # ---- inference: Euler ODE ----
            B = context.shape[0]
            x = torch.randn(B, self.chunk_size, self.action_dim, device=context.device)
            dt = 1.0 / self.num_flow_steps
            for i in range(self.num_flow_steps):
                t = torch.full((B,), i * dt, device=context.device)
                v = self.flow_head(x, t, context)
                x = x + v * dt
            return x


# ---------------------------------------------------------------------------
# Policy wrapper  (same interface as ACTPolicy in policy.py)
# ---------------------------------------------------------------------------

class ACTFlowPolicy(nn.Module):

    def __init__(self, args_override: dict):
        super().__init__()
        self.model = _build_act_flow_model(args_override)
        self.chunk_size = args_override["chunk_size"]
        self.USE_EEF_LOSS = True
        self.hand_eef_weight  = args_override.get("hand_eef_weight", 2.0)
        self.head_eef_weight  = args_override.get("head_eef_weight", 0.0)  # 0 = 不额外加权

        patch_h, patch_w = 16, 22
        self.transform = v2.Compose([
            v2.Resize((patch_h * 14, patch_w * 14)),
            v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        param_dicts = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "backbone" not in n and p.requires_grad
                ]
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if "backbone" in n and p.requires_grad
                ],
                "lr": args_override["lr_backbone"],
            },
        ]
        self.optimizer = torch.optim.AdamW(
            param_dicts, lr=args_override["lr"], weight_decay=1e-4
        )

    def __call__(self, image, qpos, actions=None, is_pad=None, conditioning_dict=None):
        image = self.transform(image)

        if actions is not None:  # training
            actions = actions[:, : self.chunk_size]
            is_pad  = is_pad[:, : self.chunk_size]

            v_pred, v_target, is_pad = self.model(
                qpos, image, actions, is_pad, conditioning_dict
            )

            mask = ~is_pad.unsqueeze(-1)                     # (B, T, 1)
            all_mse = F.mse_loss(v_pred, v_target, reduction="none")  # (B, T, action_dim)
            all_mse = all_mse * mask

            loss_dict = {}
            loss_dict["fm"] = all_mse.mean()
            loss_dict["loss"] = loss_dict["fm"]

            if self.USE_EEF_LOSS:
                loss_dict["hand_eef_loss"] = (
                    all_mse[:, :, hdt.constants.OUTPUT_LEFT_EEF].mean()
                    + all_mse[:, :, hdt.constants.OUTPUT_RIGHT_EEF].mean()
                )
                loss_dict["head_eef_loss"] = all_mse[:, :, hdt.constants.OUTPUT_HEAD_EEF].mean()
                loss_dict["loss"] = (
                    loss_dict["loss"]
                    + loss_dict["hand_eef_loss"] * self.hand_eef_weight
                    + loss_dict["head_eef_loss"] * self.head_eef_weight
                )

            return loss_dict

        else:  # inference
            return self.model(qpos, image, conditioning_dict=conditioning_dict)

    def configure_optimizers(self):
        return self.optimizer


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

def _build_act_flow_model(args_override: dict) -> ACTFlowMatching:
    parser = argparse.ArgumentParser(
        "ACT Flow Matching builder", parents=[get_args_parser()]
    )
    args = parser.parse_args([])
    for k, v in args_override.items():
        setattr(args, k, v)

    backbone = build_backbone(args)

    model = ACTFlowMatching(
        backbone=backbone,
        state_dim=args_override["state_dim"],
        action_dim=args_override["action_dim"],
        hidden_dim=args_override["hidden_dim"],
        enc_layers=args_override["enc_layers"],
        fm_layers=args_override["fm_layers"],
        nheads=args_override["nheads"],
        dim_feedforward=args_override["dim_feedforward"],
        camera_names=args_override["camera_names"],
        image_feature_strategy=args_override["image_feature_strategy"],
        chunk_size=args_override["chunk_size"],
        num_flow_steps=args_override.get("num_flow_steps", 10),
        use_language_conditioning=args_override.get("use_language_conditioning", False),
    )
    model.cuda()

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ACTFlowMatching parameters: {n_params / 1e6:.2f}M")
    return model
