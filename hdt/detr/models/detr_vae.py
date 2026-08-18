# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torch.autograd import Variable
from .backbone import build_backbone
from .transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
import os
import time

import IPython
e = IPython.embed


class FrozenPatchTargetEncoder(nn.Module):
    """
    Frozen visual encoder used ONLY to produce the Future-DINO query/target tokens.

    This must NOT be the trunk backbone: if the encoder that produces the target is
    also being trained, the target moves with the predictor and the world loss can be
    driven to zero by collapsing the features instead of by predicting the future.
    The trunk backbone (resnet18 or whatever) stays trainable as before -- only the
    *target* space is frozen.

    Deliberately NOT registered in the parent module tree (the parent holds it in a
    plain list) so that:
      - its ~22M params never enter the optimizer or the checkpoint,
      - DDP does not have to bucket/allreduce them.
    Device placement is therefore done lazily in the parent's forward().
    """

    DINOV2_EMBED_DIM = {
        'dinov2_vits14': 384,
        'dinov2_vitb14': 768,
        'dinov2_vitl14': 1024,
        'dinov2_vitg14': 1536,
    }

    # Frozen V-JEPA 2 / 2.1 *video* teachers, keyed by our short name ->
    # (torch.hub entrypoint, checkpoint file under dl.fbaipublicfiles.com/vjepa2,
    #  key inside that checkpoint, embed dim).
    #
    # Why bother: measured on our own data, DINOv2-S patch tokens are a near-useless
    # target once the camera moves -- same-state cosine 0.517 vs a different-state
    # floor of 0.480. DINOv2 is image-only self-supervision so its tokens are pinned
    # to the image grid. V-JEPA 2 is trained on video, and the 2.1 release explicitly
    # optimizes *temporally consistent dense* features, which is the property a world
    # target actually needs. omega-0 uses V-JEPA features for the same reason.
    VJEPA2_MODELS = {
        # name            hub entrypoint             checkpoint file                 key             dim
        'vjepa2_1_vitb': ('vjepa2_1_vit_base_384',  'vjepa2_1_vitb_dist_vitG_384', 'ema_encoder',     768),
        'vjepa2_1_vitl': ('vjepa2_1_vit_large_384', 'vjepa2_1_vitl_dist_vitG_384', 'ema_encoder',    1024),
        'vjepa2_vitl':   ('vjepa2_vit_large',       'vitl',                        'target_encoder', 1024),
        'vjepa2_vith':   ('vjepa2_vit_huge',        'vith',                        'target_encoder', 1280),
    }
    # Pinned commit, and we build with pretrained=False on purpose: upstream main has
    # VJEPA_BASE_URL hardwired to http://localhost:8300 ("for testing"), so their
    # pretrained=True path downloads nothing. We fetch the real weights ourselves.
    VJEPA2_HUB_REF = 'facebookresearch/vjepa2:204698b45b3712590f06245fbfba32d3be539812'
    VJEPA2_WEIGHT_URL = 'https://dl.fbaipublicfiles.com/vjepa2/{}.pt'

    # Frozen Wan *VAE* teachers. Note what these are NOT: Wan is a 3D causal video
    # autoencoder trained to RECONSTRUCT pixels, not a DINO-style discriminative
    # encoder. Its latent keeps everything needed to redraw the frame (texture,
    # lighting, exact colour), which is precisely the stuff DINO throws away.
    # That is why it is worth testing as a SECOND target species rather than a
    # replacement: ST-WAM regresses both, and in their ablation the VAE half is the
    # load-bearing one (LIBERO-Plus 72.8 -> 63.5 without DINO, but -> 39.7 without
    # the VAE). omega-0's future target is a frozen Wan latent too.
    #   name           -> (local dir, spatial compression, latent channels)
    # Wan2.2 TI2V-5B: 16x spatial => a 224x304 frame becomes 14x19 = 266 tokens of
    # dim 48, the same order as DINOv2-S's 352, so the head sees a comparable budget.
    # The 2.8GB weights are not fetched on demand, so on a new machine either put them
    # at this default or point WAN_VAE_DIR at wherever they landed.
    WAN_VAE_MODELS = {
        'wan22_vae': os.environ.get(
            'WAN_VAE_DIR',
            os.path.expanduser('~/.cache/huggingface/wan22_vae')),
    }
    # The VAE encoder is 2.8GB of fp32 weights and is by far the most expensive thing
    # in the step (0.51s/step for 128 frames at 224x304 vs 0.35s for the whole rest of
    # training). bf16 costs 0.016 max abs latent error against fp32 -- irrelevant for a
    # regression target -- and halves both time and memory, so the teacher is pinned to
    # bf16 regardless of the trunk dtype.
    WAN_DTYPE = torch.bfloat16
    # Frames per encode call. 128 frames at once peaks at 12.6GiB; 32 at a time peaks
    # at 4.0GiB with no measurable slowdown.
    WAN_ENCODE_CHUNK = 32
    # ACTPolicy.transform hands us ImageNet-normalized pixels; the VAE wants [-1, 1].
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)

    def __init__(self, name: str = 'dinov2_vits14', target_resolution_hw=None):
        super().__init__()
        self.name = name
        # Optional [H, W] the teacher's input is resized to. Must be a multiple of the
        # patch size. Only needed for the video teachers, whose patch size (16) does
        # not divide the 224x308 grid ACTPolicy.transform produces for the trunk.
        self.target_resolution_hw = tuple(target_resolution_hw) if target_resolution_hw else None
        self.tubelet_size = 1
        if name.startswith('dinov2'):
            if name not in self.DINOV2_EMBED_DIM:
                raise ValueError(f"Unsupported Future-DINO target encoder {name}")
            self.body = torch.hub.load('facebookresearch/dinov2', name)
            self.kind = 'dinov2'
            self.patch_size = 14
            self.embed_dim = self.DINOV2_EMBED_DIM[name]
        elif name in self.VJEPA2_MODELS:
            entry, ckpt_file, ckpt_key, embed_dim = self.VJEPA2_MODELS[name]
            # The hub entrypoints return (encoder, predictor); we only ever want the
            # encoder -- there is no rollout here, just a target space.
            encoder, _predictor = torch.hub.load(
                self.VJEPA2_HUB_REF, entry, pretrained=False, trust_repo=True)
            del _predictor
            state = torch.hub.load_state_dict_from_url(
                self.VJEPA2_WEIGHT_URL.format(ckpt_file), map_location='cpu')
            if ckpt_key not in state:
                raise KeyError(f"{ckpt_file}.pt has no '{ckpt_key}' key; "
                               f"available: {sorted(state.keys())[:10]}")
            enc_state = {k.replace('module.', '').replace('backbone.', ''): v
                         for k, v in state[ckpt_key].items()}
            # strict=False: the released checkpoints still carry a sincos pos_embed
            # that these RoPE encoders never use. Anything ELSE missing is a real
            # mismatch and must not be silently tolerated.
            missing, unexpected = encoder.load_state_dict(enc_state, strict=False)
            real_missing = [k for k in missing if 'pos_embed' not in k]
            if real_missing:
                raise RuntimeError(
                    f"V-JEPA2 teacher {name}: {len(real_missing)} weights failed to load, "
                    f"e.g. {real_missing[:5]}")
            print(f"V-JEPA2 teacher {name}: loaded {ckpt_file}.pt['{ckpt_key}'] "
                  f"({len(unexpected)} unused keys, {len(missing)} pos_embed skipped)")
            del state, enc_state
            self.body = encoder
            self.kind = 'vjepa2'
            self.patch_size = 16
            # Tubelet: the patch embed fuses this many consecutive frames into one
            # token group, so clip_frames must be a multiple of it.
            self.tubelet_size = int(getattr(encoder, 'tubelet_size', 2))
            self.embed_dim = embed_dim
        elif name in self.WAN_VAE_MODELS:
            from diffusers import AutoencoderKLWan
            vae = AutoencoderKLWan.from_pretrained(self.WAN_VAE_MODELS[name],
                                                   torch_dtype=self.WAN_DTYPE)
            # Only the encoder half is ever used; dropping the decoder saves ~1.4GB.
            if hasattr(vae, 'decoder'):
                del vae.decoder
            self.body = vae
            self.kind = 'wan_vae'
            self.patch_size = int(vae.config.scale_factor_spatial)   # 16
            self.embed_dim = int(vae.config.z_dim)                   # 48
            # Per-channel standardization published with the checkpoint. Without it the
            # 48 latent channels have wildly different scales (std 0.35 .. 1.69) and the
            # Huber term would be dominated by a handful of them.
            self.register_buffer('wan_latents_mean',
                                 torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1))
            self.register_buffer('wan_latents_std',
                                 torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1))
            self.register_buffer('imagenet_mean',
                                 torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer('imagenet_std',
                                 torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))
        elif name in ('resnet18', 'resnet34'):
            # Fallback for offline machines. Weaker targets than DINOv2 but still a
            # *frozen* teacher, which is the property that actually matters.
            m = getattr(torchvision.models, name)(pretrained=True)
            self.body = nn.Sequential(*(list(m.children())[:-2]))
            self.kind = 'resnet'
            self.patch_size = 32
            self.embed_dim = 512
        else:
            raise ValueError(f"Unsupported Future-DINO target encoder {name}")

        self.body.eval()
        for p in self.body.parameters():
            p.requires_grad_(False)

    @property
    def expects_clip(self):
        """True for video teachers, which take [N, K, 3, H, W] instead of [N, 3, H, W]."""
        return self.kind == 'vjepa2'

    @property
    def fixed_dtype(self):
        """dtype this teacher must stay in, or None to follow the trunk's dtype."""
        return self.WAN_DTYPE if self.kind == 'wan_vae' else None

    @torch.no_grad()
    def _forward_wan_vae(self, x):
        """[N, 3, H, W] ImageNet-normalized -> [N, h*w, z_dim] standardized latents."""
        # ImageNet-normalized -> [0,1] -> [-1,1], the range the VAE was trained on.
        x = x * self.imagenet_std.to(x.dtype) + self.imagenet_mean.to(x.dtype)
        x = x.clamp(0.0, 1.0) * 2.0 - 1.0
        H, W = x.shape[-2:]
        tH, tW = self._snap_to_patch_grid(H, W)     # 224x308 -> 224x304 (16 | both)
        if (tH, tW) != (H, W):
            x = F.interpolate(x, size=(tH, tW), mode='bilinear', align_corners=False)
        out_dtype = x.dtype
        x = x.to(self.WAN_DTYPE).unsqueeze(2)       # [N, 3, T=1, H, W]
        # A causal 3D VAE maps T=1 to T'=1, so a single frame comes back as one latent
        # frame -- the 4x temporal compression never engages here.
        lats = []
        for i in range(0, x.shape[0], self.WAN_ENCODE_CHUNK):
            lats.append(self.body.encode(x[i:i + self.WAN_ENCODE_CHUNK]).latent_dist.mode())
        lat = torch.cat(lats, dim=0).float()        # [N, z, 1, h, w]
        assert lat.shape[2] == 1, f"expected 1 latent frame, got {lat.shape[2]}"
        lat = (lat - self.wan_latents_mean.float()) / self.wan_latents_std.float()
        return lat.squeeze(2).flatten(2).transpose(1, 2).to(out_dtype)   # [N, h*w, z]

    def _snap_to_patch_grid(self, H, W):
        if self.target_resolution_hw:
            return self.target_resolution_hw
        p = self.patch_size
        return (max(p, round(H / p) * p), max(p, round(W / p) * p))

    @torch.no_grad()
    def forward(self, x):
        """
        x: [N, 3, H, W] for image teachers, or [N, K, 3, H, W] for the video teachers.
           Already ImageNet-normalized by ACTPolicy.transform (V-JEPA 2 uses the same
           ImageNet mean/std, so no separate preprocessing is needed).
        returns: [N, P, D] patch tokens (no CLS / register tokens). For a video teacher
           P = (K / tubelet) * (H/16) * (W/16), i.e. spatio-temporal tokens.
        """
        self.body.eval()
        if self.kind == 'wan_vae':
            assert x.ndim == 4, (
                f"Wan VAE teacher takes a frame [N,3,H,W], got {tuple(x.shape)}")
            return self._forward_wan_vae(x)
        if self.kind == 'vjepa2':
            assert x.ndim == 5, (
                f"video teacher {self.name} needs a clip [N,K,C,H,W], got {tuple(x.shape)}")
            N, K, C, H, W = x.shape
            assert K % self.tubelet_size == 0, (
                f"future_dino.clip_frames={K} must be a multiple of the tubelet size "
                f"{self.tubelet_size}")
            tH, tW = self._snap_to_patch_grid(H, W)
            if (tH, tW) != (H, W):
                x = F.interpolate(x.reshape(N * K, C, H, W), size=(tH, tW),
                                  mode='bilinear', align_corners=False)
                x = x.reshape(N, K, C, tH, tW)
            # These encoders want [N, C, T, H, W]. They use RoPE with
            # handle_nonsquare_inputs=True, so T/H/W need not match the 64x384x384
            # pretraining shape -- only the patch/tubelet divisibility above matters.
            return self.body(x.permute(0, 2, 1, 3, 4))
        if self.kind == 'dinov2':
            p = self.patch_size
            H, W = x.shape[-2:]
            if H % p or W % p:
                # ACTPolicy.transform already resizes to a multiple of 14; this only
                # fires when the encoder is called directly (tests, probes) and keeps
                # current and future frames on an identical grid either way.
                x = F.interpolate(x, size=(max(p, round(H / p) * p), max(p, round(W / p) * p)),
                                  mode='bilinear', align_corners=False)
            # forward_features()['x_norm_patchtokens'] already drops CLS and the
            # register tokens, i.e. doc S2.2 remove_cls_and_register_tokens.
            return self.body.forward_features(x)['x_norm_patchtokens']
        feat = self.body(x)                      # [N, D, h, w]
        return feat.flatten(2).transpose(1, 2)   # [N, P, D]


class FutureDINOHead(nn.Module):
    """
    Future-DINO Head for predicting future visual latent features.
    Uses Transformer Decoder to predict future DINO patch tokens from
    current DINO tokens (as queries) and shared features (as memory).
    """
    def __init__(
        self,
        dino_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_patches: int = 150,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dino_dim = dino_dim
        self.hidden_dim = hidden_dim
        # `num_patches` is an upper bound: the table is sliced to the actual patch
        # count at forward time, so it only has to be >= P (asserted below).
        self.num_patches = num_patches

        # Project current DINO tokens to hidden_dim
        self.query_proj = nn.Linear(dino_dim, hidden_dim)

        # Position embedding for patches
        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )

        # Horizon embedding (single horizon for now)
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        # Output projection back to DINO dimension
        self.output_proj = nn.Linear(hidden_dim, dino_dim)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.horizon_embedding, std=0.02)

    def forward(
        self,
        current_dino_tokens: torch.Tensor,
        hat_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            current_dino_tokens: [B, P, D_dino] - Current frame DINO patch features
            hat_memory: [B, N, hidden_size] - Shared features from transformer

        Returns:
            pred_future_dino: [B, P, D_dino] - Predicted future DINO patch features
        """
        # Project queries
        queries = self.query_proj(current_dino_tokens)  # [B, P, hidden_dim]

        assert queries.shape[1] <= self.num_patches, (
            f"Future-DINO head got {queries.shape[1]} patches but its position table only "
            f"holds {self.num_patches}; raise future_dino.num_patches in the model config.")

        # Add position and horizon embeddings
        queries = queries + self.position_embedding[:, :queries.shape[1], :]
        queries = queries + self.horizon_embedding

        # Transformer Decoder
        future_tokens = self.decoder(
            tgt=queries,
            memory=hat_memory,
        )

        # Project back to DINO dimension
        return self.output_proj(future_tokens)


class FutureDINOLoss(nn.Module):
    """
    Loss function for Future-DINO Head.
    Combines cosine similarity loss and Huber loss.
    """
    def __init__(self, huber_weight: float = 0.1, normalize_target: bool = True,
                 target_norm: str = 'l2'):
        super().__init__()
        self.huber_weight = huber_weight
        self.normalize_target = normalize_target
        # 'l2'        -- unit-norm each token (the original behaviour)
        # 'layernorm' -- zero-mean/unit-var each token instead. EgoWAM and RAE both
        #                LayerNorm the DINO target rather than L2-normalizing it:
        #                L2 throws away the token norm, which carries how *salient*
        #                a patch is, and leaves every target on one hypersphere so
        #                the cosine term dominates. Kept off by default so existing
        #                runs are bit-identical.
        assert target_norm in ('l2', 'layernorm')
        self.target_norm = target_norm

    def forward(
        self,
        predicted: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> dict:
        """
        Args:
            predicted: [B, P, D] - Predicted future DINO features
            target: [B, P, D] - Target future DINO features
            mask: [B, P] - Optional mask for valid patches (1=valid, 0=invalid)

        Returns:
            dict with 'cosine_loss', 'huber_loss', 'loss' keys
        """
        # Hard stop-gradient on the teacher. The encoder is already frozen and run
        # under no_grad, this is belt-and-braces so the loss can never train a target.
        target = target.detach()
        if self.normalize_target:
            if self.target_norm == 'layernorm':
                # Per-token LayerNorm (no affine): keeps the relative magnitude
                # structure inside a token that L2 discards.
                target = F.layer_norm(target, (target.shape[-1],))
            else:
                # doc S2.2 tokenwise_normalize: unit-norm each patch token so the Huber
                # term is scale-free and comparable across patches/images.
                target = F.normalize(target, dim=-1)

        # Cosine similarity loss
        cos_sim = F.cosine_similarity(predicted, target, dim=-1)  # [B, P]
        cosine_loss = (1.0 - cos_sim)  # [B, P]

        # Huber loss
        huber_loss = F.smooth_l1_loss(predicted, target, reduction='none')  # [B, P, D]
        huber_loss = huber_loss.mean(dim=-1)  # [B, P]

        if mask is not None:
            # Apply mask
            mask = mask.float()  # [B, P]
            cosine_loss = (cosine_loss * mask).sum() / (mask.sum() + 1e-8)
            huber_loss = (huber_loss * mask).sum() / (mask.sum() + 1e-8)
        else:
            cosine_loss = cosine_loss.mean()
            huber_loss = huber_loss.mean()

        total_loss = cosine_loss + self.huber_weight * huber_loss

        return {
            'cosine_loss': cosine_loss,
            'huber_loss': huber_loss,
            'loss': total_loss,
        }


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries, camera_names, image_feature_strategy, use_language_conditioning, future_dino_config=None, future_vae_config=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            future_dino_config: config dict for Future-DINO Head (optional)
            future_vae_config: config dict for a SECOND future-target head whose teacher
                is a frozen Wan VAE instead of DINO (optional, ST-WAM style). It shares
                the dataloader's future frame -- and therefore the horizon -- with the
                DINO head, so future_dino must be enabled for it to have a frame to
                look at (set future_dino.weight = 0 for a VAE-only arm, the same way
                ab0/ab1 keep the head attached but silent).
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.use_language_conditioning = use_language_conditioning
        self.image_feature_strategy = image_feature_strategy
        assert self.image_feature_strategy in ['ACT_linear', 'linear', 'linear4', 'dpt']

        if self.image_feature_strategy == 'ACT_linear':
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        elif self.image_feature_strategy == 'linear':
            # input proj goes up to 2x for cls token
            self.input_proj = nn.Conv2d(backbones[0].num_channels * 2, hidden_dim, kernel_size=1)
        elif self.image_feature_strategy == 'linear4':
            # input proj goes up to 2x for cls token
            self.input_proj = nn.Conv2d(backbones[0].num_channels * 2 * 4, hidden_dim, kernel_size=1)
        else:
            raise ValueError(f"Strategy {self.image_feature_strategy} not supported")
        
        self.backbones = nn.ModuleList(backbones)
        
        if self.use_language_conditioning:
            # TODO(roger): temporary way to handle language conditioning
            # work only with language generated by generate_easy_language.py
            self.input_proj_robot_state = nn.Linear(state_dim + 4096 * 2, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

        # Future-DINO Head (optional auxiliary head)
        self.use_future_dino_head = False
        self.future_dino_head = None
        self.future_dino_loss_fn = None
        self.future_dino_weight = 0.0
        self.future_dino_warmup_steps = 0
        self._training_step = 0

        # Held in a plain list so nn.Module never registers it: keeps the frozen
        # teacher out of state_dict(), out of the optimizer and out of DDP.
        self._future_dino_target_encoder = []
        self.future_dino_ablation = 'none'
        # 1 = single-frame teacher (DINOv2 / resnet), the original behaviour.
        # >1 = video teacher (V-JEPA 2): query and target are both K-frame clips.
        self.future_dino_clip_frames = 1

        if future_dino_config and future_dino_config.get('enabled', False):
            target_encoder = FrozenPatchTargetEncoder(
                future_dino_config.get('target_encoder', 'dinov2_vits14'),
                target_resolution_hw=future_dino_config.get('target_resolution_hw'),
            )
            self._future_dino_target_encoder.append(target_encoder)
            self.future_dino_clip_frames = int(future_dino_config.get('clip_frames', 1))
            if target_encoder.expects_clip:
                assert self.future_dino_clip_frames >= target_encoder.tubelet_size, (
                    f"{target_encoder.name} is a video teacher: set future_dino.clip_frames "
                    f">= {target_encoder.tubelet_size} (got {self.future_dino_clip_frames})")
            else:
                assert self.future_dino_clip_frames == 1, (
                    f"{target_encoder.name} is an image teacher: future_dino.clip_frames "
                    f"must be 1 (got {self.future_dino_clip_frames})")
            # Queries and targets both come from the frozen teacher, so they live in
            # the same space and the head only has to model the *change*.
            dino_dim = target_encoder.embed_dim
            self.use_future_dino_head = True
            self.future_dino_head = FutureDINOHead(
                dino_dim=dino_dim,
                hidden_dim=hidden_dim,
                num_layers=future_dino_config.get('num_layers', 4),
                num_heads=future_dino_config.get('num_heads', 8),
                num_patches=future_dino_config.get('num_patches', 1024),
                dropout=future_dino_config.get('dropout', 0.1),
            )
            self.future_dino_loss_fn = FutureDINOLoss(
                huber_weight=future_dino_config.get('huber_weight', 0.1),
                normalize_target=future_dino_config.get('normalize_target', True),
                target_norm=future_dino_config.get('target_norm', 'l2'),
            )
            self.future_dino_weight = future_dino_config.get('weight', 0.3)
            self.future_dino_warmup_steps = future_dino_config.get('warmup_steps', 1000)
            # doc S9 ablations: 'none' | 'shuffled' (break the temporal pairing) |
            # 'current' (predict the current frame, i.e. plain reconstruction).
            self.future_dino_ablation = future_dino_config.get('ablation', 'none')
            assert self.future_dino_ablation in ('none', 'shuffled', 'current')
            print(f"Future-DINO Head enabled: target_encoder={target_encoder.name} "
                  f"(dim={dino_dim}, frozen), weight={self.future_dino_weight}, "
                  f"warmup_steps={self.future_dino_warmup_steps}, "
                  f"clip_frames={self.future_dino_clip_frames}, "
                  f"ablation={self.future_dino_ablation}")

        # ---- Second future-target species: frozen Wan VAE (ST-WAM style) ----------
        # Entirely parallel to the DINO head above and deliberately kept as its own
        # block rather than folded into it: the DINO path has to stay bit-identical so
        # the ab0..ab7 runs remain the comparison baseline.
        self.use_future_vae_head = False
        self.future_vae_head = None
        self.future_vae_loss_fn = None
        self.future_vae_weight = 0.0
        self.future_vae_ablation = 'none'
        self._future_vae_target_encoder = []

        if future_vae_config and future_vae_config.get('enabled', False):
            assert self.use_future_dino_head, (
                "future_vae rides on the future frame the Future-DINO path asks the "
                "dataloader for; enable future_dino (weight 0 is fine) as well.")
            assert self.future_dino_clip_frames == 1, (
                "future_vae is a single-frame teacher; it cannot share a clip-mode "
                f"future_image (clip_frames={self.future_dino_clip_frames}).")
            vae_encoder = FrozenPatchTargetEncoder(
                future_vae_config.get('target_encoder', 'wan22_vae'),
                target_resolution_hw=future_vae_config.get('target_resolution_hw'),
            )
            self._future_vae_target_encoder.append(vae_encoder)
            self.use_future_vae_head = True
            self.future_vae_head = FutureDINOHead(
                dino_dim=vae_encoder.embed_dim,
                hidden_dim=hidden_dim,
                num_layers=future_vae_config.get('num_layers', 4),
                num_heads=future_vae_config.get('num_heads', 8),
                num_patches=future_vae_config.get('num_patches', 1024),
                dropout=future_vae_config.get('dropout', 0.1),
            )
            # Different default from the DINO head on purpose. The VAE latent is
            # already per-channel standardized by latents_mean/std, so re-normalizing
            # each 48-d token would throw away the one thing a reconstruction latent
            # carries that DINO does not: absolute magnitude. And because the target is
            # a reconstruction code rather than a direction in a discriminative space,
            # the regression (Huber) term is the meaningful one here, not cosine.
            self.future_vae_loss_fn = FutureDINOLoss(
                huber_weight=future_vae_config.get('huber_weight', 1.0),
                normalize_target=future_vae_config.get('normalize_target', False),
                target_norm=future_vae_config.get('target_norm', 'l2'),
            )
            self.future_vae_weight = future_vae_config.get('weight', 1.0)
            self.future_vae_ablation = future_vae_config.get('ablation', 'none')
            assert self.future_vae_ablation in ('none', 'shuffled', 'current')
            print(f"Future-VAE Head enabled: target_encoder={vae_encoder.name} "
                  f"(dim={vae_encoder.embed_dim}, frozen, {vae_encoder.WAN_DTYPE}), "
                  f"weight={self.future_vae_weight}, "
                  f"huber_weight={self.future_vae_loss_fn.huber_weight}, "
                  f"normalize_target={self.future_vae_loss_fn.normalize_target}, "
                  f"ablation={self.future_vae_ablation}")

    def get_current_step(self):
        """Get current training step."""
        return self._training_step

    def set_training_step(self, step):
        """Set current training step."""
        self._training_step = step
    
    def _future_vae_forward(self, image, future_image, hs, conditioning_dict):
        """Wan-VAE twin of the Future-DINO block in forward().

        Same shape as the K==1 DINO path, but its own frozen teacher, own head and own
        weight, so the two species can be traded off independently (ST-WAM runs the VAE
        at 1.0 and DINO at 0.02). Returns the loss dict, or None if disabled.
        """
        B_img, num_cam = image.shape[0], image.shape[1]
        C_img, H_img, W_img = image.shape[2:]
        assert future_image.shape == image.shape, (
            f"future_image {tuple(future_image.shape)} must match image {tuple(image.shape)}")
        enc_input = torch.cat([image, future_image], dim=0).reshape(
            2 * B_img * num_cam, C_img, H_img, W_img)

        vae_encoder = self._future_vae_target_encoder[0]
        # Lazy placement, same reason as the DINO teacher: it is held in a plain list so
        # the parent's .cuda() never reaches it. Device only -- the VAE stays in bf16.
        enc_param = next(vae_encoder.parameters())
        if enc_param.device != image.device:
            vae_encoder.to(device=image.device)

        with torch.no_grad():
            tokens = vae_encoder(enc_input)                 # [2*B*cam, P_cam, z]
            P_cam, D_vae = tokens.shape[1], tokens.shape[2]
            tokens = tokens.reshape(2 * B_img, num_cam * P_cam, D_vae)
            current_vae_tokens, future_target = tokens[:B_img], tokens[B_img:]

            if self.future_vae_ablation == 'shuffled':
                future_target = future_target[torch.randperm(B_img, device=future_target.device)]
            elif self.future_vae_ablation == 'current':
                future_target = current_vae_tokens

        pred_future_vae = self.future_vae_head(
            current_dino_tokens=current_vae_tokens,
            hat_memory=hs,
        )
        assert pred_future_vae.shape == future_target.shape, (
            f"Future-VAE shape mismatch: pred {tuple(pred_future_vae.shape)} vs "
            f"target {tuple(future_target.shape)}")

        patch_mask = None
        if conditioning_dict is not None and conditioning_dict.get('future_valid') is not None:
            future_valid = conditioning_dict['future_valid'].to(pred_future_vae.dtype)
            patch_mask = future_valid.reshape(B_img, 1).expand(-1, pred_future_vae.shape[1])

        loss_dict = self.future_vae_loss_fn(
            predicted=pred_future_vae,
            target=future_target,
            mask=patch_mask,
        )
        # No warmup knob here: the DINO measurement (world gradient ~1% of the
        # trajectory gradient on the shared trunk) is what retired warmup, and this
        # head attaches to the same trunk through the same memory.
        loss_dict['effective_weight'] = self.future_vae_weight
        return loss_dict

    def get_features_and_pos(self, image):
        """
        image (B, num_cam, C, H, W): image observation
        strategy (str): 'ACT_linear', 'linear', 'linear4', 'dpt'.
            - 'ACT_linear': linear projection of image features. Equivalent to implementation in original ACT.
            - 'linear': similar to ACT_linear but with CLS token concatenated along the channel dimension.
                        Same as linear in Dinov2 depth estimation.
            - 'linear4': Similar to linear, but uses 4 layers of features instead of 1.
            - 'dpt': similar to the DPT decoder
        """
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        B, num_cam, C, H, W = image.shape
        featuress, poss = self.backbones[0](image.reshape(B*num_cam, C, H, W))

        if self.image_feature_strategy in ['ACT_linear', 'linear', 'linear4']:
            featuress = featuress[-1]
            output_B, output_C, output_H, output_W = featuress.shape
            assert output_B == B*num_cam
            pos = poss[-1]
            featuress = featuress.view(B, num_cam, output_C, output_H, output_W) # take the last layer feature

            for cam_id, cam_name in enumerate(self.camera_names):
                features = featuress[:, cam_id]
                projected_features = self.input_proj(features)
                all_cam_features.append(projected_features)
                all_cam_pos.append(pos/2+ cam_id - 0.5)
        else:
            raise ValueError(f"Strategy {self.image_feature_strategy} not supported")
        
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)

        return src, pos

    def forward(self, qpos, image, env_state, actions=None, is_pad=None, conditioning_dict=None, future_image=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        future_image: batch, num_cam, channel, height, width (optional, for Future-DINO)
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        assert self.backbones is not None
        src, pos = self.get_features_and_pos(image)

        # proprioception features
        if self.use_language_conditioning:
            language_embedding = conditioning_dict['language_embeddings'].flatten(start_dim=1)
            state_input = torch.cat([qpos, language_embedding], axis=1)
            proprio_input = self.input_proj_robot_state(state_input)
        else:
            proprio_input = self.input_proj_robot_state(qpos)

        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)

        # Future-DINO loss computation (optional)
        future_dino_loss_dict = None
        if self.use_future_dino_head and future_image is not None and self.future_dino_head is not None:
            B_img, num_cam = image.shape[0], image.shape[1]
            K = self.future_dino_clip_frames
            if K > 1:
                # Video-teacher mode. The dataloader packs BOTH clips into one tensor,
                # [B, 2K, num_cam, C, H, W]: the first K frames end at t (they become
                # the head's queries), the last K end at t+H (the target). Query and
                # target are the same kind of object -- a short clip from the same
                # frozen teacher -- so the head still only has to model the change.
                assert future_image.ndim == 6 and future_image.shape[1] == 2 * K, (
                    f"expected future_image [B,{2 * K},num_cam,C,H,W] for clip_frames={K}, "
                    f"got {tuple(future_image.shape)}")
                C_img, H_img, W_img = future_image.shape[-3:]

                def _as_clips(clip):   # [B,K,num_cam,C,H,W] -> [B*num_cam,K,C,H,W]
                    return clip.permute(0, 2, 1, 3, 4, 5).reshape(
                        B_img * num_cam, K, C_img, H_img, W_img)

                # index along dim0 is (sample * num_cam + cam), current block then
                # future block -- the same layout the K==1 path produces below.
                enc_input = torch.cat([_as_clips(future_image[:, :K]),
                                       _as_clips(future_image[:, K:])], dim=0)
            else:
                assert future_image.shape == image.shape, (
                    f"future_image {tuple(future_image.shape)} must match image {tuple(image.shape)}")
                C_img, H_img, W_img = image.shape[2:]
                enc_input = torch.cat([image, future_image], dim=0).reshape(
                    2 * B_img * num_cam, C_img, H_img, W_img)

            target_encoder = self._future_dino_target_encoder[0]
            # Lazy device/dtype placement: the encoder is intentionally not part of the
            # module tree, so .cuda()/.to() on the parent never reaches it.
            enc_param = next(target_encoder.parameters())
            if enc_param.device != image.device or enc_param.dtype != image.dtype:
                target_encoder.to(device=image.device, dtype=image.dtype)

            # One frozen pass over current+future together. Both query and target
            # tokens come from the SAME frozen teacher, so no gradient can reach it
            # and the target space cannot collapse.
            with torch.no_grad():
                tokens = target_encoder(enc_input)
                P_cam, D_dino = tokens.shape[1], tokens.shape[2]
                # fold cameras into the patch axis, mirroring get_features_and_pos()
                tokens = tokens.reshape(2 * B_img, num_cam * P_cam, D_dino)
                current_dino_tokens, future_target = tokens[:B_img], tokens[B_img:]

                if self.future_dino_ablation == 'shuffled':
                    # doc S9.2: destroy the temporal pairing. If the world loss still
                    # helps, the gain is not coming from future prediction.
                    future_target = future_target[torch.randperm(B_img, device=future_target.device)]
                elif self.future_dino_ablation == 'current':
                    # doc S9.3: degenerate to current-frame reconstruction.
                    future_target = current_dino_tokens

            # Memory MUST be the shared trunk output, not the pre-trunk `src`.
            # `hs` is exactly the feature the action head consumes, so the world
            # gradient lands on the representation used for trajectory prediction
            # (doc S3). Using `src` here only reached backbone+input_proj and left
            # the transformer trunk with literally zero world gradient.
            hat_memory = hs  # [B, num_queries, hidden_dim]

            # Predict future DINO tokens
            pred_future_dino = self.future_dino_head(
                current_dino_tokens=current_dino_tokens,
                hat_memory=hat_memory,
            )  # [B, P, D]

            assert pred_future_dino.shape == future_target.shape, (
                f"Future-DINO shape mismatch: pred {tuple(pred_future_dino.shape)} vs "
                f"target {tuple(future_target.shape)}")

            # Samples whose t+H ran past the end of the episode were clamped by the
            # dataloader; masking them out avoids teaching "the future is static".
            patch_mask = None
            if conditioning_dict is not None and conditioning_dict.get('future_valid') is not None:
                future_valid = conditioning_dict['future_valid'].to(pred_future_dino.dtype)
                patch_mask = future_valid.reshape(B_img, 1).expand(-1, pred_future_dino.shape[1])

            future_dino_loss_dict = self.future_dino_loss_fn(
                predicted=pred_future_dino,
                target=future_target,
                mask=patch_mask,
            )

            # Apply warmup. warmup_steps <= 0 means "no warmup": full weight from step 0.
            # (The old `current_step / max(warmup_steps, 1)` form silently gave weight 0 on
            # step 0 even with warmup disabled.) Default is 0 -- the measured world
            # contribution to the shared trunk gradient is ~1% of the trajectory one, so
            # there is nothing to ramp in. See act_with_future_dino.yaml.
            current_step = self.get_current_step()
            if self.future_dino_warmup_steps > 0:
                warmup_factor = min(1.0, current_step / self.future_dino_warmup_steps)
            else:
                warmup_factor = 1.0
            effective_weight = self.future_dino_weight * warmup_factor
            future_dino_loss_dict['effective_weight'] = effective_weight

            # Second target species, folded into the same dict under vae_* keys so the
            # (a_hat, is_pad_hat, [mu, logvar], dict) contract every caller relies on
            # does not change.
            if self.use_future_vae_head:
                vae_loss_dict = self._future_vae_forward(
                    image, future_image, hs, conditioning_dict)
                future_dino_loss_dict['vae_cosine_loss'] = vae_loss_dict['cosine_loss']
                future_dino_loss_dict['vae_huber_loss'] = vae_loss_dict['huber_loss']
                future_dino_loss_dict['vae_loss'] = vae_loss_dict['loss']
                future_dino_loss_dict['vae_effective_weight'] = vae_loss_dict['effective_weight']

        return a_hat, is_pad_hat, [mu, logvar], future_dino_loss_dict

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim
    action_dim = args.action_dim

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    # backbone = build_backbone(args)
    # backbones.append(backbone)
    # for _ in args.camera_names:
    backbone = build_backbone(args)
    backbones.append(backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    # Get Future-DINO config if available
    future_dino_config = getattr(args, 'future_dino_config', None)
    future_vae_config = getattr(args, 'future_vae_config', None)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        image_feature_strategy=args.image_feature_strategy,
        use_language_conditioning=args.use_language_conditioning,
        future_dino_config=future_dino_config,
        future_vae_config=future_vae_config,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

