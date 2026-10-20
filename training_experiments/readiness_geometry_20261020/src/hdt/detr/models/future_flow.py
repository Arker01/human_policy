"""
Future 3D-flow world head (EgoWAM's third world target).

EgoWAM (arXiv 2607.08436) fixes the trunk, the action head and the data mixture and
swaps only the world-prediction target, comparing Pixel VAE / DINO features / 3D flow.
Their finding is that DINO buys OOD generalization (up to 4x on unseen objects and
scenes) while 3D flow buys *in-domain* accuracy (+20-30%), and that the two are
complementary. That split matches what we measured: our DINO head moved the OOD axes
(background/compound, 11.9% -> 9.5% relative degradation from ab1 to ab4) and left the
clean axis untouched. Every run we have ever trained sits at 40.6-44.8mm clean.

So this head exists to move the one axis nothing has moved.

Two deliberate departures from EgoWAM, both documented in the plan:

1. The target is precomputed offline by scripts/preprocess/flow_target.py rather than
   produced by a teacher encoder at train time. A dense 3D point tracker is far too
   heavy to run in the training loop, and unlike DINO/VAE features the flow target does
   not depend on any weights we are training, so caching it costs nothing in fidelity.

2. The head regresses the trajectory directly instead of using EgoWAM's flow-matching
   decoder. HAT_future_DINO_head_implementation.md S4.3 is explicit about this for the
   first version of any world head here: "不建议第一版直接照搬 diffusion / flow matching
   ... 当前目的只是验证 auxiliary world supervision 是否改善 HAT". Same call, same reason.

The queries are *learned* anchor embeddings rather than projected image tokens, which is
where this head structurally differs from FutureDINOHead. resnet18 has output stride 32,
so a 240x320 frame yields an 8x10 = 80 token map -- it cannot supply the 1200 anchors the
30x40 stride-8 grid needs. Learned queries also decouple the head from trunk resolution,
so the same head works unchanged on the V-JEPA trunk (14x19 grid) if we try it there.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_pos_embed(grid_h: int, grid_w: int, dim: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position table for a grid_h x grid_w anchor grid.

    Returns [1, grid_h * grid_w, dim], row-major so index (r, c) -> r * grid_w + c,
    which is the order scripts/preprocess/flow_target.py writes anchors in.

    Sinusoidal rather than learned because the anchors ARE a geometric grid: neighbouring
    anchors carry neighbouring 3D points, and a fixed table hands the decoder that metric
    structure for free instead of making it discover the grid from scratch.
    """
    assert dim % 4 == 0, f"2D sincos position embedding needs dim divisible by 4, got {dim}"
    omega = torch.arange(dim // 4, dtype=torch.float32) / (dim / 4.0)
    omega = 1.0 / (10000.0 ** omega)                                    # [dim/4]

    rows = torch.arange(grid_h, dtype=torch.float32)
    cols = torch.arange(grid_w, dtype=torch.float32)
    # meshgrid with indexing='ij' keeps row-major order to match the anchor layout.
    grid_r, grid_c = torch.meshgrid(rows, cols, indexing='ij')

    out = []
    for coord in (grid_r.reshape(-1), grid_c.reshape(-1)):              # [P] each
        ang = coord[:, None] * omega[None, :]                           # [P, dim/4]
        out += [torch.sin(ang), torch.cos(ang)]
    return torch.cat(out, dim=1)[None]                                  # [1, P, dim]


class FutureFlowHead(nn.Module):
    """Predicts the 3D displacement trajectory of a fixed anchor grid.

    For anchor frame t, anchor a and step k in 1..horizon, the target is where the scene
    point under anchor a has moved by frame t+k, expressed in the *camera frame at t* so
    that ego-motion is factored out (EgoWAM's desideratum D3). Static background is then
    ~0 and only genuinely moving things -- the hands and the pillow -- carry signal.

    hat_memory is [B, num_queries, hidden_dim] and num_queries is the action chunk length,
    which is 100 here, exactly the horizon EgoWAM regresses. Token k of the memory is the
    trunk's representation of chunk step k, so cross-attention lines the flow trajectory up
    with the action trajectory step for step rather than through a bottleneck.
    """

    def __init__(
        self,
        grid_hw=(30, 40),
        horizon: int = 100,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.grid_h, self.grid_w = int(grid_hw[0]), int(grid_hw[1])
        self.num_anchors = self.grid_h * self.grid_w
        self.horizon = int(horizon)
        self.hidden_dim = hidden_dim

        # Learned content per anchor, on top of the fixed geometric position table.
        self.anchor_embedding = nn.Parameter(torch.randn(1, self.num_anchors, hidden_dim) * 0.02)
        self.register_buffer(
            'position_embedding',
            build_2d_sincos_pos_embed(self.grid_h, self.grid_w, hidden_dim),
            persistent=False,
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            norm_first=True,
            dropout=dropout,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # One 3-vector per (anchor, step). Folded out of the channel dim rather than
        # decoded autoregressively -- S4.3 again, keep the first version one-step.
        self.output_proj = nn.Linear(hidden_dim, self.horizon * 3)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.anchor_embedding, std=0.02)
        # Start at zero displacement. The scene is mostly static and the target is mostly
        # near-zero metres, so a zero-init output layer starts the head at the right answer
        # for background anchors and keeps the initial world gradient small on the shared
        # trunk -- the same property that let us retire warmup for the DINO head.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, hat_memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hat_memory: [B, N, hidden_dim] - shared HAT features (N = action chunk length)

        Returns:
            pred_flow: [B, horizon, num_anchors, 3] - predicted 3D displacement in metres,
                       in the camera frame of the anchor frame
        """
        B = hat_memory.shape[0]
        queries = self.anchor_embedding + self.position_embedding                # [1, P, H]
        queries = queries.expand(B, -1, -1)                                      # [B, P, H]

        tokens = self.decoder(tgt=queries, memory=hat_memory)                    # [B, P, H]
        flow = self.output_proj(tokens)                                          # [B, P, K*3]
        flow = flow.reshape(B, self.num_anchors, self.horizon, 3)
        return flow.permute(0, 2, 1, 3).contiguous()                             # [B, K, P, 3]


class FutureFlowLoss(nn.Module):
    """Masked Huber on 3D displacement.

    Huber and not cosine: FutureDINOLoss leans on cosine because a feature target only has
    a meaningful direction, but a displacement has a meaningful *magnitude* -- "the pillow
    moved 4cm left" is the whole content of the target. Cosine would throw that away.

    Three masks multiply together:
      anchor_valid [B, P]  -- EgoWAM discards anchors whose displacement never clears a
                              movement threshold (2mm robot / 10mm human). Most of a
                              tabletop scene is static, so this is what stops the loss
                              from being dominated by predicting zeros.
      flow_valid   [B, K]  -- step k falls off the end of the episode.
      future_valid [B]     -- the sample's future frame was clamped; same flag the DINO
                              path already carries through conditioning_dict.
    """

    def __init__(self, huber_beta: float = 0.01, target_scale: float = 1.0):
        super().__init__()
        # Targets are metres. 0.01 puts the quadratic/linear knee at 1cm, which is roughly
        # where real manipulation displacement lives, so genuine motion gets the smooth
        # quadratic regime and tracker outliers get the linear one.
        self.huber_beta = huber_beta
        self.target_scale = target_scale

    def forward(self, predicted, target, anchor_valid=None, flow_valid=None, future_valid=None):
        """
        Args:
            predicted:    [B, K, P, 3]
            target:       [B, K, P, 3]
            anchor_valid: [B, P]   or None
            flow_valid:   [B, K]   or None
            future_valid: [B]      or None
        """
        assert predicted.shape == target.shape, (
            f"Future-flow shape mismatch: pred {tuple(predicted.shape)} vs "
            f"target {tuple(target.shape)}")

        if self.target_scale != 1.0:
            predicted = predicted * self.target_scale
            target = target * self.target_scale

        B, K, P, _ = predicted.shape
        mask = predicted.new_ones(B, K, P)
        if anchor_valid is not None:
            mask = mask * anchor_valid.to(mask.dtype).reshape(B, 1, P)
        if flow_valid is not None:
            mask = mask * flow_valid.to(mask.dtype).reshape(B, K, 1)
        if future_valid is not None:
            mask = mask * future_valid.to(mask.dtype).reshape(B, 1, 1)

        per_elem = F.smooth_l1_loss(predicted, target, reduction='none', beta=self.huber_beta)
        per_anchor = per_elem.mean(dim=-1)                                       # [B, K, P]

        denom = mask.sum().clamp(min=1.0)
        loss = (per_anchor * mask).sum() / denom

        with torch.no_grad():
            # Diagnostics: if valid_frac collapses to ~0 the movement threshold is wrong
            # for this data and the head is being trained on nothing.
            gt_mag = (target.norm(dim=-1) * mask).sum() / denom
            pred_mag = (predicted.norm(dim=-1) * mask).sum() / denom

        return {
            'loss': loss,
            'flow_huber': loss.detach(),
            'flow_valid_frac': mask.mean().detach(),
            'flow_gt_mag': gt_mag,
            'flow_pred_mag': pred_mag,
        }
