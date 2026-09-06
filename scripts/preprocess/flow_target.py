#!/usr/bin/env python
"""
Build EgoWAM's 3D point-flow world target offline, one h5 per episode.

RUN THIS IN THE `track4world` CONDA ENV, NOT THE TRAINING ENV:

    /home/aigc/miniconda/envs/track4world/bin/python \
        scripts/preprocess/flow_target.py \
        --episodes example_data/robot_inspire_pickup_pillow_ep_0093.hdf5 \
        --out_dir /mnt/nvme0n1/flow_target_dex5

Track4World needs CUDA 12.1 / py3.11 / its own torch; the training env is untouched.


WHAT THE TARGET IS (EgoWAM, arXiv 2607.08436, third world target)
-----------------------------------------------------------------
For anchor frame t, anchor g on a fixed stride-8 pixel grid of frame t, and step k:

    target[t, k, g] = R_t^T (P_g(t+k) - P_g(t))          in metres

where P_g(f) is the 3D position at frame f of the scene point that sat under anchor g
at frame t. Expressing the displacement in the camera frame AT t is what factors
ego-motion out (EgoWAM's desideratum D3): static background goes to ~0 and only
genuinely moving things -- the hands and the pillow -- carry signal.

EgoWAM gets its ego-motion from Aria VIO head poses. Our hdf5 has no camera pose, no
depth and no intrinsics. Track4World predicts poses from RGB alone, so this step is
self-contained -- but see "STABILISATION" below for why we do not use those poses.


THREE THINGS MEASURED ON REAL EPISODES, EACH OF WHICH BREAKS THE TARGET IF ASSUMED
----------------------------------------------------------------------------------
1. `flow_2d` / `flow_3d` are LONG-RANGE, query-to-frame-i, not adjacent-pair.
   They have T entries for T input frames (not T-1), and entry 0 is the self-map.
   Measured on 24 real frames, mean |flow_2d[i] - identity| ran
   0.01, 0.43, 1.65, 2.20, ... 9.35, 9.38 px: monotone then plateauing, i.e. drift
   away from frame 0. Adjacent-pair flow would have been roughly constant in i.
   demo.py calls the variable `all_pairwise_flows_2d`; the name is wrong.
   => No chaining. One window yields every step of the trajectory directly, and the
      only composition error is the one at window boundaries (see WINDOWS).

2. `flow_3d[i]` is an absolute 3D POSITION, in frame i's OWN camera frame -- not a
   displacement. Its magnitude is ~1.85m, matching scene depth, and flow_3d[0]
   differs from `points[0]` (the frame-0 camera pointmap) by only 13mm while
   differing from `world_points[0]` by 85mm.
   => The displacement is a difference of two flow_3d entries, and the two entries
      live in DIFFERENT camera frames, so one of them has to be transformed first.

3. The predicted camera poses are not accurate enough to do that transform.
   Stabilising with `camera_poses` left static background at 16mm median at k=20 with
   a clear left-to-right gradient -- residual ego-motion, not object motion.
   => STABILISATION (below).


STABILISATION: FIT THE RIGID TRANSFORM, DO NOT TRUST THE PREDICTED POSE
-----------------------------------------------------------------------
flow_3d gives CORRESPONDING points across frames (same track), so the frame-i ->
frame-t rigid transform can be fitted directly from the point sets by robust weighted
Procrustes. The static majority of the scene defines the transform; the hands are the
outliers that IRLS downweights, and their residual is exactly the displacement we
want. This is ego-motion factoring done from the data instead of from a pose estimate,
and it halved the static-background residual (16mm -> 8mm median at k=20).

Alignments are fitted per window frame against the window's own frame 0 and then
COMPOSED, so a window of L frames costs L fits rather than one per (anchor, step)
pair.


WHY THE MOVEMENT THRESHOLDS ARE NOT EGOWAM'S
--------------------------------------------
EgoWAM uses 2mm for robot and 10mm for human. On our data the target's own noise floor
-- static background residual after robust stabilisation -- is 8-11mm, growing with k
(2mm at k=1, 8mm at k=20, 11mm at k=39). It did not improve at higher tracker
resolution: median residual at k=10 was 7.6mm at 240x320, 7.3mm at 448x336 and 9.6mm
at 640x480, so it is intrinsic to the model on this footage, not a sampling artefact.
A 2mm threshold would therefore mark ~99% of anchors as "moving" and train the head on
noise. The thresholds below sit above the floor, where the real signal is: the moving
hands and pillow reach 30-100mm, i.e. 3-10x the floor, and thresholding there
reproduces them (verified by eye on the displacement heatmaps).


WINDOWS
-------
A VGGT-style global aggregator over a whole 200-frame episode does not fit in 24GB
(40 frames at 240x320 already peaks at 9.8GB), so the episode is cut into windows.

Window length is deliberately STRIDE + KMAX. A window queried at frame q supplies
trajectories only for frame-q pixels, so an anchor frame t = q + a can use it for
steps 1..L-1-a; making L = stride + KMAX guarantees every anchor frame in [q, q+stride)
gets its full KMAX steps out of a single window. Nothing is chained across windows, so
there is no accumulating drift -- the price is redundant compute in the overlap.

KMAX is 39 rather than the head's 100 because a 101-frame window does not fit and
chaining to reach step 100 would stack composition error on top of an ~1cm noise
floor. Steps past KMAX are MASKED in the dataloader, not clamped. Human episodes lose
nothing: slow_down_factor 4 puts chunk step 100 at raw offset 25.
"""

import argparse
import os
import sys

import cv2
import h5py
import numpy as np
import torch

TRACK4WORLD_ROOT = os.environ.get('TRACK4WORLD_ROOT', '/home/aigc/Track4World')

# Movement threshold in metres, measured ABOVE the step's own background level rather
# than in absolute metres, and set from the measured noise floor rather than from
# EgoWAM's 2mm/10mm -- see "WHY THE MOVEMENT THRESHOLDS ARE NOT EGOWAM'S" above and the
# comment at the mask itself. An anchor that never clears the threshold over the horizon
# is dropped; most of the scene is static, so this is the only thing that stops the loss
# from being dominated by predicting zeros.
#
# The two embodiments come out at the same number, unlike EgoWAM's 2mm/10mm split: the
# limit here is the tracker's own precision, which does not care which body moved.
MOVE_THRESHOLD_M = {'human': 0.025, 'robot': 0.025}


def flow_target_filename(hdf_path: str) -> str:
    """Must stay identical to hdt/data_utils_hdt.flow_target_filename.

    Duplicated rather than imported because this script runs in the track4world env,
    which does not have the training package on its path.
    """
    flat = os.path.abspath(hdf_path).lstrip(os.sep).replace(os.sep, '__')
    for ext in ('.hdf5', '.h5'):
        if flat.endswith(ext):
            flat = flat[:-len(ext)]
            break
    return flat + '.flow.h5'


# ---------------------------------------------------------------------------
# Episode reading (the two formats differ; see example_data/README.md)
# ---------------------------------------------------------------------------

def read_episode_rgb(hdf_path: str, size_wh):
    """Return (rgb uint8 [T, H, W, 3], embodiment str), resized to size_wh.

    Human episodes store `observation.image.top` uncompressed; robot episodes store
    JPEG-encoded `observation.image.left`/`right` as variable-length uint8 rows. Same
    "top if present, else the first camera" rule perturb_eval.py uses.

    The resize is not cosmetic: robot JPEGs decode at 480x640, but the anchor grid is a
    stride-8 grid on the 240x320 frame the policy actually sees, so the tracker has to
    run at the policy's resolution for "anchor g" to mean the same pixel on both sides.
    """
    with h5py.File(hdf_path, 'r') as f:
        embodiment = str(f.attrs.get('embodiment', 'default'))
        cam_keys = [k for k in f.keys() if k.startswith('observation.image.')]
        assert cam_keys, f"no camera in {hdf_path}"
        key = ('observation.image.top' if 'observation.image.top' in cam_keys
               else sorted(cam_keys)[0])
        ds = f[key]
        frames = []
        for i in range(ds.shape[0]):
            if ds.ndim == 4:
                rgb_i = np.asarray(ds[i], dtype=np.uint8)          # already RGB
            else:
                buf = np.frombuffer(np.asarray(ds[i], dtype=np.uint8), dtype=np.uint8)
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                assert bgr is not None, f"imdecode failed at frame {i} of {hdf_path}"
                rgb_i = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            if (rgb_i.shape[1], rgb_i.shape[0]) != tuple(size_wh):
                rgb_i = cv2.resize(rgb_i, tuple(size_wh), interpolation=cv2.INTER_AREA)
            frames.append(rgb_i)
    return np.stack(frames), embodiment


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------

def build_model(ckpt_path: str, config_path: str, coordinate: str, metric_scale: bool,
                seqlen: int = 16):
    """Same construction as demo.py:load_model, minus the CLI object."""
    sys.path.insert(0, TRACK4WORLD_ROOT)
    import json
    from track4world.nets.model import Track4World

    with open(config_path, 'r') as fp:
        config = json.load(fp)

    model = Track4World(**config['model'], seqlen=seqlen, use_3d=True,
                        use_model=coordinate.split('_')[-1])
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_pretrained_with_remap(state_dict)
    model.cuda()
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    if metric_scale:
        # DA3-only (README:200). Without it the output is relative scale and the
        # metre-valued movement thresholds are meaningless, and the human and robot
        # halves stop agreeing with each other.
        model.use_metric_scale = True
    return model


@torch.no_grad()
def run_window(model, rgb: np.ndarray, iters: int = 4):
    """One tracker pass over a clip of L frames, all outputs keyed by FRAME-0 pixels.

    Returns (all torch, on GPU)
        uv      [L, H, W, 2]  absolute (u, v) at frame i of the frame-0 pixel (x, y)
        pos3d   [L, H, W, 3]  its 3D position at frame i, in frame i's OWN camera frame
        conf    [L, H, W]     forward * backward visibility confidence

    Long-range, not adjacent-pair: entry 0 is the self-map. See docstring point 1.
    """
    L = rgb.shape[0]
    rgbs = torch.from_numpy(rgb).permute(0, 3, 1, 2)[None].float().cuda()   # [1,L,3,H,W]
    output, _ = model.infer(rgbs, iters=iters, sw=None,
                            is_training=False, tracking3d=True)
    # The model does not keep every head's output on one device -- visconf comes back
    # on CPU for long clips while the flow fields stay on GPU -- so pin all three here
    # instead of discovering it inside the alignment loop.
    dev = rgbs.device
    uv = output[1]['flow_2d'][0].permute(0, 2, 3, 1).float().to(dev)         # [L,H,W,2]
    pos3d = output[1]['flow_3d'][0].float().to(dev)                          # [L,H,W,3]
    vc = output[1]['visconf_maps_e'][0].float().to(dev)                      # [L,2,H,W]
    conf = (vc[:, 0] * vc[:, 1])                                             # [L,H,W]
    assert uv.shape[0] == L, f"tracker returned {uv.shape[0]} entries for {L} frames"
    return uv, pos3d, conf


# ---------------------------------------------------------------------------
# Stabilisation
# ---------------------------------------------------------------------------

def robust_align_batch(src, dst, w0, iters=6, sigma=0.01):
    """Weighted Procrustes with IRLS, n independent fits at once.

    src [n,N,3] and dst [N,3] are CORRESPONDING points (same tracks, different frames):
    flow_3d is indexed by the window's frame-0 pixels, so row p means the same physical
    track in every frame and no matching step is needed. Returns (R [n,3,3], t [n,3])
    taking each src[i] onto dst.

    The scene is mostly static, so the majority vote is the ego-motion; `sigma` = 1cm
    sets the scale at which a point stops counting as static, i.e. the hands get
    downweighted to near zero over the iterations instead of dragging the fit, and their
    residual is left over as the displacement we actually want.

    Reflections are excluded via the standard det correction -- an unconstrained SVD fit
    can otherwise return a mirror when the point cloud is nearly planar, which a tabletop
    scene very nearly is.

    Fitting frame a against frame a+k DIRECTLY, rather than composing two fits through a
    shared gauge, is what keeps the residual clean: two slightly-wrong rotations compose
    into a residual rotation, which shows up as a smooth left-to-right gradient across
    the displacement field and is indistinguishable from real motion by any per-step
    threshold. Batching over k makes the direct version cost the same.
    """
    n = src.shape[0]
    w = w0.clone()
    eye = torch.eye(3, dtype=src.dtype, device=src.device).expand(n, 3, 3)
    R, t = eye.clone(), torch.zeros(n, 3, dtype=src.dtype, device=src.device)
    for _ in range(iters):
        ws = w / w.sum(dim=1, keepdim=True).clamp(min=1e-12)             # [n,N]
        ms = (ws[..., None] * src).sum(1)                                # [n,3]
        md = (ws[..., None] * dst[None]).sum(1)                          # [n,3]
        cov = torch.einsum('nNi,nNj->nij', (src - ms[:, None]) * ws[..., None],
                           dst[None] - md[:, None])
        U, _, Vt = torch.linalg.svd(cov)
        D = eye.clone()
        D[:, 2, 2] = torch.sign(torch.linalg.det(Vt.mT @ U.mT))
        R = Vt.mT @ D @ U.mT
        t = md - torch.einsum('nij,nj->ni', R, ms)
        resid = (torch.einsum('nij,nNj->nNi', R, src) + t[:, None] - dst[None]).norm(dim=-1)
        w = w0 / (1.0 + (resid / sigma) ** 2)
    return R, t


# ---------------------------------------------------------------------------
# Anchor bookkeeping
# ---------------------------------------------------------------------------

def anchor_uv(grid_hw, image_hw):
    """Row-major stride-8 anchor centres, matching build_2d_sincos_pos_embed's order.

    Index (r, c) -> r * grid_w + c, which is what FutureFlowHead's position table
    assumes. Centres (+stride/2) rather than corners so an anchor reads the middle of
    its 8x8 cell instead of its top-left corner.
    """
    gh, gw = grid_hw
    H, W = image_hw
    sy, sx = H / gh, W / gw
    rows = (np.arange(gh) + 0.5) * sy
    cols = (np.arange(gw) + 0.5) * sx
    cc, rr = np.meshgrid(cols, rows)                     # both [gh, gw], row-major
    return np.stack([cc.reshape(-1), rr.reshape(-1)], axis=1).astype(np.float32)


def spatial_basis(grid_hw, device, dtype):
    """Quadratic basis over normalised anchor coordinates, [P, 6]."""
    gh, gw = grid_hw
    ys = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh
    xs = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw
    y, x = torch.meshgrid(ys, xs, indexing='ij')
    y, x = y.reshape(-1), x.reshape(-1)
    return torch.stack([torch.ones_like(x), x, y, x * x, y * y, x * y], dim=1)


def detrend(disp, weight, basis, iters=6, sigma=0.01):
    """Remove the smooth part of the displacement field, robustly, one fit per step.

    Even after a direct rigid fit, a residual ego-motion field survives: on a walking
    robot at k=20 the background still reads ~30mm, arranged as a smooth field centred
    on a rotation point (visible as a large low-frequency blob across the anchor grid).
    It comes from the tracker's own 3D being inconsistent under a big viewpoint change,
    and from the near-planar couch making the rigid fit ill-conditioned -- not from
    anything a threshold can separate, because it is the same magnitude as real motion.

    But it is LOW-FREQUENCY, and object motion is LOCAL, so a robust quadratic fit over
    the anchor grid separates them. Subtracting it moves the target TOWARDS EgoWAM's
    ideal (pure object motion in the anchor camera frame), not away: it is a correction
    for our stabilisation being imperfect, which EgoWAM did not need because Aria gave
    it ground-truth head pose. Measured effect at k=20: kept fraction 0.23 -> 0.16, and
    the mask goes from covering the whole floor to hugging the grippers and the pillow.

    The IRLS weights make the moving parts, which are the minority, not drag the fit.
    """
    n, P, _ = disp.shape
    w = weight.clone().clamp(min=1e-6)
    for _ in range(iters):
        bw = basis[None] * w[..., None]                              # [n,P,6]
        ata = torch.einsum('npi,npj->nij', bw, basis[None].expand(n, P, 6))
        atb = torch.einsum('npi,npj->nij', bw, disp)
        ata = ata + 1e-9 * torch.eye(6, device=disp.device, dtype=disp.dtype)[None]
        coef = torch.linalg.solve(ata, atb)                          # [n,6,3]
        resid = disp - torch.einsum('pi,nij->npj', basis, coef)
        w = weight / (1.0 + (resid.norm(dim=-1) / sigma) ** 2)
    return resid


@torch.no_grad()
def tracks_at(uv_a, conf_a, anchors, H, W, conf_threshold):
    """Which frame-0 track passes through each anchor pixel of frame q+a.

    The window only knows tracks born at frame q, but the head's anchors are a fixed
    grid on the ANCHOR frame -- it sees frame t and nothing else, so the anchor index
    has to mean a pixel of frame t. That needs the inverse of the forward flow, built
    here by scattering each frame-0 pixel into the frame-(q+a) pixel it landed on.

    An anchor with no track landing on it (occluded at q, or the track was lost) gets
    -1 and is reported as uncovered rather than silently falling back to the identity
    track, which would attach a wrong trajectory to a real anchor.

    Returns (idx [P] long, covered [P] bool) where idx indexes flattened frame-0 pixels.
    """
    flat_uv = uv_a.reshape(-1, 2)
    flat_conf = conf_a.reshape(-1)
    u = flat_uv[:, 0].round().long()
    v = flat_uv[:, 1].round().long()
    ok = ((u >= 0) & (u < W) & (v >= 0) & (v < H) &
          torch.isfinite(flat_uv).all(dim=1) & (flat_conf > conf_threshold))

    inv = torch.full((H * W,), -1, dtype=torch.long, device=uv_a.device)
    src = torch.arange(flat_uv.shape[0], device=uv_a.device)
    # Highest confidence wins a contested destination pixel: sorting ascending and
    # letting later writes overwrite makes the last (best) write the survivor.
    order = torch.argsort(torch.where(ok, flat_conf, torch.full_like(flat_conf, -1.0)))
    dst = (v.clamp(0, H - 1) * W + u.clamp(0, W - 1))[order]
    inv[dst[ok[order]]] = src[order][ok[order]]

    au = torch.from_numpy(anchors[:, 0]).to(uv_a.device).round().long().clamp(0, W - 1)
    av = torch.from_numpy(anchors[:, 1]).to(uv_a.device).round().long().clamp(0, H - 1)
    idx = inv[av * W + au]
    covered = idx >= 0
    return idx.clamp(min=0), covered


# ---------------------------------------------------------------------------
# Per-episode driver
# ---------------------------------------------------------------------------

@torch.no_grad()
def build_episode_target(model, rgb, args):
    """Run every window and fill the [T, KMAX, P, 3] target for a whole clip.

    Returns
        target  [T, KMAX, P, 3] float32  displacement in metres, anchor camera frame
        valid   [T, KMAX]       bool     step k exists (frame t+k is inside the clip)
        anchor  [T, P]          bool     anchor covered by a track AND actually moved
    """
    T, H, W = rgb.shape[0], rgb.shape[1], rgb.shape[2]
    grid_hw = (args.grid_h, args.grid_w)
    P = args.grid_h * args.grid_w
    K = args.kmax
    stride = args.stride
    L = stride + K                                     # see WINDOWS in the docstring

    anchors = anchor_uv(grid_hw, (H, W))
    basis = spatial_basis(grid_hw, 'cuda', torch.float64)
    target = np.zeros((T, K, P, 3), dtype=np.float32)
    valid = np.zeros((T, K), dtype=bool)
    anchor_ok = np.zeros((T, P), dtype=bool)

    n_win = 0
    for q in range(0, T, stride):
        end = min(q + L, T)
        if end - q < 2:
            break
        uv, pos3d, conf = run_window(model, rgb[q:end], iters=args.iters)
        L_act = uv.shape[0]
        n_win += 1

        p_flat = pos3d.reshape(L_act, -1, 3).double()               # [L,HW,3]
        c_flat = conf.reshape(L_act, -1)                            # [L,HW]
        # A few thousand points already pin down 6 DoF; fitting on all 76800 would spend
        # the time on redundant rows. Deterministic stride so reruns match.
        step = max(1, p_flat.shape[1] // args.n_fit)
        sub = torch.arange(0, p_flat.shape[1], step, device=p_flat.device)

        for a in range(min(stride, L_act - 1)):
            t = q + a
            if t >= T:
                break
            idx, covered = tracks_at(uv[a], conf[a], anchors, H, W, args.vis_threshold)
            n_steps = min(K, L_act - 1 - a)

            # One direct rigid fit per step, all k at once: frame a+k -> frame a. The
            # residual after removing that transform IS the displacement expressed in
            # the anchor frame's camera, which is EgoWAM's D3 -- no camera pose needed.
            src = p_flat[a + 1:a + 1 + n_steps][:, sub]              # [n,S,3]
            dst = p_flat[a, sub]                                    # [S,3]
            w0 = (c_flat[a + 1:a + 1 + n_steps][:, sub] *
                  c_flat[a, sub][None]).double().clamp(min=1e-3)    # [n,S]
            R, tr = robust_align_batch(src, dst, w0, sigma=args.align_sigma)

            base = p_flat[a, idx]                                   # [P,3]
            fut = p_flat[a + 1:a + 1 + n_steps][:, idx]             # [n,P,3]
            disp = torch.einsum('nij,npj->npi', R, fut) + tr[:, None] - base[None]

            # A step is only usable if the track is still confidently visible there.
            live = c_flat[a + 1:a + 1 + n_steps][:, idx] > args.vis_threshold  # [n,P]
            keep = live & covered[None, :]

            if n_steps and not args.no_detrend:
                disp = detrend(disp, keep.to(disp.dtype), basis,
                               sigma=args.align_sigma)
            disp = torch.where(keep[..., None], disp, torch.zeros_like(disp))

            target[t, :n_steps] = disp.float().cpu().numpy()
            valid[t, :n_steps] = True
            # An anchor counts as moving if it ever clears the threshold on a step where
            # its track was alive -- but the threshold is measured ABOVE THAT STEP'S OWN
            # BACKGROUND LEVEL, not in absolute metres.
            #
            # Why: the stabilisation residual grows with k (measured p50 over the
            # anchors: 2mm at k=1, 8mm at k=10, 19mm at k=39) because 39 frames of a
            # walking robot change the viewpoint a lot. A fixed 25mm cut therefore
            # passes 44% of anchors at k=39 -- almost all of it residual ego-motion, the
            # exact "head trains on noise" failure this mask exists to prevent. The
            # signal, meanwhile, separates BETTER at large k, not worse: p90/p50 goes
            # from 3.1x at k=1 to 5.7x at k=39. So the discriminative quantity is
            # displacement relative to the static majority at the same step, which is
            # what the median picks out, and subtracting it recovers the sparse mask
            # (0.09 at k=10, 0.26 at k=39) without throwing away the long steps.
            mag = disp.norm(dim=-1)                                     # [n,P]
            if n_steps:
                live_mag = torch.where(keep, mag, torch.full_like(mag, float('nan')))
                base = torch.nanmedian(live_mag, dim=1).values           # [n]
                base = torch.nan_to_num(base, nan=0.0)
                moved = ((mag > base[:, None] + args.threshold) & keep).any(dim=0)
            else:
                moved = torch.zeros(P, dtype=torch.bool, device=disp.device)
            anchor_ok[t] = (moved & covered).cpu().numpy()

        del uv, pos3d, conf, p_flat, c_flat
        torch.cuda.empty_cache()
        if end == T:
            break

    return target, valid, anchor_ok, n_win


def process_episode(model, hdf_path, out_dir, args):
    rgb, embodiment = read_episode_rgb(hdf_path, (args.width, args.height))
    T_full = rgb.shape[0]
    is_human = 'human' in embodiment

    # EgoWAM skips the first and last 20 frames of human clips (mounting the headset and
    # putting it down are not manipulation). Kept as raw-frame padding below so h5 row t
    # still IS raw frame t.
    trim = args.human_trim if is_human else 0
    lo, hi = trim, T_full - trim
    assert hi - lo > 2, f"{hdf_path}: only {hi - lo} usable frames after trimming"

    args.threshold = MOVE_THRESHOLD_M['human' if is_human else 'robot']
    target, valid, anchor_ok, n_win = build_episode_target(model, rgb[lo:hi], args)

    P = args.grid_h * args.grid_w
    K = args.kmax
    full_target = np.zeros((T_full, K, P, 3), dtype=np.float16)
    full_valid = np.zeros((T_full, K), dtype=bool)
    full_anchor = np.zeros((T_full, P), dtype=bool)
    full_target[lo:hi] = target.astype(np.float16)
    full_valid[lo:hi] = valid
    full_anchor[lo:hi] = anchor_ok

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, flow_target_filename(hdf_path))
    with h5py.File(out_path, 'w') as f:
        f.create_dataset('flow_target', data=full_target, compression=None)
        f.create_dataset('flow_valid', data=full_valid)
        f.create_dataset('anchor_valid', data=full_anchor)
        f.attrs['source'] = os.path.abspath(hdf_path)
        f.attrs['embodiment'] = embodiment
        f.attrs['grid_hw'] = (args.grid_h, args.grid_w)
        f.attrs['image_hw'] = (args.height, args.width)
        f.attrs['horizon_raw_offsets'] = K
        f.attrs['move_threshold_m'] = args.threshold
        f.attrs['trim'] = trim
        f.attrs['units'] = 'metres, camera frame of the anchor frame'

    # Diagnostics on exactly the elements the loss will see: kept anchors on existing
    # steps. anchor_valid is [T,P] against a [T,K,P,3] target, so it broadcasts over K.
    seen = anchor_ok[:, None, :] & valid[:, :, None]
    mag = np.linalg.norm(target, axis=-1)
    kept = float(anchor_ok.mean())
    mean_mag = float(mag[seen].mean()) if seen.any() else 0.0
    bg = float(np.median(mag[valid[:, :, None] & ~anchor_ok[:, None, :]])) if (
        (~anchor_ok).any()) else 0.0
    print(f"  {os.path.basename(out_path)}  T={T_full} trim={trim} windows={n_win} "
          f"anchor_valid={kept:.3f} kept|d|={mean_mag * 1000:.0f}mm "
          f"dropped|d|median={bg * 1000:.1f}mm size={os.path.getsize(out_path)/1e6:.0f}MB")
    return kept, mean_mag, bg


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', nargs='+', required=True,
                   help='episode hdf5 paths, or directories to scan for *.hdf5')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--ckpt', default=os.path.join(TRACK4WORLD_ROOT,
                                                  'checkpoints/track4world_da3.pth'))
    p.add_argument('--config', default=os.path.join(TRACK4WORLD_ROOT,
                                                    'track4world/config/eval/v1.json'))
    # DA3 is not a free choice: metric scale is DA3-only (README:200), and without
    # metric scale the movement thresholds and the human/robot alignment both break.
    p.add_argument('--coordinate', default='world_depthanythingv3')
    p.add_argument('--height', type=int, default=240,
                   help='tracker resolution; must be the policy resolution')
    p.add_argument('--width', type=int, default=320)
    p.add_argument('--grid_h', type=int, default=30)
    p.add_argument('--grid_w', type=int, default=40)
    p.add_argument('--kmax', type=int, default=39,
                   help='raw-frame offsets stored per anchor frame; steps past this '
                        'are masked by the dataloader, not clamped')
    p.add_argument('--stride', type=int, default=16,
                   help='anchor frames served per window; window length is stride+kmax')
    p.add_argument('--iters', type=int, default=4)
    p.add_argument('--n_fit', type=int, default=8192,
                   help='points subsampled for each rigid-alignment fit')
    p.add_argument('--align_sigma', type=float, default=0.01,
                   help='IRLS scale in metres: above this a point stops counting as '
                        'static and gets downweighted out of the ego-motion fit')
    p.add_argument('--no_detrend', action='store_true',
                   help='debug: keep the smooth residual ego-motion field in the target')
    p.add_argument('--vis_threshold', type=float, default=0.3,
                   help='matches demo.py:717')
    p.add_argument('--human_trim', type=int, default=20,
                   help="EgoWAM drops the first/last 20 frames of human clips")
    p.add_argument('--limit', type=int, default=0, help='debug: first N episodes only')
    p.add_argument('--skip_existing', action='store_true')
    # Sharding by index%n rather than by contiguous block: episode length varies a
    # lot (and human/robot are interleaved in the sorted order), so round-robin
    # keeps the 8 shards within a few minutes of each other.
    p.add_argument('--num_shards', type=int, default=1)
    p.add_argument('--shard', type=int, default=0)
    args = p.parse_args()

    paths = []
    for e in args.episodes:
        if os.path.isdir(e):
            for root, _, files in os.walk(e):
                paths += [os.path.join(root, f) for f in files if f.endswith('.hdf5')]
        else:
            paths.append(e)
    paths = sorted(paths)
    if args.limit:
        paths = paths[:args.limit]
    if args.num_shards > 1:
        paths = paths[args.shard::args.num_shards]
    print(f"{len(paths)} episodes -> {args.out_dir}")

    model = build_model(args.ckpt, args.config, args.coordinate, metric_scale=True)

    stats = []
    for i, path in enumerate(paths):
        out = os.path.join(args.out_dir, flow_target_filename(path))
        if args.skip_existing and os.path.exists(out):
            print(f"[{i + 1}/{len(paths)}] skip {os.path.basename(path)}")
            continue
        print(f"[{i + 1}/{len(paths)}] {os.path.basename(path)}")
        stats.append(process_episode(model, path, args.out_dir, args))

    if stats:
        kept = np.mean([s[0] for s in stats])
        print(f"\nmean anchor_valid over {len(stats)} episodes: {kept:.3f}")
        print(f"mean |d| on kept anchors: {np.mean([s[1] for s in stats]) * 1000:.0f}mm")
        print(f"median |d| on dropped anchors (the noise floor): "
              f"{np.mean([s[2] for s in stats]) * 1000:.1f}mm")
        print("If anchor_valid is near 0 the threshold is too high for this data and "
              "the head trains on an empty mask; watch flow_valid_frac in training.")
        print("If it is near 1 the threshold is below the noise floor and the head "
              "trains on noise -- that is what EgoWAM's 2mm did here.")


if __name__ == '__main__':
    main()
