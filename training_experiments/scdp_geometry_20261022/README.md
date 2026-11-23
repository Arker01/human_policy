# SCDP-Lite on two geometry w0 base checkpoints

All additions and training outputs stay in this directory. Existing HAT source,
base checkpoints, previous SCDP runs, datasets and deployment are read-only.

Remote: `4090-168-aigc`, `/home/aigc/human_policy/training_experiments/scdp_geometry_20261022`.

| Run | GPU | Exact base |
|---|---:|---|
| geometry | 6 | isolated_20261020_readiness_geometry/checkpoints/geometry_ckpt |
| geometry_no_left_wrist | 7 | isolated_20261021_no_left_wrist/checkpoints/geometry_ckpt |

Base directories above are under `/mnt/nvme0n1/human_policy_ckpt/`. The first base
weight SHA256 matches the user-named local bundle
`b0930_20261020_replace50_nostate_vjepa_w0_lagvis_geometry_ckpt` exactly:
`a55641caf0c1532a4fef2793c07506c41cd4873f95e8edf78c0edfaae9ecf5d0`.
Its inference YAML is `act_input_vjepa_w0_nostate_lagvis_geometry.yaml`.
The second base SHA256 is
`7d0e0f736b0f12113a6940ad515a97df58d571555421dc0f66ff32182054a146`.

`configs/bases.json` records both exact source trees, checkpoint paths, hashes,
normalization/input-mask settings and dataset manifests. Each base is loaded with
its own saved training policy configuration and original isolated source tree,
with strict state_dict loading. The original source files are never patched.

The no-left-wrist base masks normalized encoder INPUT 80:89, plus the original
state-tail mask 98:128. It keeps all 128 action outputs and original labels.
SCDP uses predicted coarse wrist anchors, RGB and camera poses; measured left
wrist values do not bypass that input mask. Perturbing normalized left-wrist
inputs by +100 leaves both the original and enabled-SCDP inference bitwise equal.

Each run has independent, newly zero-initialized FiLM parameters. V-JEPA and the
entire original HAT remain frozen; only 657664 new parameters are trained. Global
batch is 64 on ONE GPU per run, LR 1e-5, 100000 steps. Checkpoints save every 1000
steps. Original source ACTPolicy is retained, including the 0.1 geometry objective,
hand-event weighting, EEF/head/waist/KL coefficients and visual latent weight 0.
Original dataset split, sampling, slowdown, resampling and augmentation remain.
The original geometry label/collation helpers are used, not recreated.

Camera sidecars are read-only inputs shared with the earlier SCDP experiment:
`../scdp_lite_20261020/calibration_sidecars_20261021/manifest.json`.
Human new uses the validated equivalent screen camera and +97.428 ms pose
alignment for RIGHT wrist only. Human old remains exact baseline fallback.
Robot raw LEFT RGB uses the supplied ZED rational K/D and verified virtual-head
frame, with the same user-approved coarse-extrinsic uncertainty. No calibration
is refitted, no new labels/data are collected, no robot control is launched.

## Status and outputs

- `logs/geometry.log`, `logs/geometry_no_left_wrist.log`.
- `logs/<run>_launch.json`: PID, command, GPU, hashes, source provenance.
- `checkpoints/geometry/step_<N>.pt`.
- `checkpoints/geometry_no_left_wrist/step_<N>.pt`.

Both are detached processes that survive the launching SSH session. The launcher
refuses duplicate run names and occupied GPUs. `src/launch_variants.py` starts
both runs; do not call it again for already-running experiments.

```bash
ssh 4090-168-aigc
cd /home/aigc/human_policy/training_experiments/scdp_geometry_20261022
tail -f logs/geometry.log
# In another terminal:
tail -f logs/geometry_no_left_wrist.log
```

## Verification and loading

15 inherited geometry/projection/masking tests pass. Actual checkpoint checks
verify default-off equality, exact no-left input invariance, geometry-loss
presence, finite nonzero SCDP gradients and frozen original weights. On real fixed
minibatches, 100-step overfit at the explicitly higher test LR 1e-3 reduces
supported-wrist normalized L1 from 0.069853 to 0.037497 for geometry and from
0.072852 to 0.037147 for no-left (46.3% / 49.0%). Formal runs reset those parameters
and use LR 1e-5. These checks establish working training, not task performance.

The base selector MUST be set before importing HAT modules. Loader checks that
checkpoint and runtime base selectors agree and strictly preserves the base mask.

```bash
PY=/home/aigc/miniconda/envs/human_policy/bin/python
SCDP_BASE_VARIANT=geometry CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "$PY" src/evaluate.py checkpoints/geometry/step_1000.pt
SCDP_BASE_VARIANT=geometry_no_left_wrist CUDA_VISIBLE_DEVICES=7 OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "$PY" src/evaluate.py checkpoints/geometry_no_left_wrist/step_1000.pt
```

For offline API loading, set `SCDP_BASE_VARIANT` to the intended run and use
`src/load_policy.py::load_scdp_policy`. Camera/normalization metadata must be passed
through the optional conditioning dict, as in the earlier SCDP experiment.
No existing online HAT configuration, checkpoint selection or chunk scheduler is
changed. Checkpoint comparison always uses each run's OWN coarse base output.
No automatic plots, videos or analysis reports are generated.
