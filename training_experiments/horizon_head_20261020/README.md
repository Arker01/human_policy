# Isolated HAT horizon-head training (2026-10-20)

Remote machine: `ssh 4090-168-aigc`. Remote root:
`/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_horizon_head`.
All changes, labels, logs, configurations and new weights live here. The existing
`/home/aigc/human_policy` and readiness/geometry experiment are read-only dependencies.
No production service or ScaleBridge file is modified and no model is deployed.

Two 100,000-step fine-tuning runs, each batch 32 x 2 GPUs, constant lr 1e-5,
chunk 100, seed 0, original action representation, replace50 split, human slow-down
4, robot target 30 Hz, original V-JEPA encoder, lag/visual augmentation,
zero_state_dims 98:128, future visual loss weight 0:

- `original`: GPU 2,3; initialized from the existing
  `b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt/policy_last.ckpt`.
- `geometry`: GPU 4,5; initialized from the previous isolated experiment's
  `checkpoints/geometry_ckpt/policy_last.ckpt` (the geometry deployment bundle's
  source checkpoint); retains geometry loss weight 0.1.

Each run uses its own frozen teacher and its own label cache. Horizon BCE weight
is 0.1, supervised only on dex5 valid steps. Original robot/human sampler retained.
Training initialization allows missing keys **only** for the new horizon head;
all existing parameters must load. No old optimizer state is resumed.

## Deliberate corrections to the design document

The strict Stage 0 phase-ratio gate **FAILED for both teachers**. At residual
3 cm, far/near median horizons are original 10/10 and geometry 10/12 frames;
all-dex5 p75-p25 gaps are 16 and 17 frames. Original results remain at
`results/{original,geometry}/stage0_anchored_error.json` with `passed: false`.
The user explicitly permits correcting unreliable assumptions in the document
and requires starting training. Runs therefore record an **exploratory override**,
not a passed acceptance gate. The phase ratio does not mathematically establish
that labels are constant; the required 15-frame spread is present. These runs
must be evaluated for useful prediction and are not deployment approval.

- A training decoder state contains a CVAE posterior derived from future actions.
  The new head uses a second decoder pass with **z=0**, sharing current visual
  features. Main action training retains its posterior path. Inference uses one
  pass and returns both actions and reliability.
- Head hazards start with bias -4 and global logit bias 2, avoiding the original
  suggested initialization's immediate cumulative-logit saturation.
- Cache rows use the dataset's logical observation index; columns already follow
  the actual `_read_action_chunk` interpolation and downsampling. Applying human
  interpolation again to these labels would slow the timeline twice. Identical
  stems under human_old/human_new have distinct path-hashed cache keys.
- Threshold decisions and the 3 cm gate use float32 errors; float16 stored error
  arrays are diagnostics. Padded steps are excluded. Orientation errors are
  rotationally re-anchored before comparison.
- The old training loop reuses `data` in its validation loop, then accidentally
  trains once on the last validation batch. Only the horizon path restores the
  saved training batch before backpropagation. The baseline execution path is
  retained; old/new baseline inference was checked bitwise equal.
- Calibration reports oracle-onset bins for diagnosis, but selects a **single**
  threshold meeting coverage in every observed bin. Future demonstrated onset is
  unavailable online. Raw and displacement/minimum-frame guarded coverage are
  reported separately; guarded execution can exceed the reliable prefix. Primary
  quality and coverage exclude episode tails without 100 available ground-truth
  steps, so future episode length cannot artificially improve predicted K.

## Entrypoints

Run from `<remote root>/src` with the existing human_policy Python environment.
These commands refuse to overwrite an existing run; do not repeat live launchers.

```bash
python label_horizon.py original
python label_horizon.py geometry
# A failed strict gate requires the recorded, explicit exploratory override:
python run_horizon.py original 2,3 --exploratory-stage0-override
python run_horizon.py geometry 4,5 --exploratory-stage0-override
```

`--smoke` selects an independent three-step checkpoint directory. Both smoke
runs exited 0. First-step horizon gradient norms were 0.180715 / 0.161692,
and action-head norms 1.414909 / 1.844767. Unit checks cover monotonicity,
masking, empty dex5 batches, gradients and cache collisions. Baseline old/new
prediction SHA-256: `0035d7274e4568b081977916459c5adbf999df12022a1def9df68115ae3b05b6`.

Training: `logs/{original,geometry}.log`, metrics in
`checkpoints/{original,geometry}_ckpt/metrics.jsonl` and CSVs. Launch PIDs and
exact arguments are saved in `configs/{original,geometry}.json`. Exit status is
written to `results/{original,geometry}_status.json`. Successful full training
is followed automatically by held-out calibration, matched teacher action-error
comparison, the original full-body MPJPE protocol for student and teacher, and
source/checkpoint SHA-256 verification. Post-training logs/status have a
`_posttrain` suffix. No figures are generated.

Load weights with `src/load_experiment.py`; call
`policy.predict_with_horizon(image, normalized_qpos)` to obtain normalized action
chunks and monotone per-step reliability. This does not perform robot I/O.
