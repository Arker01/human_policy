# HAT SCDP-Lite basic experiment

All implementation, camera sidecars, checkpoints and logs stay inside this directory.
Original HAT source, datasets, base checkpoint and deployment are read-only.
No ablations or robot control runs are launched.

## Basic run

Host: 4090-168-aigc. Remote experiment:
`/home/aigc/human_policy/training_experiments/scdp_lite_20261020/`.

The named run `basic_20261021` uses GPUs 6,7, global batch 64 (32/rank), LR 1e-5,
and 100000 steps. It starts from the specified
`b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt/policy_last.ckpt` and its
original embodiment normalization. V-JEPA and the original HAT are frozen;
only 657664 new conditioner parameters are optimized. Original ACTPolicy loss,
EEF/head/waist/KL weights, future-DINO weight 0, data split, human slowdown,
robot resampling and lag/visual augmentation are retained. No duplicate coarse
loss is added because that head is frozen. This run resets adapter parameters;
it does not continue the overfit/smoke-test weights.

- Log: `logs/basic_20261021.log`.
- Command, PID, start time: `logs/basic_20261021_launch.json`.
- Checkpoints: `checkpoints/basic_20261021/step_<step>.pt`, every 1000 steps.
- Test checkpoint: `checkpoints/ddp_smoke_20261021/step_20.pt`.

`src/launch_basic.py` starts a detached run and refuses duplicate runs or occupied
GPUs. To monitor: `tail -f logs/basic_20261021.log` on the remote host.

## Camera integration and scope

`configs/cameras_20261021.json` preserves supplied parameters and uncertainty.
`src/prepare_calibration.py` writes independent sidecars, never original HDFs.

- Human new: equivalent screen-recording K, original ARKit-head extrinsics,
  +97.428 ms pose alignment. The converter frame is recovered from head poses and
  independently verified against all recorded wrists (maximum residual 5.5e-7 m).
  Translation uses linear interpolation, rotation uses SLERP. The last 3 raw
  frames of each recording fall outside the corrected timeline and fall back.
- Human left: masked because it lacks a validated camera fit. Its coarse wrist
  is retained even when the right wrist has valid projection anchors.
- Human old: no reliable calibration, so exact original-HAT fallback. Original
  dataset membership and sampling are retained.
- Robot: serial 34051753, raw unrectified LEFT RGB, eight-coefficient OpenCV
  rational distortion. Stored images are 640x480. K is converted with pixel-centre
  consistency so the existing 320x240 resize uses the supplied K. No image
  undistortion or new image resampling is introduced.
- Robot head: verified by FK from stored root/waist joints to be torso orientation
  and torso+[0,0,.4] virtual head. The supplied virtual-head translation is used;
  URDF head_link parameters are preserved separately. Audit-only joint values do
  not enter the conditioner. Training camera transforms use unaugmented recorded
  head pose, including the episode's existing origin translation.
- Robot extrinsics remain an approved rough-projection candidate, without physical
  precision claims. Training success cannot validate the mounting transform.

Raw-frame camera availability, before trimming/resampling:

| Source | Available / total | Wrist scope |
|---|---:|---|
| train/human_new | 7874 / 8024 | right |
| train/robot_new | 43820 / 43820 | both |
| train/human_old | 0 / 14038 | baseline fallback |
| val/robot_new | 6649 / 6649 | both |
| val/human_old | 0 / 2485 | baseline fallback |

Camera availability does not guarantee a coarse future anchor projects in view.
Actual anchor validity is masked and logged separately during training.

The conditioner samples 8 uniform anchors per wrist from native 768x14x19 V-JEPA
patches. Detached coarse anchors are denormalized before K/D projection. Actual
random crop parameters are recorded for the same image stack. Zero-initialized
FiLM conditions existing action hidden states; the same original KP head generates
refined candidates. Only valid wrists' XYZ channels are replaced. Non-wrist
channels and invalid wrists preserve coarse values bit for bit. Default-off and
original return contracts are preserved. No depth, world model, labels or
pseudo high-resolution features are introduced.

## Verification and reproduction

Run on the remote host, from the experiment directory:

```bash
PY=/home/aigc/miniconda/envs/human_policy/bin/python
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 "$PY" -m unittest discover -s tests -v
PYTHONDONTWRITEBYTECODE=1 "$PY" src/prepare_calibration.py
PYTHONDONTWRITEBYTECODE=1 "$PY" src/train.py --preflight
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 "$PY" src/train.py --overfit-steps 100 --batch-size 8 --lr 0.001 --workers 2
```

15 tests pass: OpenCV distortion comparison, stored-image resize, URDF/virtual-head
transform equivalence, timing, crop, wrist indices, masked-hand exact fallback,
default-off equality, shapes and gradients. A real fixed-minibatch overfit test
reduces supported wrist normalized L1 0.079740 -> 0.035790 (55.1%) at the explicitly
higher test LR. A 20-step two-GPU run saved and reloaded exact adapter weights.
Prior-only offline checks confirm zero non-wrist differences and exact old-human
fallback. These establish working implementation, not improved robot success.

Direct training, ONLY with a NEW output directory:

```bash
CUDA_VISIBLE_DEVICES=6,7 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "$PY" -m torch.distributed.run --nproc_per_node=2 --master_port=29868 src/train.py --steps 100000 --batch-size 32 --lr 1e-5 --save-every 1000 --output /absolute/path/inside/this/experiment/checkpoints/NEW_RUN
```

`src/load_policy.py::load_scdp_policy` reloads base plus adapter. Offline inference
must provide `scdp_camera`, `scdp_action_mean`, `scdp_action_std` through the optional
conditioning dict, with the same aligned camera poses. Missing metadata falls
back. No live service or chunk scheduling is modified. Debug overlay remains an
opt-in in-memory helper; no automatic plots or analysis reports are generated.

Reload and check a saved checkpoint on the original held-out split (console only):

```bash
CUDA_VISIBLE_DEVICES=6 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 "$PY" src/evaluate.py checkpoints/basic_20261021/step_1000.pt
```

The first formal checkpoint was reloaded successfully. Its prior-only robot check
on 30 fixed observation frames gives mean right-wrist error 77.80 mm coarse and
76.29 mm refined. Non-wrist differences and old-human fallback differences remain
exactly zero. This early check is a limited sample, not a final performance result.
