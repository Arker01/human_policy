#!/usr/bin/env python3
"""Trace predicted vs ground-truth values for the worst-error dims (right/left
wrist EEF x,y — dims 30,31,80,81) over a full episode, to see whether the error
is a constant bias, a time lag, or pure noise.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "hdt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mpjpe_batch import _make_act_policy, _load_norm_stats, _read_image_frame

CKPT_DIR = _REPO_ROOT / "train_all_episodes_only_ckpt"
MODEL_YAML = _REPO_ROOT / "hdt" / "configs" / "models" / "act_resnet.yaml"
DATA_DIR = _REPO_ROOT / "data" / "all_episodes"
DEVICE = "cpu"
EPISODE = "wholebody-50_unified_V.hdf5"  # regime C, low-std, representative
DIMS = {30: "right_wrist_x", 31: "right_wrist_y", 32: "right_wrist_z",
        80: "left_wrist_x", 81: "left_wrist_y", 82: "left_wrist_z"}


def main():
    policy, meta = _make_act_policy(MODEL_YAML, CKPT_DIR / "policy_last.ckpt", device=DEVICE)
    policy.eval()
    cams = meta["camera_names"]
    stats = _load_norm_stats(CKPT_DIR / "dataset_stats.pkl", "human")
    qm, qs = stats["qpos_mean"].reshape(1, -1), stats["qpos_std"].reshape(1, -1)
    am, as_ = stats["action_mean"].reshape(1, -1), stats["action_std"].reshape(1, -1)

    with h5py.File(DATA_DIR / EPISODE, "r") as f:
        states = f["observation.state"][()]
        T = states.shape[0]
        preds = np.zeros((T, 128), dtype=np.float32)
        for t in range(T):
            imgs = []
            for cam in cams:
                key = f"observation.image.{cam}"
                imgs.append(_read_image_frame(f[key], t))
            imgs = np.stack(imgs, axis=0)
            imgs_t = torch.from_numpy(imgs).to(dtype=torch.float32) / 255.0
            imgs_t = imgs_t.permute(0, 3, 1, 2).unsqueeze(0)
            qpos = states[t : t + 1].astype(np.float32)
            qpos_n = (qpos - qm) / (qs + 1e-8)
            qpos_t = torch.from_numpy(qpos_n).float()
            with torch.no_grad():
                a_hat = policy(imgs_t, qpos_t, conditioning_dict={})
            preds[t] = a_hat[0, 0].numpy() * as_.reshape(-1) + am.reshape(-1)

    for dim, name in DIMS.items():
        gt = states[:, dim]
        pred = preds[:, dim]
        err = pred - gt
        bias = err.mean()
        # cross-correlation to detect lag: find shift that minimizes L1 error
        best_lag, best_err = 0, np.abs(err).mean()
        for lag in range(-10, 11):
            if lag == 0:
                continue
            if lag > 0:
                e = np.abs(pred[lag:] - gt[:-lag]).mean()
            else:
                e = np.abs(pred[:lag] - gt[-lag:]).mean()
            if e < best_err:
                best_err, best_lag = e, lag
        print(f"{name:16s} (dim{dim}): gt_std={gt.std():.3f} bias={bias:+.3f} "
              f"raw_mae={np.abs(err).mean():.3f} best_lag={best_lag} lagged_mae={best_err:.3f}")

    # print first/last 15 raw values for dim 30 to eyeball pattern
    print("\nfirst 20 timesteps, dim30 (right wrist x): gt vs pred")
    for t in range(0, 20):
        print(f"  t={t:3d} gt={states[t,30]:+.3f} pred={preds[t,30]:+.3f} diff={preds[t,30]-states[t,30]:+.3f}")


if __name__ == "__main__":
    main()
