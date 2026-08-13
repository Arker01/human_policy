#!/usr/bin/env python3
"""Diagnostic: compare query0 reconstruction with z=0 (eval default) vs
encoder-derived z from ground-truth actions (oracle), on seen train episodes.
If oracle z reconstruction is much better than z=0, the decoder is leaning on
the CVAE latent channel instead of learning identity mapping from qpos alone.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HDT_DIR = _REPO_ROOT / "hdt"
sys.path.insert(0, str(_HDT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mpjpe_batch import _make_act_policy, _load_norm_stats, _read_image_frame, _action_to_eval_joints

CKPT_DIR = _REPO_ROOT / "train_all_episodes_only_ckpt"
MODEL_YAML = _HDT_DIR / "configs" / "models" / "act_resnet.yaml"
CKPT_PATH = CKPT_DIR / "policy_last.ckpt"
STATS_PATH = CKPT_DIR / "dataset_stats.pkl"
DATA_DIR = _REPO_ROOT / "data" / "all_episodes"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHUNK_SIZE = 100

EPISODES = ["wholebody-1_unified_V.hdf5", "wholebody-20_unified_V.hdf5"]
NUM_STEPS = 30  # sample this many timesteps per episode for speed


def joint_l2(a128, b128):
    ja, jb = _action_to_eval_joints(a128), _action_to_eval_joints(b128)
    errs = []
    for k in ["head", "lw", "rw", "waist", "neck"]:
        errs.append(np.linalg.norm(ja[k] - jb[k]))
    for k in ["lk_world", "rk_world"]:
        errs.append(np.linalg.norm(ja[k] - jb[k], axis=-1).mean())
    return float(np.mean(errs)) * 1000.0  # mm


def main():
    policy, meta = _make_act_policy(MODEL_YAML, CKPT_PATH, device=DEVICE)
    policy.eval()
    camera_names = meta["camera_names"]
    stats = _load_norm_stats(STATS_PATH, "human")
    qpos_mean, qpos_std = stats["qpos_mean"].reshape(1, -1), stats["qpos_std"].reshape(1, -1)
    act_mean, act_std = stats["action_mean"].reshape(1, -1), stats["action_std"].reshape(1, -1)

    for ep_name in EPISODES:
        ep_path = DATA_DIR / ep_name
        with h5py.File(ep_path, "r") as f:
            states = f["observation.state"][()]
            actions = f["action"][()] if "action" in f else states
            T = states.shape[0]
            step = max(1, T // NUM_STEPS)
            ts_list = list(range(0, max(1, T - CHUNK_SIZE), step))[:NUM_STEPS]

            errs_z0, errs_oracle, errs_raw_l1_z0, errs_raw_l1_oracle = [], [], [], []
            for t in ts_list:
                imgs = []
                for cam in camera_names:
                    key = f"observation.image.{cam}"
                    if key not in f:
                        key = "observation.image.left" if "observation.image.left" in f else "observation.image.right"
                    imgs.append(_read_image_frame(f[key], t))
                imgs = np.stack(imgs, axis=0)
                imgs_t = torch.from_numpy(imgs).to(device=DEVICE, dtype=torch.float32) / 255.0
                imgs_t = imgs_t.permute(0, 3, 1, 2).unsqueeze(0)

                qpos = states[t : t + 1].astype(np.float32)
                qpos_n = (qpos - qpos_mean) / (qpos_std + 1e-8)
                qpos_t = torch.from_numpy(qpos_n).to(device=DEVICE, dtype=torch.float32)

                gt_chunk = actions[t : t + CHUNK_SIZE].astype(np.float32)
                if gt_chunk.shape[0] < CHUNK_SIZE:
                    pad = np.zeros((CHUNK_SIZE - gt_chunk.shape[0], gt_chunk.shape[1]), dtype=np.float32)
                    gt_chunk = np.concatenate([gt_chunk, pad], axis=0)
                gt_chunk_n = (gt_chunk - act_mean) / (act_std + 1e-8)
                actions_t = torch.from_numpy(gt_chunk_n).unsqueeze(0).to(device=DEVICE, dtype=torch.float32)
                is_pad = torch.zeros((1, CHUNK_SIZE), dtype=torch.bool, device=DEVICE)

                with torch.no_grad():
                    # z = 0 path (normal eval / deployment)
                    a_hat_z0 = policy(imgs_t, qpos_t, conditioning_dict={})
                    a0_z0 = a_hat_z0[0, 0].cpu().numpy() * act_std.reshape(-1) + act_mean.reshape(-1)

                    # oracle path: encoder sees ground-truth action chunk -> real mu/logvar -> z
                    # DETRVAE.forward now returns a 4th value (the Future-DINO loss
                    # dict, None when the head is off) -- unpack it or this raises.
                    a_hat_oracle = policy.model(qpos_t, policy.transform(imgs_t), None, actions_t, is_pad, {})[0]
                    a0_oracle = a_hat_oracle[0, 0].cpu().numpy() * act_std.reshape(-1) + act_mean.reshape(-1)

                gt0 = actions[t].astype(np.float32)
                errs_z0.append(joint_l2(a0_z0, gt0))
                errs_oracle.append(joint_l2(a0_oracle, gt0))
                errs_raw_l1_z0.append(np.abs(a0_z0 - gt0).mean())
                errs_raw_l1_oracle.append(np.abs(a0_oracle - gt0).mean())

            print(f"== {ep_name} ({len(ts_list)} steps) ==")
            print(f"  query0 MPJPE-ish (z=0, eval default):   {np.mean(errs_z0):.2f} mm")
            print(f"  query0 MPJPE-ish (oracle z from GT):    {np.mean(errs_oracle):.2f} mm")
            print(f"  query0 raw L1   (z=0):    {np.mean(errs_raw_l1_z0):.5f}")
            print(f"  query0 raw L1   (oracle): {np.mean(errs_raw_l1_oracle):.5f}")


if __name__ == "__main__":
    main()
