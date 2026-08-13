#!/usr/bin/env python3
"""Detect body-turning direction and walking pattern per episode, in WORLD frame
(not head-relative, since turning/walking direction is exactly what head-relative
transforms would cancel out). Both hands are analyzed (task always uses both
hands to grab two fruits simultaneously, per user clarification).
"""
import re
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mpjpe_batch import _rot6d_to_mat
import hdt.constants as C

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "all_episodes"
WAIST_SLICE = slice(89, 98)  # 3 pos + 6 rot, same convention as eval_mpjpe_batch.py


def yaw_of_rot6d(rot6d):
    R = _rot6d_to_mat(rot6d.astype(np.float32))
    forward = R[:, 0]  # body-forward basis vector, world frame
    return float(np.arctan2(forward[1], forward[0]))  # yaw in XY-plane


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def episode_metrics(path: Path):
    with h5py.File(path, "r") as f:
        states = f["observation.state"][()].astype(np.float64)
    T = states.shape[0]
    waist_pos = states[:, 89:92]
    waist_rot6 = states[:, 92:98]
    left_pos = states[:, C.OUTPUT_LEFT_EEF[:3]]
    right_pos = states[:, C.OUTPUT_RIGHT_EEF[:3]]

    n = max(1, T // 10)
    yaw_start = np.mean([yaw_of_rot6d(waist_rot6[i]) for i in range(n)])
    yaw_end = np.mean([yaw_of_rot6d(waist_rot6[i]) for i in range(T - n, T)])
    yaw_change = wrap(yaw_end - yaw_start)

    waist_disp = waist_pos[-1] - waist_pos[0]
    left_disp = left_pos[-1] - left_pos[0]
    right_disp = right_pos[-1] - right_pos[0]

    return dict(
        T=T,
        yaw_change_deg=np.degrees(yaw_change),
        waist_net_disp=waist_disp,
        left_net_disp=left_disp,
        right_net_disp=right_disp,
        waist_pos_mean=waist_pos.mean(axis=0),
    )


def main():
    paths = sorted(
        DATA_DIR.glob("wholebody-*.hdf5"),
        key=lambda p: int(re.match(r"wholebody-(\d+)_", p.name).group(1)),
    )
    for p in paths:
        orig = int(re.match(r"wholebody-(\d+)_", p.name).group(1))
        try:
            m = episode_metrics(p)
        except Exception as e:
            print(f"{orig:3d} skip: {e}")
            continue
        wd = m["waist_net_disp"]
        print(
            f"{orig:3d}  yaw_change={m['yaw_change_deg']:7.1f}deg  "
            f"waist_disp=[{wd[0]:6.2f},{wd[1]:6.2f},{wd[2]:6.2f}]  T={m['T']}"
        )


if __name__ == "__main__":
    main()
