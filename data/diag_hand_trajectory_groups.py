#!/usr/bin/env python3
"""Group episodes by recording (chronological) order and compute hand-trajectory
statistics per group, to check whether table placement / approach direction
shifts across the dataset (e.g. every ~20-30 episodes), which would explain
why train/test splits and even some "seen" episodes behave very differently.
"""
import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hdt"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_mpjpe_batch import _rot6d_to_mat
import hdt.constants as C

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "all_episodes"
INDEX_CSV = DATA_DIR / "index.csv"
GROUP_SIZE = 20

LEFT_TIP_LOCAL = np.array([1, 2])  # thumb_tip, index_tip indices within the 6-row keypoint array


def pose_mat(pos, rot6d):
    m = np.eye(4, dtype=np.float64)
    m[:3, 3] = pos
    m[:3, :3] = _rot6d_to_mat(rot6d.astype(np.float32))
    return m


def episode_metrics(path: Path):
    with h5py.File(path, "r") as f:
        states = f["observation.state"][()].astype(np.float64)

    T = states.shape[0]
    head_pos = states[:, C.OUTPUT_HEAD_EEF[:3]]
    head_rot6 = states[:, C.OUTPUT_HEAD_EEF[3:9]]
    left_pos = states[:, C.OUTPUT_LEFT_EEF[:3]]
    left_rot6 = states[:, C.OUTPUT_LEFT_EEF[3:9]]
    right_pos = states[:, C.OUTPUT_RIGHT_EEF[:3]]
    right_rot6 = states[:, C.OUTPUT_RIGHT_EEF[3:9]]
    left_kp = states[:, C.OUTPUT_LEFT_KEYPOINTS].reshape(T, 6, 3)
    right_kp = states[:, C.OUTPUT_RIGHT_KEYPOINTS].reshape(T, 6, 3)

    # hand position relative to head frame (translation+rotation)
    left_head_frame = np.zeros((T, 3))
    right_head_frame = np.zeros((T, 3))
    for t in range(T):
        hm_inv = np.linalg.inv(pose_mat(head_pos[t], head_rot6[t]))
        lp_h = hm_inv @ np.array([*left_pos[t], 1.0])
        rp_h = hm_inv @ np.array([*right_pos[t], 1.0])
        left_head_frame[t] = lp_h[:3]
        right_head_frame[t] = rp_h[:3]

    def path_len(p):
        return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())

    left_path = path_len(left_head_frame)
    right_path = path_len(right_head_frame)
    dominant = "right" if right_path >= left_path else "left"
    dom_pos = right_head_frame if dominant == "right" else left_head_frame
    dom_kp = right_kp if dominant == "right" else left_kp
    dom_path = max(left_path, right_path)

    aperture = np.linalg.norm(dom_kp[:, 1] - dom_kp[:, 2], axis=1)  # thumb-index

    disp_vec_mean = np.diff(dom_pos, axis=0).mean(axis=0)
    net_disp = float(np.linalg.norm(dom_pos[-1] - dom_pos[0]))
    bbox = dom_pos.max(axis=0) - dom_pos.min(axis=0)
    head_std = float(np.linalg.norm(head_pos.std(axis=0)))

    return dict(
        T=T,
        dominant=dominant,
        path_len=dom_path,
        mean_pos_head=dom_pos.mean(axis=0),
        bbox=bbox,
        head_std=head_std,
        move_dir_mean=disp_vec_mean,
        net_disp=net_disp,
        aperture_min=float(aperture.min()),
        aperture_p90=float(np.percentile(aperture, 90)),
    )


def main():
    import re
    paths = sorted(
        DATA_DIR.glob("wholebody-*.hdf5"),
        key=lambda p: int(re.match(r"wholebody-(\d+)_", p.name).group(1)),
    )

    all_metrics = []
    for path in paths:
        orig = int(re.match(r"wholebody-(\d+)_", path.name).group(1))
        try:
            m = episode_metrics(path)
        except Exception as e:
            print(f"skip {path.name}: {e}")
            continue
        m["new_id"] = orig
        m["orig"] = orig
        all_metrics.append(m)

    n = len(all_metrics)
    print(f"loaded {n} episodes\n")

    for start in range(0, n, GROUP_SIZE):
        chunk = all_metrics[start:start + GROUP_SIZE]
        ids = [m["new_id"] for m in chunk]
        dom_right = sum(1 for m in chunk if m["dominant"] == "right")
        path_lens = np.array([m["path_len"] for m in chunk])
        mean_pos = np.stack([m["mean_pos_head"] for m in chunk]).mean(axis=0)
        bbox = np.stack([m["bbox"] for m in chunk]).mean(axis=0)
        head_std = np.array([m["head_std"] for m in chunk]).mean()
        move_dir = np.stack([m["move_dir_mean"] for m in chunk]).mean(axis=0)
        T_mean = np.array([m["T"] for m in chunk]).mean()
        net_disp = np.array([m["net_disp"] for m in chunk]).mean()
        ap_min = np.array([m["aperture_min"] for m in chunk]).mean()
        ap_p90 = np.array([m["aperture_p90"] for m in chunk]).mean()

        print(f"== new_id {ids[0]}-{ids[-1]} (n={len(chunk)}) ==")
        print(f"  主用手(right): {dom_right}/{len(chunk)}")
        print(f"  手总路径长度(mean): {path_lens.mean():.3f} m")
        print(f"  手平均位置(head系): [{mean_pos[0]:.3f}, {mean_pos[1]:.3f}, {mean_pos[2]:.3f}]")
        print(f"  workspace bbox(mean): [{bbox[0]:.3f}, {bbox[1]:.3f}, {bbox[2]:.3f}]")
        print(f"  头部 pos std: {head_std:.4f}")
        print(f"  主手移动方向(均值): [{move_dir[0]:.4f}, {move_dir[1]:.4f}, {move_dir[2]:.4f}]")
        print(f"  时长(frames): {T_mean:.0f}  净位移: {net_disp:.3f} m")
        print(f"  抓取张口 min/p90 (mm): {ap_min*1000:.1f} / {ap_p90*1000:.1f}")
        print()


if __name__ == "__main__":
    main()
