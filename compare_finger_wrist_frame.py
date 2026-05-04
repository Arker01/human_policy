import argparse
import os
import pickle

import h5py
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT,
    "DATASETS",
    "UnifoLM_WBT",
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
)

IDX_LEFT_KPTS = np.arange(10, 28)
IDX_RIGHT_KPTS = np.arange(40, 58)


def pack_wrist_tips(tips_wrist: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 3), dtype=np.float32)
    out[1:] = tips_wrist.astype(np.float32)
    return out


def summarize(name: str, direct: np.ndarray, reference: np.ndarray):
    diff = direct - reference
    print(f"{name}:")
    print(f"  max_abs: {np.max(np.abs(diff)):.8f}")
    print(f"  rms:     {np.sqrt(np.mean(diff ** 2)):.8f}")
    print(f"  mean:    {diff.mean(axis=0)}")
    print(f"  direct first 6x3:\n{direct}")
    print(f"  ref first 6x3:\n{reference}")


def fit_all_frames(name: str, direct_tips: np.ndarray, reference_kpts: np.ndarray):
    """Fit fixed transforms with all fingertip rows, excluding the palm origin."""
    x = direct_tips.reshape(-1, 3)
    y = reference_kpts[:, 1:, :].reshape(-1, 3)

    xa = np.c_[x, np.ones(len(x))]
    affine, *_ = np.linalg.lstsq(xa, y, rcond=None)
    a = affine[:3]
    b = affine[3]
    pred = x @ a + b

    xc = x - x.mean(axis=0)
    yc = y - y.mean(axis=0)
    u, _, vt = np.linalg.svd(xc.T @ yc)
    q_reflect = u @ vt
    b_reflect = y.mean(axis=0) - x.mean(axis=0) @ q_reflect
    pred_reflect = x @ q_reflect + b_reflect

    print(f"{name} all-frame fit:")
    print(f"  affine rms/max: {np.sqrt(np.mean((pred - y) ** 2)):.8f} / {np.max(np.abs(pred - y)):.8f}")
    print(f"  affine det:     {np.linalg.det(a):.8f}")
    print(f"  affine A:\n{a}")
    print(f"  affine b:       {b}")
    print(f"  orth det:       {np.linalg.det(q_reflect):.8f}")
    print(f"  orth rms/max:   {np.sqrt(np.mean((pred_reflect - y) ** 2)):.8f} / {np.max(np.abs(pred_reflect - y)):.8f}")
    print(f"  orth Q:\n{q_reflect}")
    print(f"  orth b:         {b_reflect}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare direct FK wrist-frame finger pkl against an existing real-mode HDF5."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--finger-dir", default=None,
                        help="default: <dataset>/finger_keypoints, falling back to <dataset>/finger_keypoints_wrist")
    parser.add_argument("--hdf5-dir", default=None,
                        help="default: <dataset>/../human_policy_real")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    if args.finger_dir:
        finger_dir = args.finger_dir
    else:
        finger_dir = os.path.join(dataset, "finger_keypoints")
        if not os.path.isdir(finger_dir):
            finger_dir = os.path.join(dataset, "finger_keypoints_wrist")
    hdf5_dir = args.hdf5_dir or os.path.join(os.path.dirname(dataset.rstrip("/")), "human_policy_real")

    finger_path = os.path.join(finger_dir, f"ep_{args.episode:04d}.pkl")
    hdf5_path = os.path.join(hdf5_dir, f"{args.episode}.hdf5")

    with open(finger_path, "rb") as f:
        finger_data = pickle.load(f)
    finger_frame = finger_data.get("finger_frame", "unknown")
    if finger_frame != "wrist":
        print(f"warning: {finger_path} finger_frame={finger_frame!r}, expected 'wrist'")

    left_direct = pack_wrist_tips(finger_data["finger_pos_left"][args.frame])
    right_direct = pack_wrist_tips(finger_data["finger_pos_right"][args.frame])

    with h5py.File(hdf5_path, "r") as hf:
        states = hf["observation.state"][:]
    state = states[args.frame]
    left_ref_all = states[:, IDX_LEFT_KPTS].reshape(-1, 6, 3)
    right_ref_all = states[:, IDX_RIGHT_KPTS].reshape(-1, 6, 3)
    left_ref = left_ref_all[args.frame]
    right_ref = right_ref_all[args.frame]

    print(f"finger: {finger_path}")
    print(f"hdf5:   {hdf5_path}")
    print(f"frame:  {args.frame}")
    summarize("left", left_direct, left_ref)
    summarize("right", right_direct, right_ref)
    fit_all_frames("left", finger_data["finger_pos_left"], left_ref_all)
    fit_all_frames("right", finger_data["finger_pos_right"], right_ref_all)


if __name__ == "__main__":
    main()
