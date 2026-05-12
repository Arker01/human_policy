"""
Compute wrist-frame fingertip keypoints for Brainco/Revo2 and Inspire datasets.

The dataset path selects the hand type automatically:
  * paths containing "brainco"  -> Brainco Revo2
  * paths containing "inspire"  -> Inspire hand

Output per episode pkl:
{
    "fps": 30.0,
    "hand_type": "brainco" or "inspire",
    "finger_frame": "wrist",
    "finger_pos_left":  np.ndarray (T, 5, 3),
    "finger_pos_right": np.ndarray (T, 5, 3),
}

The five fingertips are ordered as [thumb, index, middle, ring, little].
Values are local to the corresponding wrist/hand URDF root frame; convert_to_hdf5
can pack them directly into the 128-dim hand keypoint slots.
"""

import argparse
import os
import pickle
from glob import glob
from typing import Callable

import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation as R


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT,
    "DATASETS",
    "UnifoLM_WBT",
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
)
FPS = 30.0
DEFAULT_XQY_DIR = os.path.join(PROJECT_ROOT, "DATASETS", "XQY_PKL")


def detect_hand_type(dataset: str) -> str:
    dataset_lower = dataset.lower()
    if "brainco" in dataset_lower:
        return "brainco"
    if "inspire" in dataset_lower:
        return "inspire"
    raise ValueError(
        "Could not infer hand type from --dataset. "
        "Dataset path must contain 'Brainco' or 'Inspire', or pass --hand-type."
    )


def brainco_sdk_to_q(hand6: np.ndarray) -> np.ndarray:
    """SDK order: [thumb_open, thumb_tilt, index, middle, ring, little]."""
    thumb_open, thumb_tilt, idx, mid, rng, little = hand6

    thumb_meta = thumb_tilt * 1.57
    thumb_prox = thumb_open * 1.03
    thumb_dist = thumb_prox

    idx_prox = idx * 1.41
    mid_prox = mid * 1.41
    rng_prox = rng * 1.41
    pnk_prox = little * 1.41

    return np.array(
        [
            idx_prox,
            idx_prox * 1.155,
            mid_prox,
            mid_prox * 1.155,
            pnk_prox,
            pnk_prox * 1.155,
            rng_prox,
            rng_prox * 1.155,
            thumb_meta,
            thumb_prox,
            thumb_dist,
        ],
        dtype=np.float64,
    )


def inspire_sdk_to_q(
    hand6: np.ndarray,
    thumb_mode: str = "normal",
    thumb_yaw_sign: float = 1.0,
) -> np.ndarray:
    """SDK order: [index, middle, ring, little, thumb_pitch, thumb_yaw]."""
    idx, mid, rng, little, thumb_pitch, thumb_yaw = hand6
    if thumb_mode == "swap":
        thumb_pitch, thumb_yaw = thumb_yaw, thumb_pitch
    elif thumb_mode != "normal":
        raise ValueError(f"Unsupported inspire thumb_mode={thumb_mode!r}")

    idx_prox = idx * 1.7
    mid_prox = mid * 1.7
    rng_prox = rng * 1.7
    pnk_prox = little * 1.7
    yaw = thumb_yaw_sign * (-0.1 + thumb_yaw * 1.4)
    pitch = thumb_pitch * 0.5

    return np.array(
        [
            idx_prox,
            idx_prox,
            mid_prox,
            mid_prox,
            pnk_prox,
            pnk_prox,
            rng_prox,
            rng_prox,
            yaw,
            pitch,
            pitch * 1.6,
            pitch * 2.4,
        ],
        dtype=np.float64,
    )


def reverse_inspire_open_close(hand6: np.ndarray, mode: str) -> np.ndarray:
    """Reverse Inspire open/close channels while preserving selected controls."""
    out = np.asarray(hand6, dtype=np.float64).copy()
    if mode == "none":
        return out
    if mode == "fingers_only":
        out[:4] = 1.0 - out[:4]
    elif mode == "fingers_thumb":
        out[:5] = 1.0 - out[:5]
    else:
        raise ValueError(f"Unsupported inspire reverse mode: {mode!r}")
    return out


def get_hand_config(hand_type: str) -> dict:
    if hand_type == "brainco":
        revo2_dir = os.path.join(PROJECT_ROOT, "DATASETS", "revo2_description")
        return {
            "left_urdf": os.path.join(revo2_dir, "urdf", "revo2_left_hand.urdf"),
            "right_urdf": os.path.join(revo2_dir, "urdf", "revo2_right_hand.urdf"),
            "left_tip_frames": [
                "left_thumb_tip_link",
                "left_index_tip_link",
                "left_middle_tip_link",
                "left_ring_tip_link",
                "left_pinky_tip_link",
            ],
            "right_tip_frames": [
                "right_thumb_tip_link",
                "right_index_tip_link",
                "right_middle_tip_link",
                "right_ring_tip_link",
                "right_pinky_tip_link",
            ],
            "sdk_to_q": brainco_sdk_to_q,
        }
    if hand_type == "inspire":
        inspire_dir = os.path.join(
            PROJECT_ROOT,
            "DATASETS",
            "assets",
            "inspire_hand",
        )
        return {
            "left_urdf": os.path.join(inspire_dir, "inspire_hand_left.urdf"),
            "right_urdf": os.path.join(inspire_dir, "inspire_hand_right.urdf"),
            "left_tip_frames": [
                "L_thumb_tip",
                "L_index_tip",
                "L_middle_tip",
                "L_ring_tip",
                "L_pinky_tip",
            ],
            "right_tip_frames": [
                "R_thumb_tip",
                "R_index_tip",
                "R_middle_tip",
                "R_ring_tip",
                "R_pinky_tip",
            ],
            "sdk_to_q": inspire_sdk_to_q,
        }
    raise ValueError(f"Unsupported hand_type={hand_type!r}")


def make_sdk_to_q(
    hand_type: str,
    inspire_thumb_mode: str,
    inspire_thumb_yaw_sign: str,
) -> Callable[[np.ndarray], np.ndarray]:
    if hand_type == "inspire":
        sign = -1.0 if inspire_thumb_yaw_sign == "flipped" else 1.0
        return lambda hand6: inspire_sdk_to_q(hand6, inspire_thumb_mode, sign)
    return get_hand_config(hand_type)["sdk_to_q"]


def apply_local_frame_correction(tips: np.ndarray, hand_type: str, correction: str) -> np.ndarray:
    if correction == "none":
        return tips
    if hand_type == "brainco" and correction == "rx90":
        # Revo2 standalone URDF fingers extend mostly along +Z, while the policy
        # wrist-local convention used by Inspire extends fingers mostly along -Y.
        rot = R.from_euler("x", 90.0, degrees=True)
        return rot.apply(np.asarray(tips, dtype=np.float64)).astype(np.float32)
    if hand_type == "brainco" and correction == "rx90_ry180":
        # Same as rx90, then flip the palm normal around the finger-forward axis.
        rot = R.from_euler("y", 180.0, degrees=True) * R.from_euler("x", 90.0, degrees=True)
        return rot.apply(np.asarray(tips, dtype=np.float64)).astype(np.float32)
    raise ValueError(f"Unsupported local frame correction {correction!r} for hand_type={hand_type!r}")


def load_model(urdf_path: str, tip_frame_names: list[str]):
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    tip_frame_ids = [model.getFrameId(name) for name in tip_frame_names]
    return model, data, tip_frame_ids


def compute_fingertips_wrist(
    hand6: np.ndarray,
    model,
    data,
    tip_frame_ids: list[int],
    sdk_to_q: Callable[[np.ndarray], np.ndarray],
    zero_hand: bool = False,
) -> np.ndarray:
    q = np.zeros(model.nq, dtype=np.float64) if zero_hand else sdk_to_q(hand6)
    pin.framesForwardKinematics(model, data, q)

    tips = np.zeros((5, 3), dtype=np.float32)
    for i, fid in enumerate(tip_frame_ids):
        tips[i] = data.oMf[fid].translation.astype(np.float32)
    return tips


def xqy_path_for_episode(xqy_dir: str, xqy_prefix: str, ep_idx: int) -> str:
    path = os.path.join(xqy_dir, f"{xqy_prefix}_ep_{ep_idx:04d}.pkl")
    if os.path.exists(path):
        return path
    matches = sorted(glob(os.path.join(xqy_dir, f"*{xqy_prefix}*_ep_{ep_idx:04d}.pkl")))
    if matches:
        return matches[0]
    raise FileNotFoundError(path)


def body_index(body_list: list[str], *names: str) -> int:
    for name in names:
        if name in body_list:
            return body_list.index(name)
    raise ValueError(f"None of these body names found: {names}")


def get_optional_array(data: dict, *names: str):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    return None


class XQYAccess:
    def __init__(self, data: dict):
        self.data = data
        self.body_list = data["link_body_list"]
        self.pos_w = np.asarray(data["body_pos_w"])
        self.quat_w = np.asarray(data["body_quat_w"])
        self.idx = {
            "lhb": body_index(self.body_list, "L_hand_base_link", "left_hand_base_link"),
            "rhb": body_index(self.body_list, "R_hand_base_link", "right_hand_base_link"),
        }

    def hand_base_pos(self, t: int, side: str) -> np.ndarray:
        return self.pos_w[t, self.idx["lhb" if side == "left" else "rhb"]]

    def direct_wrist_pos(self, t: int, side: str) -> np.ndarray:
        key_candidates = (
            ("left_wrist_pos", "l_wrist_pos", "wrist_pos_left")
            if side == "left"
            else ("right_wrist_pos", "r_wrist_pos", "wrist_pos_right")
        )
        arr = get_optional_array(self.data, *key_candidates)
        if arr is not None:
            return arr[t]
        return self.hand_base_pos(t, side)

    def direct_wrist_quat(self, t: int, side: str) -> np.ndarray:
        key = "left_wrist_rot" if side == "left" else "right_wrist_rot"
        if key in self.data:
            return np.asarray(self.data[key])[t]
        raise KeyError(f"{key} not found in XQY pkl")


def root_pose_from_robot_q(robot_q: np.ndarray) -> tuple[np.ndarray, R]:
    robot_q = np.asarray(robot_q, dtype=np.float64)
    if robot_q.shape[0] < 7:
        raise ValueError(f"robot_q_current must have at least 7 values, got {robot_q.shape}")
    root_pos = robot_q[0:3]
    quat_wxyz = robot_q[3:7]
    root_rot = R.from_quat([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
    return root_pos, root_rot


def wrist_pose_from_row(row, side: str, source: str, xqy: XQYAccess | None, t: int) -> tuple[np.ndarray, R]:
    if source == "real-ee":
        ee = np.asarray(row["observation.state.ee_state"], dtype=np.float64)
        robot_q = np.asarray(row["observation.state.robot_q_current"], dtype=np.float64)
        root_pos, root_rot = root_pose_from_robot_q(robot_q)
        if side == "left":
            local_pos = ee[0:3]
            local_euler = ee[3:6]
        else:
            local_pos = ee[6:9]
            local_euler = ee[9:12]
        return root_pos + root_rot.apply(local_pos), root_rot * R.from_euler("xyz", local_euler)
    if source == "xqy3_raw":
        if xqy is None:
            raise ValueError("xqy data is required for --wrist-pose-source xqy3_raw")
        return xqy.direct_wrist_pos(t, side), R.from_quat(xqy.direct_wrist_quat(t, side))
    raise ValueError(source)


def local_tips_to_world(tips_local: np.ndarray, wrist_pos: np.ndarray, wrist_rot: R) -> np.ndarray:
    return (wrist_pos[None, :] + wrist_rot.apply(np.asarray(tips_local, dtype=np.float64))).astype(np.float32)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute wrist-frame fingertip keypoints from hand_state."
    )
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help="path to the LeRobot dataset folder; hand type is inferred from this path",
    )
    parser.add_argument(
        "--hand-type",
        default="auto",
        choices=["auto", "brainco", "inspire"],
        help="override dataset-name based hand type detection",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="output folder (default: <dataset>/finger_keypoints)",
    )
    parser.add_argument(
        "--finger-frame",
        default="wrist",
        choices=["wrist", "world"],
        help="output fingertip frame; world uses --wrist-pose-source to place FK tips globally",
    )
    parser.add_argument(
        "--wrist-pose-source",
        default="real-ee",
        choices=["real-ee", "xqy3_raw"],
        help="wrist pose used when --finger-frame world",
    )
    parser.add_argument("--xqy-dir", default=DEFAULT_XQY_DIR)
    parser.add_argument("--xqy-prefix", default=None, help="default: basename(dataset)")
    parser.add_argument(
        "--head-link",
        default="head_mocap",
        help="kept for compatibility; not used by fingertip FK",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="limit number of episodes (default: all)",
    )
    parser.add_argument(
        "-reverse",
        "--reverse",
        action="store_true",
        help=(
            "for Inspire only: reverse open/close channels before FK "
            "(dims 0-4 become 1-value; thumb lateral tilt dim 5 is unchanged)"
        ),
    )
    parser.add_argument(
        "--zero-hand",
        action="store_true",
        help="ignore hand_state and run hand FK with all URDF hand DOF positions set to zero",
    )
    parser.add_argument(
        "--inspire-reverse-mode",
        default="fingers_thumb",
        choices=["none", "fingers_only", "fingers_thumb"],
        help="Inspire-only open/close reversal mode used when --reverse is set",
    )
    parser.add_argument(
        "--inspire-thumb-mode",
        default="normal",
        choices=["normal", "swap"],
        help="Inspire-only thumb channel interpretation",
    )
    parser.add_argument(
        "--inspire-thumb-yaw-sign",
        default="normal",
        choices=["normal", "flipped"],
        help="Inspire-only sign for the thumb lateral/yaw joint mapping",
    )
    parser.add_argument(
        "--local-frame-correction",
        default="none",
        choices=["none", "brainco_rx90", "brainco_rx90_ry180"],
        help="optional post-FK correction for fingertip local frame conventions",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = os.path.abspath(args.dataset)
    hand_type = detect_hand_type(dataset) if args.hand_type == "auto" else args.hand_type
    out_dir = args.out_dir or os.path.join(dataset, "finger_keypoints")
    os.makedirs(out_dir, exist_ok=True)

    config = get_hand_config(hand_type)
    model_l, data_l, tips_l = load_model(config["left_urdf"], config["left_tip_frames"])
    model_r, data_r, tips_r = load_model(config["right_urdf"], config["right_tip_frames"])
    sdk_to_q = make_sdk_to_q(hand_type, args.inspire_thumb_mode, args.inspire_thumb_yaw_sign)

    parquet_files = sorted(glob(os.path.join(dataset, "data", "*", "*.parquet")))
    print(f"dataset: {dataset}")
    print(f"hand_type: {hand_type}")
    print(f"finger_frame: {args.finger_frame}")
    if args.finger_frame == "world":
        print(f"wrist_pose_source: {args.wrist_pose_source}")
    print(f"reverse: {bool(args.reverse)}")
    if hand_type == "inspire":
        print(f"inspire_reverse_mode: {args.inspire_reverse_mode}")
        print(f"inspire_thumb_mode: {args.inspire_thumb_mode}")
        print(f"inspire_thumb_yaw_sign: {args.inspire_thumb_yaw_sign}")
    print(f"local_frame_correction: {args.local_frame_correction}")
    print(f"zero_hand: {bool(args.zero_hand)}")
    print(f"out_dir: {out_dir}")
    print(f"Found {len(parquet_files)} parquet files")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset}/data/*/*.parquet")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    xqy_dir = os.path.abspath(args.xqy_dir)
    xqy_prefix = args.xqy_prefix or os.path.basename(dataset.rstrip("/"))

    episode_ids = sorted(df["episode_index"].unique())
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    print(f"Total episodes: {len(episode_ids)}")

    for ep_idx in episode_ids:
        out_path = os.path.join(out_dir, f"ep_{ep_idx:04d}.pkl")
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        t_count = len(ep_df)
        xqy = None
        if args.finger_frame == "world" and args.wrist_pose_source == "xqy3_raw":
            with open(xqy_path_for_episode(xqy_dir, xqy_prefix, ep_idx), "rb") as f:
                xqy = XQYAccess(pickle.load(f))

        left = np.zeros((t_count, 5, 3), dtype=np.float32)
        right = np.zeros((t_count, 5, 3), dtype=np.float32)

        for t, row in ep_df.iterrows():
            hand = np.array(row["observation.state.hand_state"], dtype=np.float64)
            left_hand = hand[0:6]
            right_hand = hand[6:12]
            if args.reverse and hand_type == "inspire":
                left_hand = reverse_inspire_open_close(left_hand, args.inspire_reverse_mode)
                right_hand = reverse_inspire_open_close(right_hand, args.inspire_reverse_mode)
            left_tips = compute_fingertips_wrist(
                left_hand, model_l, data_l, tips_l, sdk_to_q, zero_hand=args.zero_hand
            )
            right_tips = compute_fingertips_wrist(
                right_hand, model_r, data_r, tips_r, sdk_to_q, zero_hand=args.zero_hand
            )
            if args.local_frame_correction == "brainco_rx90":
                correction = "rx90"
            elif args.local_frame_correction == "brainco_rx90_ry180":
                correction = "rx90_ry180"
            else:
                correction = "none"
            left_tips = apply_local_frame_correction(left_tips, hand_type, correction)
            right_tips = apply_local_frame_correction(right_tips, hand_type, correction)
            if args.finger_frame == "world":
                left_pos, left_rot = wrist_pose_from_row(row, "left", args.wrist_pose_source, xqy, t)
                right_pos, right_rot = wrist_pose_from_row(row, "right", args.wrist_pose_source, xqy, t)
                left_tips = local_tips_to_world(left_tips, left_pos, left_rot)
                right_tips = local_tips_to_world(right_tips, right_pos, right_rot)
            left[t] = left_tips
            right[t] = right_tips

        out = {
            "fps": FPS,
            "hand_type": hand_type,
            "finger_frame": args.finger_frame,
            "wrist_pose_source": args.wrist_pose_source if args.finger_frame == "world" else None,
            "reverse": bool(args.reverse and hand_type == "inspire"),
            "zero_hand": bool(args.zero_hand),
            "inspire_reverse_mode": args.inspire_reverse_mode if hand_type == "inspire" else None,
            "inspire_thumb_mode": args.inspire_thumb_mode if hand_type == "inspire" else None,
            "inspire_thumb_yaw_sign": args.inspire_thumb_yaw_sign if hand_type == "inspire" else None,
            "local_frame_correction": args.local_frame_correction,
            "finger_pos_left": left,
            "finger_pos_right": right,
        }
        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        print(f"  ep {ep_idx:04d}: T={t_count} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
