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


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT,
    "DATASETS",
    "UnifoLM_WBT",
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
)
FPS = 30.0


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


def inspire_sdk_to_q(hand6: np.ndarray) -> np.ndarray:
    """SDK order: [index, middle, ring, little, thumb_pitch, thumb_yaw]."""
    idx, mid, rng, little, thumb_pitch, thumb_yaw = hand6

    idx_prox = idx * 1.7
    mid_prox = mid * 1.7
    rng_prox = rng * 1.7
    pnk_prox = little * 1.7
    yaw = -0.1 + thumb_yaw * 1.4
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
            "Maniptrans_YS",
            "maniptrans_envs",
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
) -> np.ndarray:
    q = sdk_to_q(hand6)
    pin.framesForwardKinematics(model, data, q)

    tips = np.zeros((5, 3), dtype=np.float32)
    for i, fid in enumerate(tip_frame_ids):
        tips[i] = data.oMf[fid].translation.astype(np.float32)
    return tips


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
        choices=["wrist"],
        help="kept for command compatibility; output is always wrist-relative",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="limit number of episodes (default: all)",
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
    sdk_to_q = config["sdk_to_q"]

    parquet_files = sorted(glob(os.path.join(dataset, "data", "*", "*.parquet")))
    print(f"dataset: {dataset}")
    print(f"hand_type: {hand_type}")
    print("finger_frame: wrist")
    print(f"out_dir: {out_dir}")
    print(f"Found {len(parquet_files)} parquet files")
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {dataset}/data/*/*.parquet")

    df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    episode_ids = sorted(df["episode_index"].unique())
    if args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]
    print(f"Total episodes: {len(episode_ids)}")

    for ep_idx in episode_ids:
        out_path = os.path.join(out_dir, f"ep_{ep_idx:04d}.pkl")
        ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
        t_count = len(ep_df)

        left = np.zeros((t_count, 5, 3), dtype=np.float32)
        right = np.zeros((t_count, 5, 3), dtype=np.float32)

        for t, row in ep_df.iterrows():
            hand = np.array(row["observation.state.hand_state"], dtype=np.float64)
            left[t] = compute_fingertips_wrist(hand[0:6], model_l, data_l, tips_l, sdk_to_q)
            right[t] = compute_fingertips_wrist(hand[6:12], model_r, data_r, tips_r, sdk_to_q)

        out = {
            "fps": FPS,
            "hand_type": hand_type,
            "finger_frame": "wrist",
            "finger_pos_left": left,
            "finger_pos_right": right,
        }
        with open(out_path, "wb") as f:
            pickle.dump(out, f)

        print(f"  ep {ep_idx:04d}: T={t_count} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
