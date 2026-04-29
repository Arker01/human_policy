"""
Compute finger keypoint positions (fingertips in world frame) for each episode.
Uses Brainco Revo2 hand URDF + pinocchio FK.

ee_state  [12]: left(pos3 + euler_xyz3) + right(pos3 + euler_xyz3)
hand_state[12]: left[thumb_open, thumb_tilt, index, middle, ring, little]
               + right[same order], all in [0, 1]

Output per episode pkl:
{
    'fps': float,
    'finger_pos_left':  np.ndarray (T, 5, 3),  # [thumb, index, middle, ring, little]
    'finger_pos_right': np.ndarray (T, 5, 3),
}
"""

import os
import pickle
import numpy as np
import pandas as pd
import pinocchio as pin
from scipy.spatial.transform import Rotation as R
from glob import glob

# ── paths ──────────────────────────────────────────────────────────────────────
URDF_LEFT  = "/home/ubuntu/DATA1/shengyin/humanoid/revo2_description/urdf/revo2_left_hand.urdf"
URDF_RIGHT = "/home/ubuntu/DATA1/shengyin/humanoid/revo2_description/urdf/revo2_right_hand.urdf"
DATA_DIR   = "/home/ubuntu/DATA1/shengyin/humanoid/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher"
OUT_DIR    = os.path.join(DATA_DIR, "finger_keypoints")
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 30.0

# ── load pinocchio models ──────────────────────────────────────────────────────
model_l = pin.buildModelFromUrdf(URDF_LEFT)
data_l  = model_l.createData()
model_r = pin.buildModelFromUrdf(URDF_RIGHT)
data_r  = model_r.createData()

# Fingertip frame IDs (verified from model.frames)
# order: [thumb, index, middle, ring, little]
TIP_FRAMES_LEFT  = [
    model_l.getFrameId("left_thumb_tip_link"),
    model_l.getFrameId("left_index_tip_link"),
    model_l.getFrameId("left_middle_tip_link"),
    model_l.getFrameId("left_ring_tip_link"),
    model_l.getFrameId("left_pinky_tip_link"),
]
TIP_FRAMES_RIGHT = [
    model_r.getFrameId("right_thumb_tip_link"),
    model_r.getFrameId("right_index_tip_link"),
    model_r.getFrameId("right_middle_tip_link"),
    model_r.getFrameId("right_ring_tip_link"),
    model_r.getFrameId("right_pinky_tip_link"),
]

# ── joint order (from pinocchio, verified):
# left:  [index_prox, index_dist, middle_prox, middle_dist,
#          pinky_prox, pinky_dist, ring_prox,  ring_dist,
#          thumb_metacarpal, thumb_prox, thumb_dist]
# SDK hand_state per hand: [thumb_open, thumb_tilt, index, middle, ring, little]
# Joint limits (upper rad): thumb_metacarpal=1.57, thumb_prox/dist=1.03, others_prox=1.41
# Mimic multipliers: thumb_dist=1.0×prox, finger_dist=1.155×prox

def sdk_to_q(hand6):
    """Convert 6 SDK [0,1] values to 11-dim pinocchio q vector."""
    thumb_open, thumb_tilt, idx, mid, rng, little = hand6

    thumb_meta = thumb_tilt * 1.57
    thumb_prox = thumb_open * 1.03
    thumb_dist = thumb_prox * 1.0       # mimic ×1.0

    idx_prox   = idx    * 1.41
    idx_dist   = idx_prox * 1.155       # mimic ×1.155

    mid_prox   = mid    * 1.41
    mid_dist   = mid_prox * 1.155

    rng_prox   = rng    * 1.41
    rng_dist   = rng_prox * 1.155

    pnk_prox   = little * 1.41
    pnk_dist   = pnk_prox * 1.155

    # pinocchio joint order:
    # 0 index_prox, 1 index_dist, 2 middle_prox, 3 middle_dist,
    # 4 pinky_prox, 5 pinky_dist, 6 ring_prox,  7 ring_dist,
    # 8 thumb_metacarpal, 9 thumb_prox, 10 thumb_dist
    return np.array([
        idx_prox, idx_dist,
        mid_prox, mid_dist,
        pnk_prox, pnk_dist,
        rng_prox, rng_dist,
        thumb_meta, thumb_prox, thumb_dist,
    ])


def ee_to_se3(pos3, euler_xyz3):
    """pos3 (xyz) + euler_xyz3 (radians) → pin.SE3"""
    rot = R.from_euler("xyz", euler_xyz3).as_matrix()
    return pin.SE3(rot, pos3)


def compute_fingertips(hand6, ee_pos, ee_euler, model, data, tip_frame_ids):
    """
    Returns (5, 3) fingertip positions in world frame.
    hand6: [0,1]^6 SDK values
    ee_pos: (3,) world-frame wrist position
    ee_euler: (3,) world-frame wrist Euler XYZ rotation (rad)
    """
    q = sdk_to_q(hand6)
    pin.framesForwardKinematics(model, data, q)

    wrist_se3 = ee_to_se3(ee_pos, ee_euler)

    tips = np.zeros((5, 3))
    for i, fid in enumerate(tip_frame_ids):
        tip_in_hand  = data.oMf[fid]
        tip_in_world = wrist_se3 * tip_in_hand
        tips[i] = tip_in_world.translation
    return tips


# ── load all parquet files ─────────────────────────────────────────────────────
parquet_files = sorted(glob(os.path.join(DATA_DIR, "data", "*", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files")

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

episode_ids = sorted(df["episode_index"].unique())
print(f"Total episodes: {len(episode_ids)}")

# ── process each episode ───────────────────────────────────────────────────────
for ep_idx in episode_ids:
    ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    T = len(ep_df)

    tips_left  = np.zeros((T, 5, 3))
    tips_right = np.zeros((T, 5, 3))

    for t, row in ep_df.iterrows():
        ee    = np.array(row["observation.state.ee_state"],  dtype=np.float64)
        hand  = np.array(row["observation.state.hand_state"], dtype=np.float64)

        left_pos,   left_euler  = ee[0:3],  ee[3:6]
        right_pos,  right_euler = ee[6:9],  ee[9:12]
        hand_left,  hand_right  = hand[0:6], hand[6:12]

        tips_left[t]  = compute_fingertips(hand_left,  left_pos,  left_euler,
                                           model_l, data_l, TIP_FRAMES_LEFT)
        tips_right[t] = compute_fingertips(hand_right, right_pos, right_euler,
                                           model_r, data_r, TIP_FRAMES_RIGHT)

    out = {
        "fps": FPS,
        "finger_pos_left":  tips_left,   # (T, 5, 3) [thumb,index,middle,ring,little]
        "finger_pos_right": tips_right,  # (T, 5, 3)
    }

    out_path = os.path.join(OUT_DIR, f"ep_{ep_idx:04d}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"  ep {ep_idx:04d}: T={T} → {out_path}")

print("Done.")
