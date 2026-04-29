"""
Compute finger keypoint positions (fingertips in world frame) for Inspire hand datasets.

Dataset: G1_WBT_Inspire_* format (LeRobot parquet)
URDF:    Maniptrans_YS/maniptrans_envs/assets/inspire_hand/inspire_hand_{left,right}.urdf

hand_state per hand [0,1], order: [index, middle, ring, little, thumb_pitch, thumb_yaw]
  Convention: 0.0 = open, 1.0 = close

Joint limits (rad):
  index/middle/ring/little proximal : [0,   1.7]
  thumb_proximal_yaw (lateral tilt) : [-0.1, 1.3]
  thumb_proximal_pitch (open/close) : [0,   0.5]

Mimic joints (pinocchio does NOT enforce these → set manually):
  index/middle/ring/little intermediate = proximal × 1.0
  thumb_intermediate = thumb_pitch × 1.6
  thumb_distal       = thumb_pitch × 2.4

Pinocchio joint order (nq=12, same for left L_ and right R_):
  0  {s}_index_proximal_joint
  1  {s}_index_intermediate_joint     (mimic ×1.0)
  2  {s}_middle_proximal_joint
  3  {s}_middle_intermediate_joint    (mimic ×1.0)
  4  {s}_pinky_proximal_joint
  5  {s}_pinky_intermediate_joint     (mimic ×1.0)
  6  {s}_ring_proximal_joint
  7  {s}_ring_intermediate_joint      (mimic ×1.0)
  8  {s}_thumb_proximal_yaw_joint
  9  {s}_thumb_proximal_pitch_joint
  10 {s}_thumb_intermediate_joint     (mimic ×1.6)
  11 {s}_thumb_distal_joint           (mimic ×2.4)

Output per episode pkl:
{
    'fps': float,
    'finger_pos_left':  np.ndarray (T, 5, 3),  # [thumb, index, middle, ring, little], world frame
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
URDF_LEFT  = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/maniptrans_envs/assets/inspire_hand/inspire_hand_left.urdf"
URDF_RIGHT = "/home/ubuntu/DATA1/shengyin/humanoid/Maniptrans_YS/maniptrans_envs/assets/inspire_hand/inspire_hand_right.urdf"

# ── dataset path: modify this for each new Inspire dataset ────────────────────
DATASET = "/home/ubuntu/DATA1/shengyin/humanoid/DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/G1_WB_Dex5_Collect_Clothes"

OUT_DIR = os.path.join(DATASET, "finger_keypoints")
os.makedirs(OUT_DIR, exist_ok=True)

FPS = 30.0

# ── load pinocchio models ──────────────────────────────────────────────────────
model_l = pin.buildModelFromUrdf(URDF_LEFT)
data_l  = model_l.createData()
model_r = pin.buildModelFromUrdf(URDF_RIGHT)
data_r  = model_r.createData()

# Fingertip frame IDs (verified): [thumb, index, middle, ring, little]
TIP_FRAMES_LEFT = [
    model_l.getFrameId("L_thumb_tip"),
    model_l.getFrameId("L_index_tip"),
    model_l.getFrameId("L_middle_tip"),
    model_l.getFrameId("L_ring_tip"),
    model_l.getFrameId("L_pinky_tip"),
]
TIP_FRAMES_RIGHT = [
    model_r.getFrameId("R_thumb_tip"),
    model_r.getFrameId("R_index_tip"),
    model_r.getFrameId("R_middle_tip"),
    model_r.getFrameId("R_ring_tip"),
    model_r.getFrameId("R_pinky_tip"),
]


def sdk_to_q(hand6: np.ndarray) -> np.ndarray:
    """
    Convert 6 SDK [0,1] values → 12-dim pinocchio q (radians).

    SDK order: [index, middle, ring, little, thumb_pitch, thumb_yaw]
    Dataset convention: 0 = open, 1 = close
      → angle = lower + sdk × (upper − lower)

    Pinocchio joint order:
      0  index_proximal
      1  index_intermediate   (mimic index_proximal ×1.0)
      2  middle_proximal
      3  middle_intermediate  (mimic middle_proximal ×1.0)
      4  pinky_proximal
      5  pinky_intermediate   (mimic pinky_proximal ×1.0)
      6  ring_proximal
      7  ring_intermediate    (mimic ring_proximal ×1.0)
      8  thumb_proximal_yaw
      9  thumb_proximal_pitch
      10 thumb_intermediate   (mimic thumb_pitch ×1.6)
      11 thumb_distal         (mimic thumb_pitch ×2.4)
    """
    idx, mid, rng, little, thumb_pitch, thumb_yaw = hand6

    idx_prox  = idx         * 1.7
    mid_prox  = mid         * 1.7
    rng_prox  = rng         * 1.7
    pnk_prox  = little      * 1.7
    yaw       = -0.1 + thumb_yaw   * 1.4   # [-0.1, 1.3]
    pitch     =         thumb_pitch * 0.5   # [0,   0.5]

    # mimic joints
    idx_inter = idx_prox * 1.0
    mid_inter = mid_prox * 1.0
    rng_inter = rng_prox * 1.0
    pnk_inter = pnk_prox * 1.0
    thm_inter = pitch    * 1.6
    thm_dist  = pitch    * 2.4

    return np.array([
        idx_prox,  idx_inter,
        mid_prox,  mid_inter,
        pnk_prox,  pnk_inter,
        rng_prox,  rng_inter,
        yaw, pitch, thm_inter, thm_dist,
    ])


def compute_fingertips(hand6, ee_pos, ee_euler, model, data, tip_frame_ids):
    """
    Returns (5, 3) fingertip positions in world frame.
    hand6:    [0,1]^6 SDK values (Inspire order)
    ee_pos:   (3,) wrist position in world frame
    ee_euler: (3,) wrist Euler XYZ rotation (rad)
    """
    q = sdk_to_q(hand6)
    pin.framesForwardKinematics(model, data, q)

    rot = R.from_euler("xyz", ee_euler).as_matrix()
    wrist_se3 = pin.SE3(rot, ee_pos)

    tips = np.zeros((5, 3))
    for i, fid in enumerate(tip_frame_ids):
        tip_world = wrist_se3 * data.oMf[fid]
        tips[i] = tip_world.translation
    return tips


# ── load all parquet files ─────────────────────────────────────────────────────
parquet_files = sorted(glob(os.path.join(DATASET, "data", "*", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files")

df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

episode_ids = sorted(df["episode_index"].unique())
print(f"Total episodes: {len(episode_ids)}")

# ── process each episode ───────────────────────────────────────────────────────
for ep_idx in episode_ids:
    out_path = os.path.join(OUT_DIR, f"ep_{ep_idx:04d}.pkl")
    if os.path.exists(out_path):
        print(f"  ep {ep_idx:04d}: skip (exists)")
        continue

    ep_df = df[df["episode_index"] == ep_idx].reset_index(drop=True)
    T = len(ep_df)

    tips_left  = np.zeros((T, 5, 3))
    tips_right = np.zeros((T, 5, 3))

    for t in range(T):
        ee   = np.array(ep_df.at[t, "observation.state.ee_state"],  dtype=np.float64)
        hand = np.array(ep_df.at[t, "observation.state.hand_state"], dtype=np.float64)

        left_pos,  left_euler  = ee[0:3],  ee[3:6]
        right_pos, right_euler = ee[6:9],  ee[9:12]
        hand_left,  hand_right = hand[0:6], hand[6:12]

        tips_left[t]  = compute_fingertips(hand_left,  left_pos,  left_euler,
                                           model_l, data_l, TIP_FRAMES_LEFT)
        tips_right[t] = compute_fingertips(hand_right, right_pos, right_euler,
                                           model_r, data_r, TIP_FRAMES_RIGHT)

    out = {
        "fps": FPS,
        "finger_pos_left":  tips_left,   # (T, 5, 3) [thumb, index, middle, ring, little]
        "finger_pos_right": tips_right,
    }
    with open(out_path, "wb") as f:
        pickle.dump(out, f)

    print(f"  ep {ep_idx:04d}: T={T} → {out_path}")

print("Done.")
