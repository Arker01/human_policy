"""
Convert Inspire hand dataset → human_policy_lr processed HDF5 format.

Dataset: G1_WBT_Inspire_Collect_Clothes_MainCamOnly/G1_WB_Dex5_Collect_Clothes
Output:  G1_WBT_Inspire_Collect_Clothes_MainCamOnly/human_policy/{ep_idx}.hdf5

HDF5 structure (same as Brainco pipeline):
  observation.state  (T, 128) float32
  action             (T, 128) float32   ← state[t+1], last frame copies itself
  observation.image.left   (T, N) uint8  ← JPEG compressed from cam_0
  observation.image.right  (T, N) uint8  ← JPEG compressed from cam_1
"""

import os
import pickle
import subprocess
import numpy as np
import pandas as pd
import h5py
import cv2
from scipy.spatial.transform import Rotation as R
from glob import glob

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET = "/home/ubuntu/DATA1/shengyin/humanoid/DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/G1_WB_Dex5_Collect_Clothes"
HEAD_DIR   = os.path.join(DATASET, "head_pose_track")
FINGER_DIR = os.path.join(DATASET, "finger_keypoints")
DATA_DIR   = os.path.join(DATASET, "data")
VIDEO_L_DIR = os.path.join(DATASET, "videos/observation.images.cam_0/chunk-000")
VIDEO_R_DIR = os.path.join(DATASET, "videos/observation.images.cam_1/chunk-000")
META_EP     = os.path.join(DATASET, "meta/episodes")

# ── mode: "global" | "rel" | "real" ───────────────────────────────────────────
# "real": wrist pos from L/R_hand_base_link, wrist rot from left/right_wrist_yaw_link (PKL)
MODE       = "global"
PKL_DIR    = "/home/ubuntu/qingyaoxu/TWIST/data/track_dataset/unifolm_wbt_pkl"
PKL_PREFIX = os.path.basename(DATASET.rstrip("/"))   # G1_WB_Dex5_Collect_Clothes

OUT_DIR = "/home/ubuntu/DATA1/shengyin/humanoid/DATASETS/UnifoLM_WBT/G1_WBT_Inspire_Collect_Clothes_MainCamOnly/human_policy"
os.makedirs(OUT_DIR, exist_ok=True)

JPEG_QUALITY = 50
IMAGE_HW     = (480, 640)

# ── 128-dim indices (same layout as Brainco) ───────────────────────────────────
VEC_SIZE       = 128
IDX_HEAD_EEF   = np.arange(0,  9)
IDX_LEFT_KPTS  = np.arange(10, 28)
IDX_RIGHT_EEF  = np.arange(30, 39)
IDX_RIGHT_KPTS = np.arange(40, 58)
IDX_LEFT_EEF   = np.arange(80, 89)
# [100:126] qpos → zeros

# ── helpers ────────────────────────────────────────────────────────────────────

def rot6d(mat: np.ndarray) -> np.ndarray:
    m = np.asarray(mat, dtype=np.float32)
    return np.concatenate([m[:, 0], m[:, 1]])

def quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    return rot6d(R.from_quat(q).as_matrix())

def euler_xyz_to_rot6d(e: np.ndarray) -> np.ndarray:
    return rot6d(R.from_euler("xyz", e).as_matrix())

def euler_xyz_to_mat4(pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_euler("xyz", euler).as_matrix()
    mat[:3, 3]  = pos
    return mat

def inv_se3(mat: np.ndarray) -> np.ndarray:
    Ri = mat[:3, :3].T
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Ri
    Ti[:3, 3]  = -Ri @ mat[:3, 3]
    return Ti

def tips_in_wrist_frame(tips_world: np.ndarray, wrist_mat4: np.ndarray) -> np.ndarray:
    """tips_world (5,3) → (6,3): row0=[0,0,0] palm, rows1-5 fingertips in wrist frame."""
    inv_w = inv_se3(wrist_mat4)
    out = np.zeros((6, 3), dtype=np.float32)
    for i in range(5):
        hom = np.array([*tips_world[i], 1.0])
        out[i + 1] = (inv_w @ hom)[:3]
    return out

def build_vec(head_pos, head_quat_xyzw,
              left_pos, left_euler, right_pos, right_euler,
              left_tips_world, right_tips_world) -> np.ndarray:
    vec = np.zeros(VEC_SIZE, dtype=np.float32)
    vec[IDX_HEAD_EEF]   = np.concatenate([head_pos.astype(np.float32),
                                           quat_xyzw_to_rot6d(head_quat_xyzw)])
    vec[IDX_LEFT_EEF]   = np.concatenate([left_pos.astype(np.float32),
                                           euler_xyz_to_rot6d(left_euler)])
    vec[IDX_RIGHT_EEF]  = np.concatenate([right_pos.astype(np.float32),
                                           euler_xyz_to_rot6d(right_euler)])
    left_mat  = euler_xyz_to_mat4(left_pos,  left_euler)
    right_mat = euler_xyz_to_mat4(right_pos, right_euler)
    vec[IDX_LEFT_KPTS]  = tips_in_wrist_frame(left_tips_world,  left_mat).reshape(-1)
    vec[IDX_RIGHT_KPTS] = tips_in_wrist_frame(right_tips_world, right_mat).reshape(-1)
    return vec

def extract_frames(video_path: str, from_ts: float, n_frames: int) -> list:
    H, W = IMAGE_HW
    cmd = ["ffmpeg", "-loglevel", "error",
           "-ss", str(from_ts), "-i", video_path,
           "-vframes", str(n_frames),
           "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    raw = np.frombuffer(subprocess.run(cmd, stdout=subprocess.PIPE).stdout, dtype=np.uint8)
    fs = H * W * 3
    frames = [raw[i * fs:(i + 1) * fs].reshape(H, W, 3) for i in range(len(raw) // fs)]
    pad = frames[-1] if frames else np.zeros((H, W, 3), dtype=np.uint8)
    while len(frames) < n_frames:
        frames.append(pad)
    return frames

def write_compressed_images(frames: list, key: str, hf: h5py.File):
    encoded = []
    for fr in frames:
        _, buf = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        encoded.append(buf.flatten())
    max_len = max(len(e) for e in encoded)
    ds = hf.create_dataset(key, (len(encoded), max_len), dtype=np.uint8)
    for i, e in enumerate(encoded):
        ds[i, :len(e)] = e

# ── load parquet data ──────────────────────────────────────────────────────────
parquet_files = sorted(glob(os.path.join(DATA_DIR, "*", "*.parquet")))
print(f"Found {len(parquet_files)} parquet files")

df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
df_all = df_all.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

# load episode meta (may be chunked parquet)
meta_files = sorted(glob(os.path.join(META_EP, "*", "*.parquet")))
meta_ep = pd.concat([pd.read_parquet(f) for f in meta_files], ignore_index=True)
meta_ep = meta_ep.set_index("episode_index")

episode_ids = sorted(df_all["episode_index"].unique())
print(f"Total episodes: {len(episode_ids)}")

# ── process each episode ───────────────────────────────────────────────────────
for ep_idx in episode_ids:
    out_path = os.path.join(OUT_DIR, f"{ep_idx}.hdf5")
    if os.path.exists(out_path):
        print(f"  ep {ep_idx:04d}: skip (exists)")
        continue

    ep_df   = df_all[df_all["episode_index"] == ep_idx].reset_index(drop=True)
    T       = len(ep_df)
    ep_meta = meta_ep.loc[ep_idx]

    # head & finger keypoints
    with open(os.path.join(HEAD_DIR,   f"ep_{ep_idx:04d}.pkl"), "rb") as f:
        head_data   = pickle.load(f)
    with open(os.path.join(FINGER_DIR, f"ep_{ep_idx:04d}.pkl"), "rb") as f:
        finger_data = pickle.load(f)

    head_pos_w  = head_data["head_pos_w"]    # (T, 3)
    head_quat_w = head_data["head_quat_w"]   # (T, 4) xyzw
    tips_left   = finger_data["finger_pos_left"]   # (T, 5, 3)
    tips_right  = finger_data["finger_pos_right"]  # (T, 5, 3)

    # load body PKL for "real" mode
    if MODE == "real":
        pkl_path = os.path.join(PKL_DIR, f"{PKL_PREFIX}_ep_{ep_idx:04d}.pkl")
        with open(pkl_path, "rb") as f:
            body_data = pickle.load(f)
        body_list   = body_data["link_body_list"]
        body_pos_w  = np.array(body_data["body_pos_w"])   # (T, N, 3)
        body_quat_w = np.array(body_data["body_quat_w"])  # (T, N, 4) xyzw
        lhb_idx = body_list.index("L_hand_base_link")
        rhb_idx = body_list.index("R_hand_base_link")
        lwy_idx = body_list.index("left_wrist_yaw_link")
        rwy_idx = body_list.index("right_wrist_yaw_link")

    # build state vectors
    states = np.zeros((T, VEC_SIZE), dtype=np.float32)
    for t in range(T):
        ee = np.array(ep_df.at[t, "observation.state.ee_state"])
        if MODE == "real":
            left_pos    = body_pos_w[t, lhb_idx].astype(np.float32)
            right_pos   = body_pos_w[t, rhb_idx].astype(np.float32)
            left_euler  = R.from_quat(body_quat_w[t, lwy_idx]).as_euler("xyz").astype(np.float32)
            right_euler = R.from_quat(body_quat_w[t, rwy_idx]).as_euler("xyz").astype(np.float32)
            # finger_pos 是 head-relative，转到世界坐标系
            R_head = R.from_quat(head_quat_w[t]).as_matrix()
            tips_left_t  = (R_head @ tips_left[t].T).T  + head_pos_w[t]
            tips_right_t = (R_head @ tips_right[t].T).T + head_pos_w[t]
        else:  # global
            left_pos,  left_euler  = ee[0:3], ee[3:6]
            right_pos, right_euler = ee[6:9], ee[9:12]
            tips_left_t  = tips_left[t]
            tips_right_t = tips_right[t]
        states[t] = build_vec(
            head_pos=head_pos_w[t],        head_quat_xyzw=head_quat_w[t],
            left_pos=left_pos,             left_euler=left_euler,
            right_pos=right_pos,           right_euler=right_euler,
            left_tips_world=tips_left_t,   right_tips_world=tips_right_t,
        )

    actions = np.zeros_like(states)
    actions[:-1] = states[1:]
    actions[-1]  = states[-1]

    # extract video frames
    l_file  = int(ep_meta["videos/observation.images.cam_0/file_index"])
    r_file  = int(ep_meta["videos/observation.images.cam_1/file_index"])
    l_ts    = float(ep_meta["videos/observation.images.cam_0/from_timestamp"])
    r_ts    = float(ep_meta["videos/observation.images.cam_1/from_timestamp"])

    l_video = os.path.join(VIDEO_L_DIR, f"file-{l_file:03d}.mp4")
    r_video = os.path.join(VIDEO_R_DIR, f"file-{r_file:03d}.mp4")

    frames_l = extract_frames(l_video, l_ts, T)
    frames_r = extract_frames(r_video, r_ts, T)

    # write HDF5
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("observation.state", data=states,  dtype=np.float32)
        hf.create_dataset("action",            data=actions, dtype=np.float32)
        write_compressed_images(frames_l, "observation.image.left",  hf)
        write_compressed_images(frames_r, "observation.image.right", hf)

    print(f"  ep {ep_idx:04d}: T={T} → {out_path}")

print("Done.")
