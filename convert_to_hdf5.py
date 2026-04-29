"""
Convert 300 episodes → human_policy_lr processed HDF5 format.

Output: human_policy_lr/data/{episode_index}.hdf5  (one file per episode)

HDF5 structure:
  observation.state  (T, 128) float32
  action             (T, 128) float32
  observation.image.left   (T, max_jpeg_len) uint8   [JPEG compressed]
  observation.image.right  (T, max_jpeg_len) uint8   [JPEG compressed]

128-dim layout (from hdt/constants.py):
  [0:9]    head EEF     = pos3 + rot6d
  [10:28]  left kpts    = 6 × xyz  (palm=[0,0,0] in wrist frame + 5 fingertips in wrist frame)
  [30:39]  right EEF    = pos3 + rot6d
  [40:58]  right kpts   = 6 × xyz
  [80:89]  left EEF     = pos3 + rot6d
  [100:126] qpos        = 0
"""

import os
import pickle
import argparse
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import subprocess
import numpy as np
import pandas as pd
import h5py
import cv2
from scipy.spatial.transform import Rotation as R
from glob import glob

# ── args ───────────────────────────────────────────────────────────────────────
_BASE = "/home/ubuntu/DATA1/shengyin/humanoid/DATASETS/UnifoLM_WBT"
_DEFAULT_DATASET = os.path.join(_BASE, "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher")

parser = argparse.ArgumentParser(description="Convert episodes to HDF5")
parser.add_argument("--dataset",      default=_DEFAULT_DATASET,
                    help="path to the LeRobot dataset folder")
parser.add_argument("--head-dir",     default=os.path.join(_BASE, "head_pose_track"),
                    help="folder with ep_XXXX.pkl head-pose files")
parser.add_argument("--finger-dir",   default=os.path.join(_BASE, "finger_keypoints"),
                    help="folder with ep_XXXX.pkl finger-keypoint files")
parser.add_argument("--out-dir",      default=None,
                    help="output folder (default: <dataset>/../human_policy_<mode>)")
parser.add_argument("--mode",         default="global", choices=["global", "rel", "real"],
                    help="'global': ee_state in world frame; 'rel': ee_state in head frame; "
                         "'real': wrist pos/rot from PKL body_pos_w (L/R_hand_base_link + wrist_yaw_link)")
parser.add_argument("--pkl-dir",      default="/home/ubuntu/qingyaoxu/TWIST/data/track_dataset/unifolm_wbt_pkl",
                    help="folder with per-episode PKL files (used for --mode real)")
parser.add_argument("--max-episodes", type=int, default=None,
                    help="limit number of episodes (default: all)")
parser.add_argument("--jpeg-quality", type=int, default=50,
                    help="JPEG compression quality 1-100 (default: 50)")
args = parser.parse_args()

# ── paths ──────────────────────────────────────────────────────────────────────
DATASET     = args.dataset
HEAD_DIR    = args.head_dir
FINGER_DIR  = args.finger_dir
DATA_DIR    = os.path.join(DATASET, "data")
VIDEO_L_DIR = os.path.join(DATASET, "videos/observation.images.head_stereo_left/chunk-000")
VIDEO_R_DIR = os.path.join(DATASET, "videos/observation.images.head_stereo_right/chunk-000")
META_EP     = os.path.join(DATASET, "meta/episodes")

MODE         = args.mode
MAX_EPISODES = args.max_episodes
JPEG_QUALITY = args.jpeg_quality
PKL_DIR      = args.pkl_dir
PKL_PREFIX   = os.path.basename(DATASET.rstrip("/"))

OUT_DIR = args.out_dir or os.path.join(
    os.path.dirname(DATASET.rstrip("/")), f"human_policy_{MODE}")
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_HW = (480, 640)   # H, W from info.json

# ── 128-dim indices ─────────────────────────────────────────────────────────────
VEC_SIZE            = 128
IDX_HEAD_EEF        = np.arange(0,  9)    # pos3 + rot6d
IDX_LEFT_KPTS       = np.arange(10, 28)   # 6 × 3
IDX_RIGHT_EEF       = np.arange(30, 39)
IDX_RIGHT_KPTS      = np.arange(40, 58)
IDX_LEFT_EEF        = np.arange(80, 89)
# [100:126] left as zeros (per user instruction)

# ── helpers ────────────────────────────────────────────────────────────────────

def rot6d(mat3x3: np.ndarray) -> np.ndarray:
    """First two columns of rotation matrix, flattened → 6 floats."""
    m = np.asarray(mat3x3, dtype=np.float32)
    return np.concatenate([m[:, 0], m[:, 1]])

def quat_xyzw_to_rot6d(q: np.ndarray) -> np.ndarray:
    return rot6d(R.from_quat(q).as_matrix())

def euler_xyz_to_rot6d(e: np.ndarray) -> np.ndarray:
    return rot6d(R.from_euler("xyz", e).as_matrix())

def euler_xyz_to_mat4(pos: np.ndarray, euler: np.ndarray) -> np.ndarray:
    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = R.from_euler("xyz", euler).as_matrix()
    mat[:3, 3] = pos
    return mat

def inv_se3(mat: np.ndarray) -> np.ndarray:
    Ri = mat[:3, :3].T
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = Ri
    Ti[:3, 3] = -Ri @ mat[:3, 3]
    return Ti

def eef_head_rel_to_world(pos_local: np.ndarray, euler_global: np.ndarray,
                           head_pos_w: np.ndarray, head_quat_w: np.ndarray):
    """Convert EEF position from head-relative frame → world frame.
    NOTE: euler is already in global frame, no rotation needed."""
    R_head = R.from_quat(head_quat_w).as_matrix()
    pos_world = R_head @ pos_local.astype(np.float64) + head_pos_w
    return pos_world.astype(np.float32), euler_global.astype(np.float32)

def tips_in_wrist_frame(tips_world: np.ndarray, wrist_mat4: np.ndarray) -> np.ndarray:
    """tips_world (5,3) → (6,3): row0=[0,0,0] (palm), rows1-5=fingertips in wrist frame.
    Stores R_lw @ delta so that visualization (R_wl @ kp + wrist_pos) recovers delta + wrist_pos."""
    R_lw = wrist_mat4[:3, :3]      # local→world rotation
    wrist_pos = wrist_mat4[:3, 3]
    out = np.zeros((6, 3), dtype=np.float32)
    for i in range(5):
        delta = tips_world[i].astype(np.float64) - wrist_pos
        delta[0] = -delta[0]       # mirror across x=wrist_x so fingers appear at +x in world
        out[i + 1] = (R_lw @ delta).astype(np.float32)
    return out

def build_vec(head_pos, head_quat_xyzw,
              left_pos, left_euler, right_pos, right_euler,
              left_tips_world, right_tips_world) -> np.ndarray:
    vec = np.zeros(VEC_SIZE, dtype=np.float32)

    vec[IDX_HEAD_EEF]  = np.concatenate([head_pos.astype(np.float32),
                                          quat_xyzw_to_rot6d(head_quat_xyzw)])
    vec[IDX_LEFT_EEF]  = np.concatenate([left_pos.astype(np.float32),
                                          euler_xyz_to_rot6d(left_euler)])
    vec[IDX_RIGHT_EEF] = np.concatenate([right_pos.astype(np.float32),
                                          euler_xyz_to_rot6d(right_euler)])

    left_mat  = euler_xyz_to_mat4(left_pos,  left_euler)
    right_mat = euler_xyz_to_mat4(right_pos, right_euler)
    vec[IDX_LEFT_KPTS]  = tips_in_wrist_frame(left_tips_world,  left_mat).reshape(-1)
    vec[IDX_RIGHT_KPTS] = tips_in_wrist_frame(right_tips_world, right_mat).reshape(-1)
    # [100:126] stays zero per user instruction
    return vec

def encode_jpeg(frame_bgr: np.ndarray) -> np.ndarray:
    _, buf = cv2.imencode(".jpg", frame_bgr,
                          [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.flatten()

def extract_frames(video_path: str, from_ts: float, n_frames: int) -> list:
    """
    Extract n_frames from video_path starting at from_ts (seconds).
    Uses ffmpeg pipe to support AV1 codec. Returns list of (H,W,3) uint8 BGR arrays.
    """
    H, W = IMAGE_HW
    cmd = [
        "ffmpeg", "-loglevel", "error",
        "-ss", str(from_ts),
        "-i", video_path,
        "-vframes", str(n_frames),
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE)
    raw = np.frombuffer(result.stdout, dtype=np.uint8)

    frame_size = H * W * 3
    n_decoded = len(raw) // frame_size
    frames = []
    for i in range(n_decoded):
        frames.append(raw[i * frame_size:(i + 1) * frame_size].reshape(H, W, 3))

    # pad with last frame if video ended early
    pad = frames[-1] if frames else np.zeros((H, W, 3), dtype=np.uint8)
    while len(frames) < n_frames:
        frames.append(pad)

    return frames

def write_compressed_images(frames_bgr: list, key: str, hf: h5py.File):
    encoded = [encode_jpeg(f) for f in frames_bgr]
    max_len = max(len(e) for e in encoded)
    ds = hf.create_dataset(key, (len(encoded), max_len), dtype=np.uint8)
    for i, e in enumerate(encoded):
        ds[i, :len(e)] = e

# ── load all parquet episodes ──────────────────────────────────────────────────
parquet_files = sorted(glob(os.path.join(DATA_DIR, "*", "*.parquet")))
df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
df_all = df_all.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

meta_ep = pd.read_parquet(META_EP)
meta_ep = meta_ep.set_index("episode_index")

episode_ids = sorted(df_all["episode_index"].unique())
if MAX_EPISODES is not None:
    episode_ids = episode_ids[:MAX_EPISODES]
print(f"Total episodes to process: {len(episode_ids)}")

# ── process each episode ───────────────────────────────────────────────────────
for ep_idx in episode_ids:
    out_path = os.path.join(OUT_DIR, f"{ep_idx}.hdf5")
    if os.path.exists(out_path):
        print(f"  ep {ep_idx:04d}: skip (exists)")
        continue

    ep_df    = df_all[df_all["episode_index"] == ep_idx].reset_index(drop=True)
    T        = len(ep_df)
    ep_meta  = meta_ep.loc[ep_idx]

    # ── load head & finger pkl ──
    with open(os.path.join(HEAD_DIR,   f"ep_{ep_idx:04d}.pkl"), "rb") as f:
        head_data = pickle.load(f)
    with open(os.path.join(FINGER_DIR, f"ep_{ep_idx:04d}.pkl"), "rb") as f:
        finger_data = pickle.load(f)

    head_pos_w   = head_data["head_pos_w"]    # (T, 3)
    head_quat_w  = head_data["head_quat_w"]   # (T, 4) xyzw
    tips_left    = finger_data["finger_pos_left"]   # (T, 5, 3) world
    tips_right   = finger_data["finger_pos_right"]  # (T, 5, 3) world

    # ── load body PKL for "real" mode ──
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

    # ── build state vectors ──
    states = np.zeros((T, VEC_SIZE), dtype=np.float32)
    for t in range(T):
        ee = np.array(ep_df.at[t, "observation.state.ee_state"])
        if MODE == "real":
            left_pos    = body_pos_w[t, lhb_idx].astype(np.float32)
            right_pos   = body_pos_w[t, rhb_idx].astype(np.float32)
            left_euler  = R.from_quat(body_quat_w[t, lwy_idx]).as_euler("xyz").astype(np.float32)
            right_euler = R.from_quat(body_quat_w[t, rwy_idx]).as_euler("xyz").astype(np.float32)
            # finger_pos_left/right 是 head-relative，需转到世界坐标系
            R_head = R.from_quat(head_quat_w[t]).as_matrix()
            tips_left_t  = (R_head @ tips_left[t].T).T  + head_pos_w[t]
            tips_right_t = (R_head @ tips_right[t].T).T + head_pos_w[t]
        elif MODE == "rel":
            left_pos,  left_euler  = eef_head_rel_to_world(
                ee[0:3], ee[3:6], head_pos_w[t], head_quat_w[t])
            right_pos, right_euler = eef_head_rel_to_world(
                ee[6:9], ee[9:12], head_pos_w[t], head_quat_w[t])
            # rel 模式 wrist 已转到世界坐标，finger 同样需要转换
            R_head = R.from_quat(head_quat_w[t]).as_matrix()
            tips_left_t  = (R_head @ tips_left[t].T).T  + head_pos_w[t]
            tips_right_t = (R_head @ tips_right[t].T).T + head_pos_w[t]
        else:  # global
            left_pos,  left_euler  = ee[0:3], ee[3:6]
            right_pos, right_euler = ee[6:9], ee[9:12]
            tips_left_t  = tips_left[t]
            tips_right_t = tips_right[t]
        states[t] = build_vec(
            head_pos=head_pos_w[t],
            head_quat_xyzw=head_quat_w[t],
            left_pos=left_pos,  left_euler=left_euler,
            right_pos=right_pos, right_euler=right_euler,
            left_tips_world=tips_left_t,
            right_tips_world=tips_right_t,
        )

    # action[t] = state[t+1]; last frame copies itself
    actions = np.zeros_like(states)
    actions[:-1] = states[1:]
    actions[-1]  = states[-1]

    # ── extract video frames ──
    left_file_idx  = int(ep_meta["videos/observation.images.head_stereo_left/file_index"])
    right_file_idx = int(ep_meta["videos/observation.images.head_stereo_right/file_index"])
    left_from_ts   = float(ep_meta["videos/observation.images.head_stereo_left/from_timestamp"])
    right_from_ts  = float(ep_meta["videos/observation.images.head_stereo_right/from_timestamp"])

    left_video  = os.path.join(VIDEO_L_DIR, f"file-{left_file_idx:03d}.mp4")
    right_video = os.path.join(VIDEO_R_DIR, f"file-{right_file_idx:03d}.mp4")

    frames_left  = extract_frames(left_video,  left_from_ts,  T)
    frames_right = extract_frames(right_video, right_from_ts, T)

    # ── write HDF5 ──
    with h5py.File(out_path, "w") as hf:
        hf.create_dataset("observation.state", data=states,  dtype=np.float32)
        hf.create_dataset("action",            data=actions, dtype=np.float32)
        write_compressed_images(frames_left,  "observation.image.left",  hf)
        write_compressed_images(frames_right, "observation.image.right", hf)

    print(f"  ep {ep_idx:04d}: T={T} → {out_path}")

print("Done.")
