"""
Convert UnifoLM WBT episodes to human_policy training HDF5.

This is a standalone replacement for the older convert_to_hdf5 scripts.

Inputs:
  * wrist-relative finger keypoints:
      <dataset>/finger_keypoints/ep_XXXX.pkl
  * head/wrist body data:
      <xqy-dir>/<xqy-prefix>_ep_XXXX.pkl
  * original LeRobot parquet:
      <dataset>/data/chunk-*/file-*.parquet
  * stereo videos from the dataset metadata.

Wrist modes:
  EE:   wrist position from ee_state, converted from head-relative to world;
        wrist rotation from ee_state global Euler.
  XQY1: wrist position from L/R_hand_base_link;
        wrist rotation from left/right_wrist_yaw_link.
  XQY2: wrist position from L/R_hand_base_link;
        wrist rotation assembled from wrist roll/pitch/yaw links.
  XQY3: wrist position from explicit pkl wrist pos fields when present,
        otherwise L/R_hand_base_link; wrist rotation from left/right_wrist_rot.
"""

import argparse
import json
import os
import pickle
import subprocess
import warnings
from glob import glob

import cv2
import h5py
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


warnings.filterwarnings("ignore", category=DeprecationWarning)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET = os.path.join(
    PROJECT_ROOT,
    "DATASETS",
    "UnifoLM_WBT",
    "G1_WBT_Brainco_Collect_Plates_Into_Dishwasher",
)
DEFAULT_XQY_DIR = os.path.join(PROJECT_ROOT, "DATASETS", "XQY_PKL")
DEFAULT_PH2D_REFERENCE = os.path.join(
    PROJECT_ROOT,
    "DATASETS",
    "PH2D",
    "402-pick_on_color_pad_right-2025_01_09-16_36_15",
    "processed_episode_0.hdf5",
)

VEC_SIZE = 128
IDX_HEAD_EEF = np.arange(0, 9)
IDX_LEFT_KPTS = np.arange(10, 28)
IDX_RIGHT_EEF = np.arange(30, 39)
IDX_RIGHT_KPTS = np.arange(40, 58)
IDX_LEFT_EEF = np.arange(80, 89)
IDX_QPOS = np.arange(100, 126)

DEFAULT_IMAGE_HW = (480, 640)

XQY3_WRIST_ROT_TO_YAW_LINK = {
    "left": R.from_matrix(
        np.array(
            [
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    ),
    "right": R.from_matrix(
        np.array(
            [
                [0.0, -1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
    ),
}

PH2D_ALIGNED_FINGER_LOCAL_ROT = {
    "left": np.array(
        [
            [-0.98947971, 0.02501830, 0.14249204],
            [-0.13897305, -0.43808587, -0.88812570],
            [0.04020435, -0.89858492, 0.43695395],
        ],
        dtype=np.float64,
    ),
    "right": np.array(
        [
            [-0.80359132, -0.56954821, 0.17278839],
            [0.51014629, -0.50957456, 0.69288132],
            [-0.30658075, 0.64494078, 0.70004260],
        ],
        dtype=np.float64,
    ),
}

_PH2D_WORLD_DIR_CACHE = {}


def rot6d(mat: np.ndarray) -> np.ndarray:
    """PyTorch3D 6D rotation format: first two matrix rows flattened."""
    mat = np.asarray(mat, dtype=np.float32)
    return mat[:2, :].reshape(-1)


def quat_xyzw_to_rot6d(quat: np.ndarray) -> np.ndarray:
    return rot6d(R.from_quat(quat).as_matrix())


def euler_xyz_to_rot6d(euler: np.ndarray) -> np.ndarray:
    return rot6d(R.from_euler("xyz", euler).as_matrix())


def pack_wrist_tips(tips_wrist: np.ndarray) -> np.ndarray:
    out = np.zeros((6, 3), dtype=np.float32)
    out[1:] = np.asarray(tips_wrist, dtype=np.float32)
    return out


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(PROJECT_ROOT, path))


def read_meta_episodes(dataset: str) -> pd.DataFrame:
    meta_path = os.path.join(dataset, "meta", "episodes")
    if os.path.isdir(meta_path):
        files = sorted(glob(os.path.join(meta_path, "*", "*.parquet")))
        if not files:
            files = sorted(glob(os.path.join(meta_path, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No episode meta parquet files under {meta_path}")
        meta = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        meta = pd.read_parquet(meta_path)
    return meta.set_index("episode_index")


def read_info(dataset: str) -> dict:
    path = os.path.join(dataset, "meta", "info.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        return json.load(f)


def image_hw_from_info(info: dict, video_key: str) -> tuple[int, int]:
    feat = info.get("features", {}).get(video_key, {})
    shape = feat.get("shape")
    if shape and len(shape) >= 2:
        return int(shape[0]), int(shape[1])
    return DEFAULT_IMAGE_HW


def auto_video_keys(meta: pd.DataFrame) -> tuple[str, str]:
    cols = meta.columns
    candidates = []
    for c in cols:
        if c.startswith("videos/") and c.endswith("/file_index"):
            candidates.append(c[len("videos/") : -len("/file_index")])

    preferred_pairs = [
        ("observation.images.head_stereo_left", "observation.images.head_stereo_right"),
        ("observation.images.cam_0", "observation.images.cam_1"),
        ("observation.images.left", "observation.images.right"),
    ]
    for left, right in preferred_pairs:
        if left in candidates and right in candidates:
            return left, right

    left_like = [k for k in candidates if "left" in k.lower() or k.endswith("cam_0")]
    right_like = [k for k in candidates if "right" in k.lower() or k.endswith("cam_1")]
    if left_like and right_like:
        return left_like[0], right_like[0]
    raise KeyError(f"Could not auto-detect left/right video keys from: {candidates}")


def extract_video_frames(
    dataset: str,
    ep_meta: pd.Series,
    video_key: str,
    n_frames: int,
    image_hw: tuple[int, int],
) -> list[np.ndarray]:
    chunk_idx = int(ep_meta.get(f"videos/{video_key}/chunk_index", 0))
    file_idx = int(ep_meta[f"videos/{video_key}/file_index"])
    from_ts = float(ep_meta[f"videos/{video_key}/from_timestamp"])
    video_path = os.path.join(
        dataset,
        "videos",
        video_key,
        f"chunk-{chunk_idx:03d}",
        f"file-{file_idx:03d}.mp4",
    )
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)

    height, width = image_hw
    cmd = [
        "ffmpeg",
        "-loglevel",
        "error",
        "-ss",
        str(from_ts),
        "-i",
        video_path,
        "-vframes",
        str(n_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode("utf-8", errors="ignore").strip())

    raw = np.frombuffer(proc.stdout, dtype=np.uint8)
    frame_size = height * width * 3
    frames = [
        raw[i * frame_size : (i + 1) * frame_size].reshape(height, width, 3)
        for i in range(len(raw) // frame_size)
    ]
    pad = frames[-1] if frames else np.zeros((height, width, 3), dtype=np.uint8)
    while len(frames) < n_frames:
        frames.append(pad)
    return frames


def write_jpeg_dataset(hf: h5py.File, key: str, frames_bgr: list[np.ndarray], quality: int):
    encoded = []
    for frame in frames_bgr:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError(f"Failed to JPEG encode frame for {key}")
        encoded.append(buf.flatten())
    max_len = max(len(x) for x in encoded)
    ds = hf.create_dataset(key, (len(encoded), max_len), dtype=np.uint8)
    for i, arr in enumerate(encoded):
        ds[i, : len(arr)] = arr


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
    def __init__(self, data: dict, head_link: str):
        self.data = data
        self.body_list = data["link_body_list"]
        self.pos_w = np.asarray(data["body_pos_w"])
        self.quat_w = np.asarray(data["body_quat_w"])
        self.idx = {
            "head": body_index(self.body_list, head_link),
            "lhb": body_index(self.body_list, "L_hand_base_link", "left_hand_base_link"),
            "rhb": body_index(self.body_list, "R_hand_base_link", "right_hand_base_link"),
            "lwy": body_index(self.body_list, "left_wrist_yaw_link"),
            "rwy": body_index(self.body_list, "right_wrist_yaw_link"),
            "lwr": body_index(self.body_list, "left_wrist_roll_link"),
            "rwr": body_index(self.body_list, "right_wrist_roll_link"),
            "lwp": body_index(self.body_list, "left_wrist_pitch_link"),
            "rwp": body_index(self.body_list, "right_wrist_pitch_link"),
        }

    def head(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        return self.pos_w[t, self.idx["head"]], self.quat_w[t, self.idx["head"]]

    def body_pose(self, t: int, link_name: str) -> tuple[np.ndarray, np.ndarray]:
        idx = body_index(self.body_list, link_name)
        return self.pos_w[t, idx], self.quat_w[t, idx]

    def hand_base_pos(self, t: int, side: str) -> np.ndarray:
        return self.pos_w[t, self.idx["lhb" if side == "left" else "rhb"]]

    def yaw_link_quat(self, t: int, side: str) -> np.ndarray:
        return self.quat_w[t, self.idx["lwy" if side == "left" else "rwy"]]

    def hand_base_quat(self, t: int, side: str) -> np.ndarray:
        return self.quat_w[t, self.idx["lhb" if side == "left" else "rhb"]]

    def roll_pitch_yaw_quat(self, t: int, side: str) -> np.ndarray:
        roll_idx = self.idx["lwr" if side == "left" else "rwr"]
        pitch_idx = self.idx["lwp" if side == "left" else "rwp"]
        yaw_idx = self.idx["lwy" if side == "left" else "rwy"]
        roll = R.from_quat(self.quat_w[t, roll_idx]).as_euler("xyz")[0]
        pitch = R.from_quat(self.quat_w[t, pitch_idx]).as_euler("xyz")[1]
        yaw = R.from_quat(self.quat_w[t, yaw_idx]).as_euler("xyz")[2]
        return R.from_euler("xyz", [roll, pitch, yaw]).as_quat()

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

    def direct_wrist_quat(self, t: int, side: str, rot_frame: str) -> np.ndarray:
        key = "left_wrist_rot" if side == "left" else "right_wrist_rot"
        if key not in self.data:
            return self.yaw_link_quat(t, side)
        quat_xyzw = np.asarray(self.data[key])[t]
        if rot_frame == "raw":
            return quat_xyzw
        if rot_frame == "yaw_link_aligned":
            return (R.from_quat(quat_xyzw) * XQY3_WRIST_ROT_TO_YAW_LINK[side]).as_quat()
        raise ValueError(f"Unsupported XQY3 wrist rot frame: {rot_frame}")


def ee_wrist_pose(
    ee: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left_pos = ee[0:3].astype(np.float64)
    right_pos = ee[6:9].astype(np.float64)
    left_quat = R.from_euler("xyz", ee[3:6]).as_quat()
    right_quat = R.from_euler("xyz", ee[9:12]).as_quat()
    return left_pos, left_quat, right_pos, right_quat


def xqy_wrist_pose(xqy: XQYAccess, mode: str, t: int, xqy3_wrist_rot_frame: str):
    left_pos = xqy.hand_base_pos(t, "left")
    right_pos = xqy.hand_base_pos(t, "right")
    if mode == "XQY1":
        left_quat = xqy.hand_base_quat(t, "left")
        right_quat = xqy.hand_base_quat(t, "right")
    elif mode == "XQY_PH2D":
        left_quat = xqy.yaw_link_quat(t, "left")
        right_quat = xqy.yaw_link_quat(t, "right")
    elif mode == "XQY2":
        left_quat = xqy.roll_pitch_yaw_quat(t, "left")
        right_quat = xqy.roll_pitch_yaw_quat(t, "right")
    elif mode == "XQY3":
        left_pos = xqy.direct_wrist_pos(t, "left")
        right_pos = xqy.direct_wrist_pos(t, "right")
        left_quat = xqy.direct_wrist_quat(t, "left", xqy3_wrist_rot_frame)
        right_quat = xqy.direct_wrist_quat(t, "right", xqy3_wrist_rot_frame)
    else:
        raise ValueError(mode)
    return left_pos, left_quat, right_pos, right_quat


def build_state(
    head_pos: np.ndarray,
    head_quat: np.ndarray,
    left_pos: np.ndarray,
    left_quat: np.ndarray,
    right_pos: np.ndarray,
    right_quat: np.ndarray,
    left_tips_wrist: np.ndarray,
    right_tips_wrist: np.ndarray,
    robot_q: np.ndarray | None,
) -> np.ndarray:
    state = np.zeros(VEC_SIZE, dtype=np.float32)
    state[IDX_HEAD_EEF] = np.concatenate([head_pos.astype(np.float32), quat_xyzw_to_rot6d(head_quat)])
    state[IDX_LEFT_EEF] = np.concatenate([left_pos.astype(np.float32), quat_xyzw_to_rot6d(left_quat)])
    state[IDX_RIGHT_EEF] = np.concatenate([right_pos.astype(np.float32), quat_xyzw_to_rot6d(right_quat)])
    state[IDX_LEFT_KPTS] = pack_wrist_tips(left_tips_wrist).reshape(-1)
    state[IDX_RIGHT_KPTS] = pack_wrist_tips(right_tips_wrist).reshape(-1)
    if robot_q is not None:
        q = np.asarray(robot_q, dtype=np.float32)
        state[IDX_QPOS[: min(len(IDX_QPOS), len(q))]] = q[: len(IDX_QPOS)]
    return state


def ph2d_world_dirs(reference_path: str, side: str) -> tuple[np.ndarray, np.ndarray]:
    key = (reference_path, side)
    if key in _PH2D_WORLD_DIR_CACHE:
        return _PH2D_WORLD_DIR_CACHE[key]

    from hdt.inference_utils import get_eef_kpts_from_prediction

    with h5py.File(reference_path, "r") as hf:
        action = hf["action"][:]

    out = []
    for t in (0, action.shape[0] - 1):
        decoded = get_eef_kpts_from_prediction(action[t])
        if side == "left":
            wrist = decoded["left_wrist_mat"]
            kpts = decoded["left_hand_kpts"]
        else:
            wrist = decoded["right_wrist_mat"]
            kpts = decoded["right_hand_kpts"]
        palm_world = (wrist @ np.r_[kpts[0], 1.0])[:3]
        dirs = []
        for idx in (4, 9, 14, 19, 24):
            vec = (wrist @ np.r_[kpts[idx], 1.0])[:3] - palm_world
            dirs.append(vec / (np.linalg.norm(vec) + 1e-9))
        out.append(np.asarray(dirs, dtype=np.float64))

    _PH2D_WORLD_DIR_CACHE[key] = (out[0], out[1])
    return _PH2D_WORLD_DIR_CACHE[key]


def transform_finger_tips(
    tips: np.ndarray,
    side: str,
    mode: str,
    wrist_quat: np.ndarray,
    frame_idx: int,
    frame_count: int,
    ph2d_reference: str,
) -> np.ndarray:
    if mode != "XQY_PH2D":
        return tips

    ref_first, ref_last = ph2d_world_dirs(ph2d_reference, side)
    alpha = 0.0 if frame_count <= 1 else frame_idx / float(frame_count - 1)
    dirs = (1.0 - alpha) * ref_first + alpha * ref_last
    dirs = dirs / (np.linalg.norm(dirs, axis=1, keepdims=True) + 1e-9)

    lengths = np.linalg.norm(np.asarray(tips, dtype=np.float64), axis=1)
    desired_world = dirs * lengths[:, None]
    wrist_rot = R.from_quat(wrist_quat).as_matrix()
    return (wrist_rot.T @ desired_world.T).T.astype(np.float32)


def task_description(ep_meta: pd.Series) -> str:
    tasks = ep_meta.get("tasks", "")
    if isinstance(tasks, np.ndarray):
        return ", ".join(map(str, tasks.tolist()))
    if isinstance(tasks, list):
        return ", ".join(map(str, tasks))
    return str(tasks)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert UnifoLM WBT data to human_policy HDF5.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="LeRobot dataset folder")
    parser.add_argument("--mode", required=True, choices=["EE", "XQY1", "XQY2", "XQY3", "XQY_PH2D"])
    parser.add_argument("--finger-dir", default=None, help="default: <dataset>/finger_keypoints")
    parser.add_argument("--xqy-dir", default=DEFAULT_XQY_DIR)
    parser.add_argument("--xqy-prefix", default=None, help="default: basename(dataset)")
    parser.add_argument("--ph2d-reference", default=DEFAULT_PH2D_REFERENCE,
                        help="reference HDF5 used by XQY_PH2D for global finger direction templates")
    parser.add_argument("--out-dir", default=None, help="default: <dataset>/hdf5_<mode>")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--episode", type=int, default=None, help="convert one episode only")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--left-video-key", default=None, help="default: auto")
    parser.add_argument("--right-video-key", default=None, help="default: auto")
    parser.add_argument("--head-link", default="head_mocap",
                        help="XQY body link used as head pose (default: head_mocap; use head_link to reproduce old behavior)")
    parser.add_argument("--ee-origin-link", default=None,
                        help="deprecated; EE mode now treats ee_state position as world position")
    parser.add_argument("--xqy3-wrist-rot-frame", default="raw",
                        choices=["yaw_link_aligned", "raw"],
                        help="how to interpret left/right_wrist_rot in XQY3 (default: raw hand-base-aligned quaternion)")
    parser.add_argument("--no-images", action="store_true", help="write zero image placeholders")
    parser.add_argument("--image-height", type=int, default=None)
    parser.add_argument("--image-width", type=int, default=None)
    parser.add_argument("--no-qpos", action="store_true", help="leave [100:126] qpos zeros")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = resolve_path(args.dataset)
    finger_dir = resolve_path(args.finger_dir) if args.finger_dir else os.path.join(dataset, "finger_keypoints")
    xqy_dir = resolve_path(args.xqy_dir)
    xqy_prefix = args.xqy_prefix or os.path.basename(dataset.rstrip("/"))
    out_dir = resolve_path(args.out_dir) if args.out_dir else os.path.join(dataset, f"hdf5_{args.mode}")
    os.makedirs(out_dir, exist_ok=True)

    parquet_files = sorted(glob(os.path.join(dataset, "data", "*", "*.parquet")))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {dataset}/data/*/*.parquet")
    df_all = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)
    df_all = df_all.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    meta = read_meta_episodes(dataset)
    info = read_info(dataset)
    if args.left_video_key and args.right_video_key:
        left_video_key, right_video_key = args.left_video_key, args.right_video_key
    else:
        left_video_key, right_video_key = auto_video_keys(meta)

    left_hw = (args.image_height, args.image_width) if args.image_height and args.image_width else image_hw_from_info(info, left_video_key)
    right_hw = (args.image_height, args.image_width) if args.image_height and args.image_width else image_hw_from_info(info, right_video_key)

    episode_ids = sorted(df_all["episode_index"].unique())
    if args.episode is not None:
        episode_ids = [args.episode]
    elif args.max_episodes is not None:
        episode_ids = episode_ids[: args.max_episodes]

    print(f"dataset: {dataset}")
    print(f"mode: {args.mode}")
    print(f"finger_dir: {finger_dir}")
    print(f"xqy_prefix: {xqy_prefix}")
    print(f"head_link: {args.head_link}")
    if args.mode == "EE":
        print("ee_position_frame: world")
    print(f"out_dir: {out_dir}")
    print(f"video keys: {left_video_key}, {right_video_key}")
    print(f"episodes: {len(episode_ids)}")

    for ep_idx in episode_ids:
        out_path = os.path.join(out_dir, f"{ep_idx}.hdf5")
        if os.path.exists(out_path) and not args.overwrite:
            print(f"  ep {ep_idx:04d}: skip exists")
            continue

        ep_df = df_all[df_all["episode_index"] == ep_idx].reset_index(drop=True)
        if ep_df.empty:
            raise ValueError(f"episode_index={ep_idx} not found in parquet data")
        ep_meta = meta.loc[ep_idx]

        with open(os.path.join(finger_dir, f"ep_{ep_idx:04d}.pkl"), "rb") as f:
            finger = pickle.load(f)
        if finger.get("finger_frame") != "wrist":
            raise ValueError(f"Expected finger_frame='wrist' for ep {ep_idx}, got {finger.get('finger_frame')!r}")

        with open(xqy_path_for_episode(xqy_dir, xqy_prefix, ep_idx), "rb") as f:
            xqy_data = pickle.load(f)
        xqy = XQYAccess(xqy_data, args.head_link)

        t_count = min(len(ep_df), len(finger["finger_pos_left"]), xqy.pos_w.shape[0])
        states = np.zeros((t_count, VEC_SIZE), dtype=np.float32)
        for t in range(t_count):
            head_pos, head_quat = xqy.head(t)
            ee = np.asarray(ep_df.at[t, "observation.state.ee_state"], dtype=np.float64)
            if args.mode == "EE":
                left_pos, left_quat, right_pos, right_quat = ee_wrist_pose(ee)
            else:
                left_pos, left_quat, right_pos, right_quat = xqy_wrist_pose(
                    xqy, args.mode, t, args.xqy3_wrist_rot_frame
                )

            robot_q = None if args.no_qpos else ep_df.at[t, "observation.state.robot_q_current"]
            states[t] = build_state(
                head_pos=head_pos,
                head_quat=head_quat,
                left_pos=left_pos,
                left_quat=left_quat,
                right_pos=right_pos,
                right_quat=right_quat,
                left_tips_wrist=transform_finger_tips(
                    finger["finger_pos_left"][t],
                    "left",
                    args.mode,
                    left_quat,
                    t,
                    t_count,
                    args.ph2d_reference,
                ),
                right_tips_wrist=transform_finger_tips(
                    finger["finger_pos_right"][t],
                    "right",
                    args.mode,
                    right_quat,
                    t,
                    t_count,
                    args.ph2d_reference,
                ),
                robot_q=robot_q,
            )

        actions = np.zeros_like(states)
        actions[:-1] = states[1:]
        actions[-1] = states[-1]

        if args.no_images:
            left_frames = [np.zeros((*left_hw, 3), dtype=np.uint8) for _ in range(t_count)]
            right_frames = [np.zeros((*right_hw, 3), dtype=np.uint8) for _ in range(t_count)]
        else:
            left_frames = extract_video_frames(dataset, ep_meta, left_video_key, t_count, left_hw)
            right_frames = extract_video_frames(dataset, ep_meta, right_video_key, t_count, right_hw)

        with h5py.File(out_path, "w") as hf:
            hf.create_dataset("observation.state", data=states, dtype=np.float32)
            hf.create_dataset("action", data=actions, dtype=np.float32)
            write_jpeg_dataset(hf, "observation.image.left", left_frames, args.jpeg_quality)
            write_jpeg_dataset(hf, "observation.image.right", right_frames, args.jpeg_quality)
            hf.attrs["sim"] = np.bool_(False)
            hf.attrs["embodiment"] = f"human_mocap_{args.mode.lower()}"
            hf.attrs["description"] = task_description(ep_meta)

        print(f"  ep {ep_idx:04d}: T={t_count} -> {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()
