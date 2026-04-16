#!/usr/bin/env python3
"""
将标注类动捕 HDF5（如 data/0.hdf5，含 transforms/*）转为 hdt/main.py 训练所需的 HDF5 格式：

  必填数据集：
    - `action`: 形状 (T, 128)，float32
    - `observation.state`: 形状 (T, 128)，作为观测状态（使用当前帧的姿态/关键点）
    - `observation.image.top`: 形状 (T, H, W, 3)，目前默认为全零占位图或从视频提取

推荐流程：
  python3 data/mocap_hdf5_to_processed.py -i data/0.hdf5 -o data/processed_episode_0.hdf5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import cv2

import hdt.constants as C


def _matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)

def _save_compressed_imgs_hdf5(imgs_chw: np.ndarray, key: str, hf: h5py.File, jpeg_quality: int = 50) -> None:
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    num_imgs = imgs_chw.shape[0]
    compressed_len_list = []
    encoded_img_list = []
    for i in range(num_imgs):
        img = imgs_chw[i].transpose(1, 2, 0)
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        _, img_encode = cv2.imencode(".jpg", img_bgr, encode_param)
        encoded_img_list.append(img_encode)
        compressed_len_list.append(len(img_encode))
    max_len = int(max(compressed_len_list)) if compressed_len_list else 0
    hf.create_dataset(key, (num_imgs, max_len), dtype=np.uint8)
    for i in range(num_imgs):
        hf[key][i, : compressed_len_list[i]] = encoded_img_list[i].flatten()

def _load_video_left_right(
    video_path: str,
    image_hw: tuple[int, int],
    *,
    sbs: bool = False,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    T = video_len if max_frames is None else min(video_len, int(max_frames))
    H, W = int(image_hw[0]), int(image_hw[1])
    left_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
    right_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
    i = 0
    while cap.isOpened() and i < T:
        ret, frame = cap.read()
        if not ret:
            break
        if sbs:
            ww = frame.shape[1]
            left_frame = frame[:, : ww // 2]
            right_frame = frame[:, ww // 2 :]
        else:
            left_frame = frame
            right_frame = frame
        left_resized = cv2.resize(left_frame, (W, H), interpolation=cv2.INTER_AREA)
        right_resized = cv2.resize(right_frame, (W, H), interpolation=cv2.INTER_AREA)
        left_imgs[i] = cv2.cvtColor(left_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        right_imgs[i] = cv2.cvtColor(right_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        i += 1
    cap.release()
    return left_imgs[:i], right_imgs[:i]

def _load_T(cam_to_sim: str | None) -> np.ndarray | None:
    if cam_to_sim is None:
        return None
    cam_to_sim = str(cam_to_sim).strip()
    if not cam_to_sim:
        return None

    if cam_to_sim.lower() == "default":
        # A commonly used camera->sim axis remap (x,y,z)->(-z,x,y)
        R = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        return T

    p = Path(cam_to_sim)
    if p.exists() and p.suffix.lower() == ".npy":
        T = np.load(str(p)).astype(np.float32)
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix in {p}, got {T.shape}")
        return T

    if p.exists() and p.suffix.lower() == ".json":
        obj = json.load(open(p, "r"))
        if isinstance(obj, list) and len(obj) == 16:
            T = np.array(obj, dtype=np.float32).reshape(4, 4)
        elif isinstance(obj, list) and len(obj) == 4 and all(isinstance(r, list) and len(r) == 4 for r in obj):
            T = np.array(obj, dtype=np.float32)
        else:
            raise ValueError(f"Unsupported json format for transform in {p}")
        if T.shape != (4, 4):
            raise ValueError(f"Expected 4x4 matrix in {p}, got {T.shape}")
        return T

    # Comma/space separated 16 floats
    parts = [x for x in cam_to_sim.replace(",", " ").split() if x]
    if len(parts) != 16:
        raise ValueError("--cam-to-sim must be 'default', a .npy/.json path, or 16 floats")
    T = np.array([float(x) for x in parts], dtype=np.float32).reshape(4, 4)
    return T

def _apply_T(T_map: np.ndarray | None, T_in: np.ndarray) -> np.ndarray:
    if T_map is None:
        return T_in
    return (T_map @ T_in).astype(np.float32)

def _make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = R.astype(np.float32)
    T[:3, 3] = t.astype(np.float32)
    return T

# 与 Inspire 手 25 点布局中 RETARGETTING_INDICES 对应的 6 组语义顺序（与 cmd_dict2policy 中展平顺序一致）
_LEFT_TIP_NAMES = [
    "leftThumbTip",
    "leftIndexFingerTip",
    "leftMiddleFingerTip",
    "leftRingFingerTip",
    "leftLittleFingerTip",
]
_RIGHT_TIP_NAMES = [
    "rightThumbTip",
    "rightIndexFingerTip",
    "rightMiddleFingerTip",
    "rightRingFingerTip",
    "rightLittleFingerTip",
]


def _T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _point_cam_from_T(T: np.ndarray) -> np.ndarray:
    """齐次变换下关节原点在世界/相机坐标系中的位置。"""
    return T[:3, 3].astype(np.float64)


def _fingertips_in_wrist_frame(
    f: h5py.File,
    frame_idx: int,
    wrist_name: str,
    tip_names: list[str],
) -> np.ndarray:
    """返回 (6, 3)：第 0 行为掌心 [0,0,0]，其余为各指尖在手腕坐标系下的位置。"""
    T_w = np.asarray(f["transforms"][wrist_name][frame_idx], dtype=np.float64)
    inv_w = _T_inv(T_w)
    out = np.zeros((6, 3), dtype=np.float64)
    for i, name in enumerate(tip_names, start=1):
        T_tip = np.asarray(f["transforms"][name][frame_idx], dtype=np.float64)
        p_cam = _point_cam_from_T(T_tip)
        hom = np.array([p_cam[0], p_cam[1], p_cam[2], 1.0])
        out[i] = (inv_w @ hom)[:3]
    return out

def _fingertips_in_ref_frame(
    f: h5py.File,
    frame_idx: int,
    *,
    T_wrist: np.ndarray,
    tip_names: list[str],
    T_map: np.ndarray | None,
    inv_ref: np.ndarray | None,
) -> np.ndarray:
    """返回 (6,3)：第0行为 wrist 原点在 ref 下的位置，其余为各指尖在 ref 下的位置。"""
    T_w = _apply_T(T_map, np.asarray(T_wrist, dtype=np.float32))
    if inv_ref is not None:
        T_w = (inv_ref @ T_w).astype(np.float32)
    out = np.zeros((6, 3), dtype=np.float32)
    out[0] = T_w[:3, 3]
    for i, name in enumerate(tip_names, start=1):
        T_tip = _apply_T(T_map, np.asarray(f["transforms"][name][frame_idx], dtype=np.float32))
        if inv_ref is not None:
            T_tip = (inv_ref @ T_tip).astype(np.float32)
        out[i] = T_tip[:3, 3]
    return out


def _build_action_row(
    f: h5py.File,
    frame_idx: int,
    head_bone: str,
    *,
    zero_head_translation: bool,
    pos_scale: float,
    T_map: np.ndarray | None,
    rel_frame: str,
    inv_ref_head0: np.ndarray | None,
    still_head: bool,
    kpts_frame: str,
) -> np.ndarray:
    T_head = _apply_T(T_map, np.asarray(f["transforms"][head_bone][frame_idx], dtype=np.float32))
    T_lw = _apply_T(T_map, np.asarray(f["transforms"]["leftHand"][frame_idx], dtype=np.float32))
    T_rw = _apply_T(T_map, np.asarray(f["transforms"]["rightHand"][frame_idx], dtype=np.float32))

    if still_head:
        still = getattr(C, "STILL_HEAD_MAT", np.eye(3, dtype=np.float32))
        T_head[:3, :3] = np.asarray(still, dtype=np.float32)

    if rel_frame != "none":
        if rel_frame == "head0":
            if inv_ref_head0 is None:
                raise ValueError("inv_ref_head0 is required when rel_frame=head0")
            inv_ref = inv_ref_head0
        elif rel_frame == "head_each":
            inv_ref = _T_inv(T_head.astype(np.float64)).astype(np.float32)
        else:
            raise ValueError(f"Unknown rel_frame: {rel_frame}")
        T_lw = (inv_ref @ T_lw).astype(np.float32)
        T_rw = (inv_ref @ T_rw).astype(np.float32)
    else:
        inv_ref = None

    head_rot6d = _matrix_to_rotation_6d(T_head[:3, :3])
    lw_rot6d = _matrix_to_rotation_6d(T_lw[:3, :3])
    rw_rot6d = _matrix_to_rotation_6d(T_rw[:3, :3])

    # 训练用 processed 数据（cet/utils.py cmd_dict2policy, human）会把头部平移置零，只保留旋转；
    # 可视化动捕时若仍置零，get_eef_kpts_from_prediction 会把头画在世界原点，而手腕在相机坐标真实位置，
    # 会出现「头在手腕下方/错位」。默认保留颈部/头骨骼平移以便 plot_keypoints 几何正确。
    scale = float(pos_scale)
    head_pos = np.zeros(3, dtype=np.float32) if zero_head_translation else (T_head[:3, 3] * scale).astype(np.float32)
    head_action = np.concatenate([head_pos, head_rot6d])
    left_wrist_action = np.concatenate([(T_lw[:3, 3] * scale).astype(np.float32), lw_rot6d])
    right_wrist_action = np.concatenate([(T_rw[:3, 3] * scale).astype(np.float32), rw_rot6d])

    if kpts_frame == "wrist":
        left_k = (_fingertips_in_wrist_frame(f, frame_idx, "leftHand", _LEFT_TIP_NAMES) * scale).astype(np.float32)
        right_k = (_fingertips_in_wrist_frame(f, frame_idx, "rightHand", _RIGHT_TIP_NAMES) * scale).astype(np.float32)
    elif kpts_frame == "head":
        inv_k = inv_ref
        if inv_k is None:
            inv_k = _T_inv(T_head.astype(np.float64)).astype(np.float32)
        left_k = (_fingertips_in_ref_frame(
            f,
            frame_idx,
            T_wrist=f["transforms"]["leftHand"][frame_idx],
            tip_names=_LEFT_TIP_NAMES,
            T_map=T_map,
            inv_ref=inv_k,
        ) * scale).astype(np.float32)
        right_k = (_fingertips_in_ref_frame(
            f,
            frame_idx,
            T_wrist=f["transforms"]["rightHand"][frame_idx],
            tip_names=_RIGHT_TIP_NAMES,
            T_map=T_map,
            inv_ref=inv_k,
        ) * scale).astype(np.float32)
    elif kpts_frame == "world":
        left_k = (_fingertips_in_ref_frame(
            f,
            frame_idx,
            T_wrist=f["transforms"]["leftHand"][frame_idx],
            tip_names=_LEFT_TIP_NAMES,
            T_map=T_map,
            inv_ref=None,
        ) * scale).astype(np.float32)
        right_k = (_fingertips_in_ref_frame(
            f,
            frame_idx,
            T_wrist=f["transforms"]["rightHand"][frame_idx],
            tip_names=_RIGHT_TIP_NAMES,
            T_map=T_map,
            inv_ref=None,
        ) * scale).astype(np.float32)
    else:
        raise ValueError(f"Unknown kpts_frame: {kpts_frame}")

    action = np.zeros(C.ACTION_STATE_VEC_SIZE, dtype=np.float32)
    action[C.OUTPUT_HEAD_EEF] = head_action
    action[C.OUTPUT_LEFT_EEF] = left_wrist_action
    action[C.OUTPUT_RIGHT_EEF] = right_wrist_action
    action[C.OUTPUT_LEFT_KEYPOINTS] = left_k.reshape(-1)
    action[C.OUTPUT_RIGHT_KEYPOINTS] = right_k.reshape(-1)
    return action


def convert_mocap_to_processed(
    src_path: str | Path,
    dst_path: str | Path,
    head_bone: str = "neck4",
    copy_root_attrs: bool = True,
    zero_head_translation: bool = False,
    pos_scale: float = 1.0,
    video_path: str | None = None,
    image_hw: tuple[int, int] = (480, 640),
    cam_to_sim: str | None = None,
    head_as_origin: bool = False,
    shift_action: bool = True,
    state_from_prev_action: bool = False,
    write_left_right: bool = False,
    compress_images: bool = False,
    sbs: bool = False,
    rel_frame: str = "none",
    still_head: bool = False,
    kpts_frame: str = "wrist",
) -> None:
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with h5py.File(src_path, "r") as f_in:
        T_map = _load_T(cam_to_sim)
        T = f_in["transforms"][head_bone].shape[0]
        required = ["leftHand", "rightHand", head_bone] + _LEFT_TIP_NAMES + _RIGHT_TIP_NAMES
        for name in required:
            if name not in f_in["transforms"]:
                raise KeyError(f"缺少 transforms/{name}，当前文件可能不是全身动捕格式")

        inv_ref_head0 = None
        if rel_frame == "head0":
            T_head0 = _apply_T(T_map, np.asarray(f_in["transforms"][head_bone][0], dtype=np.float32))
            if still_head:
                still = getattr(C, "STILL_HEAD_MAT", np.eye(3, dtype=np.float32))
                T_head0[:3, :3] = np.asarray(still, dtype=np.float32)
            inv_ref_head0 = _T_inv(T_head0.astype(np.float64)).astype(np.float32)

        actions = np.zeros((T, C.ACTION_STATE_VEC_SIZE), dtype=np.float32)
        for i in range(T):
            actions[i] = _build_action_row(
                f_in,
                i,
                head_bone,
                zero_head_translation=zero_head_translation,
                pos_scale=pos_scale,
                T_map=T_map,
                rel_frame=rel_frame,
                inv_ref_head0=inv_ref_head0,
                still_head=still_head,
                kpts_frame=kpts_frame,
            )

        if state_from_prev_action:
            actions_target = actions.copy()
            states = actions.copy()
            if T > 1:
                states[1:] = actions[:-1]
                states[0] = actions[0]
        else:
            states = actions.copy()
            actions_target = actions.copy()
            if shift_action and T > 1:
                actions_target[:-1] = actions[1:]

        if head_as_origin:
            head_t = states[:, C.OUTPUT_HEAD_EEF][:, 0:3].copy()
            states[:, C.OUTPUT_HEAD_EEF][:, 0:3] = 0.0
            states[:, C.OUTPUT_LEFT_EEF][:, 0:3] -= head_t
            states[:, C.OUTPUT_RIGHT_EEF][:, 0:3] -= head_t
            actions_target[:, C.OUTPUT_HEAD_EEF][:, 0:3] = 0.0
            actions_target[:, C.OUTPUT_LEFT_EEF][:, 0:3] -= head_t
            actions_target[:, C.OUTPUT_RIGHT_EEF][:, 0:3] -= head_t

        with h5py.File(dst_path, "w") as f_out:
            f_out.create_dataset("action", data=actions_target, compression="gzip", compression_opts=4)
            f_out.create_dataset("observation.state", data=states, compression="gzip", compression_opts=4)
            if write_left_right:
                if video_path and os.path.exists(video_path):
                    left_imgs, right_imgs = _load_video_left_right(
                        video_path, image_hw, sbs=sbs, max_frames=T
                    )
                else:
                    H, W = int(image_hw[0]), int(image_hw[1])
                    left_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
                    right_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
                if compress_images:
                    _save_compressed_imgs_hdf5(left_imgs, "observation.image.left", f_out)
                    _save_compressed_imgs_hdf5(right_imgs, "observation.image.right", f_out)
                else:
                    f_out.create_dataset("observation.image.left", data=left_imgs, compression="gzip", compression_opts=4)
                    f_out.create_dataset("observation.image.right", data=right_imgs, compression="gzip", compression_opts=4)
            else:
                H, W = int(image_hw[0]), int(image_hw[1])
                images = np.zeros((T, H, W, 3), dtype=np.uint8)
                if video_path and os.path.exists(video_path):
                    cap = cv2.VideoCapture(video_path)
                    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for i in range(min(T, video_len)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                        images[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    cap.release()
                f_out.create_dataset("observation.image.top", data=images, compression="gzip", compression_opts=4)
            
            f_out.attrs["sim"] = np.bool_(False)
            f_out.attrs["embodiment"] = "human_mocap_annotated"
            if copy_root_attrs and len(f_in.attrs):
                desc = f_in.attrs.get("description", "")
                if desc is not None and hasattr(desc, "decode"):
                    desc = desc.decode()
                f_out.attrs["description"] = str(desc) if desc else ""
            else:
                f_out.attrs["description"] = ""

    print(f"已写入 {dst_path}，action shape = {actions.shape}，包含 state 和 image.top")


def _pose_from_flat_16(flat_16: list[float]) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(flat_16, dtype=np.float32).reshape(4, 4, order="F")
    R = M[:3, :3]
    t = M[3, :3]
    return R, t


def _kpts_from_skeleton_joints(joints_flat: list[float]) -> np.ndarray:
    joints = np.asarray(joints_flat, dtype=np.float32).reshape(25, 4, 4)
    return joints[C.RETARGETTING_INDICES, :3, 3]

def _kpts_in_wrist_frame_from_skeleton_joints(joints_flat: list[float], T_wrist: np.ndarray) -> np.ndarray:
    joints = np.asarray(joints_flat, dtype=np.float64).reshape(25, 4, 4)
    inv_w = _T_inv(np.asarray(T_wrist, dtype=np.float64))
    out = np.zeros((len(C.RETARGETTING_INDICES), 3), dtype=np.float64)
    for i, j_idx in enumerate(C.VALID_RETARGETTING_INDICES, start=1):
        p = joints[int(j_idx), :3, 3]
        hom = np.array([p[0], p[1], p[2], 1.0], dtype=np.float64)
        out[i] = (inv_w @ hom)[:3]
    return out.astype(np.float32)


def convert_wholebody_json_to_processed(
    src_json_path: str | Path,
    dst_path: str | Path,
    copy_root_attrs: bool = True,
    zero_head_translation: bool = False,
    pos_scale: float = 1.0,
    video_path: str | None = None,
    image_hw: tuple[int, int] = (240, 320),
    cam_to_sim: str | None = None,
    head_as_origin: bool = False,
    shift_action: bool = True,
    state_from_prev_action: bool = False,
    write_left_right: bool = False,
    compress_images: bool = False,
    sbs: bool = False,
    rel_frame: str = "none",
    still_head: bool = False,
    kpts_frame: str = "wrist",
) -> None:
    src_json_path = Path(src_json_path)
    dst_path = Path(dst_path)
    frames = json.load(open(src_json_path, "r"))
    T = len(frames)
    T_map = _load_T(cam_to_sim)

    inv_ref_head0 = None
    if rel_frame == "head0" and T > 0:
        head_R0, head_t0 = _pose_from_flat_16(frames[0]["head"])
        T_head0 = _make_T(head_R0, head_t0)
        T_head0 = _apply_T(T_map, T_head0)
        if still_head:
            still = getattr(C, "STILL_HEAD_MAT", np.eye(3, dtype=np.float32))
            T_head0[:3, :3] = np.asarray(still, dtype=np.float32)
        inv_ref_head0 = _T_inv(T_head0.astype(np.float64)).astype(np.float32)

    actions = np.zeros((T, C.ACTION_STATE_VEC_SIZE), dtype=np.float32)
    scale = float(pos_scale)
    for i, fr in enumerate(frames):
        head_R, head_t = _pose_from_flat_16(fr["head"])
        lw_R, lw_t = _pose_from_flat_16(fr["leftWrist"])
        rw_R, rw_t = _pose_from_flat_16(fr["rightWrist"])

        T_head_raw = _make_T(head_R, head_t)
        T_lw_raw = _make_T(lw_R, lw_t)
        T_rw_raw = _make_T(rw_R, rw_t)

        if kpts_frame == "wrist":
            left_k = (_kpts_in_wrist_frame_from_skeleton_joints(fr["leftSkeleton"]["joints"], T_lw_raw) * scale).astype(np.float32)
            right_k = (_kpts_in_wrist_frame_from_skeleton_joints(fr["rightSkeleton"]["joints"], T_rw_raw) * scale).astype(np.float32)

        T_head = T_head_raw
        T_lw = T_lw_raw
        T_rw = T_rw_raw

        T_head = _apply_T(T_map, T_head)
        T_lw = _apply_T(T_map, T_lw)
        T_rw = _apply_T(T_map, T_rw)

        if still_head:
            still = getattr(C, "STILL_HEAD_MAT", np.eye(3, dtype=np.float32))
            T_head[:3, :3] = np.asarray(still, dtype=np.float32)

        if rel_frame != "none":
            if rel_frame == "head0":
                if inv_ref_head0 is None:
                    raise ValueError("inv_ref_head0 is required when rel_frame=head0")
                inv_ref = inv_ref_head0
            elif rel_frame == "head_each":
                inv_ref = _T_inv(T_head.astype(np.float64)).astype(np.float32)
            else:
                raise ValueError(f"Unknown rel_frame: {rel_frame}")
            T_lw = (inv_ref @ T_lw).astype(np.float32)
            T_rw = (inv_ref @ T_rw).astype(np.float32)
        else:
            inv_ref = None

        if kpts_frame != "wrist":
            left_pts = np.asarray(fr["leftSkeleton"]["joints"], dtype=np.float64).reshape(25, 4, 4)
            right_pts = np.asarray(fr["rightSkeleton"]["joints"], dtype=np.float64).reshape(25, 4, 4)
            left_sel = left_pts[C.VALID_RETARGETTING_INDICES, :3, 3].astype(np.float64)
            right_sel = right_pts[C.VALID_RETARGETTING_INDICES, :3, 3].astype(np.float64)

            if T_map is not None:
                Rm = T_map[:3, :3].astype(np.float64)
                tm = T_map[:3, 3].astype(np.float64)
                left_sel = (left_sel @ Rm.T) + tm
                right_sel = (right_sel @ Rm.T) + tm

            if kpts_frame == "head":
                inv_k = inv_ref
                if inv_k is None:
                    inv_k = _T_inv(T_head.astype(np.float64)).astype(np.float32)
                Rh = inv_k[:3, :3].astype(np.float64)
                th = inv_k[:3, 3].astype(np.float64)
                left_sel = (left_sel @ Rh.T) + th
                right_sel = (right_sel @ Rh.T) + th
                left_palm = T_lw[:3, 3].astype(np.float32)
                right_palm = T_rw[:3, 3].astype(np.float32)
            elif kpts_frame == "world":
                left_palm = (_apply_T(T_map, T_lw_raw)[:3, 3] if T_map is not None else T_lw_raw[:3, 3]).astype(np.float32)
                right_palm = (_apply_T(T_map, T_rw_raw)[:3, 3] if T_map is not None else T_rw_raw[:3, 3]).astype(np.float32)
            else:
                raise ValueError(f"Unknown kpts_frame: {kpts_frame}")

            left_k = np.zeros((6, 3), dtype=np.float32)
            right_k = np.zeros((6, 3), dtype=np.float32)
            left_k[0] = left_palm
            right_k[0] = right_palm
            left_k[1:] = (left_sel * float(scale)).astype(np.float32)
            right_k[1:] = (right_sel * float(scale)).astype(np.float32)

        head_R = T_head[:3, :3]
        head_t = T_head[:3, 3] * scale
        lw_R = T_lw[:3, :3]
        lw_t = T_lw[:3, 3] * scale
        rw_R = T_rw[:3, :3]
        rw_t = T_rw[:3, 3] * scale

        head_rot6d = _matrix_to_rotation_6d(head_R)
        lw_rot6d = _matrix_to_rotation_6d(lw_R)
        rw_rot6d = _matrix_to_rotation_6d(rw_R)
        head_pos = np.zeros(3, dtype=np.float32) if zero_head_translation else head_t.astype(np.float32)
        head_action = np.concatenate([head_pos, head_rot6d])
        left_wrist_action = np.concatenate([lw_t.astype(np.float32), lw_rot6d])
        right_wrist_action = np.concatenate([rw_t.astype(np.float32), rw_rot6d])
        action = np.zeros(C.ACTION_STATE_VEC_SIZE, dtype=np.float32)
        action[C.OUTPUT_HEAD_EEF] = head_action
        action[C.OUTPUT_LEFT_EEF] = left_wrist_action
        action[C.OUTPUT_RIGHT_EEF] = right_wrist_action
        action[C.OUTPUT_LEFT_KEYPOINTS] = left_k.reshape(-1)
        action[C.OUTPUT_RIGHT_KEYPOINTS] = right_k.reshape(-1)
        actions[i] = action
    if state_from_prev_action:
        actions_target = actions.copy()
        states = actions.copy()
        if T > 1:
            states[1:] = actions[:-1]
            states[0] = actions[0]
    else:
        states = actions.copy()
        actions_target = actions.copy()
        if shift_action and T > 1:
            actions_target[:-1] = actions[1:]
    if head_as_origin:
        head_t = states[:, C.OUTPUT_HEAD_EEF][:, 0:3].copy()
        states[:, C.OUTPUT_HEAD_EEF][:, 0:3] = 0.0
        states[:, C.OUTPUT_LEFT_EEF][:, 0:3] -= head_t
        states[:, C.OUTPUT_RIGHT_EEF][:, 0:3] -= head_t
        actions_target[:, C.OUTPUT_HEAD_EEF][:, 0:3] = 0.0
        actions_target[:, C.OUTPUT_LEFT_EEF][:, 0:3] -= head_t
        actions_target[:, C.OUTPUT_RIGHT_EEF][:, 0:3] -= head_t
    with h5py.File(dst_path, "w") as f_out:
        f_out.create_dataset("action", data=actions_target, compression="gzip", compression_opts=4)
        f_out.create_dataset("observation.state", data=states, compression="gzip", compression_opts=4)
        if write_left_right:
            if video_path and os.path.exists(video_path):
                left_imgs, right_imgs = _load_video_left_right(
                    str(video_path), image_hw, sbs=sbs, max_frames=T
                )
            else:
                H, W = int(image_hw[0]), int(image_hw[1])
                left_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
                right_imgs = np.zeros((T, 3, H, W), dtype=np.uint8)
            if compress_images:
                _save_compressed_imgs_hdf5(left_imgs, "observation.image.left", f_out)
                _save_compressed_imgs_hdf5(right_imgs, "observation.image.right", f_out)
            else:
                f_out.create_dataset("observation.image.left", data=left_imgs, compression="gzip", compression_opts=4)
                f_out.create_dataset("observation.image.right", data=right_imgs, compression="gzip", compression_opts=4)
        else:
            H, W = int(image_hw[0]), int(image_hw[1])
            images = np.zeros((T, H, W, 3), dtype=np.uint8)
            if video_path and os.path.exists(video_path):
                cap = cv2.VideoCapture(str(video_path))
                video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                for i in range(min(T, video_len)):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
                    images[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cap.release()
            f_out.create_dataset("observation.image.top", data=images, compression="gzip", compression_opts=4)
        f_out.attrs["sim"] = np.bool_(False)
        f_out.attrs["embodiment"] = "human_mocap_annotated"
        if copy_root_attrs:
            f_out.attrs["description"] = ""
        else:
            f_out.attrs["description"] = ""


def main() -> None:
    p = argparse.ArgumentParser(
        description="动捕数据 -> 包含训练所需 action, state, image.top 的 HDF5"
    )
    p.add_argument(
        "--input",
        "-i",
        type=str,
        default=str(Path(__file__).resolve().parent / "0.hdf5"),
        help="输入：标注动捕 HDF5（含 transforms/*）或 wholebody-*.json",
    )
    p.add_argument(
        "--output",
        "-o",
        type=str,
        default=str(Path(__file__).resolve().parent / "processed_from_0.hdf5"),
        help="输出：训练用 HDF5（包含 action, observation.state, observation.image.top）",
    )
    p.add_argument(
        "--format",
        type=str,
        default="auto",
        choices=["auto", "hdf5", "wholebody_json"],
        help="输入格式",
    )
    p.add_argument(
        "--cam-to-sim",
        type=str,
        default=None,
        help="可选：4x4 外参，将输入坐标系变换到 MuJoCo/机器人坐标系；支持 'default'、.npy、.json、或 16 个浮点数",
    )
    p.add_argument(
        "--rel-frame",
        type=str,
        default="none",
        choices=["none", "head0", "head_each"],
        help="将左右手腕位姿表达在相对参考系下：none=原始；head0=相对首帧头部参考系；head_each=相对每帧头部参考系",
    )
    p.add_argument(
        "--still-head",
        action="store_true",
        help="将头部旋转固定为 STILL_HEAD_MAT（更接近 post_process_zed 的常用设置）",
    )
    p.add_argument(
        "--head-as-origin",
        action="store_true",
        help="将 head 平移作为原点（head 平移置零，wrist 平移减去 head 平移），更适合 RobotController 的 head_pos 设定",
    )
    p.add_argument(
        "--no-head-as-origin",
        action="store_true",
        help="覆盖 --for-mujoco 的默认行为：不做 head-as-origin 平移处理",
    )
    p.add_argument(
        "--no-shift-action",
        action="store_true",
        help="不把 action 向前 shift 一帧（用于 MuJoCo replay 更直观；训练数据建议不要加）",
    )
    p.add_argument(
        "--state-from-prev-action",
        action="store_true",
        help="将 observation.state 设为 action 的上一时刻（使 action[t] == state[t+1]，对齐官方 human_avp 语义）",
    )
    p.add_argument(
        "--for-mujoco",
        action="store_true",
        help="快捷模式：等价于 --cam-to-sim default --head-as-origin --no-shift-action --write-left-right --compress-images",
    )
    p.add_argument(
        "--video",
        "-v",
        type=str,
        default=None,
        help="可选：对应的视频文件路径（用于提取图像观测）",
    )
    p.add_argument(
        "--head-bone",
        type=str,
        default="neck4",
        help="用作头部朝向的骨骼（无独立 head 时用颈骨，如 neck4）",
    )
    p.add_argument("--no-copy-attrs", action="store_true", help="不从输入复制 description 等根属性")
    p.add_argument(
        "--zero-head-translation",
        action="store_true",
        help="头部平移置零（训练时通常设为 True，以头为坐标原点）",
    )
    p.add_argument(
        "--no-zero-head-translation",
        action="store_true",
        help="覆盖 --for-mujoco 的默认行为：保留 head 平移",
    )
    p.add_argument(
        "--pos-scale",
        type=float,
        default=1.0,
        help="平移/关键点的尺度缩放（例如原数据为毫米则设为 0.001）",
    )
    p.add_argument(
        "--kpts-frame",
        type=str,
        default=None,
        choices=["wrist", "head", "world"],
        help="手部关键点的参考系：wrist=手腕坐标；head=头部坐标；world=原始/映射后的世界坐标",
    )
    p.add_argument(
        "--image-hw",
        type=int,
        nargs=2,
        default=[240, 320],
        help="图像高宽",
    )
    p.add_argument("--write-left-right", action="store_true", help="写入 observation.image.left/right（否则写 observation.image.top）")
    p.add_argument("--compress-images", action="store_true", help="将图像压缩为 JPG 并写成 (T,max_len) uint8（类似 post_process_zed 输出）")
    p.add_argument("--sbs", action="store_true", help="视频为左右拼接（side-by-side stereo），会自动切半为 left/right")
    args = p.parse_args()

    fmt = args.format
    if fmt == "auto":
        if str(args.input).lower().endswith(".json"):
            fmt = "wholebody_json"
        else:
            fmt = "hdf5"
    if fmt == "wholebody_json":
        kpts_frame = args.kpts_frame
        if kpts_frame is None:
            kpts_frame = "wrist"
        convert_wholebody_json_to_processed(
            args.input,
            args.output,
            copy_root_attrs=not args.no_copy_attrs,
            zero_head_translation=bool((args.zero_head_translation or args.for_mujoco) and not args.no_zero_head_translation),
            pos_scale=args.pos_scale,
            video_path=args.video,
            image_hw=(args.image_hw[0], args.image_hw[1]),
            cam_to_sim=("default" if args.for_mujoco and args.cam_to_sim is None else args.cam_to_sim),
            head_as_origin=bool((args.head_as_origin or args.for_mujoco) and not args.no_head_as_origin),
            shift_action=not (args.no_shift_action or args.for_mujoco),
            state_from_prev_action=bool(args.state_from_prev_action),
            write_left_right=(args.write_left_right or args.for_mujoco),
            compress_images=(args.compress_images or args.for_mujoco),
            sbs=args.sbs,
            rel_frame=args.rel_frame,
            still_head=args.still_head,
            kpts_frame=kpts_frame,
        )
    else:
        kpts_frame = args.kpts_frame
        if kpts_frame is None:
            kpts_frame = "wrist"
        convert_mocap_to_processed(
            args.input,
            args.output,
            head_bone=args.head_bone,
            copy_root_attrs=not args.no_copy_attrs,
            zero_head_translation=bool((args.zero_head_translation or args.for_mujoco) and not args.no_zero_head_translation),
            pos_scale=args.pos_scale,
            video_path=args.video,
            image_hw=(args.image_hw[0], args.image_hw[1]),
            cam_to_sim=("default" if args.for_mujoco and args.cam_to_sim is None else args.cam_to_sim),
            head_as_origin=bool((args.head_as_origin or args.for_mujoco) and not args.no_head_as_origin),
            shift_action=not (args.no_shift_action or args.for_mujoco),
            state_from_prev_action=bool(args.state_from_prev_action),
            write_left_right=(args.write_left_right or args.for_mujoco),
            compress_images=(args.compress_images or args.for_mujoco),
            sbs=args.sbs,
            rel_frame=args.rel_frame,
            still_head=args.still_head,
            kpts_frame=kpts_frame,
        )


if __name__ == "__main__":
    main()
