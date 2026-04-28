#!/usr/bin/env python3
"""
Xperience annotation.hdf5 -> human_policy_lr 训练/可视化 HDF5（processed_episode 格式）

用法:
  python3 data/xperience_to_processed.py \
      -i /path/to/021c.../ep1 \
      -o /path/to/021c.../ep1/processed \
      --split segments|whole \
      --rel-frame none --kpts-frame wrist --no-zero-head-translation \
      --write-left-right --compress-images --image-hw 240 320

说明:
  - 头部: Ts_world_cpf (quat_wxyz + trans)  -> 4x4
  - 手腕: hand_mocap/{l,r}_mano_hand_global_orient (9) + joints_3d[:,0,:] (世界 wrist 原点)
  - 指尖: hand_mocap/{l,r}_joints_3d[:, [4,8,12,16,20], :]
          顺序 = [thumb, index, middle, ring, little]（HOMIE-toolkit MANO parent chain）
  - 128-dim action 与 processed_episode 一致（hdt.constants）
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
from pathlib import Path

import cv2
import h5py
import numpy as np

import hdt.constants as C


MANO_FINGERTIP_INDICES = np.array([4, 8, 12, 16, 20], dtype=np.int64)
# MANO parent chain (HOMIE-toolkit / Xperience):
#   wrist(0) -> thumb(1..4) -> index(5..8) -> middle(9..12) -> ring(13..16) -> pinky(17..20)


def _fill_nan_frames(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Forward/backward fill invalid frames along axis 0.

    Invalid = NaN in any element, OR entire frame is all-zero
    (MANO estimator uses all-zero as a second "hand not visible" sentinel).
    Returns (filled_arr, invalid_frame_mask).
    """
    arr = arr.copy()
    flat = arr.reshape(arr.shape[0], -1)
    nan_mask  = np.isnan(flat).any(axis=1)
    zero_mask = np.all(flat == 0, axis=1)
    mask = nan_mask | zero_mask
    if not mask.any():
        return arr, mask
    valid_idx = np.where(~mask)[0]
    if valid_idx.size == 0:
        return arr, mask
    last_valid = valid_idx[0]
    for i in range(arr.shape[0]):
        if mask[i]:
            flat[i] = flat[last_valid]
        else:
            last_valid = i
    first_valid = valid_idx[0]
    for i in range(first_valid):
        flat[i] = flat[first_valid]
    return flat.reshape(arr.shape), mask


def _matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)


def _T_inv(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def _quat_wxyz_to_R(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    out = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = 1 - 2 * (y * y + z * z)
    out[..., 0, 1] = 2 * (x * y - z * w)
    out[..., 0, 2] = 2 * (x * z + y * w)
    out[..., 1, 0] = 2 * (x * y + z * w)
    out[..., 1, 1] = 1 - 2 * (x * x + z * z)
    out[..., 1, 2] = 2 * (y * z - x * w)
    out[..., 2, 0] = 2 * (x * z - y * w)
    out[..., 2, 1] = 2 * (y * z + x * w)
    out[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return out


def _head_mats_from_cpf(cpf: np.ndarray) -> np.ndarray:
    T = cpf.shape[0]
    R = _quat_wxyz_to_R(cpf[:, :4]).astype(np.float32)
    mats = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    mats[:, :3, :3] = R
    mats[:, :3, 3] = cpf[:, 4:7].astype(np.float32)
    return mats


def _wrist_mats_from_mano(global_orient_9: np.ndarray, joints_3d: np.ndarray) -> np.ndarray:
    T = global_orient_9.shape[0]
    mats = np.tile(np.eye(4, dtype=np.float32), (T, 1, 1))
    mats[:, :3, :3] = global_orient_9.reshape(T, 3, 3).astype(np.float32)
    mats[:, :3, 3] = joints_3d[:, 0, :].astype(np.float32)
    return mats


def _slugify(s: str, max_len: int = 60) -> str:
    s = re.sub(r"[^\w\s-]", "", (s or "").strip().lower())
    s = re.sub(r"[\s_-]+", "_", s)
    return s[:max_len] or "seg"


def _parse_segments(caption_obj: dict, ts_ns: np.ndarray) -> list[dict]:
    ts_int = [int(x) for x in ts_ns]
    out = []
    for s in caption_obj.get("segments", []):
        sf = int(s["start_frame"])
        ef = int(s["end_frame"])
        si = bisect.bisect_left(ts_int, sf)
        ei = bisect.bisect_left(ts_int, ef)
        si = max(0, min(si, len(ts_int) - 1))
        ei = max(0, min(ei, len(ts_int) - 1))
        out.append({
            "id": int(s.get("segment_id", len(out))),
            "start": int(si),
            "end": int(ei) + 1,
            "sub_task": str(s.get("Sub Task", "")),
            "current_action": s.get("Current Action", None),
        })
    return out


def _compute_action_row(
    T_head_w: np.ndarray,
    T_lw_w: np.ndarray,
    T_rw_w: np.ndarray,
    lk_world: np.ndarray,
    rk_world: np.ndarray,
    *,
    rel_frame: str,
    inv_ref_head0: np.ndarray | None,
    still_head: bool,
    kpts_frame: str,
    zero_head_translation: bool,
    pos_scale: float,
) -> np.ndarray:
    T_head = T_head_w.copy()
    if still_head:
        T_head[:3, :3] = np.asarray(C.STILL_HEAD_MAT, dtype=np.float32)

    if rel_frame == "head0":
        if inv_ref_head0 is None:
            raise ValueError("inv_ref_head0 required for rel_frame=head0")
        inv_ref = inv_ref_head0
    elif rel_frame == "head_each":
        inv_ref = _T_inv(T_head.astype(np.float64)).astype(np.float32)
    elif rel_frame == "none":
        inv_ref = None
    else:
        raise ValueError(f"rel_frame={rel_frame}")

    if inv_ref is not None:
        T_lw = (inv_ref @ T_lw_w).astype(np.float32)
        T_rw = (inv_ref @ T_rw_w).astype(np.float32)
    else:
        T_lw = T_lw_w.astype(np.float32)
        T_rw = T_rw_w.astype(np.float32)

    def _xform_pts(Tm: np.ndarray, pts: np.ndarray) -> np.ndarray:
        Rm = Tm[:3, :3].astype(np.float64)
        tm = Tm[:3, 3].astype(np.float64)
        return (pts.astype(np.float64) @ Rm.T + tm).astype(np.float32)

    if kpts_frame == "wrist":
        # 指尖在原始 wrist 本地系；与 rel_frame 无关
        inv_lw_w = _T_inv(T_lw_w.astype(np.float64)).astype(np.float32)
        inv_rw_w = _T_inv(T_rw_w.astype(np.float64)).astype(np.float32)
        lk = _xform_pts(inv_lw_w, lk_world)
        rk = _xform_pts(inv_rw_w, rk_world)
        left_k6 = np.zeros((6, 3), dtype=np.float32)   # palm = 0 in wrist frame
        right_k6 = np.zeros((6, 3), dtype=np.float32)
        left_k6[1:] = lk
        right_k6[1:] = rk
    elif kpts_frame == "head":
        inv_k = inv_ref if inv_ref is not None else _T_inv(T_head.astype(np.float64)).astype(np.float32)
        lk = _xform_pts(inv_k, lk_world)
        rk = _xform_pts(inv_k, rk_world)
        left_k6 = np.zeros((6, 3), dtype=np.float32)
        right_k6 = np.zeros((6, 3), dtype=np.float32)
        left_k6[0] = T_lw[:3, 3]
        right_k6[0] = T_rw[:3, 3]
        left_k6[1:] = lk
        right_k6[1:] = rk
    elif kpts_frame == "world":
        left_k6 = np.zeros((6, 3), dtype=np.float32)
        right_k6 = np.zeros((6, 3), dtype=np.float32)
        left_k6[0] = T_lw_w[:3, 3]
        right_k6[0] = T_rw_w[:3, 3]
        left_k6[1:] = lk_world.astype(np.float32)
        right_k6[1:] = rk_world.astype(np.float32)
    else:
        raise ValueError(f"kpts_frame={kpts_frame}")

    scale = float(pos_scale)
    head_pos = (
        np.zeros(3, dtype=np.float32)
        if zero_head_translation
        else (T_head[:3, 3] * scale).astype(np.float32)
    )
    head_rot6d = _matrix_to_rotation_6d(T_head[:3, :3])
    lw_pos = (T_lw[:3, 3] * scale).astype(np.float32)
    rw_pos = (T_rw[:3, 3] * scale).astype(np.float32)
    lw_rot6d = _matrix_to_rotation_6d(T_lw[:3, :3])
    rw_rot6d = _matrix_to_rotation_6d(T_rw[:3, :3])

    action = np.zeros(C.ACTION_STATE_VEC_SIZE, dtype=np.float32)
    action[C.OUTPUT_HEAD_EEF] = np.concatenate([head_pos, head_rot6d])
    action[C.OUTPUT_LEFT_EEF] = np.concatenate([lw_pos, lw_rot6d])
    action[C.OUTPUT_RIGHT_EEF] = np.concatenate([rw_pos, rw_rot6d])
    action[C.OUTPUT_LEFT_KEYPOINTS] = (left_k6 * scale).reshape(-1)
    action[C.OUTPUT_RIGHT_KEYPOINTS] = (right_k6 * scale).reshape(-1)
    return action


def _compute_actions_for_range(
    T_heads_w: np.ndarray,
    T_lws_w: np.ndarray,
    T_rws_w: np.ndarray,
    lk_world_all: np.ndarray,  # (T, 5, 3)
    rk_world_all: np.ndarray,
    start: int,
    end: int,
    *,
    rel_frame: str,
    still_head: bool,
    kpts_frame: str,
    zero_head_translation: bool,
    pos_scale: float,
) -> np.ndarray:
    T = end - start
    inv_ref_head0 = None
    if rel_frame == "head0":
        T_head0 = T_heads_w[start].copy()
        if still_head:
            T_head0[:3, :3] = np.asarray(C.STILL_HEAD_MAT, dtype=np.float32)
        inv_ref_head0 = _T_inv(T_head0.astype(np.float64)).astype(np.float32)

    actions = np.zeros((T, C.ACTION_STATE_VEC_SIZE), dtype=np.float32)
    for i in range(T):
        idx = start + i
        actions[i] = _compute_action_row(
            T_heads_w[idx], T_lws_w[idx], T_rws_w[idx],
            lk_world_all[idx], rk_world_all[idx],
            rel_frame=rel_frame,
            inv_ref_head0=inv_ref_head0,
            still_head=still_head,
            kpts_frame=kpts_frame,
            zero_head_translation=zero_head_translation,
            pos_scale=pos_scale,
        )
    return actions


def _encode_stereo_jpg(
    left_path: str,
    right_path: str,
    n_frames: int,
    image_hw: tuple[int, int],
    jpeg_quality: int = 50,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    H, W = image_hw
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
    capL = cv2.VideoCapture(str(left_path))
    capR = cv2.VideoCapture(str(right_path))
    if not capL.isOpened() or not capR.isOpened():
        capL.release(); capR.release()
        raise RuntimeError(f"failed to open stereo videos: {left_path}, {right_path}")
    encL: list[np.ndarray] = []
    encR: list[np.ndarray] = []
    for _ in range(n_frames):
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not (okL and okR):
            break
        fL = cv2.resize(fL, (W, H), interpolation=cv2.INTER_AREA)
        fR = cv2.resize(fR, (W, H), interpolation=cv2.INTER_AREA)
        _, bL = cv2.imencode(".jpg", fL, encode_param)
        _, bR = cv2.imencode(".jpg", fR, encode_param)
        encL.append(bL)
        encR.append(bR)
    # 缺帧补零图（保证与 action 对齐）
    while len(encL) < n_frames:
        zero = np.zeros((H, W, 3), dtype=np.uint8)
        _, bZ = cv2.imencode(".jpg", zero, encode_param)
        encL.append(bZ)
        encR.append(bZ)
    capL.release()
    capR.release()
    return encL, encR


def _decode_stereo_raw(
    left_path: str,
    right_path: str,
    n_frames: int,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    H, W = image_hw
    left = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    right = np.zeros((n_frames, H, W, 3), dtype=np.uint8)
    capL = cv2.VideoCapture(str(left_path))
    capR = cv2.VideoCapture(str(right_path))
    if not capL.isOpened() or not capR.isOpened():
        capL.release(); capR.release()
        raise RuntimeError(f"failed to open stereo videos: {left_path}, {right_path}")
    i = 0
    while i < n_frames:
        okL, fL = capL.read()
        okR, fR = capR.read()
        if not (okL and okR):
            break
        fL = cv2.resize(fL, (W, H), interpolation=cv2.INTER_AREA)
        fR = cv2.resize(fR, (W, H), interpolation=cv2.INTER_AREA)
        left[i] = cv2.cvtColor(fL, cv2.COLOR_BGR2RGB)
        right[i] = cv2.cvtColor(fR, cv2.COLOR_BGR2RGB)
        i += 1
    capL.release()
    capR.release()
    return left, right


def _write_encoded_jpg_dataset(hf: h5py.File, key: str, enc_list: list[np.ndarray]) -> None:
    num = len(enc_list)
    lens = [int(len(e)) for e in enc_list]
    max_len = int(max(lens)) if lens else 0
    ds = hf.create_dataset(key, (num, max_len), dtype=np.uint8)
    for i, e in enumerate(enc_list):
        ds[i, : lens[i]] = e.flatten()


def _write_one_hdf5(
    dst_path: Path,
    actions: np.ndarray,
    *,
    shift_action: bool,
    state_from_prev_action: bool,
    head_as_origin: bool,
    description: str,
    embodiment: str,
    encL: list[np.ndarray] | None,
    encR: list[np.ndarray] | None,
    rawL: np.ndarray | None,
    rawR: np.ndarray | None,
    image_hw: tuple[int, int],
    write_left_right: bool,
    segment_meta: dict | None = None,
) -> None:
    T = actions.shape[0]

    if state_from_prev_action:
        states = actions.copy()
        actions_target = actions.copy()
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

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    H, W = image_hw
    with h5py.File(dst_path, "w") as fo:
        fo.create_dataset("action", data=actions_target, compression="gzip", compression_opts=4)
        fo.create_dataset("observation.state", data=states, compression="gzip", compression_opts=4)

        if write_left_right:
            if encL is not None:
                _write_encoded_jpg_dataset(fo, "observation.image.left", encL)
                _write_encoded_jpg_dataset(fo, "observation.image.right", encR)
            elif rawL is not None:
                fo.create_dataset("observation.image.left", data=rawL, compression="gzip", compression_opts=4)
                fo.create_dataset("observation.image.right", data=rawR, compression="gzip", compression_opts=4)
            else:
                zeros = np.zeros((T, H, W, 3), dtype=np.uint8)
                fo.create_dataset("observation.image.left", data=zeros, compression="gzip", compression_opts=4)
                fo.create_dataset("observation.image.right", data=zeros, compression="gzip", compression_opts=4)
        else:
            images = np.zeros((T, H, W, 3), dtype=np.uint8)
            fo.create_dataset("observation.image.top", data=images, compression="gzip", compression_opts=4)

        fo.attrs["sim"] = np.bool_(False)
        fo.attrs["embodiment"] = embodiment
        fo.attrs["description"] = description
        if segment_meta is not None:
            fo.attrs["segment_id"] = int(segment_meta.get("id", -1))
            fo.attrs["segment_start_frame"] = int(segment_meta.get("start", 0))
            fo.attrs["segment_end_frame"] = int(segment_meta.get("end", T))
            fo.attrs["sub_task"] = str(segment_meta.get("sub_task", ""))


def convert(
    ep_dir: Path,
    out_dir: Path,
    *,
    split: str,
    rel_frame: str,
    kpts_frame: str,
    zero_head_translation: bool,
    still_head: bool,
    pos_scale: float,
    shift_action: bool,
    state_from_prev_action: bool,
    head_as_origin: bool,
    write_left_right: bool,
    compress_images: bool,
    image_hw: tuple[int, int],
    min_segment_frames: int,
    embodiment: str,
    stereo_left_name: str,
    stereo_right_name: str,
) -> None:
    ann_path = ep_dir / "annotation.hdf5"
    if not ann_path.exists():
        raise FileNotFoundError(f"annotation.hdf5 not found in {ep_dir}")

    left_video = ep_dir / stereo_left_name
    right_video = ep_dir / stereo_right_name
    has_stereo = left_video.exists() and right_video.exists()

    with h5py.File(ann_path, "r") as f:
        cpf = f["full_body_mocap/Ts_world_cpf"][:]          # (T, 7) wxyz + xyz
        lgo = f["hand_mocap/left_mano_hand_global_orient"][:]   # (T, 9)
        rgo = f["hand_mocap/right_mano_hand_global_orient"][:]
        lj = f["hand_mocap/left_joints_3d"][:]              # (T, 21, 3)
        rj = f["hand_mocap/right_joints_3d"][:]
        ts_ns = f["video/device_timestamp"][:]
        cap_raw = f["caption"][()]
        if isinstance(cap_raw, bytes):
            cap_raw = cap_raw.decode()
        caption_obj = json.loads(cap_raw) if cap_raw else {}

    T = cpf.shape[0]
    main_task = str(caption_obj.get("config", {}).get("Main Task", "") or "")
    segments = _parse_segments(caption_obj, ts_ns)
    print(f"[info] frames={T}, segments={len(segments)}, main_task={main_task!r}")
    print(f"[info] stereo videos: left={left_video.name} right={right_video.name} exist={has_stereo}")

    # hand_mocap 可能含 NaN 帧（视觉遮挡），forward/backward fill
    lgo, lgo_mask = _fill_nan_frames(lgo)
    rgo, rgo_mask = _fill_nan_frames(rgo)
    lj, lj_mask = _fill_nan_frames(lj)
    rj, rj_mask = _fill_nan_frames(rj)
    nan_left = int(lj_mask.sum() | lgo_mask.sum())
    nan_right = int(rj_mask.sum() | rgo_mask.sum())
    if nan_left or nan_right:
        print(f"[info] filled NaN frames via forward/backward fill: left={nan_left}/{T}, right={nan_right}/{T}")

    T_heads_w = _head_mats_from_cpf(cpf)
    T_lws_w = _wrist_mats_from_mano(lgo, lj)
    T_rws_w = _wrist_mats_from_mano(rgo, rj)
    lk_world = lj[:, MANO_FINGERTIP_INDICES, :].astype(np.float32)  # (T, 5, 3)
    rk_world = rj[:, MANO_FINGERTIP_INDICES, :].astype(np.float32)

    # 读取 / 编码 stereo 一次
    encL = encR = rawL = rawR = None
    if write_left_right and has_stereo:
        if compress_images:
            print("[info] encoding stereo videos -> jpg in memory ...")
            encL, encR = _encode_stereo_jpg(left_video, right_video, T, image_hw)
            print(f"[info] encoded {len(encL)} frames each (jpg)")
        else:
            print("[info] decoding stereo videos -> raw uint8 ...")
            rawL, rawR = _decode_stereo_raw(left_video, right_video, T, image_hw)
            print(f"[info] decoded {rawL.shape[0]} frames each (raw)")

    if split == "whole":
        actions = _compute_actions_for_range(
            T_heads_w, T_lws_w, T_rws_w, lk_world, rk_world,
            0, T,
            rel_frame=rel_frame, still_head=still_head,
            kpts_frame=kpts_frame, zero_head_translation=zero_head_translation,
            pos_scale=pos_scale,
        )
        desc = main_task or "Xperience episode"
        _write_one_hdf5(
            out_dir / f"{ep_dir.name}_whole.hdf5",
            actions,
            shift_action=shift_action,
            state_from_prev_action=state_from_prev_action,
            head_as_origin=head_as_origin,
            description=desc,
            embodiment=embodiment,
            encL=encL, encR=encR, rawL=rawL, rawR=rawR,
            image_hw=image_hw, write_left_right=write_left_right,
            segment_meta=None,
        )
        print(f"[done] wrote whole ({T} frames) -> {out_dir / (ep_dir.name + '_whole.hdf5')}")
    elif split == "segments":
        if not segments:
            raise RuntimeError("no segments found in caption")
        skipped = 0
        for seg in segments:
            n = seg["end"] - seg["start"]
            if n < int(min_segment_frames):
                print(f"[skip] seg{seg['id']:02d} len={n} < min={min_segment_frames}")
                skipped += 1
                continue
            actions = _compute_actions_for_range(
                T_heads_w, T_lws_w, T_rws_w, lk_world, rk_world,
                seg["start"], seg["end"],
                rel_frame=rel_frame, still_head=still_head,
                kpts_frame=kpts_frame, zero_head_translation=zero_head_translation,
                pos_scale=pos_scale,
            )
            sub_slug = _slugify(seg["sub_task"]) or "seg"
            fname = f"{ep_dir.name}_seg{seg['id']:02d}_{sub_slug}.hdf5"
            seg_encL = encL[seg["start"]:seg["end"]] if encL is not None else None
            seg_encR = encR[seg["start"]:seg["end"]] if encR is not None else None
            seg_rawL = rawL[seg["start"]:seg["end"]] if rawL is not None else None
            seg_rawR = rawR[seg["start"]:seg["end"]] if rawR is not None else None
            desc = seg["sub_task"] or main_task
            _write_one_hdf5(
                out_dir / fname,
                actions,
                shift_action=shift_action,
                state_from_prev_action=state_from_prev_action,
                head_as_origin=head_as_origin,
                description=desc,
                embodiment=embodiment,
                encL=seg_encL, encR=seg_encR, rawL=seg_rawL, rawR=seg_rawR,
                image_hw=image_hw, write_left_right=write_left_right,
                segment_meta=seg,
            )
            print(f"[done] seg{seg['id']:02d} [{seg['start']}:{seg['end']}) len={n} -> {fname}")
        print(f"[summary] wrote {len(segments) - skipped} / {len(segments)} segments (skipped {skipped})")
    else:
        raise ValueError(f"split={split}")


def main() -> None:
    p = argparse.ArgumentParser(description="Xperience annotation.hdf5 -> human_policy_lr processed HDF5")
    p.add_argument("--input", "-i", type=str, required=True,
                   help="输入：episode 目录（含 annotation.hdf5, stereo_left.mp4, stereo_right.mp4）")
    p.add_argument("--output", "-o", type=str, required=True,
                   help="输出：目录路径（每个 split 文件写到该目录下）")
    p.add_argument("--split", type=str, default="whole", choices=["whole", "segments"])
    p.add_argument("--rel-frame", type=str, default="none", choices=["none", "head0", "head_each"])
    p.add_argument("--kpts-frame", type=str, default="wrist", choices=["wrist", "head", "world"])
    p.add_argument("--still-head", action="store_true")
    p.add_argument("--pos-scale", type=float, default=1.0)
    p.add_argument("--zero-head-translation", action="store_true",
                   help="头部平移置零（默认 False，保留 Xperience CPF 世界平移）")
    p.add_argument("--no-zero-head-translation", action="store_true",
                   help="显式保留头部平移（与默认一致，用于覆盖其它 flag）")
    p.add_argument("--no-shift-action", action="store_true")
    p.add_argument("--state-from-prev-action", action="store_true")
    p.add_argument("--head-as-origin", action="store_true")
    p.add_argument("--write-left-right", action="store_true", default=True,
                   help="写入 image.left/right（默认开，Xperience 有 stereo）")
    p.add_argument("--no-write-left-right", action="store_true",
                   help="关闭 left/right，改写空 image.top")
    p.add_argument("--compress-images", action="store_true", default=True,
                   help="图像 JPG 压缩存储（默认开）")
    p.add_argument("--no-compress-images", action="store_true")
    p.add_argument("--image-hw", type=int, nargs=2, default=[240, 320])
    p.add_argument("--min-segment-frames", type=int, default=0,
                   help="小于此长度的 segment 跳过（split=segments 时生效）")
    p.add_argument("--embodiment", type=str, default="human_xperience")
    p.add_argument("--stereo-left-name", type=str, default="stereo_left.mp4")
    p.add_argument("--stereo-right-name", type=str, default="stereo_right.mp4")
    args = p.parse_args()

    zero_head = bool(args.zero_head_translation) and not args.no_zero_head_translation
    write_lr = bool(args.write_left_right) and not args.no_write_left_right
    compress = bool(args.compress_images) and not args.no_compress_images

    convert(
        Path(args.input).resolve(),
        Path(args.output).resolve(),
        split=args.split,
        rel_frame=args.rel_frame,
        kpts_frame=args.kpts_frame,
        zero_head_translation=zero_head,
        still_head=bool(args.still_head),
        pos_scale=float(args.pos_scale),
        shift_action=not bool(args.no_shift_action),
        state_from_prev_action=bool(args.state_from_prev_action),
        head_as_origin=bool(args.head_as_origin),
        write_left_right=write_lr,
        compress_images=compress,
        image_hw=(int(args.image_hw[0]), int(args.image_hw[1])),
        min_segment_frames=int(args.min_segment_frames),
        embodiment=str(args.embodiment),
        stereo_left_name=str(args.stereo_left_name),
        stereo_right_name=str(args.stereo_right_name),
    )


if __name__ == "__main__":
    main()
