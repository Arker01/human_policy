#!/usr/bin/env python3
"""Convert SONIC real-robot LeRobot episodes to HAT 128-D HDF5.

The SONIC exporter does not record measured base translation.  This converter
recovers an episode-relative base trajectory with stance-foot kinematic
odometry: the lower foot is treated as stationary, contact switches are
debounced, and the pelvis pose is reconstructed from the measured IMU
orientation and 29-DoF body configuration.

The output follows the existing human_policy Dex5 layout.  Positions are
expressed in an episode frame whose heading is the initial robot heading and
whose origin is the initial stable head position.  The recovered root is
stored in both state[89:98] and qpos state[100:107].
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import hashlib
import h5py
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from inspire_motor_mapping import CSV as INSPIRE_LUT_PATH, cmd6_to_dofpos12

INSPIRE_TRANSFORM = "dds_motor6_coupled_lut_v1_open1_right_thumb_rot_1minusfeedback"
INSPIRE_LUT_SHA256 = hashlib.sha256(Path(INSPIRE_LUT_PATH).read_bytes()).hexdigest()
# LUT: thumb,index,middle,ring,pinky. Pinocchio: index,middle,pinky,ring,thumb.
INSPIRE_LUT_TO_PIN = np.array([4, 5, 6, 7, 10, 11, 8, 9, 0, 1, 2, 3])


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent
TWIST_ROOT = WORKSPACE / "TWIST"
if str(TWIST_ROOT) not in sys.path:
    sys.path.insert(0, str(TWIST_ROOT))

from twist2_transformer_fk import KinematicsModel  # noqa: E402


VEC_SIZE = 128
HEAD = slice(0, 9)
LEFT_TIPS = slice(10, 28)
RIGHT_WRIST = slice(30, 39)
RIGHT_TIPS = slice(40, 58)
LEFT_WRIST = slice(80, 89)
ROOT = slice(89, 98)
QPOS = slice(100, 126)

BODY_XML = TWIST_ROOT / "assets/g1inspire/g1_29dof_inspire_hand.xml"
HAND_ASSET = TWIST_ROOT / "assets/inspire_hand"
LEFT_HAND_URDF = HAND_ASSET / "inspire_hand_left.urdf"
RIGHT_HAND_URDF = HAND_ASSET / "inspire_hand_right.urdf"
LEFT_TIP_NAMES = ["L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip"]
RIGHT_TIP_NAMES = ["R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"]
BODY_NAMES = [
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "L_hand_base_link",
    "R_hand_base_link",
    "head_mocap",
]


def rot6d(matrix: np.ndarray) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float32)[..., :2, :].reshape(*matrix.shape[:-2], 6)


def normalize_quaternion_series(q_xyzw: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    q = np.asarray(q_xyzw, dtype=np.float64).copy()
    for index in range(1, len(q)):
        if np.dot(q[index - 1], q[index]) < 0.0:
            q[index] *= -1.0
    if sigma > 0:
        q = gaussian_filter1d(q, sigma=sigma, axis=0, mode="nearest")
    q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-9)
    return q


def body_q_from_sonic(state43: np.ndarray) -> np.ndarray:
    """Drop the two seven-joint hand blocks from SONIC's 43-D state."""
    state43 = np.asarray(state43, dtype=np.float32)
    return np.concatenate([state43[:, :22], state43[:, 29:36]], axis=1)


def canonical_root_quaternions(root_wxyz: np.ndarray) -> np.ndarray:
    q_xyzw = np.asarray(root_wxyz, dtype=np.float64)[:, [1, 2, 3, 0]]
    q_xyzw = normalize_quaternion_series(q_xyzw)
    initial_count = min(20, len(q_xyzw))
    initial_rotation = Rotation.from_quat(q_xyzw[:initial_count]).mean()
    initial_yaw = initial_rotation.as_euler("ZYX", degrees=False)[0]
    heading_inverse = Rotation.from_euler("z", -initial_yaw)
    return (heading_inverse * Rotation.from_quat(q_xyzw)).as_quat().astype(np.float32)


def rotate_offsets(quat_xyzw: np.ndarray, offset: np.ndarray) -> np.ndarray:
    flat_q = np.asarray(quat_xyzw).reshape(-1, 4)
    flat_offset = np.broadcast_to(np.asarray(offset, dtype=np.float64), (len(flat_q), 3))
    return Rotation.from_quat(flat_q).apply(flat_offset).reshape(*quat_xyzw.shape[:-1], 3)


def choose_stance(sole_pos: np.ndarray, switch_height: float = 0.012) -> np.ndarray:
    """Choose the lower foot with hysteresis and a three-frame debounce."""
    count = len(sole_pos)
    stance = np.zeros(count, dtype=np.int8)
    current = int(np.argmin(sole_pos[0, :, 2]))
    stance[0] = current
    candidate = -1
    candidate_count = 0
    last_switch = -1000
    for t in range(1, count):
        other = 1 - current
        current_is_higher = sole_pos[t, current, 2] - sole_pos[t, other, 2]
        if current_is_higher > switch_height:
            if candidate == other:
                candidate_count += 1
            else:
                candidate = other
                candidate_count = 1
            if candidate_count >= 3 and t - last_switch >= 8:
                current = other
                last_switch = t
                candidate = -1
                candidate_count = 0
        else:
            candidate = -1
            candidate_count = 0
        stance[t] = current
    return stance


def estimate_root_translation(
    km: KinematicsModel,
    body_q: np.ndarray,
    root_quat_xyzw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Stance-foot FK odometry in the initial-heading frame."""
    smooth_q = gaussian_filter1d(body_q, sigma=1.0, axis=0, mode="nearest").astype(np.float32)
    zeros = torch.zeros((len(body_q), 3), dtype=torch.float32)
    with torch.no_grad():
        body_pos, body_rot = km.forward_kinematics(
            zeros,
            torch.from_numpy(root_quat_xyzw),
            torch.from_numpy(smooth_q),
        )
    ankle_indices = [km.body_names.index(BODY_NAMES[0]), km.body_names.index(BODY_NAMES[1])]
    ankle_pos = body_pos[:, ankle_indices].cpu().numpy().astype(np.float64)
    ankle_rot = body_rot[:, ankle_indices].cpu().numpy().astype(np.float64)

    # Centre of the four sole contact markers in the G1 MuJoCo model.
    sole_pos = ankle_pos + rotate_offsets(ankle_rot, np.array([0.035, 0.0, -0.030]))
    stance = choose_stance(sole_pos)

    root = np.zeros((len(body_q), 3), dtype=np.float64)
    anchors = np.full((2, 2), np.nan, dtype=np.float64)
    anchors[stance[0]] = sole_pos[0, stance[0], :2]
    root[0, 2] = -sole_pos[0, stance[0], 2]
    switches = []
    for t in range(1, len(body_q)):
        foot = int(stance[t])
        if stance[t] != stance[t - 1] or not np.all(np.isfinite(anchors[foot])):
            anchors[foot] = root[t - 1, :2] + sole_pos[t, foot, :2]
            switches.append(t)
        root[t, :2] = anchors[foot] - sole_pos[t, foot, :2]
        root[t, 2] = -sole_pos[t, foot, 2]

    root = gaussian_filter1d(root, sigma=1.5, axis=0, mode="nearest")
    root[:, :2] -= root[0, :2]
    height_delta = np.abs(sole_pos[:, 0, 2] - sole_pos[:, 1, 2])
    confidence = np.clip(0.70 + height_delta / 0.08, 0.70, 1.0).astype(np.float32)
    horizontal_steps = np.linalg.norm(np.diff(root[:, :2], axis=0), axis=1)
    diagnostics = {
        "stance_switches": len(switches),
        "root_xy_displacement_m": float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        "root_xy_path_length_m": float(horizontal_steps.sum()),
        "root_z_range_m": float(np.ptp(root[:, 2])),
        "max_horizontal_speed_mps": float(horizontal_steps.max(initial=0.0) * 50.0),
    }
    return root.astype(np.float32), stance, confidence, diagnostics


def load_hand_model(urdf: Path, tip_names: list[str]):
    model = pin.buildModelFromUrdf(str(urdf))
    data = model.createData()
    tip_ids = [model.getFrameId(name) for name in tip_names]
    return model, data, tip_ids


def inspire_sdk_to_q(hand6: np.ndarray) -> np.ndarray:
    """DDS [pinky,ring,middle,index,thumb_bend,thumb_rot], 1=open -> q12."""
    values = np.asarray(hand6, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError(f"Expected six finite normalized Inspire motors, got {values}")
    if np.any(values < -1e-6) or np.any(values > 1.0 + 1e-6):
        raise ValueError(f"Inspire feedback must be in [0,1], got {values}")
    return cmd6_to_dofpos12(values * 1000.0)[INSPIRE_LUT_TO_PIN]


def write_inspire_metadata(hf):
    hf.attrs["inspire_transform"] = INSPIRE_TRANSFORM
    hf.attrs["inspire_left_transform"] = "dds_motor6_coupled_lut_v1_open1"
    hf.attrs["inspire_right_transform"] = INSPIRE_TRANSFORM
    hf.attrs["inspire_right_thumb_rot_preprocess"] = (
        "right_feedback[:,5] = 1 - source_feedback[:,5]"
    )
    hf.attrs["inspire_motor_order"] = "pinky,ring,middle,index,thumb_bend,thumb_rot"
    hf.attrs["inspire_motor_open_value"] = 1.0
    hf.attrs["inspire_lut_sha256"] = INSPIRE_LUT_SHA256
    hf.attrs["inspire_tip_frame"] = "hand_base_local"


def hand_tips(hand6: np.ndarray, model, data, tip_ids: list[int]) -> np.ndarray:
    output = np.zeros((len(hand6), 5, 3), dtype=np.float32)
    for t, values in enumerate(hand6):
        pin.framesForwardKinematics(model, data, inspire_sdk_to_q(values))
        for finger, frame_id in enumerate(tip_ids):
            output[t, finger] = data.oMf[frame_id].translation.astype(np.float32)
    return output


def pack_tips(tips: np.ndarray) -> np.ndarray:
    output = np.zeros((len(tips), 6, 3), dtype=np.float32)
    output[:, 1:] = tips
    return output.reshape(len(tips), 18)


def stable_head_origin(head_pos: np.ndarray, window: int = 10):
    """Match the existing rootfix convention: mean of the first ten frames."""
    window = min(window, len(head_pos))
    return head_pos[:window].mean(axis=0), (0, window)


def write_jpeg_video(hf: h5py.File, video_path: Path, frame_count: int, quality: int) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(video_path))
    encoded = []
    last = None
    for _ in range(frame_count):
        ok, frame = capture.read()
        if not ok:
            if last is None:
                raise RuntimeError(f"Could not decode any frame from {video_path}")
            frame = last
        else:
            last = frame
        success, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not success:
            raise RuntimeError(f"JPEG encoding failed for {video_path}")
        encoded.append(jpeg.reshape(-1))
    capture.release()

    max_len = max(map(len, encoded))
    dataset = hf.create_dataset("observation.image.left", (frame_count, max_len), dtype=np.uint8)
    for t, jpeg in enumerate(encoded):
        dataset[t, : len(jpeg)] = jpeg
    # SONIC has one ego camera.  HAT receives the same storage through two hard links.
    hf["observation.image.right"] = dataset
    hf["observation.image.ego_view"] = dataset
    return frame_count, max_len


def read_tasks(dataset: Path) -> dict[int, str]:
    tasks_path = dataset / "meta/tasks.jsonl"
    task_names = {}
    if tasks_path.exists():
        for line in tasks_path.read_text().splitlines():
            item = json.loads(line)
            index = int(item.get("task_index", item.get("index", len(task_names))))
            task_names[index] = str(item.get("task", item.get("description", "")))
    return task_names


def convert_episode(
    parquet_path: Path,
    video_path: Path,
    output_path: Path,
    km: KinematicsModel,
    left_hand_model,
    right_hand_model,
    task_names: dict[int, str],
    jpeg_quality: int,
) -> dict:
    columns = [
        "observation.state",
        "observation.root_orientation",
        "observation.inspire.left_position",
        "observation.inspire.right_position",
        "task_index",
    ]
    table = pq.read_table(parquet_path, columns=columns)
    state43 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
    body_q = body_q_from_sonic(state43)
    root_wxyz = np.asarray(table["observation.root_orientation"].to_pylist(), dtype=np.float64)
    root_quat_xyzw = canonical_root_quaternions(root_wxyz)
    root_pos, stance, confidence, diagnostics = estimate_root_translation(
        km, body_q, root_quat_xyzw
    )

    with torch.no_grad():
        body_pos, body_rot = km.forward_kinematics(
            torch.from_numpy(root_pos),
            torch.from_numpy(root_quat_xyzw),
            torch.from_numpy(body_q),
        )
    body_indices = [km.body_names.index(name) for name in BODY_NAMES]
    body_pos = body_pos[:, body_indices].cpu().numpy().astype(np.float32)
    body_rot = body_rot[:, body_indices].cpu().numpy().astype(np.float32)
    left_wrist_pos, right_wrist_pos, head_pos = body_pos[:, 2], body_pos[:, 3], body_pos[:, 4]
    left_wrist_rot, right_wrist_rot, head_rot = body_rot[:, 2], body_rot[:, 3], body_rot[:, 4]

    origin, origin_window = stable_head_origin(head_pos)
    root_pos -= origin
    left_wrist_pos -= origin
    right_wrist_pos -= origin
    head_pos -= origin

    left_feedback = np.asarray(table["observation.inspire.left_position"].to_pylist(), dtype=np.float32)
    right_feedback = np.asarray(
        table["observation.inspire.right_position"].to_pylist(), dtype=np.float32
    )
    # The right Inspire telemetry reports thumb rotation in the opposite
    # convention from the model/LUT used to generate hand-local fingertips.
    right_feedback = right_feedback.copy()
    right_feedback[:, 5] = 1.0 - right_feedback[:, 5]
    model_l, data_l, tips_l = left_hand_model
    model_r, data_r, tips_r = right_hand_model
    left_tips = hand_tips(left_feedback, model_l, data_l, tips_l)
    right_tips = hand_tips(right_feedback, model_r, data_r, tips_r)

    count = len(body_q)
    states = np.zeros((count, VEC_SIZE), dtype=np.float32)
    states[:, HEAD] = np.concatenate([head_pos, rot6d(Rotation.from_quat(head_rot).as_matrix())], axis=1)
    states[:, LEFT_WRIST] = np.concatenate(
        [left_wrist_pos, rot6d(Rotation.from_quat(left_wrist_rot).as_matrix())], axis=1
    )
    states[:, RIGHT_WRIST] = np.concatenate(
        [right_wrist_pos, rot6d(Rotation.from_quat(right_wrist_rot).as_matrix())], axis=1
    )
    states[:, LEFT_TIPS] = pack_tips(left_tips)
    states[:, RIGHT_TIPS] = pack_tips(right_tips)
    states[:, ROOT] = np.concatenate(
        [root_pos, rot6d(Rotation.from_quat(root_quat_xyzw).as_matrix())], axis=1
    )

    root_quat_wxyz = root_quat_xyzw[:, [3, 0, 1, 2]]
    qpos = np.concatenate([root_pos, root_quat_wxyz, body_q[:, :19]], axis=1)
    states[:, QPOS] = qpos
    actions = np.concatenate([states[1:], states[-1:]], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    task_index = int(np.asarray(table["task_index"].to_pylist()).reshape(-1)[0])
    description = task_names.get(task_index, f"task_{task_index}")
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("observation.state", data=states, dtype=np.float32)
        hf.create_dataset("action", data=actions, dtype=np.float32)
        hf.create_dataset("root_odometry_stance", data=stance, dtype=np.int8)
        hf.create_dataset("root_odometry_confidence", data=confidence, dtype=np.float32)
        hf.create_dataset("waist_valid", data=np.ones(count, dtype=np.bool_))
        write_jpeg_video(hf, video_path, count, jpeg_quality)
        hf.attrs["sim"] = np.bool_(False)
        hf.attrs["embodiment"] = "dex5"
        hf.attrs["description"] = description
        hf.attrs["fps"] = 50.0
        hf.attrs["source_parquet"] = str(parquet_path.resolve())
        hf.attrs["source_video"] = str(video_path.resolve())
        hf.attrs["rootfix_method"] = "stance_foot_fk_odometry_head_origin"
        hf.attrs["root_odometry_is_ground_truth"] = np.bool_(False)
        hf.attrs["coordinate_convention"] = "initial_heading_frame_x_forward_y_left_z_up"
        hf.attrs["head_origin_source_mean"] = origin
        hf.attrs["head_origin_translation"] = -origin
        hf.attrs["head_origin_window"] = np.asarray(origin_window, dtype=np.int32)
        write_inspire_metadata(hf)
        for key, value in diagnostics.items():
            hf.attrs[f"root_odometry_{key}"] = value

    return {"episode": parquet_path.stem, "frames": count, **diagnostics, "output": str(output_path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=WORKSPACE / "sonic_real_teleop_20260903_curated/usable_dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE / "data/sonic_real_teleop_20260903_hat_rootfix",
    )
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-images", action="store_true", help="write HDF5 without camera datasets")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    km = KinematicsModel(str(BODY_XML), key_body_names=BODY_NAMES, device="cpu")
    left_hand_model = load_hand_model(LEFT_HAND_URDF, LEFT_TIP_NAMES)
    right_hand_model = load_hand_model(RIGHT_HAND_URDF, RIGHT_TIP_NAMES)
    task_names = read_tasks(dataset)

    parquet_files = sorted((dataset / "data/chunk-000").glob("episode_*.parquet"))
    if args.episode is not None:
        parquet_files = [dataset / "data/chunk-000" / f"episode_{args.episode:06d}.parquet"]
    if not parquet_files:
        raise FileNotFoundError(f"No episode parquet files under {dataset}")

    reports = []
    for index, parquet_path in enumerate(parquet_files, 1):
        episode_index = int(parquet_path.stem.rsplit("_", 1)[1])
        video_path = (
            dataset
            / "videos/chunk-000/observation.images.ego_view"
            / f"episode_{episode_index:06d}.mp4"
        )
        output_path = output_dir / f"SONIC_Dex5_20260903_ep_{episode_index:04d}_rootfix.hdf5"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(parquet_files)}] skip existing {output_path.name}")
            continue
        if not video_path.exists() and not args.no_images:
            raise FileNotFoundError(video_path)
        if args.no_images:
            # Keep the regular converter simple: create with a temporary one-frame video is undesirable.
            raise NotImplementedError("--no-images is reserved for diagnostics; omit it for final HDF5")
        report = convert_episode(
            parquet_path,
            video_path,
            output_path,
            km,
            left_hand_model,
            right_hand_model,
            task_names,
            args.jpeg_quality,
        )
        reports.append(report)
        print(
            f"[{index}/{len(parquet_files)}] {output_path.name}: T={report['frames']} "
            f"disp={report['root_xy_displacement_m']:.3f}m "
            f"switches={report['stance_switches']}"
        )

    report_path = output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")
    print(f"Done: {len(reports)} episodes -> {output_dir}")


if __name__ == "__main__":
    main()
