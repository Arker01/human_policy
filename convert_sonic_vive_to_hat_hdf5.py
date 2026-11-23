#!/usr/bin/env python3
"""Convert 2026-09-04 SONIC episodes to HAT HDF5 using the waist Vive tracker.

The recorded tracker1 translation is the primary root-translation source.  It
is transformed from SteamVR Y-up to Z-up, rotated once into the robot's initial
heading frame, and synchronized to the 50 Hz SONIC timestamps.  SteamVR samples
with dropouts or physically impossible metre-scale jumps are rejected.  Across
those gaps, stance-foot FK supplies only the local motion while the slowly
varying Vive-minus-FK correction is interpolated, keeping the trajectory
continuous without admitting the bad tracker poses.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import torch
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation

from convert_sonic_to_hat_hdf5 import (
    BODY_NAMES,
    BODY_XML,
    HEAD,
    LEFT_HAND_URDF,
    LEFT_TIP_NAMES,
    LEFT_TIPS,
    LEFT_WRIST,
    QPOS,
    RIGHT_HAND_URDF,
    RIGHT_TIP_NAMES,
    RIGHT_TIPS,
    RIGHT_WRIST,
    ROOT,
    VEC_SIZE,
    KinematicsModel,
    body_q_from_sonic,
    canonical_root_quaternions,
    estimate_root_translation,
    hand_tips,
    load_hand_model,
    pack_tips,
    read_tasks,
    rot6d,
    stable_head_origin,
    write_jpeg_video,
    write_inspire_metadata,
)


WORKSPACE = Path(__file__).resolve().parent.parent
DEFAULT_DATASET = WORKSPACE / "sonic_real_teleop_20260904_curated"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "data/sonic_real_teleop_20260904_hat_rootfix"

# SteamVR standing universe (Y up) -> a right-handed Z-up frame.
C_VIVE_TO_ZUP = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)


def load_manifest(dataset_root: Path) -> dict[int, dict]:
    path = dataset_root / "usable_dataset/meta/quality_manifest.jsonl"
    return {
        int(item["curated_episode"]): item
        for item in (json.loads(line) for line in path.read_text().splitlines() if line.strip())
    }


def select_waist_tracker(path: Path, sonic_time: np.ndarray, root_quat: np.ndarray) -> tuple[str, dict]:
    with path.open() as stream:
        metadata = json.loads(next(stream))
        samples = {name: [] for name in metadata["trackers"]}
        for line in stream:
            row = json.loads(line)
            for name in samples:
                tracker = row.get(name)
                if tracker and tracker.get("valid") and tracker.get("matrix") is not None:
                    matrix = np.asarray(tracker["matrix"], dtype=np.float64)
                    if np.isfinite(matrix).all():
                        samples[name].append((float(row["timestamp"]), matrix))
    target_time = float(metadata["sonic_start_unix"]) + sonic_time
    imu = Rotation.from_quat(root_quat)
    diagnostics, scores = {}, {}
    for name, rows in samples.items():
        diagnostics[f"{name}_valid_samples"] = len(rows)
        if len(rows) < 10:
            continue
        ts = np.array([row[0] for row in rows])
        matrices = np.stack([row[1] for row in rows])
        indices = np.clip(np.searchsorted(ts, target_time), 0, len(ts) - 1)
        previous = np.maximum(indices - 1, 0)
        indices = np.where(abs(ts[previous] - target_time) < abs(ts[indices] - target_time), previous, indices)
        supported = abs(ts[indices] - target_time) <= 0.015
        if supported.sum() < 10:
            continue
        tracker_rot = Rotation.from_matrix(matrices[indices, :3, :3])
        anchor = np.flatnonzero(supported)[0]
        # Relative rotation angles are invariant to fixed sensor mounting and world axes.
        tracker_angle = (tracker_rot[anchor].inv() * tracker_rot).magnitude()
        imu_angle = (imu[anchor].inv() * imu).magnitude()
        error = np.rad2deg(abs(tracker_angle[supported] - imu_angle[supported]))
        score = float(np.quantile(error, 0.9))
        positions = matrices[:, :3, 3]
        radius = np.linalg.norm(positions - np.median(positions, axis=0), axis=1)
        diagnostics[f"{name}_position_radius95_m"] = float(np.quantile(radius, 0.95))
        diagnostics[f"{name}_rotation95_deg"] = float(np.rad2deg(np.quantile(
            (tracker_rot.mean().inv() * tracker_rot).magnitude(), 0.95)))
        diagnostics[f"{name}_imu_angle_error90_deg"] = score
        scores[name] = score
    if not scores:
        raise RuntimeError(f"No synchronized tracker candidates in {path}")
    selected = min(scores, key=scores.get)
    if scores[selected] > 5.0:
        raise RuntimeError(f"No tracker agrees with robot IMU in {path}: {scores}")
    diagnostics["waist_tracker_selection"] = "synchronized_relative_rotation_match_to_robot_imu"
    diagnostics["vive_tracker_name"] = selected
    return selected, diagnostics


def read_tracker1(path: Path, tracker_name: str = "tracker1") -> tuple[dict, np.ndarray, np.ndarray, dict]:
    with path.open() as stream:
        metadata = json.loads(next(stream))
        timestamps, positions = [], []
        total = 0
        for line in stream:
            total += 1
            row = json.loads(line)
            tracker = row.get(tracker_name)
            if not tracker or not tracker.get("valid", False) or tracker.get("matrix") is None:
                continue
            matrix = np.asarray(tracker["matrix"], dtype=np.float64)
            timestamps.append(float(row["timestamp"]))
            positions.append(C_VIVE_TO_ZUP @ matrix[:3, 3])

    timestamps = np.asarray(timestamps, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if len(timestamps) < 2:
        raise RuntimeError(f"Not enough valid {tracker_name} samples in {path}")
    steps = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    diagnostics = {
        "tracker_samples_total": total,
        "tracker_samples_valid": len(timestamps),
        "tracker_raw_coverage": len(timestamps) / max(total, 1),
        "tracker_raw_max_step_m": float(steps.max(initial=0.0)),
        "tracker_raw_vertical_range_m": float(np.ptp(positions[:, 2])),
    }
    return metadata, timestamps, positions, diagnostics


def sample_tracker(
    timestamps: np.ndarray, positions: np.ndarray, target_times: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # Suppress sub-frame optical jitter before sampling 120 Hz at SONIC's 50 Hz.
    smooth = gaussian_filter1d(positions, sigma=3.0, axis=0, mode="nearest")
    sampled = np.column_stack(
        [np.interp(target_times, timestamps, smooth[:, axis]) for axis in range(3)]
    )

    right = np.searchsorted(timestamps, target_times, side="left")
    left = np.clip(right - 1, 0, len(timestamps) - 1)
    right = np.clip(right, 0, len(timestamps) - 1)
    nearest = np.minimum(
        np.abs(target_times - timestamps[left]), np.abs(timestamps[right] - target_times)
    )
    bracket_gap = timestamps[right] - timestamps[left]
    supported = (nearest <= 0.015) & ((bracket_gap <= 0.050) | (left == right))
    return sampled, supported


def best_planar_rotation(target_xy: np.ndarray, source_xy: np.ndarray) -> np.ndarray:
    covariance = source_xy.T @ target_xy
    u, _, vt = np.linalg.svd(covariance)
    matrix = vt.T @ u.T
    if np.linalg.det(matrix) < 0:
        vt[-1] *= -1
        matrix = vt.T @ u.T
    return matrix


def longest_false_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in mask:
        current = 0 if value else current + 1
        longest = max(longest, current)
    return longest


def fuse_vive_translation(
    fk_root: np.ndarray,
    tracker_sampled: np.ndarray,
    tracker_supported: np.ndarray,
    planar_rotation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    fk_relative = np.asarray(fk_root, dtype=np.float64) - fk_root[0]
    vive_relative = np.asarray(tracker_sampled, dtype=np.float64) - tracker_sampled[0]
    vive_relative[:, :2] = vive_relative[:, :2] @ planar_rotation.T

    residual = vive_relative - fk_relative
    step_speed = np.r_[0.0, np.linalg.norm(np.diff(vive_relative, axis=0), axis=1) * 50.0]
    measurement_valid = (
        tracker_supported
        & (np.linalg.norm(residual[:, :2], axis=1) <= 0.25)
        & (np.abs(residual[:, 2]) <= 0.12)
        & (step_speed <= 2.0)
    )
    # Keep frame zero as the common origin when the log starts a millisecond late.
    measurement_valid[0] = True
    valid_indices = np.flatnonzero(measurement_valid)
    if len(valid_indices) < 10:
        raise RuntimeError("Too few trustworthy Vive samples after gating")

    correction = np.empty_like(residual)
    frame_indices = np.arange(len(fk_root))
    for axis in range(3):
        correction[:, axis] = np.interp(
            frame_indices, valid_indices, residual[valid_indices, axis]
        )
    # Complementary fusion: Vive controls low-frequency/global translation;
    # foot FK preserves continuous short-term motion through tracker gaps.
    correction = gaussian_filter1d(correction, sigma=5.0, axis=0, mode="nearest")
    correction -= correction[0]
    fused = np.asarray(fk_root, dtype=np.float64) + correction

    vive_clean = fk_root[0] + vive_relative
    vive_clean[~measurement_valid] = np.nan
    diagnostics = {
        "vive_measurement_valid_frames": int(measurement_valid.sum()),
        "vive_measurement_valid_fraction": float(measurement_valid.mean()),
        "vive_longest_fallback_seconds": longest_false_run(measurement_valid) / 50.0,
        "vive_xy_displacement_m": float(np.linalg.norm(vive_relative[-1, :2])),
        "fused_xy_displacement_m": float(np.linalg.norm(fused[-1, :2] - fused[0, :2])),
        "fk_xy_displacement_m": float(np.linalg.norm(fk_relative[-1, :2])),
        "fused_z_range_m": float(np.ptp(fused[:, 2])),
    }
    return fused.astype(np.float32), measurement_valid, vive_clean.astype(np.float32), diagnostics


def load_episode_inputs(dataset_root: Path, manifest: dict[int, dict], km, auto_tracker: bool = False) -> list[dict]:
    episodes = []
    parquet_dir = dataset_root / "usable_dataset/data/chunk-000"
    for parquet_path in sorted(parquet_dir.glob("episode_*.parquet")):
        episode = int(parquet_path.stem.rsplit("_", 1)[1])
        columns = [
            "observation.state",
            "observation.root_orientation",
            "observation.inspire.left_position",
            "observation.inspire.right_position",
            "timestamp",
            "task_index",
        ]
        table = pq.read_table(parquet_path, columns=columns)
        state43 = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)
        body_q = body_q_from_sonic(state43)
        root_wxyz = np.asarray(table["observation.root_orientation"].to_pylist(), dtype=np.float64)
        root_quat_xyzw = canonical_root_quaternions(root_wxyz)
        fk_root, stance, confidence, fk_diagnostics = estimate_root_translation(
            km, body_q, root_quat_xyzw
        )

        sonic_time = np.asarray(table["timestamp"].to_pylist(), dtype=np.float64)
        tracker_file = manifest[episode].get("tracker_file")
        if not tracker_file:
            tracker_path = None
            tracker_name = "fk_stance"
            tracker_meta = {"trackers": {tracker_name: {"serial": "none"}}}
            tracker_sampled = np.asarray(fk_root, dtype=np.float64)
            tracker_supported = np.ones(len(fk_root), dtype=bool)
            tracker_diagnostics = {
                "tracker_samples_total": 0,
                "tracker_samples_valid": 0,
                "tracker_raw_coverage": 0.0,
                "tracker_raw_max_step_m": 0.0,
                "tracker_raw_vertical_range_m": 0.0,
                "tracker_missing_fallback": True,
            }
            raw_clean = False
        else:
            tracker_path = dataset_root / "tracker_data" / tracker_file
            try:
                tracker_name, selection_diagnostics = (select_waist_tracker(tracker_path, sonic_time, root_quat_xyzw)
                                                        if auto_tracker else ("tracker1", {}))
                tracker_meta, tracker_ts, tracker_pos, tracker_diagnostics = read_tracker1(tracker_path, tracker_name)
                tracker_diagnostics.update(selection_diagnostics)
                target_time = float(tracker_meta["sonic_start_unix"]) + sonic_time
                tracker_sampled, tracker_supported = sample_tracker(tracker_ts, tracker_pos, target_time)
                raw_clean = (
                    tracker_diagnostics["tracker_raw_coverage"] > 0.995
                    and tracker_diagnostics["tracker_raw_max_step_m"] < 0.050
                    and tracker_diagnostics["tracker_raw_vertical_range_m"] < 0.100
                )
            except (RuntimeError, KeyError, ValueError) as exc:
                tracker_name = "fk_stance"
                tracker_meta = {"trackers": {tracker_name: {"serial": "none"}}}
                tracker_sampled = np.asarray(fk_root, dtype=np.float64)
                tracker_supported = np.ones(len(fk_root), dtype=bool)
                tracker_diagnostics = {
                    "tracker_samples_total": 0,
                    "tracker_samples_valid": 0,
                    "tracker_raw_coverage": 0.0,
                    "tracker_raw_max_step_m": 0.0,
                    "tracker_raw_vertical_range_m": 0.0,
                    "tracker_missing_fallback": True,
                    "tracker_fallback_reason": str(exc),
                }
                raw_clean = False
        episodes.append(
            {
                "episode": episode,
                "parquet_path": parquet_path,
                "tracker_path": tracker_path,
                "tracker_meta": tracker_meta,
                "tracker_name": tracker_name,
                "table": table,
                "body_q": body_q,
                "root_quat_xyzw": root_quat_xyzw,
                "fk_root": fk_root,
                "stance": stance,
                "confidence": confidence,
                "fk_diagnostics": fk_diagnostics,
                "tracker_sampled": tracker_sampled,
                "tracker_supported": tracker_supported,
                "tracker_diagnostics": tracker_diagnostics,
                "raw_clean": raw_clean,
            }
        )
    return episodes


def calibrate_dataset_yaw(episodes: list[dict]) -> tuple[np.ndarray, list[int]]:
    clean = [episode for episode in episodes if episode["raw_clean"]]
    if not clean:
        return np.eye(2, dtype=np.float64), []
    target, source = [], []
    for episode in clean:
        fk = gaussian_filter1d(episode["fk_root"], sigma=5.0, axis=0, mode="nearest")
        vive = gaussian_filter1d(episode["tracker_sampled"], sigma=5.0, axis=0, mode="nearest")
        target.append(fk[:, :2] - fk[0, :2])
        source.append(vive[:, :2] - vive[0, :2])
    return best_planar_rotation(np.concatenate(target), np.concatenate(source)), [x["episode"] for x in clean]


def convert_episode(
    episode_data: dict,
    dataset_root: Path,
    output_dir: Path,
    km,
    hand_left,
    hand_right,
    task_names: dict[int, str],
    planar_rotation: np.ndarray,
    jpeg_quality: int,
    output_prefix: str = "SONIC_Dex5_20260904",
) -> dict:
    table = episode_data["table"]
    body_q = episode_data["body_q"]
    root_quat_xyzw = episode_data["root_quat_xyzw"]
    root_pos, vive_valid, vive_clean, vive_diagnostics = fuse_vive_translation(
        episode_data["fk_root"],
        episode_data["tracker_sampled"],
        episode_data["tracker_supported"],
        planar_rotation,
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
    for positions in (root_pos, vive_clean, left_wrist_pos, right_wrist_pos, head_pos):
        positions -= origin

    model_l, data_l, tips_l = hand_left
    model_r, data_r, tips_r = hand_right
    left_feedback = np.asarray(table["observation.inspire.left_position"].to_pylist(), dtype=np.float32)
    right_feedback = np.asarray(
        table["observation.inspire.right_position"].to_pylist(), dtype=np.float32
    )
    # Match the canonical Inspire/LUT convention used by the non-Vive
    # converter and by online right-hand IK.
    right_feedback = right_feedback.copy()
    right_feedback[:, 5] = 1.0 - right_feedback[:, 5]
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
    states[:, QPOS] = np.concatenate([root_pos, root_quat_wxyz, body_q[:, :19]], axis=1)
    actions = np.concatenate([states[1:], states[-1:]], axis=0)

    episode = episode_data["episode"]
    video_path = (
        dataset_root
        / "usable_dataset/videos/chunk-000/observation.images.ego_view"
        / f"episode_{episode:06d}.mp4"
    )
    output_path = output_dir / f"{output_prefix}_ep_{episode:04d}_rootfix.hdf5"
    output_dir.mkdir(parents=True, exist_ok=True)
    task_index = int(np.asarray(table["task_index"].to_pylist()).reshape(-1)[0])
    with h5py.File(output_path, "w") as hf:
        hf.create_dataset("observation.state", data=states, dtype=np.float32)
        hf.create_dataset("action", data=actions, dtype=np.float32)
        hf.create_dataset("waist_valid", data=vive_valid, dtype=np.bool_)
        hf.create_dataset("root_translation_vive_valid", data=vive_valid, dtype=np.bool_)
        hf.create_dataset("root_translation_vive_clean", data=vive_clean, dtype=np.float32)
        hf.create_dataset("root_odometry_stance", data=episode_data["stance"], dtype=np.int8)
        hf.create_dataset("root_odometry_confidence", data=episode_data["confidence"], dtype=np.float32)
        write_jpeg_video(hf, video_path, count, jpeg_quality)
        hf.attrs["sim"] = np.bool_(False)
        hf.attrs["embodiment"] = "dex5"
        hf.attrs["description"] = task_names.get(task_index, f"task_{task_index}")
        hf.attrs["fps"] = 50.0
        hf.attrs["source_parquet"] = str(episode_data["parquet_path"].resolve())
        hf.attrs["source_video"] = str(video_path.resolve())
        if episode_data["tracker_path"] is not None:
            hf.attrs["source_vive_tracker"] = str(episode_data["tracker_path"].resolve())
        tracker_name = episode_data["tracker_name"]
        hf.attrs["vive_tracker_name"] = tracker_name
        hf.attrs["vive_tracker_serial"] = episode_data["tracker_meta"]["trackers"][tracker_name]["serial"]
        hf.attrs["root_translation_source"] = (
            "stance_foot_fk_odometry_no_tracker"
            if tracker_name == "fk_stance"
            else f"vive_{tracker_name}_gated_with_stance_fk_gap_fill"
        )
        hf.attrs["root_rotation_source"] = "robot_imu"
        hf.attrs["root_odometry_is_ground_truth"] = np.bool_(False)
        hf.attrs["coordinate_convention"] = "initial_heading_frame_x_forward_y_left_z_up"
        hf.attrs["head_origin_source_mean"] = origin
        hf.attrs["head_origin_translation"] = -origin
        hf.attrs["head_origin_window"] = np.asarray(origin_window, dtype=np.int32)
        hf.attrs["vive_planar_alignment_rotation"] = planar_rotation
        write_inspire_metadata(hf)
        for diagnostics in (
            episode_data["tracker_diagnostics"],
            episode_data["fk_diagnostics"],
            vive_diagnostics,
        ):
            for key, value in diagnostics.items():
                hf.attrs[key] = value

    return {"episode": episode, "frames": count, "output": str(output_path), **episode_data["tracker_diagnostics"], **vive_diagnostics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--output-prefix", default="SONIC_Dex5_20260904")
    parser.add_argument("--calibrate-by-source-run", action="store_true")
    parser.add_argument("--auto-waist-tracker", action="store_true")
    parser.add_argument("--episodes", type=int, nargs="+", default=None,
                        help="Write only these episode IDs; retain all episodes for yaw calibration")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest = load_manifest(dataset_root)
    if args.episodes is not None and not set(args.episodes).issubset(manifest):
        parser.error(f"Unknown episode IDs: {sorted(set(args.episodes) - set(manifest))}")
    km = KinematicsModel(str(BODY_XML), key_body_names=BODY_NAMES, device="cpu")
    episodes = load_episode_inputs(dataset_root, manifest, km, args.auto_waist_tracker)
    groups = {}
    for episode_data in episodes:
        group = manifest[episode_data["episode"]]["source_run"] if args.calibrate_by_source_run else "dataset"
        episode_data["calibration_group"] = group
        groups.setdefault(group, []).append(episode_data)
    rotations, calibrations = {}, {}
    for group, group_episodes in groups.items():
        planar_rotation, calibration_episodes = calibrate_dataset_yaw(group_episodes)
        yaw_deg = float(np.degrees(np.arctan2(planar_rotation[1, 0], planar_rotation[0, 0])))
        rotations[group] = planar_rotation
        calibrations[group] = {
            "vive_planar_alignment_yaw_deg": yaw_deg,
            "yaw_calibration_episodes": calibration_episodes,
        }
        print(f"{group}: Vive planar alignment yaw={yaw_deg:.3f} deg from episodes {calibration_episodes}", flush=True)

    hand_left = load_hand_model(LEFT_HAND_URDF, LEFT_TIP_NAMES)
    hand_right = load_hand_model(RIGHT_HAND_URDF, RIGHT_TIP_NAMES)
    task_names = read_tasks(dataset_root / "usable_dataset")
    reports = []
    for index, episode_data in enumerate(episodes, 1):
        episode = episode_data["episode"]
        if args.episodes is not None and episode not in args.episodes:
            continue
        output_path = output_dir / f"{args.output_prefix}_ep_{episode:04d}_rootfix.hdf5"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(episodes)}] skip {output_path.name}")
            continue
        report = convert_episode(
            episode_data,
            dataset_root,
            output_dir,
            km,
            hand_left,
            hand_right,
            task_names,
            rotations[episode_data["calibration_group"]],
            args.jpeg_quality,
            args.output_prefix,
        )
        report["calibration_group"] = episode_data["calibration_group"]
        reports.append(report)
        print(
            f"[{index}/{len(episodes)}] ep={episode:02d} frames={report['frames']} "
            f"vive_valid={report['vive_measurement_valid_fraction']:.1%} "
            f"fallback_max={report['vive_longest_fallback_seconds']:.2f}s "
            f"disp={report['fused_xy_displacement_m']:.3f}m"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "conversion_report.json"
    report_path.write_text(
        json.dumps(
            {
                **(calibrations["dataset"] if "dataset" in calibrations else {}),
                "calibrations": calibrations,
                "episodes": reports,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n"
    )
    print(f"Done: {len(reports)} episodes -> {output_dir}")


if __name__ == "__main__":
    main()
