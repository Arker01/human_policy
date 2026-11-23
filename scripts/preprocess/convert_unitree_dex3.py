#!/usr/bin/env python3
"""Convert Unitree's LeRobot v3 G1/Dex3 tasks to this project's 128-D HDF5.

Run with the human_policy conda environment. Source Parquet/video files are
read-only. Each episode is committed atomically and can be resumed safely.
See assets/g1_dex3/README.md for coordinate and missing-channel conventions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import xml.etree.ElementTree as ET

import av
import cv2
import h5py
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hdt.action_representation import matrix_to_rotation_6d, rotation_6d_to_matrix
from hdt.constants import (
    OUTPUT_HEAD_EEF, OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF,
    OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_KEYPOINTS, OUTPUT_NECK, OUTPUT_WAIST,
)

TASKS = {
    "block_stacking": "G1_Dex3_BlockStacking_Dataset",
    "pick_apple": "G1_Dex3_PickApple_Dataset",
    "pouring": "G1_Dex3_Pouring_Dataset",
}
CAMERAS = {
    "observation.images.cam_left_high": "left",
    "observation.images.cam_right_high": "right",
    "observation.images.cam_left_wrist": "left_wrist",
    "observation.images.cam_right_wrist": "right_wrist",
}
FORMAT_VERSION = "unitree_dex3_hat_v1"
_CACHE = {}


def ensure_head_camera_alias(f):
    """Expose one explicit training view without copying the stored JPEGs.

    `left` is the head's left camera, not the left wrist camera. Validate the
    source provenance before creating `top` as a hard link to that dataset.
    Existing conflicting aliases fail instead of replacing user data.
    """
    source = f["observation.image.left"]
    if not str(source.attrs.get("source_video", "")).startswith(
        "videos/observation.images.cam_left_high/"
    ):
        raise ValueError(f"Head-left camera provenance missing or incorrect: {f.filename}")
    pairs = [("observation.image.top", "observation.image.left"),
             ("observation.image_length.top", "observation.image_length.left")]
    for alias, target in pairs:
        if target not in f:
            raise ValueError(f"Missing head-camera dataset {target}: {f.filename}")
        if alias in f and f[alias].id != f[target].id:
            raise ValueError(f"Conflicting head-camera alias {alias}: {f.filename}")
    for alias, target in pairs:
        if alias not in f:
            f[alias] = f[target]
    f.attrs["training_camera"] = "top"
    f.attrs["training_camera_source"] = "observation.images.cam_left_high"
    f.attrs["camera_aliases"] = json.dumps({"top": "left"})


def verify_head_camera_alias(f):
    source = f["observation.image.left"]
    if f["observation.image.top"].id != source.id:
        raise ValueError(f"Training top view is not head-left: {f.filename}")
    if f["observation.image_length.top"].id != f["observation.image_length.left"].id:
        raise ValueError(f"Training top JPEG lengths mismatch: {f.filename}")
    if (f.attrs.get("training_camera") != "top"
            or f.attrs.get("training_camera_source") != "observation.images.cam_left_high"
            or not str(source.attrs.get("source_video", "")).startswith(
                "videos/observation.images.cam_left_high/")):
        raise ValueError(f"Training head-camera metadata mismatch: {f.filename}")


def joint_name(feature: str) -> str:
    """Map the dataset's declared names, including asymmetric hand ordering."""
    import re
    name = feature.removeprefix("k")
    name = re.sub(r"([a-z])([A-Z])", r"\1_\2", name).lower()
    name = re.sub(r"([a-z])(\d)", r"\1_\2", name)
    return name + "_joint"


class Dex3Kinematics:
    def __init__(self, assets: Path):
        self.model = pin.buildModelFromUrdf(str(assets / "g1_29dof_with_hand_rev_1_0.urdf"))
        self.data = self.model.createData()
        self.q0 = pin.neutral(self.model)
        pin.framesForwardKinematics(self.model, self.data, self.q0)
        # Head origin: nominal depth-camera mounting position, with torso axes.
        # This is a static reference, not a measured external-camera calibration.
        self.head_origin = self.data.oMf[self.model.getFrameId("d435_link")].translation.copy()
        self.torso_origin = self.data.oMf[self.model.getFrameId("torso_link")].translation.copy()
        self.wrists = {s: self.model.getFrameId(f"{s}_wrist_yaw_link") for s in ("left", "right")}
        self.tips = {}
        # Read actual fingertip offsets from the official retargeting URDFs;
        # the full robot URDF stops at the distal phalanges.
        for side in self.wrists:
            hand = ET.parse(assets / f"unitree_dex3_{side}.urdf").getroot()
            for finger in ("thumb", "index", "middle"):
                tip = next(j for j in hand.findall("joint")
                           if j.find("child").get("link") == f"{finger}_tip")
                parent = tip.find("parent").get("link")
                offset = np.fromstring(tip.find("origin").get("xyz"), sep=" ")
                self.tips[side, finger] = (self.model.getFrameId(parent), offset)

    def indices(self, names):
        indices = []
        for name in names:
            jid = self.model.getJointId(joint_name(name))
            if jid >= self.model.njoints:
                raise ValueError(f"Unknown dataset joint: {name}")
            indices.append(self.model.joints[jid].idx_q)
        if len(indices) != 28 or len(set(indices)) != 28:
            raise ValueError(f"Expected 28 distinct joints, got {names}")
        return np.asarray(indices)

    def convert(self, joints, names):
        joints = np.asarray(joints, dtype=np.float32)
        if joints.ndim != 2 or joints.shape[1] != 28 or not np.isfinite(joints).all():
            raise ValueError(f"Invalid raw joint array: {joints.shape}")
        out = np.zeros((len(joints), 128), dtype=np.float32)
        identity = matrix_to_rotation_6d(np.eye(3))
        for block in (OUTPUT_HEAD_EEF, OUTPUT_NECK, OUTPUT_WAIST):
            out[:, block[3:]] = identity
        # Neck/head tracker is unavailable; its identity placeholder is masked.
        out[:, OUTPUT_WAIST[:3]] = self.torso_origin - self.head_origin
        indices = self.indices(names)
        for i, row in enumerate(joints):
            q = self.q0.copy()
            q[indices] = row
            pin.framesForwardKinematics(self.model, self.data, q)
            for side, pose_block, point_block in (
                ("left", OUTPUT_LEFT_EEF, OUTPUT_LEFT_KEYPOINTS),
                ("right", OUTPUT_RIGHT_EEF, OUTPUT_RIGHT_KEYPOINTS),
            ):
                wrist = self.data.oMf[self.wrists[side]]
                out[i, pose_block[:3]] = wrist.translation - self.head_origin
                out[i, pose_block[3:]] = matrix_to_rotation_6d(wrist.rotation)
                points = np.zeros((6, 3), dtype=np.float32)
                for slot, finger in enumerate(("thumb", "index", "middle"), start=1):
                    fid, offset = self.tips[side, finger]
                    world = self.data.oMf[fid].act(offset)
                    points[slot] = wrist.actInv(world)
                out[i, point_block] = points.reshape(-1)
        # No full-body measurements are available. The 98:128 tail stays zero;
        # the complete original 28-D state/action are saved separately.
        return out


def metadata(root):
    info = json.loads((root / "meta/info.json").read_text())
    paths = sorted((root / "meta/episodes").glob("*/*.parquet"))
    episodes = [row for path in paths for row in pq.read_table(path).to_pylist()]
    episodes.sort(key=lambda row: row["episode_index"])
    if len(episodes) != info["total_episodes"]:
        raise ValueError(f"Episode count mismatch in {root}")
    if sum(row["length"] for row in episodes) != info["total_frames"]:
        raise ValueError(f"Frame count mismatch in {root}")
    return info, episodes


def feature_names(info, key):
    names = info["features"][key]["names"]
    return names[0] if len(names) == 1 and isinstance(names[0], list) else names


def repair_isolated_telemetry(values):
    """Interpolate only impossible (>2pi rad) single-frame state samples.

    Repairs never cross episode boundaries or affect action commands. Non-finite,
    boundary, or consecutive corrupt samples fail rather than being guessed.
    """
    values = np.asarray(values, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Non-finite joint telemetry")
    bad = np.abs(values) > 2 * np.pi
    repaired = values.copy()
    for t, j in np.argwhere(bad):
        if t == 0 or t == len(values) - 1 or bad[t - 1, j] or bad[t + 1, j]:
            raise ValueError(f"Non-isolated corrupt telemetry at frame={t}, joint={j}")
        repaired[t, j] = (float(values[t - 1, j]) + float(values[t + 1, j])) / 2
    return repaired, bad


def video_spec(root, info, episode, key):
    prefix = f"videos/{key}/"
    path = root / info["video_path"].format(
        video_key=key, chunk_index=episode[prefix + "chunk_index"],
        file_index=episode[prefix + "file_index"])
    start = float(episode[prefix + "from_timestamp"])
    stop = float(episode[prefix + "to_timestamp"])
    fps = float(info["fps"])
    if abs((stop - start) * fps - episode["length"]) > 0.01:
        raise ValueError(f"Invalid video interval: {path}, episode {episode['episode_index']}")
    return path, start, stop


def decode_episode(path, start, length, fps, width, height, quality):
    """Seek by PTS and require every requested frame; never pad or trim video."""
    images = []
    first_frame = int(round(start * fps))
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_count = 2
        stream.thread_type = "AUTO"
        if abs(float(stream.average_rate) - fps) > 1e-4:
            raise ValueError(f"Unexpected video FPS: {path}")
        if first_frame:
            container.seek(int(round(start / float(stream.time_base))), stream=stream,
                           backward=True, any_frame=False)
        for frame in container.decode(stream):
            pts_frame = int(round(float(frame.pts * stream.time_base) * fps))
            if pts_frame < first_frame:
                continue
            if pts_frame != first_frame + len(images):
                raise ValueError(f"Video frame gap in {path}: wanted {first_frame + len(images)}, got {pts_frame}")
            rgb = cv2.resize(frame.to_ndarray(format="rgb24"), (width, height),
                             interpolation=cv2.INTER_AREA)
            # The project's current/history/future readers call cv2.imdecode
            # WITHOUT BGR2RGB. Encode the RGB array directly so they receive RGB.
            ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if not ok:
                raise RuntimeError(f"JPEG encoding failed: {path}")
            images.append(encoded.reshape(-1))
            if len(images) == length:
                break
    if len(images) != length:
        raise ValueError(f"Short video {path}: wanted {length} frames, got {len(images)}")
    return images


def get_table(root, info, episode):
    path = root / info["data_path"].format(
        chunk_index=episode["data/chunk_index"], file_index=episode["data/file_index"])
    cache_key = str(path)
    if cache_key not in _CACHE:
        tab = pq.read_table(path).combine_chunks()
        arrays = {}
        for name in tab.column_names:
            column = tab[name].chunk(0)
            if name in ("action", "observation.state"):
                arrays[name] = column.values.to_numpy().reshape(-1, 28)
            else:
                arrays[name] = column.to_numpy()
        _CACHE[cache_key] = arrays
    arrays = _CACHE[cache_key]
    lo, hi = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
    # The upstream v3 shards store a global index column. Do not assume lo is
    # a file-local row offset when converting a future multi-shard release.
    start = int(np.searchsorted(arrays["index"], lo))
    end = int(np.searchsorted(arrays["index"], hi))
    rows = {name: values[start:end] for name, values in arrays.items()}
    length = int(episode["length"])
    if (end - start != length or not np.all(rows["episode_index"] == episode["episode_index"])
            or not np.array_equal(rows["index"], np.arange(lo, hi))
            or not np.array_equal(rows["frame_index"], np.arange(length))
            or not np.allclose(rows["timestamp"], np.arange(length) / info["fps"], atol=1e-4)):
        raise ValueError(f"Parquet indexing/timestamp mismatch: {root}, {episode['episode_index']}")
    return rows


def verify_file(path, length, cameras, signature, decode_all=False):
    with h5py.File(path, "r") as f:
        if not f.attrs.get("conversion_complete", False) or f.attrs.get("conversion_signature") != signature:
            raise ValueError(f"Incomplete or incompatible output: {path}")
        if f.attrs["embodiment"] != "dex3" or f.attrs["fps"] != 30.0:
            raise ValueError(f"Invalid embodiment/fps: {path}")
        verify_head_camera_alias(f)
        for name in ("action", "observation.state"):
            values = f[name][()]
            if values.shape != (length, 128) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name}: {path}")
            if np.any(values[:, 98:128]):
                raise ValueError(f"Nonzero unknown full-body tail: {path}")
            for block in (OUTPUT_HEAD_EEF, OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF, OUTPUT_NECK, OUTPUT_WAIST):
                r6 = values[:, block[3:]]
                # Check the stored rows themselves, before Gram-Schmidt could
                # hide a malformed or transposed rotation representation.
                if not (np.allclose(np.linalg.norm(r6[:, :3], axis=1), 1, atol=1e-5)
                        and np.allclose(np.linalg.norm(r6[:, 3:], axis=1), 1, atol=1e-5)
                        and np.allclose(np.sum(r6[:, :3] * r6[:, 3:], axis=1), 0, atol=1e-5)):
                    raise ValueError(f"Invalid rotation rows: {path}")
                matrices = rotation_6d_to_matrix(r6)
                if not np.allclose(np.linalg.det(matrices), 1, atol=1e-5):
                    raise ValueError(f"Invalid rotations: {path}")
            for block in (OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_KEYPOINTS):
                points = values[:, block].reshape(-1, 6, 3)
                if np.any(points[:, [0, 4, 5]]) or np.max(np.linalg.norm(points, axis=-1)) > 0.3:
                    raise ValueError(f"Invalid finger slots/scale: {path}")
        for name in ("source.action", "source.observation.state"):
            if f[name].shape != (length, 28):
                raise ValueError(f"Missing raw joints: {path}")
        for cam in cameras:
            dataset = f[f"observation.image.{cam}"]
            if dataset.shape[0] != length or dataset.dtype != np.uint8:
                raise ValueError(f"Image frame count mismatch: {path}")
            indices = range(length) if decode_all else sorted({0, length // 2, length - 1})
            for i in indices:
                image = cv2.imdecode(dataset[i], cv2.IMREAD_COLOR)
                if image is None or image.shape != (int(f.attrs["image_height"]), int(f.attrs["image_width"]), 3):
                    raise ValueError(f"Invalid JPEG {cam}[{i}]: {path}")


def init_worker():
    cv2.setNumThreads(1)


def convert_episode(job):
    root, output, info, episode, options = job
    root, output = Path(root), Path(output)
    eid, length = int(episode["episode_index"]), int(episode["length"])
    path = output / f"episode_{eid:06d}.hdf5"
    cameras = {key: name for key, name in CAMERAS.items() if key in info["features"]}
    signature = options["signature"]
    if path.exists():
        # An additive schema upgrade for earlier conversions: no image payload
        # or source-camera naming changes, only explicit aliases and metadata.
        with h5py.File(path, "r+") as f:
            if (not f.attrs.get("conversion_complete", False)
                    or f.attrs.get("conversion_signature") != signature):
                raise ValueError(f"Incomplete or incompatible output: {path}")
            ensure_head_camera_alias(f)
        verify_file(path, length, cameras.values(), signature)
        with h5py.File(path, "r") as f:
            repairs = f["observation.state_joint_repaired"][()].sum()
        return {"episode_index": eid, "frames": length, "bytes": path.stat().st_size,
                "repaired_joint_samples": int(repairs), "resumed": True}
    rows = get_table(root, info, episode)
    if "kinematics" not in _CACHE:
        _CACHE["kinematics"] = Dex3Kinematics(REPO / "assets/g1_dex3")
    kin = _CACHE["kinematics"]
    state_names, action_names = feature_names(info, "observation.state"), feature_names(info, "action")
    repaired_state, repair_mask = repair_isolated_telemetry(rows["observation.state"])
    if np.any(np.abs(rows["action"]) > 2 * np.pi):
        raise ValueError(f"Impossible action joint angles: {root}, episode {eid}")
    state = kin.convert(repaired_state, state_names)
    action = kin.convert(rows["action"], action_names)
    temp = path.with_suffix(".hdf5.partial")
    with h5py.File(temp, "w") as f:
        f.attrs.update({
            "embodiment": "dex3", "fps": float(info["fps"]), "sim": False,
            "description": episode["tasks"][0], "source_dataset": root.name,
            "source_episode_index": eid, "source_dataset_path": str(root.resolve()),
            "conversion_version": FORMAT_VERSION, "conversion_signature": signature,
            "conversion_complete": False, "action_time_alignment": "source action[t], no additional shift",
            "position_frame": "fixed torso axes, nominal d435 mounting position as origin",
            "head_origin_pelvis": kin.head_origin, "rotation_6d_convention": "pytorch3d first two rows",
            "wrist_frame": "left/right_wrist_yaw_link (raw G1 axes)",
            "input_finger_frame": "wrist", "missing_fingers": "ring,little: zero placeholders",
            "full_body_assumption": "unobserved legs and waist fixed at neutral; head fixed to torso",
            "robot_configuration_tail": "98:128 zero; original 28 joints retained in source.*",
            "source_state_joint_names": json.dumps(state_names),
            "source_action_joint_names": json.dumps(action_names),
            "camera_mapping": json.dumps(cameras), "image_width": options["width"],
            "image_height": options["height"], "jpeg_quality": options["jpeg_quality"],
            "image_decode_color_order": "RGB (direct cv2.imdecode, project reader convention)",
            "state_repair_method": "isolated abs(q)>2pi samples: mean of adjacent valid frames",
            "repaired_joint_samples": int(repair_mask.sum()),
        })
        for name, values in {"action": action, "observation.state": state,
                             "source.action": rows["action"],
                             "source.observation.state": rows["observation.state"]}.items():
            f.create_dataset(name, data=values.astype(np.float32), compression="lzf")
        for name in ("timestamp", "frame_index", "index", "task_index"):
            f.create_dataset(f"source.{name}", data=rows[name])
        for name in ("head_tracker_valid", "waist_valid"):
            f.create_dataset(name, data=np.zeros(length, dtype=bool))
        f.create_dataset("observation.state_joint_repaired", data=repair_mask, compression="lzf")
        f.create_dataset("hand_keypoint_valid", data=np.array([[1, 1, 1, 1, 0, 0]] * 2, dtype=bool))
        valid = np.zeros(128, dtype=bool)
        valid[OUTPUT_LEFT_EEF] = valid[OUTPUT_RIGHT_EEF] = True
        valid[OUTPUT_LEFT_KEYPOINTS[:12]] = valid[OUTPUT_RIGHT_KEYPOINTS[:12]] = True
        f.create_dataset("representation_valid", data=valid)
        for key, cam in cameras.items():
            video_path, start, stop = video_spec(root, info, episode, key)
            images = decode_episode(video_path, start, length, info["fps"],
                                    options["width"], options["height"], options["jpeg_quality"])
            lengths = np.array([len(img) for img in images], dtype=np.int32)
            packed = np.zeros((length, int(lengths.max())), dtype=np.uint8)
            for i, img in enumerate(images):
                packed[i, :len(img)] = img
            ds = f.create_dataset(f"observation.image.{cam}", data=packed,
                                  chunks=(1, packed.shape[1]), compression="lzf")
            ds.attrs.update({"source_video": str(video_path.relative_to(root)),
                             "source_from_timestamp": start, "source_to_timestamp": stop,
                             "source_first_frame": int(round(start * info["fps"]))})
            f.create_dataset(f"observation.image_length.{cam}", data=lengths)
        ensure_head_camera_alias(f)
        f.attrs["conversion_complete"] = True
    verify_file(temp, length, cameras.values(), signature)
    os.replace(temp, path)
    return {"episode_index": eid, "frames": length, "bytes": path.stat().st_size,
            "repaired_joint_samples": int(repair_mask.sum()), "resumed": False}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temp, path)


def write_splits(task, episodes, output, config_root, seed, val_fraction):
    ids = np.array([row["episode_index"] for row in episodes])
    permuted = np.random.default_rng(seed).permutation(ids)
    count = max(1, int(round(len(ids) * val_fraction))) if len(ids) > 1 else 0
    val = set(int(eid) for eid in permuted[:count])
    train_names = [f"episode_{eid:06d}.hdf5" for eid in ids if eid not in val]
    val_names = [f"episode_{eid:06d}.hdf5" for eid in ids if eid in val]
    def entry(names):
        return {"dataset_path": str(output.resolve()), "type": "robot", "file_list": names}
    config = {"train": [entry(train_names)], "val": [entry(val_names)]}
    path = config_root / f"unitree_g1_dex3_{task}.json"
    write_json(path, config)
    return {"config": str(path), "train_episodes": len(train_names), "val_episodes": len(val_names),
            "train_episode_indices": [int(eid) for eid in ids if eid not in val],
            "val_episode_indices": sorted(val), "seed": seed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=REPO / "https-huggingface-co-datasets-unitreerobotics-g1-3/outputs/datasets")
    parser.add_argument("--output-root", type=Path, default=REPO / "data/unitree_g1_dex3")
    parser.add_argument("--config-root", type=Path, default=REPO / "configs/datasets")
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.width < 1 or args.height < 1 or not 1 <= args.jpeg_quality <= 100:
        parser.error("Invalid workers, image size or JPEG quality")
    if not 0 < args.val_fraction < 1:
        parser.error("val-fraction must be between 0 and 1")
    options = {"width": args.width, "height": args.height, "jpeg_quality": args.jpeg_quality}
    model_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (REPO / "assets/g1_dex3").glob("*.urdf")}
    signature_payload = {"version": FORMAT_VERSION, **options, "models": model_hashes,
                         "telemetry_repair": "isolated_abs_gt_2pi_v1"}
    options["signature"] = hashlib.sha256(json.dumps(signature_payload, sort_keys=True).encode()).hexdigest()
    report = {"format": FORMAT_VERSION, "conversion_signature": options["signature"],
              "options": options, "model_sha256": model_hashes, "tasks": {}, "errors": []}
    jobs = []
    task_meta = {}
    for task in args.tasks:
        root = args.source_root / TASKS[task]
        info, episodes = metadata(root)
        if float(info["fps"]) != 30.0:
            raise ValueError(f"Expected native 30Hz input: {root}")
        if args.limit_episodes:
            episodes = episodes[:args.limit_episodes]
        output = args.output_root / task / "episodes"
        output.mkdir(parents=True, exist_ok=True)
        task_meta[task] = (info, episodes, output)
        report["tasks"][task] = {"source": str(root.resolve()), "expected_episodes": len(episodes),
                                  "expected_frames": sum(row["length"] for row in episodes), "episodes": []}
        for episode in episodes:
            for key in CAMERAS:
                if key in info["features"]:
                    path, _, _ = video_spec(root, info, episode, key)
                    if not path.is_file():
                        raise FileNotFoundError(path)
            jobs.append((str(root), str(output), info, episode, options))
    started = time.monotonic()
    if args.verify_only:
        for root, output, info, episode, options in jobs:
            path = Path(output) / f"episode_{episode['episode_index']:06d}.hdf5"
            cams = [name for key, name in CAMERAS.items() if key in info["features"]]
            verify_file(path, episode["length"], cams, options["signature"])
        print(f"Verified {len(jobs)} episodes", flush=True)
        return
    args.output_root.mkdir(parents=True, exist_ok=True)
    report_path = args.output_root / "conversion_report.json"
    completed = 0
    with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker,
                             mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(convert_episode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            task = next(t for t, name in TASKS.items() if name == Path(job[0]).name)
            try:
                result = future.result()
                report["tasks"][task]["episodes"].append(result)
                completed += 1
                if completed % 10 == 0 or completed == len(jobs):
                    elapsed = time.monotonic() - started
                    print(f"[{completed}/{len(jobs)}] {elapsed:.1f}s, {task} episode {result['episode_index']}", flush=True)
                    report["elapsed_seconds"] = elapsed
                    write_json(report_path, report)
            except Exception as exc:
                error = {"task": task, "episode_index": job[3]["episode_index"], "error": repr(exc)}
                report["errors"].append(error)
                print(f"ERROR {error}", flush=True)
    for task, (info, episodes, output) in task_meta.items():
        summary = report["tasks"][task]
        summary["episodes"].sort(key=lambda row: row["episode_index"])
        summary["converted_episodes"] = len(summary["episodes"])
        summary["converted_frames"] = sum(row["frames"] for row in summary["episodes"])
        summary["bytes"] = sum(row["bytes"] for row in summary["episodes"])
        summary["repaired_joint_samples"] = sum(row["repaired_joint_samples"] for row in summary["episodes"])
        if summary["converted_episodes"] == len(episodes):
            summary["split"] = write_splits(task, episodes, output, args.config_root, args.seed, args.val_fraction)
    if not report["errors"]:
        combined = {"train": [], "val": []}
        for task in args.tasks:
            config = json.loads(Path(report["tasks"][task]["split"]["config"]).read_text())
            for split in combined:
                combined[split].extend(config[split])
        write_json(args.config_root / "unitree_g1_dex3_all.json", combined)
    report["elapsed_seconds"] = time.monotonic() - started
    report["status"] = "complete" if not report["errors"] else "failed"
    write_json(report_path, report)
    print(json.dumps({task: {k: v for k, v in summary.items() if k not in ("episodes", "split")}
                      for task, summary in report["tasks"].items()}, indent=2), flush=True)
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
