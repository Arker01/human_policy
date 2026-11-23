#!/usr/bin/env python3
"""Convert aligned G1 JSON/RGB recordings to HAT's bimanual Dex3 HDF5 format.

Uses the same named-joint FK and nominal head reference as convert_unitree_dex3.
The recording's color images are the head RGB view; depth/lidar are not inputs.
Source files are read-only. Completed outputs are verified before resuming.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys

import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.preprocess.convert_unitree_dex3 import Dex3Kinematics, write_json
from hdt.action_representation import rotation_6d_to_matrix
from hdt.constants import (OUTPUT_HEAD_EEF, OUTPUT_NECK, OUTPUT_WAIST,
                          OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF,
                          OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_KEYPOINTS)

VERSION = "g1_raw_json_dex3_hat_v1"
# G1 arm controller order and Dex3 DDS motor order. The left hand's last two
# fingers are MIDDLE, INDEX; the right hand's are INDEX, MIDDLE.
JOINT_NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_rows(episode):
    rows = json.loads((episode / "data.json").read_text())
    if not rows or any(r["robot_type"] != "g1" for r in rows):
        raise ValueError(f"Empty/non-G1 episode: {episode}")
    state = np.asarray([r["states"]["arm_state"] + r["states"]["hand_state"] for r in rows], np.float32)
    action = np.asarray([r["actions"]["sol_q"] + r["actions"]["left_angles"]
                         + r["actions"]["right_angles"] for r in rows], np.float32)
    timestamp = np.asarray([r["time"] for r in rows], np.float64)
    for array in [state, action]:
        if array.shape != (len(rows), 28) or not np.isfinite(array).all() or (np.abs(array) > 2*np.pi).any():
            raise ValueError(f"Invalid joint telemetry in {episode}")
    dt = np.diff(timestamp)
    if not np.isfinite(timestamp).all() or not ((dt > 0.025) & (dt < 0.045)).all():
        raise ValueError(f"Missing/non-30Hz timestamps: {episode}")
    if abs((len(rows)-1)/(timestamp[-1]-timestamp[0])-30) > 0.05:
        raise ValueError(f"Recording is not nominal 30Hz: {episode}")
    return rows, state, action, timestamp


def encode_head(episode, row):
    relative = Path(row["image"])
    if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "color":
        raise ValueError(f"Unexpected head image: {relative}")
    image = cv2.imread(str(episode / relative), cv2.IMREAD_COLOR)
    if image is None or image.shape != (480, 640, 3):
        raise ValueError(f"Missing/invalid head RGB image: {episode / relative}")
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (320, 240), interpolation=cv2.INTER_AREA)
    # HAT calls imdecode without BGR2RGB. Match the established RGB-on-decode
    # convention of convert_unitree_dex3 rather than swapping training colors.
    ok, encoded = cv2.imencode(".jpg", rgb, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise ValueError(f"JPEG encoding failed: {episode / relative}")
    return encoded.reshape(-1)


def verify(path, episode, signature, kin=None):
    rows, raw_state, raw_action, timestamp = read_rows(episode)
    kin = kin or Dex3Kinematics(REPO / "assets/g1_dex3")
    length = len(rows)
    with h5py.File(path, "r") as f:
        assert f.attrs["conversion_complete"] and f.attrs["conversion_signature"] == signature, path
        assert f.attrs["source_data_sha256"] == digest(episode / "data.json"), path
        assert f.attrs["embodiment"] == "dex3" and f.attrs["fps"] == 30.0, path
        assert f.attrs["training_camera"] == "top" and f.attrs["training_camera_source"] == "color", path
        assert f["observation.image.top"].id == f["observation.image.head"].id, path
        for key, raw in [("observation.state", raw_state), ("action", raw_action)]:
            value = f[key][()]
            assert value.shape == (length, 128) and value.dtype == np.float32 and np.isfinite(value).all(), path
            np.testing.assert_array_equal(f["source."+key][()], raw)
            np.testing.assert_allclose(value, kin.convert(raw, JOINT_NAMES), atol=1e-7)
            assert not value[:, 98:].any(), path
            for block in [OUTPUT_HEAD_EEF, OUTPUT_NECK, OUTPUT_WAIST, OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF]:
                rotation = rotation_6d_to_matrix(value[:, block[3:]])
                np.testing.assert_allclose(rotation @ rotation.transpose(0, 2, 1),
                                           np.broadcast_to(np.eye(3), rotation.shape), atol=5e-6)
            for block in [OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF, OUTPUT_LEFT_KEYPOINTS[3:12], OUTPUT_RIGHT_KEYPOINTS[3:12]]:
                assert np.ptp(value[:, block], axis=0).max() > 1e-4, (path, key, block)
        np.testing.assert_array_equal(f["source.timestamp"][()], timestamp)
        valid = f["representation_valid"][()]
        assert valid[OUTPUT_LEFT_EEF].all() and valid[OUTPUT_RIGHT_EEF].all(), path
        assert valid[13:22].all() and valid[43:52].all(), path
        np.testing.assert_array_equal(f["hand_keypoint_valid"][()], [[1,1,1,1,0,0]]*2)
        lengths = f["observation.image_length.top"][()]
        assert lengths.shape == (length,) and (lengths > 0).all(), path
        for index in sorted({0, length//2, length-1}):
            expected = encode_head(episode, rows[index])
            np.testing.assert_array_equal(f["observation.image.top"][index, :lengths[index]], expected)
            assert cv2.imdecode(expected, cv2.IMREAD_COLOR).shape == (240, 320, 3), path
    return {"episode_index": int(episode.name.split("_")[-1]), "frames": length,
            "path": str(path), "sha256": digest(path), "bytes": path.stat().st_size,
            "source_data_sha256": digest(episode / "data.json"),
            "measured_fps": float((length-1)/(timestamp[-1]-timestamp[0])),
            "both_hands_valid_and_moving": True, "full_fk_checked": True}


def convert(episode, output, signature, description, verify_only=False):
    cv2.setNumThreads(1)
    path = output / f"episode_{int(episode.name.split('_')[-1]):06d}.hdf5"
    kin = Dex3Kinematics(REPO / "assets/g1_dex3")
    if path.exists() or verify_only:
        return verify(path, episode, signature, kin)
    rows, raw_state, raw_action, timestamp = read_rows(episode)
    state, action = kin.convert(raw_state, JOINT_NAMES), kin.convert(raw_action, JOINT_NAMES)
    images = [encode_head(episode, row) for row in rows]
    lengths = np.asarray([len(img) for img in images], np.int32)
    packed = np.zeros((len(rows), int(lengths.max())), np.uint8)
    for i, img in enumerate(images):
        packed[i, :len(img)] = img
    temporary = path.with_suffix(".hdf5.partial")
    with h5py.File(temporary, "w") as f:
        f.attrs.update({"embodiment": "dex3", "fps": 30.0, "sim": False,
            "description": description, "source_dataset_path": str(episode.parent),
            "source_episode_index": int(episode.name.split("_")[-1]),
            "source_data_sha256": digest(episode / "data.json"),
            "conversion_version": VERSION, "conversion_signature": signature,
            "conversion_complete": False, "training_camera": "top", "training_camera_source": "color",
            "source_joint_names": json.dumps(JOINT_NAMES),
            "action_time_alignment": "aligned data.json action[t], no additional shift",
            "position_frame": "fixed torso axes, nominal d435 mounting position as origin",
            "head_origin_pelvis": kin.head_origin,
            "rotation_6d_convention": "pytorch3d first two rows",
            "input_finger_frame": "wrist", "wrist_frame": "left/right_wrist_yaw_link",
            "missing_fingers": "ring,little: zero placeholders; both Dex3 hands retained",
            "full_body_assumption": "same as prior Dex3 training: legs/waist neutral, head fixed to torso",
            "robot_configuration_tail": "98:128 zero; raw measured legs/IMU/odometry retained in source.*",
            "image_width": 320, "image_height": 240, "jpeg_quality": 80,
            "image_decode_color_order": "RGB (direct cv2.imdecode, project reader convention)",
            "image_preprocessing": "full 640x480 head RGB resized to 320x240; no FOV remapping without calibration"})
        arrays = {"observation.state": state, "action": action,
                  "source.observation.state": raw_state, "source.action": raw_action,
                  "source.timestamp": timestamp, "source.frame_index": np.arange(len(rows)),
                  "source.leg_state": np.asarray([r["states"]["leg_state"] for r in rows], np.float32),
                  "source.armtime": np.asarray([r["actions"]["armtime"] for r in rows]),
                  "source.iktime": np.asarray([r["actions"]["iktime"] for r in rows]),
                  "source.head_rmat": np.asarray([r["actions"]["head_rmat"] for r in rows]),
                  "source.left_pose": np.asarray([r["actions"]["left_pose"] for r in rows]),
                  "source.right_pose": np.asarray([r["actions"]["right_pose"] for r in rows]),
                  "head_tracker_valid": np.zeros(len(rows), bool),
                  "waist_valid": np.zeros(len(rows), bool),
                  "hand_keypoint_valid": np.array([[1,1,1,1,0,0]]*2, bool)}
        for section in ["imu", "odometry"]:
            for key in rows[0]["states"][section]:
                arrays[f"source.{section}.{key}"] = np.asarray([r["states"][section][key] for r in rows])
        valid = np.zeros(128, bool)
        for block in [OUTPUT_LEFT_EEF, OUTPUT_RIGHT_EEF, OUTPUT_LEFT_KEYPOINTS[:12], OUTPUT_RIGHT_KEYPOINTS[:12]]:
            valid[block] = True
        arrays["representation_valid"] = valid
        for key, value in arrays.items():
            f.create_dataset(key, data=value, compression="lzf")
        image = f.create_dataset("observation.image.head", data=packed, chunks=(1, packed.shape[1]), compression="lzf")
        image.attrs["source_images"] = "color/frame_*.jpg (head RGB)"
        f.create_dataset("observation.image_length.head", data=lengths)
        f.create_dataset("source.image_path", data=[r["image"] for r in rows], dtype=h5py.string_dtype())
        f["observation.image.top"] = image
        f["observation.image_length.top"] = f["observation.image_length.head"]
        f.attrs["conversion_complete"] = True
    record = verify(temporary, episode, signature, kin)
    os.replace(temporary, path)
    record["path"] = str(path)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    episodes = sorted(source.glob("episode_*/data.json"), key=lambda p:int(p.parent.name.split("_")[-1]))
    if len(episodes) < 2:
        raise ValueError("Need multiple episodes for disjoint train/val splits")
    output.mkdir(parents=True, exist_ok=True)
    (output/"episodes").mkdir(exist_ok=True)
    signature_data = {"version": VERSION, "converter_sha256": digest(__file__),
                      "fk_converter_sha256": digest(REPO/"scripts/preprocess/convert_unitree_dex3.py"),
                      "urdfs": {p.name:digest(p) for p in (REPO/"assets/g1_dex3").glob("*.urdf")}}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    description = json.loads((source/"metadata/metadata.json").read_text())["title"]
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(convert, p.parent, output/"episodes", signature, description, args.verify_only) for p in episodes]
        for future in as_completed(futures):
            record = future.result(); records.append(record)
            print(f"PASS episode {record['episode_index']}: {record['frames']} frames, both hands + head RGB", flush=True)
    records.sort(key=lambda r:r["episode_index"])
    ids = np.asarray([r["episode_index"] for r in records])
    val = set(np.random.default_rng(42).permutation(ids)[:max(1,round(len(ids)*0.1))].tolist())
    config = {split:[{"dataset_path": str(output/"episodes"), "type":"robot", "file_list":
                         [f"episode_{i:06d}.hdf5" for i in ids if (int(i) in val)==(split=="val")]}]
              for split in ["train","val"]}
    write_json(output/"dataset.json", config)
    archive = source.with_suffix(".zip")
    report = {"status":"pass", "format":VERSION, "signature":signature, "signature_inputs":signature_data,
              "source_root":str(source), "archive_sha256":digest(archive) if archive.exists() else None,
              "episodes":len(records), "frames":sum(r["frames"] for r in records), "files":records,
              "split_seed":42, "train_episodes":len(ids)-len(val), "val_episodes":len(val),
              "train_val_disjoint":True, "all_frames_preserved":True,
              "camera":"head color -> observation.image.top", "embodiment":"dex3",
              "both_hands_valid":True, "joint_names":JOINT_NAMES,
              "reference_geometry":"same fixed nominal head/torso reference as prior Dex3 training"}
    write_json(output/"conversion_report.json", report)
    print(json.dumps({k:v for k,v in report.items() if k not in ["files","signature_inputs","joint_names"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
