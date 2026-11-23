#!/usr/bin/env python3
"""Audit the converted G1/Dex3 corpus against the actual ACT training loader.

Read-only for all HDF5 files. Writes training_compatibility_report.json and logs.
The runtime phase uses load_data's real eight-worker DataLoaders; local process
communication must be allowed by the environment running this check.
"""
from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
import os
from pathlib import Path
import sys
import time

import cv2
import h5py
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from hdt import constants as C
from hdt.action_representation import convert_action_chunk
from hdt.data_utils_hdt import load_data
from hdt.modeling.utils import make_visual_encoder
from scripts.preprocess.convert_unitree_dex3 import (
    TASKS, Dex3Kinematics, metadata, feature_names, get_table,
    repair_isolated_telemetry, verify_file,
)


def save(path, report):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def check_schema(root, source_root):
    conversion = json.loads((root / "conversion_report.json").read_text())
    signature = conversion["conversion_signature"]
    kin = Dex3Kinematics(REPO / "assets/g1_dex3")
    tasks, all_paths = {}, set()
    pose_blocks = [C.OUTPUT_HEAD_EEF, C.OUTPUT_LEFT_EEF, C.OUTPUT_RIGHT_EEF,
                   C.OUTPUT_NECK, C.OUTPUT_WAIST]
    occupied = set(np.concatenate(pose_blocks + [C.OUTPUT_LEFT_KEYPOINTS, C.OUTPUT_RIGHT_KEYPOINTS]).tolist())
    reserved = sorted(set(range(98)) - occupied)
    for task, source_name in TASKS.items():
        source = source_root / source_name
        info, episodes = metadata(source)
        descriptions = Counter()
        lengths, repairs, distinct_frames = [], [], 0
        extrema = {key: {"min": np.full(128, np.inf), "max": np.full(128, -np.inf)}
                   for key in ["action", "observation.state"]}
        for ordinal, ep in enumerate(episodes):
            path = root / task / "episodes" / f"episode_{ep['episode_index']:06d}.hdf5"
            verify_file(path, ep["length"], ["top"], signature)
            rows = get_table(source, info, ep)
            with h5py.File(path, "r") as f:
                length = int(ep["length"])
                assert length >= 2 and f["action"].shape == f["observation.state"].shape == (length, 128), path
                assert isinstance(f.attrs["description"], str) and f.attrs["description"], path
                descriptions[str(f.attrs["description"]) ] += 1
                assert f.attrs["source_episode_index"] == ep["episode_index"], path
                assert f.attrs["source_dataset"] == source_name, path
                assert not bool(f.attrs["sim"]), path
                action, state = f["action"][()], f["observation.state"][()]
                distinct_frames += int(np.any(action != state, axis=1).sum())
                for key, array in [("action", action), ("observation.state", state)]:
                    assert array.dtype == np.float32 and np.isfinite(array).all(), (path, key)
                    assert not np.any(array[:, reserved]) and not np.any(array[:, 98:]), (path, key)
                    for point_block in [C.OUTPUT_LEFT_KEYPOINTS, C.OUTPUT_RIGHT_KEYPOINTS]:
                        points = array[:, point_block].reshape(-1, 6, 3)
                        assert not np.any(points[:, [0, 4, 5]]), (path, key)
                    for block in [C.OUTPUT_HEAD_EEF, C.OUTPUT_NECK, C.OUTPUT_WAIST]:
                        np.testing.assert_array_equal(array[:, block], np.broadcast_to(array[0, block], (length, 9)))
                    extrema[key]["min"] = np.minimum(extrema[key]["min"], array.min(0))
                    extrema[key]["max"] = np.maximum(extrema[key]["max"], array.max(0))
                for name in ["action", "observation.state"]:
                    raw = f["source." + name][()]
                    assert raw.shape == (length, 28) and raw.dtype == np.float32 and np.isfinite(raw).all(), (path, name)
                    np.testing.assert_array_equal(raw, rows[name])
                for name in ["timestamp", "frame_index", "index", "task_index"]:
                    assert f["source." + name].shape == (length,), (path, name)
                    np.testing.assert_array_equal(f["source." + name][()], rows[name])
                np.testing.assert_allclose(f["source.timestamp"][()], np.arange(length) / 30, atol=1e-4)
                assert np.all(np.diff(f["source.timestamp"][()]) > 0), path
                fixed, mask = repair_isolated_telemetry(rows["observation.state"])
                assert f["observation.state_joint_repaired"].dtype == np.bool_, path
                np.testing.assert_array_equal(f["observation.state_joint_repaired"][()], mask)
                assert f.attrs["repaired_joint_samples"] == int(mask.sum()), path
                for t, j in np.argwhere(mask):
                    repairs.append({"episode": int(ep["episode_index"]), "frame": int(t), "joint": int(j)})
                for key in ["head_tracker_valid", "waist_valid"]:
                    assert f[key].shape == (length,) and f[key].dtype == np.bool_ and not np.any(f[key][()]), (path, key)
                np.testing.assert_array_equal(f["hand_keypoint_valid"][()], [[1, 1, 1, 1, 0, 0]] * 2)
                valid = f["representation_valid"][()]
                assert valid.shape == (128,) and valid.dtype == np.bool_, path
                assert not np.any(valid[98:]), path
                selected = sorted({0, length // 2, length - 1} | set(np.where(mask.any(1))[0].tolist()))
                np.testing.assert_allclose(action[selected], kin.convert(rows["action"][selected], feature_names(info, "action")), atol=1e-7)
                np.testing.assert_allclose(state[selected], kin.convert(fixed[selected], feature_names(info, "observation.state")), atol=1e-7)
                lengths_jpeg = f["observation.image_length.top"][()]
                assert lengths_jpeg.shape == (length,) and np.all(lengths_jpeg > 0), path
                assert np.all(lengths_jpeg <= f["observation.image.top"].shape[1]), path
                assert f.attrs["image_decode_color_order"].startswith("RGB"), path
                for t in [0, length // 2, length - 1]:
                    encoded = f["observation.image.top"][t]
                    n = int(lengths_jpeg[t])
                    assert encoded[0] == 255 and encoded[1] == 216, path
                    assert encoded[n - 2] == 255 and encoded[n - 1] == 217, path
            all_paths.add(str(path.resolve()))
            lengths.append(length)
            if (ordinal + 1) % 100 == 0:
                print(f"SCHEMA {task}: {ordinal + 1}/{len(episodes)}", flush=True)
        tasks[task] = {"episodes": len(episodes), "frames": sum(lengths),
                       "min_episode_frames": min(lengths), "max_episode_frames": max(lengths),
                       "state_action_distinct_frames": distinct_frames,
                       "descriptions": dict(descriptions), "isolated_telemetry_repairs": repairs,
                       "field_extrema": {key: {name: array.tolist() for name, array in values.items()}
                                         for key, values in extrema.items()}}
    configs = {}
    for task in list(TASKS) + ["all"]:
        path = REPO / "configs/datasets" / f"unitree_g1_dex3_{task}.json"
        config = json.loads(path.read_text())
        assert set(config) == {"train", "val"}, path
        splits = {}
        for split in ["train", "val"]:
            paths = []
            for entry in config[split]:
                paths.extend(str((Path(entry["dataset_path"]) / name).resolve()) for name in entry["file_list"])
            assert len(paths) == len(set(paths)) and all(Path(p).is_file() for p in paths), (path, split)
            splits[split] = set(paths)
        assert not (splits["train"] & splits["val"]), path
        expected = all_paths if task == "all" else {p for p in all_paths if str(root / task / "episodes") in p}
        assert splits["train"] | splits["val"] == expected, path
        configs[task] = {"train_episodes": len(splits["train"]), "val_episodes": len(splits["val"]),
                         "all_files_covered": True, "no_train_val_overlap": True}
    return {"status": "pass", "episodes": len(all_paths), "frames": sum(t["frames"] for t in tasks.values()),
            "tasks": tasks, "configs": configs,
            "checks": ["float32 T-by-128 finite state/action", "all 5 pose rotations use valid 6D rows",
                       "128-D block layout and zero full-body tail", "raw joints/index/timestamps equal source Parquet",
                       "30Hz aligned timeline", "sparse FK of measured state and command action",
                       "explicit head camera and JPEG lengths", "masks and repair provenance", "train/val coverage and separation"]}


def assert_finite_batch(batch, size, chunk, future, history):
    assert len(batch) == (6 if future else 5)
    image, state, action, pad, conditioning = batch[:5]
    assert image.shape == (size, 1, 3, 240, 320) and image.dtype == torch.float32
    assert state.shape == (size, 128) and state.dtype == torch.float32
    assert action.shape == (size, chunk, 128) and action.dtype == torch.float32
    assert pad.shape == (size, chunk) and pad.dtype == torch.bool
    for tensor in [image, state, action]:
        assert torch.isfinite(tensor).all()
    if future:
        assert torch.isfinite(batch[5]).all() and batch[5].shape[0] == size
        assert conditioning["future_valid"].shape == (size,)
    if history:
        assert conditioning["hist_image"].shape == (size, 2, 1, 3, 240, 320)
        assert conditioning["hist_qpos"].shape == (size, 2, 128)
        assert torch.isfinite(conditioning["hist_qpos"]).all()
    return {"image": list(image.shape), "state": list(state.shape), "action": list(action.shape),
            "is_pad": list(pad.shape), "all_tensors_finite": True}


def check_episode_boundaries(dataset):
    checked, trimmed, resampled = 0, 0, 0
    stats = dataset.norm_stats["dex3"]
    for i, path in enumerate(dataset.dataset_paths):
        assert dataset.source_embodiment_list[i] == "dex3"
        timeline = dataset.episode_frame_indices[i]
        assert len(timeline) >= 2 and dataset.episode_len_list[i] > 0, path
        trimmed += int(dataset.valid_start_list[i] > 0)
        resampled += int(dataset.source_fps_list[i] > 30.000001)
        last = int(dataset.episode_len_list[i]) - 1
        # Near the end, exactly one real action should remain for either mode.
        sample = dataset.read_one(i, last)
        assert int((~sample[3]).sum()) == 1 and not sample[3][0] and sample[3][1:].all(), path
        assert torch.isfinite(sample[1]).all() and torch.isfinite(sample[2]).all(), path
        with h5py.File(path, "r") as f:
            anchor = f["observation.state"][timeline[last]]
            target_index = last + int(dataset.action_representation == "relative_chunk")
            target = f["action"][timeline[target_index]:timeline[target_index] + 1]
            expected = convert_action_chunk(anchor, target, dataset.action_representation)[0]
        expected = (expected - stats["action_mean"]) / (stats["action_std"] + 1e-6)
        np.testing.assert_allclose(sample[2][0].numpy(), expected, rtol=2e-5, atol=2e-5)
        checked += 1
    return {"episodes_checked": checked, "last_action_and_padding_correct": True,
            "auto_trimmed_episodes": trimmed, "fps_resampled_episodes": resampled}


def check_runtime(chunk, batch_size, batch_path):
    _, preprocessor = make_visual_encoder("ACT", {})
    checks = []
    settings = [(task, "original", False) for task in list(TASKS) + ["all"]]
    settings.append(("all", "relative_chunk", True))
    for task, mode, extended in settings:
        started = time.monotonic()
        path = REPO / "configs/datasets" / f"unitree_g1_dex3_{task}.json"
        train, val, stats = load_data(
            str(REPO), {"image_resolution_hw": [240, 320]}, str(path), ["top"], chunk,
            batch_size, batch_size, preprocessor, 0.0, slow_down_factor=1,
            future_image_enabled=extended, future_horizon=45 if extended else 0,
            future_clip_frames=4 if extended else 1,
            history_offsets=[5, 10] if extended else None, action_representation=mode)
        assert set(stats) == {"dex3"}
        normalization = {}
        for key in ["action_mean", "action_std", "qpos_mean", "qpos_std"]:
            value = stats["dex3"][key]
            assert value.shape == (128,) and value.dtype == np.float32 and np.isfinite(value).all()
            if key.endswith("std"):
                assert np.all(value >= np.float32(0.01))
            normalization[key] = {"min": float(value.min()), "max": float(value.max())}
        assert val.dataset.norm_stats is train.dataset.norm_stats
        # Deterministic boundary checks must not invoke training ColorJitter.
        boundary = {}
        if task == "all":
            old_train_flag = train.dataset.train
            train.dataset.train = False
            try:
                boundary["train"] = check_episode_boundaries(train.dataset)
                boundary["val"] = check_episode_boundaries(val.dataset)
            finally:
                train.dataset.train = old_train_flag
        batches = {}
        for split, loader in [("train", train), ("val", val)]:
            iterator = iter(loader)
            try:
                sample = next(iterator)
                batches[split] = assert_finite_batch(sample, batch_size, chunk, extended, extended)
                if split == "val" and task == "all" and mode == "original":
                    torch.save({"batch": sample, "action_representation": mode}, batch_path)
            finally:
                if hasattr(iterator, "_shutdown_workers"):
                    iterator._shutdown_workers()
        checks.append({"task": task, "action_representation": mode,
                       "train_episodes": len(train.dataset.dataset_paths), "val_episodes": len(val.dataset.dataset_paths),
                       "num_workers": train.num_workers, "chunk_size": chunk, "batch_size": batch_size,
                       "future_clip_and_history": extended, "batch_shapes": batches,
                       "normalization": normalization, "boundary_checks": boundary,
                       "elapsed_seconds": round(time.monotonic() - started, 2), "status": "pass"})
        print(f"RUNTIME PASS {task}/{mode}: {checks[-1]['elapsed_seconds']}s", flush=True)
        del train, val, iterator
        gc.collect()
    return {"status": "pass", "checks": checks, "model_batch_path": str(batch_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["schema", "runtime", "all"], default="all")
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    np.random.seed(42)
    root = REPO / "data/unitree_g1_dex3"
    report_path = root / "training_compatibility_report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    report.pop("error", None)
    report.update({"status": "running", "scope": "128-D ACT, ee control, one head camera, original and relative_chunk"})
    report["known_limitations"] = [
        {"field": "description", "task": "pick_apple", "episodes": 201,
         "issue": "Upstream label says red cup, dataset/task is apple; review before language conditioning."},
        {"field": "robot/head/waist geometry", "issue": "Fixed torso/head assumption; unknown 98:128 tail and ring/little fingers are zero."},
        {"field": "*_valid / representation_valid", "issue": "Audit masks are not automatically consumed by ACT loss; neck loss is explicitly zeroed by policy."},
        {"field": "old_action", "issue": "Not present: qpos control mode is not supported by these 128-D EE HDF5 files."},
        {"field": "flow_target", "issue": "Optional flow branch needs separately precomputed flow-target files."},
    ]
    save(report_path, report)
    try:
        if args.phase in ["schema", "all"]:
            report["schema"] = check_schema(root, REPO / "https-huggingface-co-datasets-unitreerobotics-g1-3/outputs/datasets")
            save(report_path, report)
        if args.phase in ["runtime", "all"]:
            report["runtime"] = check_runtime(args.chunk_size, args.batch_size, Path("/tmp/unitree_dex3_training_batch.pt"))
        report["status"] = "pass" if all(report.get(key, {}).get("status") == "pass" for key in ["schema", "runtime"]) else "partial"
    except BaseException as exc:
        report["status"] = "failed"
        report["error"] = repr(exc)
        save(report_path, report)
        raise
    save(report_path, report)
    print(f"{report['status']}: {report_path}", flush=True)


if __name__ == "__main__":
    main()
