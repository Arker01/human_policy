#!/usr/bin/env python3
"""Prepare immutable D0 datasets/configs/manifests for native baselines.

Run this on 4090-168.  Existing HDF5, checkpoint, and config files are never
overwritten: an existing generated file must validate byte-for-byte or the script
stops.  Observation/image datasets remain external links to the root-fixed SAIL
episodes; only the 76-D SONIC target is materialized.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq
import yaml


EXP = Path("/mnt/nvme0n1/human_policy_ckpt/sail_ablation_20261109")
LOCAL_REPO = Path("/home/nerv/qingyaoxu/GR00T-WholeBodyControl")
RELEASE = LOCAL_REPO / "gear_sonic_deploy/policy/release"
RELEASE_HASHES = {
    "model_encoder.onnx": "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
    "model_decoder.onnx": "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
}
SOURCE_DIRS = {
    "walk": Path("/mnt/nvme0n1/DATASETS/sonic_walk-far_hat_rootfix"),
    "turn": Path("/mnt/nvme0n1/DATASETS/sonic_turn-in-place_hat_rootfix"),
}
RUNS = {
    "walk_sonic_native_d0": ("walk_far", "walk", "sonic_native", 167),
    "turn_sonic_native_d0": ("turn_in_place", "turn", "sonic_native", 167),
    "walk_hat_native_d0": ("walk_far", "walk", "hat_native", 100),
    "turn_hat_native_d0": ("turn_in_place", "turn", "hat_native", 100),
}
CHECKPOINT_AUDIT = {
    "selected": "release/original",
    "selection_evidence": [
        "~/.bash_history explicitly launches start_body.sh --sonic-version original",
        "recorded teleop.encoder_input is zero because release encoder is 1762-D; low_latency is 1247-D",
        "approximate decoder replay ranks release best for both tasks",
    ],
    "exact_replay_limitation": (
        "raw episodes do not store the exact decoder body_dq/base_ang_vel inputs; "
        "the replay ranking is not a bitwise reproduction"
    ),
    "normalized_action_mae_best_offset": {
        "walk": {"release": 0.5425, "sonic_v1_1": 0.9663, "low_latency": 1.0444},
        "turn": {"release": 0.3612, "sonic_v1_1": 0.9554, "low_latency": 0.8888},
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    text = json.dumps(value, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"refusing to overwrite different file: {path}")
        return
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def read_native_action(parquet: Path) -> np.ndarray:
    table = pq.read_table(parquet, columns=[
        "action.motion_token",
        "action.inspire.left_command",
        "action.inspire.right_command",
    ])
    pieces = [np.asarray(table.column(i).to_pylist(), dtype=np.float32)
              for i in range(3)]
    action = np.concatenate(pieces, axis=1)
    if action.ndim != 2 or action.shape[1] != 76:
        raise ValueError(f"expected Nx76 native action in {parquet}, got {action.shape}")
    return action


def build_episode(source: Path, destination: Path) -> dict:
    with h5py.File(source, "r") as src:
        parquet = Path(str(src.attrs["source_parquet"]))
        n = int(src["observation.state"].shape[0])
        action = read_native_action(parquet)
        if len(action) != n:
            raise ValueError(f"row mismatch: {source}={n}, {parquet}={len(action)}")
        if destination.exists():
            with h5py.File(destination, "r") as old:
                if old["action"].shape != (n, 76):
                    raise RuntimeError(f"invalid existing generated episode: {destination}")
                if not np.array_equal(old["action"][:], action):
                    raise RuntimeError(f"different existing generated action: {destination}")
            return {"episode": destination.name, "frames": n,
                    "source_hdf5": str(source), "source_parquet": str(parquet),
                    "sha256": sha256(destination)}

        tmp = destination.with_suffix(".hdf5.tmp")
        with h5py.File(tmp, "w") as dst:
            dst.create_dataset("action", data=action, compression="gzip", shuffle=True)
            for key in src.keys():
                if key != "action":
                    dst[key] = h5py.ExternalLink(str(source), f"/{key}")
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.attrs["native_action_interface"] = "64D_post_fsq_plus_12D_inspire"
            dst.attrs["native_action_fps"] = 50.0
            dst.attrs["source_hdf5"] = str(source)
            dst.attrs["source_hdf5_sha256"] = sha256(source)
            dst.attrs["source_parquet_sha256"] = sha256(parquet)
        tmp.replace(destination)
    return {"episode": destination.name, "frames": n,
            "source_hdf5": str(source), "source_parquet": str(parquet),
            "sha256": sha256(destination)}


def config_entry(directory: Path, names: list[str]) -> dict:
    return {"dataset_path": str(directory), "type": "robot",
            "file_list": names, "source_name": directory.name}


def main() -> None:
    for name in ("configs", "manifests", "checkpoints", "logs", "status", "data_sources"):
        (EXP / name).mkdir(parents=True, exist_ok=True)

    split_manifest = json.loads((EXP / "configs/splits_manifest.json").read_text())
    split = split_manifest["splits"]
    model_base = yaml.safe_load((EXP / "configs/model_no_dropout.yaml").read_text())

    episode_rows: dict[str, list[dict]] = {}
    native_configs: dict[str, Path] = {}
    for task in ("walk", "turn"):
        source_key = f"{task}_sonic"
        out_dir = EXP / f"data_sources/sonic_native_{task}"
        out_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        cfg = {"train": [], "val": []}
        for split_name in ("train", "val"):
            names = list(split[source_key][split_name])
            for name in names:
                source = SOURCE_DIRS[task] / name
                row = build_episode(source, out_dir / name)
                row["split"] = "test" if split_name == "val" else "train"
                rows.append(row)
            cfg[split_name].append(config_entry(out_dir, names))
        cfg_path = EXP / f"configs/{task}_sonic_native_d0.dataset.json"
        atomic_json(cfg_path, cfg)
        episode_rows[task] = rows
        native_configs[task] = cfg_path

    sonic_model = json.loads(json.dumps(model_base))
    sonic_model["common"]["action_dim"] = 76
    sonic_model["common"]["action_chunk_size"] = 167
    sonic_model["common"]["chunk_fps"] = 50
    sonic_model_path = EXP / "configs/model_sonic_native.yaml"
    sonic_text = yaml.safe_dump(sonic_model, sort_keys=False)
    if sonic_model_path.exists() and sonic_model_path.read_text() != sonic_text:
        raise RuntimeError(f"refusing to overwrite different file: {sonic_model_path}")
    if not sonic_model_path.exists():
        sonic_model_path.write_text(sonic_text)

    # release uses 32-level FSQ in each scalar: {-16,...,15}/16.
    atomic_json(EXP / "configs/sonic_release_fsq_levels.json", {
        "levels_per_scalar": 32,
        "values": (np.arange(-16, 16, dtype=np.float32) / 16.0).tolist(),
        "projection": "nearest_value_per_dimension",
    })

    checkpoint_audit = dict(CHECKPOINT_AUDIT)
    checkpoint_audit["files"] = {
        "encoder": {"path": str(RELEASE / "model_encoder.onnx"),
                    "sha256": RELEASE_HASHES["model_encoder.onnx"]},
        "decoder": {"path": str(RELEASE / "model_decoder.onnx"),
                    "sha256": RELEASE_HASHES["model_decoder.onnx"]},
    }

    for run, (task_name, task, variant, chunk) in RUNS.items():
        if variant == "sonic_native":
            dataset = native_configs[task]
            model = sonic_model_path
            rows = episode_rows[task]
        else:
            dataset = EXP / f"configs/{task}_d0.dataset.json"
            model = EXP / "configs/model_no_dropout.yaml"
            rows = []
            cfg = json.loads(dataset.read_text())
            for split_name in ("train", "val"):
                for name in cfg[split_name][0]["file_list"]:
                    path = Path(cfg[split_name][0]["dataset_path"]) / name
                    with h5py.File(path, "r") as f:
                        frames = int(f["action"].shape[0])
                    rows.append({"episode": name, "frames": frames,
                                 "split": "test" if split_name == "val" else "train",
                                 "path": str(path), "sha256": sha256(path)})
        manifest = {
            "name": run, "status": "planned", "task": task_name,
            "data_pool": "D0", "variant": variant, "seed": 0,
            "steps": 100000, "dataset_config": str(dataset),
            "model_config": str(model), "episodes": rows,
            "source_summary": {
                "episodes": len(rows),
                "train_episodes": sum(x["split"] == "train" for x in rows),
                "test_episodes": sum(x["split"] == "test" for x in rows),
                "frames": sum(x["frames"] for x in rows),
            },
            "config": {
                "state_dim": 128, "action_dim": 76 if variant == "sonic_native" else 128,
                "chunk_size": chunk, "anchor_fps": 30,
                "label_fps": 50 if variant == "sonic_native" else 30,
                "action_representation": "original", "zero_state_dims": "98:128",
                "batch_size": 64, "lr": 1e-5, "kl_weight": 10,
                "geometry_supervision": False,
                "root_supervision": variant != "hat_native",
            },
            "checkpoint_dir": str(EXP / f"checkpoints/{run}_ckpt"),
            "checkpoint": str(EXP / f"checkpoints/{run}_ckpt/policy_last.ckpt"),
            "checkpoint_sha256": None,
        }
        if variant == "sonic_native":
            manifest["sonic_checkpoint_audit"] = checkpoint_audit
        atomic_json(EXP / f"manifests/{run}.json", manifest)

    atomic_json(EXP / "configs/sonic_checkpoint_audit.json", checkpoint_audit)
    print("prepared", ", ".join(RUNS))


if __name__ == "__main__":
    main()
