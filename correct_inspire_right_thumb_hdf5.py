#!/usr/bin/env python3
"""Correct right Inspire thumb-rotation fingertips in derived HAT HDF5 files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

CONVERTER_ROOT = Path("/root/shengyin/human_policy")
sys.path.insert(0, str(CONVERTER_ROOT))

from convert_sonic_to_hat_hdf5 import (  # noqa: E402
    RIGHT_HAND_URDF,
    RIGHT_TIP_NAMES,
    hand_tips,
    load_hand_model,
    pack_tips,
)

RIGHT_TIPS = slice(40, 58)
TRANSFORM = "dds_motor6_coupled_lut_v1_open1_right_thumb_rot_1minusfeedback"
PREPROCESS = "right_feedback[:,5] = 1 - source_feedback[:,5]"


def correct_file(path: Path, hand_model) -> tuple[int, float]:
    with h5py.File(path, "r") as hf:
        source = Path(str(hf.attrs["source_parquet"]))
        count = int(hf["observation.state"].shape[0])
        if hf["action"].shape != (count, 128):
            raise ValueError(f"Unexpected action shape in {path}: {hf['action'].shape}")

    table = pq.read_table(source, columns=["observation.inspire.right_position"])
    feedback = np.asarray(
        table["observation.inspire.right_position"].to_pylist(), dtype=np.float32
    )
    if feedback.shape != (count, 6):
        raise ValueError(
            f"Telemetry/HDF5 length mismatch for {path}: {feedback.shape} vs {(count, 6)}"
        )
    feedback = feedback.copy()
    feedback[:, 5] = 1.0 - feedback[:, 5]

    model, data, tip_ids = hand_model
    packed = pack_tips(hand_tips(feedback, model, data, tip_ids))
    action_packed = np.concatenate([packed[1:], packed[-1:]], axis=0)

    with h5py.File(path, "r+") as hf:
        previous = np.asarray(hf["observation.state"][:, RIGHT_TIPS], dtype=np.float32)
        hf["observation.state"][:, RIGHT_TIPS] = packed
        hf["action"][:, RIGHT_TIPS] = action_packed
        hf.attrs["inspire_transform"] = TRANSFORM
        hf.attrs["inspire_left_transform"] = "dds_motor6_coupled_lut_v1_open1"
        hf.attrs["inspire_right_transform"] = TRANSFORM
        hf.attrs["inspire_right_thumb_rot_preprocess"] = PREPROCESS
        hf.attrs["inspire_thumb_rot_correction_source"] = (
            "observation.inspire.right_position from source_parquet"
        )
        hf.flush()

    delta = float(np.max(np.abs(previous - packed)))
    return count, delta


def verify_file(path: Path) -> None:
    with h5py.File(path, "r") as hf:
        state = hf["observation.state"][:, RIGHT_TIPS]
        action = hf["action"][:, RIGHT_TIPS]
        if not np.allclose(action[:-1], state[1:], atol=1e-7, rtol=0.0):
            raise ValueError(f"Action/state shift check failed: {path}")
        if not np.allclose(action[-1], state[-1], atol=1e-7, rtol=0.0):
            raise ValueError(f"Final action/state check failed: {path}")
        if str(hf.attrs.get("inspire_right_thumb_rot_preprocess", "")) != PREPROCESS:
            raise ValueError(f"Correction metadata missing: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args()
    hand_model = load_hand_model(RIGHT_HAND_URDF, RIGHT_TIP_NAMES)
    total_files = total_frames = 0
    max_delta = 0.0
    for directory in args.directories:
        files = sorted(directory.glob("*.hdf5"))
        if not files:
            raise FileNotFoundError(f"No HDF5 files in {directory}")
        for index, path in enumerate(files, 1):
            frames, delta = correct_file(path, hand_model)
            verify_file(path)
            total_files += 1
            total_frames += frames
            max_delta = max(max_delta, delta)
            print(
                f"[{directory.name} {index}/{len(files)}] {path.name}: "
                f"frames={frames} max_delta_m={delta:.6f}",
                flush=True,
            )
    print(
        f"corrected_files={total_files} corrected_frames={total_frames} "
        f"max_delta_m={max_delta:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
