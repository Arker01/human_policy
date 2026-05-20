#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HDT_DIR = _REPO_ROOT / "hdt"
if str(_HDT_DIR) not in sys.path:
    sys.path.insert(0, str(_HDT_DIR))

import hdt.constants as C


def _kp0_norms(arr: np.ndarray, keypoint_indices: np.ndarray) -> np.ndarray:
    keypoints = arr[:, keypoint_indices].reshape(arr.shape[0], 6, 3)
    return np.linalg.norm(keypoints[:, 0, :], axis=1)


def _set_kp0_zero(
    arr: np.ndarray,
    keypoint_indices: np.ndarray,
    *,
    threshold: float,
) -> tuple[np.ndarray, int]:
    fixed = arr.copy()
    keypoints = fixed[:, keypoint_indices].copy().reshape(fixed.shape[0], 6, 3)
    norms = np.linalg.norm(keypoints[:, 0, :], axis=1)
    mask = norms > threshold
    keypoints[mask, 0, :] = 0.0
    fixed[:, keypoint_indices] = keypoints.reshape(fixed.shape[0], -1)
    return fixed, int(mask.sum())


def _summarize(norms: np.ndarray) -> dict[str, float | int]:
    if norms.size == 0:
        return {"count": 0, "bad_frames": 0, "max": 0.0, "mean": 0.0}
    return {
        "count": int(norms.size),
        "bad_frames": int((norms > 0).sum()),
        "max": float(norms.max()),
        "mean": float(norms.mean()),
    }


def _copy_all_objects(src: h5py.File, dst: h5py.File, skip_names: set[str]) -> None:
    for key in src.keys():
        if key in skip_names:
            continue
        src.copy(key, dst)
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def _print_stats(label: str, left_norms: np.ndarray, right_norms: np.ndarray) -> None:
    left = _summarize(left_norms)
    right = _summarize(right_norms)
    print(
        f"{label}: "
        f"left bad={left['bad_frames']}/{left['count']} max={left['max']:.6f} mean={left['mean']:.6f} | "
        f"right bad={right['bad_frames']}/{right['count']} max={right['max']:.6f} mean={right['mean']:.6f}"
    )


def _analyze_dataset(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    left_norms = _kp0_norms(arr, C.OUTPUT_LEFT_KEYPOINTS)
    right_norms = _kp0_norms(arr, C.OUTPUT_RIGHT_KEYPOINTS)
    return left_norms, right_norms


def process_file(
    input_path: Path,
    output_path: Path | None,
    *,
    fix_action: bool,
    fix_state: bool,
    threshold: float,
) -> dict[str, int | bool]:
    with h5py.File(input_path, "r") as f_in:
        action = f_in["action"][()]
        state = f_in["observation.state"][()] if "observation.state" in f_in else None

        action_left_before, action_right_before = _analyze_dataset(action)
        state_left_before, state_right_before = (
            _analyze_dataset(state) if state is not None else (None, None)
        )

        print(f"\n{input_path}")
        _print_stats("  action before", action_left_before, action_right_before)
        if state is not None:
            _print_stats("  state  before", state_left_before, state_right_before)

        fixed_action = action
        fixed_state = state
        action_fixed_frames = 0
        state_fixed_frames = 0

        if fix_action:
            fixed_action, action_fixed_frames = _set_kp0_zero(
                action, C.OUTPUT_RIGHT_KEYPOINTS, threshold=threshold
            )
        if fix_state and state is not None:
            fixed_state, state_fixed_frames = _set_kp0_zero(
                state, C.OUTPUT_RIGHT_KEYPOINTS, threshold=threshold
            )

        action_left_after, action_right_after = _analyze_dataset(fixed_action)
        _print_stats("  action after ", action_left_after, action_right_after)
        if fixed_state is not None:
            state_left_after, state_right_after = _analyze_dataset(fixed_state)
            _print_stats("  state  after ", state_left_after, state_right_after)

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(output_path, "w") as f_out:
                skip_names = {"action"}
                if state is not None:
                    skip_names.add("observation.state")
                _copy_all_objects(f_in, f_out, skip_names)
                f_out.create_dataset("action", data=fixed_action, compression="gzip", compression_opts=4)
                if fixed_state is not None:
                    f_out.create_dataset(
                        "observation.state",
                        data=fixed_state,
                        compression="gzip",
                        compression_opts=4,
                    )
                f_out.attrs["kp0_checked"] = True
                f_out.attrs["kp0_fix_threshold"] = float(threshold)
                f_out.attrs["kp0_action_fixed_frames"] = int(action_fixed_frames)
                f_out.attrs["kp0_state_fixed_frames"] = int(state_fixed_frames)

    return {
        "action_fixed_frames": int(action_fixed_frames),
        "state_fixed_frames": int(state_fixed_frames),
        "had_state": bool(state is not None),
    }


def collect_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    files = sorted(input_path.glob("episode_*.hdf5"))
    if files:
        return files
    files = sorted(input_path.glob("*.hdf5"))
    if files:
        return files
    return sorted(input_path.glob("*.h5"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch check/fix hand keypoint0 so wrist-frame kp0 is zero."
    )
    parser.add_argument("input", type=Path, help="Input HDF5 file or directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Write fixed files to a new directory. If omitted, runs check-only unless --inplace is used.",
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite input files in place.",
    )
    parser.add_argument(
        "--fix-action",
        action="store_true",
        help="Fix action right-hand kp0. Recommended when the issue is present.",
    )
    parser.add_argument(
        "--fix-state",
        action="store_true",
        help="Fix observation.state right-hand kp0 as well.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=1e-8,
        help="Only zero kp0 when its norm is above this threshold.",
    )
    args = parser.parse_args()

    if args.inplace and args.output_dir is not None:
        raise ValueError("--inplace and --output-dir cannot be used together.")

    files = collect_files(args.input)
    if not files:
        raise FileNotFoundError(f"No HDF5 files found under: {args.input}")

    check_only = not args.inplace and args.output_dir is None
    if check_only and (args.fix_action or args.fix_state):
        print("Warning: fix flags were provided, but no output mode was selected. Running check-only.")

    total_action_fixed = 0
    total_state_fixed = 0
    changed_files = 0

    for input_path in files:
        if args.output_dir is not None:
            output_path = args.output_dir / input_path.name
        elif args.inplace:
            output_path = input_path.with_suffix(input_path.suffix + ".tmp")
        else:
            output_path = None

        result = process_file(
            input_path,
            output_path,
            fix_action=(not check_only and args.fix_action),
            fix_state=(not check_only and args.fix_state),
            threshold=float(args.threshold),
        )

        if args.inplace and output_path is not None:
            output_path.replace(input_path)

        total_action_fixed += int(result["action_fixed_frames"])
        total_state_fixed += int(result["state_fixed_frames"])
        if int(result["action_fixed_frames"]) > 0 or int(result["state_fixed_frames"]) > 0:
            changed_files += 1

    mode = "check-only"
    if args.output_dir is not None:
        mode = f"write-to={args.output_dir}"
    elif args.inplace:
        mode = "inplace"

    print("\nSummary")
    print(f"  files: {len(files)}")
    print(f"  mode: {mode}")
    print(f"  changed_files: {changed_files}")
    print(f"  action_fixed_frames: {total_action_fixed}")
    print(f"  state_fixed_frames: {total_state_fixed}")


if __name__ == "__main__":
    main()
