#!/usr/bin/env python3
"""Inject the real XQY pelvis/root pose into converted UnifoLM HDF5 files.

The 128-D converter leaves indices 89:98 (root xyz + rotation 6D) at zero.
XQY root poses are in the original world frame, while converted HDF5 poses are
head-origin recentered.  This postprocessor applies the HDF5's recorded
``head_origin_translation`` to root xyz and writes the first two rows of the
root rotation matrix to both state and action.

Examples:
  python fix_unifolm_root.py --input-dir data/converted \
      --xqy-dir /mnt/nvme0n1/DATASETS/XQY_PKL \
      --output-dir data/rootfix --embodiment auto

  python fix_unifolm_root.py --input episode.hdf5 --xqy root.pkl \
      --output episode_rootfix.hdf5 --embodiment dex5
"""

import argparse
import os
import pickle
import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


ROOT_POS = slice(89, 92)
ROOT_ROT6D = slice(92, 98)
DATASETS = ("observation.state", "action")


def root_from_xqy(hdf5_path: Path, xqy_path: Path) -> np.ndarray:
    with h5py.File(hdf5_path, "r") as src:
        if "head_origin_translation" not in src.attrs:
            raise KeyError(f"{hdf5_path}: missing head_origin_translation")
        translation = np.asarray(src.attrs["head_origin_translation"], np.float32)
        lengths = {name: len(src[name]) for name in DATASETS}
    if translation.shape != (3,):
        raise ValueError(f"{hdf5_path}: bad head_origin_translation shape {translation.shape}")
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{hdf5_path}: state/action length mismatch: {lengths}")

    # XQY files are trusted local preprocessing artifacts, not user uploads.
    with xqy_path.open("rb") as stream:
        xqy = pickle.load(stream)
    root_pos = np.asarray(xqy["root_pos"], np.float32)
    root_quat = np.asarray(xqy["root_rot"], np.float32)  # scipy xyzw order
    length = next(iter(lengths.values()))
    if len(root_pos) != length or len(root_quat) != length:
        raise ValueError(
            f"length mismatch for {hdf5_path.name}: HDF5={length}, "
            f"root_pos={len(root_pos)}, root_rot={len(root_quat)}"
        )

    root = np.empty((length, 9), np.float32)
    root[:, :3] = root_pos + translation
    root[:, 3:] = (
        Rotation.from_quat(root_quat).as_matrix()[:, :2, :].reshape(length, 6)
    ).astype(np.float32)
    return root


def fix_file(
    input_path: Path,
    xqy_path: Path,
    output_path: Path,
    embodiment: str,
    overwrite: bool,
) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"output exists (pass --overwrite): {output_path}")
    root = root_from_xqy(input_path, xqy_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Copy first so images, dataset creation properties, groups and attrs remain intact.
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    os.close(fd)
    temporary_path = Path(temporary)
    try:
        shutil.copyfile(input_path, temporary_path)
        with h5py.File(temporary_path, "r+") as dst:
            for name in DATASETS:
                if dst[name].ndim != 2 or dst[name].shape[1] < ROOT_ROT6D.stop:
                    raise ValueError(f"{input_path}: invalid {name} shape {dst[name].shape}")
                dst[name][:, ROOT_POS] = root[:, :3]
                dst[name][:, ROOT_ROT6D] = root[:, 3:]
            dst.attrs["embodiment"] = embodiment
            dst.attrs["rootfix_source"] = str(xqy_path.resolve())
            dst.attrs["rootfix_method"] = "xqy_root_in_head_origin_frame"
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--input-dir", type=Path)
    parser.add_argument("--xqy", type=Path, help="XQY PKL for single-file mode")
    parser.add_argument("--xqy-dir", type=Path, help="directory of same-stem PKLs")
    parser.add_argument("--output", type=Path, help="output HDF5 for single-file mode")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--pattern", default="*.hdf5", help="input glob in directory mode")
    parser.add_argument(
        "--embodiment",
        default="auto",
        help="output attribute; 'auto' infers dex5/brainco from each filename",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.input is not None:
        if args.xqy is None or args.output is None:
            raise ValueError("single-file mode requires --xqy and --output")
        jobs = [(args.input, args.xqy, args.output)]
    else:
        if args.xqy_dir is None or args.output_dir is None:
            raise ValueError("directory mode requires --xqy-dir and --output-dir")
        inputs = sorted(args.input_dir.glob(args.pattern))
        if not inputs:
            raise FileNotFoundError(f"no *.hdf5 files in {args.input_dir}")
        jobs = [
            (path, args.xqy_dir / f"{path.stem}.pkl", args.output_dir / f"{path.stem}_rootfix.hdf5")
            for path in inputs
        ]

    missing = [str(xqy) for _, xqy, _ in jobs if not xqy.is_file()]
    if missing:
        preview = "\n  ".join(missing[:10])
        raise FileNotFoundError(f"missing {len(missing)} XQY PKLs:\n  {preview}")
    print(f"validated {len(jobs)} input/PKL pairs")
    if args.dry_run:
        return
    for index, (source, xqy, output) in enumerate(jobs, 1):
        embodiment = args.embodiment
        if embodiment == "auto":
            lowered = source.name.lower()
            if "dex5" in lowered:
                embodiment = "dex5"
            elif "brainco" in lowered:
                embodiment = "brainco"
            else:
                raise ValueError(f"cannot infer embodiment from {source.name}")
        fix_file(source, xqy, output, embodiment, args.overwrite)
        print(f"[{index}/{len(jobs)}] {output}")


if __name__ == "__main__":
    main()
