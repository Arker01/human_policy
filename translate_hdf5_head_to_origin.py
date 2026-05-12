#!/usr/bin/env python3
"""Translate human_policy HDF5 episodes so the initial stable head is at origin."""

import argparse
import os
import re
from glob import glob

import h5py
import numpy as np


IDX_HEAD_POS = slice(0, 3)
IDX_RIGHT_WRIST_POS = slice(30, 33)
IDX_LEFT_WRIST_POS = slice(80, 83)
IDX_QPOS_ROOT_POS = slice(100, 103)


def choose_stable_head_window(head_pos: np.ndarray, search_frames: int, window: int) -> tuple[int, int]:
    n = head_pos.shape[0]
    if n == 0:
        raise ValueError("empty episode")
    search_n = min(max(1, search_frames), n)
    win = min(max(1, window), search_n)
    best_start = 0
    best_score = float("inf")
    for start in range(0, search_n - win + 1):
        segment = head_pos[start : start + win]
        centered = segment - segment.mean(axis=0, keepdims=True)
        score = float(np.mean(np.sum(centered * centered, axis=1)))
        if score < best_score:
            best_score = score
            best_start = start
    return best_start, best_start + win


def translate_vec128(data: np.ndarray, translation: np.ndarray) -> np.ndarray:
    out = np.asarray(data).copy()
    out[:, IDX_HEAD_POS] += translation
    out[:, IDX_RIGHT_WRIST_POS] += translation
    out[:, IDX_LEFT_WRIST_POS] += translation
    if out.shape[1] >= IDX_QPOS_ROOT_POS.stop:
        out[:, IDX_QPOS_ROOT_POS] += translation
    return out


def copy_episode_with_translation(
    input_path: str,
    output_path: str,
    *,
    search_frames: int,
    stable_window: int,
    overwrite: bool,
    external_images: bool,
) -> dict:
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with h5py.File(input_path, "r") as src:
        states = src["observation.state"][:]
        head_pos = states[:, IDX_HEAD_POS].astype(np.float64)
        start, end = choose_stable_head_window(head_pos, search_frames, stable_window)
        head_origin = head_pos[start:end].mean(axis=0).astype(np.float32)
        translation = (-head_origin).astype(np.float32)

        with h5py.File(output_path, "w") as dst:
            for key in src.keys():
                if key == "observation.state":
                    dst.create_dataset(key, data=translate_vec128(states, translation), dtype=src[key].dtype)
                elif key == "action":
                    actions = src[key][:]
                    dst.create_dataset(key, data=translate_vec128(actions, translation), dtype=src[key].dtype)
                elif external_images and key.startswith("observation.image."):
                    dst[key] = h5py.ExternalLink(os.path.abspath(input_path), key)
                else:
                    src.copy(key, dst)

            for k, v in src.attrs.items():
                dst.attrs[k] = v
            dst.attrs["head_origin_translation"] = translation
            dst.attrs["head_origin_source_mean"] = head_origin
            dst.attrs["head_origin_window"] = np.array([start, end], dtype=np.int32)
            dst.attrs["head_origin_input"] = os.path.abspath(input_path)
            dst.attrs["head_origin_external_images"] = np.bool_(external_images)

    return {
        "input": input_path,
        "output": output_path,
        "window": (start, end),
        "head_origin": head_origin.tolist(),
        "translation": translation.tolist(),
    }


def episode_index_from_hdf5_name(path: str) -> int:
    name = os.path.splitext(os.path.basename(path))[0]
    match = re.search(r"(\d+)$", name)
    if match is None:
        raise ValueError(f"Cannot infer episode index from {path}")
    return int(match.group(1))


def output_name_for_episode(xqy_prefix: str, ep_idx: int) -> str:
    return f"{xqy_prefix}_ep_{ep_idx:04d}.hdf5"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=None, help="single input HDF5")
    parser.add_argument("--output", default=None, help="single output HDF5")
    parser.add_argument("--hdf5-dir", default=None, help="batch input directory containing <episode>.hdf5")
    parser.add_argument("--out-dir", default="/root/shengyin/DATASETS/human_policy/convert_UnifoLM_WBT")
    parser.add_argument("--xqy-prefix", default=None, help="output basename prefix, e.g. G1_WBT_Brainco_...")
    parser.add_argument("--search-frames", type=int, default=20)
    parser.add_argument("--stable-window", type=int, default=10)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--external-images",
        action="store_true",
        help="store observation.image.* as HDF5 external links to avoid duplicating JPEG datasets",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
        if args.output is None:
            if args.xqy_prefix is None:
                raise ValueError("--output or --xqy-prefix is required with --input")
            ep_idx = episode_index_from_hdf5_name(args.input)
            args.output = os.path.join(args.out_dir, output_name_for_episode(args.xqy_prefix, ep_idx))
        result = copy_episode_with_translation(
            args.input,
            args.output,
            search_frames=args.search_frames,
            stable_window=args.stable_window,
            overwrite=args.overwrite,
            external_images=args.external_images,
        )
        print(result)
        return

    if args.hdf5_dir is None or args.xqy_prefix is None:
        raise ValueError("Use either --input or both --hdf5-dir and --xqy-prefix")

    files = sorted(glob(os.path.join(args.hdf5_dir, "*.hdf5")))
    if not files:
        raise FileNotFoundError(f"No hdf5 files under {args.hdf5_dir}")
    for input_path in files:
        ep_idx = episode_index_from_hdf5_name(input_path)
        output_path = os.path.join(args.out_dir, output_name_for_episode(args.xqy_prefix, ep_idx))
        result = copy_episode_with_translation(
            input_path,
            output_path,
            search_frames=args.search_frames,
            stable_window=args.stable_window,
            overwrite=args.overwrite,
            external_images=args.external_images,
        )
        print(result)


if __name__ == "__main__":
    main()
