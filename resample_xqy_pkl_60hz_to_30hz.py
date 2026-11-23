"""Downsample 60Hz XQY_PKL retargeting pkls back to the original 30fps sample grid.

Newly downloaded UnifoLM_WBT batches ship XQY_PKL companion pkls named
``<prefix>_ep_XXXX_60hz_new.pkl`` with fps=60.0 / source_fps=30.0: every
per-frame array was linearly upsampled 2x from the original 30fps samples.
convert_unifolm_to_hdf5.py indexes the XQY pkl frame-for-frame against the
30fps LeRobot parquet, so it needs the original 30fps samples back, named
``<prefix>_ep_XXXX.pkl`` (no suffix) so xqy_path_for_episode() finds them
directly.

Even indices of the 2x-upsampled array (0, 2, 4, ...) are exactly the
original 30fps samples (new_time[2*j] = 2*j/60 = j/30 = old_time[j]), so
downsampling is a lossless [::2] slice, not an approximation.
"""

import argparse
import glob
import os
import pickle

import numpy as np
import pandas as pd


def parquet_frame_counts(dataset: str) -> pd.Series:
    files = sorted(glob.glob(os.path.join(dataset, "data", "*", "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files under {dataset}/data/*/*.parquet")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.groupby("episode_index").size()


def downsample_episode(data: dict, expected_t30: int) -> dict:
    t60 = data["root_pos"].shape[0]
    if (t60 + 1) // 2 != expected_t30:
        raise ValueError(f"frame count mismatch: xqy T60={t60} -> expected T30={(t60 + 1) // 2}, parquet has {expected_t30}")

    out = {}
    for key, value in data.items():
        if isinstance(value, np.ndarray) and value.ndim >= 1 and value.shape[0] == t60:
            out[key] = value[::2].copy()
        else:
            out[key] = value
    if out["root_pos"].shape[0] != expected_t30:
        raise ValueError(f"downsample produced {out['root_pos'].shape[0]} frames, expected {expected_t30}")
    out["fps"] = 30.0
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="UnifoLM_WBT/<task> dir, used to validate target frame counts")
    parser.add_argument("--xqy-dir", default="/root/shengyin/DATASETS/XQY_PKL")
    parser.add_argument("--xqy-prefix", default=None, help="default: basename(dataset)")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dataset = os.path.abspath(args.dataset)
    xqy_prefix = args.xqy_prefix or os.path.basename(dataset.rstrip("/"))
    counts = parquet_frame_counts(dataset)

    n_done, n_skip = 0, 0
    for ep_idx, expected_t30 in sorted(counts.items()):
        out_path = os.path.join(args.xqy_dir, f"{xqy_prefix}_ep_{ep_idx:04d}.pkl")
        if os.path.exists(out_path) and not args.overwrite:
            n_skip += 1
            continue
        in_path = os.path.join(args.xqy_dir, f"{xqy_prefix}_ep_{ep_idx:04d}_60hz_new.pkl")
        with open(in_path, "rb") as f:
            data = pickle.load(f)
        out = downsample_episode(data, int(expected_t30))
        with open(out_path, "wb") as f:
            pickle.dump(out, f)
        n_done += 1
        print(f"  ep {ep_idx:04d}: T60={data['root_pos'].shape[0]} -> T30={out['root_pos'].shape[0]} -> {out_path}")

    print(f"Done. converted={n_done} skipped_existing={n_skip} total_episodes={len(counts)}")


if __name__ == "__main__":
    main()
