#!/usr/bin/env python3
"""把 50 FPS 的 sonic 真机数据时间降采样到指定帧率，写出新的 hdf5。

为什么要做：仓库里所有已有语料都是 30 fps（docs/FUTURE_DINO_CHANGES.md 的
"30fps 下手腕一帧才动 4mm" 正好对上 dex5_train_s93 实测的 4.57mm/帧，
clip_stride=4 → 133ms 也只有 30fps 成立）。sonic 是 50 fps，直接混进去的话
人类半边一步 33ms、机器人半边一步 20ms，同一个 action head 要同时拟合两种
时间步，chunk=100 在两边代表的真实时长也不一样。

不做的事：不压缩 episode 的真实时长。sonic 的 demo 本来就比人类慢（采集时
不熟练），那是真实的 domain 差异，降采样只统一 dt，不去抹平它。

图像是定长 JPEG 字节块（每帧 padding 到本文件的最大长度），按帧号取子集即可，
不需要解码重编码。ego_view 与 left/right 逐字节相同（单目 ego 相机的三份拷贝），
只保留 dataloader 会用到的 left/right。
"""
import argparse
import os
import glob
import numpy as np
import h5py

DROP_KEYS = {"observation.image.ego_view"}


def resample_indices(T: int, src_fps: float, dst_fps: float) -> np.ndarray:
    """最近邻取帧。50->30 是 5:3，不整除，所以用 round 而不是 stride。"""
    step = src_fps / dst_fps
    n_out = int(np.floor(T / step))
    idx = np.round(np.arange(n_out) * step).astype(np.int64)
    return np.clip(idx, 0, T - 1)


def convert_one(src: str, dst: str, src_fps: float, dst_fps: float) -> tuple:
    with h5py.File(src, "r") as fi:
        T = fi["action"].shape[0]
        idx = resample_indices(T, src_fps, dst_fps)
        with h5py.File(dst, "w") as fo:
            for k in fi.keys():
                if k in DROP_KEYS:
                    continue
                d = fi[k]
                if d.shape[0] != T:
                    fo.create_dataset(k, data=d[()], compression=d.compression)
                    continue
                # 逐块读，避免 1500x22456 的图像整块进内存的峰值
                out = d[()][idx]
                fo.create_dataset(k, data=out, compression=d.compression)
            for ak, av in fi.attrs.items():
                fo.attrs[ak] = av
            fo.attrs["fps"] = float(dst_fps)
            fo.attrs["resampled_from_fps"] = float(src_fps)
            fo.attrs["resampled_source"] = src
            fo.attrs["dropped_keys"] = ",".join(sorted(DROP_KEYS))
    return T, len(idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_dir", default="/mnt/nvme0n1/DATASETS/sonic_real_teleop_20260903_hat_rootfix")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--src_fps", type=float, default=50.0)
    ap.add_argument("--dst_fps", type=float, default=30.0)
    ap.add_argument("--skip_existing", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    fs = sorted(glob.glob(os.path.join(args.src_dir, "*.hdf5")))
    if not fs:
        raise SystemExit(f"没有 hdf5: {args.src_dir}")

    tot_in = tot_out = 0
    for i, src in enumerate(fs):
        dst = os.path.join(args.out_dir, os.path.basename(src))
        if args.skip_existing and os.path.exists(dst):
            continue
        tin, tout = convert_one(src, dst, args.src_fps, args.dst_fps)
        tot_in += tin
        tot_out += tout
        print(f"[{i+1}/{len(fs)}] {os.path.basename(src)}  {tin} -> {tout}", flush=True)

    print(f"done: {len(fs)} episodes, {tot_in} -> {tot_out} frames "
          f"({args.src_fps} -> {args.dst_fps} fps), out={args.out_dir}")


if __name__ == "__main__":
    main()
