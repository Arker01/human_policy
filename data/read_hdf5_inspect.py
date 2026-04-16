#!/usr/bin/env python3
"""
读取并摘要 HDF5 结构，用于理解本仓库里两类常见文件：

1) 标注 / 动捕类（例如 data/0.hdf5）
   - 根节点 attrs：任务描述、会话视频名、LLM 标注等
   - camera/intrinsic: (3, 3) 相机内参
   - transforms/<关节名>: (T, 4, 4) float32，每帧齐次变换矩阵
   - confidences/<关节名>: (T,) float32，与 transforms 同名字一一对应
   - transforms/camera: (T, 4, 4) 相机位姿（若存在）

2) 处理后训练用（例如 processed_episode_*.hdf5）
   - 通常含 /action、/obs/... 等扁平或分组路径，见 hdt/data_utils_hdt.py、plot_keypoints.py
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _attrs_to_jsonable(attrs: h5py.AttributeManager) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in attrs:
        v = attrs[k]
        if isinstance(v, np.ndarray) and v.dtype == object:
            out[k] = [str(x) for x in v.tolist()]
        elif isinstance(v, (np.ndarray, np.generic)):
            out[k] = np.asarray(v).tolist()
        else:
            out[k] = v
    return out


def iter_hdf5_tree(
    group: h5py.Group | h5py.File, prefix: str = ""
) -> list[tuple[str, str, tuple[int, ...] | None, str]]:
    """
    返回列表项: (kind, path, shape_or_none, dtype_str)
    kind 为 'dataset' 或 'group'。
    """
    rows: list[tuple[str, str, tuple[int, ...] | None, str]] = []
    for name in sorted(group.keys()):
        path = f"{prefix}/{name}" if prefix else name
        obj = group[name]
        if isinstance(obj, h5py.Dataset):
            rows.append(("dataset", path, obj.shape, str(obj.dtype)))
        else:
            rows.append(("group", path, None, ""))
            rows.extend(iter_hdf5_tree(obj, path))
    return rows


def summarize_hdf5(path: str | Path, max_attr_chars: int = 2000) -> None:
    path = Path(path)
    with h5py.File(path, "r") as f:
        print(f"文件: {path.resolve()}")
        print("\n--- 根属性 (attrs) ---")
        root_attrs = _attrs_to_jsonable(f.attrs)
        s = json.dumps(root_attrs, ensure_ascii=False, indent=2)
        if len(s) > max_attr_chars:
            print(s[:max_attr_chars] + "\n... [截断]")
        else:
            print(s)

        print("\n--- 组 / 数据集树 ---")
        tree = iter_hdf5_tree(f)
        # 按路径分组统计：数据集数量、最大时间步 T
        ts: set[int] = set()
        for kind, p, shape, dtype in tree:
            if kind != "dataset" or shape is None or len(shape) < 1:
                continue
            t0 = int(shape[0])
            # 序列帧：常见为 (T,4,4) 或 (T,)；排除小矩阵如内参 (3,3)
            if len(shape) == 2 and shape == (3, 3):
                continue
            if t0 > 1:
                ts.add(t0)
        if ts:
            print(f"推断时间步 T（来自首维，已排除 3x3 等）: {sorted(ts)}")

        for kind, p, shape, dtype in tree:
            if kind == "group":
                print(f"[GRP] {p}/")
            else:
                sh = shape if shape is not None else ()
                print(f"[DS ] {p}  shape={sh}  dtype={dtype}")

        # 简要语义说明（针对 0.hdf5 这类）
        has_transforms = "transforms" in f and isinstance(f["transforms"], h5py.Group)
        has_conf = "confidences" in f and isinstance(f["confidences"], h5py.Group)
        if has_transforms and has_conf:
            t_keys = set(f["transforms"].keys())
            c_keys = set(f["confidences"].keys())
            only_t = t_keys - c_keys
            only_c = c_keys - t_keys
            print("\n--- 语义提示（标注/动捕类）---")
            print(
                "transforms/* 与 confidences/* 按关节名对齐；"
                "每帧一条 4x4 齐次矩阵，confidences 为标量置信度。"
            )
            if only_t or only_c:
                print(f"仅 transforms 有: {sorted(only_t)[:8]}{'...' if len(only_t) > 8 else ''}")
                print(f"仅 confidences 有: {sorted(only_c)[:8]}{'...' if len(only_c) > 8 else ''}")


def load_frame_transforms(
    path: str | Path, frame_index: int = 0
) -> dict[str, np.ndarray]:
    """读取某一帧所有 transforms/* 关节矩阵，返回 {关节名: (4, 4)}。"""
    out: dict[str, np.ndarray] = {}
    with h5py.File(path, "r") as f:
        if "transforms" not in f:
            raise KeyError("无 transforms 组，可能不是标注类 HDF5")
        g = f["transforms"]
        for name in g.keys():
            ds = g[name]
            if not isinstance(ds, h5py.Dataset):
                continue
            arr = np.asarray(ds[frame_index])
            if arr.shape == (4, 4):
                out[str(name)] = arr
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="打印 HDF5 结构与根属性")
    parser.add_argument(
        "hdf5_path",
        nargs="?",
        default=str(Path(__file__).resolve().parent / "0.hdf5"),
        help="HDF5 路径（默认: 与本脚本同目录的 0.hdf5）",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="若设置，额外打印该帧 transforms 中各关节 4x4 矩阵形状校验信息",
    )
    args = parser.parse_args()

    summarize_hdf5(args.hdf5_path)

    if args.frame is not None:
        print(f"\n--- 第 {args.frame} 帧 transforms 抽样 ---")
        mats = load_frame_transforms(args.hdf5_path, args.frame)
        for i, (k, m) in enumerate(sorted(mats.items())[:5]):
            print(f"  {k}: det(R)≈{np.linalg.det(m[:3, :3]):.4f}")
        print(f"  ... 共 {len(mats)} 个关节")


if __name__ == "__main__":
    main()
