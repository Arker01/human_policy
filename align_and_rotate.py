#!/usr/bin/env python3
import h5py
import numpy as np
import os
from glob import glob
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


def make_rotation_matrix(axis, angle_deg):
    """Create a rotation matrix using the original script convention.

    The old code used angle_rad = -deg2rad(angle_deg), where positive angle
    means clockwise. Keep that convention so configured angles match existing
    align_and_rotate.py usage.
    """
    angle_rad = -np.deg2rad(angle_deg)

    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=np.float32)
    elif axis == 'y':
        return np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=np.float32)
    elif axis == 'z':
        return np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=np.float32)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def make_composed_rotation_matrix(rotation_sequence):
    R = np.eye(3, dtype=np.float32)
    for axis, angle_deg in rotation_sequence:
        R = make_rotation_matrix(axis, angle_deg) @ R
    return R


def rotate_point_cloud(points, rotation_matrix):
    """Apply the configured coordinate-frame rotation to points."""
    if points.ndim == 1:
        return rotation_matrix @ points
    return (rotation_matrix @ points.T).T


def rotate_6d_rotation(rotation_6d, rotation_matrix):
    """对6D旋转表示进行旋转（使用 PyTorch3D 标准实现）"""
    import torch

    rotation_6d = torch.as_tensor(rotation_6d, dtype=torch.float32)
    R = torch.as_tensor(rotation_matrix, dtype=torch.float32)

    # 如果是单个6D向量，添加batch维度
    if rotation_6d.ndim == 1:
        rotation_6d = rotation_6d.unsqueeze(0)

    # 将6D转换为旋转矩阵
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)

    # 应用旋转（矩阵乘法）
    rotation_matrix_rotated = torch.matmul(R, rotation_matrix)

    # 转换回6D表示
    rotation_6d_rotated = matrix_to_rotation_6d(rotation_matrix_rotated)

    # 移除batch维度（如果需要）
    if rotation_6d_rotated.shape[0] == 1:
        rotation_6d_rotated = rotation_6d_rotated.squeeze(0)

    return rotation_6d_rotated.numpy()


# 旋转参数配置：一次性先绕 x 轴 -90 度，再绕 z 轴 90 度
ROTATION_SEQUENCE = [('x', -90), ('z', 90)]
ROTATION_MATRIX = make_composed_rotation_matrix(ROTATION_SEQUENCE)

# 头部位置索引
HEAD_POS_IDX = np.arange(0, 3)
HEAD_ROT_IDX = np.arange(3, 9)
# 左手腕位置
LEFT_EEF_POS_IDX = np.arange(80, 83)
LEFT_EEF_ROT_IDX = np.arange(83, 89)
# 右手腕位置  
RIGHT_EEF_POS_IDX = np.arange(30, 33)
RIGHT_EEF_ROT_IDX = np.arange(33, 39)
# 后脑 tracker/head tracker 位置
HEAD_TRACKER_POS_IDX = np.arange(58, 61)
HEAD_TRACKER_ROT_IDX = np.arange(61, 67)
# 腰部 tracker 位置
WAIST_POS_IDX = np.arange(89, 92)
WAIST_ROT_IDX = np.arange(92, 98)

# 左手关键点索引 (10-27: 6个点 × 3维)
LEFT_KEYPOINTS_IDX = np.arange(10, 28)
# 右手关键点索引 (40-57: 6个点 × 3维)
RIGHT_KEYPOINTS_IDX = np.arange(40, 58)

# 所有需要平移的绝对位置
ALL_POS_IDX = np.concatenate([
    HEAD_POS_IDX,
    LEFT_EEF_POS_IDX,
    RIGHT_EEF_POS_IDX,
    HEAD_TRACKER_POS_IDX,
    WAIST_POS_IDX,
])

POSE_BLOCKS = [
    (HEAD_POS_IDX, HEAD_ROT_IDX),
    (LEFT_EEF_POS_IDX, LEFT_EEF_ROT_IDX),
    (RIGHT_EEF_POS_IDX, RIGHT_EEF_ROT_IDX),
    (HEAD_TRACKER_POS_IDX, HEAD_TRACKER_ROT_IDX),
    (WAIST_POS_IDX, WAIST_ROT_IDX),
]


def pose_block_has_data(arr, pos_idx, rot_idx, eps=1e-6):
    return np.any(np.abs(arr[:, pos_idx]) > eps) or np.any(np.abs(arr[:, rot_idx]) > eps)


def transform_pose_blocks(arr, first_head_pos):
    for pos_idx, _ in POSE_BLOCKS:
        if pose_block_has_data(arr, pos_idx, _):
            arr[:, pos_idx] -= first_head_pos[0:3]
            arr[:, pos_idx] = rotate_point_cloud(arr[:, pos_idx], ROTATION_MATRIX)

    for _, rot_idx in POSE_BLOCKS:
        if np.any(np.abs(arr[:, rot_idx]) > 1e-6):
            for i in range(len(arr)):
                arr[i, rot_idx] = rotate_6d_rotation(arr[i, rot_idx], ROTATION_MATRIX)
    return arr

def process_directory(input_dir, output_dir):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 HDF5 文件
    hdf5_files = sorted(glob(os.path.join(input_dir, "*.hdf5")))
    print(f"Found {len(hdf5_files)} files")

    for hdf5_path in hdf5_files:
        with h5py.File(hdf5_path, "r") as f:
            actions = f["action"][()]
            state = f["observation.state"][()] if "observation.state" in f else None
            image_top = f["observation.image.top"][()] if "observation.image.top" in f else None
            extra_datasets = {}
            for name in f.keys():
                if name not in ("action", "observation.state", "observation.image.top"):
                    extra_datasets[name] = f[name][()]
            attrs = dict(f.attrs)

        # 步骤1: 获取第一帧的头部位置作为偏移量
        first_head_pos = actions[0, HEAD_POS_IDX].copy()
        print(f"Processing {os.path.basename(hdf5_path)}: first head pos = {first_head_pos}")

        # 步骤2: 所有绝对位置坐标减去第一帧头部位置
        actions = transform_pose_blocks(actions, first_head_pos)

        # 更新 state（如果存在）
        if state is not None:
            state = transform_pose_blocks(state, first_head_pos)

        # 保存变换后的数据
        output_path = os.path.join(output_dir, os.path.basename(hdf5_path))
        with h5py.File(output_path, "w") as f:
            f.create_dataset("action", data=actions)
            if state is not None:
                f.create_dataset("observation.state", data=state)
            if image_top is not None:
                f.create_dataset("observation.image.top", data=image_top)
            for name, data in extra_datasets.items():
                f.create_dataset(name, data=data)
            for k, v in attrs.items():
                f.attrs[k] = v

    print(f"Done! Transformed data saved to {output_dir}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Align and rotate converted HDF5 files")
    parser.add_argument("--input", "-i", default="/root/shengyin/rubbish_vive1")
    parser.add_argument("--output", "-o", default="/root/shengyin/rubbish_vive2")
    args = parser.parse_args()

    process_directory(args.input, args.output)


if __name__ == "__main__":
    main()
