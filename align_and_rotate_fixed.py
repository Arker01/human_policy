#!/usr/bin/env python3
import h5py
import numpy as np
import os
from glob import glob
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

def rotate_point_cloud(points, axis='y', angle_deg=90):
    """绕指定轴旋转点云"""
    angle_rad = -np.deg2rad(angle_deg)
    
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    if points.ndim == 1:
        return R @ points
    else:
        return (R @ points.T).T

def rotate_6d_rotation(rotation_6d, axis='x', angle_deg=90):
    """对6D旋转表示进行旋转（使用 PyTorch3D 标准实现）"""
    import torch
    
    # 确保输入是 torch tensor
    if not isinstance(rotation_6d, torch.Tensor):
        rotation_6d = torch.tensor(rotation_6d, dtype=torch.float32)
    
    # 如果是单个6D向量，添加batch维度
    if rotation_6d.ndim == 1:
        rotation_6d = rotation_6d.unsqueeze(0)
    
    # 将6D转换为旋转矩阵
    rotation_matrix = rotation_6d_to_matrix(rotation_6d)
    
    # 生成旋转矩阵
    angle_rad = -np.deg2rad(angle_deg)
    if axis == 'x':
        R = torch.tensor([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ], dtype=torch.float32)
    elif axis == 'y':
        R = torch.tensor([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ], dtype=torch.float32)
    elif axis == 'z':
        R = torch.tensor([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ], dtype=torch.float32)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    
    # 应用旋转（矩阵乘法）
    rotation_matrix_rotated = torch.matmul(R, rotation_matrix)
    
    # 转换回6D表示
    rotation_6d_rotated = matrix_to_rotation_6d(rotation_matrix_rotated)
    
    # 移除batch维度（如果需要）
    if rotation_6d_rotated.shape[0] == 1:
        rotation_6d_rotated = rotation_6d_rotated.squeeze(0)
    
    return rotation_6d_rotated.numpy() if isinstance(rotation_6d, torch.Tensor) else rotation_6d_rotated

def rotate_matrix(matrix, axis='x', angle_deg=90):
    """对旋转矩阵进行旋转"""
    angle_rad = -np.deg2rad(angle_deg)
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(angle_rad), -np.sin(angle_rad)],
            [0, np.sin(angle_rad), np.cos(angle_rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(angle_rad), 0, np.sin(angle_rad)],
            [0, 1, 0],
            [-np.sin(angle_rad), 0, np.cos(angle_rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(angle_rad), -np.sin(angle_rad), 0],
            [np.sin(angle_rad), np.cos(angle_rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return R @ matrix

# 数据目录
input_dir = "/data1/zxlei/dataset/ego_converted"
output_dir = "/data1/zxlei/dataset/ego_converted2_test"

# 旋转参数配置
ROTATION_AXIS = 'z'  # 旋转轴: 'x', 'y', 'z'
ROTATION_ANGLE = 90  # 旋转角度（顺时针为正）

# 头部位置索引
HEAD_POS_IDX = np.arange(0, 3)
HEAD_ROT_IDX = np.arange(3, 9)
# 左手腕位置
LEFT_EEF_POS_IDX = np.arange(80, 83)
LEFT_EEF_ROT_IDX = np.arange(83, 89)
# 右手腕位置  
RIGHT_EEF_POS_IDX = np.arange(30, 33)
RIGHT_EEF_ROT_IDX = np.arange(33, 39)

# 左手关键点索引 (10-27: 6个点 × 3维)
LEFT_KEYPOINTS_IDX = np.arange(10, 28)
# 右手关键点索引 (40-57: 6个点 × 3维)
RIGHT_KEYPOINTS_IDX = np.arange(40, 58)

# 所有需要平移的绝对位置
ALL_POS_IDX = np.concatenate([HEAD_POS_IDX, LEFT_EEF_POS_IDX, RIGHT_EEF_POS_IDX])

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
        attrs = dict(f.attrs)
    
    # 步骤1: 获取第一帧的头部位置作为偏移量
    first_head_pos = actions[0, HEAD_POS_IDX].copy()
    print(f"Processing {os.path.basename(hdf5_path)}: first head pos = {first_head_pos}")
    
    # 步骤2: 所有绝对位置坐标减去第一帧头部位置
    actions[:, HEAD_POS_IDX] -= first_head_pos[0:3]
    actions[:, LEFT_EEF_POS_IDX] -= first_head_pos[0:3]
    actions[:, RIGHT_EEF_POS_IDX] -= first_head_pos[0:3]
    
    # 步骤3: 顺时针旋转90度（绕X轴）

    # 旋转绝对位置
    actions[:, HEAD_POS_IDX] = rotate_point_cloud(actions[:, HEAD_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
    actions[:, LEFT_EEF_POS_IDX] = rotate_point_cloud(actions[:, LEFT_EEF_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
    actions[:, RIGHT_EEF_POS_IDX] = rotate_point_cloud(actions[:, RIGHT_EEF_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

    # 旋转EEF的6D旋转表示
    for i in range(len(actions)):
        actions[i, HEAD_ROT_IDX] = rotate_6d_rotation(actions[i, HEAD_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
        actions[i, LEFT_EEF_ROT_IDX] = rotate_6d_rotation(actions[i, LEFT_EEF_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
        actions[i, RIGHT_EEF_ROT_IDX] = rotate_6d_rotation(actions[i, RIGHT_EEF_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

    # 旋转左手关键点 (6个点，每点3维)
    # for i in range(6):
    #     start = 10 + i * 3
    #     end = start + 3
    #     actions[:, start:end] = rotate_point_cloud(actions[:, start:end], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

    # # 旋转右手关键点
    # for i in range(6):
    #     start = 40 + i * 3
    #     end = start + 3
    #     actions[:, start:end] = rotate_point_cloud(actions[:, start:end], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

    # 更新 state（如果存在）
    if state is not None:
        state[:, HEAD_POS_IDX] -= first_head_pos[0:3]
        state[:, LEFT_EEF_POS_IDX] -= first_head_pos[0:3]
        state[:, RIGHT_EEF_POS_IDX] -= first_head_pos[0:3]

        state[:, HEAD_POS_IDX] = rotate_point_cloud(state[:, HEAD_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
        state[:, LEFT_EEF_POS_IDX] = rotate_point_cloud(state[:, LEFT_EEF_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
        state[:, RIGHT_EEF_POS_IDX] = rotate_point_cloud(state[:, RIGHT_EEF_POS_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

        for i in range(len(state)):
            state[i, HEAD_ROT_IDX] = rotate_6d_rotation(state[i, HEAD_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
            state[i, LEFT_EEF_ROT_IDX] = rotate_6d_rotation(state[i, LEFT_EEF_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
            state[i, RIGHT_EEF_ROT_IDX] = rotate_6d_rotation(state[i, RIGHT_EEF_ROT_IDX], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)

        # for i in range(6):
        #     start = 10 + i * 3
        #     end = start + 3
        #     state[:, start:end] = rotate_point_cloud(state[:, start:end], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
        #     start = 40 + i * 3
        #     end = start + 3
        #     state[:, start:end] = rotate_point_cloud(state[:, start:end], axis=ROTATION_AXIS, angle_deg=ROTATION_ANGLE)
    
    # 保存变换后的数据
    output_path = os.path.join(output_dir, os.path.basename(hdf5_path))
    with h5py.File(output_path, "w") as f:
        f.create_dataset("action", data=actions)
        if state is not None:
            f.create_dataset("observation.state", data=state)
        if image_top is not None:
            f.create_dataset("observation.image.top", data=image_top)
        for k, v in attrs.items():
            f.attrs[k] = v

print(f"Done! Transformed data saved to {output_dir}")