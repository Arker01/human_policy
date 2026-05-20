#!/usr/bin/env python3
"""检查手腕是否都在头部下方（z轴）"""

import h5py
import numpy as np
import os
from glob import glob

# 数据目录
input_dir = "/data1/zxlei/dataset/mocap_converted_x_negated"

# 索引定义
HEAD_POS_IDX = np.arange(0, 3)    # 头部位置
LEFT_EEF_POS_IDX = np.arange(80, 83)   # 左手腕位置
RIGHT_EEF_POS_IDX = np.arange(30, 33)  # 右手腕位置

# 获取所有HDF5文件
hdf5_files = sorted(glob(os.path.join(input_dir, "*.hdf5")))
print(f"找到 {len(hdf5_files)} 个HDF5文件")

# 统计结果
total_files = len(hdf5_files)
total_frames = 0
valid_files = 0
invalid_files = []
invalid_frames_details = []

for hdf5_path in hdf5_files:
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][()]
    
    num_frames = len(actions)
    total_frames += num_frames
    
    # 检查每帧
    head_z = actions[:, HEAD_POS_IDX[2]]  # z坐标
    left_wrist_z = actions[:, LEFT_EEF_POS_IDX[2]]
    right_wrist_z = actions[:, RIGHT_EEF_POS_IDX[2]]
    
    # 手腕在头部下方的条件：wrist_z <= head_z
    left_valid = left_wrist_z <= head_z
    right_valid = right_wrist_z <= head_z
    
    all_valid = np.all(left_valid) and np.all(right_valid)
    
    if all_valid:
        valid_files += 1
        print(f"✅ {os.path.basename(hdf5_path)}: 所有 {num_frames} 帧手腕都在头部下方")
    else:
        invalid_files.append(os.path.basename(hdf5_path))
        
        # 找出无效帧
        invalid_left_frames = np.where(~left_valid)[0]
        invalid_right_frames = np.where(~right_valid)[0]
        
        invalid_frames_details.append({
            'file': os.path.basename(hdf5_path),
            'total_frames': num_frames,
            'invalid_left': len(invalid_left_frames),
            'invalid_right': len(invalid_right_frames),
            'example_frames': {
                'left': invalid_left_frames[:3] if len(invalid_left_frames) > 0 else [],
                'right': invalid_right_frames[:3] if len(invalid_right_frames) > 0 else []
            }
        })
        
        print(f"❌ {os.path.basename(hdf5_path)}: 发现手腕在头部上方的帧")
        if len(invalid_left_frames) > 0:
            idx = invalid_left_frames[0]
            print(f"   左手腕: 帧 {idx}, head_z={head_z[idx]:.3f}, wrist_z={left_wrist_z[idx]:.3f}")
        if len(invalid_right_frames) > 0:
            idx = invalid_right_frames[0]
            print(f"   右手腕: 帧 {idx}, head_z={head_z[idx]:.3f}, wrist_z={right_wrist_z[idx]:.3f}")

# 输出统计
print("\n" + "="*60)
print(f"检查结果汇总:")
print(f"  总文件数: {total_files}")
print(f"  总帧数: {total_frames}")
print(f"  全部符合条件: {valid_files}")
print(f"  存在问题: {len(invalid_files)}")

if invalid_files:
    print("\n存在问题的文件:")
    for detail in invalid_frames_details:
        print(f"  - {detail['file']}:")
        print(f"      总帧数: {detail['total_frames']}")
        print(f"      左手腕异常: {detail['invalid_left']} 帧")
        print(f"      右手腕异常: {detail['invalid_right']} 帧")