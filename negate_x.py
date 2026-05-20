#!/usr/bin/env python3
"""将所有绝对位置的x坐标取相反数（关键点不动）"""

import h5py
import numpy as np
import os
from glob import glob

# 数据目录
input_dir = "/data1/zxlei/dataset/mocap_converted2"
output_dir = "/data1/zxlei/dataset/mocap_converted_x_negated"

# 位置索引定义
HEAD_POS_IDX = np.arange(0, 3)        # 头部位置 [x, y, z]
LEFT_EEF_POS_IDX = np.arange(80, 83)   # 左手腕位置
RIGHT_EEF_POS_IDX = np.arange(30, 33)  # 右手腕位置

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 获取所有HDF5文件
hdf5_files = sorted(glob(os.path.join(input_dir, "*.hdf5")))
print(f"找到 {len(hdf5_files)} 个HDF5文件")

for hdf5_path in hdf5_files:
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"][()]
        state = f["observation.state"][()] if "observation.state" in f else None
        image_top = f["observation.image.top"][()] if "observation.image.top" in f else None
        attrs = dict(f.attrs)
    
    print(f"处理 {os.path.basename(hdf5_path)}")
    
    # 只对绝对位置的x坐标取相反数，关键点（相对坐标）保持不变
    
    # 头部位置x
    actions[:, HEAD_POS_IDX[0]] *= -1
    
    # 左手腕位置x
    actions[:, LEFT_EEF_POS_IDX[0]] *= -1
    
    # 右手腕位置x
    actions[:, RIGHT_EEF_POS_IDX[0]] *= -1
    
    # 注意：关键点是相对手掌的偏移量，不进行x取反
    
    # 对state做同样处理
    if state is not None:
        state[:, HEAD_POS_IDX[0]] *= -1
        state[:, LEFT_EEF_POS_IDX[0]] *= -1
        state[:, RIGHT_EEF_POS_IDX[0]] *= -1
        # 关键点同样保持不变
    
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
        f.attrs['x_negated'] = True
    
    print(f"  已保存到 {output_path}")

print(f"\n完成！数据保存到 {output_dir}")
print("注意：仅修改了绝对位置（头部、左右手位置），关键点保持不变")