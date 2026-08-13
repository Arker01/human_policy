#!/usr/bin/env python3
"""验证模型在 eval_whole_last 数据上的 head 输出是否为 0"""

import os
import sys
import h5py
import torch
import yaml
import numpy as np

# 添加项目路径和 hdt 路径
sys.path.insert(0, '/home/aigc/human_policy')
sys.path.insert(0, '/home/aigc/human_policy/hdt')

# 现在可以正确导入了
from hdt import constants
from hdt.policy import ACTPolicy

def convert_numeric_values(d):
    """递归地将字符串类型的数值转换为正确的数值类型"""
    if isinstance(d, dict):
        for k, v in d.items():
            d[k] = convert_numeric_values(v)
    elif isinstance(d, str):
        # 尝试转换为数值
        try:
            # 先尝试转换为整数
            return int(d)
        except ValueError:
            try:
                # 再尝试转换为浮点数
                return float(d)
            except ValueError:
                # 保持字符串
                return d
    return d

def load_checkpoint(ckpt_path):
    """加载模型 checkpoint"""
    print(f"Loading checkpoint: {ckpt_path}")
    # 使用 CPU 加载
    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    return checkpoint

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    # 转换数值类型
    config = convert_numeric_values(config)
    return config

def load_hdf5_data(file_path):
    """加载单个 HDF5 数据文件"""
    with h5py.File(file_path, 'r') as f:
        data = {
            'observation.image': f['observation.image.top'][()],
            'observation.state': f['observation.state'][()],
            'action': f['action'][()]
        }
    return data

def check_head_output(policy, data_dir):
    """检查 head 输出的 xyz 值，同时输出 GT"""
    head_xyz_pred = []
    head_xyz_gt = []
    
    # 获取所有 HDF5 文件
    hdf5_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hdf5')])
    
    print(f"Found {len(hdf5_files)} files in {data_dir}")
    
    for idx, file_name in enumerate(hdf5_files[:5]):  # 只检查前5个文件
        file_path = os.path.join(data_dir, file_name)
        print(f"\nProcessing {file_name}...")
        
        try:
            data = load_hdf5_data(file_path)
            
            # 准备输入数据
            image = torch.from_numpy(data['observation.image']).float()
            state = torch.from_numpy(data['observation.state']).float()
            action_gt = torch.from_numpy(data['action']).float()  # GT action
            
            # 图像形状: (timesteps, H, W, C) -> 需要转换为 (timesteps, C, H, W)
            image = image.permute(0, 3, 1, 2)  # 交换通道位置
            
            num_timesteps = image.shape[0]
            print(f"  Total timesteps: {num_timesteps}")
            
            # 遍历所有帧
            for t in range(num_timesteps):
                # 准备当前帧的输入
                state_t = state[t:t+1]  # (1, state_dim)
                image_t = image[t:t+1].unsqueeze(1)  # (1, 1, C, H, W)
                
                # 推理（注意参数顺序：image在前，state/qpos在后）
                with torch.no_grad():
                    # 获取模型输出
                    action_pred = policy(image_t, state_t, None, None, None)
                    
                    # 提取 head 输出 (OUTPUT_HEAD_EEF = [0, 1, 2, 3, 4, 5, 6, 7, 8])
                    head_output = action_pred[:, :, constants.OUTPUT_HEAD_EEF]
                    
                    # 提取 GT head 输出（当前帧的 action）
                    head_gt = action_gt[t:t+1][:, constants.OUTPUT_HEAD_EEF]
                    
                    # 提取 xyz 分量（前3个维度：索引 0, 1, 2）
                    head_xyz = head_output[:, :, :3]  # xyz
                    head_xyz_gt_val = head_gt[:, :3]   # xyz
                    
                    # 保存 xyz 输出
                    head_xyz_pred.append(head_xyz.cpu().numpy())
                    # 重复 GT 以匹配预测的形状（100 queries）
                    head_xyz_gt_repeated = head_xyz_gt_val.unsqueeze(1).repeat(1, head_xyz.shape[1], 1).cpu().numpy()
                    head_xyz_gt.append(head_xyz_gt_repeated)
            
            # 打印单个文件的 xyz 值示例（前3帧，每帧前3个 query）
            print(f"  === Predicted Head XYZ (first 3 timesteps, first 3 queries each) ===")
            for t in range(min(3, num_timesteps)):
                pred_t = head_xyz_pred[-num_timesteps + t]
                print(f"  Timestep {t}:")
                for i in range(min(3, pred_t.shape[1])):
                    x_val = float(pred_t[0, i, 0])
                    y_val = float(pred_t[0, i, 1])
                    z_val = float(pred_t[0, i, 2])
                    print(f"    Query {i}: x={x_val:.6f}, y={y_val:.6f}, z={z_val:.6f}")
            
            print(f"  === GT Head XYZ (first 3 timesteps) ===")
            for t in range(min(3, num_timesteps)):
                gt_t = head_xyz_gt[-num_timesteps + t]
                x_val = float(gt_t[0, 0, 0])
                y_val = float(gt_t[0, 0, 1])
                z_val = float(gt_t[0, 0, 2])
                print(f"  Timestep {t}: x={x_val:.6f}, y={y_val:.6f}, z={z_val:.6f}")
                
        except Exception as e:
            print(f"  Error processing {file_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总统计
    if head_xyz_pred and head_xyz_gt:
        # 合并所有数据
        all_xyz_pred = np.concatenate([h.reshape(-1, 3) for h in head_xyz_pred])
        all_xyz_gt = np.concatenate([h.reshape(-1, 3) for h in head_xyz_gt])
        
        print(f"\n===== Overall Head XYZ Statistics =====")
        
        # 分别统计 x, y, z
        for dim, label in enumerate(['X', 'Y', 'Z']):
            print(f"\n--- {label} Dimension ---")
            print(f"  Predicted:")
            print(f"    Mean: {np.mean(all_xyz_pred[:, dim]):.6f}")
            print(f"    Max: {np.max(all_xyz_pred[:, dim]):.6f}")
            print(f"    Min: {np.min(all_xyz_pred[:, dim]):.6f}")
            print(f"    Std: {np.std(all_xyz_pred[:, dim]):.6f}")
            
            print(f"  Ground Truth:")
            print(f"    Mean: {np.mean(all_xyz_gt[:, dim]):.6f}")
            print(f"    Max: {np.max(all_xyz_gt[:, dim]):.6f}")
            print(f"    Min: {np.min(all_xyz_gt[:, dim]):.6f}")
            print(f"    Std: {np.std(all_xyz_gt[:, dim]):.6f}")
            
            print(f"  Error (pred - gt):")
            print(f"    Mean error: {np.mean(all_xyz_pred[:, dim] - all_xyz_gt[:, dim]):.6f}")

if __name__ == '__main__':
    # 配置路径
    ckpt_path = '/home/aigc/human_policy/train1_finetune_task2_ckpt/policy_last.ckpt'
    config_path = '/home/aigc/human_policy/hdt/configs/models/act_resnet.yaml'
    data_dir = '/home/aigc/human_policy/data/eval_whole_last'
    
    # 检查路径是否存在
    if not os.path.exists(ckpt_path):
        print(f"Error: Checkpoint not found at {ckpt_path}")
        exit(1)
    
    if not os.path.exists(config_path):
        print(f"Error: Config not found at {config_path}")
        exit(1)
    
    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found at {data_dir}")
        exit(1)
    
    # 加载配置
    config = load_config(config_path)
    
    # 合并 common 和 model 配置
    policy_config = {}
    policy_config.update(config['common'])
    policy_config.update(config['model'])
    
    # 添加训练时使用的参数（从 checkpoint 形状推断）
    policy_config['num_queries'] = 100  # checkpoint 显示使用的是 100
    policy_config['chunk_size'] = 100
    
    # 打印配置信息
    print(f"Loaded config:")
    print(f"  state_dim: {policy_config.get('state_dim')}")
    print(f"  action_dim: {policy_config.get('action_dim')}")
    print(f"  backbone: {policy_config.get('backbone')}")
    print(f"  lr_backbone: {policy_config.get('lr_backbone')}")
    print(f"  lr_backbone type: {type(policy_config.get('lr_backbone'))}")
    print(f"  num_queries: {policy_config.get('num_queries')}")
    print(f"  chunk_size: {policy_config.get('chunk_size')}")
    
    # 加载模型
    checkpoint = load_checkpoint(ckpt_path)
    
    # 创建并加载模型（使用 CPU）
    policy = ACTPolicy(policy_config)
    policy.load_state_dict(checkpoint)
    policy.eval()
    
    print(f"\nModel loaded successfully!")
    print(f"Backbone: {config['model']['backbone']}")
    print(f"Device: CPU")
    
    # 检查 head 输出
    check_head_output(policy, data_dir)