#!/usr/bin/env python3
"""对转换后的HDF5数据进行整体绕坐标轴旋转"""

import h5py
import numpy as np
import torch
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from pathlib import Path
from tqdm import tqdm
import argparse


def get_rotation_matrix(axis, angle_degrees):
    """
    生成绕指定坐标轴旋转的矩阵
    
    参数:
    axis: 'x', 'y', 或 'z'
    angle_degrees: 旋转角度（度）
    
    返回:
    3x3 旋转矩阵
    """
    angle = np.radians(angle_degrees)
    c, s = np.cos(angle), np.sin(angle)
    
    if axis == 'x':
        return np.array([
            [1, 0, 0],
            [0, c, -s],
            [0, s, c]
        ])
    elif axis == 'y':
        return np.array([
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c]
        ])
    elif axis == 'z':
        return np.array([
            [c, -s, 0],
            [s, c, 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")


def apply_rotation_to_position(pos, rotation_matrix):
    """
    对位置向量应用旋转
    
    参数:
    pos: 3维位置向量
    rotation_matrix: 3x3 旋转矩阵
    
    返回:
    旋转后的位置向量
    """
    return np.dot(rotation_matrix, pos)


def apply_rotation_to_rotation(rot_6d, rotation_matrix):
    """
    对6D旋转表示应用全局旋转
    
    参数:
    rot_6d: 6维旋转表示
    rotation_matrix: 3x3 全局旋转矩阵
    
    返回:
    旋转后的6D旋转表示
    """
    # 将6D转换为旋转矩阵
    rot_mat = rotation_6d_to_matrix(torch.tensor(rot_6d).unsqueeze(0)).numpy()[0]
    
    # 应用全局旋转（左乘）
    rot_mat_new = np.dot(rotation_matrix, rot_mat)
    
    # 转换回6D表示
    rot_6d_new = matrix_to_rotation_6d(torch.tensor(rot_mat_new)).numpy()
    
    return rot_6d_new


def apply_rotation_to_action(action, rotation_matrix):
    """
    对128维action应用全局旋转
    
    参数:
    action: 128维向量
    rotation_matrix: 3x3 全局旋转矩阵
    
    返回:
    旋转后的128维向量
    """
    action_rotated = action.copy()
    
    # 1. 头部位置 (0:3) 和旋转 (3:9)
    head_pos = action[0:3]
    head_rot_6d = action[3:9]
    
    action_rotated[0:3] = apply_rotation_to_position(head_pos, rotation_matrix)
    action_rotated[3:9] = apply_rotation_to_rotation(head_rot_6d, rotation_matrix)
    
    # 2. 左手关键点 (10:28) - 不需要旋转（相对于手腕）
    # 保持不变
    
    # 3. 右手腕位置 (30:33) 和旋转 (33:39)
    rw_pos = action[30:33]
    rw_rot_6d = action[33:39]
    
    action_rotated[30:33] = apply_rotation_to_position(rw_pos, rotation_matrix)
    action_rotated[33:39] = apply_rotation_to_rotation(rw_rot_6d, rotation_matrix)
    
    # 4. 右手关键点 (40:58) - 不需要旋转（相对于手腕）
    # 保持不变
    
    # 5. 左手腕位置 (80:83) 和旋转 (83:89)
    lw_pos = action[80:83]
    lw_rot_6d = action[83:89]
    
    action_rotated[80:83] = apply_rotation_to_position(lw_pos, rotation_matrix)
    action_rotated[83:89] = apply_rotation_to_rotation(lw_rot_6d, rotation_matrix)
    
    return action_rotated


def rotate_hdf5_file(input_path, output_path, rotation_matrix):
    """
    对单个HDF5文件应用旋转
    
    参数:
    input_path: 输入HDF5文件路径
    output_path: 输出HDF5文件路径
    rotation_matrix: 3x3 旋转矩阵
    """
    with h5py.File(input_path, 'r') as f_in:
        actions = f_in['action'][()]
        states = f_in['observation.state'][()]
        
        # 复制所有属性
        attrs = dict(f_in.attrs)
        
        # 检查是否有图像数据
        has_image = 'observation.image.top' in f_in
        if has_image:
            images = f_in['observation.image.top'][()]
    
    # 对所有帧应用旋转
    actions_rotated = np.zeros_like(actions)
    for i in range(len(actions)):
        actions_rotated[i] = apply_rotation_to_action(actions[i], rotation_matrix)
    
    states_rotated = actions_rotated.copy()
    
    # 保存旋转后的数据
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('action', data=actions_rotated, compression='gzip', compression_opts=4)
        f_out.create_dataset('observation.state', data=states_rotated, compression='gzip', compression_opts=4)
        if has_image:
            f_out.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
        
        # 复制属性并更新旋转信息
        for k, v in attrs.items():
            f_out.attrs[k] = v
        
        # 记录旋转信息
        rotation_info = attrs.get('rotation_info', [])
        rotation_info.append({
            'axis': rotation_matrix.get('axis', 'unknown'),
            'angle': rotation_matrix.get('angle', 0),
            'matrix': rotation_matrix.get('matrix', np.eye(3)).tolist()
        })
        f_out.attrs['rotation_info'] = str(rotation_info)


def batch_rotate(input_dir, output_dir, axis, angle, file_list=None):
    """
    批量旋转HDF5文件
    
    参数:
    input_dir: 输入目录
    output_dir: 输出目录
    axis: 旋转轴 ('x', 'y', 'z')
    angle: 旋转角度（度）
    file_list: 指定要处理的文件列表（可选）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成旋转矩阵
    rotation_matrix = get_rotation_matrix(axis, angle)
    
    # 创建旋转信息字典
    rotation_info = {
        'axis': axis,
        'angle': angle,
        'matrix': rotation_matrix
    }
    
    print(f"旋转矩阵（绕{axis}轴旋转{angle}度）:")
    print(rotation_matrix)
    print()
    
    # 获取文件列表
    if file_list:
        hdf5_files = [Path(f) for f in file_list]
    else:
        hdf5_files = sorted(input_path.glob('*.hdf5'))
    
    print(f"找到 {len(hdf5_files)} 个HDF5文件")
    print()
    
    # 批量处理
    results = {'success': [], 'failed': []}
    
    for hdf5_file in tqdm(hdf5_files, desc="旋转进度"):
        output_file = output_path / hdf5_file.name
        
        try:
            rotate_hdf5_file(hdf5_file, output_file, rotation_matrix)
            results['success'].append(hdf5_file.name)
        except Exception as e:
            results['failed'].append((hdf5_file.name, str(e)))
            print(f"  ❌ 处理失败 {hdf5_file.name}: {e}")
    
    # 输出结果
    print(f"\n处理完成:")
    print(f"  成功: {len(results['success'])}")
    print(f"  失败: {len(results['failed'])}")
    
    if results['failed']:
        print("\n失败的文件:")
        for name, error in results['failed']:
            print(f"  - {name}: {error}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='对HDF5数据进行整体绕坐标轴旋转',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 绕z轴旋转90度
  python rotate_hdf5.py -i input_dir -o output_dir -a z -d 90
  
  # 绕x轴旋转-45度
  python rotate_hdf5.py -i input_dir -o output_dir -a x -d -45
  
  # 只处理指定文件
  python rotate_hdf5.py -i input_dir -o output_dir -a y -d 180 --files file1.hdf5 file2.hdf5
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入目录')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('-a', '--axis', type=str, required=True, choices=['x', 'y', 'z'],
                        help='旋转轴 (x, y, 或 z)')
    parser.add_argument('-d', '--degrees', type=float, required=True,
                        help='旋转角度（度）')
    parser.add_argument('--files', nargs='+', type=str,
                        help='指定要处理的文件列表（可选）')
    
    args = parser.parse_args()
    
    batch_rotate(
        input_dir=args.input,
        output_dir=args.output,
        axis=args.axis,
        angle=args.degrees,
        file_list=args.files
    )


if __name__ == '__main__':
    main()