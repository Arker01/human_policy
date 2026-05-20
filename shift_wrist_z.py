#!/usr/bin/env python3
"""将手腕的z轴坐标向负方向平移指定距离"""

import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse


def shift_wrist_z_coordinate(actions, shift_distance):
    """
    对action数组中的手腕z坐标进行平移
    
    参数:
    actions: (num_frames, 128) 的action数组
    shift_distance: 平移距离（正值表示向负方向移动）
    
    返回:
    平移后的actions数组
    """
    actions_shifted = actions.copy()
    
    # 右手腕位置索引 (30:33) - z坐标是索引32
    RIGHT_EEF_POS_IDX = np.arange(30, 33)
    RIGHT_EEF_Z_IDX = 32
    
    # 左手腕位置索引 (80:83) - z坐标是索引82  
    LEFT_EEF_POS_IDX = np.arange(80, 83)
    LEFT_EEF_Z_IDX = 82
    
    # 头部位置索引 (0:3) - z坐标是索引2（可选，根据需求决定是否平移）
    HEAD_POS_IDX = np.arange(0, 3)
    HEAD_Z_IDX = 2
    
    # 向负方向平移手腕z坐标
    actions_shifted[:, RIGHT_EEF_Z_IDX] -= shift_distance
    actions_shifted[:, LEFT_EEF_Z_IDX] -= shift_distance
    
    # 可选：如果需要同时平移头部（保持相对位置）
    # actions_shifted[:, HEAD_Z_IDX] -= shift_distance
    
    return actions_shifted


def process_single_file(input_path, output_path, shift_distance):
    """
    处理单个HDF5文件
    
    参数:
    input_path: 输入文件路径
    output_path: 输出文件路径
    shift_distance: z轴平移距离
    """
    with h5py.File(input_path, 'r') as f_in:
        actions = f_in['action'][()]
        states = f_in['observation.state'][()] if 'observation.state' in f_in else None
        
        # 复制所有属性
        attrs = dict(f_in.attrs)
        
        # 检查是否有图像数据
        has_image = 'observation.image.top' in f_in
        if has_image:
            images = f_in['observation.image.top'][()]
    
    # 平移手腕z坐标
    actions_shifted = shift_wrist_z_coordinate(actions, shift_distance)
    
    # 如果有state，同样处理
    if states is not None:
        states_shifted = shift_wrist_z_coordinate(states, shift_distance)
    else:
        states_shifted = None
    
    # 保存处理后的数据
    with h5py.File(output_path, 'w') as f_out:
        f_out.create_dataset('action', data=actions_shifted, compression='gzip', compression_opts=4)
        
        if states_shifted is not None:
            f_out.create_dataset('observation.state', data=states_shifted, compression='gzip', compression_opts=4)
        
        if has_image:
            f_out.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
        
        # 复制属性
        for k, v in attrs.items():
            f_out.attrs[k] = v
        
        # 记录平移信息
        f_out.attrs['z_shift_distance'] = shift_distance
        f_out.attrs['z_shift_direction'] = 'negative'


def batch_process(input_dir, output_dir, shift_distance, file_list=None):
    """
    批量处理HDF5文件
    
    参数:
    input_dir: 输入目录
    output_dir: 输出目录
    shift_distance: z轴平移距离
    file_list: 指定文件列表（可选）
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"将手腕z坐标向负方向平移 {shift_distance} 单位")
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
    
    for hdf5_file in tqdm(hdf5_files, desc="处理进度"):
        output_file = output_path / hdf5_file.name
        
        try:
            process_single_file(hdf5_file, output_file, shift_distance)
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
        description='将手腕的z轴坐标向负方向平移指定距离',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 将手腕z坐标向负方向平移0.5单位
  python shift_wrist_z.py -i input_dir -o output_dir -d 0.5
  
  # 平移更大距离
  python shift_wrist_z.py -i input_dir -o output_dir -d 1.0
  
  # 只处理指定文件
  python shift_wrist_z.py -i input_dir -o output_dir -d 0.3 --files episode_0.hdf5 episode_1.hdf5
        """
    )
    
    parser.add_argument('-i', '--input', type=str, required=True,
                        help='输入目录')
    parser.add_argument('-o', '--output', type=str, required=True,
                        help='输出目录')
    parser.add_argument('-d', '--distance', type=float, required=True,
                        help='向负方向平移的距离（正值）')
    parser.add_argument('--files', nargs='+', type=str,
                        help='指定要处理的文件列表（可选）')
    
    args = parser.parse_args()
    
    batch_process(
        input_dir=args.input,
        output_dir=args.output,
        shift_distance=args.distance,
        file_list=args.files
    )


if __name__ == '__main__':
    main()