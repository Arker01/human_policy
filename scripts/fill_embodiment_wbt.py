#!/usr/bin/env python3
"""
为WBT数据文件填充正确的embodiment字段（Dex5, Inspire, Brainco）
"""

import argparse
import glob
import h5py
import os

def fill_embodiment(input_dir):
    """为目录中的所有hdf5文件填充embodiment字段"""
    files = sorted(glob.glob(os.path.join(input_dir, '*.hdf5')))
    print(f"找到 {len(files)} 个文件")
    
    count_dex5 = 0
    count_inspire = 0
    count_brainco = 0
    count_unknown = 0
    
    for filepath in files:
        filename = os.path.basename(filepath)
        
        # 根据文件名确定型号
        if 'Dex5' in filename:
            embodiment = 'robot_Dex5'
            count_dex5 += 1
        elif 'Inspire' in filename:
            embodiment = 'robot_Inspire'
            count_inspire += 1
        elif 'Brainco' in filename:
            embodiment = 'robot_Brainco'
            count_brainco += 1
        else:
            print(f"  未知型号: {filename}")
            count_unknown += 1
            continue
        
        # 打开文件并写入embodiment字段
        with h5py.File(filepath, 'r+') as f:
            f.attrs['embodiment'] = embodiment
        
        print(f"  {filename[:50]}... -> {embodiment}")
    
    print(f"\n处理完成！")
    print(f"  Dex5: {count_dex5} 个文件")
    print(f"  Inspire: {count_inspire} 个文件")
    print(f"  Brainco: {count_brainco} 个文件")
    print(f"  未知型号: {count_unknown} 个文件")

def main():
    parser = argparse.ArgumentParser(description='为WBT数据文件填充embodiment字段')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='输入数据目录')
    args = parser.parse_args()
    
    fill_embodiment(args.input_dir)

if __name__ == '__main__':
    main()