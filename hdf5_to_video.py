#!/usr/bin/env python3
"""将HDF5文件中的observation.image转换为视频"""

import h5py
import numpy as np
import cv2
from pathlib import Path
import argparse
import os

def hdf5_to_video(hdf5_path, output_dir=None, fps=15, image_key='observation.image.top'):
    """
    将HDF5文件中的图片序列转换为视频
    
    Args:
        hdf5_path: HDF5文件路径
        output_dir: 输出目录，默认为当前目录
        fps: 视频帧率
        image_key: 图片数据的键名
    """
    hdf5_path = Path(hdf5_path)
    
    if not hdf5_path.exists():
        print(f"错误：文件不存在: {hdf5_path}")
        return
    
    if output_dir is None:
        output_dir = hdf5_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # 检查可用的图片键
            available_keys = [k for k in f.keys() if 'image' in k.lower()]
            if not available_keys:
                print(f"错误：{hdf5_path.name} 中没有找到图片数据集")
                print(f"可用的数据集: {list(f.keys())}")
                return
            
            if image_key not in f:
                print(f"警告：指定的键 '{image_key}' 不存在")
                print(f"可用的图片键: {available_keys}")
                image_key = available_keys[0]
            
            images = f[image_key][:]
            print(f"找到 {len(images)} 帧图片")
            print(f"图片形状: {images.shape}")
            
            if len(images) == 0:
                print("错误：图片数据为空")
                return
            
            # 获取图片尺寸
            img = images[0]
            # 处理不同的形状格式 (可能是 (H, W, 3) 或 (3, H, W))
            if img.shape[0] == 3:  # (3, H, W) 格式
                img = img.transpose(1, 2, 0)
                height, width = img.shape[:2]
            else:  # (H, W, 3) 格式
                height, width = img.shape[:2]
            
            # 确保图片是 uint8 类型
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            
            # 构建输出视频路径
            video_name = hdf5_path.stem + '.mp4'
            video_path = output_dir / video_name
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"错误：无法创建视频文件: {video_path}")
                return
            
            # 写入每一帧
            print(f"正在生成视频...")
            for i, img in enumerate(images):
                # 处理不同的形状格式
                if img.shape[0] == 3:
                    img = img.transpose(1, 2, 0)
                
                # 确保图片是 uint8 类型
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # OpenCV 期望 BGR 格式，而图片可能是 RGB 格式
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                out.write(img_bgr)
                
                # 显示进度
                if (i + 1) % 50 == 0 or i == len(images) - 1:
                    print(f"进度: [{i+1}/{len(images)}]", end='\r')
            
            out.release()
            print(f"\n视频已保存到: {video_path}")
            
            # 显示统计信息
            print(f"\n统计信息:")
            print(f"  总帧数: {len(images)}")
            print(f"  视频时长: {len(images) / fps:.2f} 秒")
            print(f"  分辨率: {width} x {height}")
            print(f"  帧率: {fps} FPS")
            
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()

def list_image_keys(hdf5_path):
    """列出HDF5文件中所有的图片键"""
    with h5py.File(hdf5_path, 'r') as f:
        keys = [k for k in f.keys() if 'image' in k.lower()]
        print(f"可用的图片键:")
        for key in keys:
            data = f[key]
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")

def main():
    parser = argparse.ArgumentParser(description='将HDF5文件中的图片转换为视频')
    parser.add_argument('input', type=str, help='HDF5文件路径或包含HDF5文件的目录')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出目录')
    parser.add_argument('--fps', type=int, default=15, help='视频帧率 (默认: 15)')
    parser.add_argument('--key', '-k', type=str, default='observation.image.top',
                        help='图片数据的键名')
    parser.add_argument('--list-keys', action='store_true', help='列出文件中可用的图片键')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.list_keys:
        list_image_keys(input_path)
        return
    
    if input_path.is_file() and input_path.suffix == '.hdf5':
        # 处理单个文件
        hdf5_to_video(input_path, args.output, args.fps, args.key)
    elif input_path.is_dir():
        # 处理目录中的所有HDF5文件
        hdf5_files = sorted(input_path.glob('episode_*.hdf5'))
        print(f"找到 {len(hdf5_files)} 个HDF5文件")
        
        for i, hdf5_file in enumerate(hdf5_files):
            print(f"\n[{i+1}/{len(hdf5_files)}] 处理: {hdf5_file.name}")
            hdf5_to_video(hdf5_file, args.output, args.fps, args.key)
    else:
        print(f"错误：输入不是有效的文件或目录: {args.input}")

if __name__ == '__main__':
    main()
