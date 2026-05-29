#!/usr/bin/env python3
"""将 Arkit JSON 数据转换为标准 128 维格式"""

import json
import h5py
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
import cv2

# 指尖在 orderedJoints 数组中的索引
FINGERTIP_INDICES = [4, 9, 14, 19, 24]  # 拇指、食指、中指、无名指、小指


class ArkitJsonTo128Converter:
    def __init__(self, input_dir, output_dir, skip_missing_mp4=True, max_video_frames=1000):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.skip_missing_mp4 = skip_missing_mp4
        self.max_video_frames = max_video_frames  # 异常视频最大保留帧数
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def col_major_to_row_major(self, col_major):
        """将列主序的 4x4 矩阵转换为行主序"""
        matrix = np.array(col_major, dtype=np.float32).reshape(4, 4)
        return matrix  # 不转置，保持原始顺序

    def extract_6d_rotation(self, rotation_matrix):
        """从 3x3 旋转矩阵提取 6D 旋转表示（使用 PyTorch3D）"""
        import torch
        # 将 numpy 数组转换为 torch tensor
        rot_mat_tensor = torch.tensor(rotation_matrix, dtype=torch.float32)
        # 使用 PyTorch3D 的 matrix_to_rotation_6d 转换
        rotation_6d_tensor = matrix_to_rotation_6d(rot_mat_tensor)
        # 转换回 numpy 数组
        return rotation_6d_tensor.numpy()

    def rotation_6d_to_matrix(self, rotation_6d):
        """将 6D 旋转表示转换回 3x3 旋转矩阵（使用 PyTorch3D）"""
        import torch
        # 将 numpy 数组转换为 torch tensor
        rotation_6d_tensor = torch.tensor(rotation_6d, dtype=torch.float32)
        # 使用 PyTorch3D 的 rotation_6d_to_matrix 转换
        rot_mat_tensor = rotation_6d_to_matrix(rotation_6d_tensor)
        # 转换回 numpy 数组
        return rot_mat_tensor.numpy()

    def extract_position(self, transform_4x4):
        """从 4x4 变换矩阵提取位置"""
        return transform_4x4[:3, 3]

    def extract_fingertip_positions(self, skeleton_joints):
        """从骨架数据中提取 5 个指尖的位置（相对于手腕）
        
        skeleton_joints: 400 个 float (25个关节 × 4x4 矩阵)
        返回: 5 个 (x, y, z) 坐标列表
        """
        fingertips = []
        for idx in FINGERTIP_INDICES:
            base = idx * 16  # 每个关节占 16 个 float
            # 提取平移分量（行主序的第 3, 7, 11 位 = 第 3 列）
            x = skeleton_joints[base + 3]
            y = skeleton_joints[base + 7]
            z = skeleton_joints[base + 11]
            fingertips.append(np.array([x, y, z], dtype=np.float32))
        return fingertips

    def extract_keypoints(self, finger_tips_local):
        """提取 6 个关键点 (1 个手掌原点 + 5 个指尖)"""
        keypoints = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]  # 手掌原点
        keypoints.extend(finger_tips_local)
        return np.concatenate(keypoints)

    def extract_video_frames(self, mp4_path, target_frames, max_size_mb=5):
        """从 MP4 视频中提取帧，并压缩分辨率以控制文件大小
        
        Args:
            mp4_path: 视频文件路径
            target_frames: 目标帧数（来自坐标数据）
            max_size_mb: 最大文件大小（MB），用于计算缩放比例
            
        Returns:
            提取的帧数组，如果失败返回 None
        """
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            print(f"  警告: 无法打开视频文件 {mp4_path}")
            return None
            
        frames = []

        # 获取原始视频分辨率
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 固定缩放到训练要求的分辨率（320×240，宽×高）
        target_width = 320
        target_height = 240
        
        print(f"  视频分辨率: {orig_width}x{orig_height} -> {target_width}x{target_height}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # 缩放帧
            frame_resized = cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()

        if len(frames) == 0:
            return None

        return np.array(frames)

    def convert_single(self, json_path, mp4_path=None):
        """转换单个 JSON 文件"""
        # 读取 JSON 数据
        with open(json_path, 'r') as f:
            frames = json.load(f)
        
        num_frames = len(frames)
        print(f"  处理 {num_frames} 帧数据")
        
        actions = []
        
        for frame in frames:
            action = np.zeros(128, dtype=np.float32)
            
            # 头部 (camera/neck) - 世界坐标系
            head_matrix = self.col_major_to_row_major(frame['head'])
            action[0:3] = self.extract_position(head_matrix)
            action[3:9] = self.extract_6d_rotation(head_matrix[:3, :3])
            
            # 左手腕 - 世界坐标系
            left_wrist_matrix = self.col_major_to_row_major(frame['leftWrist'])
            action[80:83] = self.extract_position(left_wrist_matrix)
            action[83:89] = self.extract_6d_rotation(left_wrist_matrix[:3, :3])
            
            # 右手腕 - 世界坐标系
            right_wrist_matrix = self.col_major_to_row_major(frame['rightWrist'])
            action[30:33] = self.extract_position(right_wrist_matrix)
            action[33:39] = self.extract_6d_rotation(right_wrist_matrix[:3, :3])
            
            # 左手关键点（相对于左手腕）
            left_fingertips = self.extract_fingertip_positions(frame['leftSkeleton']['joints'])
            action[10:28] = self.extract_keypoints(left_fingertips)
            
            # 右手关键点（相对于右手腕）
            right_fingertips = self.extract_fingertip_positions(frame['rightSkeleton']['joints'])
            action[40:58] = self.extract_keypoints(right_fingertips)
            
            actions.append(action)
        
        actions = np.array(actions, dtype=np.float32)
        states = actions.copy()  # 状态通常与动作相同
        
        # 处理视频
        images = None
        if mp4_path and mp4_path.exists():
            images = self.extract_video_frames(mp4_path, num_frames, max_size_mb=3)
            # action 帧数只可能比视频少，extract_video_frames 会确保返回目标帧数
            # 因此无需处理 action 帧数大于视频的情况
        elif not self.skip_missing_mp4:
            print(f"  警告: 未找到视频文件 {mp4_path}，跳过")
            return None
        
        # 保存为 HDF5
        hdf5_name = json_path.with_suffix('.hdf5').name
        output_path = self.output_dir / hdf5_name
        
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('action', data=actions, compression='gzip', compression_opts=4)
            f.create_dataset('observation.state', data=states, compression='gzip', compression_opts=4)
            if images is not None:
                f.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
            f.attrs['x_negated'] = False
            f.attrs['source'] = str(json_path)
        
        return output_path

    def convert_batch(self, file_list=None):
        """批量转换"""
        if file_list is None:
            json_files = sorted(self.input_dir.glob('wholebody-*.json'))
        else:
            json_files = [Path(f) for f in file_list]
        
        print(f"找到 {len(json_files)} 个 JSON 文件")
        
        results = {'success': [], 'skipped': [], 'failed': []}
        
        for json_path in tqdm(json_files, desc="转换进度"):
            # 从 wholebody-N.json 提取编号 N
            # 文件名格式: wholebody-1.json -> wb-1.MP4
            json_name = json_path.stem  # wholebody-1
            if json_name.startswith('wholebody-'):
                mp4_base = 'wb-' + json_name.replace('wholebody-', '')
            else:
                mp4_base = json_name
            
            # 尝试多种可能的文件名（处理可能的前导空格）
            mp4_path = None
            for suffix in ['.MP4', '.mp4']:
                # 标准文件名
                candidate = self.input_dir / (mp4_base + suffix)
                if candidate.exists():
                    mp4_path = candidate
                    break
                # 带前导空格的文件名
                candidate = self.input_dir / (' ' + mp4_base + suffix)
                if candidate.exists():
                    mp4_path = candidate
                    break
                # 使用 glob 查找匹配的文件
                candidates = list(self.input_dir.glob(f'*{mp4_base}{suffix}'))
                if candidates:
                    mp4_path = candidates[0]
                    break
            
            if mp4_path is None:
                print(f"  ⚠️ 未找到视频文件: {mp4_base}.MP4")
            
            try:
                output_path = self.convert_single(json_path, mp4_path)
                if output_path:
                    results['success'].append(json_path.name)
                    print(f"  ✅ 已保存到 {output_path}")
                else:
                    results['skipped'].append(json_path.name)
            except Exception as e:
                results['failed'].append((json_path.name, str(e)))
                print(f"  ❌ 转换失败 {json_path.name}: {e}")
        
        print(f"\n转换完成:")
        print(f"  成功: {len(results['success'])}")
        print(f"  跳过: {len(results['skipped'])}")
        print(f"  失败: {len(results['failed'])}")
        
        if results['failed']:
            print("\n失败详情:")
            for name, error in results['failed']:
                print(f"  - {name}: {error}")
        
        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='将 Arkit JSON 转换为 128 维格式')
    parser.add_argument('--input', '-i', type=str,
                        default='/data1/zxlei/dataset/data-1',
                        help='输入目录')
    parser.add_argument('--output', '-o', type=str,
                        default='/data1/zxlei/dataset/arkit_converted',
                        help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=150,
                        help='最多转换的文件数量')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='从第几个文件开始（索引，从0开始）')
    parser.add_argument('--no-skip', action='store_true',
                        help='如果视频缺失则不跳过（报错）')
    parser.add_argument('--max-frames', type=int, default=300,
                        help='异常视频最大保留帧数')
    
    args = parser.parse_args()
    
    converter = ArkitJsonTo128Converter(
        input_dir=args.input,
        output_dir=args.output,
        skip_missing_mp4=not args.no_skip,
        max_video_frames=args.max_frames
    )
    
    all_files = sorted(converter.input_dir.glob('wholebody-*.json'))
    start_idx = args.start
    end_idx = args.start + args.num if args.num is not None else None
    file_list = all_files[start_idx:end_idx]
    
    print(f"文件范围: [{start_idx}, {end_idx if end_idx else len(all_files)}), 共 {len(file_list)} 个文件")
    
    converter.convert_batch(file_list)


if __name__ == '__main__':
    main()
