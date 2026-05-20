#!/usr/bin/env python3
"""将 mocap HDF5 数据转换为标准 128 维格式"""

import h5py
import numpy as np
import os
from pathlib import Path
import cv2
from tqdm import tqdm
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

MOCAP_KEYPOINTS_LEFT = [
    'leftThumbTip',
    'leftIndexFingerTip',
    'leftMiddleFingerTip',
    'leftRingFingerTip',
    'leftLittleFingerTip',
]

MOCAP_KEYPOINTS_RIGHT = [
    'rightThumbTip',
    'rightIndexFingerTip',
    'rightMiddleFingerTip',
    'rightRingFingerTip',
    'rightLittleFingerTip',
]

class MocapTo128Converter:
    def __init__(self, mocap_dir, output_dir, skip_missing_mp4=True):
        self.mocap_dir = Path(mocap_dir)
        self.output_dir = Path(output_dir)
        self.skip_missing_mp4 = skip_missing_mp4
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_6d_rotation(self, rotation_matrix):
        """从 3x3 旋转矩阵提取 6D 旋转表示（使用 PyTorch3D）"""
        import torch
        # 将 numpy 数组转换为 torch tensor
        rot_mat_tensor = torch.tensor(rotation_matrix, dtype=torch.float32)
        # 使用 PyTorch3D 的 matrix_to_rotation_6d 转换
        rotation_6d_tensor = matrix_to_rotation_6d(rot_mat_tensor)
        # 转换回 numpy 数组
        return rotation_6d_tensor.numpy()

    def extract_eef(self, transform_4x4):
        """提取 EEF (位置 + 6D 旋转)"""
        position = transform_4x4[:3, 3]
        rotation_6d = self.extract_6d_rotation(transform_4x4[:3, :3])
        return np.concatenate([position, rotation_6d])

    def extract_keypoints(self, hand_transform, finger_tips, hand_name='left'):
        """提取 6 个关键点 (1 个手掌原点 + 5 个指尖偏移)
        
        使用手腕逆变换将手指从世界坐标系转换到手腕局部坐标系
        """
        hand_pos = hand_transform[:3, 3]
        hand_rot = hand_transform[:3, :3]
        
        # 计算手腕的逆旋转矩阵（转置矩阵 = 逆矩阵，因为是正交矩阵）
        hand_rot_inv = hand_rot.T
        
        keypoints = [np.array([0.0, 0.0, 0.0])]  # 手掌原点（相对于手腕）

        for finger_tip in finger_tips:
            tip_pos_world = finger_tip[:3, 3]
            # 步骤 1: 减去手腕位置（平移到手腕原点）
            tip_pos_relative = tip_pos_world - hand_pos
            # 步骤 2: 应用逆旋转（转回手腕局部坐标系）
            tip_pos_local = np.dot(hand_rot_inv, tip_pos_relative)
            keypoints.append(tip_pos_local)

        return np.concatenate(keypoints)

    def convert_single(self, hdf5_path, mp4_path=None):
        """转换单个 HDF5 文件"""
        with h5py.File(hdf5_path, 'r') as f:
            transforms = f['transforms']
            num_frames = transforms['hip'].shape[0]

            actions = []
            states = []
            images = []

            for frame_idx in range(num_frames):
                action = np.zeros(128)

                neck = transforms['camera'][frame_idx]
                action[0:3] = neck[:3, 3]
                action[3:9] = self.extract_6d_rotation(neck[:3, :3])

                right_hand = transforms['rightHand'][frame_idx]
                action[30:33] = right_hand[:3, 3]
                action[33:39] = self.extract_6d_rotation(right_hand[:3, :3])

                left_hand = transforms['leftHand'][frame_idx]
                action[80:83] = left_hand[:3, 3]
                action[83:89] = self.extract_6d_rotation(left_hand[:3, :3])

                # 使用 extract_keypoints 方法处理手指关键点（包含旋转逆变换）
                right_finger_tips = [transforms[name][frame_idx] for name in MOCAP_KEYPOINTS_RIGHT]
                action[40:58] = self.extract_keypoints(right_hand, right_finger_tips, 'right')

                left_finger_tips = [transforms[name][frame_idx] for name in MOCAP_KEYPOINTS_LEFT]
                action[10:28] = self.extract_keypoints(left_hand, left_finger_tips, 'left')

                actions.append(action)
                states.append(action.copy())

                if mp4_path and mp4_path.exists():
                    pass

            actions = np.array(actions)
            states = np.array(states)

        if mp4_path and mp4_path.exists():
            images = self.extract_video_frames(mp4_path, num_frames, max_size_mb=50)
            if len(images) != num_frames:
                print(f"  警告: 视频帧数 ({len(images)}) 与 mocap 帧数 ({num_frames}) 不匹配")
                min_frames = min(len(images), num_frames)
                actions = actions[:min_frames]
                states = states[:min_frames]
                images = images[:min_frames]
        elif not self.skip_missing_mp4:
            print(f"  警告: 未找到视频文件 {mp4_path}，跳过")
            return None
        else:
            images = None

        output_path = self.output_dir / hdf5_path.name
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('action', data=actions, compression='gzip', compression_opts=4)
            f.create_dataset('observation.state', data=states, compression='gzip', compression_opts=4)
            if images is not None:
                f.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
            f.attrs['x_negated'] = False
            f.attrs['source'] = str(hdf5_path)

        return output_path

    def extract_video_frames(self, mp4_path, target_frames, max_size_mb=50):
        """从 MP4 视频中提取帧，并压缩分辨率以控制文件大小
        
        Args:
            mp4_path: 视频文件路径
            target_frames: 目标帧数
            max_size_mb: 最大文件大小（MB），用于计算缩放比例
        """
        cap = cv2.VideoCapture(str(mp4_path))
        frames = []

        # 获取原始视频分辨率
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 计算目标分辨率（控制文件大小在 max_size_mb 以内）
        # 单帧大小 = width * height * 3 (RGB)
        # 总大小 = num_frames * width * height * 3
        target_bytes = max_size_mb * 1024 * 1024
        scale_factor = np.sqrt(target_bytes / (target_frames * orig_width * orig_height * 3))
        scale_factor = min(scale_factor, 1.0)  # 不放大
        
        target_width = max(160, int(orig_width * scale_factor))
        target_height = max(90, int(orig_height * scale_factor))
        
        print(f"  视频分辨率: {orig_width}x{orig_height} -> {target_width}x{target_height} (缩放: {scale_factor:.2f})")

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

        if len(frames) != target_frames:
            indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
            frames = [frames[i] for i in indices]

        return np.array(frames)

    def convert_batch(self, file_list=None):
        """批量转换"""
        if file_list is None:
            hdf5_files = sorted(self.mocap_dir.glob('*.hdf5'))
        else:
            hdf5_files = [Path(f) for f in file_list]

        hdf5_files = [f for f in hdf5_files if f.name != '0.hdf5']

        print(f"找到 {len(hdf5_files)} 个 HDF5 文件")

        results = {'success': [], 'skipped': [], 'failed': []}

        for hdf5_path in tqdm(hdf5_files, desc="转换进度"):
            mp4_path = hdf5_path.with_suffix('.mp4')

            try:
                output_path = self.convert_single(hdf5_path, mp4_path)
                if output_path:
                    results['success'].append(hdf5_path.name)
                    print(f"  ✅ 已保存到 {output_path}")
                else:
                    results['skipped'].append(hdf5_path.name)
            except Exception as e:
                results['failed'].append((hdf5_path.name, str(e)))
                print(f"  ❌ 转换失败 {hdf5_path.name}: {e}")

        print(f"\n转换完成:")
        print(f"  成功: {len(results['success'])}")
        print(f"  跳过: {len(results['skipped'])}")
        print(f"  失败: {len(results['failed'])}")

        return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description='将 mocap HDF5 转换为 128 维格式')
    parser.add_argument('--input', '-i', type=str,
                        default='/data1/zxlei/dataset/part2/basic_pick_place',
                        help='输入目录')
    parser.add_argument('--output', '-o', type=str,
                        default='/data1/zxlei/dataset/ego_converted',
                        help='输出目录')
    parser.add_argument('--num', '-n', type=int, default=2000,
                        help='最多转换的文件数量（不指定则转换全部）')
    parser.add_argument('--start', '-s', type=int, default=0,
                        help='从第几个文件开始（索引，从0开始）')
    parser.add_argument('--no-skip', action='store_true',
                        help='如果视频缺失则不跳过（报错）')
    parser.add_argument('--files', nargs='+', type=str,
                        help='指定要转换的文件列表（可选）')

    args = parser.parse_args()

    converter = MocapTo128Converter(
        mocap_dir=args.input,
        output_dir=args.output,
        skip_missing_mp4=not args.no_skip
    )

    if args.files:
        file_list = [Path(f) for f in args.files]
    else:
        all_files = sorted([f for f in Path(args.input).glob('*.hdf5') if f.name != '0.hdf5'])
        start_idx = args.start
        end_idx = args.start + args.num if args.num is not None else None
        file_list = all_files[start_idx:end_idx]

        print(f"文件范围: [{start_idx}, {end_idx if end_idx else len(all_files)}), 共 {len(file_list)} 个文件")

    converter.convert_batch(file_list)


if __name__ == '__main__':
    main()
