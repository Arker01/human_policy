import h5py
import numpy as np
import os
from pathlib import Path
import cv2

def extract_frames_from_mp4(mp4_path, target_size=(240, 320)):
    """从MP4文件中提取帧"""
    cap = cv2.VideoCapture(mp4_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (target_size[1], target_size[0]))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return np.array(frames)

def align_frames_to_actions(video_frames, action_frames_count):
    """将视频帧对齐到动作帧数"""
    video_count = len(video_frames)

    if video_count == action_frames_count:
        return video_frames

    if video_count > 1000:
        keep_count = min(action_frames_count + 100, video_count)
        video_frames = video_frames[-keep_count:]
        video_count = keep_count
        print(f'  异常视频，已截取后{keep_count}帧')

    if abs(video_count - action_frames_count) <= 10:
        return video_frames

    if video_count > action_frames_count:
        indices = np.linspace(0, video_count - 1, action_frames_count, dtype=int)
        return video_frames[indices]

    return video_frames

def copy_group(src_group, dst_group):
    """复制HDF5组结构"""
    for key, item in src_group.items():
        if isinstance(item, h5py.Dataset):
            dst_group.create_dataset(key, data=item[:], compression='gzip', compression_opts=4)
        elif isinstance(item, h5py.Group):
            new_group = dst_group.create_group(key)
            copy_group(item, new_group)

    for attr_key, attr_value in src_group.attrs.items():
        dst_group.attrs[attr_key] = attr_value

def process_episode(episode_idx, mp4_dir, input_hdf5_path, output_hdf5_path):
    """处理单个episode"""
    mp4_path = mp4_dir / f' wb-{episode_idx}.MP4'

    if not mp4_path.exists():
        print(f'警告: MP4文件不存在 {mp4_path}')
        return False

    print(f'处理 episode_{episode_idx}: {mp4_path}')

    with h5py.File(input_hdf5_path, 'r') as f_in:
        action_frames = len(f_in['action'][:])
        print(f'  动作帧数: {action_frames}')

    video_frames = extract_frames_from_mp4(str(mp4_path))
    print(f'  原始视频帧数: {len(video_frames)}')

    aligned_frames = align_frames_to_actions(video_frames, action_frames)
    print(f'  对齐后帧数: {len(aligned_frames)}')

    with h5py.File(input_hdf5_path, 'r') as f_in:
        os.makedirs(os.path.dirname(output_hdf5_path) or '.', exist_ok=True)
        with h5py.File(output_hdf5_path, 'w') as f_out:
            copy_group(f_in, f_out)

            if 'observation.image.top' in f_out:
                del f_out['observation.image.top']

            f_out.create_dataset('observation.image.top', data=aligned_frames, compression='gzip', compression_opts=4)

    return True

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Extract frames from MP4 and align to action frames')
    parser.add_argument('mp4_dir', help='Directory containing MP4 files')
    parser.add_argument('input_dir', help='Input directory containing HDF5 files')
    parser.add_argument('output_dir', help='Output directory for processed HDF5 files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to process')
    args = parser.parse_args()

    mp4_dir = Path(args.mp4_dir)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(input_dir.glob('episode_*.hdf5'))

    if args.limit:
        hdf5_files = hdf5_files[:args.limit]

    print(f'找到 {len(hdf5_files)} HDF5 文件')

    success_count = 0
    fail_count = 0

    for f in hdf5_files:
        episode_idx = int(f.stem.split('_')[1])

        input_path = input_dir / f.name
        output_path = output_dir / f.name

        if process_episode(episode_idx, mp4_dir, str(input_path), str(output_path)):
            success_count += 1
        else:
            fail_count += 1

    print(f'\n处理完成: 成功 {success_count} 个, 失败 {fail_count} 个')

if __name__ == '__main__':
    main()