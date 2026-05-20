import h5py
import numpy as np
import os

# 导入常量定义
import hdt.constants

def process_hdf5_file(input_path, output_path):
    """处理单个HDF5文件"""
    with h5py.File(input_path, 'r') as f:
        # 读取数据
        state_data = f['/observation.state'][:]
        action_data = f['/action'][:]
        # 检查图像数据集是否存在
        left_image_data = f['/observation.image.left'][:] if '/observation.image.left' in f else None
        right_image_data = f['/observation.image.right'][:] if '/observation.image.right' in f else None
    
    # 处理状态数据
    processed_state = np.copy(state_data)
    # 处理action数据
    processed_action = np.copy(action_data)
    
    for i in range(state_data.shape[0]):
        # 处理状态数据
        # 提取头部坐标
        head = state_data[i, :3]
        
        # 提取右手腕和左手腕坐标
        right_wrist = state_data[i, 3:6]
        left_wrist = state_data[i, 6:9]
        
        # 提取右手手指和左手手指坐标
        right_fingers = state_data[i, 9:27]
        left_fingers = state_data[i, 27:45]
        
        # 其他数据
        other_data = state_data[i, 45:]
        
        # 头部和手腕减去头部位置，变换到相对头部的坐标系
        head_rel = head - head  # 头部相对于自己为0
        right_wrist_rel = right_wrist - head
        left_wrist_rel = left_wrist - head
        
        # 调整坐标轴顺序：x, y, z -> x, z, y（将Y轴作为高度，对应到Z轴）
        # 原始: [x, y, z]
        # 新: [x, z, y]，其中y是高度
        def adjust_axes(coord):
            return [coord[0], coord[2], coord[1]]
        
        # 调整头部、手腕和手指的坐标轴
        head_rel = adjust_axes(head_rel)
        right_wrist_rel = adjust_axes(right_wrist_rel)
        left_wrist_rel = adjust_axes(left_wrist_rel)
        
        # 对换左右手的位置
        # 新的右手腕 = 原来的左手腕
        # 新的左手腕 = 原来的右手腕
        # 新的右手手指 = 原来的左手手指
        # 新的左手手指 = 原来的右手手指
        new_right_wrist = left_wrist_rel
        new_left_wrist = right_wrist_rel
        new_right_fingers = left_fingers
        new_left_fingers = right_fingers
        
        # 重新组合数据
        processed_state[i, :3] = head_rel
        processed_state[i, 3:6] = new_right_wrist
        processed_state[i, 6:9] = new_left_wrist
        processed_state[i, 9:27] = new_right_fingers
        processed_state[i, 27:45] = new_left_fingers
        processed_state[i, 45:] = other_data
        
        # 处理action数据
        # 提取头部位置和旋转
        head_pos = action_data[i, hdt.constants.OUTPUT_HEAD_EEF[0:3]]
        head_rot = action_data[i, hdt.constants.OUTPUT_HEAD_EEF[3:]]
        
        # 提取右手腕位置和旋转
        right_wrist_pos = action_data[i, hdt.constants.OUTPUT_RIGHT_EEF[0:3]]
        right_wrist_rot = action_data[i, hdt.constants.OUTPUT_RIGHT_EEF[3:]]
        
        # 提取左手腕位置和旋转
        left_wrist_pos = action_data[i, hdt.constants.OUTPUT_LEFT_EEF[0:3]]
        left_wrist_rot = action_data[i, hdt.constants.OUTPUT_LEFT_EEF[3:]]
        
        # 提取右手手指和左手手指坐标
        right_fingers = action_data[i, hdt.constants.OUTPUT_RIGHT_KEYPOINTS]
        left_fingers = action_data[i, hdt.constants.OUTPUT_LEFT_KEYPOINTS]
        
        # 其他数据
        other_action_data = np.concatenate([
            action_data[i, 9:10],  # 头部和左手手指之间的数据
            action_data[i, 27:30],  # 左手手指和右手腕之间的数据
            action_data[i, 57:80],  # 右手手指和左手腕之间的数据
            action_data[i, 89:]     # 左手腕之后的数据
        ])
        
        # 头部和手腕减去头部位置，变换到相对头部的坐标系
        head_pos_rel = head_pos - head_pos  # 头部相对于自己为0
        right_wrist_pos_rel = right_wrist_pos - head_pos
        left_wrist_pos_rel = left_wrist_pos - head_pos
        
        # 对换左右手的位置
        # 新的右手腕 = 原来的左手腕
        # 新的左手腕 = 原来的右手腕
        # 新的右手手指 = 原来的左手手指
        # 新的左手手指 = 原来的右手手指
        new_right_wrist_pos = left_wrist_pos_rel
        new_right_wrist_rot = left_wrist_rot
        new_left_wrist_pos = right_wrist_pos_rel
        new_left_wrist_rot = right_wrist_rot
        new_right_fingers = left_fingers
        new_left_fingers = right_fingers
        
        # 重新组合数据
        processed_action[i, hdt.constants.OUTPUT_HEAD_EEF[0:3]] = head_pos_rel
        processed_action[i, hdt.constants.OUTPUT_HEAD_EEF[3:]] = head_rot
        processed_action[i, hdt.constants.OUTPUT_RIGHT_EEF[0:3]] = new_right_wrist_pos
        processed_action[i, hdt.constants.OUTPUT_RIGHT_EEF[3:]] = new_right_wrist_rot
        processed_action[i, hdt.constants.OUTPUT_LEFT_EEF[0:3]] = new_left_wrist_pos
        processed_action[i, hdt.constants.OUTPUT_LEFT_EEF[3:]] = new_left_wrist_rot
        processed_action[i, hdt.constants.OUTPUT_RIGHT_KEYPOINTS] = new_right_fingers
        processed_action[i, hdt.constants.OUTPUT_LEFT_KEYPOINTS] = new_left_fingers
    
    # 保存处理后的数据
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('/observation.state', data=processed_state)
        f.create_dataset('/action', data=processed_action)
        # 只保存存在的图像数据集
        if left_image_data is not None:
            f.create_dataset('/observation.image.left', data=left_image_data)
        if right_image_data is not None:
            f.create_dataset('/observation.image.right', data=right_image_data)

def process_all_files(input_dir, output_dir):
    """处理目录下所有HDF5文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 遍历输入目录下的所有HDF5文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.hdf5'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            print(f"Processing {filename}...")
            process_hdf5_file(input_path, output_path)
            print(f"Saved to {output_path}")

if __name__ == "__main__":
    input_dir = '/data1/zxlei/dataset/part2/convert2'
    output_dir = './processed'
    
    process_all_files(input_dir, output_dir)
    print("All files processed successfully!")