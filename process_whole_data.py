import h5py
import numpy as np
import os

def process_hdf5_file(input_path, output_path):
    """处理单个HDF5文件"""
    with h5py.File(input_path, 'r') as f:
        # 读取数据
        state_data = f['/observation.state'][:]
        action_data = f['/action'][:]
        # 检查图像数据集是否存在
        image_data = f['/observation.image.top'][:] if '/observation.image.top' in f else None
    
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
        
        # 对换左右手（vision pro采集的数据需要对换）
        # 注意：这里是交换变量值，确保后续处理使用对换后的数据
        temp_wrist = right_wrist.copy()
        temp_fingers = right_fingers.copy()
        right_wrist = left_wrist.copy()
        right_fingers = left_fingers.copy()
        left_wrist = temp_wrist
        left_fingers = temp_fingers
        
        # 其他数据
        other_data = state_data[i, 45:]
        
        # 头部和手腕减去头部位置，变换到相对头部的坐标系
        head_rel = head - head  # 头部相对于自己为0
        right_wrist_rel = right_wrist - head
        left_wrist_rel = left_wrist - head
        
        # 调整坐标轴顺序：
        # 原始: [x, y, z]
        # 新: [x, z, y]，其中：
        # 新x = 原始x（身体朝向，前后方向）
        # 新y = 原始z（左右方向，左手在y正方向，右手在y负方向）
        # 新z = 原始y（高度，确保双手在头部下方）
        def adjust_axes(coord):
            return [coord[0], coord[2], coord[1]]
        
        # 调整头部、手腕的坐标轴
        head_rel = adjust_axes(head_rel)
        right_wrist_rel = adjust_axes(right_wrist_rel)
        left_wrist_rel = adjust_axes(left_wrist_rel)
        
        # 调整手指的坐标轴
        # 手指是相对手腕的坐标，所以也需要调整坐标轴
        adjusted_right_fingers = []
        for j in range(0, len(right_fingers), 3):
            finger = right_fingers[j:j+3]
            adjusted_finger = adjust_axes(finger)
            adjusted_right_fingers.extend(adjusted_finger)
        
        adjusted_left_fingers = []
        for j in range(0, len(left_fingers), 3):
            finger = left_fingers[j:j+3]
            adjusted_finger = adjust_axes(finger)
            adjusted_left_fingers.extend(adjusted_finger)
        
        # 确保左手在y正方向，右手在y负方向
        # 如果左手y坐标小于右手y坐标，交换它们的位置
        if left_wrist_rel[1] < right_wrist_rel[1]:
            temp_wrist = right_wrist_rel.copy()
            temp_fingers = adjusted_right_fingers.copy()
            right_wrist_rel = left_wrist_rel.copy()
            adjusted_right_fingers = adjusted_left_fingers.copy()
            left_wrist_rel = temp_wrist
            adjusted_left_fingers = temp_fingers
        
        # 重新组合数据
        processed_state[i, :3] = head_rel
        processed_state[i, 3:6] = right_wrist_rel
        processed_state[i, 6:9] = left_wrist_rel
        processed_state[i, 9:27] = adjusted_right_fingers
        processed_state[i, 27:45] = adjusted_left_fingers
        processed_state[i, 45:] = other_data
        
        # 处理action数据
        # 提取头部位置和旋转
        head_pos = action_data[i, 0:3]
        head_rot = action_data[i, 3:9]
        
        # 提取右手腕位置和旋转
        right_wrist_pos = action_data[i, 30:33].copy()
        right_wrist_rot = action_data[i, 33:39].copy()
        
        # 提取左手腕位置和旋转
        left_wrist_pos = action_data[i, 80:83].copy()
        left_wrist_rot = action_data[i, 83:89].copy()
        
        # 提取手指坐标
        right_fingers = action_data[i, 40:58].copy()  # 右手手指：40-57
        left_fingers = action_data[i, 10:28].copy()   # 左手手指：10-27
        
        # 对换左右手（vision pro采集的数据需要对换）
        temp_wrist = right_wrist_pos.copy()
        temp_rot = right_wrist_rot.copy()
        temp_fingers = right_fingers.copy()
        right_wrist_pos = left_wrist_pos.copy()
        right_wrist_rot = left_wrist_rot.copy()
        right_fingers = left_fingers.copy()
        left_wrist_pos = temp_wrist
        left_wrist_rot = temp_rot
        left_fingers = temp_fingers
        
        # 头部和手腕减去头部位置，变换到相对头部的坐标系
        head_pos_rel = head_pos - head_pos  # 头部相对于自己为0
        right_wrist_pos_rel = right_wrist_pos - head_pos
        left_wrist_pos_rel = left_wrist_pos - head_pos
        
        # 调整坐标轴顺序：
        # 原始: [x, y, z]
        # 新: [x, z, y]，其中：
        # 新x = 原始x（身体朝向，前后方向）
        # 新y = 原始z（左右方向，左手在y正方向，右手在y负方向）
        # 新z = 原始y（高度，确保双手在头部下方）
        head_pos_rel = adjust_axes(head_pos_rel)
        right_wrist_pos_rel = adjust_axes(right_wrist_pos_rel)
        left_wrist_pos_rel = adjust_axes(left_wrist_pos_rel)
        
        # 调整手指的坐标轴
        adjusted_right_fingers = []
        for j in range(0, len(right_fingers), 3):
            finger = right_fingers[j:j+3]
            adjusted_finger = adjust_axes(finger)
            adjusted_right_fingers.extend(adjusted_finger)
        
        adjusted_left_fingers = []
        for j in range(0, len(left_fingers), 3):
            finger = left_fingers[j:j+3]
            adjusted_finger = adjust_axes(finger)
            adjusted_left_fingers.extend(adjusted_finger)
        
        # 确保左手在y正方向，右手在y负方向
        # 如果左手y坐标小于右手y坐标，交换它们的位置
        if left_wrist_pos_rel[1] < right_wrist_pos_rel[1]:
            temp_wrist = right_wrist_pos_rel.copy()
            temp_rot = right_wrist_rot.copy()
            temp_fingers = adjusted_right_fingers.copy()
            right_wrist_pos_rel = left_wrist_pos_rel.copy()
            right_wrist_rot = left_wrist_rot.copy()
            adjusted_right_fingers = adjusted_left_fingers.copy()
            left_wrist_pos_rel = temp_wrist
            left_wrist_rot = temp_rot
            adjusted_left_fingers = temp_fingers
        
        # 重新组合数据
        processed_action[i, 0:3] = head_pos_rel
        processed_action[i, 3:9] = head_rot
        processed_action[i, 10:28] = adjusted_left_fingers   # 左手手指
        processed_action[i, 30:33] = right_wrist_pos_rel
        processed_action[i, 33:39] = right_wrist_rot
        processed_action[i, 40:58] = adjusted_right_fingers  # 右手手指
        processed_action[i, 80:83] = left_wrist_pos_rel
        processed_action[i, 83:89] = left_wrist_rot
    
    # 保存处理后的数据
    with h5py.File(output_path, 'w') as f:
        f.create_dataset('/observation.state', data=processed_state)
        f.create_dataset('/action', data=processed_action)
        # 只保存存在的图像数据集
        if image_data is not None:
            f.create_dataset('/observation.image.top', data=image_data)

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
    input_dir = '/data1/zxlei/dataset/convert_whole/task2'
    output_dir = './processed_whole'
    
    process_all_files(input_dir, output_dir)
    print("All files processed successfully!")