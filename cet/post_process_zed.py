import h5py
import cv2
import os
import argparse
import numpy as np
import tqdm

from pathlib import Path

from multiprocessing import Pool, cpu_count

import json
import hdt.constants

from cet.utils import index_dict, cmd_dict2policy
from cet.utils import parse_id

def load_mp4_stereo(path, 
                    original_res=(1280, 720),
                    offset_w=0,
                    resize_width=320, 
                    resize_height=240, 
                    sbs=True,
                    discard_frames=0):
    """
    offset_w: move the image after cropping
    crop is automatically done by aspect ratio
    """
    video_path = path + "_stereo.mp4"
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Failed to open video files")
        return None
    timestamps = load_vid_timestamps(path)

    nb_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    num_frames_processed = nb_frames - discard_frames

    original_width = original_res[0]
    original_height = original_res[1]
    original_aspect_ratio = original_width / original_height  
    target_aspect_ratio = resize_width / resize_height

    if target_aspect_ratio > original_aspect_ratio:
        new_height = int(original_width / target_aspect_ratio)
        crop_size_h = (original_height - new_height) // 2
        crop_size_w = 0
    else:
        new_width = int(original_height * target_aspect_ratio)
        crop_size_w = (original_width - new_width) // 2
        crop_size_h = 0
    
    cropped_img_shape = (original_height - 2 * crop_size_h, original_width - 2 * crop_size_w)

    final_width = resize_width if resize_width else cropped_img_shape[1]
    final_height = resize_height if resize_height else cropped_img_shape[0]

    left_imgs = np.zeros((num_frames_processed, 3, final_height, final_width), dtype=np.uint8)
    right_imgs = np.zeros((num_frames_processed, 3, final_height, final_width), dtype=np.uint8)

    cnt = 0

    while cap.isOpened():
        ret, frame = cap.read()
        
        if not ret:
            break
        if sbs:
            orig_width = frame.shape[1]
            left_frame = frame[:, :orig_width//2]
            right_frame = frame[:, orig_width//2:]
        else:
            left_frame = frame
            right_frame = frame

        # offset_w is 0, crop_size_h is 0, crop_size_w is 160, original_height is 720, original_width is 1280
        left_cropped = left_frame[crop_size_h:original_height-crop_size_h, crop_size_w+offset_w:original_width-crop_size_w+offset_w]
        right_cropped = right_frame[crop_size_h:original_height-crop_size_h, crop_size_w+offset_w:original_width-crop_size_w+offset_w]

        if resize_width and resize_height:
            left_resized = cv2.resize(left_cropped, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
            right_resized = cv2.resize(right_cropped, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
        else:
            left_resized = left_cropped
            right_resized = right_cropped
        
        # HWC -> CHW
        left_imgs[cnt] = cv2.cvtColor(left_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        right_imgs[cnt] = cv2.cvtColor(right_resized, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        cnt += 1

        if cnt >= num_frames_processed:
            break
        if cnt % 100 == 0:
            print(f"{cnt / num_frames_processed * 100:.2f}%")
    
    cap.release()

    delta = np.diff(timestamps)[:-1]
    print("Image timestamps delta mean: ", np.mean(delta))
    print("Image timestamps delta std: ", np.std(delta))

    return left_imgs, right_imgs, timestamps[:num_frames_processed]

def load_hdf5(path):
    '''
    /obs/timestamp: [],
    /obs/qpos: [],  # 14 +12  # BUG: should be 26 but actually 28, recorded 2 more waist joints 
    /obs/qvel: [],  # 14
    /action/joint_pos: [],  # 14arm + 12hand+ head2 = 28
    /action/cmd: [] #  head16 + wrist16*2 + hand6*3*2 = 70
    '''
    input_file = path + ".hdf5"
    file = h5py.File(input_file, 'r')
    print(f"Total hdf5_frames: {file['/obs/timestamp'].shape[0]}")
    timestamps = np.array(file["/obs/timestamp"][:] * 1000, dtype=np.int64)
    
    cmd_dict = {}   
    cmd_dict['head_mat'] = np.array(file["/action/cmd/head_mat"][:])
    cmd_dict['rel_left_wrist_mat'] = np.array(file["/action/cmd/rel_left_wrist_mat"][:])
    cmd_dict['rel_right_wrist_mat'] = np.array(file["/action/cmd/rel_right_wrist_mat"][:])
    cmd_dict['rel_left_hand_keypoints'] = np.array(file["/action/cmd/rel_left_hand_keypoints"][:])
    cmd_dict['rel_right_hand_keypoints'] = np.array(file["/action/cmd/rel_right_hand_keypoints"][:])

    qpos = None
    ik_qpos = None

    return timestamps, cmd_dict, qpos, ik_qpos

def load_vid_timestamps(path):
    timestamps = json.load(open(path + "_vid_timestamps.json", "r"))["video_frame_timestamps"]
    return (np.array(timestamps)*1000.0).astype(np.int64)  # ms 

def load_meta(path):
    meta_path = path + "_meta.json"
    with open(meta_path, 'r') as f:
        meta_data = json.load(f)
    return meta_data

def match_timestamps(candidate, ref, drop_threshold=200):
    """
    Match timestamps from candidate to ref.
    
    Args:
        candidate: numpy array of timestamps to be matched
        ref: numpy array of timestamps to be matched against
        drop_threshold: threshold in milliseconds for dropping frames. Default is 200ms.
    """
    closest_indices = []
    # candidate = np.sort(candidate)
    dropped_cnt = 0
    for t in ref:
        idx = np.searchsorted(candidate, t, side="left")
        # TODO: add paranthesis to make it more readable
        if idx > 0 and (idx == len(candidate) or np.fabs(t - candidate[idx-1]) < np.fabs(t - candidate[idx])):
            matched_idx = idx - 1
        else:
            matched_idx = idx
        matched_t = candidate[matched_idx]

        if np.fabs(t - matched_t) > drop_threshold:
            dropped_cnt += 1
        else:
            closest_indices.append(matched_idx)
    return np.array(closest_indices), dropped_cnt

def find_all_episodes(path, discard_drop=True):
    match_str = "_meta.json"
    episodes = [os.path.join(path, f) for f in os.listdir(path) if f.startswith('episode') and f.endswith(match_str)]
    episodes = [os.path.basename(ep).split(match_str)[0] for ep in episodes]
    if discard_drop:
        filtered_episodes = []
        for ep in episodes:
            ep_meta = json.load(open(path + "/" + ep + "_meta.json", "r"))
            if not ep_meta["description"] == "drop":
                filtered_episodes.append(ep)
        episodes = filtered_episodes
    return episodes

def save_compressed_imgs_hdf5(imgs, image_key, hf):
    ENCODE_PARAM = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
    compressed_len_list = []
    encoded_img_list = []
    num_imgs = imgs.shape[0]
    for i in range(num_imgs):
        img = imgs[i].transpose(1, 2, 0)
        _, img_encode = cv2.imencode('.jpg', img, ENCODE_PARAM)
        encoded_img_list.append(img_encode)
        compressed_len_list.append(len(img_encode))
                
    max_len = max(compressed_len_list)
    # create dataset
    hf.create_dataset(image_key, (num_imgs, max_len), dtype=np.uint8)
    for i in range(num_imgs):
        hf[image_key][i, :compressed_len_list[i]] = encoded_img_list[i].flatten()
    
    return encoded_img_list, compressed_len_list, max_len

def process_episode(file_name, ep, args, des=None):
    meta_info = load_meta(file_name)
    src = meta_info['embodiment']
    assert src == "human_zed"
    original_res = (1280, 720)
    ZED_FREQ = 30
    # Drop last 3 seconds since it's mostly thumbs-up
    discard_frames = 3 * ZED_FREQ
    left_imgs, right_imgs, img_timestamps = load_mp4_stereo(file_name, sbs=True, discard_frames=discard_frames)
        
    hdf5_timestamps, cmd_dict, qpos, ik_qpos = load_hdf5(file_name)

    # Downsample
    left_imgs = left_imgs[::args.downsample_rate]
    right_imgs = right_imgs[::args.downsample_rate]
    img_timestamps = img_timestamps[::args.downsample_rate]

    closest_indices, dropped_cnt = match_timestamps(candidate=hdf5_timestamps, ref=img_timestamps)

    if dropped_cnt > 0:
        raise ValueError(f"Dropped {dropped_cnt} frames")

    cmd_dict = index_dict(cmd_dict, closest_indices)

    if qpos is not None:
        qpos = qpos[closest_indices]
    if ik_qpos is not None:
        ik_qpos = ik_qpos[closest_indices]

    # For human data, cmd_dict2policy does not use qpos
    policy_action, policy_states = cmd_dict2policy(cmd_dict, qpos, src=src)
    
    # Mask head inputs
    policy_states[:, hdt.constants.OUTPUT_HEAD_EEF] = 0

    os.makedirs(des, exist_ok=True)

    save_path = os.path.join(des, f"processed_{ep}.hdf5")
    print("save_path: ", save_path)
    with h5py.File(save_path, 'w') as hf:
        encoded_list_left, _, _ = save_compressed_imgs_hdf5(left_imgs, 'observation.image.left', hf)
        encoded_list_right, _, _ = save_compressed_imgs_hdf5(right_imgs, 'observation.image.right', hf)

        assert len(encoded_list_left) == len(encoded_list_right) == left_imgs.shape[0]
        assert left_imgs.shape[0] == right_imgs.shape[0] == img_timestamps.shape[0]
        assert left_imgs.shape[0] == policy_action.shape[0] == policy_states.shape[0]

        hf.create_dataset('action', data=policy_action.astype(np.float32))
        hf.create_dataset('observation.state', data=policy_states.astype(np.float32))
        # hf.create_dataset('cmds', data=cmds.astype(np.float32))
        hf.attrs['sim'] = False
        # hf.attrs['init_action'] = cmds[0].astype(np.float32)
        hf.attrs['description'] = meta_info['description']
        hf.attrs['embodiment'] = meta_info['embodiment']

def find_all_processed_episodes(path):
    if not os.path.exists(path):
        return []
    episodes = [f for f in os.listdir(path)]
    return episodes

def process_episode_wrapper(args_list):
    file_name, ep, args, des = args_list
    try:
        process_episode(*args_list)
        print(f'Processed file {file_name}')
        return True
    except Exception as e:
        print(f'Failed to process {file_name}: {str(e)}')
        return False

def main(args, path, task_name):
    all_eps = find_all_episodes(path)
    all_eps.sort()
    print("all_eps: ", all_eps)

    des = (Path(path) / "../processed" / task_name).resolve()

    if args.multiprocess:
        # Prepare arguments for parallel processing
        process_args = [(path + "/" + ep, ep, args, des) for ep in all_eps]
        # Use half of available CPU cores to avoid overwhelming the system
        num_processes = min(8, cpu_count() // 2)
        # Create process pool and process episodes in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(process_episode_wrapper, process_args)
        successful = sum(results)
    else:
        successful = 0
        for i, ep in tqdm.tqdm(enumerate(all_eps)):
            process_episode(path + "/" + ep, ep, args, des)
            successful += 1


    # Test the processed data
    episodes = find_all_processed_episodes(des)
    lens = []

    for episode in episodes:
        episode_path = des / episode
        data = h5py.File(str(episode_path), 'r')
        lens.append(data['action'].shape[0])
        data.close()

    lens = np.array(lens)
    episodes = np.array(episodes)
    
    return successful, len(all_eps)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--taskid', type=str, required=True)
    parser.add_argument('--multiprocess', action='store_true', default=False)
    parser.add_argument('--downsample_rate', type=int, default=1)
    args = parser.parse_args()

    succ_cnt_list = []
    all_cnt_list = []
    
    if os.path.exists(args.taskid) and args.taskid.endswith(".json"):
        with open(args.taskid, 'r') as f:
            dataset_config = json.load(f)
        for split in dataset_config:
            assert split in ['train', 'val'], "Only train and val splits now supported"
            task_names = dataset_config[split]
            for task_name in task_names:
                path, task_name = parse_id("../data/recordings", task_name)
                succ, all_cnt = main(args, path, task_name)
                succ_cnt_list.append(succ)
                all_cnt_list.append(all_cnt)
    else:
        path, task_name = parse_id("../data/recordings", args.taskid)
        succ, all_cnt = main(args, path, task_name)
        succ_cnt_list.append(succ)
        all_cnt_list.append(all_cnt)

    print("================================================")
    print(f"Successfully processed {sum(succ_cnt_list)}/{sum(all_cnt_list)} episodes")
    print("================================================")
