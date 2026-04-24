import numpy as np
import torch
import json
import torchvision
import os
import h5py
from torch.utils.data import DataLoader
import cv2
import pickle

import sys
import random
import torchvision.transforms.v2

from hdt.constants import *

import hdt.inference_utils

class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 data_config,
                 hdf_path_list, 
                 lang_embeds_paths, 
                 camera_names, 
                 chunk_size, 
                 episode_len_list, 
                 task_episode_cnt,
                 visual_preprocessor, 
                 cond_mask_prob,
                 control_mode="ee", 
                 train=True,
                 slow_down_factor=4):
        super(EpisodicDataset).__init__()
        self.camera_names = camera_names
        self.train = train
        self.data_config = data_config
        self.episode_len_list = episode_len_list
        self.dataset_paths = hdf_path_list
        self.task_episode_cnt = task_episode_cnt
        self.visual_preprocessor = visual_preprocessor
        self.chunk_size = chunk_size  # action length (e.g., chunk size)
        self.action_str = 'old_action' if control_mode == 'qpos' else 'action'

        self.predict_delta_action = False
        self.augment_action_space = False
        # Flag for simplifying visual learning when data is not that much
        self.SIMPLIFY_VISUAL = True
        # h5py is pretty efficient. This works only marginally for machines with fast disks (~5% improvement)
        # for NFS-based storage, this is a huge improvement
        self.load_hdf_to_cpu = False

        self.cond_mask_prob = cond_mask_prob
        self.cumulative_len = np.cumsum(self.episode_len_list)

        self.sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in self.episode_len_list])
        
        self.slow_down_factor = slow_down_factor

        # Load everything to CPU memory
        # infer language paths
        self.cached_lang_embedding_dict = {}
        for lang_embedding_path in lang_embeds_paths:
            if lang_embedding_path is None:
                continue
            with open(lang_embedding_path, 'rb') as f:
                cur_lang_embedding_dict = pickle.load(f)
                self.cached_lang_embedding_dict.update(cur_lang_embedding_dict)

        if self.load_hdf_to_cpu:
            self.cached_hdf_dict = {}
            for single_hdf_path in self.dataset_paths:
                if single_hdf_path not in self.cached_hdf_dict:
                    self.cached_hdf_dict[single_hdf_path] = {}
                    with h5py.File(single_hdf_path, 'r') as root:
                        self.cached_hdf_dict[single_hdf_path]['observation.state'] = root['observation.state'][()]
                        for cam_name in self.camera_names:
                            # If cam_name doesn't exist, try fallback cameras
                            actual_cam_name = cam_name
                            if f'observation.image.{cam_name}' not in root:
                                if 'observation.image.left' in root:
                                    actual_cam_name = 'left'
                                elif 'observation.image.right' in root:
                                    actual_cam_name = 'right'
                                else:
                                    raise KeyError(f"None of the expected cameras (top, left, right) found in {single_hdf_path}")
                            self.cached_hdf_dict[single_hdf_path][f'observation.image.{cam_name}'] = root[f'observation.image.{actual_cam_name}'][()]
                        self.cached_hdf_dict[single_hdf_path][self.action_str] =  root[self.action_str][()]
                        self.cached_hdf_dict[single_hdf_path]['attrs'] = {k: v for k, v in root.attrs.items()}
        
        self.training_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # torchvision.transforms.v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
        ])

        self.norm_stats, self.embodiment_list = self.get_norm_stats()

        SAMPLER_TYPE = 'norm_by_embodiment_and_task'
        self.episode_sampling_prob = self.get_episode_sampling_prob(SAMPLER_TYPE)

        # Load empty language embedding from the correct path
        import os
        empty_lang_embed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty_lang_embed.pt")
        empty_lang_embedding = torch.load(empty_lang_embed_path, weights_only=True).float()
        self.cached_lang_embedding_dict[''] = empty_lang_embedding
    
    def get_episode_sampling_prob(self, sampler_type):
        if sampler_type == 'uniform':
            return None  # when it is None, the default behavior is uniform sampling
        elif sampler_type == 'norm_by_embodiment':
            P_ROBOT = 0.5
            P_HUMAN = 1 - P_ROBOT
            human_idx_list = []
            robot_idx_list = []
            # Directly iterate through all episodes to collect human and robot indices
            for idx, embodiment in enumerate(self.embodiment_list):
                if "human" in embodiment:
                    human_idx_list.append(idx)
                else:
                    robot_idx_list.append(idx)
            assert len(self.embodiment_list) == len(self.episode_len_list)
            prob_arr = np.ones(len(self.episode_len_list)) / len(self.episode_len_list)
            if human_idx_list:
                prob_arr[human_idx_list] = P_HUMAN / len(human_idx_list)
            if robot_idx_list:
                prob_arr[robot_idx_list] = P_ROBOT / len(robot_idx_list)
            if not np.isclose(np.sum(prob_arr), 1):
                print("=========")
                print(f"Warning: sum of prob_arr is not 1: {np.sum(prob_arr)}. Is only one embodiment available?")
                prob_arr = prob_arr / np.sum(prob_arr)
            return prob_arr
        elif sampler_type == 'norm_by_embodiment_and_task':
            P_ROBOT = 0.5
            P_HUMAN = 1 - P_ROBOT
            human_idx_list = []
            robot_idx_list = []
            # Directly iterate through all episodes to collect human and robot indices
            for idx, embodiment in enumerate(self.embodiment_list):
                if "human" in embodiment:
                    human_idx_list.append(idx)
                else:
                    robot_idx_list.append(idx)
            assert len(self.embodiment_list) == len(self.episode_len_list)
            prob_arr = np.ones(len(self.episode_len_list)) / len(self.episode_len_list)
            if human_idx_list:
                prob_arr[human_idx_list] = P_HUMAN / len(human_idx_list)
            if robot_idx_list:
                prob_arr[robot_idx_list] = P_ROBOT / len(robot_idx_list)
            if not np.isclose(np.sum(prob_arr), 1):
                print("=========")
                print(f"Warning: sum of prob_arr is not 1: {np.sum(prob_arr)}. Is only one embodiment available?")
                prob_arr = prob_arr / np.sum(prob_arr)
            return prob_arr
        else:
            raise ValueError(f"Unknown sampler type: {sampler_type}")
    
    def get_norm_stats(self):
        if hasattr(self, 'norm_stats'):
            return self.norm_stats
        
        norm_stats_dict = {}
        embodiment_list = []
        # First, gather types of embodiments
        for single_hdf_path in self.dataset_paths:
            with h5py.File(single_hdf_path, 'r') as root:
                # Use default embodiment if not present
                embodiment = root.attrs.get('embodiment', 'default')
                if embodiment not in norm_stats_dict:
                    norm_stats_dict[embodiment] = {
                        "actions": [],
                        "states": []
                    }
                embodiment_list.append(embodiment)
        print(f"Found embodiments: {norm_stats_dict.keys()}")

        for hdf_path in self.dataset_paths:
            with h5py.File(hdf_path, 'r') as root:
                state = root['observation.state'][()]
                action = root['action'][()]
                # Use default embodiment if not present
                embodiment = root.attrs.get('embodiment', 'default')
                norm_stats_dict[embodiment]["actions"].append(torch.from_numpy(action))
                norm_stats_dict[embodiment]["states"].append(torch.from_numpy(state))
        
        SAME_NORMALIZATION = False
        if SAME_NORMALIZATION:
            all_actions = []
            all_states = []
            for emb in norm_stats_dict:
                all_actions.append(torch.cat(norm_stats_dict[emb]['actions']))
                all_states.append(torch.cat(norm_stats_dict[emb]['states']))
            
            all_actions = torch.cat(all_actions)
            all_states = torch.cat(all_states)

            # normalize action data
            action_mean = all_actions.mean(dim=0, keepdim=True).numpy().squeeze()
            action_std = all_actions.std(dim=0, keepdim=True)
            action_std = torch.clip(action_std, 1e-2, np.inf).numpy().squeeze()
            if self.predict_delta_action:
                action_mean = np.zeros_like(action_mean)

            # normalize qpos data
            qpos_mean = all_states.mean(dim=0, keepdim=True).numpy().squeeze()
            qpos_std = all_states.std(dim=0, keepdim=True)
            qpos_std = torch.clip(qpos_std, 1e-2, np.inf).numpy().squeeze()

            for emb in norm_stats_dict:
                norm_stats_dict[emb]['action_mean'] = action_mean
                norm_stats_dict[emb]['action_std'] = action_std

                norm_stats_dict[emb]['qpos_mean'] = qpos_mean
                norm_stats_dict[emb]['qpos_std'] = qpos_std

                del norm_stats_dict[emb]['actions']
                del norm_stats_dict[emb]['states']
            
        else:
            for emb in norm_stats_dict:
                norm_stats_dict[emb]['actions'] = torch.cat(norm_stats_dict[emb]['actions'])
                norm_stats_dict[emb]['states'] = torch.cat(norm_stats_dict[emb]['states'])

                # normalize action data
                norm_stats_dict[emb]['action_mean'] = norm_stats_dict[emb]['actions'].mean(dim=0, keepdim=True).numpy().squeeze()
                norm_stats_dict[emb]['action_std'] = norm_stats_dict[emb]['actions'].std(dim=0, keepdim=True)
                norm_stats_dict[emb]['action_std'] = torch.clip(norm_stats_dict[emb]['action_std'], 1e-2, np.inf).numpy().squeeze()
                if self.predict_delta_action:
                    norm_stats_dict[emb]['action_mean'] = np.zeros_like(norm_stats_dict[emb]['action_mean'])

                # normalize qpos data
                norm_stats_dict[emb]['qpos_mean'] = norm_stats_dict[emb]['states'].mean(dim=0, keepdim=True).numpy().squeeze()
                norm_stats_dict[emb]['qpos_std'] = norm_stats_dict[emb]['states'].std(dim=0, keepdim=True)
                norm_stats_dict[emb]['qpos_std'] = torch.clip(norm_stats_dict[emb]['qpos_std'], 1e-2, np.inf).numpy().squeeze()

                del norm_stats_dict[emb]['actions']
                del norm_stats_dict[emb]['states']

        return norm_stats_dict, embodiment_list
    
    def __len__(self):
        return np.iinfo(int).max

    def _locate_transition(self, index):
        assert index < self.cumulative_len[-1]
        episode_index = np.argmax(self.cumulative_len > index) # argmax returns first True index
        start_ts = index - (self.cumulative_len[episode_index] - self.episode_len_list[episode_index])
        return episode_index, start_ts

    def read_one(self, index, start_ts):
        episode_len = self.episode_len_list[index]

        start_ts = np.random.choice(episode_len)

        single_hdf_path = self.dataset_paths[index]

        if self.load_hdf_to_cpu:
            root = self.cached_hdf_dict[single_hdf_path]
        else:
            root = h5py.File(single_hdf_path, 'r')

        qpos = root['observation.state'][start_ts]

        image_dict = dict()
        for cam_name in self.camera_names:
            if self.SIMPLIFY_VISUAL or random.random() > self.cond_mask_prob:
                # If cam_name doesn't exist, try fallback cameras
                actual_cam_name = cam_name
                if f'observation.image.{cam_name}' not in root:
                    # Try 'left' first, then 'right'
                    if 'observation.image.left' in root:
                        actual_cam_name = 'left'
                    elif 'observation.image.right' in root:
                        actual_cam_name = 'right'
                    else:
                        raise KeyError(f"None of the expected cameras (top, left, right) found in {single_hdf_path}")
                
                image_dict[cam_name] = root[f'observation.image.{actual_cam_name}'][start_ts]
                if len(image_dict[cam_name].shape) == 1:
                    # Compressed JPEG format images are represented as (N,) uint8 array. N is different for every image.
                    image_dict[cam_name] = cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)
                    image_dict[cam_name] = cv2.resize(image_dict[cam_name], (self.data_config["image_resolution_hw"][1], self.data_config["image_resolution_hw"][0]))
                    assert image_dict[cam_name].shape == (self.data_config["image_resolution_hw"][0], self.data_config["image_resolution_hw"][1], 3)
                    # Images are in RGB (verified by plt.imshow) and HWC in this case
                    image_dict[cam_name] = image_dict[cam_name].transpose(2, 0, 1)  # CHW
                elif len(image_dict[cam_name].shape) == 3:
                    # Raw HWC images need to be resized and transposed to CHW
                    image_dict[cam_name] = cv2.resize(image_dict[cam_name], (self.data_config["image_resolution_hw"][1], self.data_config["image_resolution_hw"][0]))
                    image_dict[cam_name] = image_dict[cam_name].transpose(2, 0, 1)  # CHW
            else:
                image_dict[cam_name] = np.zeros((3, self.data_config["image_resolution_hw"][0], self.data_config["image_resolution_hw"][1]), dtype=np.uint8)
        all_time_action = root[self.action_str][start_ts:start_ts+self.chunk_size]

        if not self.load_hdf_to_cpu:
            lang_instruction = root.attrs['description']
            embodiment = root.attrs['embodiment']
            root.close()
        else:
            lang_instruction = root['attrs']['description']
            embodiment = root['attrs']['embodiment']
        
        if "human" in embodiment:
            SLOW_DOWN_FACTOR = self.slow_down_factor
            all_time_action = hdt.inference_utils.interpolate_128dim_action(all_time_action, all_time_action.shape[0] * SLOW_DOWN_FACTOR)
            all_time_action = all_time_action[:self.chunk_size]
        
        padded_action = np.zeros((self.chunk_size, all_time_action.shape[1]), dtype=np.float32)
        padded_action[:all_time_action.shape[0], :] = all_time_action
        
        real_len = episode_len - start_ts

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[real_len:] = True

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        image_data = self.visual_preprocessor(all_cam_images)
        if self.train:
            image_data = self.training_transforms(image_data)

        qpos_data = torch.from_numpy(qpos).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad = torch.from_numpy(is_pad).bool()

        if self.augment_action_space:
            # augment hand EEF positions
            AUG_SCALE = 0.5
            action_data[:, OUTPUT_LEFT_EEF[0:3]] = action_data[:, OUTPUT_LEFT_EEF[0:3]] + torch.rand(3) * AUG_SCALE * self.norm_stats[embodiment]["action_std"][OUTPUT_LEFT_EEF[0:3]]
            action_data[:, OUTPUT_RIGHT_EEF[0:3]] = action_data[:, OUTPUT_RIGHT_EEF[0:3]] + torch.rand(3) * AUG_SCALE * self.norm_stats[embodiment]["action_std"][OUTPUT_RIGHT_EEF[0:3]]
        
        if self.predict_delta_action:
            action_data = action_data - action_data[0]

        action_data = (action_data - self.norm_stats[embodiment]["action_mean"]) / (self.norm_stats[embodiment]["action_std"] + 1e-6)
        qpos_data = (qpos_data - self.norm_stats[embodiment]["qpos_mean"]) / (self.norm_stats[embodiment]["qpos_std"] + 1e-6) \
            if random.random() > self.cond_mask_prob else torch.zeros_like(qpos_data)
        
        if random.random() < self.cond_mask_prob:
            lang_instruction = ''
        
        if lang_instruction in self.cached_lang_embedding_dict:
            selected_embedding = self.cached_lang_embedding_dict[lang_instruction]
            if selected_embedding is not None:
                selected_embedding = selected_embedding.float()
        else:
            selected_embedding = None

        conditioning_dict = {
            "language_embeddings": selected_embedding,
            "plain_text": lang_instruction
        }

        return image_data, qpos_data, action_data, is_pad, conditioning_dict
    
    def __getitem__(self, _idx):
        max_retries = 10
        for _ in range(max_retries):
            episode_idx = np.random.choice(len(self.episode_len_list), p=self.episode_sampling_prob)
            ts_index = np.random.randint(self.sum_dataset_len_l[episode_idx], self.sum_dataset_len_l[episode_idx + 1])

            index, start_ts = self._locate_transition(ts_index)
            try:
                return self.read_one(index, start_ts)
            except (KeyError, OSError) as e:
                error_msg = str(e)
                if 'observation.image' in error_msg or 'None of the expected cameras' in error_msg:
                    continue
                raise
        raise KeyError(f"Failed to get valid sample after {max_retries} retries")

def gather_hdf_paths(base_dir, task_names, camera_names=None, extra_base_dirs=None):
    all_hdf_paths = []
    task_episode_cnt = []
    for task_item in task_names:
        found = False
        # 处理任务项，支持字符串或字典格式
        if isinstance(task_item, dict):
            task_name = task_item.get('dataset_path')
            start_idx = task_item.get('start_idx', 0)
            end_idx = task_item.get('end_idx', None)
        else:
            task_name = task_item
            start_idx = 0
            end_idx = None
        
        # 处理特殊任务名，如 convert2_1500, convert2_1000, data1_legacy_140, data1_legacy_10
        limit = None
        original_task_name = task_name
        if (task_name.startswith('convert2_') or task_name.startswith('data1_legacy_')) and task_name != 'convert2_val':
            parts = task_name.split('_')
            if len(parts) > 1 and parts[-1].isdigit():
                limit = int(parts[-1])
                original_task_name = '_'.join(parts[:-1])
        
        for cur_base_dir in [base_dir] + (extra_base_dirs or []):
            task_dir = os.path.join(cur_base_dir, original_task_name)
            if os.path.exists(task_dir) and os.path.isdir(task_dir):
                print(f"Task dir: {task_dir}")
                cur_task_cnt = 0
                hdf_files = []
                for fn in sorted(os.listdir(task_dir)):
                    if fn.endswith('.hdf5'):
                        hdf_path = os.path.join(task_dir, fn)
                        if camera_names is not None:
                            with h5py.File(hdf_path, 'r') as root:
                                has_camera = False
                                for cam_name in camera_names:
                                    if f'observation.image.{cam_name}' in root:
                                        has_camera = True
                                        break
                                    if 'observation.image.left' in root or 'observation.image.right' in root:
                                        has_camera = True
                                        break
                                if not has_camera:
                                    print(f"Skipping {hdf_path} - no camera data")
                                    continue
                        hdf_files.append(hdf_path)
                
                # 应用限制
                if limit:
                    hdf_files = hdf_files[:limit]
                
                # 应用索引范围
                hdf_files = hdf_files[start_idx:end_idx]
                
                all_hdf_paths.extend(hdf_files)
                cur_task_cnt = len(hdf_files)
                task_episode_cnt.append(cur_task_cnt)
                found = True
                break
        if not found:
            print(f"Warning: Task {task_name} not found in any base directory")
            task_episode_cnt.append(0)
    return all_hdf_paths, task_episode_cnt

def gather_lang_embeds_paths(base_dir, task_names, extra_base_dirs=None):
    lang_embeds_paths = []
    for task_item in task_names:
        # 处理任务项，支持字符串或字典格式
        if isinstance(task_item, dict):
            task_name = task_item.get('dataset_path')
        else:
            task_name = task_item
        
        fn = f"{task_name}.pkl"
        found = False
        for cur_base_dir in [base_dir] + (extra_base_dirs or []):
            path = os.path.join(cur_base_dir, fn)
            if os.path.exists(path):
                lang_embeds_paths.append(path)
                found = True
                break
        if not found:
            print(f"Warning: {fn} does not exist in any base directory")
            lang_embeds_paths.append(None)
    
    return lang_embeds_paths

def get_all_episode_len(hdf_list):
    all_episode_len = []
    for hdf_path in hdf_list:
        with h5py.File(hdf_path, 'r') as root:
            qpos = root['observation.state'][()]
        all_episode_len.append(len(qpos))

    return all_episode_len

def collate_fn(batch):
    image_data, qpos_data, action_data, is_pad, conditioning_list = zip(*batch)
    image_data = torch.stack(image_data)
    qpos_data = torch.stack(qpos_data)
    action_data = torch.stack(action_data)
    is_pad = torch.stack(is_pad)
    
    # Can be accessed by T5 tokenizer.pad_token_id
    KEYWORDS_LIST = ['language_embeddings', 'plain_text']
    ret_conditioning_dict = {}
    for keyword in KEYWORDS_LIST:
        # handle varying lengths
        cur_conditioning_list = [conditioning[keyword] for conditioning in conditioning_list]
        if keyword == 'language_embeddings':
            if any(cond is None for cond in cur_conditioning_list):
                continue
            cur_conditioning_len = [arr.shape[0] for arr in cur_conditioning_list]
            cur_conditioning_tensor = torch.nn.utils.rnn.pad_sequence(
                    cur_conditioning_list,
                    batch_first=True,
                    padding_value=0)
            valid_mask = torch.zeros(
                cur_conditioning_tensor.shape[0], cur_conditioning_tensor.shape[1], dtype=torch.bool)
            for i, l in enumerate(cur_conditioning_len):
                valid_mask[i, :l] = True
            
            ret_conditioning_dict[keyword] = cur_conditioning_tensor
            ret_conditioning_dict[keyword + '_mask'] = valid_mask
        elif keyword == 'plain_text':
            ret_conditioning_dict[keyword] = cur_conditioning_list
        
    return image_data, qpos_data, action_data, is_pad, ret_conditioning_dict

def load_data(base_dir,
              data_config,
              dataset_json_path: str,
              camera_names, 
              chunk_size, 
              batch_size_train, 
              batch_size_val, 
              visual_preprocessor,
              cond_mask_prob,
              slow_down_factor=4,
              extra_base_dirs=None):
    
    
    assert os.path.exists(dataset_json_path)

    with open(dataset_json_path, 'r') as f:
        dataset_config = json.load(f)
    
    dataset_dict = {}
    
    for split in dataset_config:
        assert split in ['train', 'val'], "Only train and val splits now supported, you gave {}".format(split)
        task_names = dataset_config[split]
        # 对任务列表进行排序，支持字典和字符串格式
        if task_names and isinstance(task_names[0], dict):
            task_names = sorted(task_names, key=lambda x: x.get('dataset_path', ''))
        else:
            task_names = sorted(task_names)
        hdf_path_list, task_episode_cnt = gather_hdf_paths(base_dir, task_names, camera_names, extra_base_dirs)
        hdf_path_list = sorted(hdf_path_list)
        lang_embeds_paths = gather_lang_embeds_paths(base_dir, task_names, extra_base_dirs)

        print("Total {} episodes for {} split".format(len(hdf_path_list), split))
        all_episode_len = get_all_episode_len(hdf_path_list)

        dataset_dict[split] = EpisodicDataset(data_config,
                                    hdf_path_list, 
                                    lang_embeds_paths, 
                                    camera_names, 
                                    chunk_size, 
                                    all_episode_len, 
                                    task_episode_cnt,
                                    visual_preprocessor, 
                                    cond_mask_prob, 
                                    train=True,
                                    slow_down_factor=slow_down_factor)
    
    train_dataset = dataset_dict['train']
    val_dataset = dataset_dict['val']

    val_dataset.norm_stats = train_dataset.get_norm_stats()
    train_dataloader = DataLoader(train_dataset, pin_memory=True, num_workers=8, batch_size=batch_size_train, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, pin_memory=True, num_workers=8, batch_size=batch_size_val, collate_fn=collate_fn)

    norm_stats = train_dataset.get_norm_stats()

    return train_dataloader, val_dataloader, norm_stats

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
