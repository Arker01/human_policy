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
        self.load_hdf_to_cpu = True

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
                            self.cached_hdf_dict[single_hdf_path][f'observation.image.{cam_name}'] = root[f'observation.image.{cam_name}'][()]
                        self.cached_hdf_dict[single_hdf_path][self.action_str] =  root[self.action_str][()]
                        self.cached_hdf_dict[single_hdf_path]['attrs'] = {k: v for k, v in root.attrs.items()}
        
        self.training_transforms = torchvision.transforms.v2.Compose([
            torchvision.transforms.v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            # torchvision.transforms.v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
        ])

        self.norm_stats, self.embodiment_list = self.get_norm_stats()

        SAMPLER_TYPE = 'norm_by_embodiment_and_task'
        self.episode_sampling_prob = self.get_episode_sampling_prob(SAMPLER_TYPE)

        empty_lang_embedding = torch.load("empty_lang_embed.pt").float()
        self.cached_lang_embedding_dict[''] = empty_lang_embedding
    
    def get_episode_sampling_prob(self, sampler_type):
        if sampler_type == 'uniform':
            return None  # when it is None, the default behavior is uniform sampling
        elif sampler_type == 'norm_by_embodiment':
            P_ROBOT = 0.5
            P_HUMAN = 1 - P_ROBOT
            human_idx_list = []
            robot_idx_list = []
            cur_idx = 0
            for task_idx in range(len(self.task_episode_cnt)):
                task_episode_cnt = self.task_episode_cnt[task_idx]
                cur_embodiment = self.embodiment_list[cur_idx]
                assert self.embodiment_list[cur_idx:cur_idx + task_episode_cnt] == [cur_embodiment] * task_episode_cnt
                if "human" in cur_embodiment:
                    human_idx_list.extend(list(range(cur_idx, cur_idx + task_episode_cnt)))
                else:
                    robot_idx_list.extend(list(range(cur_idx, cur_idx + task_episode_cnt)))
                cur_idx += task_episode_cnt
            assert len(self.embodiment_list) == len(self.episode_len_list)
            prob_arr = np.ones(len(self.episode_len_list)) / len(self.episode_len_list)
            prob_arr[human_idx_list] = P_HUMAN / len(human_idx_list)
            prob_arr[robot_idx_list] = P_ROBOT / len(robot_idx_list)
            if not np.isclose(np.sum(prob_arr), 1):
                print("=========")
                print(f"Warning: sum of prob_arr is not 1: {np.sum(prob_arr)}. Is only one embodiment available?")
                prob_arr = prob_arr / np.sum(prob_arr)
            return prob_arr
        elif sampler_type == 'norm_by_embodiment_and_task':
            P_ROBOT = 0.5
            P_HUMAN = 1 - P_ROBOT
            human_task_idx_group = []
            robot_task_idx_group = []
            cur_idx = 0
            for task_idx in range(len(self.task_episode_cnt)):
                task_episode_cnt = self.task_episode_cnt[task_idx]
                cur_embodiment = self.embodiment_list[cur_idx]
                assert self.embodiment_list[cur_idx:cur_idx + task_episode_cnt] == [cur_embodiment] * task_episode_cnt
                if "human" in cur_embodiment:
                    human_task_idx_group.append(list(range(cur_idx, cur_idx + task_episode_cnt)))
                else:
                    robot_task_idx_group.append(list(range(cur_idx, cur_idx + task_episode_cnt)))
                cur_idx += task_episode_cnt
            assert len(self.embodiment_list) == len(self.episode_len_list)
            prob_arr = np.ones(len(self.episode_len_list)) / len(self.episode_len_list)
            for task_idx_list in human_task_idx_group:
                prob_arr[task_idx_list] = P_HUMAN / (len(task_idx_list) * len(human_task_idx_group))
            for task_idx_list in robot_task_idx_group:
                prob_arr[task_idx_list] = P_ROBOT / (len(task_idx_list) * len(robot_task_idx_group))
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
                norm_stats_dict[root.attrs['embodiment']] = {
                    "actions": [],
                    "states": []
                }
                embodiment_list.append(root.attrs['embodiment'])
        print(f"Found embodiments: {norm_stats_dict.keys()}")

        for hdf_path in self.dataset_paths:
            with h5py.File(hdf_path, 'r') as root:
                state = root['observation.state'][()]
                action = root['action'][()]
                norm_stats_dict[root.attrs['embodiment']]["actions"].append(torch.from_numpy(action))
                norm_stats_dict[root.attrs['embodiment']]["states"].append(torch.from_numpy(state))
        
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
                image_dict[cam_name] = root[f'observation.image.{cam_name}'][start_ts]
                if len(image_dict[cam_name].shape) == 1:
                    # Compressed JPEG format images are represented as (N,) uint8 array. N is different for every image.
                    image_dict[cam_name] = cv2.imdecode(image_dict[cam_name], cv2.IMREAD_COLOR)
                    image_dict[cam_name] = cv2.resize(image_dict[cam_name], (self.data_config["image_resolution_hw"][1], self.data_config["image_resolution_hw"][0]))
                    assert image_dict[cam_name].shape == (self.data_config["image_resolution_hw"][0], self.data_config["image_resolution_hw"][1], 3)
                    # Images are in RGB (verified by plt.imshow) and HWC in this case
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
            selected_embedding = self.cached_lang_embedding_dict[lang_instruction].float()
        else:
            selected_embedding = None

        conditioning_dict = {
            "language_embeddings": selected_embedding,
            "plain_text": lang_instruction
        }

        return image_data, qpos_data, action_data, is_pad, conditioning_dict
    
    def __getitem__(self, _idx):
        episode_idx = np.random.choice(len(self.episode_len_list), p=self.episode_sampling_prob)
        ts_index = np.random.randint(self.sum_dataset_len_l[episode_idx], self.sum_dataset_len_l[episode_idx + 1])

        # index: index of the selected episode ID, start_ts: start timestep within the episode
        index, start_ts = self._locate_transition(ts_index)
        return self.read_one(index, start_ts)

def gather_hdf_paths(base_dir, task_names):
    all_hdf_paths = []
    task_episode_cnt = []
    for task_name in task_names:
        task_dir = os.path.join(base_dir, task_name)
        print(f"Task dir: {task_dir}")
        assert os.path.exists(task_dir) and os.path.isdir(task_dir)
        cur_task_cnt = 0
        for fn in sorted(os.listdir(task_dir)):
            if fn.endswith('.hdf5'):
                all_hdf_paths.append(os.path.join(task_dir, fn))
                cur_task_cnt += 1
        task_episode_cnt.append(cur_task_cnt)
    return all_hdf_paths, task_episode_cnt

def gather_lang_embeds_paths(base_dir, task_names):
    lang_embeds_paths = []
    for task_name in task_names:
        fn = f"{task_name}.pkl"
        if os.path.exists(os.path.join(base_dir, fn)):
            lang_embeds_paths.append(os.path.join(base_dir, fn))
        else:
            print(f"Warning: {fn} does not exist in {base_dir}")
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
              slow_down_factor=4):
    
    
    assert os.path.exists(dataset_json_path)

    with open(dataset_json_path, 'r') as f:
        dataset_config = json.load(f)
    
    dataset_dict = {}
    
    for split in dataset_config:
        assert split in ['train', 'val'], "Only train and val splits now supported, you gave {}".format(split)
        task_names = dataset_config[split]
        task_names = sorted(task_names)
        hdf_path_list, task_episode_cnt = gather_hdf_paths(base_dir, task_names)
        assert hdf_path_list == sorted(hdf_path_list)
        lang_embeds_paths = gather_lang_embeds_paths(base_dir, task_names)

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
