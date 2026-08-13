import numpy as np
import torch
import json
import re
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

EARLY_JUMP_CHECK_SLICES = (
    OUTPUT_HEAD_EEF[0:3],
    OUTPUT_RIGHT_EEF[0:3],
    OUTPUT_NECK[0:3],
    OUTPUT_LEFT_EEF[0:3],
    OUTPUT_WAIST[0:3],
)


def _detect_valid_start_from_actions(actions, *, check_frames=20, jump_threshold_m=0.3, settle_frames=0):
    if actions.shape[0] <= 1 or check_frames <= 1 or jump_threshold_m <= 0:
        return 0

    n = min(int(check_frames), int(actions.shape[0]))
    last_bad_next_frame = -1
    for sl in EARLY_JUMP_CHECK_SLICES:
        pos = actions[:n, sl]
        if pos.shape[1] != 3:
            continue
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        bad = np.where(step > float(jump_threshold_m))[0]
        if len(bad) > 0:
            last_bad_next_frame = max(last_bad_next_frame, int(bad[-1]) + 1)

    if last_bad_next_frame < 0:
        return 0
    valid_start = last_bad_next_frame + int(settle_frames)
    return min(valid_start, max(0, int(actions.shape[0]) - 1))


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
                 slow_down_factor=4,
                 auto_trim_dirty_start=True,
                 dirty_start_check_frames=20,
                 dirty_start_jump_threshold_m=0.3,
                 dirty_start_settle_frames=0,
                 future_image_enabled=False,
                 future_horizon=0):
        super(EpisodicDataset).__init__()
        # Future-DINO: when disabled (default) every code path below is byte-for-byte
        # the original one and read_one() still returns the original 5-tuple.
        self.future_image_enabled = bool(future_image_enabled)
        self.future_horizon = int(future_horizon)
        if self.future_image_enabled:
            assert self.future_horizon > 0, "future_horizon must be > 0 when future images are enabled"
        self.camera_names = camera_names
        self.train = train
        self.data_config = data_config
        self.raw_episode_len_list = np.asarray(episode_len_list, dtype=np.int64)
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
        self.auto_trim_dirty_start = bool(auto_trim_dirty_start)
        self.dirty_start_check_frames = int(dirty_start_check_frames)
        self.dirty_start_jump_threshold_m = float(dirty_start_jump_threshold_m)
        self.dirty_start_settle_frames = int(dirty_start_settle_frames)
        self.valid_start_list = self._compute_valid_start_list()
        self.episode_len_list = np.maximum(self.raw_episode_len_list - self.valid_start_list, 1)
        self.cumulative_len = np.cumsum(self.episode_len_list)

        self.sum_dataset_len_l = np.cumsum([0] + [np.sum(episode_len) for episode_len in self.episode_len_list])
        
        self.slow_down_factor = slow_down_factor

        if self.slow_down_factor <= 0:
            self.slow_down_factor = self._compute_auto_slow_down_factor()

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

    def _compute_valid_start_list(self):
        valid_starts = np.zeros(len(self.dataset_paths), dtype=np.int64)
        if not self.auto_trim_dirty_start:
            return valid_starts

        trimmed = []
        for idx, single_hdf_path in enumerate(self.dataset_paths):
            try:
                with h5py.File(single_hdf_path, 'r') as root:
                    if self.action_str not in root:
                        continue
                    n = min(self.dirty_start_check_frames, int(root[self.action_str].shape[0]))
                    actions = root[self.action_str][:n]
                valid_start = _detect_valid_start_from_actions(
                    actions,
                    check_frames=self.dirty_start_check_frames,
                    jump_threshold_m=self.dirty_start_jump_threshold_m,
                    settle_frames=self.dirty_start_settle_frames,
                )
                valid_start = min(valid_start, max(0, int(self.raw_episode_len_list[idx]) - 1))
                valid_starts[idx] = valid_start
                if valid_start > 0:
                    trimmed.append((os.path.basename(single_hdf_path), int(valid_start)))
            except (KeyError, OSError) as e:
                print(f"Warning: failed to detect dirty start for {single_hdf_path}: {e}")

        if trimmed:
            preview = ", ".join([f"{name}:{start}" for name, start in trimmed[:8]])
            more = "" if len(trimmed) <= 8 else f", ... +{len(trimmed) - 8} more"
            split = "train" if self.train else "val"
            print(f"Auto-trimmed dirty starts for {len(trimmed)}/{len(self.dataset_paths)} {split} episodes: {preview}{more}")
        return valid_starts
    
    def _compute_auto_slow_down_factor(self):
        human_frame_counts = []
        robot_frame_counts = []
        for idx, single_hdf_path in enumerate(self.dataset_paths):
            with h5py.File(single_hdf_path, 'r') as root:
                embodiment = root.attrs.get('embodiment', 'default')
                if embodiment == 'default':
                    if 'human' in single_hdf_path.lower():
                        embodiment = 'human'
                    elif 'robot' in single_hdf_path.lower() or 'dex5' in single_hdf_path.lower():
                        embodiment = 'robot'
                
                if 'human' in embodiment.lower():
                    human_frame_counts.append(self.raw_episode_len_list[idx])
                else:
                    robot_frame_counts.append(self.raw_episode_len_list[idx])
        
        if human_frame_counts and robot_frame_counts:
            human_avg = np.mean(human_frame_counts)
            robot_avg = np.mean(robot_frame_counts)
            ratio = robot_avg / human_avg
            auto_factor = max(1, round(ratio))
            print(f"Auto-computed slow_down_factor: {auto_factor} (human_avg={human_avg:.1f}, robot_avg={robot_avg:.1f}, ratio={ratio:.2f})")
            return auto_factor
        else:
            print("Warning: Cannot auto-compute slow_down_factor, using default 1")
            return 1
    
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
                
                # 如果 HDF5 中没有 embodiment，从文件路径推断
                if embodiment == 'default':
                    if 'human' in single_hdf_path.lower():
                        embodiment = 'human'
                    elif 'robot' in single_hdf_path.lower():
                        embodiment = 'robot'
                    # 否则保持为 'default'
                
                if embodiment not in norm_stats_dict:
                    norm_stats_dict[embodiment] = {
                        "actions": [],
                        "states": []
                    }
                embodiment_list.append(embodiment)
        print(f"Found embodiments: {norm_stats_dict.keys()}")

        for hdf_idx, hdf_path in enumerate(self.dataset_paths):
            with h5py.File(hdf_path, 'r') as root:
                valid_start = int(self.valid_start_list[hdf_idx])
                state = root['observation.state'][valid_start:]
                action = root['action'][valid_start:]
                # Use default embodiment if not present
                embodiment = root.attrs.get('embodiment', 'default')
                
                # 如果 HDF5 中没有 embodiment，从文件路径推断
                if embodiment == 'default':
                    if 'human' in hdf_path.lower():
                        embodiment = 'human'
                    elif 'robot' in hdf_path.lower():
                        embodiment = 'robot'
                
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

    def _resolve_cam_name(self, root, cam_name, single_hdf_path):
        if f'observation.image.{cam_name}' in root:
            return cam_name
        if 'observation.image.left' in root:
            return 'left'
        if 'observation.image.right' in root:
            return 'right'
        raise KeyError(f"None of the expected cameras (top, left, right) found in {single_hdf_path}")

    def _read_cam_images_at(self, root, raw_ts, single_hdf_path):
        """
        Future-DINO only. Decode every camera at one timestep, returning a list of
        CHW uint8 arrays. Mirrors the current-frame decoding in read_one() but has no
        cond_mask branch: the future frame is a supervision *target*, blanking it to
        zeros would be meaningless.
        """
        images = []
        for cam_name in self.camera_names:
            actual_cam_name = self._resolve_cam_name(root, cam_name, single_hdf_path)
            img = root[f'observation.image.{actual_cam_name}'][raw_ts]
            if len(img.shape) == 1:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (self.data_config["image_resolution_hw"][1],
                                   self.data_config["image_resolution_hw"][0]))
            images.append(img.transpose(2, 0, 1))  # CHW
        return images

    def read_one(self, index, start_ts):
        episode_len = int(self.episode_len_list[index])
        raw_start_ts = int(self.valid_start_list[index]) + int(start_ts)

        single_hdf_path = self.dataset_paths[index]

        if self.load_hdf_to_cpu:
            root = self.cached_hdf_dict[single_hdf_path]
        else:
            root = h5py.File(single_hdf_path, 'r')

        qpos = root['observation.state'][raw_start_ts]

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
                
                image_dict[cam_name] = root[f'observation.image.{actual_cam_name}'][raw_start_ts]
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
        all_time_action = root[self.action_str][raw_start_ts:raw_start_ts+self.chunk_size]

        # ---- Future-DINO: read the frame at t+H (disabled by default) ----
        future_cam_images = None
        future_valid = 1.0
        if self.future_image_enabled:
            if self.load_hdf_to_cpu:
                _embodiment = root['attrs'].get('embodiment', 'default')
            else:
                _embodiment = root.attrs.get('embodiment', 'default')
            horizon = self.future_horizon
            if "human" in _embodiment:
                # Human action chunks are time-compressed below by slow_down_factor:
                # chunk_size actions only span chunk_size/factor raw frames. Scale the
                # visual horizon identically, otherwise the world target sits far
                # beyond the actions the trunk is being asked to predict.
                horizon = max(1, int(round(horizon / max(self.slow_down_factor, 1))))
            last_raw_ts = int(self.raw_episode_len_list[index]) - 1
            future_ts = raw_start_ts + horizon
            if future_ts > last_raw_ts:
                # Clamp, but mark invalid so the loss masks it out. Training on the
                # clamped frame would teach "the future is static" near episode ends.
                future_ts = last_raw_ts
                future_valid = 0.0
            future_cam_images = self._read_cam_images_at(root, future_ts, single_hdf_path)

        if not self.load_hdf_to_cpu:
            lang_instruction = root.attrs.get('description', '')
            embodiment = root.attrs.get('embodiment', 'default')
            root.close()
        else:
            lang_instruction = root['attrs'].get('description', '')
            embodiment = root['attrs'].get('embodiment', 'default')
        
        if "human" in embodiment:
            SLOW_DOWN_FACTOR = self.slow_down_factor
            all_time_action = hdt.inference_utils.interpolate_128dim_action(all_time_action, all_time_action.shape[0] * SLOW_DOWN_FACTOR)
            all_time_action = all_time_action[:self.chunk_size]
        
        padded_action = np.zeros((self.chunk_size, all_time_action.shape[1]), dtype=np.float32)
        padded_action[:all_time_action.shape[0], :] = all_time_action
        
        real_len = episode_len - int(start_ts)

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[real_len:] = True

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        future_image_data = None
        if self.future_image_enabled:
            # Preprocess + augment current and future in ONE call: torchvision v2
            # samples the ColorJitter params once per call, so both frames get the
            # identical photometric transform (doc S2.3).
            n_cam = all_cam_images.shape[0]
            stacked = np.concatenate([all_cam_images, np.stack(future_cam_images, axis=0)], axis=0)
            stacked = self.visual_preprocessor(stacked)
            if self.train:
                stacked = self.training_transforms(stacked)
            image_data, future_image_data = stacked[:n_cam], stacked[n_cam:]
        else:
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

        action_mean = torch.from_numpy(self.norm_stats[embodiment]["action_mean"]).float()
        action_std = torch.from_numpy(self.norm_stats[embodiment]["action_std"]).float()
        qpos_mean = torch.from_numpy(self.norm_stats[embodiment]["qpos_mean"]).float()
        qpos_std = torch.from_numpy(self.norm_stats[embodiment]["qpos_std"]).float()
        
        action_data = (action_data - action_mean) / (action_std + 1e-6)
        qpos_data = (qpos_data - qpos_mean) / (qpos_std + 1e-6) \
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

        if self.future_image_enabled:
            conditioning_dict["future_valid"] = torch.tensor(future_valid, dtype=torch.float32)
            return image_data, qpos_data, action_data, is_pad, conditioning_dict, future_image_data

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

def _numeric_sort_key(fn):
    m = re.search(r'(\d+)', fn)
    return (int(m.group(1)), fn) if m else (float('inf'), fn)

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
            file_list = task_item.get('file_list', None)
        else:
            task_name = task_item
            start_idx = 0
            end_idx = None
            file_list = None
        
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
                if file_list is not None:
                    # Explicit file list (caller controls exact membership/order,
                    # bypassing listdir + sort entirely). Used to avoid the
                    # lexicographic-vs-numeric sort mismatch below.
                    filenames = file_list
                else:
                    # NOTE: plain sorted(os.listdir(...)) is a *lexicographic*
                    # string sort, e.g. "wholebody-10" sorts before
                    # "wholebody-2" -- this previously caused a biased
                    # train/val split. Sort numerically by embedded episode id.
                    filenames = sorted(os.listdir(task_dir), key=_numeric_sort_key)
                for fn in filenames:
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
    # 6-tuple only when the Future-DINO dataloader path is enabled; otherwise this is
    # the original 5-tuple and nothing below changes.
    future_image_data = None
    if len(batch[0]) == 6:
        image_data, qpos_data, action_data, is_pad, conditioning_list, future_list = zip(*batch)
        future_image_data = torch.stack(future_list)
    else:
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

    if future_image_data is not None:
        ret_conditioning_dict['future_valid'] = torch.stack(
            [conditioning['future_valid'] for conditioning in conditioning_list])
        return image_data, qpos_data, action_data, is_pad, ret_conditioning_dict, future_image_data

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
              extra_base_dirs=None,
              auto_trim_dirty_start=True,
              dirty_start_check_frames=20,
              dirty_start_jump_threshold_m=0.3,
              dirty_start_settle_frames=0,
              future_image_enabled=False,
              future_horizon=0):


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
                                    train=(split == 'train'),
                                    slow_down_factor=slow_down_factor,
                                    auto_trim_dirty_start=auto_trim_dirty_start,
                                    dirty_start_check_frames=dirty_start_check_frames,
                                    dirty_start_jump_threshold_m=dirty_start_jump_threshold_m,
                                    dirty_start_settle_frames=dirty_start_settle_frames,
                                    future_image_enabled=future_image_enabled,
                                    future_horizon=future_horizon)
    
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
