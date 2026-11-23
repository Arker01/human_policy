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
from hdt.action_representation import ACTION_REPRESENTATIONS, convert_action_chunk

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


def flow_target_filename(hdf_path: str) -> str:
    """Map a source episode hdf5 to its flow-target h5 basename.

    Single source of truth for the naming, imported by both this dataloader and
    scripts/preprocess/flow_target.py so the two can never drift.

    The absolute path is flattened rather than hashed: episodes come from several
    base dirs (base_dir plus extra_base_dirs) so basenames collide, but a flattened
    absolute path stays unique AND stays readable, which matters when a
    "flow target not found" error needs to be traced back to one episode.
    """
    flat = os.path.abspath(hdf_path).lstrip(os.sep).replace(os.sep, '__')
    for ext in ('.hdf5', '.h5'):
        if flat.endswith(ext):
            flat = flat[:-len(ext)]
            break
    return flat + '.flow.h5'


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
                 future_horizon=0,
                 future_clip_frames=1,
                 future_clip_stride=1,
                 flow_target_enabled=False,
                 flow_target_dir=None,
                 flow_grid_hw=(30, 40),
                 flow_horizon=100,
                 history_offsets=None,
                 lag_aug_config=None,
                 visual_aug_config=None,
                 action_representation="original",
                 robot_target_fps=30.0):
        super(EpisodicDataset).__init__()
        # Future-DINO: when disabled (default) every code path below is byte-for-byte
        # the original one and read_one() still returns the original 5-tuple.
        self.future_image_enabled = bool(future_image_enabled)
        self.future_horizon = int(future_horizon)
        # clip_frames == 1 (default) keeps the single-frame teacher path bit-identical.
        # clip_frames > 1 is video-teacher mode (V-JEPA 2): read a K-frame clip ending
        # at t AND one ending at t+H, spaced clip_stride raw frames apart.
        self.future_clip_frames = max(1, int(future_clip_frames))
        self.future_clip_stride = max(1, int(future_clip_stride))
        if self.future_image_enabled:
            assert self.future_horizon > 0, "future_horizon must be > 0 when future images are enabled"
        # ---- EgoWAM 3D flow target (disabled by default) ----
        # Read straight off disk from scripts/preprocess/flow_target.py, so unlike the
        # DINO/VAE targets this needs no future frame and is independent of the flags
        # above. Files are opened lazily per worker: h5py handles do not survive fork.
        self.flow_target_enabled = bool(flow_target_enabled)
        self.flow_target_dir = flow_target_dir
        self.flow_grid_h, self.flow_grid_w = int(flow_grid_hw[0]), int(flow_grid_hw[1])
        self.flow_num_anchors = self.flow_grid_h * self.flow_grid_w
        self.flow_horizon = int(flow_horizon)
        self._flow_handles = {}
        if self.flow_target_enabled:
            assert self.flow_target_dir and os.path.isdir(self.flow_target_dir), \
                f"flow_target_dir not found: {self.flow_target_dir}"

        # ---- Real observation history (disabled by default) ----
        # Offsets are frames on the common logical 30fps timeline and are NOT scaled
        # by slow_down_factor. Thus [5, 10] means 0.17s/0.33s ago for both a native
        # 30fps episode and robot teleop resampled online from 50fps.
        self.history_offsets = [int(o) for o in (history_offsets or [])]
        self.history_enabled = len(self.history_offsets) > 0
        self.camera_names = camera_names
        self.train = train
        self.data_config = data_config
        self.raw_episode_len_list = np.asarray(episode_len_list, dtype=np.int64)
        self.dataset_paths = hdf_path_list
        self.task_episode_cnt = task_episode_cnt
        self.visual_preprocessor = visual_preprocessor
        self.chunk_size = chunk_size  # action length (e.g., chunk size)
        self.action_str = 'old_action' if control_mode == 'qpos' else 'action'
        self.action_representation = str(action_representation)
        if self.action_representation not in ACTION_REPRESENTATIONS:
            raise ValueError(
                f"action_representation must be one of {ACTION_REPRESENTATIONS}, "
                f"got {self.action_representation!r}")
        self.robot_target_fps = (
            None if robot_target_fps is None else float(robot_target_fps)
        )
        if self.robot_target_fps is not None and self.robot_target_fps <= 0:
            raise ValueError("robot_target_fps must be positive or None")

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
        self.source_fps_list, self.source_embodiment_list = self._read_episode_metadata()
        self.valid_start_list = self._compute_valid_start_list()
        self.episode_frame_indices = self._build_episode_frame_indices()
        self.episode_len_list = np.asarray(
            [max(1, len(indices) - (self.action_representation == 'relative_chunk'))
             for indices in self.episode_frame_indices], dtype=np.int64)
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
        
        # ---- Execution-lag state augmentation (disabled by default) ----
        # Perturbs the INPUT wrist state only; the action chunk label is untouched.
        # Teaches "the wrist is behind the plan -> keep pushing it to the bottle mouth
        # and close" instead of reading the current position as part of the plan.
        # Train split only, so val/loss stays comparable to the un-augmented runs.
        self.lag_aug = dict(lag_aug_config) if lag_aug_config else None
        if self.lag_aug is not None:
            assert not self.history_enabled, (
                "lag_aug perturbs the current state only; the un-perturbed history "
                "states would let the trunk extrapolate the true current wrist pose "
                "and undo the augmentation. Decide how history should lag first.")
            # Right arm is the manipulator in this task: OUTPUT_RIGHT_EEF is 3 pos + 6D
            # rot, OUTPUT_RIGHT_KEYPOINTS are palm-LOCAL fingertips, which is why the
            # finger block can be desynced from the wrist without contradicting it.
            _w0 = int(OUTPUT_RIGHT_EEF[0])
            self.lag_wrist_pos_slice = slice(_w0, _w0 + 3)
            self.lag_wrist_slice = slice(_w0, int(OUTPUT_RIGHT_EEF[-1]) + 1)
            self.lag_finger_slice = slice(int(OUTPUT_RIGHT_KEYPOINTS[0]),
                                          int(OUTPUT_RIGHT_KEYPOINTS[-1]) + 1)

        # ---- Visual localization augmentation (disabled by default) ----
        # RandomResizedCrop back to the SAME resolution, so every shape downstream
        # (backbone input, DINO patch grid) is unchanged. The random crop window gives
        # the translation jitter for free, without the black borders a pad-based shift
        # would introduce.
        self.visual_aug = dict(visual_aug_config) if visual_aug_config else None
        if self.visual_aug is not None:
            assert not self.flow_target_enabled, (
                "visual_aug crops the frame but the flow target is precomputed on a "
                "fixed image-space anchor grid; the two would silently disagree.")

        _train_tf = []
        if self.visual_aug is not None:
            _h, _w = (int(self.data_config["image_resolution_hw"][0]),
                      int(self.data_config["image_resolution_hw"][1]))
            _train_tf.append(torchvision.transforms.v2.RandomResizedCrop(
                size=(_h, _w),
                scale=tuple(self.visual_aug["scale"]),
                ratio=tuple(self.visual_aug["ratio"]),
                antialias=True,
            ))
        _train_tf.append(
            torchvision.transforms.v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4))
        # torchvision.transforms.v2.GaussianBlur(kernel_size=(9,9), sigma=(0.1,2.0)),
        self.training_transforms = torchvision.transforms.v2.Compose(_train_tf)

        self.norm_stats, self.embodiment_list = self.get_norm_stats()

        SAMPLER_TYPE = 'norm_by_embodiment_and_task'
        self.episode_sampling_prob = self.get_episode_sampling_prob(SAMPLER_TYPE)

        # Load empty language embedding from the correct path.
        # 不要在这里 import os: 那会把整个 __init__ 里的 os 变成局部变量，
        # 于是上面 flow_target_dir 那句断言（第 118 行）会先撞上 UnboundLocalError。
        empty_lang_embed_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "empty_lang_embed.pt")
        empty_lang_embedding = torch.load(empty_lang_embed_path, weights_only=True).float()
        self.cached_lang_embedding_dict[''] = empty_lang_embedding

    @staticmethod
    def _infer_embodiment(path, value):
        if isinstance(value, bytes):
            value = value.decode()
        embodiment = str(value)
        if embodiment == 'default':
            lower = path.lower()
            if 'human' in lower:
                embodiment = 'human'
            elif 'robot' in lower or 'dex5' in lower or 'sonic' in lower:
                embodiment = 'robot'
        return embodiment

    def _read_episode_metadata(self):
        fps_list = []
        embodiment_list = []
        for path in self.dataset_paths:
            with h5py.File(path, 'r') as root:
                embodiment = self._infer_embodiment(
                    path, root.attrs.get('embodiment', 'default'))
                # Historical HAT corpora without an fps attribute are 30fps.  We do
                # not guess a higher rate from filenames: only explicit metadata may
                # cause a trajectory to be temporally resampled.
                fps = float(root.attrs.get('fps', 30.0))
            if not np.isfinite(fps) or fps <= 0:
                raise ValueError(f"Invalid fps={fps!r} in {path}")
            fps_list.append(fps)
            embodiment_list.append(embodiment)
        return np.asarray(fps_list, dtype=np.float64), embodiment_list

    def _build_episode_frame_indices(self):
        """Build the logical observation timeline used by every data path.

        Robot teleoperation recorded above ``robot_target_fps`` is mapped onto a
        30fps logical timeline with the same nearest-frame rule as
        scripts/preprocess/resample_sonic.py.  Images, state, action labels,
        history, and future targets all consume this one mapping.
        """
        timelines = []
        resampled = []
        for idx, (path, raw_len, valid_start, fps, embodiment) in enumerate(zip(
                self.dataset_paths, self.raw_episode_len_list, self.valid_start_list,
                self.source_fps_list, self.source_embodiment_list)):
            should_resample = (
                self.robot_target_fps is not None
                and 'human' not in embodiment.lower()
                and fps > self.robot_target_fps + 1e-6
            )
            if should_resample:
                step = fps / self.robot_target_fps
                n_out = max(1, int(np.floor(int(raw_len) / step)))
                indices = np.rint(np.arange(n_out, dtype=np.float64) * step).astype(np.int64)
                indices = np.clip(indices, 0, int(raw_len) - 1)
                indices = np.unique(indices)
                indices = indices[indices >= int(valid_start)]
                if len(indices) == 0:
                    indices = np.asarray([min(int(valid_start), int(raw_len) - 1)])
                resampled.append((os.path.basename(path), fps, len(indices), int(raw_len)))
            else:
                indices = np.arange(int(valid_start), int(raw_len), dtype=np.int64)
                if len(indices) == 0:
                    indices = np.asarray([int(raw_len) - 1], dtype=np.int64)
            timelines.append(indices)

        if resampled:
            split = "train" if self.train else "val"
            rates = sorted({float(item[1]) for item in resampled})
            before = sum(item[3] for item in resampled)
            after = sum(item[2] for item in resampled)
            print(
                f"Robot teleop FPS normalization ({split}): {len(resampled)} episodes "
                f"{rates}fps -> {self.robot_target_fps:g}fps, {before} -> {after} frames")
        return timelines

    def _raw_index(self, episode_index, logical_ts):
        timeline = self.episode_frame_indices[episode_index]
        logical_ts = min(max(int(logical_ts), 0), len(timeline) - 1)
        return int(timeline[logical_ts])

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
                    human_frame_counts.append(self.episode_len_list[idx])
                else:
                    robot_frame_counts.append(self.episode_len_list[idx])
        
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
    
    @staticmethod
    def _interpolate_rows(values, positions):
        """Linearly sample a 2-D action array at fractional frame positions."""
        positions = np.asarray(positions, dtype=np.float64)
        if len(positions) == 0:
            return np.empty((0, int(values.shape[1])), dtype=np.float32)
        lo = np.floor(positions).astype(np.int64)
        hi = np.minimum(lo + 1, int(values.shape[0]) - 1)
        first, last = int(lo.min()), int(hi.max())
        block = np.asarray(values[first:last + 1], dtype=np.float32)
        alpha = (positions - lo).astype(np.float32)[:, None]
        return block[lo - first] * (1.0 - alpha) + block[hi - first] * alpha

    def _read_action_chunk(self, index, logical_ts, states, actions):
        """Return an unpadded action chunk and its current absolute anchor."""
        embodiment = self.source_embodiment_list[index]
        raw_start = self._raw_index(index, logical_ts)
        anchor = np.asarray(states[raw_start], dtype=np.float32)
        remaining = int(self.episode_len_list[index]) - int(logical_ts)

        if self.action_representation == 'relative_chunk':
            if 'human' in embodiment.lower():
                # Human trajectories are intentionally slowed in action space.  A
                # chunk step therefore advances 1/slow_down_factor source frames.
                factor = max(int(self.slow_down_factor), 1)
                positions = raw_start + np.arange(1, self.chunk_size + 1) / factor
                valid = positions <= int(actions.shape[0]) - 1
                positions = positions[valid]
                chunk = self._interpolate_rows(actions, positions)
            else:
                timeline = self.episode_frame_indices[index]
                raw_indices = timeline[int(logical_ts) + 1:int(logical_ts) + 1 + self.chunk_size]
                chunk = np.asarray(actions[raw_indices], dtype=np.float32)
            return convert_action_chunk(anchor, chunk, self.action_representation), anchor

        if 'human' in embodiment.lower():
            raw = np.asarray(actions[raw_start:raw_start + self.chunk_size])
            chunk = hdt.inference_utils.interpolate_128dim_action(
                raw, raw.shape[0] * max(int(self.slow_down_factor), 1))
            chunk = np.asarray(chunk[:self.chunk_size], dtype=np.float32)
            # Preserve the original padding convention for the original action mode.
            chunk = chunk[:min(remaining, len(chunk))]
        else:
            timeline = self.episode_frame_indices[index]
            raw_indices = timeline[int(logical_ts):int(logical_ts) + self.chunk_size]
            chunk = np.asarray(actions[raw_indices], dtype=np.float32)
        return chunk, anchor

    @staticmethod
    def _update_moments(accumulator, values):
        values = np.asarray(values, dtype=np.float64)
        if values.size == 0:
            return
        accumulator['count'] += int(values.shape[0])
        accumulator['sum'] += values.sum(axis=0)
        accumulator['sum_sq'] += np.square(values).sum(axis=0)

    @staticmethod
    def _finish_moments(accumulator):
        count = accumulator['count']
        if count == 0:
            raise ValueError("Cannot compute normalization statistics from zero rows")
        mean = accumulator['sum'] / count
        if count > 1:
            variance = (
                accumulator['sum_sq'] - count * np.square(mean)
            ) / (count - 1)
        else:
            variance = np.zeros_like(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        return mean.astype(np.float32), np.maximum(std, 1e-2).astype(np.float32)

    def get_norm_stats(self):
        if hasattr(self, 'norm_stats'):
            return self.norm_stats

        norm_stats_dict = {}
        for embodiment in self.source_embodiment_list:
            norm_stats_dict.setdefault(embodiment, {
                'action_moments': {
                    'count': 0,
                    'sum': np.zeros(ACTION_STATE_VEC_SIZE, dtype=np.float64),
                    'sum_sq': np.zeros(ACTION_STATE_VEC_SIZE, dtype=np.float64),
                },
                'state_moments': {
                    'count': 0,
                    'sum': np.zeros(ACTION_STATE_VEC_SIZE, dtype=np.float64),
                    'sum_sq': np.zeros(ACTION_STATE_VEC_SIZE, dtype=np.float64),
                },
            })
        print(f"Found embodiments: {norm_stats_dict.keys()}")

        for index, hdf_path in enumerate(self.dataset_paths):
            embodiment = self.source_embodiment_list[index]
            timeline = self.episode_frame_indices[index]
            with h5py.File(hdf_path, 'r') as root:
                states = np.asarray(root['observation.state'])
                actions = np.asarray(root[self.action_str])
            self._update_moments(
                norm_stats_dict[embodiment]['state_moments'],
                states[timeline[:int(self.episode_len_list[index])]])

            if self.action_representation == 'original':
                self._update_moments(
                    norm_stats_dict[embodiment]['action_moments'], actions[timeline])
            else:
                # Relative values depend on both the anchor and the future horizon,
                # so their statistics must be measured after conversion rather than
                # borrowed from the absolute per-frame action distribution.
                horizon = np.arange(1, self.chunk_size + 1, dtype=np.int64)
                anchor_count = int(self.episode_len_list[index])
                for batch_start in range(0, anchor_count, 256):
                    logical = np.arange(
                        batch_start, min(batch_start + 256, anchor_count),
                        dtype=np.int64)
                    anchors = states[timeline[logical]][:, None, :]
                    if 'human' in embodiment.lower():
                        factor = max(int(self.slow_down_factor), 1)
                        positions = timeline[logical, None] + horizon[None, :] / factor
                        valid = positions <= int(actions.shape[0]) - 1
                        future = self._interpolate_rows(
                            actions,
                            np.minimum(positions, int(actions.shape[0]) - 1).reshape(-1)
                        ).reshape(
                                len(logical), self.chunk_size, ACTION_STATE_VEC_SIZE)
                    else:
                        future_logical = logical[:, None] + horizon[None, :]
                        valid = future_logical < len(timeline)
                        future_raw = timeline[
                            np.minimum(future_logical, len(timeline) - 1)]
                        future = actions[future_raw]
                    relative = convert_action_chunk(
                        anchors, future, self.action_representation)
                    self._update_moments(
                        norm_stats_dict[embodiment]['action_moments'], relative[valid])

        for embodiment, stats in norm_stats_dict.items():
            action_mean, action_std = self._finish_moments(stats.pop('action_moments'))
            qpos_mean, qpos_std = self._finish_moments(stats.pop('state_moments'))
            if self.predict_delta_action:
                action_mean = np.zeros_like(action_mean)
            stats.update({
                'action_mean': action_mean,
                'action_std': action_std,
                'qpos_mean': qpos_mean,
                'qpos_std': qpos_std,
                'action_representation': self.action_representation,
                'robot_target_fps': self.robot_target_fps,
            })

        return norm_stats_dict, list(self.source_embodiment_list)
    
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

    def _read_cam_clip_ending_at(self, root, end_ts, single_hdf_path, first_ts=0):
        """
        Future-DINO video-teacher mode. Read future_clip_frames frames ENDING at
        end_ts, spaced future_clip_stride raw frames apart, in chronological order.
        Timesteps that fall before the episode start are clamped to first_ts -- a clip
        at t=0 then just repeats frame 0, which is what actually happened (the robot
        was standing still), not a fabricated frame.

        Returns a list of K entries, each a list of per-camera CHW uint8 arrays.
        """
        K, S = self.future_clip_frames, self.future_clip_stride
        return [self._read_cam_images_at(root, max(first_ts, end_ts - (K - 1 - i) * S),
                                         single_hdf_path)
                for i in range(K)]

    def _read_flow_target(self, index, raw_start_ts, embodiment):
        """Fetch the precomputed 3D-flow trajectory anchored at raw frame raw_start_ts.

        The h5 written by scripts/preprocess/flow_target.py is indexed by RAW frame
        offset 1..R. The head, however, is indexed by CHUNK STEP 1..K, and for human
        episodes those are not the same thing: human chunks are time-compressed by
        slow_down_factor, so chunk step k lands on raw offset k / factor. Same
        correction the DINO path applies to its single horizon (see the human branch
        below), just applied to all K steps at once.

        Returns (target [K,P,3] float32, flow_valid [K] float32, anchor_valid [P] float32).
        """
        path = self.dataset_paths[index]
        handle = self._flow_handles.get(path)
        if handle is None:
            flow_path = os.path.join(self.flow_target_dir, flow_target_filename(path))
            handle = h5py.File(flow_path, 'r')
            self._flow_handles[path] = handle

        raw_offsets = int(handle['flow_target'].shape[1])
        K = self.flow_horizon
        if "human" in embodiment:
            factor = max(self.slow_down_factor, 1)
            offs = np.rint(np.arange(1, K + 1) / factor).astype(np.int64)
        else:
            # Use the actual nearest-frame timeline, including its alternating
            # 1/2-frame gaps for 50->30, rather than an average fps multiplier.
            timeline = self.episode_frame_indices[index]
            logical_ts = int(np.searchsorted(timeline, raw_start_ts))
            future_logical = logical_ts + np.arange(1, K + 1, dtype=np.int64)
            in_episode = future_logical < len(timeline)
            clipped = np.minimum(future_logical, len(timeline) - 1)
            offs = timeline[clipped] - int(raw_start_ts)
            # Values after the episode are masked below. Give them an out-of-range
            # offset so they cannot accidentally reuse the final valid flow row.
            offs[~in_episode] = raw_offsets + 1
        # The preprocessing stores only R raw offsets, and R can be smaller than the
        # horizon the head predicts: one tracker window covers ~39 frames ahead, and
        # chaining more windows would pile up drift on top of an ~1cm noise floor. So
        # chunk steps past R get MASKED, not clamped -- clamping would silently claim
        # "nothing moves after step R", which is a false target rather than a missing
        # one. Human episodes are unaffected: factor 4 puts step 100 at raw offset 25.
        oob = offs > raw_offsets
        offs = np.clip(offs, 1, raw_offsets)

        T_flow = int(handle['flow_target'].shape[0])
        t = min(int(raw_start_ts), T_flow - 1)

        # One contiguous read of [R, P, 3] (~720KB at R=100, P=1200, fp16) and then
        # index in numpy: the human offsets repeat, and h5py fancy indexing requires
        # strictly increasing indices.
        block = np.asarray(handle['flow_target'][t], dtype=np.float32)      # [R, P, 3]
        valid = np.asarray(handle['flow_valid'][t], dtype=np.float32)       # [R]
        anchor = np.asarray(handle['anchor_valid'][t], dtype=np.float32)    # [P]

        target = block[offs - 1]                                            # [K, P, 3]
        flow_valid = valid[offs - 1]                                        # [K]
        flow_valid[oob] = 0.0

        if int(raw_start_ts) >= T_flow:
            # Preprocessing trimmed further than the trajectory sampler does (human
            # clips lose 20 frames at each end). Zero the anchor mask instead of
            # training on a frame whose flow we never computed.
            anchor = np.zeros_like(anchor)

        assert target.shape == (K, self.flow_num_anchors, 3), (
            f"flow target {target.shape} != ({K}, {self.flow_num_anchors}, 3) for {path}")
        return (torch.from_numpy(target),
                torch.from_numpy(flow_valid),
                torch.from_numpy(anchor))

    def _apply_lag_aug(self, root, index, raw_start_ts, qpos):
        """Simulate "the BFM did not keep up" on the INPUT state.

        Two independent perturbations, both on the raw (pre-normalization) state so the
        thresholds stay in metres:

        1. Trajectory backtrack. Sample a target lag distance d ~ U[min_m, max_m] and
           replace the wrist block (pos + 6D rot) with the value at the MOST RECENT past
           frame that sits at least d away. "Most recent" matters: a trajectory that
           revisits a position would otherwise send the lag arbitrarily far back in time.
           A lagging wrist also has a lagging orientation, hence the whole 9-dim block.
        2. Random 3D offset on the wrist position, for tracking error that is not a pure
           lag along the path.

        Optionally the fingertip block is read at a DIFFERENT lag than the wrist, which is
        the finger-vs-wrist progress desync: fingers already closing while the wrist is
        still behind, or the reverse.

        The action chunk is deliberately not touched anywhere in here.
        """
        cfg = self.lag_aug
        ds = root['observation.state']
        first_ts = int(self.valid_start_list[index])
        # load_hdf_to_cpu hands back a VIEW into the cached episode array, so mutating
        # in place would poison the cache for every later sample.
        qpos = np.array(qpos, copy=True)

        if random.random() < cfg['prob'] and raw_start_ts > first_ts:
            target_m = random.uniform(cfg['min_m'], cfg['max_m'])
            lo = max(first_ts, raw_start_ts - int(cfg['search_frames']))
            past = np.asarray(ds[lo:raw_start_ts, self.lag_wrist_pos_slice], dtype=np.float64)
            here = np.asarray(qpos[self.lag_wrist_pos_slice], dtype=np.float64)
            dist = np.linalg.norm(past - here, axis=1)
            hit = np.nonzero(dist >= target_m)[0]
            # Nothing in the window is far enough (slow phase near the episode start):
            # take the farthest frame available rather than skipping the sample, so the
            # augmentation still fires, just at a smaller lag than requested.
            j = int(hit[-1]) if len(hit) else int(np.argmax(dist))
            lag_frames = raw_start_ts - (lo + j)
            qpos[self.lag_wrist_slice] = ds[lo + j, self.lag_wrist_slice]

            if lag_frames > 0 and random.random() < cfg['finger_prob']:
                # finger_scale > 1 lets the fingers lag MORE than the wrist as well as less.
                f_lag = int(round(random.uniform(0.0, cfg['finger_scale']) * lag_frames))
                f_ts = max(first_ts, raw_start_ts - f_lag)
                qpos[self.lag_finger_slice] = ds[f_ts, self.lag_finger_slice]

        if random.random() < cfg['offset_prob']:
            off = np.random.randn(3) * cfg['offset_sigma_m']
            norm = float(np.linalg.norm(off))
            if norm > cfg['offset_clip_m'] > 0:
                off = off * (cfg['offset_clip_m'] / norm)
            qpos[self.lag_wrist_pos_slice] = qpos[self.lag_wrist_pos_slice] + off

        return qpos

    def read_one(self, index, start_ts):
        # relative_chunk excludes the final no-future frame from sampling, while
        # future-image clamping still needs the full observation timeline length.
        episode_len = len(self.episode_frame_indices[index])
        raw_start_ts = self._raw_index(index, start_ts)

        single_hdf_path = self.dataset_paths[index]

        if self.load_hdf_to_cpu:
            root = self.cached_hdf_dict[single_hdf_path]
        else:
            root = h5py.File(single_hdf_path, 'r')

        # Build the label from the clean current pose before lag augmentation.  Lag
        # changes only what the policy observes; it must not move the demonstration
        # trajectory or its HuMI anchor.
        all_time_action, _ = self._read_action_chunk(
            index, start_ts, root['observation.state'], root[self.action_str])
        qpos = root['observation.state'][raw_start_ts]
        if self.lag_aug is not None and self.train:
            qpos = self._apply_lag_aug(root, index, raw_start_ts, qpos)

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
        # ---- Real history: read frames/states at t-o for each offset o ----
        hist_cam_images = None
        hist_qpos = None
        if self.history_enabled:
            # Clamped to the episode's first usable frame, NOT masked. A clamped slot
            # repeats frame 0, which is what actually happened (nothing had moved yet);
            # the per-slot time embedding on the model side still says which slot it
            # is, so the model can tell "0.33s ago" from "start of episode".
            hist_ts = [self._raw_index(index, int(start_ts) - o)
                       for o in self.history_offsets]
            hist_cam_images = [self._read_cam_images_at(root, ts, single_hdf_path)
                               for ts in hist_ts]
            hist_qpos = np.stack([root['observation.state'][ts] for ts in hist_ts], axis=0)

        # ---- Future-DINO: read the frame at t+H (disabled by default) ----
        future_cam_images = None
        current_clip = future_clip = None
        future_valid = 1.0
        if self.future_image_enabled:
            _embodiment = self.source_embodiment_list[index]
            horizon = self.future_horizon
            if "human" in _embodiment:
                # Human action chunks are time-compressed below by slow_down_factor:
                # chunk_size actions only span chunk_size/factor raw frames. Scale the
                # visual horizon identically, otherwise the world target sits far
                # beyond the actions the trunk is being asked to predict.
                horizon = max(1, int(round(horizon / max(self.slow_down_factor, 1))))
            future_logical_ts = int(start_ts) + horizon
            if future_logical_ts >= episode_len:
                # Clamp, but mark invalid so the loss masks it out. Training on the
                # clamped frame would teach "the future is static" near episode ends.
                future_logical_ts = episode_len - 1
                future_valid = 0.0
            future_ts = self._raw_index(index, future_logical_ts)
            if self.future_clip_frames > 1:
                # Video teacher: the query side is a clip ending at t and the target a
                # clip ending at t+H. Both are read here so that the single
                # visual_preprocessor/training_transforms call below applies the
                # identical photometric augmentation to every frame (doc S2.3).
                current_clip = [
                    self._read_cam_images_at(
                        root,
                        self._raw_index(index, int(start_ts) - (self.future_clip_frames - 1 - i)
                                        * self.future_clip_stride),
                        single_hdf_path)
                    for i in range(self.future_clip_frames)
                ]
                future_clip = [
                    self._read_cam_images_at(
                        root,
                        self._raw_index(index, future_logical_ts - (self.future_clip_frames - 1 - i)
                                        * self.future_clip_stride),
                        single_hdf_path)
                    for i in range(self.future_clip_frames)
                ]
            else:
                future_cam_images = self._read_cam_images_at(root, future_ts, single_hdf_path)

        if not self.load_hdf_to_cpu:
            lang_instruction = root.attrs.get('description', '')
            embodiment = root.attrs.get('embodiment', 'default')
            root.close()
        else:
            lang_instruction = root['attrs'].get('description', '')
            embodiment = root['attrs'].get('embodiment', 'default')
        
        padded_action = np.zeros((self.chunk_size, all_time_action.shape[1]), dtype=np.float32)
        padded_action[:all_time_action.shape[0], :] = all_time_action

        is_pad = np.zeros(self.chunk_size, dtype=bool)
        is_pad[all_time_action.shape[0]:] = True

        # new axis for different cameras
        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        # construct observations
        # History frames ride along at the TAIL of whatever stack is preprocessed
        # below, for the same reason the future frame does: torchvision v2 samples
        # ColorJitter once per call, so one call gives every timestep the identical
        # photometric transform. Separate calls would let the model spot the current
        # frame by its colour statistics alone.
        hist_block = None
        n_hist_rows = 0
        if self.history_enabled:
            hist_block = np.stack([np.stack(f, axis=0) for f in hist_cam_images], axis=0)
            n_hist = hist_block.shape[0]
            hist_block = hist_block.reshape(-1, *all_cam_images.shape[1:])
            n_hist_rows = hist_block.shape[0]

        future_image_data = None
        hist_image_data = None
        if self.future_image_enabled:
            # Preprocess + augment current and future in ONE call: torchvision v2
            # samples the ColorJitter params once per call, so both frames get the
            # identical photometric transform (doc S2.3).
            n_cam = all_cam_images.shape[0]
            if self.future_clip_frames > 1:
                K = self.future_clip_frames
                # [current frame for the trunk] + [2K clip frames for the teacher],
                # flattened as (k * n_cam + cam) so the model can fold it back to
                # [2K, n_cam, ...] and then to per-camera clips.
                clip_stack = np.stack(
                    [np.stack(frame, axis=0) for frame in (current_clip + future_clip)],
                    axis=0)                                    # [2K, n_cam, C, H, W]
                blocks = [all_cam_images,
                          clip_stack.reshape(2 * K * n_cam, *all_cam_images.shape[1:])]
                if hist_block is not None:
                    blocks.append(hist_block)
                stacked = np.concatenate(blocks, axis=0)
                stacked = self.visual_preprocessor(stacked)
                if self.train:
                    stacked = self.training_transforms(stacked)
                image_data = stacked[:n_cam]
                # len(stacked)-n_hist_rows is just len(stacked) when history is off,
                # so this slice is the original stacked[n_cam:].
                future_image_data = stacked[n_cam:len(stacked) - n_hist_rows].reshape(
                    2 * K, n_cam, *stacked.shape[1:])
            else:
                blocks = [all_cam_images, np.stack(future_cam_images, axis=0)]
                if hist_block is not None:
                    blocks.append(hist_block)
                stacked = np.concatenate(blocks, axis=0)
                stacked = self.visual_preprocessor(stacked)
                if self.train:
                    stacked = self.training_transforms(stacked)
                image_data = stacked[:n_cam]
                future_image_data = stacked[n_cam:len(stacked) - n_hist_rows]
            if hist_block is not None:
                hist_image_data = stacked[len(stacked) - n_hist_rows:].reshape(
                    n_hist, n_cam, *stacked.shape[1:])
        else:
            if hist_block is not None:
                n_cam = all_cam_images.shape[0]
                stacked = self.visual_preprocessor(
                    np.concatenate([all_cam_images, hist_block], axis=0))
                if self.train:
                    stacked = self.training_transforms(stacked)
                image_data = stacked[:n_cam]
                hist_image_data = stacked[n_cam:].reshape(n_hist, n_cam, *stacked.shape[1:])
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
        # Same single random.random() draw and the exact complementary condition as
        # before -- only hoisted into a variable so the history states below can be
        # blanked by the SAME event. Blanking the current state but keeping t-0.17s
        # would leak the very thing cond_mask is hiding.
        _state_masked = random.random() <= self.cond_mask_prob
        qpos_data = torch.zeros_like(qpos_data) if _state_masked \
            else (qpos_data - qpos_mean) / (qpos_std + 1e-6)
        
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

        if self.history_enabled:
            # Rides in conditioning_dict for the same reason the flow target does:
            # the 5-tuple / 6-tuple return contract stays untouched.
            hist_qpos_data = torch.from_numpy(hist_qpos).float()
            hist_qpos_data = torch.zeros_like(hist_qpos_data) if _state_masked \
                else (hist_qpos_data - qpos_mean) / (qpos_std + 1e-6)
            conditioning_dict["hist_image"] = hist_image_data
            conditioning_dict["hist_qpos"] = hist_qpos_data

        if self.flow_target_enabled:
            # Rides in conditioning_dict, not in the return tuple, so the 5-tuple /
            # 6-tuple contract above is untouched and a flow-only arm still takes the
            # original 5-tuple path.
            # 用上面关闭 root 之前已经取好的 embodiment：这里 root 早就 close 了
            # （第 606 行），再去读它的 attrs 会得到
            # KeyError: invalid identifier type to function。
            flow_target, flow_valid, flow_anchor_valid = self._read_flow_target(
                index, raw_start_ts, embodiment)
            conditioning_dict["flow_target"] = flow_target
            conditioning_dict["flow_valid"] = flow_valid
            conditioning_dict["flow_anchor_valid"] = flow_anchor_valid

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

    # 3D-flow target. This dict is built from an allow-list above, so keys that are
    # not named here get silently dropped -- which is exactly how a flow arm would
    # end up training with no world loss at all and no error to show for it.
    if 'flow_target' in conditioning_list[0]:
        for key in ('flow_target', 'flow_valid', 'flow_anchor_valid'):
            ret_conditioning_dict[key] = torch.stack(
                [conditioning[key] for conditioning in conditioning_list])

    # Same allow-list trap as above: without this the history arm would train on
    # the current frame only and look like a plain baseline that mysteriously costs 3x.
    if 'hist_image' in conditioning_list[0]:
        for key in ('hist_image', 'hist_qpos'):
            ret_conditioning_dict[key] = torch.stack(
                [conditioning[key] for conditioning in conditioning_list])

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
              future_horizon=0,
              future_clip_frames=1,
              future_clip_stride=1,
              flow_target_enabled=False,
              flow_target_dir=None,
              flow_grid_hw=(30, 40),
              flow_horizon=100,
              history_offsets=None,
              lag_aug_config=None,
              visual_aug_config=None,
              action_representation="original",
              robot_target_fps=30.0):


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
                                    future_horizon=future_horizon,
                                    future_clip_frames=future_clip_frames,
                                    future_clip_stride=future_clip_stride,
                                    flow_target_enabled=flow_target_enabled,
                                    flow_target_dir=flow_target_dir,
                                    flow_grid_hw=flow_grid_hw,
                                    flow_horizon=flow_horizon,
                                    history_offsets=history_offsets,
                                    lag_aug_config=lag_aug_config,
                                    visual_aug_config=visual_aug_config,
                                    action_representation=action_representation,
                                    robot_target_fps=robot_target_fps)
    
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
