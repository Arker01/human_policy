"""Conditioning utilities for the semantic-dropout and H+P ablations.

This module deliberately has no task-specific policy logic.  It only samples
missing conditioning groups and reads cached predictions made by a frozen teacher.
"""

import hashlib
import json
import os

import numpy as np
import torch


SEMANTIC_BLOCKS = {
    "head_rot": slice(3, 9),
    "root_rot": slice(92, 98),
    "right_wrist_rot": slice(33, 39),
    "right_hand_keypoints": slice(40, 58),
    "left_wrist_rot": slice(83, 89),
    "left_hand_keypoints": slice(10, 28),
}
SEMANTIC_MASK_ORDER = (
    "head_rot", "root_rot", "right_wrist_rot",
    "right_hand_keypoints", "visual_tokens", "left_wrist_rot", "left_hand_keypoints",
)


def episode_cache_key(path):
    return hashlib.sha256(os.path.abspath(path).encode("utf-8")).hexdigest()[:20]


def sample_semantic_missing(config, rng=None):
    """Apply the cap without favoring either hand's position in the mask."""
    if rng is None:
        rng = np.random
    probs = config.get("probabilities", {})
    cap = int(config.get("max_missing_blocks", 2))
    if cap < 2:
        raise ValueError("The joint head/root dropout requires at least two mask slots")
    if rng.random() < float(probs.get("head_and_root_rot", 0.05)):
        missing = {"head_rot", "root_rot"}
    else:
        hits = [
            name for name in SEMANTIC_MASK_ORDER
            if rng.random() < float(probs.get(name, 0.0))
        ]
        missing = set(rng.permutation(hits)[:cap]) if len(hits) > cap else set(hits)
    return torch.tensor([float(name in missing) for name in SEMANTIC_MASK_ORDER])



class TeacherPlanCache:
    """Lazy, worker-safe reader for frozen-teacher action chunks."""

    def __init__(self, cache_dir):
        self.cache_dir = os.path.abspath(cache_dir)
        manifest_path = os.path.join(self.cache_dir, "manifest.json")
        if not os.path.isfile(manifest_path):
            raise FileNotFoundError(f"teacher cache manifest not found: {manifest_path}")
        with open(manifest_path, "r") as f:
            self.manifest = json.load(f)
        if self.manifest.get("split") != "train":
            raise ValueError("H+P training cache must be generated from train split only")
        if int(self.manifest.get("action_dim", -1)) != 128:
            raise ValueError("teacher cache action_dim must be 128")
        self.entries = self.manifest["episodes"]
        self._arrays = {}

    def _load(self, episode_path):
        key = episode_cache_key(episode_path)
        entry = self.entries.get(key)
        if entry is None or os.path.abspath(entry["source_path"]) != os.path.abspath(episode_path):
            return None
        if key not in self._arrays:
            self._arrays[key] = np.load(os.path.join(self.cache_dir, entry["file"]), mmap_mode="r")
        return self._arrays[key]

    def previous(self, episode_path, current_t, rng=None, min_age=3, max_age=30):
        """Choose a genuinely earlier cached prediction and align its playback cursor.

        Age is sampled over a range rather than assuming that every inference was
        accepted for exactly five frames.  The cursor is the actual logical-frame
        difference between the cached teacher timestamp and current_t.
        """
        rng = np.random if rng is None else rng
        arrays = self._load(episode_path)
        if arrays is None:
            return None
        times = arrays["times"]
        if len(times) == 0 or current_t <= 0:
            return None
        target_age = int(rng.randint(int(min_age), int(max_age) + 1))
        target_t = int(current_t) - target_age
        eligible = np.flatnonzero(times <= target_t)
        if len(eligible) == 0:
            eligible = np.flatnonzero(times < int(current_t))
        if len(eligible) == 0:
            return None
        i = int(eligible[-1])
        t_prev = int(times[i])
        cursor = int(current_t) - t_prev
        chunk = np.asarray(arrays["chunks"][i], dtype=np.float32)
        valid_len = int(arrays["valid_lengths"][i])
        if not (0 < cursor < valid_len):
            return None
        return chunk, valid_len, t_prev, cursor


def anchor_remaining_plan(chunk, valid_len, cursor, anchor_count):
    remaining = int(valid_len) - int(cursor)
    out = np.zeros((int(anchor_count), 128), dtype=np.float32)
    mask = np.zeros(int(anchor_count), dtype=bool)
    times = np.zeros(int(anchor_count), dtype=np.float32)
    if remaining <= 0:
        return out, mask, times
    count = min(int(anchor_count), remaining)
    indices = np.rint(np.linspace(int(cursor), int(valid_len) - 1, count)).astype(np.int64)
    out[:count] = chunk[indices]
    mask[:count] = True
    times[:count] = (indices - int(cursor)) / 30.0
    return out, mask, times
