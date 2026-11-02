from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class ObjectLabelStore:
    """Lazy reader for immutable, per-episode object label arrays."""

    REQUIRED = (
        "uv", "target_valid", "target_quality", "present", "present_valid",
        "object_to_hand_uv", "relation_valid", "relation_quality",
    )

    def __init__(self, root: str | Path, task: str) -> None:
        self.root = Path(root)
        self.task = str(task)
        self._cache: dict[tuple[str, str], dict[str, np.ndarray] | None] = {}

    def _load(self, domain: str, episode_path: str) -> dict[str, np.ndarray] | None:
        key = (domain, Path(episode_path).stem)
        if key not in self._cache:
            path = self.root / self.task / domain / f"{key[1]}.npz"
            if not path.exists():
                self._cache[key] = None
            else:
                with np.load(path, allow_pickle=False) as archive:
                    missing = set(self.REQUIRED) - set(archive.files)
                    if missing:
                        raise ValueError(f"{path} misses label fields {sorted(missing)}")
                    self._cache[key] = {name: np.asarray(archive[name]) for name in self.REQUIRED}
        return self._cache[key]

    def sample(self, domain: str, episode_path: str, frame: int) -> dict[str, torch.Tensor]:
        data = self._load(domain, episode_path)
        if data is None or frame >= len(data["target_valid"]):
            return self.empty()
        return {
            "object_uv": torch.as_tensor(data["uv"][frame], dtype=torch.float32),
            "object_valid": torch.tensor(float(data["target_valid"][frame]), dtype=torch.float32),
            "object_quality": torch.tensor(float(data["target_quality"][frame]), dtype=torch.float32),
            "object_present": torch.tensor(float(data["present"][frame]), dtype=torch.float32),
            "object_present_valid": torch.tensor(float(data["present_valid"][frame]), dtype=torch.float32),
            "object_to_hand_uv": torch.as_tensor(
                data["object_to_hand_uv"][frame], dtype=torch.float32),
            "object_hand_relation_valid": torch.tensor(
                float(data["relation_valid"][frame]), dtype=torch.float32),
            "object_hand_relation_quality": torch.tensor(
                float(data["relation_quality"][frame]), dtype=torch.float32),
        }

    @staticmethod
    def empty() -> dict[str, torch.Tensor]:
        return {
            "object_uv": torch.zeros(2, dtype=torch.float32),
            "object_valid": torch.tensor(0.0),
            "object_quality": torch.tensor(0.0),
            "object_present": torch.tensor(0.0),
            "object_present_valid": torch.tensor(0.0),
            "object_to_hand_uv": torch.zeros(2, dtype=torch.float32),
            "object_hand_relation_valid": torch.tensor(0.0),
            "object_hand_relation_quality": torch.tensor(0.0),
        }
