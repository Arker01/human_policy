"""Online action-chunk service for the HAT -> ScaleBFM closed loop.

This module deliberately does not own cameras.  The deployment's existing
visual process supplies a ``module:function`` callback returning the camera
array expected by the selected HAT checkpoint.  Robot state arrives from
ScaleBFM and predicted chunks are published back over latest-only ZMQ sockets.
"""

from __future__ import annotations

import argparse
import importlib
import json
import pickle
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

import hdt.constants as C
from hdt.modeling.utils import make_visual_encoder


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCALEBRIDGE_ROOT = Path("/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge")
PROTOCOL_ROOT = DEFAULT_SCALEBRIDGE_ROOT


def _jsonable(value):
    """Convert protocol/NumPy values without losing their numeric contents."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


class OnlineHATInputRecorder:
    """Losslessly record the exact inputs and outputs of every HAT inference."""

    FORMAT_VERSION = 1

    def __init__(self, run_dir, metadata):
        self.run_dir = Path(run_dir)
        self.images_dir = self.run_dir / "images"
        self.actions_dir = self.run_dir / "actions"
        self.images_dir.mkdir(parents=True, exist_ok=False)
        self.actions_dir.mkdir(parents=True, exist_ok=False)
        manifest = {
            "format_version": self.FORMAT_VERSION,
            "started_at": datetime.now().astimezone().isoformat(),
            "image_storage": (
                "lossless NumPy NPZ; images is uint8 NCHW in the exact "
                "channel order presented to the policy"
            ),
            "action_storage": "NumPy NPY; denormalized HAT action shaped (T,128)",
            **_jsonable(metadata),
        }
        (self.run_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._stream = (self.run_dir / "inference_inputs.jsonl").open(
            "a", encoding="utf-8", buffering=1
        )

    @classmethod
    def create_default(cls, base_dir, metadata):
        base_dir = Path(base_dir)
        stamp = datetime.now().astimezone().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = base_dir / stamp
        suffix = 1
        while run_dir.exists():
            run_dir = base_dir / f"{stamp}_{suffix:02d}"
            suffix += 1
        return cls(run_dir, metadata)

    @staticmethod
    def _atomic_save(path, writer):
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            writer(stream)
        temporary.replace(path)

    def record(
        self, *, sequence_id, source_state, origin, images,
        policy_state_raw, policy_state_normalized, action,
        image_captured_ns, inference_started_ns, inference_finished_ns,
        chunk_generated_at_ns,
    ):
        stem = f"{int(sequence_id):06d}"
        image_path = self.images_dir / f"{stem}.npz"
        action_path = self.actions_dir / f"{stem}.npy"
        images = np.asarray(images)
        action = np.asarray(action, dtype=np.float32)
        self._atomic_save(
            image_path,
            lambda stream: np.savez_compressed(stream, images=images),
        )
        self._atomic_save(
            action_path,
            lambda stream: np.save(stream, action, allow_pickle=False),
        )
        record = {
            "record_type": "hat_inference",
            "sequence_id": int(sequence_id),
            "source_state_sequence_id": int(source_state["sequence_id"]),
            "source_state_timestamp_ns": int(source_state["timestamp_ns"]),
            "image_captured_ns": int(image_captured_ns),
            "inference_started_ns": int(inference_started_ns),
            "inference_finished_ns": int(inference_finished_ns),
            "inference_duration_s": (
                int(inference_finished_ns) - int(inference_started_ns)
            ) * 1e-9,
            "chunk_generated_at_ns": int(chunk_generated_at_ns),
            "images_file": str(image_path.relative_to(self.run_dir)),
            "actions_file": str(action_path.relative_to(self.run_dir)),
            "image_shape": list(images.shape),
            "image_dtype": str(images.dtype),
            "action_shape": list(action.shape),
            "head_origin": _jsonable(np.asarray(origin, dtype=np.float32)),
            "robot_state": _jsonable(source_state),
            "policy_state_raw": _jsonable(
                np.asarray(policy_state_raw, dtype=np.float32)
            ),
            "policy_state_normalized": _jsonable(
                np.asarray(policy_state_normalized, dtype=np.float32)
            ),
        }
        self._stream.write(json.dumps(record, separators=(",", ":")) + "\n")
        self._stream.flush()

    def close(self):
        if not self._stream.closed:
            self._stream.close()


def _install_protocol(scale_bridge_root: Path):
    root = str(scale_bridge_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from scalebridge.online.protocol import (
        make_hat_chunk,
        monotonic_ns,
        validate_robot_state,
    )
    from scalebridge.online.transport import LatestPublisher, LatestSubscriber
    return make_hat_chunk, monotonic_ns, validate_robot_state, LatestPublisher, LatestSubscriber


def _rotation_6d_to_quat_wxyz(value):
    """Match pytorch3d's row-vector 6D convention without importing it."""
    value = np.asarray(value, dtype=np.float64)
    a1, a2 = value[..., :3], value[..., 3:6]
    norm1 = np.linalg.norm(a1, axis=-1, keepdims=True)
    if np.any(norm1 < 1e-6):
        raise ValueError("HAT emitted a degenerate first 6D rotation axis")
    b1 = a1 / norm1
    a2_orth = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    norm2 = np.linalg.norm(a2_orth, axis=-1, keepdims=True)
    if np.any(norm2 < 1e-6):
        raise ValueError("HAT emitted collinear 6D rotation axes")
    b2 = a2_orth / norm2
    b3 = np.cross(b1, b2)
    matrix = np.stack((b1, b2, b3), axis=-2)
    from scipy.spatial.transform import Rotation
    xyzw = Rotation.from_matrix(matrix).as_quat()
    return np.concatenate((xyzw[..., 3:4], xyzw[..., :3]), axis=-1).astype(np.float32)


def _quat_wxyz_to_rotation_6d(value):
    """Quaternion to the first two matrix rows used by HAT/PyTorch3D 6D."""
    quat = np.asarray(value, dtype=np.float64)
    if quat.shape != (4,) or not np.isfinite(quat).all():
        raise ValueError(f"Expected one finite wxyz quaternion, got {quat.shape}")
    norm = np.linalg.norm(quat)
    if norm < 1e-6:
        raise ValueError("Cannot convert a zero quaternion to HAT rotation 6D")
    quat = quat / norm
    from scipy.spatial.transform import Rotation
    # The deployed HAT environment carries an older SciPy without
    # ``scalar_first``; convert wxyz to its native xyzw explicitly.
    matrix = Rotation.from_quat(
        [quat[1], quat[2], quat[3], quat[0]]
    ).as_matrix()
    return matrix[:2].reshape(6).astype(np.float32)


WRIST_CONVERSION_MODES = ("none", "inspire_hand_base")

# Constant mount of the Inspire hand base on the G1 wrist, taken from
# g1_29dof_inspire_hand.xml: L/R_hand_base_link sits at pos [0.0415,0,0] under
# left/right_wrist_yaw_link with these local rotations (wxyz).  The dex5
# dataset stores wrist poses in the hand-base frame (attr
# xqy3_wrist_rot_frame == 'raw'), while the simulator publishes and consumes
# wrist_yaw_link poses, so:  hand_base = wrist_yaw * mount.
_WRIST_MOUNT_POS = np.array([0.0415, 0.0, 0.0], dtype=np.float64)
_WRIST_MOUNT_QUAT_WXYZ = {
    "left": np.array([0.70710678, 0.0, 0.0, 0.70710678], dtype=np.float64),
    "right": np.array([0.0, 0.70710678, -0.70710678, 0.0], dtype=np.float64),
}


def _quat_mul_wxyz(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        (
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ),
        axis=-1,
    )


def _quat_rotate_wxyz(quat, vector):
    quat = np.asarray(quat, dtype=np.float64)
    scalar = quat[..., :1]
    axis = quat[..., 1:]
    vector = np.broadcast_to(np.asarray(vector, dtype=np.float64), axis.shape)
    uv = np.cross(axis, vector)
    return vector + 2.0 * (scalar * uv + np.cross(axis, uv))


def _g1_wrist_to_hand_base(pos, quat_wxyz, side):
    """Simulator wrist_yaw_link world pose -> dataset hand-base world pose."""
    pos = np.asarray(pos, dtype=np.float64)
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    return (
        pos + _quat_rotate_wxyz(quat, _WRIST_MOUNT_POS),
        _quat_mul_wxyz(quat, _WRIST_MOUNT_QUAT_WXYZ[side]),
    )


def _hand_base_to_g1_wrist(pos, quat_wxyz, side):
    """Dataset hand-base world pose -> simulator wrist_yaw_link world pose."""
    mount_inv = _WRIST_MOUNT_QUAT_WXYZ[side] * np.array([1.0, -1.0, -1.0, -1.0])
    quat_g1 = _quat_mul_wxyz(np.asarray(quat_wxyz, dtype=np.float64), mount_inv)
    pos_g1 = np.asarray(pos, dtype=np.float64) - _quat_rotate_wxyz(quat_g1, _WRIST_MOUNT_POS)
    return pos_g1, quat_g1


def action_chunk_to_targets(action, source_state, fps, sequence_id, generated_at_ns,
                            wrist_conversion="none", position_origin_world=None):
    """Extract only the HAT fields consumed by online ScaleBFM."""
    if wrist_conversion not in WRIST_CONVERSION_MODES:
        raise ValueError(
            f"wrist_conversion must be one of {WRIST_CONVERSION_MODES}, got {wrist_conversion!r}"
        )
    make_hat_chunk, _, validate_robot_state, _, _ = _install_protocol(PROTOCOL_ROOT)
    state = validate_robot_state(source_state)
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 2 or action.shape[1] != C.ACTION_STATE_VEC_SIZE:
        raise ValueError(f"Expected HAT action shaped (N,128), got {action.shape}")
    n = action.shape[0]

    def pose(indices):
        values = action[:, indices]
        return values[:, :3], _rotation_6d_to_quat_wxyz(values[:, 3:9])

    left_pos, left_quat = pose(C.OUTPUT_LEFT_EEF)
    right_pos, right_quat = pose(C.OUTPUT_RIGHT_EEF)
    head_pos, head_quat = pose(C.OUTPUT_HEAD_EEF)
    root_pos, root_quat = pose(C.OUTPUT_WAIST)
    if wrist_conversion == "inspire_hand_base":
        left_pos, left_quat = _hand_base_to_g1_wrist(left_pos, left_quat, "left")
        right_pos, right_quat = _hand_base_to_g1_wrist(right_pos, right_quat, "right")
        left_pos = left_pos.astype(np.float32)
        left_quat = left_quat.astype(np.float32)
        right_pos = right_pos.astype(np.float32)
        right_quat = right_quat.astype(np.float32)
    left_six = action[:, C.OUTPUT_LEFT_KEYPOINTS].reshape(n, 6, 3)
    right_six = action[:, C.OUTPUT_RIGHT_KEYPOINTS].reshape(n, 6, 3)
    fields = dict(
        sequence_id=int(sequence_id),
        source_state_sequence_id=int(state["sequence_id"]),
        generated_at_ns=int(generated_at_ns),
        fps=float(fps),
        frame_count=int(n),
        root_pos=root_pos,
        root_quat_wxyz=root_quat,
        head_pos=head_pos,
        head_quat_wxyz=head_quat,
        left_wrist_pos=left_pos,
        left_wrist_quat_wxyz=left_quat,
        right_wrist_pos=right_pos,
        right_wrist_quat_wxyz=right_quat,
        # HAT stores [palm, thumb, index, middle, ring, pinky].
        left_fingertip_local=left_six[:, 1:6],
        right_fingertip_local=right_six[:, 1:6],
    )
    if position_origin_world is not None:
        fields["position_origin_world"] = np.asarray(
            position_origin_world, dtype=np.float32
        )
    return make_hat_chunk(**fields)


STATE_TAIL_MODES = ("full", "qpos-only", "zero")


class HeadOrigin:
    """The per-episode origin every dex5 position channel is measured from.

    The converter subtracts the head world position of an episode's opening
    frames from head, both wrists and the waist, and leaves rotations alone.
    The trained statistics show it plainly: head z averages -0.08, waist -0.47
    and the wrists -0.55, i.e. metres below a point at standing head height.
    Feeding raw world positions puts z more than ten standard deviations out of
    distribution, so the same offset has to be measured online.
    """

    def __init__(self, frames):
        self.frames = int(frames)
        self._samples = []
        self._origin = np.zeros(3, dtype=np.float32)

    @property
    def locked(self):
        return len(self._samples) >= self.frames

    def observe(self, head_pos_world):
        """Accumulate opening frames and return the offset to use right now."""
        if not self.locked:
            self._samples.append(np.asarray(head_pos_world, dtype=np.float64))
            self._origin = np.mean(self._samples, axis=0).astype(np.float32)
            if self.locked:
                print(
                    f"[online_hat] head origin locked to {self._origin.tolist()} "
                    f"after {self.frames} frames",
                    flush=True,
                )
        return self._origin


def robot_state_to_policy_state(robot_state, state_tail="full", origin=None,
                                wrist_conversion="none"):
    """Build the robot embodiment's 128-D observation used during training.

    Every position is expressed relative to ``origin``, the session's opening
    head position, matching the converter's head-as-origin recentering.
    Rotations stay in the world frame, as they do in the dataset.

    Dims 100:128 follow the dex5 dataset layout: ``[100:103]`` root XYZ and
    ``[103:107]`` root quaternion in w,x,y,z order, both copied from the root
    already written into the first 100 dims; ``[107:126]`` joints 0..18 of the
    G1 29-DOF standard order (both legs, waist, left shoulder and elbow);
    ``[126:128]`` zero.  Everything here is the robot's measured state for this
    instant -- nothing is read from a recording.

    ``state_tail`` selects how much of that block is populated: ``full``,
    ``qpos-only`` (zero 100:107, keep the joints) or ``zero`` (mask 100:128).
    """
    if state_tail not in STATE_TAIL_MODES:
        raise ValueError(f"state_tail must be one of {STATE_TAIL_MODES}, got {state_tail!r}")
    if wrist_conversion not in WRIST_CONVERSION_MODES:
        raise ValueError(
            f"wrist_conversion must be one of {WRIST_CONVERSION_MODES}, got {wrist_conversion!r}"
        )
    _, _, validate_robot_state, _, _ = _install_protocol(PROTOCOL_ROOT)
    state = validate_robot_state(robot_state)
    out = np.zeros(C.ACTION_STATE_VEC_SIZE, dtype=np.float32)
    if origin is None:
        origin = np.zeros(3, dtype=np.float32)
    else:
        origin = np.asarray(origin, dtype=np.float32).reshape(3)

    def put_pose(indices, position_key, quaternion_key, wrist_side=None):
        position = np.asarray(state[position_key], dtype=np.float64)
        quaternion = np.asarray(state[quaternion_key], dtype=np.float64)
        if wrist_side is not None and wrist_conversion == "inspire_hand_base":
            position, quaternion = _g1_wrist_to_hand_base(position, quaternion, wrist_side)
        out[indices[:3]] = position.astype(np.float32) - origin
        out[indices[3:9]] = _quat_wxyz_to_rotation_6d(quaternion)

    put_pose(
        C.OUTPUT_HEAD_EEF, "head_pos_world", "head_quat_world_wxyz"
    )
    put_pose(
        C.OUTPUT_LEFT_EEF,
        "left_wrist_pos_world",
        "left_wrist_quat_world_wxyz",
        wrist_side="left",
    )
    put_pose(
        C.OUTPUT_RIGHT_EEF,
        "right_wrist_pos_world",
        "right_wrist_quat_world_wxyz",
        wrist_side="right",
    )
    put_pose(
        C.OUTPUT_WAIST, "root_pos_world", "root_quat_world_wxyz"
    )
    # Training stores [palm origin, thumb, index, middle, ring, pinky] in each
    # wrist-local hand block.  These are measured/reconstructed by ScaleBridge
    # from the current Inspire motor state.  Leaving them at the zero-filled
    # default puts real fingertips 9--12 standard deviations out of the dex5
    # training distribution and makes the otherwise state-driven policy emit
    # an asymmetric arm pose before the online controller has even started.
    out[C.OUTPUT_LEFT_KEYPOINTS] = np.asarray(
        state["left_hand_keypoints_local"], dtype=np.float32
    ).reshape(-1)
    out[C.OUTPUT_RIGHT_KEYPOINTS] = np.asarray(
        state["right_hand_keypoints_local"], dtype=np.float32
    ).reshape(-1)
    if state_tail == "zero":
        return out
    if state_tail == "full":
        # Same root as OUTPUT_WAIST, so whatever recentering the first 100 dims
        # carry is inherited instead of being recomputed here.
        out[100:103] = out[C.OUTPUT_WAIST[:3]]
        # q and -q encode the same physical rotation, but this raw quaternion is
        # consumed by a linear network input (unlike the sign-invariant 6D pose
        # blocks above).  The dex5 training converter canonicalized the nearly
        # upright root to positive w.  The real G1 IMU commonly publishes the
        # equivalent negative-w representative; passing that through verbatim
        # puts qpos[103] roughly 120 standard deviations outside training.
        root_quat = np.asarray(
            state["root_quat_world_wxyz"], dtype=np.float32
        ).copy()
        if root_quat[0] < 0.0:
            root_quat *= -1.0
        out[103:107] = root_quat
    out[107:126] = state["body_q19"]
    return out


def _load_stats(path, embodiment):
    with open(path, "rb") as stream:
        try:
            all_stats = pickle.load(stream)
        except ModuleNotFoundError as exc:
            # NumPy 2 pickles refer to numpy._core; the deployed HAT Python 3.8
            # environment currently carries NumPy 1.x under numpy.core.
            if not (exc.name or "").startswith("numpy._core"):
                raise
            sys.modules.setdefault("numpy._core", np.core)
            sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
            sys.modules.setdefault("numpy._core.numeric", np.core.numeric)
            # OpenCV 4.13 imports the NumPy 2 spelling of this binary module.
            # Alias it explicitly so NumPy 1.x does not initialize the same
            # extension a second time under a different module name.
            sys.modules.setdefault(
                "numpy._core._multiarray_umath", np.core._multiarray_umath
            )
            sys.modules.setdefault("numpy._core.umath", np.core.umath)
            stream.seek(0)
            all_stats = pickle.load(stream)
    stats = all_stats[embodiment] if embodiment in all_stats else all_stats
    required = ("qpos_mean", "qpos_std", "action_mean", "action_std")
    missing = [key for key in required if key not in stats]
    if missing:
        raise KeyError(f"Normalization stats missing {missing}: {path}")
    result = {key: np.asarray(stats[key], dtype=np.float32) for key in required}
    for key, value in result.items():
        if value.shape != (C.ACTION_STATE_VEC_SIZE,):
            raise ValueError(f"{key} must be shaped (128,), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or infinity")
    return result


def _load_state_dict(policy, state):
    """Load trained weights, tolerating training-only auxiliary heads.

    The Future-DINO world head is supervision-only: it exists in the checkpoint
    but never runs at inference.  Missing keys are still fatal -- those would be
    a genuinely wrong checkpoint, not an unused head.  ``str.removeprefix`` is
    3.9+, and the deployed HAT environment is Python 3.8.
    """
    cleaned = {}
    for key, value in state.items():
        cleaned[key[len("module."):] if key.startswith("module.") else key] = value
    result = policy.load_state_dict(cleaned, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"Checkpoint is missing required weights: {result.missing_keys}")
    if result.unexpected_keys:
        prefixes = sorted({key.split(".")[0] for key in result.unexpected_keys})
        print(
            f"[online_hat] ignoring {len(result.unexpected_keys)} training-only "
            f"weights from {prefixes}",
            flush=True,
        )
    return policy


def _parse_zero_state_dims(value):
    """``"98:128"`` or ``[98, 128]`` -> ``(98, 128)``; absent -> ``None``."""
    if value is None:
        return None
    if isinstance(value, str):
        lo, hi = value.split(":")
        return (int(lo), int(hi))
    return tuple(int(item) for item in value)


def _act_config(config, chunk_size):
    model = config["model"]
    common = config["common"]
    act_config = {
        "lr": 1e-4,
        "num_queries": chunk_size,
        "chunk_size": chunk_size,
        "kl_weight": model["kl_weight"],
        "hidden_dim": model["hidden_dim"],
        "dim_feedforward": model["dim_feedforward"],
        "lr_backbone": float(model["lr_backbone"]),
        "backbone": model["backbone"],
        "enc_layers": model["enc_layers"],
        "dec_layers": model["dec_layers"],
        "nheads": model["nheads"],
        "camera_names": common["camera_names"],
        "state_dim": common["state_dim"],
        "action_dim": common.get("action_dim", common["state_dim"]),
        "image_feature_strategy": model["image_feature_strategy"],
        "use_language_conditioning": model.get("use_language_conditioning", False),
        "query0_extra_weight": model.get("query0_extra_weight", 0.0),
    }
    # Optional state-block ablation, additive: an absent key leaves every
    # previously trained checkpoint untouched.  It has to be honoured here
    # because the mask lives in the checkpoint as a buffer, and the
    # ``strict=False`` load below would drop it without a word -- the model would
    # then be fed the full 128-D state it never saw during training, which reads
    # as a large regression rather than as the config mistake it is.
    zero_state_dims = _parse_zero_state_dims(model.get("zero_state_dims"))
    if zero_state_dims is not None:
        act_config["zero_state_dims"] = zero_state_dims
    return act_config


class OnlineHATPolicy:
    def __init__(self, *, config_path, checkpoint, stats_path, embodiment,
                 device="cuda", state_tail="full", wrist_conversion="none"):
        self.device = torch.device(device)
        if state_tail not in STATE_TAIL_MODES:
            raise ValueError(f"state_tail must be one of {STATE_TAIL_MODES}, got {state_tail!r}")
        if wrist_conversion not in WRIST_CONVERSION_MODES:
            raise ValueError(
                f"wrist_conversion must be one of {WRIST_CONVERSION_MODES}, got {wrist_conversion!r}"
            )
        self.state_tail = state_tail
        self.wrist_conversion = wrist_conversion
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.policy_class = str(self.config["common"]["policy_class"]).upper()
        self.chunk_size = int(self.config["common"].get("action_chunk_size", 64))
        self.camera_names = tuple(self.config["common"]["camera_names"])
        self.stats = _load_stats(stats_path, embodiment)
        self.policy, self.visual_preprocessor, self.is_torchscript = self._load_policy(
            Path(checkpoint)
        )
        empty_embedding = torch.load(
            ROOT / "hdt" / "empty_lang_embed.pt",
            map_location="cpu",
            weights_only=True,
        ).float()
        if empty_embedding.ndim == 2:
            empty_embedding = empty_embedding.unsqueeze(0)
        self.empty_language_embedding = empty_embedding.to(self.device)

    def _load_policy(self, checkpoint):
        visual_config = None
        if self.policy_class == "RDT":
            visual_config = {"visual_backbone": self.config["model"]["backbone"]}
        visual_encoder, preprocessor = make_visual_encoder(
            self.policy_class, visual_config
        )
        is_torchscript = False
        if self.policy_class == "ACT":
            try:
                policy = torch.jit.load(str(checkpoint), map_location=self.device)
                is_torchscript = True
            except RuntimeError:
                from hdt.policy import ACTPolicy
                policy = ACTPolicy(_act_config(self.config, self.chunk_size))
                state = torch.load(checkpoint, map_location="cpu", weights_only=False)
                if "model_state_dict" in state:
                    state = state["model_state_dict"]
                _load_state_dict(policy, state)
        elif self.policy_class == "RDT":
            from hdt.modeling.modeling_hdt import HumanDiffusionTransformer
            dataset = self.config["dataset"]
            model = self.config["model"]
            policy = HumanDiffusionTransformer(
                action_dim=self.config["common"]["state_dim"],
                pred_horizon=self.chunk_size,
                config=self.config,
                lang_token_dim=model["lang_token_dim"],
                img_token_dim=model["img_token_dim"],
                state_token_dim=model["state_token_dim"],
                max_lang_cond_len=dataset["tokenizer_max_length"],
                visual_encoder=visual_encoder,
                lang_pos_embed_config=[("lang", -dataset["tokenizer_max_length"])],
                dtype=torch.float32,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if "model_state_dict" in state:
                state = state["model_state_dict"]
            _load_state_dict(policy, state)
        else:
            raise ValueError(f"Online HAT supports ACT or RDT, got {self.policy_class}")
        return policy.eval().to(self.device), preprocessor, is_torchscript

    def prepare_policy_state(self, robot_state, origin=None):
        raw = robot_state_to_policy_state(
            robot_state, self.state_tail, origin, self.wrist_conversion
        )
        normalized = (
            raw - self.stats["qpos_mean"]
        ) / (self.stats["qpos_std"] + 1e-6)
        return raw.astype(np.float32), normalized.astype(np.float32)

    @torch.inference_mode()
    def infer(self, images, robot_state, origin=None, normalized_state=None):
        images = np.asarray(images)
        if (
            images.ndim != 4
            or images.shape[0] != len(self.camera_names)
            or images.shape[1] != 3
        ):
            raise ValueError(
                f"Images must be (camera,C,H,W) for {self.camera_names}, got {images.shape}"
            )
        image_tensor = self.visual_preprocessor(images).float().unsqueeze(0).to(self.device)
        if normalized_state is None:
            _, state = self.prepare_policy_state(robot_state, origin)
        else:
            state = np.asarray(normalized_state, dtype=np.float32)
            if state.shape != (C.ACTION_STATE_VEC_SIZE,):
                raise ValueError(
                    "normalized_state must be shaped "
                    f"({C.ACTION_STATE_VEC_SIZE},), got {state.shape}"
                )
        qpos = torch.from_numpy(state).unsqueeze(0).to(self.device)
        if self.is_torchscript:
            output = self.policy(image_tensor, qpos)
        else:
            conditioning = {}
            if (
                self.policy_class == "RDT"
                or self.config["model"].get("use_language_conditioning", False)
            ):
                conditioning = {
                    "language_embeddings": self.empty_language_embedding,
                    "language_embeddings_mask": torch.ones(
                        self.empty_language_embedding.shape[:2],
                        dtype=torch.bool,
                        device=self.device,
                    ),
                }
            output = self.policy(
                image_tensor, qpos, conditioning_dict=conditioning
            )
        if isinstance(output, (tuple, list)):
            output = output[0]
        output = output[0].detach().float().cpu().numpy()
        expected = (self.chunk_size, C.ACTION_STATE_VEC_SIZE)
        if output.shape != expected:
            raise ValueError(f"HAT output must be shaped {expected}, got {output.shape}")
        if not np.isfinite(output).all():
            raise ValueError("HAT output contains NaN or infinity")
        return output * self.stats["action_std"] + self.stats["action_mean"]


def _load_callback(spec):
    if ":" not in spec:
        raise ValueError("--image-source must be module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Image source is not callable: {spec}")
    return function


def main():
    global PROTOCOL_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--embodiment", default="h1_inspire")
    parser.add_argument("--image-source", required=True, help="module:function returning (camera,C,H,W)")
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--chunk-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--replan-frames", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--state-tail", choices=STATE_TAIL_MODES, default="full",
        help="how much of state[100:128] to populate: full, qpos-only (zero the "
             "root at 100:107), or zero (mask the whole block)",
    )
    parser.add_argument(
        "--origin-frames", type=int, default=10,
        help="how many opening frames to average into the head origin every "
             "position is measured from, matching the converter's "
             "head-as-origin recentering; 0 keeps raw world positions",
    )
    parser.add_argument(
        "--wrist-frame-conversion", choices=WRIST_CONVERSION_MODES, default="none",
        help="convert wrist poses between the G1 wrist_yaw_link frame published "
             "by the simulator and the Inspire hand-base ('raw') frame the dex5 "
             "dataset stores: G1->hand_base on the input state, hand_base->G1 on "
             "the output chunk; 'none' keeps the current behaviour",
    )
    parser.add_argument(
        "--record-dir", type=Path,
        help="exact directory for lossless HAT inference records; by default a "
             "timestamped directory is created under online_hat_records",
    )
    parser.add_argument(
        "--disable-recording", action="store_true",
        help="disable automatic lossless recording of HAT inputs and outputs",
    )
    parser.add_argument("--scalebridge-root", type=Path, default=DEFAULT_SCALEBRIDGE_ROOT)
    args = parser.parse_args()
    if args.fps <= 0 or args.replan_frames <= 0:
        parser.error("--fps and --replan-frames must be positive")
    if args.origin_frames < 0:
        parser.error("--origin-frames must not be negative")

    PROTOCOL_ROOT = args.scalebridge_root
    make_hat_chunk, monotonic_ns, validate_robot_state, LatestPublisher, LatestSubscriber = _install_protocol(args.scalebridge_root)
    policy = OnlineHATPolicy(
        config_path=args.config,
        checkpoint=args.checkpoint,
        stats_path=args.stats,
        embodiment=args.embodiment,
        device=args.device,
        state_tail=args.state_tail,
        wrist_conversion=args.wrist_frame_conversion,
    )
    print(f"[online_hat] wrist frame conversion: {args.wrist_frame_conversion}", flush=True)
    recorder = None
    if not args.disable_recording:
        recorder_metadata = {
            "config": str(args.config.resolve()),
            "checkpoint": str(args.checkpoint.resolve()),
            "stats": str(args.stats.resolve()),
            "embodiment": args.embodiment,
            "image_source": args.image_source,
            "camera_names": list(policy.camera_names),
            "state_tail": args.state_tail,
            "origin_frames": args.origin_frames,
            "wrist_frame_conversion": args.wrist_frame_conversion,
            "fps": args.fps,
            "replan_frames": args.replan_frames,
        }
        if args.record_dir is None:
            recorder = OnlineHATInputRecorder.create_default(
                ROOT / "online_hat_records", recorder_metadata
            )
        else:
            recorder = OnlineHATInputRecorder(args.record_dir, recorder_metadata)
        print(f"[online_hat] recording inputs to {recorder.run_dir}", flush=True)
    image_source = _load_callback(args.image_source)
    source_video_stop = None
    if recorder is not None:
        image_source_module = importlib.import_module(image_source.__module__)
        source_video_start = getattr(
            image_source_module, "start_video_recording", None
        )
        candidate_stop = getattr(
            image_source_module, "stop_video_recording", None
        )
        if callable(source_video_start) and callable(candidate_stop):
            source_video_path = (
                recorder.run_dir / "zed_source_30fps_h264.mp4"
            )
            source_video_start(source_video_path, fps=args.fps)
            source_video_stop = candidate_stop
            print(
                f"[online_hat] recording full source video to "
                f"{source_video_path}",
                flush=True,
            )
    state_sub = LatestSubscriber(args.state_endpoint, bind=False)
    chunk_pub = LatestPublisher(args.chunk_endpoint, bind=True)
    head_origin = HeadOrigin(args.origin_frames)
    sequence_id = 0
    next_replan_ns = 0
    replan_ns = int(args.replan_frames / args.fps * 1e9)

    def _handle_sigterm(signum, frame):
        print(
            "[online_hat] SIGTERM received; finalizing recordings ...",
            flush=True,
        )
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        while True:
            state = state_sub.receive_blocking(timeout_ms=1000)
            if state is None:
                continue
            state = validate_robot_state(state)
            origin = head_origin.observe(state["head_pos_world"])
            now = monotonic_ns()
            if now < next_replan_ns:
                continue
            images = image_source(camera_names=policy.camera_names, robot_state=state)
            image_captured_ns = monotonic_ns()
            policy_state_raw, policy_state_normalized = policy.prepare_policy_state(
                state, origin
            )
            inference_started_ns = monotonic_ns()
            action = policy.infer(
                images, state, origin, normalized_state=policy_state_normalized
            )
            inference_finished_ns = monotonic_ns()
            chunk_generated_at_ns = monotonic_ns()
            chunk = action_chunk_to_targets(
                action, state, args.fps, sequence_id, chunk_generated_at_ns,
                wrist_conversion=args.wrist_frame_conversion,
                position_origin_world=origin,
            )
            chunk_pub.send(chunk)
            if recorder is not None:
                recorder.record(
                    sequence_id=sequence_id,
                    source_state=state,
                    origin=origin,
                    images=images,
                    policy_state_raw=policy_state_raw,
                    policy_state_normalized=policy_state_normalized,
                    action=action,
                    image_captured_ns=image_captured_ns,
                    inference_started_ns=inference_started_ns,
                    inference_finished_ns=inference_finished_ns,
                    chunk_generated_at_ns=chunk_generated_at_ns,
                )
            sequence_id += 1
            next_replan_ns = now + replan_ns
    except KeyboardInterrupt:
        print("[online_hat] stopping; finalizing recordings ...", flush=True)
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm_handler)
        if source_video_stop is not None:
            source_video_stop()
        if recorder is not None:
            recorder.close()
        state_sub.close()
        chunk_pub.close()


if __name__ == "__main__":
    main()
