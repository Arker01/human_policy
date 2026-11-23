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
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

import hdt.constants as C
from hdt.action_representation import ACTION_REPRESENTATIONS, to_absolute_chunk
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
        chunk_generated_at_ns, heading_quat_wxyz=None, history_images=None,
        history_policy_state_raw=None, history_policy_state_normalized=None,
        history_source_states=None, history_captured_ns=None,
        action_anchor=None, action_anchor_info=None,
    ):
        stem = f"{int(sequence_id):06d}"
        image_path = self.images_dir / f"{stem}.npz"
        action_path = self.actions_dir / f"{stem}.npy"
        images = np.asarray(images)
        action = np.asarray(action, dtype=np.float32)
        image_payload = {"images": images}
        if history_images is not None:
            image_payload["history_images"] = np.asarray(history_images)
        self._atomic_save(
            image_path,
            lambda stream: np.savez_compressed(stream, **image_payload),
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
            "initial_heading_quat_wxyz": (
                None
                if heading_quat_wxyz is None
                else _jsonable(np.asarray(heading_quat_wxyz, dtype=np.float32))
            ),
            "policy_state_raw": _jsonable(
                np.asarray(policy_state_raw, dtype=np.float32)
            ),
            "policy_state_normalized": _jsonable(
                np.asarray(policy_state_normalized, dtype=np.float32)
            ),
        }
        if action_anchor is not None:
            record["action_anchor"] = _jsonable(
                np.asarray(action_anchor, dtype=np.float32)
            )
            record["action_anchor_info"] = _jsonable(action_anchor_info)
        if history_images is not None:
            record.update({
                "history_image_shape": list(np.asarray(history_images).shape),
                "history_source_state_sequence_ids": [
                    int(item["sequence_id"]) for item in history_source_states
                ],
                "history_captured_ns": [int(item) for item in history_captured_ns],
                "history_policy_state_raw": _jsonable(
                    np.asarray(history_policy_state_raw, dtype=np.float32)
                ),
                "history_policy_state_normalized": _jsonable(
                    np.asarray(history_policy_state_normalized, dtype=np.float32)
                ),
            })
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
HEADING_FRAME_MODES = ("initial", "world")

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


def _quat_conjugate_wxyz(quat):
    return np.asarray(quat, dtype=np.float64) * np.array([1.0, -1.0, -1.0, -1.0])


def _heading_forward_xy(quat_wxyz):
    """Where the body's local +x points in the world xy plane.

    Sign-invariant in the quaternion (q and -q rotate identically), unlike
    reading the yaw angle off the components, which matters because the live
    root stream carries w < 0.
    """
    forward = _quat_rotate_wxyz(np.asarray(quat_wxyz, dtype=np.float64), [1.0, 0.0, 0.0])
    xy = forward[..., :2]
    norm = np.linalg.norm(xy, axis=-1)
    if np.any(norm < 1e-6):
        raise ValueError("Root +x axis is vertical; heading is undefined")
    return xy / norm[..., None]


def _heading_quat_wxyz(forward_xy):
    """Yaw-only quaternion whose +x axis matches ``forward_xy``.

    Same construction as ScaleBridge's ``quat_heading`` so both ends of the
    loop agree on what "heading" means.
    """
    half = 0.5 * np.arctan2(forward_xy[1], forward_xy[0])
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)], dtype=np.float64)


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
                            wrist_conversion="none", position_origin_world=None,
                            action_representation="original"):
    """Extract only the HAT fields consumed by online ScaleBFM."""
    if wrist_conversion not in WRIST_CONVERSION_MODES:
        raise ValueError(
            f"wrist_conversion must be one of {WRIST_CONVERSION_MODES}, got {wrist_conversion!r}"
        )
    if action_representation not in ACTION_REPRESENTATIONS:
        raise ValueError(
            f"action_representation must be one of {ACTION_REPRESENTATIONS}, "
            f"got {action_representation!r}"
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
        action_representation=action_representation,
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
    """The per-episode frame every dex5 world channel is measured in.

    The converter subtracts the head world position of an episode's opening
    frames from head, both wrists and the waist.  The trained statistics show
    it plainly: head z averages -0.08, waist -0.47 and the wrists -0.55, i.e.
    metres below a point at standing head height.  Feeding raw world positions
    puts z more than ten standard deviations out of distribution, so the same
    offset has to be measured online.

    The training HDF5s additionally declare
    ``coordinate_convention = initial_heading_frame_x_forward_y_left_z_up``:
    every episode's root yaw starts at ~0 degrees.  The Vive/IMU world yaw is
    arbitrary (2026-09-07 runs sat at -37..-43 degrees throughout), which put
    the root rotation dims 3-5 sigma out of distribution.  So the opening
    frames also lock a yaw-only heading quaternion, measured from the root the
    same way ScaleBridge's ``quat_heading`` does, and every world pose is
    expressed in that initial-heading frame:

        p_hat = R_yaw0^-1 (p_world - head_origin)
        R_hat = R_yaw0^-1 R_world

    The output side needs no inverse of its own: ScaleBridge's protocol
    alignment anchors each accepted chunk by
    ``heading(measured root) * heading(predicted frame0)^-1`` about the
    predicted frame-0 root.  With the input transform in place the predicted
    frame-0 heading is the measured heading minus yaw0, so that anchor *is*
    R_yaw0 (plus the model's own frame-0 heading error, which anchoring to the
    measured root absorbs by construction).
    """

    def __init__(self, frames):
        self.frames = int(frames)
        self._samples = []
        self._forward_samples = []
        self._origin = np.zeros(3, dtype=np.float32)
        self._heading = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    @property
    def locked(self):
        return len(self._samples) >= self.frames

    def observe(self, head_pos_world, root_quat_world_wxyz):
        """Accumulate opening frames; return the (origin, heading) to use now."""
        if not self.locked:
            self._samples.append(np.asarray(head_pos_world, dtype=np.float64))
            self._origin = np.mean(self._samples, axis=0).astype(np.float32)
            # Averaging unit xy directions instead of yaw angles keeps the
            # mean well-defined across the +/-180 degree wrap.
            self._forward_samples.append(_heading_forward_xy(root_quat_world_wxyz))
            mean_forward = np.mean(self._forward_samples, axis=0)
            norm = np.linalg.norm(mean_forward)
            if norm < 1e-6:
                raise ValueError(
                    "Opening root headings cancel out; cannot define the "
                    "initial-heading frame"
                )
            self._heading = _heading_quat_wxyz(mean_forward / norm)
            if self.locked:
                yaw_deg = float(
                    np.degrees(2.0 * np.arctan2(self._heading[3], self._heading[0]))
                )
                print(
                    f"[online_hat] head origin locked to {self._origin.tolist()}, "
                    f"initial heading {yaw_deg:+.1f} deg, "
                    f"after {self.frames} frames",
                    flush=True,
                )
        return self._origin, self._heading


def robot_state_to_policy_state(robot_state, state_tail="full", origin=None,
                                wrist_conversion="none", root_quat_mean_w=None,
                                heading_quat_wxyz=None):
    """Build the robot embodiment's 128-D observation used during training.

    Every position is expressed relative to ``origin``, the session's opening
    head position, matching the converter's head-as-origin recentering.

    ``heading_quat_wxyz`` is the session's opening yaw-only root heading
    (see ``HeadOrigin``).  When given, every world position and orientation --
    including the raw root quaternion at 103:107 -- is rotated into that
    initial-heading frame, matching the training HDF5s'
    ``coordinate_convention = initial_heading_frame_x_forward_y_left_z_up``.
    Wrist-local finger keypoints and joint angles are frame-local and stay
    untouched.  ``None`` keeps the previous behaviour of leaving rotations in
    the Vive/IMU world frame.

    Dims 100:128 follow the dex5 dataset layout: ``[100:103]`` root XYZ and
    ``[103:107]`` root quaternion in w,x,y,z order, both copied from the root
    already written into the first 100 dims; ``[107:126]`` joints 0..18 of the
    G1 29-DOF standard order (both legs, waist, left shoulder and elbow);
    ``[126:128]`` zero.  Everything here is the robot's measured state for this
    instant -- nothing is read from a recording.

    ``state_tail`` selects how much of that block is populated: ``full``,
    ``qpos-only`` (zero 100:107, keep the joints) or ``zero`` (mask 100:128).

    ``root_quat_mean_w`` is the checkpoint's ``qpos_mean[103]``; see the
    canonicalization comment on dim 103 below.  ``None`` passes the measured
    quaternion through verbatim, which is only correct when the caller has
    already checked which representative the checkpoint was trained on.
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
    heading_inv = None
    if heading_quat_wxyz is not None:
        heading_inv = _quat_conjugate_wxyz(
            np.asarray(heading_quat_wxyz, dtype=np.float64).reshape(4)
        )

    def to_initial_heading(position, quaternion):
        position = position - origin.astype(np.float64)
        if heading_inv is None:
            return position, quaternion
        return (
            _quat_rotate_wxyz(heading_inv, position),
            _quat_mul_wxyz(heading_inv, quaternion),
        )

    def put_pose(indices, position_key, quaternion_key, wrist_side=None):
        position = np.asarray(state[position_key], dtype=np.float64)
        quaternion = np.asarray(state[quaternion_key], dtype=np.float64)
        if wrist_side is not None and wrist_conversion == "inspire_hand_base":
            position, quaternion = _g1_wrist_to_hand_base(position, quaternion, wrist_side)
        position, quaternion = to_initial_heading(position, quaternion)
        out[indices[:3]] = position.astype(np.float32)
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
        # blocks above).  Which of the two representatives a checkpoint saw is a
        # property of its training set, not something to assume: b10/c30 all
        # report qpos_mean[103] = -0.99, so forcing w >= 0 here fed +0.99 into a
        # dim whose std is 0.014, i.e. +137 sigma, which saturated the state
        # encoder and collapsed the c30 models onto their action mean.  Align to
        # the checkpoint's own mean instead.
        root_quat = np.asarray(
            state["root_quat_world_wxyz"], dtype=np.float64
        )
        if heading_inv is not None:
            root_quat = _quat_mul_wxyz(heading_inv, root_quat)
        root_quat = root_quat.astype(np.float32)
        if root_quat_mean_w is not None and root_quat[0] * float(root_quat_mean_w) < 0.0:
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
    for key in required:
        value = result[key]
        if value.shape != (C.ACTION_STATE_VEC_SIZE,):
            raise ValueError(f"{key} must be shaped (128,), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{key} contains NaN or infinity")
    representation = str(stats.get("action_representation", "original"))
    if representation not in ACTION_REPRESENTATIONS:
        raise ValueError(
            f"Unknown action_representation={representation!r} in {path}")
    result["action_representation"] = representation
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


def _parse_history_config(config):
    history = config.get("model", {}).get("history", {}) or {}
    if not history.get("enabled", False):
        return None
    offsets = tuple(int(item) for item in history.get("offsets", ()))
    if not offsets:
        raise ValueError("model.history.enabled=true requires non-empty offsets")
    if any(item <= 0 for item in offsets):
        raise ValueError(f"History offsets must be positive, got {offsets}")
    if len(set(offsets)) != len(offsets):
        raise ValueError(f"History offsets must be unique, got {offsets}")
    token_pool = int(history.get("token_pool", 2))
    if token_pool <= 0:
        raise ValueError(f"model.history.token_pool must be positive, got {token_pool}")
    return {"offsets": list(offsets), "token_pool": token_pool}


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
    history_config = _parse_history_config(config)
    if history_config is not None:
        act_config["history_config"] = history_config
    return act_config


class OnlineHATPolicy:
    def __init__(self, *, config_path, checkpoint, stats_path, embodiment,
                 device="cuda", state_tail="full", wrist_conversion="none",
                 state_ood_limit=6.0):
        self.device = torch.device(device)
        if state_tail not in STATE_TAIL_MODES:
            raise ValueError(f"state_tail must be one of {STATE_TAIL_MODES}, got {state_tail!r}")
        if wrist_conversion not in WRIST_CONVERSION_MODES:
            raise ValueError(
                f"wrist_conversion must be one of {WRIST_CONVERSION_MODES}, got {wrist_conversion!r}"
            )
        self.state_tail = state_tail
        self.wrist_conversion = wrist_conversion
        self.state_ood_limit = float(state_ood_limit)
        self._state_ood_checked = False
        self._state_ood_warned_at = 0.0
        with open(config_path, "r", encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self.policy_class = str(self.config["common"]["policy_class"]).upper()
        self.chunk_size = int(self.config["common"].get("action_chunk_size", 64))
        # The rate the chunk's frames were sampled at during training.  Both
        # sample_chunk() and _latency_start_frame() on the ScaleBridge side key
        # off the fps carried in the chunk, so publishing the wrong one plays
        # the whole reference at the wrong speed (a 30 Hz checkpoint served at
        # --fps 10 runs the reference at a third of the intended rate).
        chunk_fps = self.config["common"].get("chunk_fps")
        self.chunk_fps = None if chunk_fps is None else float(chunk_fps)
        self.camera_names = tuple(self.config["common"]["camera_names"])
        self.history_config = _parse_history_config(self.config)
        self.history_offsets = tuple(
            () if self.history_config is None else self.history_config["offsets"]
        )
        history_section = self.config.get("model", {}).get("history", {}) or {}
        self.history_fps = float(history_section.get("fps", 30.0))
        if self.history_offsets and self.history_fps <= 0.0:
            raise ValueError(
                f"model.history.fps must be positive, got {self.history_fps}"
            )
        if self.history_offsets and self.policy_class != "ACT":
            raise ValueError("Online history observations are currently supported only for ACT")
        self.zero_state_dims = _parse_zero_state_dims(
            self.config.get("model", {}).get("zero_state_dims")
        )
        self.stats = _load_stats(stats_path, embodiment)
        self.action_representation = self.stats["action_representation"]
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
            if is_torchscript and self.history_offsets:
                raise ValueError(
                    "History ACT checkpoints must be loaded as eager state dicts; "
                    "the TorchScript online signature has no history inputs"
                )
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

    def prepare_policy_state(self, robot_state, origin=None, heading_quat_wxyz=None):
        raw = robot_state_to_policy_state(
            robot_state,
            self.state_tail,
            origin,
            self.wrist_conversion,
            root_quat_mean_w=float(self.stats["qpos_mean"][103]),
            heading_quat_wxyz=heading_quat_wxyz,
        )
        normalized = (
            raw - self.stats["qpos_mean"]
        ) / (self.stats["qpos_std"] + 1e-6)
        self._check_state_in_distribution(raw, normalized)
        return raw.astype(np.float32), normalized.astype(np.float32)

    def _describe_state_dim(self, raw, masked, dim):
        return (
            f"dim {dim} {abs(float(masked[dim])):.1f} sigma "
            f"(raw {raw[dim]:+.4f} vs train {self.stats['qpos_mean'][dim]:+.4f}"
            f" +/- {self.stats['qpos_std'][dim]:.4f})"
        )

    def _check_state_in_distribution(self, raw, normalized):
        """Refuse to start on a state the checkpoint's encoder cannot read.

        A single dim far outside training saturates the state encoder and the
        policy falls back on its action mean, which on a real robot looks like
        a plausible but completely open-loop motion rather than a crash -- the
        2026-09-06 runs walked to a memorized world pose with qpos[103] at +137
        sigma.  The first state therefore hard-fails; later ones only warn,
        because raising mid-episode would drop the reference while the robot is
        moving.

        The worst dims are always printed on the first state, not only on
        failure: the online hand keypoints currently sit around 8 sigma, well
        below the saturating class but still worth seeing every run.
        """
        masked = normalized.copy()
        if self.zero_state_dims is not None:
            masked[self.zero_state_dims[0]:self.zero_state_dims[1]] = 0.0
        ranked = np.argsort(np.abs(masked))[::-1]
        worst = int(ranked[0])
        deviation = float(abs(masked[worst]))
        first_check = not self._state_ood_checked
        self._state_ood_checked = True
        if first_check:
            summary = ", ".join(
                self._describe_state_dim(raw, masked, int(dim)) for dim in ranked[:3]
            )
            print(f"[online_hat] state vs training, worst dims: {summary}", flush=True)
        if deviation <= self.state_ood_limit:
            return
        message = (
            f"policy state {self._describe_state_dim(raw, masked, worst)} exceeds "
            f"--state-ood-limit {self.state_ood_limit:.1f}"
        )
        if first_check:
            raise RuntimeError(
                message + ". The checkpoint cannot consume this observation; fix "
                "the conversion or raise --state-ood-limit to override."
            )
        now = time.monotonic()
        if now - self._state_ood_warned_at > 5.0:
            self._state_ood_warned_at = now
            print(f"[online_hat] WARNING: {message}", flush=True)

    @torch.inference_mode()
    def infer(self, images, robot_state, origin=None, normalized_state=None,
              heading_quat_wxyz=None, history_images=None,
              normalized_history_states=None, action_anchor=None):
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
            _, state = self.prepare_policy_state(robot_state, origin, heading_quat_wxyz)
        else:
            state = np.asarray(normalized_state, dtype=np.float32)
            if state.shape != (C.ACTION_STATE_VEC_SIZE,):
                raise ValueError(
                    "normalized_state must be shaped "
                    f"({C.ACTION_STATE_VEC_SIZE},), got {state.shape}"
                )
        qpos = torch.from_numpy(state).unsqueeze(0).to(self.device)
        history_count = len(self.history_offsets)
        history_conditioning = {}
        if history_count:
            history_images = np.asarray(history_images)
            expected_image_prefix = (
                history_count, len(self.camera_names), 3
            )
            if (
                history_images.ndim != 5
                or history_images.shape[:3] != expected_image_prefix
            ):
                raise ValueError(
                    "history_images must be shaped "
                    f"({history_count},camera,C,H,W), got {history_images.shape}"
                )
            history_states = np.asarray(
                normalized_history_states, dtype=np.float32
            )
            expected_state_shape = (history_count, C.ACTION_STATE_VEC_SIZE)
            if history_states.shape != expected_state_shape:
                raise ValueError(
                    "normalized_history_states must be shaped "
                    f"{expected_state_shape}, got {history_states.shape}"
                )
            flat_history_images = history_images.reshape(
                history_count * len(self.camera_names),
                *history_images.shape[2:],
            )
            processed_history = self.visual_preprocessor(
                flat_history_images
            ).float()
            processed_history = processed_history.reshape(
                history_count,
                len(self.camera_names),
                *processed_history.shape[1:],
            )
            history_conditioning = {
                "hist_image": processed_history.unsqueeze(0).to(self.device),
                "hist_qpos": torch.from_numpy(history_states)
                .unsqueeze(0)
                .to(self.device),
            }
        elif history_images is not None or normalized_history_states is not None:
            raise ValueError("History inputs were provided to a non-history checkpoint")
        if self.is_torchscript:
            output = self.policy(image_tensor, qpos)
        else:
            conditioning = dict(history_conditioning)
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
        output = output * self.stats["action_std"] + self.stats["action_mean"]
        if getattr(self, "action_representation", "original") == "relative_chunk":
            if action_anchor is None:
                action_anchor, _ = self.prepare_policy_state(
                    robot_state, origin, heading_quat_wxyz
                )
            output = to_absolute_chunk(
                np.asarray(action_anchor, dtype=np.float32), output)
        return output


def _load_callback(spec):
    if ":" not in spec:
        raise ValueError("--image-source must be module:function")
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    if not callable(function):
        raise TypeError(f"Image source is not callable: {spec}")
    return function


class RecoveringImageSource:
    """Skip inference during declared transient camera errors, without stale reuse."""

    def __init__(self, callback, recoverable_errors):
        self.callback = callback
        self.recoverable_errors = recoverable_errors
        self.failed_at = None
        self.last_warning = float("-inf")

    def __call__(self, **kwargs):
        try:
            images = self.callback(**kwargs)
        except self.recoverable_errors as exc:
            now = time.monotonic()
            if self.failed_at is None:
                self.failed_at = now
            if now - self.last_warning >= 5.0:
                print(f"[online_hat] Camera unavailable: {exc}. "
                      "Pausing chunk publication; waiting for fresh images.", flush=True)
                self.last_warning = now
            return None
        if self.failed_at is not None:
            print(f"[online_hat] Camera recovered after "
                  f"{time.monotonic() - self.failed_at:.2f}s; resuming inference. "
                  "If ScaleBridge latched hold, press R1 after recovery.", flush=True)
            self.failed_at = None
            self.last_warning = float("-inf")
        return images


class TimedObservationHistory:
    """Time-indexed camera/state pairs used by history-enabled checkpoints."""

    def __init__(self, offsets, fps):
        self.offsets = tuple(int(item) for item in offsets)
        self.fps = float(fps)
        if self.fps <= 0.0:
            raise ValueError(f"History fps must be positive, got {self.fps}")
        if any(item <= 0 for item in self.offsets):
            raise ValueError(f"History offsets must be positive, got {self.offsets}")
        self._samples = deque()
        self._next_id = 0
        self._retention_ns = int(
            (max(self.offsets, default=0) / self.fps + 1.0) * 1e9
        )

    def append(self, captured_ns, state, images):
        captured_ns = int(captured_ns)
        if self._samples and captured_ns < self._samples[-1]["captured_ns"]:
            raise ValueError("Observation timestamps must be monotonic")
        sample = {
            "id": self._next_id,
            "captured_ns": captured_ns,
            "state": state,
            "images": np.asarray(images).copy(),
        }
        self._next_id += 1
        self._samples.append(sample)
        cutoff = captured_ns - self._retention_ns
        while len(self._samples) > 1 and self._samples[1]["captured_ns"] < cutoff:
            self._samples.popleft()
        return sample

    def pending_after(self, sample_id):
        return [item for item in self._samples if item["id"] > sample_id]

    def select(self, current):
        """Select configured past slots, clamping startup to the first sample."""
        eligible = [
            item for item in self._samples if item["id"] <= current["id"]
        ]
        if not eligible:
            raise RuntimeError("Current observation is no longer in history")
        selected = []
        for offset in self.offsets:
            target_ns = current["captured_ns"] - int(offset / self.fps * 1e9)
            selected.append(
                min(eligible, key=lambda item: abs(item["captured_ns"] - target_ns))
            )
        return selected


class BackgroundObservationSource:
    """Continuously pair robot states with camera frames during model inference."""

    def __init__(self, *, subscriber_factory, endpoint, validator, image_source,
                 camera_names, history_offsets, history_fps, monotonic_ns):
        self.subscriber_factory = subscriber_factory
        self.endpoint = endpoint
        self.validator = validator
        self.image_source = image_source
        self.camera_names = tuple(camera_names)
        self.monotonic_ns = monotonic_ns
        self.history = TimedObservationHistory(history_offsets, history_fps)
        self._condition = threading.Condition()
        self._stop = threading.Event()
        self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="online-hat-observation-history",
            daemon=True,
        )
        self._thread.start()

    def _run(self):
        subscriber = None
        try:
            # The ZMQ socket is created and used in this worker only.
            subscriber = self.subscriber_factory(self.endpoint, bind=False)
            while not self._stop.is_set():
                state = subscriber.receive_blocking(timeout_ms=100)
                if state is None:
                    continue
                state = self.validator(state)
                images = self.image_source(
                    camera_names=self.camera_names, robot_state=state
                )
                if images is None:
                    continue
                with self._condition:
                    self.history.append(self.monotonic_ns(), state, images)
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._condition.notify_all()
        finally:
            if subscriber is not None:
                subscriber.close()

    def wait_for_new(self, after_id, timeout_s=1.0):
        deadline = time.monotonic() + float(timeout_s)
        with self._condition:
            while True:
                pending = self.history.pending_after(after_id)
                if pending:
                    current = pending[-1]
                    return pending, current, self.history.select(current)
                if self._error is not None:
                    raise RuntimeError(
                        f"Background observation source failed: {self._error}"
                    ) from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return None
                self._condition.wait(remaining)

    def close(self):
        self._stop.set()
        with self._condition:
            self._condition.notify_all()
        self._thread.join(timeout=2.0)


SCHEDULED_ANCHOR_POSITION_BLOCKS = (
    ("head", C.OUTPUT_HEAD_EEF[:3]),
    ("left_wrist", C.OUTPUT_LEFT_EEF[:3]),
    ("right_wrist", C.OUTPUT_RIGHT_EEF[:3]),
    ("waist", C.OUTPUT_WAIST[:3]),
)


class ScheduledAnchorPlanner:
    """Choose HuMI's inverse anchor from the previous absolute plan.

    Relative chunk row zero represents ``t+1``. The next observation can be
    early or late because camera and inference latency vary, so its anchor is
    selected from the previous chunk using RobotState timestamps rather than
    a fixed ``replan_frames - 1`` row. Missing or out-of-horizon timing falls
    back to the measured pose instead of extrapolating a stale plan.

    A zero ``max_position_error_m`` leaves the real-robot threshold disabled
    until it has been tuned. A positive value compares head, both wrists and
    waist. Neck is excluded because the online state does not populate that
    training-only block, and wrist-local finger points are not world tracking
    errors. Crossing the limit re-anchors this replan to actual state.
    """

    def __init__(self, fps, max_position_error_m=0.0):
        self.fps = float(fps)
        self.max_position_error_m = float(max_position_error_m)
        if self.fps <= 0.0:
            raise ValueError(f"Scheduled-anchor fps must be positive, got {self.fps}")
        if self.max_position_error_m < 0.0:
            raise ValueError(
                "scheduled-anchor max position error must be non-negative"
            )
        self.reset()

    def reset(self):
        self.previous_action = None
        self.previous_source_timestamp_ns = None

    @staticmethod
    def _state_row(value, name):
        value = np.asarray(value, dtype=np.float32)
        if value.shape != (C.ACTION_STATE_VEC_SIZE,):
            raise ValueError(f"{name} must be shaped (128,), got {value.shape}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} contains NaN or infinity")
        return value

    @staticmethod
    def _actual(actual, mode, **details):
        return np.array(actual, copy=True), {"mode": mode, **details}

    def select(self, actual, source_timestamp_ns):
        actual = self._state_row(actual, "actual anchor")
        source_timestamp_ns = int(source_timestamp_ns)
        if self.previous_action is None:
            return self._actual(actual, "actual_first")

        elapsed_frames = (
            source_timestamp_ns - self.previous_source_timestamp_ns
        ) * 1e-9 * self.fps
        anchor_row = int(np.floor(elapsed_frames + 0.5)) - 1
        timing = {
            "elapsed_frames": float(elapsed_frames),
            "scheduled_row": int(anchor_row),
            "previous_source_timestamp_ns": int(
                self.previous_source_timestamp_ns
            ),
        }
        if anchor_row < 0 or anchor_row >= len(self.previous_action):
            return self._actual(actual, "actual_timing_reanchor", **timing)

        scheduled = self._state_row(
            self.previous_action[anchor_row], "scheduled anchor"
        )
        errors = {
            name: float(np.linalg.norm(scheduled[indices] - actual[indices]))
            for name, indices in SCHEDULED_ANCHOR_POSITION_BLOCKS
        }
        maximum = max(errors.values(), default=0.0)
        details = {
            **timing,
            "position_errors_m": errors,
            "max_position_error_m": float(maximum),
        }
        if self.max_position_error_m > 0.0 and maximum > self.max_position_error_m:
            details["limit_m"] = self.max_position_error_m
            return self._actual(actual, "actual_safety_reanchor", **details)
        return np.array(scheduled, copy=True), {"mode": "scheduled", **details}

    def update(self, absolute_action, source_timestamp_ns):
        action = np.asarray(absolute_action, dtype=np.float32)
        if action.ndim != 2 or action.shape[1:] != (C.ACTION_STATE_VEC_SIZE,):
            raise ValueError(
                f"absolute scheduled action must be shaped (N,128), got {action.shape}"
            )
        if len(action) == 0:
            raise ValueError("absolute scheduled action must contain at least one row")
        if not np.isfinite(action).all():
            raise ValueError("absolute scheduled action contains NaN or infinity")
        self.previous_action = np.array(action, copy=True)
        self.previous_source_timestamp_ns = int(source_timestamp_ns)


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
    parser.add_argument(
        "--fps", type=float, default=None,
        help="chunk playback rate; defaults to the checkpoint config's "
             "common.chunk_fps and must match it unless --allow-fps-mismatch",
    )
    parser.add_argument(
        "--allow-fps-mismatch", action="store_true",
        help="serve chunks at --fps even when the checkpoint was trained at a "
             "different rate",
    )
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
        "--heading-frame", choices=HEADING_FRAME_MODES, default="initial",
        help="'initial' (default) rotates every world pose into the frame of "
             "the opening root heading averaged over --origin-frames, matching "
             "the training data's initial_heading_frame convention; 'world' "
             "keeps the pre-2026-09-08 behaviour of Vive/IMU-frame "
             "rotations, which sat 3-5 sigma outside training",
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
    parser.add_argument(
        "--state-ood-limit", type=float, default=10.0,
        help="refuse to start when any policy state dim is further than this "
             "many standard deviations from the checkpoint's training mean. The "
             "default clears the online hand keypoints (~8 sigma today) while "
             "still catching the encoder-saturating class; the worst dims are "
             "printed on startup either way",
    )
    parser.add_argument(
        "--scheduled-anchor-max-error-m", type=float, default=0.0,
        help="relative_chunk only: re-anchor to the measured pose when the "
             "largest head/wrist/waist position error against the scheduled "
             "anchor exceeds this distance; 0 disables the untuned threshold",
    )
    parser.add_argument("--scalebridge-root", type=Path, default=DEFAULT_SCALEBRIDGE_ROOT)
    args = parser.parse_args()
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be positive")
    if args.replan_frames <= 0:
        parser.error("--replan-frames must be positive")
    if args.origin_frames < 0:
        parser.error("--origin-frames must not be negative")
    if args.scheduled_anchor_max_error_m < 0.0:
        parser.error("--scheduled-anchor-max-error-m must not be negative")

    # Resolved before the checkpoint is loaded so a mismatch fails immediately
    # rather than after a minute of model loading.
    with open(args.config, "r", encoding="utf-8") as stream:
        checkpoint_fps = yaml.safe_load(stream)["common"].get("chunk_fps")
    checkpoint_fps = None if checkpoint_fps is None else float(checkpoint_fps)
    if checkpoint_fps is None:
        if args.fps is None:
            parser.error(
                f"{args.config} has no common.chunk_fps, so --fps must be given "
                "explicitly (it is the rate the checkpoint's chunk frames were "
                "sampled at, not a display rate)"
            )
    elif args.fps is None:
        args.fps = checkpoint_fps
    elif args.fps != checkpoint_fps and not args.allow_fps_mismatch:
        parser.error(
            f"--fps {args.fps:g} contradicts the checkpoint's training rate "
            f"{checkpoint_fps:g} Hz; the reference would play at "
            f"{args.fps / checkpoint_fps:.2f}x speed. Drop --fps to use the "
            "checkpoint's rate, or pass --allow-fps-mismatch on purpose."
        )
    print(
        f"[online_hat] chunk fps: {args.fps:g}"
        + (
            ""
            if checkpoint_fps is None or args.fps == checkpoint_fps
            else f" (OVERRIDING checkpoint's {checkpoint_fps:g} Hz)"
        ),
        flush=True,
    )
    print(f"[online_hat] wrist frame conversion: {args.wrist_frame_conversion}", flush=True)
    print(
        f"[online_hat] heading frame: {args.heading_frame}"
        + (
            " (poses rotated into the opening root heading; ScaleBridge's "
            "chunk heading anchor supplies the inverse on output)"
            if args.heading_frame == "initial"
            else ""
        ),
        flush=True,
    )

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
        state_ood_limit=args.state_ood_limit,
    )
    print(f"[online_hat] action representation: {policy.action_representation}", flush=True)
    if policy.action_representation == "relative_chunk":
        print(
            "[online_hat] scheduled-anchor safety: "
            + (
                f"re-anchor above {args.scheduled_anchor_max_error_m:.3f} m"
                if args.scheduled_anchor_max_error_m > 0.0
                else "distance threshold disabled; errors are still recorded"
            ),
            flush=True,
        )
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
            "heading_frame": args.heading_frame,
            "wrist_frame_conversion": args.wrist_frame_conversion,
            "fps": args.fps,
            "replan_frames": args.replan_frames,
            "history_offsets": list(policy.history_offsets),
            "history_fps": policy.history_fps,
            "action_representation": policy.action_representation,
            "scheduled_anchor_max_error_m": args.scheduled_anchor_max_error_m,
        }
        if args.record_dir is None:
            recorder = OnlineHATInputRecorder.create_default(
                ROOT / "online_hat_records", recorder_metadata
            )
        else:
            recorder = OnlineHATInputRecorder(args.record_dir, recorder_metadata)
        print(f"[online_hat] recording inputs to {recorder.run_dir}", flush=True)
    image_source = _load_callback(args.image_source)
    image_source_module = importlib.import_module(image_source.__module__)
    recovering_images = RecoveringImageSource(
        image_source, getattr(image_source_module, "RECOVERABLE_IMAGE_ERRORS", ())
    )
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
    chunk_pub = LatestPublisher(args.chunk_endpoint, bind=True)
    state_sub = None
    observation_source = None
    last_observation_id = -1
    if policy.history_offsets:
        observation_source = BackgroundObservationSource(
            subscriber_factory=LatestSubscriber,
            endpoint=args.state_endpoint,
            validator=validate_robot_state,
            image_source=recovering_images,
            camera_names=policy.camera_names,
            history_offsets=policy.history_offsets,
            history_fps=policy.history_fps,
            monotonic_ns=monotonic_ns,
        )
        print(
            f"[online_hat] observation history enabled: current + offsets "
            f"{list(policy.history_offsets)} at {policy.history_fps:g} Hz",
            flush=True,
        )
    else:
        state_sub = LatestSubscriber(args.state_endpoint, bind=False)
    head_origin = HeadOrigin(args.origin_frames)
    sequence_id = 0
    next_replan_ns = 0
    replan_ns = int(args.replan_frames / args.fps * 1e9)
    scheduled_anchor_planner = ScheduledAnchorPlanner(
        args.fps, args.scheduled_anchor_max_error_m
    )

    def _handle_sigterm(signum, frame):
        print(
            "[online_hat] SIGTERM received; finalizing recordings ...",
            flush=True,
        )
        raise KeyboardInterrupt

    previous_sigterm_handler = signal.signal(signal.SIGTERM, _handle_sigterm)
    try:
        while True:
            history_samples = []
            if observation_source is not None:
                observed = observation_source.wait_for_new(
                    last_observation_id, timeout_s=1.0
                )
                if observed is None:
                    continue
                pending, current, history_samples = observed
                last_observation_id = current["id"]
                for item in pending:
                    origin, opening_heading = head_origin.observe(
                        item["state"]["head_pos_world"],
                        item["state"]["root_quat_world_wxyz"],
                    )
                state = current["state"]
                images = current["images"]
                image_captured_ns = current["captured_ns"]
                now = image_captured_ns
            else:
                state = state_sub.receive_blocking(timeout_ms=1000)
                if state is None:
                    continue
                state = validate_robot_state(state)
                origin, opening_heading = head_origin.observe(
                    state["head_pos_world"], state["root_quat_world_wxyz"]
                )
                now = monotonic_ns()
                if now < next_replan_ns:
                    continue
                images = recovering_images(
                    camera_names=policy.camera_names, robot_state=state
                )
                if images is None:
                    next_replan_ns = monotonic_ns() + int(0.1 * 1e9)
                    continue
                image_captured_ns = monotonic_ns()
            heading = opening_heading if args.heading_frame == "initial" else None
            if now < next_replan_ns:
                continue
            policy_state_raw, policy_state_normalized = policy.prepare_policy_state(
                state, origin, heading
            )
            history_images = None
            history_state_raw = None
            history_state_normalized = None
            if history_samples:
                history_images = np.stack(
                    [item["images"] for item in history_samples], axis=0
                )
                history_state_pairs = [
                    policy.prepare_policy_state(item["state"], origin, heading)
                    for item in history_samples
                ]
                history_state_raw = np.stack(
                    [item[0] for item in history_state_pairs], axis=0
                )
                history_state_normalized = np.stack(
                    [item[1] for item in history_state_pairs], axis=0
                )
            inference_started_ns = monotonic_ns()
            action_anchor = None
            action_anchor_info = None
            if policy.action_representation == "relative_chunk":
                if not head_origin.locked:
                    # Never carry a scheduled row across a moving origin or
                    # opening-heading estimate.
                    scheduled_anchor_planner.reset()
                action_anchor, action_anchor_info = scheduled_anchor_planner.select(
                    policy_state_raw, state["timestamp_ns"]
                )
                if action_anchor_info["mode"] in (
                    "actual_timing_reanchor", "actual_safety_reanchor"
                ):
                    print(
                        f"[online_hat] HuMI {action_anchor_info['mode']}: "
                        f"{action_anchor_info}",
                        flush=True,
                    )
            action = policy.infer(
                images,
                state,
                origin,
                normalized_state=policy_state_normalized,
                heading_quat_wxyz=heading,
                history_images=history_images,
                normalized_history_states=history_state_normalized,
                action_anchor=action_anchor,
            )
            if policy.action_representation == "relative_chunk":
                if head_origin.locked:
                    scheduled_anchor_planner.update(action, state["timestamp_ns"])
                else:
                    scheduled_anchor_planner.reset()
            inference_finished_ns = monotonic_ns()
            chunk_generated_at_ns = monotonic_ns()
            chunk = action_chunk_to_targets(
                action, state, args.fps, sequence_id, chunk_generated_at_ns,
                wrist_conversion=args.wrist_frame_conversion,
                position_origin_world=origin,
                action_representation=policy.action_representation,
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
                    heading_quat_wxyz=heading,
                    history_images=history_images,
                    history_policy_state_raw=history_state_raw,
                    history_policy_state_normalized=history_state_normalized,
                    history_source_states=[
                        item["state"] for item in history_samples
                    ],
                    history_captured_ns=[
                        item["captured_ns"] for item in history_samples
                    ],
                    action_anchor=action_anchor,
                    action_anchor_info=action_anchor_info,
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
        if observation_source is not None:
            observation_source.close()
        if state_sub is not None:
            state_sub.close()
        chunk_pub.close()


if __name__ == "__main__":
    main()
