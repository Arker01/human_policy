"""Action-space conversions shared by training, evaluation, and deployment.

``relative_chunk`` follows the representation described in ``HuMI.md``: every
future pose/keypoint in a chunk is expressed relative to the observation at the
chunk anchor.  Translation and point channels use a simple difference in the
dataset coordinate frame; rotation channels use ``R_anchor.T @ R_future``.

The remaining robot-configuration tail is deliberately kept absolute.  It is
not part of the HAT keypoint pose and keeping it unchanged preserves the
existing 128-D model/output contract.
"""

from __future__ import annotations

import numpy as np

from hdt.constants import (
    ACTION_STATE_VEC_SIZE,
    OUTPUT_HEAD_EEF,
    OUTPUT_LEFT_EEF,
    OUTPUT_LEFT_KEYPOINTS,
    OUTPUT_NECK,
    OUTPUT_RIGHT_EEF,
    OUTPUT_RIGHT_KEYPOINTS,
    OUTPUT_WAIST,
)


ACTION_REPRESENTATIONS = ("original", "relative_chunk")
POSE_BLOCKS = (
    OUTPUT_HEAD_EEF,
    OUTPUT_RIGHT_EEF,
    OUTPUT_NECK,
    OUTPUT_LEFT_EEF,
    OUTPUT_WAIST,
)
POINT_BLOCKS = (OUTPUT_LEFT_KEYPOINTS, OUTPUT_RIGHT_KEYPOINTS)


def _check_128(name: str, value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.shape[-1] != ACTION_STATE_VEC_SIZE:
        raise ValueError(
            f"{name} must end in {ACTION_STATE_VEC_SIZE} dimensions, got {value.shape}"
        )
    return value


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """NumPy equivalent of pytorch3d.transforms.rotation_6d_to_matrix."""
    value = np.asarray(rotation_6d)
    a1, a2 = value[..., :3], value[..., 3:6]
    eps = np.finfo(value.dtype if np.issubdtype(value.dtype, np.floating) else np.float32).eps
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=-1, keepdims=True), eps)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=-1, keepdims=True), eps)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    """NumPy equivalent of pytorch3d.transforms.matrix_to_rotation_6d."""
    matrix = np.asarray(matrix)
    return matrix[..., :2, :].reshape(matrix.shape[:-2] + (6,))


def to_relative_chunk(anchor: np.ndarray, future: np.ndarray) -> np.ndarray:
    """Convert absolute future 128-D rows to one current-frame-relative chunk.

    ``anchor`` may be one row or have broadcastable leading dimensions (for
    example ``(B,1,128)`` for ``future`` shaped ``(B,H,128)``).  The returned
    array has the same shape and dtype as ``future``.
    """
    anchor = _check_128("anchor", anchor)
    future = _check_128("future", future)
    relative = np.array(future, copy=True)
    for indices in POSE_BLOCKS:
        relative[..., indices[:3]] = future[..., indices[:3]] - anchor[..., indices[:3]]
        anchor_rotation = rotation_6d_to_matrix(anchor[..., indices[3:9]])
        future_rotation = rotation_6d_to_matrix(future[..., indices[3:9]])
        delta_rotation = np.matmul(
            np.swapaxes(anchor_rotation, -1, -2), future_rotation)
        relative[..., indices[3:9]] = matrix_to_rotation_6d(delta_rotation)
    for indices in POINT_BLOCKS:
        relative[..., indices] = future[..., indices] - anchor[..., indices]
    return relative


def to_absolute_chunk(anchor: np.ndarray, relative: np.ndarray) -> np.ndarray:
    """Invert :func:`to_relative_chunk` using the supplied scheduled anchor."""
    anchor = _check_128("anchor", anchor)
    relative = _check_128("relative", relative)
    absolute = np.array(relative, copy=True)
    for indices in POSE_BLOCKS:
        absolute[..., indices[:3]] = relative[..., indices[:3]] + anchor[..., indices[:3]]
        anchor_rotation = rotation_6d_to_matrix(anchor[..., indices[3:9]])
        delta_rotation = rotation_6d_to_matrix(relative[..., indices[3:9]])
        future_rotation = np.matmul(anchor_rotation, delta_rotation)
        absolute[..., indices[3:9]] = matrix_to_rotation_6d(future_rotation)
    for indices in POINT_BLOCKS:
        absolute[..., indices] = relative[..., indices] + anchor[..., indices]
    return absolute


def convert_action_chunk(
    anchor: np.ndarray, action: np.ndarray, representation: str
) -> np.ndarray:
    if representation == "original":
        return np.asarray(action)
    if representation == "relative_chunk":
        return to_relative_chunk(anchor, action)
    raise ValueError(
        f"action representation must be one of {ACTION_REPRESENTATIONS}, got {representation!r}"
    )
