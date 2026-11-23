import numpy as np

from hdt.action_representation import (
    POINT_BLOCKS,
    POSE_BLOCKS,
    rotation_6d_to_matrix,
    to_absolute_chunk,
    to_relative_chunk,
)


IDENTITY_6D = np.asarray([1, 0, 0, 0, 1, 0], dtype=np.float32)


def _valid_actions(shape, seed=0):
    rng = np.random.default_rng(seed)
    value = rng.normal(size=shape).astype(np.float32)
    for indices in POSE_BLOCKS:
        # QR gives valid rotations without making the test depend on PyTorch3D.
        matrix = rng.normal(size=shape[:-1] + (3, 3))
        q, _ = np.linalg.qr(matrix)
        det = np.linalg.det(q)
        q[..., :, -1] *= np.where(det < 0, -1, 1)[..., None]
        value[..., indices[3:9]] = q[..., :2, :].reshape(shape[:-1] + (6,))
    return value


def test_relative_chunk_round_trip_preserves_pose_and_keypoints():
    anchor = _valid_actions((128,), seed=1)
    future = _valid_actions((13, 128), seed=2)

    relative = to_relative_chunk(anchor, future)
    restored = to_absolute_chunk(anchor, relative)

    represented = np.concatenate(
        [np.asarray(indices) for indices in POSE_BLOCKS + POINT_BLOCKS])
    np.testing.assert_allclose(restored[:, represented], future[:, represented], atol=1e-5)


def test_every_rotation_is_relative_to_the_same_anchor():
    anchor = np.zeros(128, dtype=np.float32)
    future = np.zeros((3, 128), dtype=np.float32)
    for indices in POSE_BLOCKS:
        anchor[indices[3:9]] = IDENTITY_6D
        future[:, indices[3:9]] = IDENTITY_6D

    future[:, 0] = [1.0, 2.0, 3.0]
    relative = to_relative_chunk(anchor, future)

    np.testing.assert_allclose(relative[:, 0], [1.0, 2.0, 3.0])
    for indices in POSE_BLOCKS:
        matrices = rotation_6d_to_matrix(relative[:, indices[3:9]])
        np.testing.assert_allclose(
            matrices, np.broadcast_to(np.eye(3), matrices.shape), atol=1e-6)


def test_batched_anchor_matches_per_chunk_conversion():
    anchor = _valid_actions((4, 1, 128), seed=3)
    future = _valid_actions((4, 7, 128), seed=4)
    batched = to_relative_chunk(anchor, future)
    separate = np.stack(
        [to_relative_chunk(anchor[i, 0], future[i]) for i in range(len(anchor))])
    np.testing.assert_allclose(batched, separate, atol=1e-6)
