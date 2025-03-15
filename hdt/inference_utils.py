import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from scipy.interpolate import interp1d
from pytransform3d.rotations import euler_from_matrix
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

import hdt.constants

def get_eef_kpts_from_prediction(action):
    left_wrist_mat = np.eye(4)
    left_wrist_mat[0:3, 3] = action[hdt.constants.OUTPUT_LEFT_EEF[0:3]]
    left_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[hdt.constants.OUTPUT_LEFT_EEF[3:]]).unsqueeze(0)).numpy()

    left_hand_keypoints = np.zeros((25,3))
    left_hand_keypoints[hdt.constants.RETARGETTING_INDICES] = action[hdt.constants.OUTPUT_LEFT_KEYPOINTS].reshape((6,3))

    right_wrist_mat = np.eye(4)
    right_wrist_mat[0:3, 3] = action[hdt.constants.OUTPUT_RIGHT_EEF[0:3]]
    right_wrist_mat[0:3, 0:3] = rotation_6d_to_matrix(torch.tensor(action[hdt.constants.OUTPUT_RIGHT_EEF[3:]]).unsqueeze(0)).numpy()

    right_hand_keypoints = np.zeros((25,3))
    right_hand_keypoints[hdt.constants.RETARGETTING_INDICES] = action[hdt.constants.OUTPUT_RIGHT_KEYPOINTS].reshape((6,3))

    head_mat = np.eye(4)
    head_mat[0:3, 3] = action[hdt.constants.OUTPUT_HEAD_EEF[0:3]]
    head_rmat = rotation_6d_to_matrix(torch.tensor(action[hdt.constants.OUTPUT_HEAD_EEF[3:]]).unsqueeze(0)).squeeze().numpy()
    head_mat[0:3, 0:3] = head_rmat

    return {
        'left_wrist_mat': left_wrist_mat,
        'left_hand_kpts': left_hand_keypoints,
        'right_wrist_mat': right_wrist_mat,
        'right_hand_kpts': right_hand_keypoints,
        'head_mat': head_mat
    }

def rotation_matrix_to_yp(rotation_matrix):
    """
    Convert a rotation matrix to yaw-pitch (YP) angles.

    Parameters:
    rotation_matrix (3x3 array): Rotation matrix.

    Returns:
    tuple: YP angles in radians (yaw, pitch).
    """
    rpy_angles = euler_from_matrix(rotation_matrix, i=0, j=1, k=2, extrinsic=False)
    yaw_pitch = (rpy_angles[2], rpy_angles[1])  
    return yaw_pitch

def manual_slerp(quat1, quat2, t):
    """
    Manual implementation of Spherical Linear Interpolation (SLERP) for quaternions.
    """
    dot_product = np.dot(quat1, quat2)

    # If the dot product is negative, the quaternions have opposite handed-ness and slerp won't take
    # the shorter path. Fix by reversing one quaternion.
    if dot_product < 0.0:
        quat1 = -quat1
        dot_product = -dot_product
    
    # Clamp value to stay within domain of acos()
    dot_product = np.clip(dot_product, -1.0, 1.0)

    theta_0 = np.arccos(dot_product)  # angle between input vectors
    theta = theta_0 * t  # angle between v0 and result

    quat2 = quat2 - quat1 * dot_product
    quat2 = quat2 / np.linalg.norm(quat2)

    return quat1 * np.cos(theta) + quat2 * np.sin(theta)

def interpolate_128dim_action(action_nd, target_m):
    """
    Interpolate 128dim action from $n$ actions to $m$ actions.
    The format of 128dim follows the OUTPUT definition in hdt.constants. 
    """
    target_t = np.linspace(0, action_nd.shape[0] - 1, target_m)
    
    interpolated_action = np.zeros((target_m, 128))
    for idx in range(target_m):
        curr_t = target_t[idx]
        src_prev_t = int(curr_t)
        src_next_t = min(src_prev_t + 1, action_nd.shape[0] - 1)
        assert src_prev_t >= 0 and src_next_t < action_nd.shape[0], "Index out of bounds: {} {} {}".format(idx, src_prev_t, src_next_t)
        src_prev_action = action_nd[src_prev_t]
        src_next_action = action_nd[src_next_t]
        # This interpolates 3D positions / qpos
        interpolated_action[idx] = (src_next_action - src_prev_action) * (curr_t - src_prev_t) + src_prev_action
        # TODO(roger): conceptually, 6D rotation is not a linear space, so we need to use SLERP
        # however, we tried simple interpolation on the robot and it worked well
    return interpolated_action

def interpolate_se3(T1, T2, t):
    """
    Interpolates between two SE(3) poses.
    
    :param T1: First SE(3) matrix.
    :param T2: Second SE(3) matrix.
    :param t: Interpolation factor (0 <= t <= 1).
    :return: Interpolated SE(3) matrix.
    """
    if np.isclose(T1 - T2, 0).all():
        return T1
    # Decompose matrices into rotation (as quaternion) and translation
    rot1, trans1 = T1[:3, :3], T1[:3, 3]
    rot2, trans2 = T2[:3, :3], T2[:3, 3]
    quat1, quat2 = R.from_matrix(rot1).as_quat(), R.from_matrix(rot2).as_quat()

    # Spherical linear interpolation (SLERP) for rotation
    # Manual SLERP for rotation
    interp_quat = manual_slerp(quat1, quat2, t)
    interp_rot = R.from_quat(interp_quat).as_matrix()

    # Linear interpolation for translation
    interp_trans = interp1d([0, 1], np.vstack([trans1, trans2]), axis=0)(t)

    # Recompose SE(3) matrix
    T_interp = np.eye(4)
    T_interp[:3, :3] = interp_rot
    T_interp[:3, 3] = interp_trans

    return T_interp