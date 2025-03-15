import h5py
import numpy as np
import pinocchio as pin
from typing import Dict, Any, Tuple

import hdt.constants
from cet.utils import fk_cmd_dict2policy

# CONSTANTS
SCALES = [ 1.1, 1.1, 1.1, 1.1, 1.2, 1.1, 1.1, 1.1, 1.1, 1.2 ]
target_task_link_names = [ "L_thumb_tip", "L_index_tip", "L_middle_tip", "L_ring_tip", "L_pinky_tip", "R_thumb_tip", "R_index_tip", "R_middle_tip", "R_ring_tip", "R_pinky_tip"]
SCALES_DIC = {link_name: scale for link_name, scale in zip(target_task_link_names, SCALES)}
HAND_LOWER_LIMIT = np.array([0, 0, 0, 0, -0.1, 0])
HAND_UPPER_LIMIT = np.array([1.7, 1.7, 1.7, 1.7, 1.3, 0.5])

def load_hdf5(path, offset=10):  # offset 10ms
    input_file = path + ".hdf5"
    file = h5py.File(input_file, 'r')

    print(f"Total hdf5_frames: {file['/obs/timestamp'].shape[0]}")
    timestamps = np.array(file["/obs/timestamp"][:] * 1000, dtype=np.int64) - offset
    cmd_dict = {}
    for k in file["/action/cmd"].keys():
        cmd_dict[k] = np.array(file["/action/cmd"][k][()])
    qpos = np.array(file["/obs/qpos"][()])
    ik_qpos = np.array(file["/action/joint_pos"][()])

    return timestamps, cmd_dict, qpos, ik_qpos

class ArmInterface:
    def __init__(self, urdf_path, arm_indices, wrist_name):
        self.arm_indices = arm_indices

        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.dof = self.model.nq

        # lock joints
        locked_joint_ids = list(set(range(self.dof)) - set(self.arm_indices))
        locked_joint_ids = [id + 1 for id in locked_joint_ids]  # account for universe joint
        self.model = pin.buildReducedModel(self.model, locked_joint_ids, np.zeros(self.dof))
        self.arm_dof = self.model.nq

        self.data: pin.Data = self.model.createData()

        self.wrist_id = self.model.getFrameId(wrist_name)

    def compute_ee_pose(self, joint_pos: np.ndarray):
        assert joint_pos.shape[0] == self.arm_dof

        pin.forwardKinematics(self.model, self.data, joint_pos)
        oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, self.wrist_id)
        postion = oMf.translation - hdt.constants.H1_HEAD_POS
        rotation = oMf.rotation
        retargeted_pose = np.eye(4)
        retargeted_pose[:3, :3] = rotation
        retargeted_pose[:3, 3] = postion
        return retargeted_pose

class HandInterface:
    def __init__(self, urdf_path, hand_indices, target_task_link_names, wrist_link_name):
        self.hand_indices = hand_indices

        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.dof = self.model.nq

        # lock joints
        locked_joint_ids = list(set(range(self.dof)) - set(self.hand_indices))
        locked_joint_ids = [id + 1 for id in locked_joint_ids]  # account for universe joint
        self.model = pin.buildReducedModel(self.model, locked_joint_ids, np.zeros(self.dof))
        self.hand_dof = self.model.nq

        self.data: pin.Data = self.model.createData()
        self.wrist_link_name = wrist_link_name

        self.wrist_link_id = self.model.getFrameId(wrist_link_name)
        pin.forwardKinematics(self.model, self.data, pin.neutral(self.model))
        self.wrist_link_pose = pin.updateFramePlacement(
            self.model, self.data, self.wrist_link_id
        )

        self.target_task_link_ids = {
            link_name: self.model.getFrameId(link_name)
            for link_name in target_task_link_names
        }

    def compute_all_keypoints(self, joint_pos: np.ndarray):
        joint_pos = self.generate_full_qpos(joint_pos)

        assert joint_pos.shape[0] == self.hand_dof
        
        pin.forwardKinematics(self.model, self.data, joint_pos)
        
        ee_poses = {}
        for link_name, frame_id in self.target_task_link_ids.items():
            oMf: pin.SE3 = pin.updateFramePlacement(self.model, self.data, frame_id)
            ee_poses[link_name] = oMf.homogeneous

        retargeted_keypoints = self.retarget_keypoints(ee_poses)

        return retargeted_keypoints
    
    def retarget_keypoints(self, ee_poses):
        retargeted_keypoints = {}
        for link_name, ee_pose in ee_poses.items():
            scale = SCALES_DIC[link_name]
            ee_position = ee_pose[:3, 3] / scale
            ee_position = (ee_position - self.wrist_link_pose.translation) @ self.wrist_link_pose.rotation

            adjusted_pose = ee_pose.copy()
            adjusted_pose[:3, 3] = ee_position
            retargeted_keypoints[link_name] = adjusted_pose

        return retargeted_keypoints

    def generate_full_qpos(self, input_qpos: np.ndarray) -> np.ndarray:
        if input_qpos.shape[0] != 6:
            raise ValueError("Input qpos must have 6 elements.")

        full_qpos = np.zeros(12)

        full_qpos[0] = input_qpos[0]   # 20
        full_qpos[2] = input_qpos[1]   # 22
        full_qpos[4] = input_qpos[2]   # 24
        full_qpos[6] = input_qpos[3]   # 26
        full_qpos[8] = input_qpos[4]   # 28
        full_qpos[9] = input_qpos[5]   # 29

        full_qpos[1] = full_qpos[0]                    # 21 mimics 20
        full_qpos[3] = full_qpos[2]                    # 23 mimics 22
        full_qpos[5] = full_qpos[4]                    # 25 mimics 24
        full_qpos[7] = full_qpos[6]                    # 27 mimics 26
        full_qpos[10] = 1.6 * full_qpos[9]             # 30 mimics 29 with multiplier 1.6
        full_qpos[11] = 2.4 * full_qpos[9]             # 31 mimics 29 with multiplier 2.4

        return full_qpos

# Step 3: for hand change into 0-1, we need to convert it back
def convert_hand_actions_back(hand_actions, hand_lower_limit, hand_upper_limit):
    hand_actions = 1 - hand_actions
    hand_actions = hand_actions * (hand_upper_limit - hand_lower_limit) + hand_lower_limit
    return hand_actions


class FKCmdDictGenerator:
    def __init__(self, urdf_path: str, left_arm_indices: list, right_arm_indices: list, 
                 left_wrist_name: str, right_wrist_name: str, target_task_link_names: list, hdf5_timestamps: list=None):
        self.urdf_path = urdf_path
        self.left_arm_indices = left_arm_indices
        self.right_arm_indices = right_arm_indices
        self.left_wrist_name = left_wrist_name
        self.right_wrist_name = right_wrist_name
        self.hdf5_timestamps = hdf5_timestamps

        # for arm
        self.left_arm_interface = ArmInterface(self.urdf_path, self.left_arm_indices, self.left_wrist_name)
        self.right_arm_interface = ArmInterface(self.urdf_path, self.right_arm_indices, self.right_wrist_name)

        # for hand
        self.left_hand_interface = HandInterface(self.urdf_path, hdt.constants.H1_LEFT_HAND_INDICES, target_task_link_names, self.left_wrist_name)
        self.right_hand_interface = HandInterface(self.urdf_path, hdt.constants.H1_RIGHT_HAND_INDICES, target_task_link_names, self.right_wrist_name)

    def compute_arm_fk(self, left_qpos: np.ndarray, right_qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        left_ee_pose = self.left_arm_interface.compute_ee_pose(left_qpos)
        left_position = left_ee_pose[:3, 3]
        left_rotation = left_ee_pose[:3, :3]

        right_ee_pose = self.right_arm_interface.compute_ee_pose(right_qpos)
        right_position = right_ee_pose[:3, 3]
        right_rotation = right_ee_pose[:3, :3]

        return left_position, left_rotation, right_position, right_rotation
    
    def compute_hand_fk(self, left_hand_qpos: np.ndarray, right_hand_qpos: np.ndarray, norm_qpos_to_urdf: bool) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        if norm_qpos_to_urdf:
            left_hand_qpos = convert_hand_actions_back(left_hand_qpos, HAND_LOWER_LIMIT, HAND_UPPER_LIMIT)
            right_hand_qpos = convert_hand_actions_back(right_hand_qpos, HAND_LOWER_LIMIT, HAND_UPPER_LIMIT)
        left_hand_ee_pose = self.left_hand_interface.compute_all_keypoints(left_hand_qpos)
        right_hand_ee_pose = self.right_hand_interface.compute_all_keypoints(right_hand_qpos)

        left_hand_ee_position = []
        left_hand_ee_position.append((0,0,0))
        for link_name, ee_pose in left_hand_ee_pose.items():
            if link_name.startswith("L_"):
                left_hand_ee_position.append(ee_pose[:3, 3])
        
        right_hand_ee_position = []
        right_hand_ee_position.append((0,0,0))
        for link_name, ee_pose in right_hand_ee_pose.items():
            if link_name.startswith("R_"):
                right_hand_ee_position.append(ee_pose[:3, 3])

        return left_hand_ee_position, right_hand_ee_position
    
    def compute_fk(self, qpos: np.ndarray, norm_qpos_to_urdf: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # for arm
        left_arm_qpos = qpos[hdt.constants.H1_QPOS_LEFT_ARM_INDICES]
        right_arm_qpos = qpos[hdt.constants.H1_QPOS_RIGHT_ARM_INDICES]
        left_fk_pos, left_fk_rot, right_fk_pos, right_fk_rot = self.compute_arm_fk(left_arm_qpos, right_arm_qpos)

        # for hand
        left_hand_qpos = qpos[hdt.constants.H1_QPOS_LEFT_HAND_INDICES]
        right_hand_qpos = qpos[hdt.constants.H1_QPOS_RIGHT_HAND_INDICES]
        left_hand_ee_position, right_hand_ee_position = self.compute_hand_fk(left_hand_qpos, right_hand_qpos, norm_qpos_to_urdf)

        return left_fk_pos, left_fk_rot, right_fk_pos, right_fk_rot, left_hand_ee_position, right_hand_ee_position


    def generate_fk_cmd_dict(self, cmd_dict: Dict[str, Any], qpos: np.ndarray, norm_qpos_to_urdf: bool) -> Dict[str, Any]:
        fk_cmd_dict = {"head_mat":[], "rel_left_wrist_mat": [], "rel_right_wrist_mat": [], "rel_left_hand_keypoints": [], "rel_right_hand_keypoints": []}
        position_errors = {"left": [], "right": []}
        rotation_errors = {"left": [], "right": []}
        hand_position_errors = {"left": [], "right": []}
        
        assert self.hdf5_timestamps is not None, "hdf5_timestamps wasn't passed properly"

        for idx in range(len(self.hdf5_timestamps)):
            _qpos = qpos[idx]

            # FK computations
            _, left_fk_pos, left_fk_rot, right_fk_pos, right_fk_rot, left_hand_ee_position, right_hand_ee_position, _fk_cmd_dict\
                = self.generate_fk_obs(_qpos, cmd_dict["head_mat"][idx], norm_qpos_to_urdf)

            # Command positions and rotations
            cmd_left_pos = cmd_dict["rel_left_wrist_mat"][idx, :3, 3]
            cmd_left_rot = cmd_dict["rel_left_wrist_mat"][idx, :3, :3]
            cmd_right_pos = cmd_dict["rel_right_wrist_mat"][idx, :3, 3]
            cmd_right_rot = cmd_dict["rel_right_wrist_mat"][idx, :3, :3]
            
            cmd_left_hand_keypoints = cmd_dict["rel_left_hand_keypoints"][idx][hdt.constants.RETARGETTING_INDICES]
            cmd_right_hand_keypoints = cmd_dict["rel_right_hand_keypoints"][idx][hdt.constants.RETARGETTING_INDICES]

            # Error calculations
            position_errors["left"].append(np.linalg.norm(left_fk_pos - cmd_left_pos))
            rotation_errors["left"].append(np.linalg.norm(left_fk_rot - cmd_left_rot))

            position_errors["right"].append(np.linalg.norm(right_fk_pos - cmd_right_pos))
            rotation_errors["right"].append(np.linalg.norm(right_fk_rot - cmd_right_rot))

            hand_position_errors["left"].append(np.linalg.norm(left_hand_ee_position - cmd_left_hand_keypoints))
            hand_position_errors["right"].append(np.linalg.norm(right_hand_ee_position - cmd_right_hand_keypoints))

            # Store FK results
            fk_cmd_dict["rel_left_wrist_mat"].append(_fk_cmd_dict["rel_left_wrist_mat"])

            fk_cmd_dict["rel_right_wrist_mat"].append(_fk_cmd_dict["rel_right_wrist_mat"])

            fk_cmd_dict["rel_left_hand_keypoints"].append(_fk_cmd_dict["rel_left_hand_keypoints"])
            fk_cmd_dict["rel_right_hand_keypoints"].append(_fk_cmd_dict["rel_right_hand_keypoints"])

        # Convert lists to arrays
        fk_cmd_dict["head_mat"] = np.array(cmd_dict["head_mat"])
        fk_cmd_dict["rel_left_wrist_mat"] = np.array(fk_cmd_dict["rel_left_wrist_mat"])
        fk_cmd_dict["rel_right_wrist_mat"] = np.array(fk_cmd_dict["rel_right_wrist_mat"])
        fk_cmd_dict["rel_left_hand_keypoints"] = np.array(fk_cmd_dict["rel_left_hand_keypoints"])
        fk_cmd_dict["rel_right_hand_keypoints"] = np.array(fk_cmd_dict["rel_right_hand_keypoints"])

        # Calculate mean errors
        mean_position_error = {
            "left": np.mean(position_errors["left"]),
            "right": np.mean(position_errors["right"])
        }
        mean_rotation_error = {
            "left": np.mean(rotation_errors["left"]),
            "right": np.mean(rotation_errors["right"])
        }

        hand_position_errors = {
            "left": np.mean(hand_position_errors["left"]) / 5,
            "right": np.mean(hand_position_errors["right"]) / 5
        }

        return fk_cmd_dict, mean_position_error, mean_rotation_error, hand_position_errors
    
    def generate_fk_obs(self, qpos, head_mat, norm_qpos_to_urdf):
        fk_cmd_dict = {"head_mat":[], "rel_left_wrist_mat": [], "rel_right_wrist_mat": [], "rel_left_hand_keypoints": [], "rel_right_hand_keypoints": []}

        # FK computations
        left_fk_pos, left_fk_rot, right_fk_pos, right_fk_rot, left_hand_ee_position, right_hand_ee_position = self.compute_fk(qpos, norm_qpos_to_urdf)

        # Store FK results
        fk_cmd_dict["rel_left_wrist_mat"].append(np.block([
            [left_fk_rot, left_fk_pos[:, None]],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ]))

        fk_cmd_dict["rel_right_wrist_mat"].append(np.block([
            [right_fk_rot, right_fk_pos[:, None]],
            [np.zeros((1, 3)), np.ones((1, 1))]
        ]))

        fk_cmd_dict["rel_left_hand_keypoints"].append(left_hand_ee_position)
        fk_cmd_dict["rel_right_hand_keypoints"].append(right_hand_ee_position)

        # Convert lists to arrays
        #! need head input 
        fk_cmd_dict["head_mat"] = np.array(head_mat.reshape(1, 4, 4))
        fk_cmd_dict["rel_left_wrist_mat"] = np.array(fk_cmd_dict["rel_left_wrist_mat"])
        fk_cmd_dict["rel_right_wrist_mat"] = np.array(fk_cmd_dict["rel_right_wrist_mat"])
        fk_cmd_dict["rel_left_hand_keypoints"] = np.array(fk_cmd_dict["rel_left_hand_keypoints"])
        fk_cmd_dict["rel_right_hand_keypoints"] = np.array(fk_cmd_dict["rel_right_hand_keypoints"])

        policy_state = fk_cmd_dict2policy(fk_cmd_dict, num_timesteps = 1)

        return policy_state, left_fk_pos, left_fk_rot, right_fk_pos, right_fk_rot, left_hand_ee_position, right_hand_ee_position, fk_cmd_dict
