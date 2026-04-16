import mujoco as mj
import mujoco.viewer as mjv
from cet.sim_mujoco import MujocoSim

import cv2
import yaml
import h5py
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import hdt.constants
from cet.utils_fk import FKCmdDictGenerator, target_task_link_names
from cet.eval_6d import load_policy, get_norm_stats, normalize_input
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
import torch

def _fix_action_for_mujoco(
    actions_gt: np.ndarray,
    *,
    pos_only: bool = False,
    scale: float = 1.0,
    swap_lr: bool = False,
    flip_x: bool = False,
    flip_y: bool = False,
    flip_z: bool = False,
) -> np.ndarray:
    actions_gt = np.asarray(actions_gt, dtype=np.float32).copy()
    R_map = np.array([[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    head_slice = hdt.constants.OUTPUT_HEAD_EEF
    lw_slice = hdt.constants.OUTPUT_LEFT_EEF
    rw_slice = hdt.constants.OUTPUT_RIGHT_EEF

    head_t = actions_gt[:, head_slice][:, 0:3]
    lw_t = actions_gt[:, lw_slice][:, 0:3]
    rw_t = actions_gt[:, rw_slice][:, 0:3]

    lw_rel = ((lw_t - head_t) @ R_map.T) * float(scale)
    rw_rel = ((rw_t - head_t) @ R_map.T) * float(scale)
    if swap_lr:
        lw_rel, rw_rel = rw_rel, lw_rel
    if flip_x:
        lw_rel[:, 0] *= -1.0
        rw_rel[:, 0] *= -1.0
    if flip_y:
        lw_rel[:, 1] *= -1.0
        rw_rel[:, 1] *= -1.0
    if flip_z:
        lw_rel[:, 2] *= -1.0
        rw_rel[:, 2] *= -1.0

    actions_gt[:, head_slice][:, 0:3] = 0.0
    actions_gt[:, lw_slice][:, 0:3] = lw_rel
    actions_gt[:, rw_slice][:, 0:3] = rw_rel

    if pos_only:
        ident6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        actions_gt[:, head_slice][:, 3:9] = ident6d
        actions_gt[:, lw_slice][:, 3:9] = ident6d
        actions_gt[:, rw_slice][:, 3:9] = ident6d
    else:
        head_6d = actions_gt[:, head_slice][:, 3:9]
        lw_6d = actions_gt[:, lw_slice][:, 3:9]
        rw_6d = actions_gt[:, rw_slice][:, 3:9]

        head_R = rotation_6d_to_matrix(torch.tensor(head_6d)).numpy()
        lw_R = rotation_6d_to_matrix(torch.tensor(lw_6d)).numpy()
        rw_R = rotation_6d_to_matrix(torch.tensor(rw_6d)).numpy()

        head_R2 = (R_map @ head_R.transpose(0, 2, 1)).transpose(0, 2, 1)
        lw_R2 = (R_map @ lw_R.transpose(0, 2, 1)).transpose(0, 2, 1)
        rw_R2 = (R_map @ rw_R.transpose(0, 2, 1)).transpose(0, 2, 1)

        actions_gt[:, head_slice][:, 3:9] = matrix_to_rotation_6d(torch.tensor(head_R2)).numpy()
        actions_gt[:, lw_slice][:, 3:9] = matrix_to_rotation_6d(torch.tensor(lw_R2)).numpy()
        actions_gt[:, rw_slice][:, 3:9] = matrix_to_rotation_6d(torch.tensor(rw_R2)).numpy()

    return actions_gt

def _load_hdf5(hdf5_path):
    """ Load hdf5 file """
    with h5py.File(hdf5_path, 'r') as data:
        actions_gt = np.array(data['action'])
        if 'observation.image.left' in data and 'observation.image.right' in data:
            left_imgs = np.array(data['observation.image.left'])
            right_imgs = np.array(data['observation.image.right'])
        elif 'observation.image.top' in data:
            top_imgs = np.array(data['observation.image.top'])
            left_imgs = top_imgs
            right_imgs = top_imgs
        else:
            raise KeyError("HDF5 must contain observation.image.left/right or observation.image.top")
        states = np.array(data['observation.state'])

        if len(left_imgs.shape) == 2:
            # compressed images
            assert len(right_imgs.shape) == 2
            assert left_imgs.shape[0] == right_imgs.shape[0]
            # Decompress
            left_img_list = []
            right_img_list = []
            for i in range(left_imgs.shape[0]):
                left_img = cv2.imdecode(left_imgs[i], cv2.IMREAD_COLOR)
                right_img = cv2.imdecode(right_imgs[i], cv2.IMREAD_COLOR)
                left_img_list.append(left_img.transpose((2, 0, 1)))
                right_img_list.append(right_img.transpose((2, 0, 1)))
            # BCHW format
            left_imgs = np.stack(left_img_list, axis=0)
            right_imgs = np.stack(right_img_list, axis=0)
        elif len(left_imgs.shape) == 4:
            # Raw images, could be BHWC or BCHW
            if left_imgs.shape[-1] == 3:
                left_imgs = left_imgs.transpose(0, 3, 1, 2)
                right_imgs = right_imgs.transpose(0, 3, 1, 2)

        init_action = actions_gt[0]
        init_left_img = left_imgs[0]
        init_right_img = right_imgs[0]

        return actions_gt, left_imgs, right_imgs, states, init_action, init_left_img, init_right_img

def process_image(img, original_res=(1280, 720), offset_w=0, resize_width=320, resize_height=240, sbs=True):
    original_width = original_res[0]
    original_height = original_res[1]
    original_aspect_ratio = original_width / original_height
    target_aspect_ratio = resize_width / resize_height

    if target_aspect_ratio > original_aspect_ratio:
        new_height = int(original_width / target_aspect_ratio)
        crop_size_h = (original_height - new_height) // 2
        crop_size_w = 0
    else:
        new_width = int(original_height * target_aspect_ratio)
        crop_size_w = (original_width - new_width) // 2
        crop_size_h = 0

    cropped_img_shape = (original_height - 2 * crop_size_h, original_width - 2 * crop_size_w)

    img_cropped = img[crop_size_h:original_height-crop_size_h, crop_size_w+offset_w:original_width-crop_size_w+offset_w]
    img_resized = cv2.resize(img_cropped, (resize_width, resize_height), interpolation=cv2.INTER_AREA)
    img_resized = img_resized.transpose(2, 0, 1)
    
    return img_resized   

def main(args, player, policy_rollout):
    """ Main function to evaluate the policy in the simulator """
    device = args['device']
    
    # Load initial dataset and model
    actions_gt, left_imgs, right_imgs, states, init_action, init_left_img, init_right_img = _load_hdf5(args['hdf_file_path'])
    if args.get("coord_fix", False):
        actions_gt = _fix_action_for_mujoco(
            actions_gt,
            pos_only=args.get("pos_only", False),
            scale=args.get("coord_scale", 1.0),
            swap_lr=args.get("swap_lr", False),
            flip_x=args.get("flip_x", False),
            flip_y=args.get("flip_y", False),
            flip_z=args.get("flip_z", False),
        )
    if policy_rollout:
        norm_stats = get_norm_stats(args['norm_stats_path'], embodiment_name="h1_inspire_sim")
        policy, visual_preprocessor = load_policy(args['model_path'], args['policy_config_path'], device)
    
    # Constants
    # By default, robot URDF starts with both arms laid down
    # The zeroing phase is to bring the arms to the front with preset qpos
    ZEROING_LENGTH = 50
    INIT_FIRST_ACTION_LENGTH = 180
    assert INIT_FIRST_ACTION_LENGTH > ZEROING_LENGTH + 10
        
    GT_TIME_FIX = False
    output, act = None, None
    act_index = 0
    
    predicted_list, gt_list, record_list = [], [], []
        
    # urdf_path = str(Path(__file__).resolve().parent.parent / "assets" / "h1_inspire" / "urdf" / "h1_inspire.urdf")
    urdf_path = str(Path(__file__).resolve().parent.parent / "assets" / "h1_inspire_sim" / "urdf" / "h1_inspire.urdf")
    # urdf_path = "/home/tairanh/Workspace/cross-embodiment-transformer/assets/h1_inspire/urdf/h1_inspire.urdf"
    generator = FKCmdDictGenerator(
        urdf_path, 
        hdt.constants.H1_LEFT_ARM_INDICES, 
        hdt.constants.H1_RIGHT_ARM_INDICES, 
        "L_hand_base_link", 
        "R_hand_base_link", 
        target_task_link_names
    )
    
    # Launch simulator viewer
    with mj.viewer.launch_passive(player.model, player.data) as viewer:
        player.setup_viewer(viewer, preset=args.get("viewer_preset", "front"))

        end_time = states.shape[0] + INIT_FIRST_ACTION_LENGTH
        if policy_rollout:
            # 600 extra steps because policy predictions don't match ground truth
            end_time += 600
            
        # Main simulation loop
        for t in tqdm(range(end_time)):
            # Fetch images from simulator and process them
            cur_left_img = process_image(player.get_camera_image(0))
            cur_right_img = process_image(player.get_camera_image(1))
                                
            if t < ZEROING_LENGTH:
                print("Initializing: Zeroing position")
                player.step_init(actions_gt[0], viewer, init_state=True)
                continue
            elif t < INIT_FIRST_ACTION_LENGTH:
                print("Initializing: Mimicking actions")
                player.step_init(actions_gt[0], viewer)
                continue
                
            # Inference phase: 
            t_start = t - INIT_FIRST_ACTION_LENGTH

            # Fix GT_TIME to complete the rollout trajectory
            t_gt = min(t_start, len(actions_gt) - 1)  # Ensures valid index
            GT_TIME_FIX = GT_TIME_FIX or (t_start >= len(actions_gt))
                
            # Extract ground truth data
            cur_action_gt = actions_gt[t_gt] 
            cur_left_img_gt = left_imgs[t_gt]
            cur_right_img_gt  = right_imgs[t_gt]
            cur_state_gt = states[t_gt]
            
            # Fetch qpos from sim and generate FK observations
            cur_state = np.array(player.data.qpos[hdt.constants.H1_ALL_INDICES][13:]) #  38-dim: 7*2 arms, 12*2 fingers
            # NOTE: head_mat is simply masked from training for now.
            fk_cmd_dict, _, _, _, _, _, _, _ = generator.generate_fk_obs(qpos=cur_state, head_mat=np.zeros((4, 4)), norm_qpos_to_urdf=False)

            if policy_rollout:
                qpos_data, image_data = normalize_input(fk_cmd_dict[0], cur_left_img, cur_right_img, 
                                                        norm_stats, visual_preprocessor, match_human=True)
            
                # If output is exhausted, generate new predictions
                if output is None or act_index == chunk_size - 0:
                    print("Predicted Chunk exhausted at", t_start)
                    output = policy(image_data, qpos_data)[0].detach().cpu().numpy() # (chuck_size,action_dim)
                    output = output * norm_stats["action_std"] + norm_stats["action_mean"]
                    act_index = 0
                    
                # Select current action
                act = output[act_index]
                act_index += 1
            else:
                act = cur_action_gt
            
            if args['plot']:
                predicted_list.append(act[hdt.constants.OUTPUT_RIGHT_EEF[3:6]])
                gt_list.append(cur_action_gt[hdt.constants.OUTPUT_RIGHT_EEF[3:6]])
                
            # Step the simulator
            player.step_init(act, viewer)  
            record_list.append(act)
        
        # Plot the predicted and ground truth actions
        if args['plot']:
            plt.figure()
            for i in range(3):
                plt.plot([x[i] for x in predicted_list], label=f'pred_{i}')
            for i in range(3):
                plt.plot([x[i] for x in gt_list], label=f'gt_{i}')
            plt.legend()
            plt.show()

    player.end()
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval HDT policy on figures w/ optional sim visualization', add_help=False)
    parser.add_argument('--hdf_file_path', type=str, help='hdf file path', required=True)
    parser.add_argument('--norm_stats_path', type=str, help='norm stats path', required=False)
    parser.add_argument('--model_path', type=str, help='model path', required=False)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=64)
    parser.add_argument('--policy_config_path', type=str, help='policy config path', required=False)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--tasktype', type=str, help='Scene setup', required=True, choices=['microwave', 'tap', 'pour', 'pickplace', 'wiping', 'pepsi', 'h1_only'])
    parser.add_argument('--device', type=str, help='Device', default="cuda")
    parser.add_argument('--coord_fix', action='store_true')
    parser.add_argument('--pos_only', action='store_true')
    parser.add_argument('--coord_scale', type=float, default=1.0)
    parser.add_argument('--swap_lr', action='store_true')
    parser.add_argument('--flip_x', action='store_true')
    parser.add_argument('--flip_y', action='store_true')
    parser.add_argument('--flip_z', action='store_true')
    parser.add_argument('--viewer_preset', type=str, default="front", choices=["default", "front", "side", "top"])
    args = vars(parser.parse_args())

    if args['model_path'] is not None:
        assert args['norm_stats_path'] is not None
        assert args['policy_config_path'] is not None
        policy_rollout = True
    else:
        policy_rollout = False

    chunk_size = args['chunk_size']

    root_path = str(Path(__file__).resolve().parent.parent)
    config_path = root_path + "/configs/all_tasks.yml"

    task_config = yaml.safe_load(open(config_path, "r"))["tasks"]
    config_files = [task_config["h1_inspire_sim"]["file"]]

    player = MujocoSim(config_files, root_path=root_path, task_id=0, tasktype=args['tasktype'], cfgs=None)

    main(args, player, policy_rollout)
