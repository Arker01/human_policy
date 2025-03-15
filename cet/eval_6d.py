import h5py
import pickle
import torch
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

import yaml
import hdt.constants
from hdt.modeling.utils import make_visual_encoder

def get_norm_stats(data_path, embodiment_name="h1_inspire"):
    with open(data_path, "rb") as f:
        norm_stats = pickle.load(f)
    norm_stats = norm_stats[embodiment_name]
    return norm_stats

def load_policy(policy_path, policy_config_path, device):
    with open(policy_config_path, "r") as fp:
        policy_config = yaml.safe_load(fp)
    policy_type = policy_config["common"]["policy_class"]

    if policy_type == "ACT":
        policy = torch.jit.load(policy_path, map_location=device).eval().to(device)

        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos):
                return self.policy(image, qpos)
            
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        visual_encoder, visual_preprocessor = make_visual_encoder("ACT", None)
        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "RDT":
        visual_encoder, visual_preprocessor = make_visual_encoder("RDT", {"visual_backbone": "MASKCLIP"})

        from hdt.modeling.modeling_hdt import HumanDiffusionTransformer
        policy = HumanDiffusionTransformer(
            action_dim=policy_config["common"]["state_dim"],
            pred_horizon=policy_config["common"]["action_chunk_size"],
            config=policy_config,
            lang_token_dim=policy_config["model"]["lang_token_dim"],
            img_token_dim=policy_config["model"]["img_token_dim"],
            state_token_dim=policy_config["model"]["state_token_dim"],
            max_lang_cond_len=policy_config["dataset"]["tokenizer_max_length"],
            visual_encoder=visual_encoder,
            lang_pos_embed_config=[
                # Similarly, no initial pos embed for language
                ("lang", -policy_config["dataset"]["tokenizer_max_length"]),
            ],
            dtype=torch.float32,
        )
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos):
                return self.policy(image, qpos, conditioning_dict={})
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().to(device)

        return my_policy_wrapper, visual_preprocessor
    elif policy_type == "DP":
        visual_encoder, visual_preprocessor = make_visual_encoder("DP", {"visual_backbone": "MASKCLIP"})
        
        from hdt.modeling.modeling_vanilla_dp import DiffusionPolicy
        policy = DiffusionPolicy(action_dim=128,
            chunk_size=64,
            img_token_dim=visual_encoder.hidden_size,
            state_token_dim=128,
            num_inference_timesteps=20,
            visual_encoder=visual_encoder)
        checkpoint = torch.load(policy_path, map_location=device)
        new_state_dict = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        policy.load_state_dict(new_state_dict, strict=True)
        class polciy_wrapper(torch.nn.Module):
            def __init__(self, policy):
                super().__init__()
                self.policy = policy

            @torch.no_grad()
            def forward(self, image, qpos, cond_dict):
                return self.policy(image, qpos, conditioning_dict=cond_dict)
    
        my_policy_wrapper = polciy_wrapper(policy)
        my_policy_wrapper.eval().cuda()

        return my_policy_wrapper, visual_preprocessor
    else:
        raise ValueError("Invalid policy type: {}".format(policy_type))

def normalize_input(state, left_img_chw, right_img_chw, norm_stats, visual_preprocessor, match_human = False):
    """
    Args
        - state: np.array of shape (N,) e.g., 26 in qpos-only H1 data
        - left_img_chw: np.array of shape (3, H, W) in uint8 [0, 255]
        - right_img_chw: np.array of shape (3, H, W) in uint8 [0, 255]
        - norm_stats: dict with keys "qpos_mean", "qpos_std", "action_mean", "action_std"
        - visual_preprocessor: function that takes in BCHW UINT8 image and returns processed BCHW image
    """
    image_data = np.stack([left_img_chw, right_img_chw], axis=0)
    image_data = visual_preprocessor(image_data).float()
    B, C, H, W = image_data.shape
    image_data = image_data.view((1, B, C, H, W)).to(device='cuda')

    qpos_data = torch.zeros(128, dtype=torch.float32)
    if match_human:
        qpos_data = torch.from_numpy(state).float()
    else:
        qpos_data[hdt.constants.QPOS_INDICES] = torch.from_numpy(state).float()
    qpos_data = (qpos_data - norm_stats["qpos_mean"]) / (norm_stats["qpos_std"] + 1e-6)
    qpos_data = qpos_data.view((1, -1)).to(device='cuda')

    return (qpos_data, image_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Eval HDT policy on figures w/ optional sim visualization', add_help=False)
    parser.add_argument('--hdf_file_path', type=str, help='hdf file path', required=True)
    parser.add_argument('--norm_stats_path', type=str, help='norm stats path', required=True)
    parser.add_argument('--model_path', type=str, help='model path', required=True)
    parser.add_argument('--lang_embeddings_path', type=str, help='lang embeddings path', required=True)
    parser.add_argument('--chunk_size', type=int, help='chunk size', default=64)
    parser.add_argument('--model_cfg_path', type=str, help='path to model cfg yaml', required=True)
    parser.add_argument('--plot', action='store_true')

    args = vars(parser.parse_args())

    chunk_size = args['chunk_size']
    device = "cuda"

    with h5py.File(args['hdf_file_path'], 'r') as data:
        actions = np.array(data['action'])
        left_imgs = np.array(data['observation.image.left'])
        right_imgs = np.array(data['observation.image.right'])
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

        init_action = actions[0]
        init_left_img = left_imgs[0]
        init_right_img = right_imgs[0]

    norm_stats = get_norm_stats(args['norm_stats_path'])

    policy, visual_preprocessor = load_policy(args['model_path'], args['model_cfg_path'], device)

    # Reset robot and the environment
    output = None
    act = None
    act_index = 0

    if args['plot']:
        predicted_list = []
        gt_list = []
        record_list = []

    for t in tqdm(range(states.shape[0])):
            print("step", t)
            t_start = t

            # Select offseted episodes
            cur_action = actions[t_start]
            cur_left_img = left_imgs[t_start]
            cur_right_img = right_imgs[t_start]
            # cur_state = states[t_start][hdt.constants.QPOS_INDICES]

            cur_state = states[t_start]

            qpos_data, image_data = normalize_input(cur_state, cur_left_img, cur_right_img, norm_stats, visual_preprocessor, match_human = True)

            #! here to mask data
            # qpos_data[:, hdt.constants.OUTPUT_LEFT_KEYPOINTS] = 0
            # qpos_data[:, hdt.constants.OUTPUT_RIGHT_KEYPOINTS] = 0
            qpos_data[:, hdt.constants.OUTPUT_HEAD_EEF] = 0

            if output is None or act_index == chunk_size - 10:
                output = policy(image_data, qpos_data)[0].detach().cpu().numpy() # (chuck_size,action_dim)
                output = output * norm_stats["action_std"] + norm_stats["action_mean"]
                act_index = 0
            act = output[act_index]
            act_index += 1

            if args['plot']:

                # print("act", act.shape)
                # print("gt", cur_action.shape)
                
                predicted_list.append(act[hdt.constants.OUTPUT_RIGHT_EEF[:3]])
                gt_list.append(cur_action[hdt.constants.OUTPUT_RIGHT_EEF[:3]])
                
                # predicted_list.append(act[:7])
                # gt_list.append(cur_action[:7])

            img = np.concatenate((cur_left_img.transpose((1, 2, 0)), cur_right_img.transpose((1, 2, 0))), axis=1)

            plt.cla()
            plt.title('VisionPro View')
            plt.imshow(img, aspect='equal')
            plt.pause(0.001)

            act = act.astype(np.float32)

    if args['plot']:
        print("plotting")
        plt.figure()

        for i in range(3):  # x, y, z
            plt.plot([x[i] for x in predicted_list], label=f'Predicted {i}', linestyle='--')  
            plt.plot([x[i] for x in gt_list], label=f'Ground Truth {i}') 

        plt.legend()
        plt.title("Predicted vs Ground Truth")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        plt.show()

        input("Press Enter to exit...")

    with open('record_list.pkl', 'wb') as f:  # Open file in binary write mode
        pickle.dump(record_list, f)
    