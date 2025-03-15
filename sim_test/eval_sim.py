import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from einops import rearrange
import time
import cv2
from torchvision import transforms
from sim_env import BOX_POSE
from sim_env import make_sim_env

# Sim FPS
FPS = 50
DEVICE = 'cuda'

SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top', 'left_wrist', 'right_wrist']
    },
}

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def main(args):
    set_seed(1)
    # command line parameters
    onscreen_render = args['onscreen_render']
    task_name = args['task_name']

    # get task parameters
    task_config = SIM_TASK_CONFIGS[task_name]
    # num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']

    config = {
        'episode_len': episode_len,
        'onscreen_render': onscreen_render,
        'task_name': task_name,
        'seed': 0,
        'camera_names': camera_names,
        'norm_stats_path': args['norm_stats_path'],
        'model_path': args['model_path'],
        'model_cfg_path': args['model_cfg_path'],
        'lang_embeddings_path': args['lang_embeddings_path'],
    }

    success_rate, avg_return = eval_bc(config, num_rollouts=50)

    print(f'{success_rate=} {avg_return=}')
    print()

    exit()

def eval_bc(config, num_rollouts=50):
    set_seed(1000)
    onscreen_render = config['onscreen_render']
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    # temporal_agg = config['temporal_agg']
    onscreen_cam = 'angle'

    from cet.eval_6d import get_norm_stats, load_policy

    # load policy and stats
    norm_stats = get_norm_stats(config['norm_stats_path'], 'sim_aloha')

    policy, visual_preprocessor = load_policy(config['model_path'], config['model_cfg_path'], DEVICE)

    def pre_process(s_qpos):
        ret = (s_qpos - norm_stats['qpos_mean']) / norm_stats['qpos_std']
        return ret
    
    def post_process(a):
        ret = a * norm_stats['action_std'] + norm_stats['action_mean']
        return ret

    env = make_sim_env(task_name)
    env_max_reward = env.task.max_reward

    query_frequency = 63
    num_queries = query_frequency # chunk size
    temporal_agg = True
    if temporal_agg:
        query_frequency = 10

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

    episode_returns = []
    highest_rewards = []
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        ### set task
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose() # used in sim reset
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose()) # used in sim reset

        ts = env.reset()

        ### onscreen render
        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()
            # Write cv2 video
            writer = cv2.VideoWriter(f'rollout_{rollout_id}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (640, 480))

        ### evaluation loop
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, 128]).to(DEVICE)

        image_list = [] # for visualization
        qpos_list = []
        target_qpos_list = []
        rewards = []
        # if use_actuator_net:
        #     norm_episode_all_base_actions = [actuator_norm(np.zeros(history_len, 2)).tolist()]
        with torch.inference_mode():
            time0 = time.time()
            DT = 1 / FPS
            culmulated_delay = 0 
            for t in range(max_timesteps):
                time1 = time.time()
                ### update onscreen render and wait for DT
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    writer.write(image[:, :, ::-1])

                ### process previous timestep to get qpos and image_list
                time2 = time.time()
                obs = ts.observation
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos'])
                qpos_data = np.zeros(128, dtype=np.float32)
                qpos_data[:len(qpos_numpy)] = qpos_numpy
                qpos_numpy = qpos_data

                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().unsqueeze(0).to(DEVICE)
                if t % query_frequency == 0:
                    camera_names = ['top', 'top']
                    curr_images = []
                    for cam_name in camera_names:
                        curr_image = ts.observation['images'][cam_name]
                        if onscreen_render and cam_name == 'top':
                            plt_img.set_data(curr_image)
                            plt.pause(0.03)
                        # cv2.imshow(cam_name, curr_image[:, :, ::-1])
                        # cv2.waitKey(1)
                        if cam_name == 'top':
                            curr_image = cv2.resize(curr_image, (640, 480))
                            curr_image = curr_image[120:120+240, 160:160+320, :]
                        else:
                            curr_image = cv2.resize(curr_image, (320, 240))
                        curr_image = rearrange(curr_image, 'h w c -> c h w')
                        curr_images.append(curr_image)

                    image_data = np.stack(curr_images, axis=0)
                    image_data = visual_preprocessor(image_data).float()
                    B, C, H, W = image_data.shape
                    image_data = image_data.view((1, B, C, H, W)).to(DEVICE)
                    # curr_image = get_image(ts, camera_names)
                # print('get image: ', time.time() - time2)

                if t == 0:
                    # warm up
                    for _ in range(10):
                        output = policy(image_data, qpos)
                    print('network warm up done')
                    time1 = time.time()

                ### query policy
                time3 = time.time()
                if t % query_frequency == 0:
                    all_actions = policy(image_data, qpos)
                raw_action = all_actions[:, t % query_frequency]
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions[:, :num_queries]
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).unsqueeze(dim=1).to(DEVICE)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)

                ### post-process actions
                time4 = time.time()
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action[:14]

                ### step the environment
                time5 = time.time()
                ts = env.step(target_qpos)
                # print('step env: ', time.time() - time5)

                ### for visualization
                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)
                duration = time.time() - time1
                sleep_time = max(0, DT - duration)
                # print(sleep_time)
                time.sleep(sleep_time)
                # time.sleep(max(0, DT - duration - culmulated_delay))
                if duration >= DT:
                    culmulated_delay += (duration - DT)
                    # print(f'Warning: step duration: {duration:.3f} s at step {t} longer than DT: {DT} s, culmulated delay: {culmulated_delay:.3f} s')

            print(f'Avg fps {max_timesteps / (time.time() - time0)}')
            plt.close()
        
        if config['onscreen_render']:
            writer.release()

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    return success_rate, avg_return

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--norm_stats_path', type=str, help='norm stats path', required=True)
    parser.add_argument('--model_path', type=str, help='model path', required=True)
    parser.add_argument('--lang_embeddings_path', type=str, help='lang embeddings path', required=True)
    parser.add_argument('--model_cfg_path', type=str, help='path to model cfg yaml', required=True)

    parser.add_argument('--task_name', action='store', type=str, help='task_name', default="sim_transfer_cube_scripted")
    
    main(vars(parser.parse_args()))
