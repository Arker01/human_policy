import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import h5py
import os
from pathlib import Path

from hdt.inference_utils import get_eef_kpts_from_prediction

def _expand_hand_25_from_tips(hand_kpts_25: np.ndarray) -> np.ndarray:
    hand_kpts_25 = np.asarray(hand_kpts_25, dtype=np.float32)
    if hand_kpts_25.shape != (25, 3):
        raise ValueError(f"Expected hand_kpts_25 shape (25,3), got {hand_kpts_25.shape}")
    palm = hand_kpts_25[0].copy()
    out = hand_kpts_25.copy()
    for finger in range(5):
        base = finger * 5
        tip = base + 4
        tip_pos = hand_kpts_25[tip]
        if np.allclose(tip_pos, 0):
            continue
        alphas = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        for j, a in enumerate(alphas):
            idx = base + j
            if idx == 0:
                out[idx] = palm
            else:
                out[idx] = palm + a * (tip_pos - palm)
    return out


def load_cmd_tuple_hdf5(path, *, full_hand=False, max_frames=None):
    data_list = []

    with h5py.File(path, 'r') as file:
        # Processed HDF5
        assert "/action" in file
        frame_count = file["/action"].shape[0]
        if max_frames is not None:
            frame_count = min(frame_count, int(max_frames))
        for i in range(frame_count):
            cur_cmd_dict = get_eef_kpts_from_prediction(file["/action"][i])
            # Format post-processed data to match expected structure
            head_mat = cur_cmd_dict['head_mat']
            left_wrist_mat = cur_cmd_dict['left_wrist_mat']
            right_wrist_mat = cur_cmd_dict['right_wrist_mat']
            left_hand_kpts = cur_cmd_dict['left_hand_kpts']
            right_hand_kpts = cur_cmd_dict['right_hand_kpts']

            if full_hand:
                left_hand_kpts = _expand_hand_25_from_tips(left_hand_kpts)
                right_hand_kpts = _expand_hand_25_from_tips(right_hand_kpts)
            
            # Create skeleton joint structures from the keypoints
            left_skeleton_joints = np.zeros((25, 4, 4))
            right_skeleton_joints = np.zeros((25, 4, 4))
            
            # Set identity matrices for each joint
            for j in range(25):
                left_skeleton_joints[j] = np.eye(4)
                right_skeleton_joints[j] = np.eye(4)
            
            # Fill in the finger positions
            for j in range(min(25, len(left_hand_kpts))):
                left_skeleton_joints[j, 3, 0:3] = left_hand_kpts[j]
            
            for j in range(min(25, len(right_hand_kpts))):
                right_skeleton_joints[j, 3, 0:3] = right_hand_kpts[j]
            
            # Construct data dictionary matching expected format
            data = {
                'head': head_mat.flatten(order="F").tolist(),
                'rightWrist': right_wrist_mat.flatten(order="F").tolist(),
                'leftWrist': left_wrist_mat.flatten(order="F").tolist(),
                'rightSkeleton': {
                    'joints': right_skeleton_joints.reshape(-1).tolist()
                },
                'leftSkeleton': {
                    'joints': left_skeleton_joints.reshape(-1).tolist()
                }
            }
            data_list.append(data)

    return data_list

def _load_act_policy(model_cfg_path, chunk_size, camera_names):
    import yaml
    import torch
    from hdt.policy import ACTPolicy

    with open(model_cfg_path, "r") as fp:
        trainer_config = yaml.safe_load(fp)

    policy_config = {
        "lr": 1e-4,
        "num_queries": chunk_size,
        "kl_weight": trainer_config["model"]["kl_weight"],
        "hidden_dim": trainer_config["model"]["hidden_dim"],
        "chunk_size": chunk_size,
        "dim_feedforward": trainer_config["model"]["dim_feedforward"],
        "lr_backbone": float(trainer_config["model"]["lr_backbone"]),
        "backbone": trainer_config["model"]["backbone"],
        "enc_layers": trainer_config["model"]["enc_layers"],
        "dec_layers": trainer_config["model"]["dec_layers"],
        "nheads": trainer_config["model"]["nheads"],
        "camera_names": camera_names,
        "state_dim": 128,
        "action_dim": 128,
        "image_feature_strategy": trainer_config["model"]["image_feature_strategy"],
        "use_language_conditioning": trainer_config["model"]["use_language_conditioning"],
    }

    policy = ACTPolicy(policy_config)
    policy.eval()
    return policy

def predict_episode_to_hdf5(
    episode_hdf5_path,
    out_hdf5_path,
    ckpt_path,
    model_cfg_path,
    *,
    chunk_size=100,
    camera_name="top",
    max_steps=None,
    device=None,
):
    import torch
    import cv2
    import pickle
    from hdt.modeling.utils import make_visual_encoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = _load_act_policy(model_cfg_path, chunk_size, [camera_name]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    _, visual_preprocessor = make_visual_encoder("ACT", {})

    with h5py.File(episode_hdf5_path, "r") as root:
        embodiment = str(root.attrs.get("embodiment", "human_mocap_annotated"))
        ckpt_dir = os.path.dirname(ckpt_path)
        if os.path.basename(ckpt_dir).startswith("policy_iter_"):
            ckpt_dir = os.path.dirname(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found next to ckpt: {stats_path}")
        with open(stats_path, "rb") as f:
            loaded = pickle.load(f)
        norm_stats = loaded[0] if isinstance(loaded, tuple) and len(loaded) == 2 else loaded
        if embodiment not in norm_stats:
            raise KeyError(f"embodiment={embodiment} not found in dataset_stats.pkl: {list(norm_stats.keys())}")
        ns = norm_stats[embodiment]
        qpos_mean = ns["qpos_mean"].astype(np.float32)
        qpos_std = ns["qpos_std"].astype(np.float32)
        action_mean = ns["action_mean"].astype(np.float32)
        action_std = ns["action_std"].astype(np.float32)

        states = root["observation.state"][()]
        images = root[f"observation.image.{camera_name}"]
        T = states.shape[0]
        if max_steps is not None:
            T = min(T, int(max_steps))

        pred_actions = np.zeros((T, 128), dtype=np.float32)

        for t in range(T):
            img = images[t]
            if len(img.shape) == 1:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3:
                if img.shape[-1] == 3:
                    img = img
                else:
                    img = img[:, :, :3]
            else:
                raise ValueError(f"Unsupported image shape at t={t}: {img.shape}")

            if img.shape[0] != 240 or img.shape[1] != 320:
                img = cv2.resize(img, (320, 240))

            img_nchw = img.transpose(2, 0, 1)[None, ...].astype(np.uint8)
            img_tensor = visual_preprocessor(img_nchw).unsqueeze(0).to(device)

            qpos_raw = states[t].astype(np.float32)
            qpos_norm = (qpos_raw - qpos_mean) / (qpos_std + 1e-6)
            qpos = torch.from_numpy(qpos_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                a_hat = policy(img_tensor, qpos, conditioning_dict=None)
            if isinstance(a_hat, dict):
                raise ValueError("Policy returned a loss dict; expected trajectory predictions.")
            pred_norm = a_hat[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_actions[t] = pred_norm * (action_std + 1e-6) + action_mean

    os.makedirs(os.path.dirname(out_hdf5_path) or ".", exist_ok=True)
    with h5py.File(out_hdf5_path, "w") as f_out:
        f_out.create_dataset("action", data=pred_actions, compression="gzip", compression_opts=4)
        f_out.attrs["sim"] = np.bool_(False)
        f_out.attrs["embodiment"] = "predicted"
        f_out.attrs["description"] = f"predicted_from:{os.path.basename(episode_hdf5_path)}"

def _transform_points(points, transform_mat):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = np.dot(transform_mat, points_h.T).T
    return transformed[:, :3]

def _action_to_eval_joints(action_128: np.ndarray) -> dict:
    cmd = get_eef_kpts_from_prediction(action_128)
    head_mat = cmd["head_mat"]
    lw_mat = cmd["left_wrist_mat"]
    rw_mat = cmd["right_wrist_mat"]
    lk_local = cmd["left_hand_kpts"]
    rk_local = cmd["right_hand_kpts"]

    lk_world = _transform_points(lk_local, lw_mat)
    rk_world = _transform_points(rk_local, rw_mat)

    return {
        "head": head_mat[:3, 3].astype(np.float32),
        "lw": lw_mat[:3, 3].astype(np.float32),
        "rw": rw_mat[:3, 3].astype(np.float32),
        "lk_world": lk_world.astype(np.float32),
        "rk_world": rk_world.astype(np.float32),
    }

def evaluate_mpjpe(gt_hdf5_path: str, pred_hdf5_path: str, save_json_path: str | None = None) -> dict:
    with h5py.File(gt_hdf5_path, "r") as f_gt, h5py.File(pred_hdf5_path, "r") as f_pr:
        gt_actions = f_gt["action"][()]
        pr_actions = f_pr["action"][()]

    T = min(gt_actions.shape[0], pr_actions.shape[0])
    if T <= 0:
        raise ValueError("No overlapping timesteps between gt and prediction.")

    all_joint_err = []
    hand_err = []
    wrist_err = []
    head_err = []

    for t in range(T):
        gt_j = _action_to_eval_joints(gt_actions[t])
        pr_j = _action_to_eval_joints(pr_actions[t])

        gt_hand = np.concatenate([gt_j["lk_world"], gt_j["rk_world"]], axis=0)  # (12, 3)
        pr_hand = np.concatenate([pr_j["lk_world"], pr_j["rk_world"]], axis=0)
        gt_wrist = np.stack([gt_j["lw"], gt_j["rw"]], axis=0)  # (2, 3)
        pr_wrist = np.stack([pr_j["lw"], pr_j["rw"]], axis=0)
        gt_head = gt_j["head"][None, :]
        pr_head = pr_j["head"][None, :]

        gt_all = np.concatenate([gt_head, gt_wrist, gt_hand], axis=0)  # (15, 3)
        pr_all = np.concatenate([pr_head, pr_wrist, pr_hand], axis=0)

        all_joint_err.append(np.linalg.norm(pr_all - gt_all, axis=1))
        hand_err.append(np.linalg.norm(pr_hand - gt_hand, axis=1))
        wrist_err.append(np.linalg.norm(pr_wrist - gt_wrist, axis=1))
        head_err.append(np.linalg.norm(pr_head - gt_head, axis=1))

    all_joint_err = np.concatenate(all_joint_err, axis=0)
    hand_err = np.concatenate(hand_err, axis=0)
    wrist_err = np.concatenate(wrist_err, axis=0)
    head_err = np.concatenate(head_err, axis=0)

    metrics = {
        "timesteps_compared": int(T),
        "joints_per_timestep": 15,
        "mpjpe_all_m": float(np.mean(all_joint_err)),
        "mpjpe_hand_m": float(np.mean(hand_err)),
        "mpjpe_wrist_m": float(np.mean(wrist_err)),
        "mpjpe_head_m": float(np.mean(head_err)),
        "mpjpe_all_mm": float(np.mean(all_joint_err) * 1000.0),
        "mpjpe_hand_mm": float(np.mean(hand_err) * 1000.0),
        "mpjpe_wrist_mm": float(np.mean(wrist_err) * 1000.0),
        "mpjpe_head_mm": float(np.mean(head_err) * 1000.0),
    }

    if save_json_path is not None:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics

def main(input_file, *, full_hand=False, max_frames=None):
    # Processed HDF5
    datas = load_cmd_tuple_hdf5(input_file, full_hand=full_hand, max_frames=max_frames)
    
    # Prepare data for animation
    frames = []
    for data in datas:
        head_mat = np.array(data['head']).reshape(4, 4, order="F")
        right_wrist_mat = np.array(data['rightWrist']).reshape(4, 4, order="F")
        left_wrist_mat = np.array(data['leftWrist']).reshape(4, 4, order="F")
        
        right_fingers = np.array(data["rightSkeleton"]["joints"]).reshape(25, 4, 4)[:, 3, 0:3]
        left_fingers = np.array(data["leftSkeleton"]["joints"]).reshape(25, 4, 4)[:, 3, 0:3]
        
        # Extract positions (translation vectors) from transformation matrices
        head_pos = head_mat[:3, 3]
        right_wrist_pos = right_wrist_mat[:3, 3]
        left_wrist_pos = left_wrist_mat[:3, 3]
        
        # Extract rotation axes (first 3 columns of rotation matrix)
        def get_axes(mat, scale=0.1):
            R = mat[:3, :3]
            pos = mat[:3, 3]
            x_axis = pos + R[:, 0] * scale
            y_axis = pos + R[:, 1] * scale
            z_axis = pos + R[:, 2] * scale
            return pos, x_axis, y_axis, z_axis
        
        head_pos, head_x, head_y, head_z = get_axes(head_mat)
        rw_pos, rw_x, rw_y, rw_z = get_axes(right_wrist_mat)
        lw_pos, lw_x, lw_y, lw_z = get_axes(left_wrist_mat)
        
        # Transform finger positions from local wrist frame to world frame
        def transform_points(points, transform_mat):
            # Convert to homogeneous coordinates
            points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            # Transform points
            transformed = np.dot(transform_mat, points_h.T).T
            return transformed[:, :3]  # Return only xyz coordinates

        right_fingers_world = transform_points(right_fingers, right_wrist_mat)
        left_fingers_world = transform_points(left_fingers, left_wrist_mat)
        
        frame_data = {
            'positions': {
                'head': head_pos,
                'right_wrist': right_wrist_pos,
                'left_wrist': left_wrist_pos
            },
            'axes': {
                'head': (head_x, head_y, head_z),
                'right_wrist': (rw_x, rw_y, rw_z),
                'left_wrist': (lw_x, lw_y, lw_z)
            },
            'fingers': {
                'right': right_fingers_world,
                'left': left_fingers_world
            }
        }
        frames.append(frame_data)
    
    # Create figure
    fig = go.Figure()
    
    # Add initial positions
    colors = {'head': 'blue', 'right_wrist': 'red', 'left_wrist': 'green'}
    axis_colors = ['red', 'green', 'blue']  # x, y, z axes colors
    
    def add_coordinate_frame(pos, axes, name, base_color):
        # Add position marker
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            name=f"{name}_position",
            marker=dict(size=8, color=base_color)
        ))
        
        # Add coordinate axes
        for i, (axis_end, color) in enumerate(zip(axes, axis_colors)):
            fig.add_trace(go.Scatter3d(
                x=[pos[0], axis_end[0]],
                y=[pos[1], axis_end[1]],
                z=[pos[2], axis_end[2]],
                mode='lines',
                name=f"{name}_{['x', 'y', 'z'][i]}_axis",
                line=dict(color=color, width=3)
            ))
    
    def add_hand_keypoints(pos, fingers, name, color):
        # Add finger keypoints
        fig.add_trace(go.Scatter3d(
            x=fingers[:, 0], y=fingers[:, 1], z=fingers[:, 2],
            mode='markers',
            name=f"{name}_fingers",
            marker=dict(size=4, color=color, opacity=0.7)
        ))
    
    # Initial frame
    first_frame = frames[0]
    # Add origin coordinate frame
    
    
    for part in ['head', 'right_wrist', 'left_wrist']:
        pos = first_frame['positions'][part]
        axes = first_frame['axes'][part]
        add_coordinate_frame(pos, axes, part, colors[part])
    
    # Add initial hand keypoints
    add_hand_keypoints(first_frame['positions']['right_wrist'], 
                      first_frame['fingers']['right'], 
                      'right', colors['right_wrist'])
    add_hand_keypoints(first_frame['positions']['left_wrist'], 
                      first_frame['fingers']['left'], 
                      'left', colors['left_wrist'])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.73, y=1.0, z=0.5)
            ),
            # xaxis=dict(range=[-0.3, 0.3]),
            # yaxis=dict(range=[0, 1.4]),
            # zaxis=dict(range=[-0.5, 0.1]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="3D Transformation Visualization",
        margin=dict(l=0, r=0, t=36, b=0),
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons',
            'direction': 'left',
            'showactive': True
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(i),
                'method': 'animate'
            } for i in range(len(frames))]
        }]
    )
    
    # Create animation frames
    fig_frames = []
    for i, frame in enumerate(frames):
        frame_traces = []
        
        # Add origin to each frame
        # frame_traces.append(go.Scatter3d(
        #     x=[0], y=[0], z=[0],
        #     mode='markers',
        #     marker=dict(size=8, color='black')
        # ))
        
        # Add origin axes to each frame
        # for axis_end, color in zip(origin_axes, axis_colors):
        #     frame_traces.append(go.Scatter3d(
        #         x=[0, axis_end[0]],
        #         y=[0, axis_end[1]],
        #         z=[0, axis_end[2]],
        #         mode='lines',
        #         line=dict(color=color, width=3)
        #     ))
            
        for part in ['head', 'right_wrist', 'left_wrist']:
            pos = frame['positions'][part]
            axes = frame['axes'][part]
            
            # Position marker
            frame_traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=8, color=colors[part])
            ))
            
            # Coordinate axes
            for axis_end, color in zip(axes, axis_colors):
                frame_traces.append(go.Scatter3d(
                    x=[pos[0], axis_end[0]],
                    y=[pos[1], axis_end[1]],
                    z=[pos[2], axis_end[2]],
                    mode='lines',
                    line=dict(color=color, width=3)
                ))
        
        # Add finger keypoints to each frame
        frame_traces.append(go.Scatter3d(
            x=frame['fingers']['right'][:, 0],
            y=frame['fingers']['right'][:, 1],
            z=frame['fingers']['right'][:, 2],
            mode='markers',
            marker=dict(size=4, color=colors['right_wrist'], opacity=0.7)
        ))
        
        frame_traces.append(go.Scatter3d(
            x=frame['fingers']['left'][:, 0],
            y=frame['fingers']['left'][:, 1],
            z=frame['fingers']['left'][:, 2],
            mode='markers',
            marker=dict(size=4, color=colors['left_wrist'], opacity=0.7)
        ))
        
        fig_frames.append(go.Frame(data=frame_traces, name=str(i)))
    
    fig.frames = fig_frames
    
    return fig, frames

def _save_html(fig, out_path):
    import plotly.io as pio

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pio.write_html(fig, out_path, auto_open=False, include_plotlyjs="cdn")

def _save_mp4(fig, out_path, *, fps=20, width=960, height=720, frames=None):
    import subprocess
    import tempfile

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg to export mp4.")

    if frames is not None:
        _save_mp4_matplotlib(frames, out_path, fps=fps, width=width, height=height)
        return

    # Fallback: kaleido-based rendering (slow)
    import plotly.io as pio
    try:
        _ = pio.kaleido.scope
    except Exception as e:
        raise RuntimeError("plotly kaleido is required to export video frames. Please install kaleido.") from e

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fr in enumerate(fig.frames):
            frame_fig = go.Figure(data=fr.data, layout=fig.layout)
            png_bytes = frame_fig.to_image(format="png", width=width, height=height, scale=1)
            frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            with open(frame_path, "wb") as f:
                f.write(png_bytes)

        cmd = (
            f"ffmpeg -y -framerate {int(fps)} -i {os.path.join(tmpdir, 'frame_%06d.png')} "
            f"-c:v libx264 -pix_fmt yuv420p {out_path}"
        )
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def _save_mp4_matplotlib(frames, out_path, *, fps=20, width=960, height=720):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import subprocess
    import tempfile

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    colors = {'head': 'blue', 'right_wrist': 'red', 'left_wrist': 'green'}
    axis_colors = ['red', 'green', 'blue']

    all_pts = np.concatenate(
        [np.stack([f['positions'][p] for p in ('head', 'right_wrist', 'left_wrist')])
         for f in frames], axis=0
    )
    margin = 0.1
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    with tempfile.TemporaryDirectory() as tmpdir:
        from tqdm import tqdm
        for i, frame in enumerate(tqdm(frames, desc="Rendering frames", unit="frame")):
            fig_m = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax = fig_m.add_subplot(111, projection='3d')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.view_init(elev=15, azim=210)  # View from -x with y+ on the left.
            ax.set_title(f"Frame {i}")
            fig_m.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)

            for part in ('head', 'right_wrist', 'left_wrist'):
                pos = frame['positions'][part]
                axes_ends = frame['axes'][part]
                c = colors[part]
                ax.scatter(*pos, color=c, s=60, zorder=5, label=f"{part} position")
                for axis_end, ac, ax_name in zip(axes_ends, axis_colors, ('x', 'y', 'z')):
                    ax.plot([pos[0], axis_end[0]], [pos[1], axis_end[1]], [pos[2], axis_end[2]],
                            color=ac, linewidth=2, label=f"{part}_{ax_name}_axis")

            for side, c in (('right', colors['right_wrist']), ('left', colors['left_wrist'])):
                pts = frame['fingers'][side]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=15, alpha=0.7,
                           label=f"{side}_fingers")

            ax.legend(loc='upper left', fontsize=6, markerscale=0.8)

            frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            fig_m.savefig(frame_path, dpi=dpi)
            plt.close(fig_m)

        cmd = (
            f"ffmpeg -y -framerate {int(fps)} -i {os.path.join(tmpdir, 'frame_%06d.png')} "
            f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
            f"-c:v libx264 -pix_fmt yuv420p {out_path}"
        )
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot processed data from HDF5 file')
    parser.add_argument('--file', '-f', type=str,
                        default="/home/embodied/human-policy/data/processed_episode_0.hdf5",
                        help='Path to the processed/predicted HDF5 file (must contain dataset: action)')
    parser.add_argument('--predict_episode', type=str, default=None,
                        help='If set, run policy on this episode HDF5 and write predictions to --out, then visualize --out')
    parser.add_argument('--out', type=str, default="/home/embodied/human-policy/data/predicted_episode.hdf5",
                        help='Output HDF5 path for predicted actions')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint state_dict (e.g., .../pytorch_model.bin or policy_last.ckpt)')
    parser.add_argument('--model_cfg', type=str, default="/home/embodied/human-policy/hdt/configs/models/act_resnet.yaml",
                        help='Path to model yaml (ACT config)')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size (num_queries) used by ACT')
    parser.add_argument('--camera', type=str, default="top", help='Camera name inside episode hdf5: observation.image.<camera>')
    parser.add_argument('--max_steps', type=int, default=None, help='Limit number of steps to predict')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default auto)')
    parser.add_argument('--full_hand', action='store_true', help='Expand each hand from 5 fingertip keypoints to an approximate 25-joint chain for visualization')
    parser.add_argument('--save_html', type=str, default=None, help='If set, save the interactive visualization to an HTML file')
    parser.add_argument('--save_mp4', type=str, default=None, help='If set, export the animation to an MP4 file (requires kaleido + ffmpeg)')
    parser.add_argument('--fps', type=int, default=20, help='FPS for MP4 export')
    parser.add_argument('--max_seconds', type=float, default=None, help='Limit visualization/export to the first N seconds, using --fps to convert seconds to frames')
    parser.add_argument('--width', type=int, default=960, help='Frame width for MP4 export')
    parser.add_argument('--height', type=int, default=720, help='Frame height for MP4 export')
    parser.add_argument('--eval_mpjpe', action='store_true', help='Evaluate MPJPE between prediction and ground-truth episode actions')
    parser.add_argument('--gt_file', type=str, default=None, help='Ground-truth episode file for MPJPE. Default: --predict_episode')
    parser.add_argument('--metrics_out', type=str, default=None, help='Optional output JSON path for metrics')
    args = parser.parse_args()
    max_frames = None
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise ValueError("--max_seconds must be positive")
        max_frames = int(np.ceil(args.max_seconds * args.fps))

    if args.predict_episode is not None:
        if args.ckpt is None:
            raise ValueError("--ckpt is required when --predict_episode is set")
        predict_episode_to_hdf5(
            args.predict_episode,
            args.out,
            args.ckpt,
            args.model_cfg,
            chunk_size=args.chunk_size,
            camera_name=args.camera,
            max_steps=args.max_steps,
            device=args.device,
        )
        fig, frames = main(args.out, full_hand=args.full_hand, max_frames=max_frames)
        if args.eval_mpjpe:
            gt_file = args.gt_file if args.gt_file is not None else args.predict_episode
            metrics = evaluate_mpjpe(gt_file, args.out, save_json_path=args.metrics_out)
            print("===== MPJPE Metrics =====")
            for k, v in metrics.items():
                print(f"{k}: {v}")
    else:
        fig, frames = main(args.file, full_hand=args.full_hand, max_frames=max_frames)
        if args.eval_mpjpe:
            if args.gt_file is None:
                raise ValueError("--gt_file is required when using --eval_mpjpe without --predict_episode")
            metrics = evaluate_mpjpe(args.gt_file, args.file, save_json_path=args.metrics_out)
            print("===== MPJPE Metrics =====")
            for k, v in metrics.items():
                print(f"{k}: {v}")

    if args.save_html is not None:
        _save_html(fig, args.save_html)
    if args.save_mp4 is not None:
        _save_mp4(fig, args.save_mp4, fps=args.fps, width=args.width, height=args.height, frames=frames)

    if args.save_html is None and args.save_mp4 is None:
        fig.show()