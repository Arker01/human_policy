import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import h5py

from hdt.inference_utils import get_eef_kpts_from_prediction

def load_cmd_tuple_hdf5(path):
    data_list = []

    with h5py.File(path, 'r') as file:
        # Processed HDF5
        assert "/action" in file
        for i in range(file["/action"].shape[0]):
            cur_cmd_dict = get_eef_kpts_from_prediction(file["/action"][i])
            # Format post-processed data to match expected structure
            head_mat = cur_cmd_dict['head_mat']
            left_wrist_mat = cur_cmd_dict['left_wrist_mat']
            right_wrist_mat = cur_cmd_dict['right_wrist_mat']
            left_hand_kpts = cur_cmd_dict['left_hand_kpts']
            right_hand_kpts = cur_cmd_dict['right_hand_kpts']
            
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

def main(input_file):
    # Processed HDF5
    datas = load_cmd_tuple_hdf5(input_file)
    
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
                up=dict(x=0, y=1, z=0),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
            # xaxis=dict(range=[-0.3, 0.3]),
            # yaxis=dict(range=[0, 1.4]),
            # zaxis=dict(range=[-0.5, 0.1]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="3D Transformation Visualization",
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
    
    # Show the figure
    fig.show()
    
    return frames

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot processed data from HDF5 file')
    parser.add_argument('--file', '-f', type=str, 
                        default="./recordings/processed/303-grasp_coke_random-2024_12_12-19_13_53/processed_episode_10.hdf5",
                        help='Path to the processed HDF5 file')
    args = parser.parse_args()

    main(args.file)
