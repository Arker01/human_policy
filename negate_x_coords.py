#!/usr/bin/env python3
"""将task1_convert目录下的HDF5文件中头部和手腕位置的x坐标取相反数"""

import h5py
import numpy as np
import os
from pathlib import Path

def negate_x_in_action(action):
    new_action = action.copy()
    # 头部位置 x: index 0
    new_action[:, 0] = -action[:, 0]
    # 右手腕位置 x: index 30
    new_action[:, 30] = -action[:, 30]+0.5
    # 左手腕位置 x: index 80
    new_action[:, 80] = -action[:, 80]+0.5
    return new_action

def negate_x_in_state(state):
    new_state = state.copy()
    # 头部位置 x: index 0
    new_state[:, 0] = -state[:, 0]
    # 右手腕位置 x: index 30
    new_state[:, 30] = -state[:, 30]
    # 左手腕位置 x: index 80
    new_state[:, 80] = -state[:, 80]
    return new_state

def process_file(input_path, output_path):
    with h5py.File(input_path, 'r') as f_in:
        actions = f_in['action'][:]
        states = f_in['observation.state'][:] if 'observation.state' in f_in else None
        images = f_in['observation.image.top'][:] if 'observation.image.top' in f_in else None

        # 对 x 坐标取相反数
        negated_actions = negate_x_in_action(actions)
        negated_states = negate_x_in_state(states) if states is not None else None

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('action', data=negated_actions, compression='gzip', compression_opts=4)
            if images is not None:
                f_out.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
            if negated_states is not None:
                f_out.create_dataset('observation.state', data=negated_states, compression='gzip', compression_opts=4)
            # 复制属性
            for key in f_in.attrs:
                f_out.attrs[key] = f_in.attrs[key]
            f_out.attrs['x_negated'] = True

def main():
    input_dir = Path('/home/embodied/human-policy/data/task1_convert')
    output_dir = Path('/data1/zxlei/dataset/task1_convert_negated_x')
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(input_dir.glob('episode_*.hdf5'))
    print(f'Found {len(hdf5_files)} HDF5 files in {input_dir}')

    for i, f in enumerate(hdf5_files):
        output_path = output_dir / f.name
        print(f'[{i+1}/{len(hdf5_files)}] Processing {f.name}')
        process_file(str(f), str(output_path))

    print('Done!')

if __name__ == '__main__':
    main()