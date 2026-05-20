import h5py
import numpy as np
import os
from pathlib import Path

COORD_TRANSFORM_MATRIX = np.array([
    [1, 0, 0],
    [0, -1, 0],
    [0, 0, 1]
], dtype=np.float32)

def transform_coord(points):
    if points.shape[-1] != 3:
        raise ValueError(f"Expected points with shape (..., 3), got {points.shape}")
    orig_shape = points.shape
    points_2d = points.reshape(-1, 3)
    transformed = np.dot(points_2d, COORD_TRANSFORM_MATRIX.T)
    return transformed.reshape(orig_shape)

def convert_action(action_128):
    new_action = action_128.copy()
    # 转换头部位置
    new_action[0:3] = transform_coord(action_128[0:3].reshape(1, 3)).flatten()
    
    # 转换右手腕和手指位置
    for i in range(30, 46, 3):
        if i+2 < 128:
            new_action[i:i+3] = transform_coord(action_128[i:i+3].reshape(1, 3)).flatten()
    
    # 转换左手腕和手指位置
    for i in range(80, 96, 3):
        if i+2 < 128:
            new_action[i:i+3] = transform_coord(action_128[i:i+3].reshape(1, 3)).flatten()
    
    return new_action

def convert_hdf5(input_path, output_path):
    with h5py.File(input_path, 'r') as f_in:
        actions = f_in['action'][:]
        images = f_in['observation.image.top'][:] if 'observation.image.top' in f_in else None
        states = f_in['observation.state'][:] if 'observation.state' in f_in else None

        converted_actions = np.zeros_like(actions)
        for i in range(len(actions)):
            converted_actions[i] = convert_action(actions[i])

        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        with h5py.File(output_path, 'w') as f_out:
            f_out.create_dataset('action', data=converted_actions, compression='gzip', compression_opts=4)
            if images is not None:
                f_out.create_dataset('observation.image.top', data=images, compression='gzip', compression_opts=4)
            if states is not None:
                f_out.create_dataset('observation.state', data=states, compression='gzip', compression_opts=4)
            for key in f_in.attrs:
                f_out.attrs[key] = f_in.attrs[key]
            f_out.attrs['coord_transform'] = 'x,-y,z'

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Convert HDF5 data to (x,-y,z) coordinate system')
    parser.add_argument('input_dir', help='Input directory containing HDF5 files')
    parser.add_argument('output_dir', help='Output directory for converted HDF5 files')
    parser.add_argument('--limit', type=int, default=None, help='Limit number of files to convert')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    hdf5_files = sorted(input_dir.glob('episode_*.hdf5'))
    
    if args.limit:
        hdf5_files = hdf5_files[:args.limit]
    
    print(f'Found {len(hdf5_files)} HDF5 files in {input_dir}')

    for i, f in enumerate(hdf5_files):
        output_path = output_dir / f.name
        print(f'[{i+1}/{len(hdf5_files)}] Converting {f.name} -> {output_path}')
        convert_hdf5(str(f), str(output_path))

    print('Done!')

if __name__ == '__main__':
    main()