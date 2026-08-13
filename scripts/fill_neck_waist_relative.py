#!/usr/bin/env python3

import argparse
import glob
import h5py
import numpy as np
import os

def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = a1 / np.linalg.norm(a1) if np.linalg.norm(a1) > 1e-8 else a1
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / np.linalg.norm(b2) if np.linalg.norm(b2) > 1e-8 else b2
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1).astype(np.float32)
    return R

def _mat_to_rot6d(R: np.ndarray) -> np.ndarray:
    return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)

def compute_relative_transform(head_pos, head_rot6d, target_pos, target_rot6d):
    rel_pos = target_pos - head_pos
    rel_rot6d = target_rot6d - head_rot6d
    return rel_pos, rel_rot6d

def apply_relative_transform(head_pos, head_rot6d, rel_pos, rel_rot6d):
    target_pos = head_pos + rel_pos
    target_rot6d = head_rot6d + rel_rot6d
    return target_pos, target_rot6d

def get_reference_offsets(reference_file):
    print(f"Getting offsets from: {os.path.basename(reference_file)}")
    with h5py.File(reference_file, 'r') as f:
        action = f['action'][()]
        head_pos = action[0, 0:3]
        head_rot6d = action[0, 3:9]
        neck_pos = action[0, 58:61]
        neck_rot6d = action[0, 61:67]
        waist_pos = action[0, 89:92]
        waist_rot6d = action[0, 92:98]
    neck_rel_pos, neck_rel_rot6d = compute_relative_transform(
        head_pos, head_rot6d, neck_pos, neck_rot6d
    )
    waist_rel_pos, waist_rel_rot6d = compute_relative_transform(
        head_pos, head_rot6d, waist_pos, waist_rot6d
    )
    print(f"Head pos: {head_pos}")
    print(f"Neck rel pos: {neck_rel_pos}")
    print(f"Waist rel pos: {waist_rel_pos}")
    return {
        'neck': (neck_rel_pos, neck_rel_rot6d),
        'waist': (waist_rel_pos, waist_rel_rot6d)
    }

def process_file(filepath, offsets, output_dir):
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)
    with h5py.File(filepath, 'r') as f:
        action = f['action'][()].copy()
        datasets = {}
        for key in f.keys():
            datasets[key] = f[key][()]
    neck_rel_pos, neck_rel_rot6d = offsets['neck']
    waist_rel_pos, waist_rel_rot6d = offsets['waist']
    num_frames = action.shape[0]
    for t in range(num_frames):
        head_pos = action[t, 0:3]
        head_rot6d = action[t, 3:9]
        neck_pos, neck_rot6d = apply_relative_transform(
            head_pos, head_rot6d, neck_rel_pos, neck_rel_rot6d
        )
        waist_pos, waist_rot6d = apply_relative_transform(
            head_pos, head_rot6d, waist_rel_pos, waist_rel_rot6d
        )
        action[t, 58:61] = neck_pos
        action[t, 61:67] = neck_rot6d
        action[t, 89:92] = waist_pos
        action[t, 92:98] = waist_rot6d
    with h5py.File(output_path, 'w') as f:
        f['action'] = action
        for key, value in datasets.items():
            if key != 'action':
                f[key] = value
    return filename

def main():
    parser = argparse.ArgumentParser(description='Fill neck and waist positions for wholebody data')
    parser.add_argument('--reference-dir', type=str,
                        default='/home/aigc/human_policy/data/all_episodes',
                        help='Reference data directory')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Input data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output data directory')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    ref_files = sorted(glob.glob(os.path.join(args.reference_dir, '*.hdf5')))
    if not ref_files:
        raise ValueError(f"No hdf5 files found in reference dir: {args.reference_dir}")
    reference_file = ref_files[0]
    offsets = get_reference_offsets(reference_file)
    input_files = sorted(glob.glob(os.path.join(args.input_dir, '*.hdf5')))
    print(f"\nFound {len(input_files)} files to process")
    processed_count = 0
    for filepath in input_files:
        print(f"Processing: {os.path.basename(filepath)}")
        try:
            process_file(filepath, offsets, args.output_dir)
            processed_count += 1
        except Exception as e:
            print(f"  Failed: {e}")
    print(f"\nProcessing complete! Total processed: {processed_count}")
    print(f"Output directory: {args.output_dir}")

if __name__ == '__main__':
    main()