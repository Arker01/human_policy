"""Update existing HDF5 right-hand tips from source feedback, without reconversion."""
import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pyarrow.parquet as pq

from convert_sonic_to_hat_hdf5 import (
    RIGHT_HAND_URDF, RIGHT_TIP_NAMES, RIGHT_TIPS,
    hand_tips, load_hand_model, pack_tips,
)

TRANSFORM = 'dds_motor6_coupled_lut_v1_open1_right_thumb_rot_1minuscmd'


def untouched_digest(hf):
    """Hash other columns, all auxiliary datasets, and compressed image bytes."""
    digest = hashlib.sha256()
    mask = np.ones(128, dtype=bool)
    mask[RIGHT_TIPS] = False
    for name in ['observation.state', 'action']:
        digest.update(hf[name][:, mask].tobytes())

    def visit(name, obj):
        if not isinstance(obj, h5py.Dataset) or name in ['observation.state', 'action']:
            return
        digest.update(name.encode())
        value = obj[()]
        if value.dtype.kind == 'O':
            for row in value.flat:
                digest.update(np.asarray(row).tobytes())
        else:
            digest.update(value.tobytes())
    hf.visititems(visit)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.output_dir.glob('SONIC_Dex5_20260930_ep_*_rootfix.hdf5'))
    if len(files) != 79 or {int(p.stem.split('_ep_')[1].split('_')[0]) for p in files} != set(range(79)):
        raise RuntimeError('Expected exactly the 79 official episodes 0-78')
    # Resolve every source and validate dimensions before any overwrite.
    sources = {}
    for path in files:
        with h5py.File(path, 'r') as hf:
            source = Path(hf.attrs['source_parquet'])
            if not source.is_file():
                raise FileNotFoundError(source)
            cmd = np.asarray(pq.read_table(source, columns=['observation.inspire.right_position'])[
                'observation.inspire.right_position'].to_pylist(), dtype=np.float32)
            if cmd.shape != (len(hf['observation.state']), 6):
                raise ValueError(f'Frame mismatch: {path}')
            if hf['action'].shape != (len(cmd), 128) or not np.isfinite(cmd).all():
                raise ValueError(f'Invalid episode: {path}')
            if np.any(cmd < 0) or np.any(cmd > 1):
                raise ValueError(f'Out-of-range feedback: {path}')
            sources[path] = cmd
    model, data, ids = load_hand_model(RIGHT_HAND_URDF, RIGHT_TIP_NAMES)
    total = 0
    for index, path in enumerate(files, 1):
        cmd = sources[path].copy()
        cmd[:, 5] = 1 - cmd[:, 5]
        tips = pack_tips(hand_tips(cmd, model, data, ids))
        with h5py.File(path, 'r+') as hf:
            before = untouched_digest(hf)
            old_state = hf['observation.state'][:, RIGHT_TIPS]
            old_action = hf['action'][:, RIGHT_TIPS]
            state = hf['observation.state'][:]
            state[:, RIGHT_TIPS] = tips
            action_tips = np.concatenate([tips[1:], tips[-1:]], axis=0)
            if not np.array_equal(hf['action'][:, :10], np.concatenate([state[1:, :10], state[-1:, :10]], axis=0)):
                raise ValueError(f'Unexpected action alignment: {path}')
            try:
                hf['observation.state'][:, RIGHT_TIPS] = tips
                hf['action'][:, RIGHT_TIPS] = action_tips
                hf.flush()
                assert np.array_equal(hf['observation.state'][:, RIGHT_TIPS], tips)
                assert np.array_equal(hf['action'][:, RIGHT_TIPS], action_tips)
                assert before == untouched_digest(hf), f'Unrelated data changed: {path}'
            except BaseException:
                hf['observation.state'][:, RIGHT_TIPS] = old_state
                hf['action'][:, RIGHT_TIPS] = old_action
                hf.flush()
                raise
            hf.attrs['inspire_transform'] = TRANSFORM
            hf.attrs['inspire_right_thumb_rot_preprocess'] = 'cmd[:,5] = 1 - source_feedback[:,5]'
            hf.attrs['inspire_left_transform'] = 'dds_motor6_coupled_lut_v1_open1'
            hf.attrs['inspire_right_transform'] = TRANSFORM
            hf.attrs['inspire_hand_update_date'] = '2026-10-03'
            total += len(tips)
        print(f'[{index}/79] {path.name}: {len(tips)} frames, other datasets unchanged', flush=True)
    report_path = args.output_dir / 'conversion_report.json'
    report = json.loads(report_path.read_text())
    update = dict(transform=TRANSFORM, side='right', motor_index=5,
                  preprocessing='1 - original parquet feedback', updated_episodes=79,
                  updated_frames=total, unrelated_datasets_sha256_verified=True,
                  existing_mp4_files_regenerated=False, date='2026-10-03')
    report['inspire_hand_update'] = update
    for episode in report.get('episodes', []):
        episode['inspire_transform'] = TRANSFORM
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(f'Done: 79 episodes, {total} frames; overwritten in place.', flush=True)


if __name__ == '__main__':
    main()
