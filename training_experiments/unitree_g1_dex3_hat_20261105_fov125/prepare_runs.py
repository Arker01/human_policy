"""Prepare the visual-data retrain against the recorded October 31 experiment."""
import copy
import json
from pathlib import Path
import shutil

import cv2
import h5py
import numpy as np
import yaml


def prepare_runs(root, repo, checkpoint_root, python, tasks, stamp, write_json, digest):
    if (root / 'manifest.json').exists():
        raise RuntimeError('Already prepared; refusing to overwrite.')
    previous_root = repo / 'training_experiments/unitree_g1_dex3_hat_20261031'
    previous = json.loads((previous_root / 'manifest.json').read_text())
    source_hashes = {str(p.relative_to(root / 'source')): digest(p)
                     for p in sorted((root / 'source').rglob('*')) if p.is_file()}
    assert source_hashes == previous['source_hashes'], 'Training source must match previous run exactly'
    baseline_command = previous['baseline']['command']
    model_source = Path(baseline_command[baseline_command.index('--model_cfg_path') + 1])
    assert digest(model_source) == previous['model_sha256']
    shutil.copyfile(model_source, root / 'configs/model.yaml')
    model = yaml.safe_load(model_source.read_text())
    assert model['model']['zero_state_dims'] == '98:128'
    assert model['common']['camera_names'] == ['top']
    assert model['common']['action_dim'] == model['common']['state_dim'] == 128
    assert model['model']['backbone'] == 'vjepa2_1_vitb'
    data_root = Path('/mnt/nvme0n1/lirui_backup/public_data')
    folders = dict(block_stacking='block', pick_apple='pick_apple', pouring='pour')
    jobs, audits = {}, {}
    for task, gpu in zip(tasks, [2, 3, 4]):
        old_job = previous['jobs'][task]
        config = json.loads(Path(old_job['dataset_config']).read_text())
        assert digest(old_job['dataset_config']) == old_job['dataset_sha256']
        new_dir = data_root / folders[task] / 'episodes_g1_fov125'
        processing_path = new_dir / 'processing_report.json'
        processing = json.loads(processing_path.read_text())
        assert processing['complete'] is True
        report_files = {entry['file']: entry for entry in processing['files']}
        split_names = {split: [name for entry in config[split] for name in entry['file_list']]
                       for split in ['train', 'val']}
        expected = split_names['train'] + split_names['val']
        assert len(expected) == len(set(expected))
        assert set(expected) == set(p.name for p in new_dir.glob('*.hdf5')) == set(report_files)
        counts = {split: len(names) for split, names in split_names.items()}
        assert counts['train'] == old_job['train_episodes']
        assert counts['val'] == old_job['val_episodes']
        frames, decoded = 0, 0
        max_left_range = np.zeros(3)
        for split in ['train', 'val']:
            for entry in config[split]:
                old_dir = Path(entry['dataset_path'])
                for name in entry['file_list']:
                    path = new_dir / name
                    record = report_files[name]
                    assert path.stat().st_size == record['output_bytes'], path
                    with h5py.File(path, 'r') as f, h5py.File(old_dir / name, 'r') as old:
                        length = len(f['action'])
                        assert length == record['frames']
                        assert str(f.attrs['embodiment']) == 'dex3'
                        assert float(f.attrs['fps']) == 30.0
                        assert bool(f.attrs['conversion_complete'])
                        transform = json.loads(f.attrs['visual_preprocessing'])
                        assert transform == processing['transform']
                        assert f['observation.image.top'].id == f['observation.image.left'].id
                        assert f['observation.image_length.top'].id == f['observation.image_length.left'].id
                        for key in ['action', 'observation.state']:
                            array = f[key][()]
                            assert array.shape == (length, 128) and array.dtype == np.float32
                            assert np.isfinite(array).all(), (path, key)
                            np.testing.assert_array_equal(array, old[key][()], err_msg=f'{path}: {key}')
                        max_left_range = np.maximum(max_left_range, np.ptp(f['action'][:, 80:83], axis=0))
                        valid = f['representation_valid'][()]
                        assert valid[80:89].all() and valid[13:22].all(), path
                        np.testing.assert_array_equal(valid, old['representation_valid'][()])
                        for t in sorted({0, length // 2, length - 1}):
                            image = cv2.imdecode(f['observation.image.top'][t], cv2.IMREAD_COLOR)
                            assert image is not None and image.shape == (240, 320, 3), (path, t)
                            decoded += 1
                        frames += length
                entry['dataset_path'] = str(new_dir)
        assert frames == processing['frames'] and len(expected) == processing['episodes']
        # Check complete content provenance on one episode per task, in addition
        # to the all-episode numeric/schema and JPEG checks above.
        sample_name = sorted(expected)[0]
        assert digest(new_dir / sample_name) == report_files[sample_name]['output_sha256']
        config_path = root / f'configs/{task}.json'
        write_json(config_path, config)
        replacements = {
            '--exptid': str(checkpoint_root / task),
            '--dataset_json_path': str(config_path),
            '--model_cfg_path': str(root / 'configs/model.yaml'),
        }
        command = old_job['command'].copy()
        command[0] = python
        command[2] = str(root / 'source/hdt/main.py')
        for key, value in replacements.items():
            command[command.index(key) + 1] = value
        restored = command.copy()
        restored[2] = old_job['command'][2]
        for key in replacements:
            restored[restored.index(key) + 1] = old_job['command'][old_job['command'].index(key) + 1]
        assert restored == old_job['command']
        assert command[command.index('--zero_state_dims') + 1] == '98:128'
        jobs[task] = dict(task=task, gpu_index=gpu, command=command,
                          dataset_config=str(config_path), dataset_sha256=digest(config_path),
                          checkpoint_dir=str(checkpoint_root / (task + '_ckpt')),
                          log=str(root / f'logs/{task}.log'),
                          status_file=str(root / f'status/{task}.json'),
                          train_episodes=counts['train'], val_episodes=counts['val'], status='prepared')
        audits[task] = dict(dataset_path=str(new_dir), episodes=len(expected), frames=frames,
                            **counts, jpeg_frames_decoded=decoded, state_action_exactly_equal_previous=True,
                            train_val_membership_exactly_equal_previous=True, no_train_val_overlap=True,
                            all_files_covered=True, all_numeric_values_finite=True,
                            left_eef_and_three_fingers_valid=True,
                            max_left_eef_range_m=max_left_range.tolist(),
                            transform=processing['transform'], processing_report_sha256=digest(processing_path),
                            content_hash_checked_episode=sample_name)
        print(f'PASS {task}: {len(expected)} episodes, {frames} frames, {decoded} JPEGs, GPU {gpu}', flush=True)
    weights = copy.deepcopy(previous['pretrained_weights'])
    for path, record in weights.items():
        assert Path(path).is_file() and digest(path) == record['sha256'], path
    write_json(root / 'data_preflight.json', dict(status='pass', checked_at=stamp(), tasks=audits))
    manifest = dict(status='prepared', prepared_at=stamp(), previous_experiment=str(previous_root),
                    previous_manifest_sha256=digest(previous_root / 'manifest.json'),
                    user_selected_configuration='same standard HAT as 20261031, updated visual data; retain both hands',
                    checkpoint_root=str(checkpoint_root), source_hashes=source_hashes,
                    model_sha256=digest(root / 'configs/model.yaml'), pretrained_weights=weights,
                    same_training_arguments_verified=True, same_training_source_verified=True,
                    same_model_bytes_verified=True,
                    model_provenance='Restored from original baseline YAML matching previous recorded SHA256; old experiment YAML has since been commented/edited.',
                    state_zero_range='98:128; excludes both hands',
                    left_hand_input_indices=list(range(10, 28)) + list(range(80, 89)),
                    left_hand_supervision='Kept for all tasks; existing ACT loss supervises both hands. No right-hand-only mask.',
                    initialization=previous['initialization'], jobs=jobs)
    write_json(root / 'manifest.json', manifest)
    print('Prepared all three runs; model, source, training arguments and split memberships match previous run.', flush=True)
