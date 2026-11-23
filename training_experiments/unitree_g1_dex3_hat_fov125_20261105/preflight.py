"""Read-only corpus audit and regression checks against the historical source."""
from pathlib import Path
import hashlib
import json
import sys


TASK_FOLDERS = {'block_stacking': 'block', 'pick_apple': 'pick_apple', 'pouring': 'pour'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def audit_data(root, previous, data_root):
    import cv2
    import h5py
    import numpy as np

    result = {'status': 'pass', 'tasks': {}}
    for task, folder in TASK_FOLDERS.items():
        directory = data_root / folder / 'episodes_g1_fov125'
        old_config = previous / f'configs/{task}.json'
        old = json.loads(old_config.read_text())
        new = json.loads((directory / 'dataset_config.json').read_text())
        report = json.loads((directory / 'processing_report.json').read_text())
        assert report['complete'] and not (directory / 'INCOMPLETE').exists(), directory
        assert report['split_source_sha256'] == sha256(old_config)
        assert report['transform']['version'] == 'g1_fov125_v1'
        expected, splits = {}, {}
        for split in ('train', 'val'):
            assert len(new[split]) == len(old[split])
            names = []
            for before, after in zip(old[split], new[split]):
                assert before['file_list'] == after['file_list'], (task, split)
                for name in before['file_list']:
                    expected[name] = Path(before['dataset_path']) / name
                    names.append(name)
            assert len(names) == len(set(names))
            splits[split] = set(names)
        assert not (splits['train'] & splits['val'])
        assert set(expected) == {p.name for p in directory.glob('*.hdf5')}
        rows = {r['file']: r for r in report['files']}
        assert set(rows) == set(expected)
        frames = samples = 0
        left_state_range = np.zeros(27)
        left_action_range = np.zeros(27)
        left_dims = np.r_[10:28, 80:89]
        for ordinal, (name, source) in enumerate(sorted(expected.items()), 1):
            path = directory / name
            row = rows[name]
            assert path.stat().st_size == row['output_bytes'], path
            assert sha256(path) == row['output_sha256'], path
            with h5py.File(path, 'r') as f, h5py.File(source, 'r') as old_f:
                length = row['frames']
                assert f.attrs['embodiment'] == 'dex3' and f.attrs['fps'] == 30.0
                assert json.loads(f.attrs['visual_preprocessing']) == report['transform']
                assert f['observation.image.top'].id == f['observation.image.left'].id
                assert f['observation.image_length.top'].id == f['observation.image_length.left'].id
                for key, ranges in [('action', left_action_range), ('observation.state', left_state_range)]:
                    values = f[key][()]
                    assert values.shape == (length, 128) and values.dtype == np.float32
                    assert np.isfinite(values).all() and not np.any(values[:, 98:128])
                    np.testing.assert_array_equal(values, old_f[key][()])
                    np.maximum(ranges, np.ptp(values[:, left_dims], axis=0), out=ranges)
                for key in ['representation_valid', 'hand_keypoint_valid', 'source.timestamp']:
                    np.testing.assert_array_equal(f[key][()], old_f[key][()])
                lengths = f['observation.image_length.top'][()]
                images = f['observation.image.top']
                assert images.shape[0] == length and images.dtype == np.uint8
                assert lengths.shape == (length,) and np.all((lengths > 0) & (lengths <= images.shape[1]))
                for index in sorted({0, length // 2, length - 1}):
                    encoded = images[index]
                    decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                    assert decoded is not None and decoded.shape == (240, 320, 3), (path, index)
                    assert encoded[0] == 255 and encoded[1] == 216
                    assert not np.array_equal(encoded, old_f['observation.image.top'][index]), (path, index)
                    samples += 1
                frames += length
            if ordinal % 50 == 0 or ordinal == len(expected):
                print(f'AUDIT {task}: {ordinal}/{len(expected)} verified (SHA256, actions/state, sampled images)', flush=True)
        assert frames == report['frames']
        assert left_state_range.max() > 0 and left_action_range.max() > 0
        result['tasks'][task] = {
            'episodes': len(expected), 'frames': frames, 'train': len(splits['train']), 'val': len(splits['val']),
            'all_output_sha256_verified': True, 'all_actions_and_states_equal_previous': True,
            'same_disjoint_complete_splits': True, 'decoded_changed_image_samples': samples,
            'left_state_max_range': float(left_state_range.max()),
            'left_action_max_range': float(left_action_range.max()),
            'processing_report_sha256': sha256(directory / 'processing_report.json'),
        }
    (root / 'configs/data_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def check_left_hand(root):
    import torch
    from torch import nn

    sys.path[:0] = [str(root / 'source'), str(root / 'source/hdt'), str(root / 'source/hdt/detr')]
    from hdt.policy import ACTPolicy
    from hdt.constants import OUTPUT_LEFT_EEF, OUTPUT_LEFT_KEYPOINTS, OUTPUT_NECK
    from hdt.detr.models.detr_vae import DETRVAE

    torch.set_num_threads(4)

    class TinyTransformer(nn.Module):
        d_model = 8

        def forward(self, src, mask, query, pos, latent, proprio, extra):
            return (proprio[:, None, :].expand(-1, len(query), -1),)

    class TinyEncoder(nn.Module):
        def forward(self, values, **kwargs):
            return values

    backbone = nn.Identity()
    backbone.num_channels = 8
    model = DETRVAE([backbone], TinyTransformer(), TinyEncoder(), 128, 128, 3,
                    ['top'], 'ACT_linear', False, zero_state_dims=(98, 128))
    model.get_features_and_pos = lambda images: (torch.zeros(2, 8, 1, 1), torch.zeros(2, 8, 1, 1))
    captured = {}
    def capture(name):
        def hook(module, args):
            captured[name] = args[0].detach().clone()
        return hook
    model.encoder_joint_proj.register_forward_pre_hook(capture('encoder'))
    model.input_proj_robot_state.register_forward_pre_hook(capture('decoder'))
    qpos = torch.arange(1, 129, dtype=torch.float32).repeat(2, 1)
    with torch.no_grad():
        model(qpos, torch.zeros(2, 1, 3, 8, 8), None,
              torch.zeros(2, 3, 128), torch.zeros(2, 3, dtype=torch.bool))
    left = torch.as_tensor(list(OUTPUT_LEFT_EEF) + list(OUTPUT_LEFT_KEYPOINTS))
    assert torch.all(model.state_keep_mask[left] == 1)
    assert torch.all(model.state_keep_mask[98:128] == 0)
    for values in captured.values():
        assert torch.equal(values[:, left], qpos[:, left])
        assert not torch.any(values[:, 98:128])
    assert set(captured) == {'encoder', 'decoder'}

    class Predictions(nn.Module):
        num_queries = 3

        def __init__(self):
            super().__init__()
            self.prediction = nn.Parameter(torch.ones(2, 3, 128))

        def forward(self, *args, **kwargs):
            zeros = torch.zeros(2, 32)
            return self.prediction, None, (zeros, zeros), None

    policy = ACTPolicy.__new__(ACTPolicy)
    nn.Module.__init__(policy)
    policy.model = Predictions()
    policy.transform = nn.Identity()
    policy.kl_weight = 10
    policy.query0_extra_weight = 0
    policy.USE_EEF_LOSS = True
    losses = policy(torch.zeros(2, 1, 3, 8, 8), qpos,
                    torch.zeros(2, 3, 128), torch.zeros(2, 3, dtype=torch.bool))
    losses['loss'].backward()
    gradient = policy.model.prediction.grad
    assert torch.all(gradient[:, :, left] > 0), 'Left action dimensions are masked from the loss'
    assert torch.all(gradient[:, :, OUTPUT_NECK] == 0)
    result = {
        'status': 'pass', 'source': 'exact 20261031 snapshot', 'zero_state_dims': [98, 128],
        'left_wrist_dims': [80, 89], 'left_finger_dims': [10, 28],
        'left_state_preserved_at_encoder_and_decoder': True,
        'every_left_action_dimension_has_positive_loss_gradient': True,
        'neck_loss_mask_preserved': True,
        'previous_snapshot_left_mask_found': False,
    }
    (root / 'configs/left_hand_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print('LEFT HAND: input and loss-gradient regression checks passed', flush=True)
    return result
