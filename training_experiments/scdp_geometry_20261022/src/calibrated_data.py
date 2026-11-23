"""Read-only calibration sidecars and crop-aware dataset adapter."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as TF
import data_utils_hdt as data
from spatial_conditioner import resize_crop_affine


class RecordedCrop(nn.Module):
    def __init__(self, crop):
        super().__init__()
        self.crop, self.last = crop, None

    def forward(self, images):
        self.last = v2.RandomResizedCrop.get_params(images, self.crop.scale, self.crop.ratio)
        return TF.resized_crop(images, *self.last, self.crop.size,
                               interpolation=self.crop.interpolation, antialias=self.crop.antialias)


class CalibrationStore:
    """Manifest keys are absolute HDF paths, never just ambiguous episode numbers.

    Each entry identifies its RGB projection model and KP frame/units/provenance, with an NPZ
    containing K [3,3] or [N,3,3], T_cam_from_kp [N,4,4], valid [N], raw_hw [2].
    Extrinsics are explicitly per raw observation frame, including head motion.
    """
    def __init__(self, manifest=None):
        self.entries = json.loads(Path(manifest).read_text()) if manifest else {}
        self.cache = {}

    def load(self, path):
        entry = self.entries.get(str(Path(path).resolve()))
        if entry is None:
            return None
        if path in self.cache:
            return self.cache[path]
        for field in ('npz', 'rgb_key', 'embodiment', 'kp_frame', 'kp_units', 'provenance', 'rectified'):
            if field not in entry:
                raise ValueError(f'{path}: missing calibration field {field}')
        if entry['kp_units'] != 'm':
            raise ValueError('KP units must be metres')
        with np.load(entry['npz'], allow_pickle=False) as z:
            arrays = {k: z[k].copy() for k in ('K', 'T_cam_from_kp', 'valid', 'raw_hw')}
            arrays['D'] = z['D'].copy() if 'D' in z else np.zeros(8,np.float32)
            arrays['hand_valid'] = z['hand_valid'].copy() if 'hand_valid' in z else np.ones(2,bool)
        if arrays['D'].shape!=(8,) or arrays['hand_valid'].shape!=(2,):
            raise ValueError(f'{path}: invalid distortion/hand mask shape')
        if entry['rectified'] is False and entry.get('distortion_model')!='opencv_rational_8':
            raise ValueError(f'{path}: raw camera requires explicit OpenCV rational distortion model')
        if not np.isfinite(arrays['D']).all():
            raise ValueError(f'{path}: nonfinite distortion coefficients')
        import h5py
        with h5py.File(path, 'r') as f:
            n = len(f['action'])
            if str(f.attrs['embodiment']) != entry['embodiment'] or entry['rgb_key'] not in f:
                raise ValueError(f'{path}: camera/embodiment mismatch')
            # The baseline uses top, then left, then right. Do not calibrate a
            # different physical camera just because both are present in the file.
            selected = next((k for k in ('observation.image.top', 'observation.image.left',
                                         'observation.image.right') if k in f), None)
            if selected != entry['rgb_key']:
                raise ValueError(f'{path}: calibration is for {entry["rgb_key"]}, HAT reads {selected}')
            im = f[selected][0]
            if im.ndim == 1:
                import cv2
                im = cv2.imdecode(im, cv2.IMREAD_COLOR)
            if tuple(im.shape[:2]) != tuple(arrays['raw_hw']):
                raise ValueError(f'{path}: K must refer to decoded stored RGB size')
        if arrays['T_cam_from_kp'].shape != (n, 4, 4) or arrays['valid'].shape != (n,):
            raise ValueError(f'{path}: extrinsics/valid must match raw observation timeline')
        if arrays['K'].shape not in ((3, 3), (n, 3, 3)) or arrays['raw_hw'].shape != (2,):
            raise ValueError(f'{path}: invalid intrinsic/image size shape')
        use = arrays['valid'].astype(bool)
        ts = arrays['T_cam_from_kp'][use]
        if len(ts):
            if not np.isfinite(ts).all() or not np.allclose(ts[:, 3], [0, 0, 0, 1], atol=1e-5):
                raise ValueError(f'{path}: invalid rigid transforms')
            r = ts[:, :3, :3]
            if not np.allclose(r @ r.transpose(0, 2, 1), np.eye(3), atol=1e-3) or not np.allclose(np.linalg.det(r), 1, atol=1e-3):
                raise ValueError(f'{path}: camera rotation is not SO(3)')
            k = arrays['K'] if arrays['K'].ndim == 2 else arrays['K'][use]
            if not np.isfinite(k).all() or (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any():
                raise ValueError(f'{path}: invalid focal length')
        self.cache[path] = arrays
        return arrays


class CalibratedDataset(torch.utils.data.Dataset):
    def __init__(self, baseline, store, baseline_stats):
        self.base, self.store = baseline, store
        self.base.norm_stats = baseline_stats
        self.crop = None
        if baseline.visual_aug is not None:
            original = baseline.training_transforms.transforms[0]
            if not isinstance(original, v2.RandomResizedCrop):
                raise ValueError('Unexpected image preprocessing; audit before proceeding')
            self.crop = RecordedCrop(original)
            baseline.training_transforms.transforms[0] = self.crop

    def __len__(self):
        return len(self.base)

    def read_one(self, index, start_ts):
        if self.crop:
            self.crop.last = None
        item = list(self.base.read_one(index, start_ts))
        path = self.base.dataset_paths[index]
        calibration = self.store.load(path)
        raw_ts = self.base._raw_index(index, start_ts)
        if calibration is None:
            camera = dict(K=torch.zeros(3, 3), T_cam_from_kp=torch.zeros(4, 4),
                          D=torch.zeros(8),hand_valid=torch.zeros(2,dtype=torch.bool),
                          raw_hw=torch.zeros(2), pixel_from_raw=torch.eye(3), valid=torch.tensor(False))
        else:
            k = calibration['K'] if calibration['K'].ndim == 2 else calibration['K'][raw_ts]
            crop = self.crop.last if self.crop else None
            camera = dict(K=torch.as_tensor(k).float(),
                D=torch.as_tensor(calibration['D']).float(),hand_valid=torch.as_tensor(calibration['hand_valid']).bool(),
                T_cam_from_kp=torch.as_tensor(calibration['T_cam_from_kp'][raw_ts]).float(),
                raw_hw=torch.as_tensor(calibration['raw_hw']).float(),
                pixel_from_raw=resize_crop_affine(calibration['raw_hw'],
                    self.base.data_config['image_resolution_hw'], crop),
                valid=torch.tensor(bool(calibration['valid'][raw_ts])))
        embodiment = self.base.source_embodiment_list[index]
        stats = self.base.norm_stats[embodiment]
        item[4].update(scdp_camera=camera,
                       scdp_action_mean=torch.as_tensor(stats['action_mean']).float(),
                       scdp_action_std=torch.as_tensor(stats['action_std']).float())
        return tuple(item)

    def __getitem__(self, _index):
        # Same episode/time sampling as the baseline; no retry that hides geometry errors.
        i = np.random.choice(len(self.base.episode_len_list), p=self.base.episode_sampling_prob)
        t = np.random.randint(self.base.sum_dataset_len_l[i], self.base.sum_dataset_len_l[i + 1])
        episode, start = self.base._locate_transition(t)
        return self.read_one(episode, start)


def collate(batch):
    output = list(data.collate_fn(batch))
    for key in ('scdp_action_mean', 'scdp_action_std'):
        output[4][key] = torch.stack([x[4][key] for x in batch])
    output[4]['scdp_camera'] = {
        key: torch.stack([x[4]['scdp_camera'][key] for x in batch])
        for key in batch[0][4]['scdp_camera']}
    return tuple(output)
