"""Actual checkpoint/data forward/backward, without invented camera calibration."""
import runtime
import json
import pickle
import h5py
import torch
from hat_adapter import SCDPHAT
from calibrated_data import CalibrationStore, CalibratedDataset, collate
from data_utils_hdt import EpisodicDataset
from modeling.utils import make_visual_encoder

torch.set_num_threads(4)
torch.manual_seed(123)
policy, cfg = runtime.build_policy()
policy.eval()
entries = json.loads(runtime.DATASET.read_text())['val']
entry = next(e for e in entries if e['type'] == 'human')
path = str(runtime.Path(entry['dataset_path']) / entry['file_list'][0])
with h5py.File(path) as f:
    n = len(f['action'])
_, preprocess = make_visual_encoder('ACT', {})
base = EpisodicDataset(cfg['data'], [path], [None], ['top'], 100, [n], [1],
                       preprocess, 0., train=False, slow_down_factor=4, robot_target_fps=30.)
stats = pickle.load(open(runtime.BASE / 'dataset_stats.pkl', 'rb'))
ds = CalibratedDataset(base, CalibrationStore(), stats)
images, qpos, truth, pad, cond = collate([ds.read_one(0, 0), ds.read_one(0, min(5, len(base.episode_frame_indices[0])-1))])
images, qpos, truth, pad = [x.cuda() for x in (images, qpos, truth, pad)]
with torch.no_grad():
    baseline = policy(images, qpos)
    repeat = policy(images, qpos)
print('Baseline repeat maximum difference:', float((baseline-repeat).abs().max()), flush=True)
wrapper = SCDPHAT(policy.model).cuda()
policy.model = wrapper
policy.eval()
with torch.no_grad():
    disabled = policy(images, qpos, conditioning_dict=cond)
print('Disabled maximum difference:', float((baseline-disabled).abs().max()), flush=True)
assert torch.equal(baseline, disabled), 'Disabled path changed baseline output'
wrapper.conditioner.enabled = True
wrapper.base.requires_grad_(False)
policy.eval()
with torch.no_grad():
    frozen_baseline = wrapper.base(qpos, policy.transform(images), None)[0]
print('Freeze-only rounding difference:', float((baseline-frozen_baseline).abs().max()), flush=True)
with torch.no_grad():
    fallback = policy(images, qpos, conditioning_dict=cond)
assert torch.equal(frozen_baseline, fallback), 'Missing-camera path changed frozen baseline output'
print('REAL prior forward: disabled and missing-calibration fallback are BITWISE EQUAL', flush=True)
print('Feature / hidden shapes:', wrapper.debug['feature_shape'], wrapper.debug['hidden_shape'], flush=True)
# Original ACTPolicy loss, original posterior, actual labels. This checks backward
# plumbing only: zero valid projections mean zero conditioner learning signal.
losses = policy(images, qpos, truth, pad, conditioning_dict=cond)
losses['loss'].backward()
grads = [p.grad for p in wrapper.conditioner.parameters()]
assert all(g is not None and torch.isfinite(g).all() for g in grads)
assert all(p.grad is None for p in wrapper.base.parameters())
print('REAL minibatch backward loss:', float(losses['loss'].detach()),
      'valid_projection_fraction:', float(wrapper.debug['valid_fraction']),
      'conditioner_gradient_sum:', sum(float(g.abs().sum()) for g in grads), flush=True)
print('Not a real-data overfit test: calibration is missing. No training started.', flush=True)
