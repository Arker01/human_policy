"""Read one existing online observation; exercise real inference without networking/writes."""
import os
import json
from pathlib import Path

import numpy as np
import torch
import hdt.online_hat_service as entry

bundle = Path(os.environ['HAT_BUNDLE'])
repo = Path(__file__).resolve().parents[3]
record_dir = repo / 'online_hat_records/2026-10-21_15-18-27'
record = json.loads(next((record_dir / 'inference_inputs.jsonl').open()))
with np.load(record_dir / record['images_file']) as saved:
    images = saved[saved.files[0]]
policy = entry.original.OnlineHATPolicy(
    config_path=bundle / os.environ['HAT_CONFIG'], checkpoint=bundle / 'policy_last.ckpt',
    stats_path=bundle / 'dataset_stats.pkl', embodiment='dex5', state_tail='full',
    wrist_conversion='inspire_hand_base', device='cuda', state_ood_limit=1000)
raw, normalized = policy.prepare_policy_state(record['robot_state'], record['head_origin'], record['initial_heading_quat_wxyz'])
output = policy.infer(images, record['robot_state'], record['head_origin'], normalized,
                      record['initial_heading_quat_wxyz'])
debug = policy.scdp_wrapper.debug
assert output.shape == (100, 128) and np.isfinite(output).all()
valid = debug['valid'].sum((0, 1)).tolist()
assert sum(valid) > 0, 'Recorded observation must exercise SCDP valid sampling'
keep = torch.ones(128, dtype=torch.bool, device='cuda')
keep[[30, 31, 32, 80, 81, 82]] = False
assert torch.equal(debug['refined'][..., keep], debug['coarse'][..., keep])
delta = float((debug['refined'] - debug['coarse']).abs().max())
assert delta > 0, 'Adapter must change at least one wrist prediction'
if 'no_left_wrist' in bundle.name:
    perturbed = normalized.copy()
    perturbed[80:89] += 100
    changed = policy.infer(images, record['robot_state'], record['head_origin'], perturbed,
                           record['initial_heading_quat_wxyz'])
    assert np.array_equal(output, changed), 'Left-wrist input masking must remain exact'
print('LOCAL INFERENCE PASSED', dict(bundle=bundle.name, step=policy.scdp_step,
      output_shape=output.shape, valid_anchors_left_right=valid, max_normalized_refinement=delta,
      feature_shape=debug['feature_shape'], input_mask=policy.scdp_wrapper.base.state_keep_mask.tolist()))
