"""Actual base checkpoint: input mask, geometry objective and SCDP gradients."""
import runtime
import json,pickle
from pathlib import Path
import torch
from data_utils_hdt import EpisodicDataset,get_all_episode_len
from modeling.utils import make_visual_encoder
from calibrated_data import CalibratedDataset,CalibrationStore,collate
from hat_adapter import SCDPHAT
from train import move


def main():
    torch.set_num_threads(4);torch.manual_seed(5)
    policy,cfg=runtime.build_policy();policy.eval()
    entries=json.loads(runtime.DATASET.read_text())['train']
    chosen=[next(e for e in entries if Path(e['dataset_path']).name==source) for source in ('human_new','robot_new')]
    paths=[str(Path(e['dataset_path'])/e['file_list'][0]) for e in chosen]
    _,preprocessor=make_visual_encoder('ACT',{})
    base=EpisodicDataset(cfg['data'],paths,[None,None],['top'],100,get_all_episode_len(paths),[1,1],
        preprocessor,0.,train=False,slow_down_factor=4,robot_target_fps=30.,action_representation='original')
    stats=pickle.load(open(runtime.BASE/'dataset_stats.pkl','rb'))
    ds=CalibratedDataset(base,CalibrationStore(runtime.CALIBRATION_MANIFEST),stats)
    batch=collate([ds.read_one(i,min(25,int(base.episode_len_list[i])-1)) for i in range(2)])
    image,qpos,truth,pad,cond=[move(x,'cuda') for x in batch[:5]]
    for k in ('action_mean','action_std','hand_event_weights'):assert k in cond,k
    with torch.no_grad():reference=policy(image,qpos,conditioning_dict=cond)
    if runtime.BASE_VARIANT=='geometry_no_left_wrist':
        changed=qpos.clone();changed[:,80:89]+=100
        with torch.no_grad():masked=policy(image,changed,conditioning_dict=cond)
        assert torch.equal(reference,masked),'Left wrist reached no-left-wrist checkpoint encoder'
        print('NO LEFT WRIST: exact invariance to +100 normalized input perturbation',flush=True)
    wrapper=SCDPHAT(policy.model,dict(enabled=False)).cuda();policy.model=wrapper;policy.eval()
    with torch.no_grad():disabled=policy(image,qpos,conditioning_dict=cond)
    assert torch.equal(reference,disabled),'Disabled wrapper changed the selected base'
    wrapper.conditioner.enabled=True;wrapper.base.requires_grad_(False);policy.eval()
    if runtime.BASE_VARIANT=='geometry_no_left_wrist':
        with torch.no_grad():
            before=policy(image,qpos,conditioning_dict=cond)
            after=policy(image,changed,conditioning_dict=cond)
        assert torch.equal(before,after),'SCDP bypassed the no-left-wrist encoder mask'
        print('SCDP ENABLED: exact invariance to masked left-wrist perturbation',flush=True)
    torch.manual_seed(123)
    losses=policy(image,qpos,truth,pad,conditioning_dict=cond)
    assert 'geometry_loss' in losses and torch.isfinite(losses['geometry_loss'])
    assert wrapper.debug['valid'].any(),'Real data has no valid projection'
    losses['loss'].backward()
    assert all(p.grad is None for p in wrapper.base.parameters())
    gradient=sum(float(p.grad.abs().sum()) for p in wrapper.conditioner.parameters() if p.grad is not None)
    assert gradient>0 and all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.conditioner.parameters())
    print('PASS',dict(base=runtime.BASE_VARIANT,disabled_bitwise_equal=True,geometry_loss=float(losses['geometry_loss']),
        geometry_error_cm=float(losses['geometry_error_cm']),valid_anchor_fraction=float(wrapper.debug['valid_fraction']),
        conditioner_gradient_sum=gradient,original_model_frozen=True),flush=True)


if __name__=='__main__':main()
