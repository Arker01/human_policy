"""Read-only training-loader and full-model CPU checks for the tripod dataset."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
sys.path[:0] = [str(ROOT/'source'), str(ROOT/'source/hdt'), str(ROOT/'source/hdt/detr'), str(REPO)]

import cv2
import h5py
import numpy as np
import torch
import yaml
from hdt import constants as C
from hdt.data_utils_hdt import load_data
from hdt.modeling.utils import make_visual_encoder
from hdt.policy import ACTPolicy


def main():
    assert not torch.cuda.is_available(), 'Run this preflight with CUDA_VISIBLE_DEVICES empty'
    torch.set_num_threads(4)
    torch.manual_seed(0)
    cfg = yaml.safe_load((ROOT/'configs/model.yaml').read_text())
    _, processor = make_visual_encoder('ACT', {})
    train, val, stats = load_data(
        str(REPO/'data'), cfg['data'], str(ROOT/'configs/tripod.json'), ['top'],
        100, 64, 64, processor, 0.0, slow_down_factor=4,
        future_image_enabled=True, future_horizon=45, future_clip_frames=1,
        lag_aug_config={'prob':0.5, 'min_m':0.02, 'max_m':0.08, 'search_frames':450,
                        'offset_prob':0.25, 'offset_sigma_m':0.02, 'offset_clip_m':0.08,
                        'finger_prob':0.25, 'finger_scale':2.0},
        visual_aug_config={'scale':(0.75,1.0), 'ratio':(4/3*0.9,4/3/0.9)},
        action_representation='original', robot_target_fps=30.0)
    assert len(train.dataset.dataset_paths)==36 and len(val.dataset.dataset_paths)==4
    assert not set(train.dataset.dataset_paths)&set(val.dataset.dataset_paths)
    assert set(stats)=={'dex3'} and val.dataset.norm_stats is train.dataset.norm_stats
    for key in ['action_mean','action_std','qpos_mean','qpos_std']:
        assert stats['dex3'][key].shape==(128,) and np.isfinite(stats['dex3'][key]).all()
        if key.endswith('std'): assert (stats['dex3'][key]>=0.01).all()
    report = {'status':'running', 'episodes':40, 'frames':24528,
              'train_episodes':36,'val_episodes':4, 'train_val_disjoint':True,
              'embodiment':'dex3','camera_names':['top'], 'batches':{}}
    validation_batch = None
    for name,loader in [('train',train),('val',val)]:
        iterator = iter(loader)
        try:
            batch = next(iterator)
            image,state,action,pad,conditioning,future = batch
            assert image.shape==(64,1,3,240,320) and state.shape==(64,128)
            assert action.shape==(64,100,128) and pad.shape==(64,100)
            for value in [image,state,action,future]: assert torch.isfinite(value).all()
            assert future.shape==(64,1,3,240,320)
            assert all(e=='dex3' for e in loader.dataset.source_embodiment_list)
            report['batches'][name] = {'image':list(image.shape),'state':list(state.shape),
                'action':list(action.shape),'future':list(future.shape),'workers':loader.num_workers}
            if name=='val': validation_batch=batch
        finally:
            iterator._shutdown_workers()
    # Independent reader check: direct cv2 decode must match the actual RGB
    # tensor produced by read_one on the deterministic validation split.
    sample = val.dataset.read_one(0,0)
    first_frame = val.dataset.episode_frame_indices[0][0]
    with h5py.File(val.dataset.dataset_paths[0],'r') as f:
        decoded = cv2.imdecode(f['observation.image.top'][first_frame],cv2.IMREAD_COLOR)
        np.testing.assert_allclose(sample[0][0].numpy(),decoded.transpose(2,0,1)/255,atol=1e-7)
    # Check padding and target alignment on every train/val episode without
    # stochastic augmentation. A final sample must contain one real action.
    for loader in [train,val]:
        dataset=loader.dataset
        was_training=dataset.train
        dataset.train=False
        try:
            for index,length in enumerate(dataset.episode_len_list):
                sample=dataset.read_one(index,int(length)-1)
                assert (~sample[3]).sum()==1 and not sample[3][0]
        finally:
            dataset.train=was_training
    print('PASS real train/val batch64, head RGB, normalization, all episode boundaries',flush=True)
    model_config = {key:value for key,value in cfg['model'].items() if key not in ['future_dino','zero_state_dims']}
    model_config.update(lr=1e-5,lr_backbone=1e-5,num_queries=100,chunk_size=100,
        state_dim=128,action_dim=128,camera_names=['top'],zero_state_dims=(98,128),
        future_dino_config=cfg['model']['future_dino'],query0_extra_weight=0.0)
    policy=ACTPolicy(model_config)
    policy.train()
    both=list(C.OUTPUT_LEFT_EEF)+list(C.OUTPUT_LEFT_KEYPOINTS)+list(C.OUTPUT_RIGHT_EEF)+list(C.OUTPUT_RIGHT_KEYPOINTS)
    assert (policy.model.state_keep_mask[both]==1).all()
    captured={}
    def capture_input(name):
        def hook(module,args): captured[name]=args[0].detach().clone()
        return hook
    policy.model.encoder_joint_proj.register_forward_pre_hook(capture_input('encoder'))
    policy.model.input_proj_robot_state.register_forward_pre_hook(capture_input('decoder'))
    def capture_prediction(module,args,result):
        result.retain_grad(); captured['prediction']=result
    policy.model.action_head.register_forward_hook(capture_prediction)
    image,state,action,pad,conditioning,future=validation_batch
    conditioning={key:value[:1] if isinstance(value,(torch.Tensor,list)) else value for key,value in conditioning.items()}
    losses=policy(image[:1],state[:1],action[:1],pad[:1],conditioning,future[:1])
    assert all(torch.isfinite(v).all() for v in losses.values())
    losses['loss'].backward()
    for name in ['encoder','decoder']:
        torch.testing.assert_close(captured[name][:,both],state[:1,both])
        assert not captured[name][:,98:].any()
    gradient=captured['prediction'].grad
    hands={}
    for side,wrist,fingers in [('left',C.OUTPUT_LEFT_EEF,C.OUTPUT_LEFT_KEYPOINTS),
                               ('right',C.OUTPUT_RIGHT_EEF,C.OUTPUT_RIGHT_KEYPOINTS)]:
        dims=list(wrist)+list(fingers)
        per_dimension=gradient[:,:,dims].abs().sum((0,1))
        assert (per_dimension>0).all(), (side,per_dimension)
        hands[side]={'state_preserved_at_encoder_and_decoder':True,
                     'every_action_dimension_has_gradient':True,
                     'min_dimension_gradient_sum':float(per_dimension.min())}
    assert not gradient[:,:,C.OUTPUT_NECK].any()
    gradients=[p.grad for p in policy.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    report.update(status='pass',full_vjepa_act_cpu_forward_backward=True,
        future_dino_teacher_evaluated=True,losses={k:float(v.detach()) for k,v in losses.items()},
        both_hands=hands,finite_parameter_gradients=len(gradients),
        state_zero_range='98:128 only',existing_neck_loss_mask='58:67',
        rgb_matches_real_loader=True,all_episode_end_padding_checked=True,
        policy_source=str(Path(sys.modules['hdt.policy'].__file__).resolve()))
    destination=ROOT/'preflight_report.json'
    destination.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
