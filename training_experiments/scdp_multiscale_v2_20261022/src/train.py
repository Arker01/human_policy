"""Independent basic SCDP trainer. Preflight refuses an all-unusable calibration set.

    CUDA_VISIBLE_DEVICES=6,7 python -m torch.distributed.run --nproc_per_node=2 \
      --master_port=29867 .../src/train.py --calibration-manifest /path/to/manifest.json

The default preflight creates no artifacts and initializes no CUDA context.
"""
import runtime
import argparse
import json
import pickle
import time
import hashlib
from pathlib import Path
import numpy as np
import torch
from calibrated_data import CalibrationStore, CalibratedDataset, collate
from hat_adapter import SCDPHAT


def preflight(store):
    import h5py
    counts = {}
    for split, entries in json.loads(runtime.DATASET.read_text()).items():
        for entry in entries:
            group = f'{split}/{Path(entry["dataset_path"]).name}'
            total, usable = 0, 0
            for name in entry['file_list']:
                path = str(Path(entry['dataset_path']) / name)
                with h5py.File(path, 'r') as f:
                    total += len(f['action'])
                calibration = store.load(path)
                if calibration is not None:
                    usable += int(np.count_nonzero(calibration['valid']))
            counts[group] = (total, usable)
    print('Calibration coverage (raw frames):', counts, flush=True)
    if not any(v[1] for k, v in counts.items() if k.startswith('train/')):
        raise SystemExit('BLOCKED: no calibrated training frames. Need real RGB K + per-observation T_cam_from_kp; no GPU training launched.')
    return counts


def move(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, dict):
        return {k: move(v, device) for k, v in x.items()}
    return x


def wrist_l1(pred,truth,pad,debug):
    ids=torch.tensor([80,81,82,30,31,32],device=pred.device)
    mask=debug['valid'].any(1).repeat_interleave(3,dim=-1)[:,None,:]&(~pad)[...,None]
    error=(pred[...,ids]-truth[...,ids]).abs()
    return (error*mask).sum()/mask.sum().clamp_min(1)


def offline_check(policy,wrapper,ds,device):
    """Fixed-frame, prior-only same-observation check; original validation split."""
    policy.eval();totals={}
    for group in ('human_old','robot_new'):
        pairs=[]
        for i,path in enumerate(ds.base.dataset_paths):
            if group not in str(path):continue
            n=int(ds.base.episode_len_list[i])
            pairs.extend((i,int(t)) for t in np.linspace(0,n-1,3).round())
        sums=np.zeros(3);count=0;nonwrist_max=0.;projection_sum=0.;samples=0
        for start in range(0,len(pairs),8):
            batch=collate([ds.read_one(i,t) for i,t in pairs[start:start+8]])
            image,qpos,truth,pad,cond=[move(x,device) for x in batch[:5]]
            with torch.no_grad():
                refined=policy(image,qpos,conditioning_dict=cond)
                coarse=wrapper.debug['coarse'];debug=wrapper.debug
                scale=cond['scdp_action_std'][:,None]+1e-6
                for j,pred in enumerate((coarse,refined)):
                    delta=(pred[...,30:33]-truth[...,30:33])*scale[...,30:33]
                    error=torch.linalg.vector_norm(delta,dim=-1)
                    sums[j]+=float(error[~pad].sum())
                count+=int((~pad).sum());samples+=len(image)
                projection_sum+=float(debug['valid'].float().mean())*len(image)
                keep=torch.ones(128,dtype=torch.bool,device=device);keep[[80,81,82,30,31,32]]=False
                nonwrist_max=max(nonwrist_max,float((refined[...,keep]-coarse[...,keep]).abs().max()))
        if count:
            totals[group]=dict(frames=len(pairs),coarse_right_wrist_m=sums[0]/count,
                refined_right_wrist_m=sums[1]/count,valid_anchor_fraction=projection_sum/max(samples,1),
                nonwrist_max_abs_difference=nonwrist_max)
    print('OFFLINE prior validation',json.dumps(totals),flush=True)
    return totals


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--calibration-manifest',default=str(runtime.CALIBRATION_MANIFEST))
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--steps', type=int, default=100000)
    p.add_argument('--overfit-steps', type=int, default=0, help='Repeat a single real minibatch; no checkpoint output')
    p.add_argument('--batch-size', type=int, default=64, help='Per process; each variant runs on its own GPU')
    p.add_argument('--lr',type=float,default=1e-5)
    p.add_argument('--save-every',type=int,default=1000)
    p.add_argument('--spatial-mode', choices=['single','five_scale_v2'], default='five_scale_v2')
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output', type=Path, default=runtime.ROOT/'checkpoints'/runtime.BASE_VARIANT)
    p.set_defaults(**runtime.EXPERIMENT['training'])
    args = p.parse_args()
    if not args.output.resolve().is_relative_to(runtime.ROOT.resolve()):
        p.error('Outputs must stay within the independent experiment folder')
    if args.output.exists() and any(args.output.glob('step_*.pt')):
        p.error('Output already has checkpoints; choose a new run directory')
    if min(args.steps,args.batch_size,args.save_every)<1:
        p.error('steps, batch size and save interval must be positive')
    store = CalibrationStore(args.calibration_manifest)
    preflight(store)
    if args.preflight:
        return
    from accelerate import Accelerator
    from data_utils_hdt import load_data, set_seed
    from modeling.utils import make_visual_encoder
    accelerator = Accelerator()
    torch.cuda.set_device(accelerator.local_process_index)
    torch.set_num_threads(4)
    set_seed(0+accelerator.process_index)
    policy, cfg = runtime.build_policy()
    conditioner_config=dict(runtime.EXPERIMENT['conditioner'],spatial_mode=args.spatial_mode)
    wrapper = SCDPHAT(policy.model, conditioner_config).cuda()
    policy.model = wrapper
    policy.optimizer = torch.optim.AdamW(wrapper.conditioner.parameters(), lr=args.lr, weight_decay=1e-4)
    _, preprocess = make_visual_encoder('ACT', {})
    lag = dict(prob=.5, min_m=.02, max_m=.08, search_frames=450, offset_prob=.25,
               offset_sigma_m=.02, offset_clip_m=.08, finger_prob=.25, finger_scale=2.)
    visual = dict(scale=(.75, 1.), ratio=(4/3 * .9, 4/3 / .9))
    tr, va, computed_stats = load_data(str(runtime.REPO/'data'), cfg['data'], str(runtime.DATASET),
        ['top'], 100, args.batch_size, args.batch_size, preprocess, 0., 4,
        future_image_enabled=True, future_horizon=45, lag_aug_config=lag,
        visual_aug_config=visual, action_representation='original', robot_target_fps=30.)
    stats = pickle.load(open(runtime.BASE / 'dataset_stats.pkl', 'rb'))
    # A drift in data or normalization must be investigated, not silently accepted.
    if isinstance(computed_stats,tuple):computed_stats=computed_stats[0]
    for emb in stats:
        for key in ('action_mean', 'action_std', 'qpos_mean', 'qpos_std'):
            if not np.array_equal(stats[emb][key], computed_stats[emb][key]):
                raise RuntimeError(f'Baseline training normalization differs: {emb}/{key}')
    tr = torch.utils.data.DataLoader(CalibratedDataset(tr.dataset, store, stats),
        batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate, pin_memory=True,
        persistent_workers=args.workers>0)
    va = CalibratedDataset(va.dataset, store, stats)
    # ACTPolicy overrides __call__; wrap it so Accelerator/DDP receives a standard
    # nn.Module.forward while still using the original policy loss unchanged.
    class LossModule(torch.nn.Module):
        def __init__(self, model):
            super().__init__(); self.policy = model
        def forward(self, *a, **kw):
            return self.policy(*a, **kw)
    model, optimizer, tr = accelerator.prepare(LossModule(policy), policy.optimizer, tr)
    iterator, fixed = iter(tr), None
    initial_wrist=None;last_wrist=None;skipped=0;start_time=time.monotonic()
    if accelerator.is_main_process:
        print('RUN',json.dumps(dict(base=str(runtime.BASE),global_batch_size=args.batch_size*accelerator.num_processes,
            trainable_parameters=sum(p.numel() for p in wrapper.conditioner.parameters()),
            frozen_baseline_parameters=sum(p.numel() for p in wrapper.base.parameters()),
            lr=args.lr,steps=args.overfit_steps or args.steps,output=str(args.output),
            calibration_manifest=args.calibration_manifest)),flush=True)
    for step in range(args.overfit_steps or args.steps):
        model.train()
        if fixed is None:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(tr); batch = next(iterator)
            if args.overfit_steps:
                fixed = batch
        else:
            batch = fixed
        image, qpos, truth, pad, cond, future = [move(x, accelerator.device) for x in batch]
        optimizer.zero_grad(set_to_none=True)
        if args.overfit_steps:torch.manual_seed(123+accelerator.process_index)
        losses = model(image, qpos, truth, pad, conditioning_dict=cond, future_image=future)
        valid = wrapper.debug['valid_fraction'].detach()
        if accelerator.reduce(valid, reduction='sum').item() == 0:
            skipped+=1
            if skipped>=10:raise RuntimeError('Ten consecutive globally unprojectable minibatches; inspect geometry')
            print(f'step={step} globally invalid minibatch; baseline fallback, no update',flush=True)
            continue
        skipped=0
        current_wrist=wrist_l1(wrapper.debug['refined'],truth,pad,wrapper.debug)
        if initial_wrist is None:initial_wrist=float(current_wrist.detach())
        last_wrist=float(current_wrist.detach())
        accelerator.backward(losses['loss'])
        if not all(p.grad is None or torch.isfinite(p.grad).all() for p in wrapper.conditioner.parameters()):
            raise RuntimeError('Nonfinite conditioner gradient')
        accelerator.clip_grad_norm_(wrapper.conditioner.parameters(), .1)
        optimizer.step()
        if step==1 and args.spatial_mode=='five_scale_v2':
            for level,module in enumerate(wrapper.conditioner.pyramid.levels):
                value=sum(float(p.grad.abs().sum()) for p in module.parameters() if p.grad is not None)
                if not value>0:raise RuntimeError(f'Pyramid level {level} received no gradient after FiLM warm start')
            print('PYRAMID',wrapper.debug['pyramid_shapes'],'sources',wrapper.debug['pyramid_source_layers'],'all five levels receive nonzero gradients',flush=True)
        if step % 10 == 0 and accelerator.is_main_process:
            counts=wrapper.debug['valid'].sum((0,1)).tolist()
            print(f'step={step} loss={float(losses["loss"].detach()):.6f} wrist_l1={last_wrist:.6f} valid_anchors={float(valid):.4f} valid_left_right={counts} elapsed_s={time.monotonic()-start_time:.1f}', flush=True)
            if runtime.BASE_VARIANT!='original_20261004' and 'geometry_loss' not in losses:
                raise RuntimeError('The selected geometry policy loss was not retained')
            if 'geometry_loss' in losses: print(f'geometry_loss={float(losses["geometry_loss"].detach()):.6f} geometry_error_cm={float(losses["geometry_error_cm"].detach()):.4f}',flush=True)
        if not args.overfit_steps and ((step + 1) % args.save_every == 0 or step + 1 == args.steps):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                args.output.mkdir(parents=True, exist_ok=True)
                payload=dict(conditioner=wrapper.conditioner.state_dict(),
                    optimizer=optimizer.state_dict(), step=step+1, base=str(runtime.BASE),
                    base_variant=runtime.BASE_VARIANT,base_source_root=str(runtime.SOURCE_ROOT),
                    base_weights_sha256=runtime.BASE_CONFIG['weights_sha256'],
                    input_mask_ranges=runtime.BASE_CONFIG['input_mask_ranges'],
                    conditioner_config=conditioner_config,
                    calibration_manifest=args.calibration_manifest,
                    calibration_manifest_sha256=hashlib.sha256(Path(args.calibration_manifest).read_bytes()).hexdigest(),
                    original_loss='Selected base ACTPolicy unchanged, including geometry/event weights where present',
                    seed=0,version='v2',spatial_mode=args.spatial_mode,
                    pyramid=runtime.EXPERIMENT['pyramid'] if args.spatial_mode=='five_scale_v2' else None,
                    lr=args.lr,global_batch_size=args.batch_size*accelerator.num_processes)
                destination=args.output/f'step_{step+1}.pt';temporary=destination.with_suffix('.tmp')
                torch.save(payload,temporary);temporary.replace(destination)
                print('CHECKPOINT',destination,flush=True)
                reference=runtime.BASE_CONFIG['v1_reference_checkpoint']
                reference_step=int(Path(reference).stem.split('_')[-1])
                if step+1==reference_step and step+1<args.steps:
                    import subprocess,sys
                    result=runtime.ROOT/'results'/runtime.BASE_VARIANT/f'comparison_step_{step+1}.json'
                    subprocess.run([sys.executable,'-u',str(runtime.ROOT/'src/evaluate_protocol.py'),
                        '--checkpoint',str(destination),'--reference',reference,'--output',str(result)],check=True)
    # Fixed held-out prior-only mini-evaluation; no posterior action leakage.
    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        if args.overfit_steps:
            print('OVERFIT',dict(initial_supported_wrist_l1=initial_wrist,last_supported_wrist_l1=last_wrist,
                ratio=last_wrist/max(initial_wrist,1e-12)),flush=True)
            if last_wrist>=initial_wrist*.98:raise RuntimeError('Fixed minibatch did not improve wrist loss by 2%')
        offline_check(policy,wrapper,va,accelerator.device)
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == '__main__':
    main()
