"""Run the existing MPJPE protocol on the shared held-out split, without editing it."""
import os,sys,json,pathlib,importlib.util,time,pickle,argparse
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'src/hdt')]
from load_experiment import load_experiment
from policy import ACTPolicy
spec=importlib.util.spec_from_file_location('original_mpjpe', '/home/aigc/human_policy/data/eval_mpjpe_batch.py')
core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('name');ap.add_argument('--max-steps',type=int);args=ap.parse_args()
    torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    started=time.time()
    if args.name=='original':
        ck=pathlib.Path('/mnt/nvme0n1/human_policy_ckpt/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt')
        config=json.load(open(ROOT/'checkpoints/geometry_ckpt/training_config.json'))
        policy_args=config['config']['policy_config'].copy();policy_args['experiment_variant']='baseline'
        policy=ACTPolicy(policy_args)
        policy.load_state_dict(torch.load(ck/'policy_last.ckpt',map_location='cpu',weights_only=True),strict=True)
        policy=policy.cuda().eval()
    else:
        ck=ROOT/'checkpoints'/(args.name+'_ckpt')
        policy,_,config=load_experiment(ck)
    split=json.load(open(ROOT/'configs/replace50.json'))
    results={}
    for half in ('robot','human'):
        per_file=[]
        with torch.inference_mode():
            for group in split['val']:
                if group['type']!=half:continue
                for filename in sorted(group['file_list']):
                    ep=pathlib.Path(group['dataset_path'])/filename
                    with core.h5py.File(ep,'r') as f:
                        emb=f.attrs.get('embodiment');emb=emb.decode() if isinstance(emb,bytes) else str(emb)
                    stats=core._load_norm_stats(ck/'dataset_stats.pkl',emb)
                    pred,indices=core._predict_actions_for_episode(policy,ep,stats=stats,camera_names=['top'],device='cuda',max_steps=args.max_steps)
                    with core.h5py.File(ep,'r') as f:gt=f['action'][indices]
                    m=core.eval_arrays_mpjpe(gt,pred);m['file']=str(ep);per_file.append(m)
                    print(json.dumps(dict(model=args.name,half=half,episode=len(per_file),file=filename,frames=m['timesteps_compared'],all_mm=m['mpjpe_all_mm'],elapsed_s=time.time()-started)),flush=True)
        summary={k:float(np.mean([x[k] for x in per_file])) for k in per_file[0] if k.startswith('mpjpe_')}
        summary.update(num_pairs_evaluated=len(per_file),total_timesteps_compared=sum(x['timesteps_compared'] for x in per_file),total_timesteps_skipped=sum(x['timesteps_skipped'] for x in per_file))
        results[half]=dict(summary=summary,per_file=per_file)
    payload=dict(model=args.name,checkpoint=str(ck/'policy_last.ckpt'),protocol='original eval_mpjpe_batch.py, query0, full held-out episodes, per-episode macro-average, original dirty-start defaults',max_steps=args.max_steps,elapsed_s=time.time()-started,results=results)
    out=ROOT/'results/mpjpe_comparison';out.mkdir(exist_ok=True)
    suffix='_smoke' if args.max_steps else ''
    (out/(args.name+suffix+'.json')).write_text(json.dumps(payload,indent=2))
    print('COMPLETED '+json.dumps({h:r['summary'] for h,r in results.items()}),flush=True)
if __name__=='__main__':main()
