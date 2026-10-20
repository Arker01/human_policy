"""Fixed-frame, prior-only offline validation. No claim of robot success rate."""
import os
os.environ['HDT_EXPERIMENT']='readiness'  # collect auxiliary labels for every model
import sys,pathlib,json,pickle,argparse
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/hdt'))
from load_experiment import load_experiment
from policy import ACTPolicy
from modeling.utils import make_visual_encoder
from data_utils_hdt import load_data,collate_fn,set_seed
from auxiliary_tasks import scene_tips,rotation_6d_rows,dataset_aux


def auc(target,prob):
    p=prob[target>0];n=prob[target==0]
    if not len(p) or not len(n):return None
    d=p[:,None]-n[None,:]
    return float(((d>0)+.5*(d==0)).mean())


def evaluate(name, smoke=False):
    torch.set_num_threads(4);set_seed(2718)
    suffix='_smoke' if smoke else ''
    ckpt=ROOT/'checkpoints'/(name+suffix+'_ckpt')
    reference=ROOT/'checkpoints'/('geometry'+suffix+'_ckpt')
    if name=='original':
        config=json.load(open(reference/'training_config.json'))
        args=config['config']['policy_config'].copy();args['experiment_variant']='baseline'
        policy=ACTPolicy(args)
        origin=pathlib.Path('/mnt/nvme0n1/human_policy_ckpt/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt')
        policy.load_state_dict(torch.load(origin/'policy_last.ckpt',map_location='cuda',weights_only=True),strict=True)
        policy.eval();stats=pickle.load(open(origin/'dataset_stats.pkl','rb'))
    else:policy,stats,config=load_experiment(ckpt)
    import yaml
    cfg=yaml.safe_load(open(ROOT/'configs/baseline.yaml'))
    _,preprocessor=make_visual_encoder('ACT',config['config']['policy_config'])
    _,val_loader,newstats=load_data('/home/aigc/human_policy/data',cfg['data'],str(ROOT/'configs/replace50.json'),['top'],100,16,16,preprocessor,0.,4,auto_trim_dirty_start=True,robot_target_fps=30.,action_representation='original')
    for e in stats:
        for k in ('action_mean','action_std','qpos_mean','qpos_std'):
            assert np.array_equal(stats[e][k],newstats[e][k]),(e,k)
    ds=val_loader.dataset
    frames={}
    for i in range(len(ds.dataset_paths)):
        n=int(ds.episode_len_list[i])
        uniform=set(np.linspace(0,n-1,4 if smoke else 32).astype(int).tolist())
        for t in uniform:frames[(i,t)]=True
        dataset_aux(ds,i,0)
        target,mask,_=ds._aux_cache[i]
        timeline=ds.episode_frame_indices[i][:n]
        if smoke:
            for col in (2,3):
                ids=np.flatnonzero((mask[timeline,col]>0)&(target[timeline,col]>0))
                if len(ids): frames.setdefault((i,int(ids[len(ids)//2])),False)
        if not smoke:
            # All held-out positives, a deterministic spread of stage negatives.
            for col in (2,3):
                for cls in (0,1):
                    ids=np.flatnonzero((mask[timeline,col]>0)&(target[timeline,col]==cls))
                    if len(ids)>200:ids=ids[np.linspace(0,len(ids)-1,200).astype(int)]
                    for t in ids:frames.setdefault((i,int(t)),False)
    items=sorted(frames)
    totals={};ready_p=[];ready_t=[];ready_m=[]
    def add(key,values,mask):
        selected=values[mask]
        if selected.numel():
            total,count=totals.get(key,(0.,0));totals[key]=(total+float(selected.sum()),count+selected.numel())
    with torch.inference_mode():
        for start in range(0,len(items),16):
            pairs=items[start:start+16]
            data=collate_fn([ds.read_one(i,t) for i,t in pairs])
            image,qpos,truth,pad,cond=data[:5]
            image,qpos,truth,pad=image.cuda(),qpos.cuda(),truth.cuda(),pad.cuda()
            # Intentionally pass no future actions: the CVAE posterior is unavailable
            # in deployment. Readiness gets current observation only as well.
            pred=policy(image,qpos)
            mu=cond['action_mean'].cuda()[:,None];std=cond['action_std'].cuda()[:,None]+1e-6
            pr=pred*std+mu;gt=truth*std+mu
            wrist=torch.linalg.vector_norm(pr[:,:,30:33]-gt[:,:,30:33],dim=-1)*100
            tip=torch.linalg.vector_norm(scene_tips(pr)[:,:,1]-scene_tips(gt)[:,:,1],dim=-1).mean(-1)*100
            rp=rotation_6d_rows(pr[:,:,33:39]);rg=rotation_6d_rows(gt[:,:,33:39]);cos=((rp@rg.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2
            angle=torch.rad2deg(torch.acos(cos.clamp(-1,1)))
            event=cond['hand_event_weights'][:,:,1].cuda()>1
            valid=~pad;uniform=torch.tensor([frames[x] for x in pairs],device='cuda')[:,None]
            robot=torch.tensor([ds.source_embodiment_list[i]=='dex5' for i,t in pairs],device='cuda')[:,None]
            for group,sel in (('all',uniform),('robot',uniform&robot),('human',uniform&~robot)):
                for metric,values in (('right_wrist_cm',wrist),('right_fingertips_cm',tip),('right_rotation_deg',angle)):
                    add(group+'/chunk/'+metric,values,valid&sel)
                    add(group+'/query0/'+metric,values[:,:1],valid[:,:1]&sel)
                    add(group+'/event_chunk/'+metric,values,valid&sel&event)
                    add(group+'/event_query0/'+metric,values[:,:1],valid[:,:1]&sel&event[:,:1])
                    ready_close=((cond['readiness_target'][:,2]>0)&(cond['readiness_mask'][:,2]>0)).cuda()[:,None]
                    group_sel = torch.ones_like(robot) if group=='all' else (robot if group=='robot' else ~robot)
                    add(group+'/ready_close_query0/'+metric,values[:,:1],valid[:,:1]&group_sel&ready_close)
            if policy.model.readiness_logits is not None:
                ready_p.append(policy.model.readiness_logits.sigmoid().cpu().numpy())
                ready_t.append(cond['readiness_target'].numpy());ready_m.append(cond['readiness_mask'].numpy())
            if start%160==0:print(name,'evaluation',start,'/',len(items),flush=True)
    result=dict(name=name,smoke=smoke,frames=len(items),uniform_frames=sum(frames.values()),mode='prior_z_zero_no_future_actions',metrics={k:dict(mean=s/n,count=n) for k,(s,n) in totals.items()})
    if ready_p:
        pp,tt,mm=map(np.concatenate,(ready_p,ready_t,ready_m));heads={}
        for c,label in enumerate(('left_close','left_open','right_close','right_open')):
            p=pp[mm[:,c]>0,c];t=tt[mm[:,c]>0,c]
            pos=t>0;neg=~pos
            heads[label]=dict(positives=int(pos.sum()),negatives=int(neg.sum()),auc=auc(t,p),recall_at_05=float((p[pos]>=.5).mean()) if pos.any() else None,specificity_at_05=float((p[neg]<.5).mean()) if neg.any() else None)
        result['readiness']=heads
    assert all(np.isfinite(v['mean']) for v in result['metrics'].values())
    (ROOT/'results'/(name+suffix+'_fixed_validation.json')).write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--smoke',action='store_true');args=p.parse_args();evaluate(args.name,args.smoke)
