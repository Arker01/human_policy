"""Held-out calibration, prior-only quality and matched teacher action regression."""
import os
os.environ['HDT_EXPERIMENT']='baseline'
import sys,pathlib,json,pickle,argparse,collections
import numpy as np
import torch,yaml
from scipy.stats import spearmanr
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/hdt'))
from policy import ACTPolicy
from load_experiment import load_experiment
from modeling.utils import make_visual_encoder
from data_utils_hdt import load_data,collate_fn,set_seed
from auxiliary_tasks import scene_tips,rotation_6d_rows
from label_horizon import Episode,stage_dist

STAGES=[('>60',lambda d:d>60),('40-60',lambda d:(d>=40)&(d<=60)),('25-40',lambda d:(d>=25)&(d<40)),('15-25',lambda d:(d>=15)&(d<25)),('8-15',lambda d:(d>=8)&(d<15)),('0-8',lambda d:(d>=0)&(d<8)),('<0',lambda d:d<0),('no_close_event',lambda d:~np.isfinite(d))]

def main(name,smoke=False):
 torch.set_num_threads(4);set_seed(2718)
 run=name+('_smoke' if smoke else '')
 ck=ROOT/'checkpoints'/(run+'_ckpt')
 policy,stats,config=load_experiment(ck)
 meta=json.loads((ROOT/'labels'/name/'manifest.json').read_text())
 teacher_args=config['config']['policy_config'].copy();teacher_args['experiment_variant']='baseline' if name=='original' else 'geometry'
 teacher=ACTPolicy(teacher_args).cuda().eval()
 teacher.load_state_dict(torch.load(meta['teacher'],map_location='cpu',weights_only=True),strict=True)
 _,prep=make_visual_encoder('ACT',teacher_args)
 _,val,newstats=load_data('/home/aigc/human_policy/data',yaml.safe_load(open(ROOT/'configs/baseline.yaml'))['data'],str(ROOT/'configs/replace50.json'),['top'],100,16,16,prep,0.,4,auto_trim_dirty_start=True,robot_target_fps=30.,action_representation='original')
 for e in stats:
  for k in ('action_mean','action_std','qpos_mean','qpos_std'):assert np.array_equal(stats[e][k],newstats[e][k])
 ds=val.dataset;pred_k=[];true_k=[];distance=[];prefix=[];valids=[];guarded=[];probs=[];metrics=collections.defaultdict(list)
 with torch.inference_mode():
  for i,path in enumerate(ds.dataset_paths):
   emb=ds.source_embodiment_list[i];robot=emb=='dex5'
   record=meta['episodes'][str(pathlib.Path(path).resolve())]
   with np.load(ROOT/'labels'/name/record['file']) as z: target_k=z['K_eps003'].copy()
   episode_distance=stage_dist(ds,i) if robot else None
   positions=np.arange(int(ds.episode_len_list[i]))
   if smoke:positions=np.unique(np.linspace(0,len(positions)-1,8).astype(int))
   for start in range(0,len(positions),16):
    times=positions[start:start+16];data=collate_fn([ds.read_one(i,int(t)) for t in times]);image,qpos,truth,pad=data[:4]
    actions,reliability=policy.predict_with_horizon(image.cuda(),qpos.cuda())
    ref=teacher(image.cuda(),qpos.cuda())
    mu=torch.as_tensor(stats[emb]['action_mean'],device='cuda');std=torch.as_tensor(stats[emb]['action_std'],device='cuda')+1e-6
    pr=actions*std+mu;ref=ref*std+mu;gt=truth.cuda()*std+mu
    p=scene_tips(pr)[:,:,1];g=scene_tips(gt)[:,:,1]
    shape=torch.linalg.vector_norm((p-p[:,:1])-(g-g[:,:1]),dim=-1).mean(-1)
    group='dex5' if robot else 'human'
    for model,values in [('student',pr),('teacher',ref)]:
     wrist=torch.linalg.vector_norm(values[:,:,30:33]-gt[:,:,30:33],dim=-1)*100
     tip=torch.linalg.vector_norm(scene_tips(values)[:,:,1]-g,dim=-1).mean(-1)*100
     for metric,vals in [('right_wrist_cm',wrist),('right_tips_cm',tip)]:
      metrics[group+'/'+model+'/query0/'+metric].extend(vals[:,0].cpu().tolist())
      metrics[group+'/'+model+'/chunk/'+metric].extend(vals[~pad.cuda()].cpu().tolist())
    if robot:
     v=(~pad).numpy();prob=reliability.cpu().numpy();cumulative=np.maximum.accumulate(shape.cpu().numpy(),axis=1)
     prefix.append(cumulative);valids.append(v);probs.append(prob);true_k.extend(target_k[times].tolist());distance.extend(episode_distance[times].tolist())
     path_length=torch.linalg.vector_norm(torch.diff(pr[:,:,30:33],dim=1),dim=-1).cumsum(-1).cpu().numpy()
     # Counts include the starting waypoint; reaching diff index 0 needs 2 frames.
     disp=np.array([np.searchsorted(row,.01)+2 for row in path_length]);guarded.extend(np.maximum(10,disp).clip(1,100).tolist())
   print(run,'held-out',i+1,'/',len(ds.dataset_paths),flush=True)
 prob=np.concatenate(probs);errs=np.concatenate(prefix);v=np.concatenate(valids);dist=np.asarray(distance);truth=np.asarray(true_k);floor=np.asarray(guarded)
 # Episode ends censor K: clipping a prediction by ground-truth availability
 # artificially inflates correlation. Primary quality/coverage uses full chunks.
 full=v.all(1);censored=int((~full).sum())
 prob,errs,v,dist,truth,floor=[a[full] for a in (prob,errs,v,dist,truth,floor)];N=len(truth)
 assert N>0,'No uncensored held-out chunks'
 table={}
 for tau in (.5,.7,.8,.9,.95):
  k=((prob>=tau)&v).sum(1);coverage=np.ones(N,bool);active=k>0
  coverage[active]=errs[np.flatnonzero(active),k[active]-1]<=.03
  guarded_k=np.maximum(k,floor).clip(1,100)
  guarded_k=np.minimum(guarded_k,v.sum(1))
  guard_covered=errs[np.arange(N),guarded_k-1]<=.03
  groups={}
  for label,sel in [('all',np.ones(N,bool))]+[(label,fn(dist)) for label,fn in STAGES]:
   if not sel.any():continue
   nonempty=sel&active
   groups[label]=dict(frames=int(sel.sum()),median_K=float(np.median(k[sel])),mean_K=float(np.mean(k[sel])),zero_fraction=float((k[sel]==0).mean()),coverage=float(coverage[sel].mean()),nonempty_coverage=float(coverage[nonempty].mean()) if nonempty.any() else None,guarded_coverage=float(guard_covered[sel].mean()),median_guarded_K=float(np.median(guarded_k[sel])))
  table[str(tau)]=groups
 # Choose one deployable threshold using per-stage validation, without an oracle
 # future close-onset at runtime. Empty executions cannot establish calibration.
 candidates=[]
 for tau,groups in table.items():
  if all(g['nonempty_coverage'] is not None and g['nonempty_coverage']>=.9 for g in groups.values()):candidates.append(tau)
 chosen=max(candidates,key=lambda t:table[t]['all']['median_K']) if candidates else None
 k=((prob>=float(chosen or .5))&v).sum(1)
 rho=float(spearmanr(k,truth).statistic) if np.std(k)>0 and np.std(truth)>0 else None
 result=dict(name=run,split='val_only_25_episodes',quality_population='dex5 with all 100 ground-truth steps available; no oracle length clipping',quality_frames=N,censored_frames_excluded=censored,quality_tau=float(chosen or .5),smoke=smoke,eps_res=.03,eps_cons=.03,tau=float(chosen) if chosen else None,calibrated=chosen is not None,selection='maximize median raw K subject to nonempty residual coverage >=0.9 in every observed phase; deployment guard reported separately',table=table,spearman=rho,mae_frames=float(np.abs(k-truth).mean()),metrics={key:dict(mean=float(np.mean(values)),count=len(values)) for key,values in metrics.items()},label_teacher_sha256=meta['teacher_sha256'])
 key='dex5/student/query0/right_wrist_cm';teacher_key='dex5/teacher/query0/right_wrist_cm'
 result['dex5_query0_wrist_regression_cm']=result['metrics'][key]['mean']-result['metrics'][teacher_key]['mean']
 out=ROOT/'results'/run;out.mkdir(parents=True,exist_ok=True);(out/'calibration.json').write_text(json.dumps(result,indent=2,allow_nan=False))
 print(json.dumps({key:value for key,value in result.items() if key not in ('metrics','table')},indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('name',choices=['original','geometry']);p.add_argument('--smoke',action='store_true');a=p.parse_args();main(a.name,a.smoke)
