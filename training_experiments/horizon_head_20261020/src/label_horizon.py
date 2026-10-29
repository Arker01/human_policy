"""Sequential prior-only frozen-teacher scan. Writes only the requested label cache/statistics."""
import os
os.environ['HDT_EXPERIMENT']='baseline'
import sys, pathlib, json, pickle, hashlib, argparse, collections, time
import numpy as np
import torch
import yaml
ROOT=pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/hdt'))
from policy import ACTPolicy
from modeling.utils import make_visual_encoder
from data_utils_hdt import load_data, collate_fn, set_seed
from auxiliary_tasks import scene_tips, rotation_6d_rows, transition_labels

ORIGINAL=pathlib.Path('/mnt/nvme0n1/human_policy_ckpt/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt')
GEOMETRY=pathlib.Path('/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_readiness_geometry/checkpoints/geometry_ckpt')
def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()
def label_key(path):
    return pathlib.Path(path).stem+'_'+hashlib.sha256(str(pathlib.Path(path).resolve()).encode()).hexdigest()[:12]
class Episode(torch.utils.data.Dataset):
    def __init__(self, ds, i): self.ds,self.i=ds,i
    def __len__(self): return int(self.ds.episode_len_list[self.i])
    def __getitem__(self,t): return self.ds.read_one(self.i,t)

def stage_dist(ds,i):
    import h5py
    with h5py.File(ds.dataset_paths[i],'r') as f: states=f['observation.state'][:]
    events=transition_labels(states,ds.source_fps_list[i])[3]
    raw_onsets=[e[2] for e in events if e[0]==1 and e[1]=='close']
    timeline=ds.episode_frame_indices[i]
    onsets=np.array([np.searchsorted(timeline,t) for t in raw_onsets])
    dist=np.full(len(timeline),np.nan,dtype=np.float32)
    for t in range(len(timeline)):
        future=onsets[onsets>=t]
        if len(future): dist[t]=future.min()-t
        elif len(onsets): dist[t]=onsets.max()-t
    return dist

def main(name):
    torch.set_num_threads(4); set_seed(0)
    out=ROOT/'labels'/name;out.mkdir(parents=True,exist_ok=True)
    teacher=ORIGINAL if name=='original' else GEOMETRY
    cfg=json.load(open(GEOMETRY/'training_config.json'))
    args=cfg['config']['policy_config'].copy();args['experiment_variant']='baseline' if name=='original' else 'geometry'
    policy=ACTPolicy(args).cuda().eval(); policy.requires_grad_(False)
    policy.load_state_dict(torch.load(teacher/'policy_last.ckpt',map_location='cpu',weights_only=True),strict=True)
    stats=pickle.load(open(teacher/'dataset_stats.pkl','rb'))
    _,prep=make_visual_encoder('ACT',args)
    train,val,newstats=load_data('/home/aigc/human_policy/data',yaml.safe_load(open(ROOT/'configs/baseline.yaml'))['data'],str(ROOT/'configs/replace50.json'),['top'],100,16,16,prep,0.,4,auto_trim_dirty_start=True,robot_target_fps=30.,action_representation='original')
    for emb in stats:
        for k in ('action_mean','action_std','qpos_mean','qpos_std'): assert np.array_equal(stats[emb][k],newstats[emb][k]),(emb,k)
    manifest=dict(teacher=str(teacher/'policy_last.ckpt'),teacher_sha256=sha(teacher/'policy_last.ckpt'),split_sha256=sha(ROOT/'configs/replace50.json'),delta=10,epsilons=[.02,.03,.05],chunk_size=100,slow_down_factor=4,robot_target_fps=30,mode='prior_z_zero_no_future_actions_no_augmentation',episodes={})
    start_time=time.monotonic()
    for split,ds in [('val',val.dataset),('train',train.dataset)]:
        ds.train=False
        for i,path in enumerate(ds.dataset_paths):
            emb=ds.source_embodiment_list[i];n=int(ds.episode_len_list[i]);key=label_key(path);file=out/(key+'.npz')
            metadata=dict(path=str(pathlib.Path(path).resolve()),split=split,embodiment=emb,frames=n,file=file.name,timeline_sha256=hashlib.sha256(ds.episode_frame_indices[i].tobytes()).hexdigest())
            if file.exists():
                with np.load(file) as z: assert z['shape_err'].shape==(n,100)
                manifest['episodes'][metadata['path']]=metadata
                print(name,'cached',split,i,n,flush=True);continue
            shape=np.zeros((n,100),np.float32);cons=np.zeros_like(shape);wrist=np.zeros_like(shape);angle=np.zeros_like(shape);valid=np.zeros((n,100),bool)
            ring=collections.deque(maxlen=11);cursor=0
            loader=torch.utils.data.DataLoader(Episode(ds,i),batch_size=16,num_workers=4,collate_fn=collate_fn,pin_memory=True)
            with torch.inference_mode():
                for data in loader:
                    image,qpos,truth,pad=data[:4];b=len(image)
                    pr=policy(image.cuda(),qpos.cuda());gt=truth.cuda()
                    mu=torch.as_tensor(stats[emb]['action_mean'],device='cuda');std=torch.as_tensor(stats[emb]['action_std'],device='cuda')+1e-6
                    pr=pr*std+mu;gt=gt*std+mu
                    p=scene_tips(pr)[:,:,1];g=scene_tips(gt)[:,:,1]
                    err=torch.linalg.vector_norm((p-p[:,:1])-(g-g[:,:1]),dim=-1).mean(-1).cpu().numpy()
                    we=torch.linalg.vector_norm((pr[:,:,30:33]-pr[:,:1,30:33])-(gt[:,:,30:33]-gt[:,:1,30:33]),dim=-1).cpu().numpy()
                    rp=rotation_6d_rows(pr[:,:,33:39]);rg=rotation_6d_rows(gt[:,:,33:39])
                    # Left re-anchor each orientation at query 0 before comparison.
                    rp=rp[:,:1].transpose(-1,-2)@rp;rg=rg[:,:1].transpose(-1,-2)@rg
                    aa=torch.rad2deg(torch.acos((((rp@rg.transpose(-1,-2)).diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1))).cpu().numpy()
                    points=p.cpu().numpy()
                    shape[cursor:cursor+b]=err;wrist[cursor:cursor+b]=we;angle[cursor:cursor+b]=aa;valid[cursor:cursor+b]=~pad.numpy()
                    for j in range(b):
                        t=cursor+j;ring.append((t,points[j]))
                        if len(ring)==11:
                            old_t,old=ring[0]
                            # Both chunks translated to their shared handoff frame.
                            c=np.linalg.norm((old[10:]-old[10])-(points[j,:90]-points[j,0]),axis=-1).mean(-1)
                            overlap=valid[t,:90]&valid[old_t,10:]
                            cons[old_t,10:]=np.where(overlap,c,0.)
                    cursor+=b
            combined=np.maximum(np.maximum.accumulate(shape,axis=1),np.maximum.accumulate(cons,axis=1))
            arrays={f'reliable_eps{int(e*100):03d}':((combined<=e)&valid).astype(np.uint8) for e in (.02,.03,.05)}
            arrays.update(shape_err=shape.astype(np.float16),cons=cons.astype(np.float16),wrist_err=wrist.astype(np.float16),rotation_err_deg=angle.astype(np.float16),shape_H_eps003=((np.maximum.accumulate(shape,axis=1)<=.03)&valid).sum(1).astype(np.int16),valid=valid,stage_distance=stage_dist(ds,i),raw_indices=ds.episode_frame_indices[i],K_eps003=arrays['reliable_eps003'].sum(1).astype(np.int16),successor_valid=np.arange(n)+10<n)
            # Exact float32 threshold decisions saved above; float16 errors are diagnostics.
            np.savez_compressed(file,**arrays)
            manifest['episodes'][metadata['path']]=metadata
            (out/'manifest.partial.json').write_text(json.dumps(manifest,indent=2))
            print(name,split,i+1,'/',len(ds.dataset_paths),'frames',n,'elapsed',round(time.monotonic()-start_time),flush=True)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summarize(name,manifest)

def summarize(name,manifest):
    rows=collections.defaultdict(lambda:collections.defaultdict(list));all_h=[];far=[];near=[]
    bins=[('>60',lambda d:d>60),('40-60',lambda d:(d>=40)&(d<=60)),('25-40',lambda d:(d>=25)&(d<40)),('15-25',lambda d:(d>=15)&(d<25)),('8-15',lambda d:(d>=8)&(d<15)),('0-8',lambda d:(d>=0)&(d<8)),('<0',lambda d:d<0)]
    for m in manifest['episodes'].values():
        with np.load(ROOT/'labels'/name/m['file']) as z:
            d=z['stage_distance'];v=z['valid'];shape=z['shape_err'].astype(np.float32)
            for e in (.02,.03,.05):
                # Gate uses signal A only, as specified; padded steps are excluded.
                h=z['shape_H_eps003'] if e==.03 else ((np.maximum.accumulate(shape,axis=1)<=e)&v).sum(1)
                if m['embodiment']=='dex5' and e==.03:
                    all_h.extend(h.tolist());far.extend(h[d>60].tolist());near.extend(h[(d>=0)&(d<15)].tolist())
                for b,sel in [('all',np.ones(len(d),bool))]+[(b,fn(d)) for b,fn in bins]:
                    key=f"{m['split']}/{m['embodiment']}/{b}/eps{e}"
                    rows[key]['H'].extend(h[sel].tolist())
                    for label,field in [('shape_m','shape_err'),('wrist_m','wrist_err'),('rotation_deg','rotation_err_deg')]:
                        for k in (0,9,19,29,59,99):
                            mask=sel&v[:,k];rows[key][label+f'@{k+1}'].extend(z[field][mask,k].astype(float).tolist())
    stats={k:{field:dict(count=len(v),quantiles=np.percentile(v,[10,25,50,75,90]).tolist(),mean=float(np.mean(v))) for field,v in fields.items() if len(v)} for k,fields in rows.items()}
    fm=float(np.median(far)) if far else None;nm=float(np.median(near)) if near else None
    spread=float(np.percentile(all_h,75)-np.percentile(all_h,25))
    passed=fm is not None and nm is not None and fm>=2*nm and spread>=15
    result=dict(name=name,passed=passed,far_median=fm,near_median=nm,all_p25_p75_gap=spread,far_count=len(far),near_count=len(near),gate_population='all train+val dex5 frames; residual-only horizon; valid steps only',stats=stats)
    result_path=ROOT/'results'/name/'stage0_anchored_error.json';result_path.parent.mkdir(parents=True,exist_ok=True);result_path.write_text(json.dumps(result,indent=2))
    print('STAGE0',name,'passed',passed,'far',fm,'near',nm,'IQR',spread,flush=True)
    if not passed: raise SystemExit('动态范围不足，Horizon Head 只会学到常数')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('name',choices=['original','geometry']);a=p.parse_args();main(a.name)
