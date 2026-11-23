"""Shared prior-only query-0 protocol for baseline, v1 and v2; no image artifacts.

Matches original eval_mpjpe_batch.py: 30 Hz nearest-frame robot timeline,
qpos std +1e-8, output std (no epsilon), dirty-start defaults, 16 scene joints,
per-episode macro averaging. Closing onset from the existing GT hand aperture
detector is a grasp proxy; pre-close 0.5 seconds are measured in source time.
"""
import runtime
import argparse
import contextlib
import importlib.util
import json
import pickle
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from calibrated_data import CalibrationStore
from spatial_conditioner import SpatialTrajectoryConditioner, resize_crop_affine
from hat_adapter import SCDPHAT
from load_policy import load_scdp_policy


def import_file(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def camera_at(calibration, raw):
    if calibration is None:
        return dict(K=torch.zeros(3,3),D=torch.zeros(8),T_cam_from_kp=torch.eye(4),
            pixel_from_raw=torch.eye(3),raw_hw=torch.zeros(2),valid=torch.tensor(False),
            hand_valid=torch.zeros(2,dtype=torch.bool))
    k=calibration['K'] if calibration['K'].ndim==2 else calibration['K'][raw]
    return dict(K=torch.as_tensor(k).float(),D=torch.as_tensor(calibration['D']).float(),
        T_cam_from_kp=torch.as_tensor(calibration['T_cam_from_kp'][raw]).float(),
        pixel_from_raw=resize_crop_affine(calibration['raw_hw'],(240,320)),
        raw_hw=torch.as_tensor(calibration['raw_hw']).float(),valid=torch.tensor(bool(calibration['valid'][raw])),
        hand_valid=torch.as_tensor(calibration['hand_valid']).bool())


def conditioning(calibration, indices, stats, device):
    cameras=[camera_at(calibration,int(i)) for i in indices]
    return dict(scdp_camera={k:torch.stack([c[k] for c in cameras]).to(device) for k in cameras[0]},
        scdp_action_mean=torch.as_tensor(stats['action_mean'],device=device).float()[None].expand(len(indices),-1),
        scdp_action_std=torch.as_tensor(stats['action_std'],device=device).float()[None].expand(len(indices),-1))


@contextlib.contextmanager
def single_scale(wrapper, conditioner):
    previous, multiscale=wrapper.conditioner,wrapper.multiscale
    wrapper.conditioner,wrapper.multiscale=conditioner,False
    try:yield
    finally:wrapper.conditioner,wrapper.multiscale=previous,multiscale


class EvaluationEngine:
    def __init__(self, checkpoint=None, reference=None, device='cuda'):
        self.device=device
        if checkpoint:
            self.policy,self.stats,self.config=load_scdp_policy(checkpoint,device)
            self.step=torch.load(checkpoint,map_location='cpu',weights_only=False)['step']
        else:
            self.policy,self.config=runtime.build_policy()
            self.policy.model=SCDPHAT(self.policy.model,dict(enabled=True,anchors=8,film_dim=256,
                spatial_mode='five_scale_v2',pyramid_channels=128))
            self.policy.to(device).eval()
            self.stats=pickle.load(open(runtime.BASE/'dataset_stats.pkl','rb'));self.step=0
        self.wrapper=self.policy.model
        self.mode='v2' if self.wrapper.multiscale else 'v1'
        self.reference=None;self.reference_step=None
        if reference:
            saved=torch.load(reference,map_location='cpu',weights_only=False)
            if Path(saved['base']).resolve()!=runtime.BASE.resolve():raise ValueError('V1 comparison uses another base')
            if saved['step']!=self.step:raise ValueError('V1/v2 comparisons require the same number of training steps')
            self.reference=SpatialTrajectoryConditioner(feature_dim=768,hidden_dim=512,film_dim=256,
                anchors=8,enabled=True).to(device).eval()
            self.reference.load_state_dict(saved['conditioner'],strict=True)
            self.reference_step=saved['step']

    @torch.inference_mode()
    def predict(self, image, qpos, cond):
        result={self.mode:self.policy(image,qpos,conditioning_dict=cond)}
        result['baseline']=self.wrapper.debug['coarse'].clone()
        if self.reference is not None:
            with single_scale(self.wrapper,self.reference):
                result['v1']=self.policy(image,qpos,conditioning_dict=cond)
        return result


def image_at(core, hdf, index):
    key=next(k for k in ('observation.image.top','observation.image.left','observation.image.right') if k in hdf)
    image=core._read_image_frame(hdf[key],int(index))
    if image.shape[-1]==4:image=image[...,:3]
    if image.shape[:2]!=(240,320):image=cv2.resize(image,(320,240))
    return torch.from_numpy(image.copy()).permute(2,0,1).float().unsqueeze(0)/255.


def timeline(length,fps,embodiment):
    if 'human' not in embodiment.lower() and fps>30.+1e-6:
        return np.unique(np.clip(np.rint(np.arange(max(1,int(np.floor(length/(fps/30.)))))*(fps/30.)).astype(int),0,length-1))
    return np.arange(length)


def evaluate(engine, max_frames=None, batch_size=64):
    core=import_file('original_mpjpe',runtime.REPO/'data/eval_mpjpe_batch.py')
    aux=import_file('eval_hand_events',Path(json.loads((runtime.ROOT/'configs/bases.json').read_text())['geometry']['source_root'])/'hdt/auxiliary_tasks.py')
    store=CalibrationStore(runtime.CALIBRATION_MANIFEST)
    metrics={};preclose={};episode_count=0
    for entry in json.loads(runtime.DATASET.read_text())['val']:
        for filename in sorted(entry['file_list']):
            path=Path(entry['dataset_path'])/filename;calibration=store.load(str(path))
            with h5py.File(path,'r') as h:
                embodiment=str(h.attrs['embodiment']);fps=float(h.attrs.get('fps',30.))
                states=h['observation.state'][:];truth=h['action'][:];stats=engine.stats[embodiment]
                indices=timeline(len(truth),fps,embodiment)
                dirty=core._detect_valid_start_from_actions(truth[indices],check_frames=20,jump_threshold_m=.3,settle_frames=0)
                indices=indices[dirty:]
                if max_frames and len(indices)>max_frames:
                    indices=indices[np.linspace(0,len(indices)-1,max_frames).round().astype(int)]
                targets,masks,weights,events=aux.transition_labels(states,fps)
                closing=np.zeros((len(states),2),bool)
                for hand,kind,onset,end in events:
                    if kind=='close':closing[max(0,onset-int(round(.5*fps))):onset,hand]=True
                predictions={};gt=[]
                for start in range(0,len(indices),batch_size):
                    raw=indices[start:start+batch_size]
                    image=torch.stack([image_at(core,h,i) for i in raw]).to(engine.device)
                    qpos=torch.as_tensor((states[raw]-stats['qpos_mean'])/(stats['qpos_std']+1e-8),device=engine.device).float()
                    cond=conditioning(calibration,raw,stats,engine.device)
                    output=engine.predict(image,qpos,cond)
                    gt.append(truth[raw])
                    for mode,pred in output.items():
                        physical=pred[:,0].cpu().numpy()*stats['action_std']+stats['action_mean']
                        predictions.setdefault(mode,[]).append(physical)
                        if calibration is None:continue
                        for hand,offset in enumerate((80,30)):
                            good=closing[raw,hand]&calibration['valid'][raw]&calibration['hand_valid'][hand]
                            if not good.any():continue
                            delta=physical[good,offset:offset+3]-truth[raw[good],offset:offset+3]
                            rotation=calibration['T_cam_from_kp'][raw[good],:3,:3]
                            error=np.einsum('bij,bj->bi',rotation,delta)*1000
                            key=embodiment+'/'+mode+'/'+('left' if hand==0 else 'right')
                            preclose.setdefault(key,[]).append(error)
                for mode,parts in predictions.items():
                    values=core.eval_arrays_mpjpe(np.concatenate(gt),np.concatenate(parts),dirty_start_check_frames=0)
                    metrics.setdefault(embodiment+'/'+mode,[]).append(values)
                episode_count+=1
                print('EVAL episode',episode_count,path.name,'observations',len(indices),flush=True)
    summary={}
    for key,values in metrics.items():
        summary[key]={k:float(np.mean([v[k] for v in values])) for k in values[0] if k.startswith('mpjpe_') and k.endswith('_mm')}
        summary[key].update(episodes=len(values),observations=sum(v['timesteps_compared'] for v in values))
    wrist={}
    for key,parts in preclose.items():
        error=np.concatenate(parts)
        wrist[key]=dict(frames=len(error),camera_xyz_mae_mm=np.abs(error).mean(0).tolist(),
                        wrist_l2_mm=float(np.linalg.norm(error,axis=-1).mean()))
    return dict(version=engine.mode,base=runtime.BASE_VARIANT,step=engine.step,v1_step=engine.reference_step,
        query0_protocol='original eval_mpjpe_batch.py, 16 joints, per-episode macro average, prior only',
        max_frames_per_episode=max_frames,query0=summary,preclose_05s=wrist,
        preclose_definition='GT finger closing onset proxy, source-recording seconds; camera axes right/down/forward; frame-weighted',
        bottle_translation_response='pending: requires valid fixed-state bottle-only paired observations; whole-image shift is not a substitute')


def evaluate_bottle_pairs(engine, manifest):
    """Read real verified bottle-shift pairs; force identical state and camera inputs."""
    from spatial_conditioner import project_wrist
    core=import_file('pair_mpjpe',runtime.REPO/'data/eval_mpjpe_batch.py')
    store=CalibrationStore(runtime.CALIBRATION_MANIFEST);results=[]
    for pair in json.loads(Path(manifest).read_text()):
        if pair.get('bottle_only_change_verified') is not True:
            raise ValueError('Pair needs independent verification that only the bottle moved')
        path=Path(pair['episode']);a,b=int(pair['frame_a']),int(pair['frame_b'])
        calibration=store.load(str(path))
        if calibration is None:raise ValueError('Paired episode needs supported camera calibration')
        with h5py.File(path,'r') as h:
            state=h['observation.state'][a];other=h['observation.state'][b]
            # Reject motion of hands/head/root, rather than treating it as bottle response.
            points=np.r_[0:3,10:28,30:33,40:58,80:83,89:92]
            rotations=np.r_[3:9,33:39,83:89,92:98]
            if np.max(np.abs(state[points]-other[points]))>.005 or np.max(np.abs(state[rotations]-other[rotations]))>.01:
                raise ValueError('Pair includes camera/robot motion beyond fixed-state tolerances')
            stats=engine.stats[str(h.attrs['embodiment'])]
            image=torch.stack([image_at(core,h,a),image_at(core,h,b)]).to(engine.device)
            normalized=(state-stats['qpos_mean'])/(stats['qpos_std']+1e-8)
            qpos=torch.as_tensor(np.stack([normalized,normalized]),device=engine.device).float()
            cond=conditioning(calibration,[a,a],stats,engine.device)
            predictions=engine.predict(image,qpos,cond)
        bottle_delta=np.asarray(pair['bottle_uv_b'],float)-np.asarray(pair['bottle_uv_a'],float)
        if bottle_delta.shape!=(2,) or np.linalg.norm(bottle_delta)<3:raise ValueError('Bottle shift must be at least 3 pixels at 320x240')
        query=int(pair.get('query_index',0))
        if not 0<=query<100:raise ValueError('Invalid query index')
        for mode,pred in predictions.items():
            physical=pred*torch.as_tensor(stats['action_std'],device=engine.device)+torch.as_tensor(stats['action_mean'],device=engine.device)
            xyz=physical[:,query,torch.tensor([[80,81,82],[30,31,32]],device=engine.device)][:,None]
            grid,valid=project_wrist(xyz,cond['scdp_camera'])
            uv=((grid+1)*grid.new_tensor([320,240])/2-.5).cpu().numpy()[:,0]
            for hand,name in enumerate(('left','right')):
                good=bool(valid[:,0,hand].all())
                delta=uv[1,hand]-uv[0,hand]
                results.append(dict(mode=mode,hand=name,query=query,valid=good,
                    wrist_delta_mm=((xyz[1,0,hand]-xyz[0,0,hand])*1000).cpu().tolist(),
                    projected_wrist_delta_px=delta.tolist() if good else None,
                    bottle_delta_px=bottle_delta.tolist(),
                    response_gain=float(np.dot(delta,bottle_delta)/np.dot(bottle_delta,bottle_delta)) if good else None))
    return dict(protocol='real verified bottle-only pairs; exactly fixed qpos and current-head camera conditioning',pairs=results)


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--reference')
    p.add_argument('--max-frames',type=int);p.add_argument('--output',type=Path)
    p.add_argument('--pairs',type=Path)
    args=p.parse_args();torch.set_num_threads(4);torch.manual_seed(0);np.random.seed(0)
    if args.output and not args.output.resolve().is_relative_to(runtime.ROOT.resolve()):raise ValueError('Output outside v2')
    engine=EvaluationEngine(args.checkpoint,args.reference)
    result=evaluate(engine,args.max_frames)
    if args.pairs:result['bottle_translation_response']=evaluate_bottle_pairs(engine,args.pairs)
    print('EVALUATION',json.dumps(result),flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
