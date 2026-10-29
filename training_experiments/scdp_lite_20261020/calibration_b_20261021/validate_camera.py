"""Fresh recording holdout and leave-one-training-recording-out stability."""
import json,pathlib,contextlib,io
import numpy as np
from fit_camera import Correspondences,fit,predict,metrics,make_overlays

ROOT=pathlib.Path(__file__).resolve().parent
candidate=json.loads((ROOT/'wrist_optimized_time.json').read_text())
params=np.array(min(candidate['multistart'],key=lambda x:x['cost'])['params'])
newrecords=json.loads((ROOT/'detections_167_176_182_188_195_202.json').read_text())
fresh=Correspondences(newrecords)
pred,cam,_=predict(params,fresh.points(params[10]))
error=np.linalg.norm(pred-fresh.uv,axis=1)
report=dict(candidate_source='wrist_optimized_time.json',locked_before_fresh_holdout=True,
    fresh_episodes=sorted(map(int,set(fresh.ep))),frames=len(fresh.frame_records),
    detected_sides=sorted(set(x[2] for x in fresh.items)),
    wrist=metrics(error[fresh.joint==0]),fingertips_diagnostic=metrics(error[fresh.joint>0]),
    by_episode={str(e):metrics(error[(fresh.ep==e)&(fresh.joint==0)]) for e in sorted(set(fresh.ep))})
print('FRESH HOLDOUT',json.dumps(report,indent=2),flush=True)
make_overlays(fresh,params,np.zeros(len(fresh.items),dtype=bool),ROOT/'fresh_holdout_overlays')
make_overlays(fresh,params,np.zeros(len(fresh.items),dtype=bool),ROOT/'fresh_wrist_overlays',wrist_only=True)
original=json.loads(next(ROOT.glob('detections_155_156_158_160*.json')).read_text())
corr=Correspondences(original)
training=candidate['train_episodes'];jack=[]
for dropped in training:
    with contextlib.redirect_stdout(io.StringIO()):
        result,p,_,_,_=fit(corr,[e for e in training if e!=dropped],wrist_only=True,focals=(180,))
    projected,_,_=predict(p,fresh.points(p[10]))
    difference=np.linalg.norm(projected[fresh.joint==0]-pred[fresh.joint==0],axis=1)
    jack.append(dict(dropped_episode=dropped,K=result['K'],time_offset_s=float(p[10]),
                     wrist_projection_change=metrics(difference),params=p.tolist()))
    print('jackknife dropped',dropped,'mean change',float(difference.mean()),flush=True)
report['jackknife']=jack
intrinsics=np.array([[j['K'][0][0],j['K'][1][1],j['K'][0][2],j['K'][1][2]] for j in jack])
report['jackknife_intrinsics_ranges']=dict(zip(['fx','fy','cx','cy'],
    [[float(v.min()),float(v.max())] for v in intrinsics.T]))
report['jackknife_time_offset_range_s']=[min(j['time_offset_s'] for j in jack),max(j['time_offset_s'] for j in jack)]
(ROOT/'validation.json').write_text(json.dumps(report,indent=2))
candidate['fresh_holdout']=report['wrist'];candidate['fresh_holdout_episodes']=report['fresh_episodes']
candidate['scope']='human_new Vision Pro system screen recordings; empirically validated RIGHT WRIST only'
candidate['status']='preliminary equivalent camera; all-point/left-hand/old-session/robot validation not passed'
candidate['automatic_training_enabled']=False
candidate['jackknife_intrinsics_ranges']=report['jackknife_intrinsics_ranges']
candidate['jackknife_time_offset_range_s']=report['jackknife_time_offset_range_s']
candidate['T_optical_from_kp_frame_formula']='T_optical_from_arkit_head @ inverse(T_V_from_head_at_video_time) @ T_V_from_kp_frame'
candidate['integration_warning']='Use original ARKit head (not tracker pose). Preserve measured temporal alignment. No current-pose or predicted wrist was used to define camera extrinsics. Do not directly plug T_optical_from_arkit_head into the HAT KP-frame transform slot.'
(ROOT/'human_new_equivalent_camera.json').write_text(json.dumps(candidate,indent=2))
print('WROTE human_new_equivalent_camera.json and validation.json',flush=True)
