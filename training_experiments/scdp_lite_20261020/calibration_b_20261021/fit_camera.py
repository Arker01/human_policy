"""Robust equivalent screenshot camera calibration; never a physical calibration."""
import argparse,json,pathlib
import cv2,numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

ROOT=pathlib.Path(__file__).resolve().parent
OPTICAL_FROM_ARKIT=np.diag([1.,-1.,-1.])
JOINTS=np.array([0,4,9,14,19,24]);MP_JOINTS=np.array([0,4,8,12,16,20])
NAMES=['wrist','thumb_tip','index_tip','middle_tip','ring_tip','little_tip']


def pose_track(path,side):
    frames=json.loads(pathlib.Path(path).read_text())['frames'];track=[];valid=[]
    for frame in frames:
        head=np.asarray(frame['head']).reshape(4,4)
        wrist=np.asarray(frame[f'{side}Wrist']).reshape(4,4)
        skeleton=np.asarray(frame[f'{side}Skeleton']['joints']).reshape(-1,4,4)
        points=skeleton[JOINTS,:3,3].copy();points[0]=0
        world=points@wrist[:3,:3].T+wrist[:3,3]
        local=(world-head[:3,3])@head[:3,:3]
        track.append(local)
        valid.append(bool(frame.get(f'{side}Wrist_valid',True)) and not np.allclose(wrist,np.eye(4)))
    return np.array(track),np.array(valid),np.array([f['timestamp'] for f in frames])


class Correspondences:
    def __init__(self,records,swap=False,min_score=.8):
        self.records=records;cache={};items=[];self.frame_records=[]
        for rec in records:
            for detection in rec['hands']:
                if detection['score']<min_score:continue
                side=detection['label'].lower()
                if swap:side='left' if side=='right' else 'right'
                key=(rec['pose'],side)
                if key not in cache:cache[key]=pose_track(*key)
                track,valid,times=cache[key];i=rec['index']
                if not valid[i]:continue
                # Do not interpolate through missing ARKit tracking during fitting.
                if not valid[max(0,i-7):min(len(valid),i+8)].all():continue
                uv=np.array(detection['uv320'])[MP_JOINTS]
                if not ((uv>=0).all() and (uv<np.array([320,240])).all()):continue
                self.frame_records.append((rec,side,uv,track))
                for joint in range(6):
                    items.append((rec['episode'],rec['index'],side,joint,uv[joint],track[:,joint],times))
        if not items:raise ValueError('No reliable correspondences')
        self.items=items
        self.uv=np.array([x[4] for x in items]);self.ep=np.array([x[0] for x in items])
        self.frame=np.array([x[1] for x in items]);self.joint=np.array([x[3] for x in items])
        self.t0=np.array([x[6][x[1]] for x in items])
        # Piecewise-linear interpolation of head-relative point trajectories.
        self.tracks=[x[5] for x in items];self.times=[x[6] for x in items]
        self.near_times=[];self.near_xyz=[]
        for x in items:
            indices=np.clip(x[1]+np.arange(-9,10),0,len(x[6])-1)
            self.near_times.append(x[6][indices]);self.near_xyz.append(x[5][indices])
        self.near_times=np.array(self.near_times);self.near_xyz=np.array(self.near_xyz)

    def points(self,delta):
        query=self.t0+delta
        right=(self.near_times<=query[:,None]).sum(1).clip(1,18);left=right-1;row=np.arange(len(right))
        lo=self.near_times[row,left];hi=self.near_times[row,right]
        weight=np.clip((query-lo)/np.maximum(hi-lo,1e-9),0,1)
        return self.near_xyz[row,left]*(1-weight[:,None])+self.near_xyz[row,right]*weight[:,None]


def predict(params,points):
    fx,fy=np.exp(params[:2]);cx,cy=params[2:4]
    R=Rotation.from_rotvec(params[4:7]).as_matrix()@OPTICAL_FROM_ARKIT
    cam=points@R.T+params[7:10]
    z=np.maximum(cam[:,2],.03)
    uv=cam[:,:2]/z[:,None]*[fx,fy]+[cx,cy]
    return uv,cam,R


def metrics(errors):
    if not len(errors):return None
    return dict(points=len(errors),mean_px=float(np.mean(errors)),median_px=float(np.median(errors)),
                p90_px=float(np.percentile(errors,90)),p95_px=float(np.percentile(errors,95)),
                fraction_le_3px=float(np.mean(errors<=3)),fraction_le_5px=float(np.mean(errors<=5)),
                fraction_le_10px=float(np.mean(errors<=10)))


def fit(corr,train_eps,time_offset=True,seed=0,wrist_only=False,focals=(110,180,280)):
    train=np.isin(corr.ep,train_eps)
    # Wrist definitions differ between detectors; only anatomically matching
    # fingertips determine K/extrinsics. Wrist is an independent diagnostic.
    chosen=(corr.joint==0) if wrist_only else (corr.joint>0)
    selected=train&chosen
    uv=corr.uv[selected]
    lower=np.r_[np.log([40,40]),[-80,-60],[-.7]*3,[-.3]*3,[-.20] if time_offset else []]
    upper=np.r_[np.log([800,1000]),[400,300],[.7]*3,[.3]*3,[.20] if time_offset else []]
    frozen=corr.points(0)[selected]
    def residual(p):
        points=corr.points(p[10])[selected] if time_offset else frozen
        pred,cam,_=predict(p,points)
        # Only weak physical regularization; validation contains no fitting.
        prior=np.r_[p[4:7]*.5,p[7:10]*2]
        return np.r_[(pred-uv).ravel(), np.minimum(cam[:,2]-.03,0)*1000,prior]
    results=[]
    for focal in focals:
        x0=np.r_[np.log([focal,focal*4/3]),[160,120],np.zeros(6),[0.] if time_offset else []]
        result=least_squares(residual,x0,bounds=(lower,upper),loss='soft_l1',f_scale=2.,max_nfev=250,
                             diff_step=1e-5,ftol=1e-9,xtol=1e-9,gtol=1e-7)
        results.append(result)
        print('fit',focal,'cost',result.cost,'nfev',result.nfev,'params',result.x,flush=True)
    best=min(results,key=lambda r:r.cost)
    points=corr.points(best.x[10] if time_offset else 0)
    pred,cam,R=predict(best.x,points);error=np.linalg.norm(pred-corr.uv,axis=1)
    fx,fy=np.exp(best.x[:2]);cx,cy=best.x[2:4]
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=best.x[7:10]
    report=dict(model='fixed_equivalent_screen_recording_pinhole',physical_camera_calibration=False,
        image_hw=[240,320],K=[[fx,0,cx],[0,fy,cy],[0,0,1]],T_optical_from_arkit_head=T.tolist(),
        T_arkit_head_from_optical=np.linalg.inv(T).tolist(),pose_time_offset_seconds=float(best.x[10]) if time_offset else 0.,
        time_offset_convention='video frame i uses pose timestamp[i] + delta; pre-existing video/JSON alignment is retained',
        train_episodes=list(map(int,train_eps)),heldout_episodes=sorted(set(map(int,corr.ep[~train]))),
        fitting_landmarks=[NAMES[0]] if wrist_only else NAMES[1:],detector='MediaPipe HandLandmarker 1.1.1 float16',
        train=metrics(error[train&chosen]),heldout=metrics(error[~train&chosen]),
        train_wrist_diagnostic=metrics(error[train&(corr.joint==0)]),heldout_wrist_diagnostic=metrics(error[~train&(corr.joint==0)]),
        by_episode={str(ep):metrics(error[(corr.ep==ep)&(corr.joint>0)]) for ep in sorted(set(corr.ep))},
        by_joint={name:metrics(error[(~train)&(corr.joint==i)]) for i,name in enumerate(NAMES)},
        condition_number=float(np.linalg.cond(best.jac)),
        multistart=[dict(cost=float(r.cost),params=r.x.tolist()) for r in results],
        automatic_correspondences=True,manual_annotation_validation=False,
        warning='Equivalent recording model only. Automatic 2D detector has occlusion/anatomical errors; inspect overlays before SCDP. No robot calibration.')
    return report,best.x,pred,error,train


def make_overlays(corr,params,train,out,wrist_only=False):
    out.mkdir(exist_ok=True)
    # Deterministic held-out frames, selected without consulting fitting error.
    for ep in sorted(set(corr.ep[~train])):
        frames=[x for x in corr.frame_records if x[0]['episode']==ep]
        if not frames:continue
        for k in np.linspace(0,len(frames)-1,min(3,len(frames))).astype(int):
            rec,side,uv,track=frames[k]
            video=cv2.VideoCapture(rec['video']);video.set(cv2.CAP_PROP_POS_FRAMES,rec['index']);ok,image=video.read();video.release()
            if not ok:continue
            image=cv2.resize(image,(960,720))
            times=np.array([f['timestamp'] for f in json.loads(pathlib.Path(rec['pose']).read_text())['frames']])
            t=times[rec['index']]+(params[10] if len(params)>10 else 0)
            pts=np.array([[np.interp(t,times,track[:,j,k]) for k in range(3)] for j in range(6)])
            projected,_,_=predict(params,pts)
            selected=slice(0,1) if wrist_only else slice(None)
            for observed,predicted in zip(uv[selected],projected[selected]):
                a=tuple(np.rint(observed*3).astype(int));b=tuple(np.rint(predicted*3).astype(int))
                cv2.line(image,a,b,(255,180,0),1);cv2.circle(image,a,5,(0,255,0),2);cv2.drawMarker(image,b,(0,0,255),cv2.MARKER_CROSS,12,2)
            cv2.putText(image,f'HELD OUT ep{ep} frame{rec["index"]} {side}: green=detected red=projected',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
            if wrist_only:
                err=float(np.linalg.norm(uv[0]-projected[0]))
                cv2.putText(image,f'WRIST ONLY | error at 320x240: {err:.2f}px',(10,50),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1)
            cv2.imwrite(str(out/f'heldout_ep{ep}_f{rec["index"]}_{side}.jpg'),image)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--detections',required=True)
    parser.add_argument('--train',nargs='+',type=int,required=True);parser.add_argument('--swap',action='store_true')
    parser.add_argument('--fixed-time',action='store_true');parser.add_argument('--name',default='equivalent_camera')
    parser.add_argument('--wrist-only',action='store_true')
    args=parser.parse_args();records=json.loads(pathlib.Path(args.detections).read_text())
    corr=Correspondences(records,args.swap)
    print('correspondences',len(corr.items),'frames',len(corr.frame_records),'episodes',sorted(set(corr.ep)),flush=True)
    report,params,pred,error,train=fit(corr,args.train,not args.fixed_time,wrist_only=args.wrist_only)
    (ROOT/f'{args.name}.json').write_text(json.dumps(report,indent=2))
    np.savez_compressed(ROOT/f'{args.name}_residuals.npz',predicted=pred,observed=corr.uv,error=error,
        episode=corr.ep,frame=corr.frame,joint=corr.joint,is_train=train)
    make_overlays(corr,params,train,ROOT/f'{args.name}_overlays')
    print(json.dumps({k:report[k] for k in ('K','T_optical_from_arkit_head','pose_time_offset_seconds','train','heldout','heldout_wrist_diagnostic','condition_number')},indent=2),flush=True)

if __name__=='__main__':main()
