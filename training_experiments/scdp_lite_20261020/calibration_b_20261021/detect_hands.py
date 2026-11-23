"""Read existing recordings and store automatic image correspondences only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import argparse,json,pathlib
import cv2,numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ROOT=pathlib.Path(__file__).resolve().parent

def locate(episode):
    for folder in (pathlib.Path('/tmp/1001_human_input2'),pathlib.Path('/mnt/nvme0n1/0902_session/out_episodes')):
        pose=folder/f'wholebody-{episode}_unified_V.json'
        video=(folder if folder.name!='out_episodes' else folder.parent/'out_videos')/f'wholebody-{episode}.mp4'
        if pose.exists() and video.exists():return pose,video
    raise FileNotFoundError(episode)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--episodes',nargs='+',type=int,required=True)
    parser.add_argument('--frames',type=int,default=45);args=parser.parse_args()
    options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT/'hand_landmarker.task')),
        num_hands=2,min_hand_detection_confidence=.5,min_hand_presence_confidence=.7)
    records=[]
    with vision.HandLandmarker.create_from_options(options) as detector:
        for ep in args.episodes:
            pose,video=locate(ep);data=json.loads(pose.read_text());frames=data['frames'];cap=cv2.VideoCapture(str(video))
            n=min(len(frames),int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
            selected=np.unique(np.linspace(3,n-4,args.frames).round().astype(int));count=0
            for index in selected:
                cap.set(cv2.CAP_PROP_POS_FRAMES,int(index));ok,bgr=cap.read()
                if not ok:continue
                h,w=bgr.shape[:2];small=cv2.resize(bgr,(960,round(h*960/w)))
                rgb=cv2.cvtColor(small,cv2.COLOR_BGR2RGB)
                result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb))
                hands=[]
                for landmarks,handed in zip(result.hand_landmarks,result.handedness):
                    hands.append(dict(label=handed[0].category_name,score=handed[0].score,
                        uv320=[[p.x*320,p.y*240] for p in landmarks]))
                if hands:count+=1
                records.append(dict(episode=ep,index=int(index),pose=str(pose),video=str(video),native_hw=[h,w],
                    timestamp=frames[index]['timestamp'],hands=hands))
            cap.release();print(f'episode={ep} detected_frames={count}/{len(selected)}',flush=True)
    out=ROOT/('detections_'+'_'.join(map(str,args.episodes))+'.json')
    out.write_text(json.dumps(records));print('saved',out,flush=True)

if __name__=='__main__':main()
