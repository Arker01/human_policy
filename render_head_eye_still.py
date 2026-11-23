"""Use plot_keypoints geometry, with the perspective eye at the head point."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, "/root/shengyin/human_policy/data")
from plot_keypoints import load_cmd_tuple_hdf5

BASE = Path("/root/shengyin/human_policy/data")
OUT = BASE / "sonic_real_teleop_20260906_07_handfix_preview_20260908"
FILES = {
    "old": BASE / "sonic_real_teleop_20260906_07_hat_rootfix/SONIC_Dex5_20260906_07_ep_0000_rootfix.hdf5",
    "new": OUT / "SONIC_Dex5_20260906_07_handfix_ep_0000_rootfix.hdf5",
}
FRAME = 150  # Same action frame as 3.00 seconds in the 50 FPS videos.


def unpack(path):
    d = load_cmd_tuple_hdf5(str(path), full_hand=True, max_frames=FRAME+1)[FRAME]
    head = np.array(d["head"]).reshape(4,4,order="F")
    hands = {}
    for side in ("left", "right"):
        wrist = np.array(d[side+"Wrist"]).reshape(4,4,order="F")
        local = np.array(d[side+"Skeleton"]["joints"]).reshape(25,4,4)[:,3,:3]
        world = local @ wrist[:3,:3].T + wrist[:3,3]
        hands[side] = {"wrist": wrist, "points": world}
    return head, hands


rows = {name:unpack(path) for name,path in FILES.items()}
head, hands = rows["new"]
eye = head[:3,3]
target = np.mean([v["points"].mean(0) for v in hands.values()],axis=0)
forward = target-eye
forward /= np.linalg.norm(forward)
right = np.cross(forward,head[:3,2])
right /= np.linalg.norm(right)
up = np.cross(right,forward)
basis = np.stack([right,up,forward],axis=1)


def project(points, camera_eye):
    camera = (np.asarray(points)-camera_eye) @ basis
    if np.any(camera[...,2] <= 0):
        raise ValueError("Hand geometry behind head camera")
    return camera[...,:2]/camera[...,2,None]


all_uv = np.concatenate([project(v["points"],h[:3,3]) for h,hs in rows.values() for v in hs.values()])
aspect=4/3
half_h=max(np.max(np.abs(all_uv[:,1])),np.max(np.abs(all_uv[:,0]))/aspect)*1.22
half_w=half_h*aspect
meta={"frame":FRAME,"video_time_s":3.0,"dataset":"action", "look_direction":forward.tolist(),
      "camera_up":up.tolist(),"vertical_fov_deg":float(np.degrees(2*np.arctan(half_h))),
      "note":"Eye at each saved head point, common look-at direction toward both hands; not calibrated physical camera intrinsics/extrinsics.",
      "hand_geometry":"Existing plot_keypoints full_hand interpolation of the five fingertip labels", "eyes":{}}
for name,(head,hs) in rows.items():
    camera_eye=head[:3,3]
    meta["eyes"][name]=camera_eye.tolist()
    fig,ax=plt.subplots(figsize=(12,9),dpi=120)
    fig.subplots_adjust(left=.035,right=.965,bottom=.08,top=.87)
    ax.set_xlim(-half_w,half_w);ax.set_ylim(-half_h,half_h)
    ax.set_aspect("equal");ax.set_facecolor("#f6f8fa")
    ax.set_xticks([]);ax.set_yticks([])
    for side,color in [("left","#18824b"),("right","#d13c3c")]:
        uv=project(hs[side]["points"],camera_eye)
        wrist=hs[side]["wrist"]
        wrist_uv=project(wrist[:3,3][None],camera_eye)[0]
        ax.scatter(*wrist_uv,s=60,color=color)
        for finger in range(5):
            chain=uv[finger*5: finger*5+5]
            col="#b12a87" if finger==0 else color
            ax.plot(chain[:,0],chain[:,1],"o-",color=col,lw=1.6,ms=4)
            ax.scatter(*chain[-1],s=95,facecolors="none",edgecolors=col,lw=1.5)
        ax.annotate(side.capitalize()+" wrist",wrist_uv,xytext=(8,-18),textcoords="offset points",color=color,fontsize=12)
        thumb_uv=uv[4]
        ax.annotate("Thumb",thumb_uv,xytext=(8,10),textcoords="offset points",color="#b12a87",fontsize=11)
    fig.suptitle(("OLD labels" if name=="old" else "NEW coupled-LUT labels")+" | 3.00 s (frame 150)",fontsize=20,y=.97)
    fig.text(.5,.917,"Viewpoint at the blue head point, looking toward the hands",ha="center",fontsize=13)
    fig.text(.5,.025,"Same perspective in both images | Green: left / Red: right / Magenta: thumbs",ha="center",fontsize=12)
    fig.savefig(OUT/f"episode_0000_head_eye_3s_{name}.png")
    plt.close(fig)
old=Image.open(OUT/"episode_0000_head_eye_3s_old.png")
new=Image.open(OUT/"episode_0000_head_eye_3s_new.png")
pair=Image.new("RGB",(old.width+new.width,old.height),"white")
pair.paste(old,(0,0));pair.paste(new,(old.width,0))
pair.save(OUT/"episode_0000_head_eye_3s_comparison.jpg",quality=95)
(OUT/"episode_0000_head_eye_3s_camera.json").write_text(json.dumps(meta,indent=2)+"\n")
print(json.dumps(meta,indent=2))
