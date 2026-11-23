import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
sys.path.insert(0, '/home/nerv/qingyaoxu/human_policy')
sys.path.insert(0, '/home/nerv/qingyaoxu/human_policy/data')
import plot_keypoints as p

OUT=Path('/root/shengyin/human_policy/data/sonic_real_teleop_20260906_07_handfix_preview_20260908')
FILES={
 'old': OUT/'SONIC_Dex5_20260906_07_handfix_ep_0000_rootfix.hdf5',
 'new': OUT/'SONIC_Dex5_20260906_07_handfix_ep_0000_rootfix.hdf5',
}
# The old HDF5 is kept in the remote source directory; copied locally below.
FILES['old']=Path('/root/shengyin/human_policy/data/sonic_real_teleop_20260906_07_hat_rootfix/SONIC_Dex5_20260906_07_ep_0000_rootfix.hdf5')
FRAME=150

def make(name,path):
    _, fs=p.main(str(path),full_hand=True,max_frames=FRAME+1,draw_spine=True)
    f=fs[FRAME]
    eye=np.asarray(f['positions']['head'])
    # Convert world points into the head frame: origin at blue point, forward is head +X.
    # Existing plotting style and its standard camera are retained after this transform.
    R=np.column_stack([f['axes']['head'][i]-eye for i in range(3)])/0.1
    def tf(x): return (np.asarray(x)-eye)@R
    fig=plt.figure(figsize=(12.8,7.2),dpi=100)
    ax=fig.add_subplot(111,projection='3d')
    points=[]
    colors={'head':'blue','right_wrist':'red','left_wrist':'green','waist':'orange'}
    for part in ('head','right_wrist','left_wrist','waist'):
        if part not in f['positions']: continue
        pos=tf(f['positions'][part]); points.append(pos)
    for side in ('right','left'):
        points.append(tf(f['fingers'][side]))
    allp=np.concatenate([np.atleast_2d(x) for x in points]); m=.12
    ax.set_xlim(allp[:,0].min()-m,allp[:,0].max()+m); ax.set_ylim(allp[:,1].min()-m,allp[:,1].max()+m); ax.set_zlim(allp[:,2].min()-m,allp[:,2].max()+m)
    ax.view_init(elev=15,azim=210); ax.set_xlabel('Head X (forward)'); ax.set_ylabel('Head Y'); ax.set_zlabel('Head Z'); ax.set_title(('OLD' if name=='old' else 'NEW')+' labels | 3.00 s | eye at blue head point')
    for part,c in colors.items():
        if part not in f['positions']: continue
        pos=tf(f['positions'][part]); ax.scatter(*pos,color=c,s=60,label=part)
        for end in f['axes'][part]: ax.plot([*pos[:1],tf(end[None])[0,0]],[*pos[1:2],tf(end[None])[0,1]],[*pos[2:3],tf(end[None])[0,2]],color=c,lw=1)
    for side,c in (('right','red'),('left','green')):
        pts=tf(f['fingers'][side]); ax.scatter(pts[:,0],pts[:,1],pts[:,2],color=c,s=18,label=side+'_fingers')
    ax.legend(fontsize=7,loc='upper left'); fig.tight_layout(); fig.savefig(OUT/f'episode_0000_original_style_head_3s_{name}.png'); plt.close(fig)

for n,x in FILES.items(): make(n,x)
print('done')
