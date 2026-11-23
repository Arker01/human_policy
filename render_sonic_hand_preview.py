"""Render synchronized recorded image, coupled hand joints and HDF5 tips."""
import argparse
import json
import subprocess
from pathlib import Path

import cv2
import h5py
import imageio_ffmpeg
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pinocchio as pin
import pyarrow.parquet as pq

from convert_sonic_to_hat_hdf5 import (
    RIGHT_HAND_URDF, RIGHT_TIP_NAMES, inspire_sdk_to_q, load_hand_model,
)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", type=Path, required=True)
    ap.add_argument("--old-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    with h5py.File(args.file, "r") as f, h5py.File(args.old_file, "r") as old:
        tips = f["observation.state"][:,43:58].reshape(-1,5,3)
        old_tips = old["observation.state"][:,43:58].reshape(-1,5,3)
        fps = float(f.attrs["fps"])
        parquet_path = str(f.attrs["source_parquet"])
        source_video = str(f.attrs["source_video"])
    raw = pq.read_table(parquet_path, columns=["observation.inspire.right_position"])
    motors = np.asarray(raw.column(0).to_pylist())
    if len(motors) != len(tips) or len(old_tips) != len(tips):
        raise ValueError("Preview expects aligned original-rate episodes")
    model, data, ids = load_hand_model(RIGHT_HAND_URDF, RIGHT_TIP_NAMES)
    fingers = ["thumb", "index", "middle", "ring", "pinky"]
    joint_ids = [[i for i,name in enumerate(model.names) if name.startswith("R_" + finger + "_")]
                 for finger in fingers]
    chains, error = [], []
    for motor, label in zip(motors, tips):
        pin.framesForwardKinematics(model, data, inspire_sdk_to_q(motor))
        predicted = np.stack([data.oMf[i].translation for i in ids])
        error.append(np.linalg.norm(predicted-label,axis=1))
        chains.append([np.stack([*[data.oMi[i].translation.copy() for i in joints], predicted[k]]) * 100
                       for k,joints in enumerate(joint_ids)])
    maximum_mm = float(np.max(error) * 1000)
    if maximum_mm > .001:
        raise ValueError(f"Saved HDF5 does not match coupled FK: {maximum_mm} mm")
    colors = ["#b83280", "#d55e00", "#0072b2", "#009e73", "#7b61a8"]
    fig = plt.figure(figsize=(12.8,7.2), dpi=100, layout="constrained")
    grid = fig.add_gridspec(2,2,width_ratios=[1.25,1])
    camera = fig.add_subplot(grid[:,0])
    camera.axis("off")
    image_artist = camera.imshow(np.zeros((480,640,3),dtype=np.uint8))
    views = []
    all_points = np.concatenate([np.concatenate(c) for c in chains] + [old_tips.reshape(-1,3)*100])
    for row, (horizontal,vertical,name) in enumerate([(0,1,"X-Y"),(2,1,"Z-Y")]):
        ax = fig.add_subplot(grid[row,1])
        ax.set_title("Right hand " + name)
        ax.set_xlabel("XYZ"[horizontal] + " (cm)")
        ax.set_ylabel("XYZ"[vertical] + " (cm)")
        ax.set_xlim(all_points[:,horizontal].min()-1,all_points[:,horizontal].max()+1)
        ax.set_ylim(all_points[:,vertical].min()-1,all_points[:,vertical].max()+1)
        ax.set_aspect("equal",adjustable="box")
        ax.grid(alpha=.2)
        lines=[ax.plot([],[],"o-",color=c,linewidth=2,markersize=3,label=n)[0]
               for c,n in zip(colors,fingers)]
        markers=ax.scatter([],[],s=75,facecolors="none",edgecolors=colors,linewidths=1.5)
        previous=ax.scatter([],[],s=40,color="#666666",marker="x",alpha=.65)
        views.append((horizontal,vertical,lines,markers,previous))
        if row==0:
            ax.legend(loc="upper left",bbox_to_anchor=(1.02,1.0),fontsize=7)
    fig.suptitle("Recorded image | coupled joints + HDF5 tips (circles) | previous tips (x)",fontsize=11)
    camera.set_title("00.00 s | frame 000 | bend 0.000 / rot 0.000",fontsize=10)
    fig.canvas.draw()
    fig.set_layout_engine(None)
    capture=cv2.VideoCapture(source_video)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    command=[imageio_ffmpeg.get_ffmpeg_exe(),"-hide_banner","-loglevel","error","-y","-f","rawvideo",
             "-vcodec","rawvideo","-pix_fmt","rgb24","-s","1280x720","-r",str(fps),
             "-i","-","-an","-c:v","libx264","-preset","fast","-crf","20",
             "-pix_fmt","yuv420p","-movflags","+faststart",str(args.out)]
    encoder=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        for i, chain in enumerate(chains):
            ok,image=capture.read()
            if not ok:
                raise RuntimeError(f"Source video ended at {i}")
            image_artist.set_data(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
            camera.set_title(f"{i/fps:.2f} s | frame {i} | bend {motors[i,4]:.3f} / rot {motors[i,5]:.3f}",fontsize=10)
            for x,y,lines,markers,previous in views:
                for line, points in zip(lines,chain):
                    line.set_data(points[:,x],points[:,y])
                markers.set_offsets(tips[i][:,[x,y]]*100)
                previous.set_offsets(old_tips[i][:,[x,y]]*100)
            fig.canvas.draw()
            rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
            encoder.stdin.write(rgb.tobytes())
            if i in (0,len(chains)//2,len(chains)-1):
                cv2.imwrite(str(args.out.with_name(args.out.stem+f"_frame{i:04d}.jpg")),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
            if i%100==0:
                print(f"Rendered {i}/{len(chains)}",flush=True)
    finally:
        capture.release()
        encoder.stdin.close()
        error_text=encoder.stderr.read().decode()
        code=encoder.wait()
        plt.close(fig)
    if code:
        raise RuntimeError(error_text)
    print(json.dumps({"frames":len(tips),"fps":fps,"duration_s":len(tips)/fps,
                      "saved_tips_vs_fk_max_mm":maximum_mm,"output":str(args.out)}),flush=True)


if __name__ == "__main__":
    main()
