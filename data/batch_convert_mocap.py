#!/usr/bin/env python3
import os
import sys
import json
import re
import glob
from pathlib import Path
from tqdm import tqdm

# Add the project root to sys.path to import hdt
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.mocap_hdf5_to_processed import convert_mocap_to_processed
from data.mocap_hdf5_to_processed import (
    convert_wholebody_json_to_processed as convert_wholebody_json_to_processed_aligned,
)
import h5py
import numpy as np
import cv2
import hdt.constants as C

def _sanitize_task_name(name: str) -> str:
    name = name.strip()
    if not name:
        return "dataset"
    return re.sub(r"[^a-zA-Z0-9_\\-]+", "_", name)

def _collect_wholebody_pairs(input_dir: Path) -> list[tuple[int, Path, Path | None]]:
    pairs: list[tuple[int, Path, Path | None]] = []
    pattern = str(input_dir / "wholebody-*.json")
    for json_path_str in sorted(glob.glob(pattern)):
        json_path = Path(json_path_str)
        m = re.search(r"wholebody-(\d+)\.json$", json_path.name)
        if not m:
            continue
        idx = int(m.group(1))
        candidates = [
            input_dir / f"wb-{idx}.MP4",
            input_dir / f"wb-{idx}.mp4",
            input_dir / f"wb-{idx}.mov",
            input_dir / f"wb-{idx}.MOV",
        ]
        video_path = next((p for p in candidates if p.exists()), None)
        pairs.append((idx, json_path, video_path))
    pairs.sort(key=lambda x: x[0])
    return pairs

def _save_dataset_config(task_name: str, output_base_dir: Path) -> Path:
    output_base_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_base_dir / f"{task_name}.json"
    dataset_cfg = {"train": [task_name], "val": [task_name]}
    with open(json_path, "w") as f:
        json.dump(dataset_cfg, f, indent=4)
    return json_path

def batch_process_hdf5_range(
    input_dir: Path,
    output_task_dir: Path,
    output_base_dir: Path,
    task_name: str,
    start_idx: int,
    end_idx: int,
    *,
    zero_head_translation: bool = True,
    pos_scale: float = 1.0,
    image_hw: tuple[int, int] = (240, 320),
    cam_to_sim: str | None = None,
    head_as_origin: bool = False,
    shift_action: bool = True,
    state_from_prev_action: bool = False,
    rel_frame: str = "none",
    still_head: bool = False,
) -> None:
    """批量处理HDF5格式的动捕数据，只生成top摄像头"""
    processed_count = 0
    for i in tqdm(range(start_idx, end_idx + 1)):
        hdf5_path = input_dir / f"{i}.hdf5"
        video_path = input_dir / f"{i}.mp4"
        if not hdf5_path.exists():
            continue
        output_path = output_task_dir / f"episode_{i}.hdf5"
        if output_path.exists():
            continue
        try:
            convert_mocap_to_processed(
                src_path=hdf5_path,
                dst_path=output_path,
                video_path=str(video_path) if video_path.exists() else None,
                zero_head_translation=zero_head_translation,
                pos_scale=pos_scale,
                image_hw=image_hw,
                cam_to_sim=cam_to_sim,
                head_as_origin=head_as_origin,
                shift_action=shift_action,
                state_from_prev_action=state_from_prev_action,
                write_left_right=False,  # 强制只写top摄像头
                compress_images=False,
                sbs=False,
                rel_frame=rel_frame,
                still_head=still_head,
            )
            processed_count += 1
        except Exception as e:
            print(f"Error processing index {i}: {e}")

    json_path = _save_dataset_config(task_name, output_base_dir)
    print(f"Saved dataset config to {json_path}")
    print(f"Batch processing complete. Total processed: {processed_count}")

def batch_process_wholebody_json(
    input_dir: Path,
    output_task_dir: Path,
    output_base_dir: Path,
    task_name: str,
    start_idx: int,
    end_idx: int,
    *,
    use_aligned: bool = False,
    zero_head_translation: bool = True,
    pos_scale: float = 1.0,
    image_resolution_hw: tuple[int, int] = (240, 320),
    cam_to_sim: str | None = None,
    head_as_origin: bool = False,
    shift_action: bool = True,
    state_from_prev_action: bool = False,
    rel_frame: str = "none",
    still_head: bool = False,
    kpts_frame: str = "wrist",
) -> None:
    """批量处理wholebody JSON格式的动捕数据，只生成top摄像头"""
    processed_count = 0

    pairs = _collect_wholebody_pairs(input_dir)
    if not pairs:
        raise RuntimeError(f"No wholebody-*.json found in {input_dir}")

    for idx, json_path, video_path in tqdm(pairs):
        if int(idx) < int(start_idx) or int(idx) > int(end_idx):
            continue
        output_path = output_task_dir / f"episode_{idx}.hdf5"
        if output_path.exists():
            continue

        try:
            if use_aligned:
                convert_wholebody_json_to_processed_aligned(
                    src_json_path=json_path,
                    dst_path=output_path,
                    copy_root_attrs=True,
                    zero_head_translation=zero_head_translation,
                    pos_scale=float(pos_scale),
                    video_path=str(video_path) if video_path is not None and video_path.exists() else None,
                    image_hw=(int(image_resolution_hw[0]), int(image_resolution_hw[1])),
                    cam_to_sim=cam_to_sim,
                    head_as_origin=head_as_origin,
                    shift_action=shift_action,
                    state_from_prev_action=state_from_prev_action,
                    write_left_right=False,  # 强制只写top摄像头
                    compress_images=False,
                    sbs=False,
                    rel_frame=rel_frame,
                    still_head=still_head,
                    kpts_frame=str(kpts_frame),
                )
            else:
                # 简化版的JSON处理，只写top摄像头
                with open(json_path, "r") as f:
                    frames = json.load(f)

                T = len(frames)
                actions = np.zeros((T, C.ACTION_STATE_VEC_SIZE), dtype=np.float32)
                images = np.zeros((T, image_resolution_hw[0], image_resolution_hw[1], 3), dtype=np.uint8)

                if video_path is not None and video_path.exists():
                    cap = cv2.VideoCapture(str(video_path))
                    video_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    for i in range(min(T, video_len)):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame = cv2.resize(frame, (image_resolution_hw[1], image_resolution_hw[0]))
                        images[i] = frame
                    cap.release()

                for i, fr in enumerate(frames):
                    # 简化的姿态处理
                    head_R, head_t = _pose_from_flat_16(fr["head"])
                    lw_R, lw_t = _pose_from_flat_16(fr["leftWrist"])
                    rw_R, rw_t = _pose_from_flat_16(fr["rightWrist"])

                    head_rot6d = _matrix_to_rotation_6d(head_R)
                    lw_rot6d = _matrix_to_rotation_6d(lw_R)
                    rw_rot6d = _matrix_to_rotation_6d(rw_R)

                    head_pos = np.zeros(3, dtype=np.float32) if zero_head_translation else head_t.astype(np.float32)
                    head_action = np.concatenate([head_pos, head_rot6d])
                    left_wrist_action = np.concatenate([lw_t.astype(np.float32), lw_rot6d])
                    right_wrist_action = np.concatenate([rw_t.astype(np.float32), rw_rot6d])

                    left_k = _kpts_from_skeleton_joints(fr["leftSkeleton"]["joints"]).astype(np.float32)
                    right_k = _kpts_from_skeleton_joints(fr["rightSkeleton"]["joints"]).astype(np.float32)

                    action = np.zeros(C.ACTION_STATE_VEC_SIZE, dtype=np.float32)
                    action[C.OUTPUT_HEAD_EEF] = head_action
                    action[C.OUTPUT_LEFT_EEF] = left_wrist_action
                    action[C.OUTPUT_RIGHT_EEF] = right_wrist_action
                    action[C.OUTPUT_LEFT_KEYPOINTS] = left_k.reshape(-1)
                    action[C.OUTPUT_RIGHT_KEYPOINTS] = right_k.reshape(-1)
                    actions[i] = action

                states = actions.copy()
                actions_target = actions.copy()
                if T > 1:
                    actions_target[:-1] = actions[1:]

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(output_path, "w") as f_out:
                    f_out.create_dataset("action", data=actions_target, compression="gzip", compression_opts=4)
                    f_out.create_dataset("observation.state", data=states, compression="gzip", compression_opts=4)
                    f_out.create_dataset("observation.image.top", data=images, compression="gzip", compression_opts=4)
                    f_out.attrs["sim"] = np.bool_(False)
                    f_out.attrs["embodiment"] = "human_mocap_annotated"
                    f_out.attrs["description"] = ""

            processed_count += 1
        except Exception as e:
            print(f"Error processing {json_path.name}: {e}")

    json_path = _save_dataset_config(task_name, output_base_dir)
    print(f"Saved dataset config to {json_path}")
    print(f"Batch processing complete. Total processed: {processed_count}")

# 辅助函数
def _matrix_to_rotation_6d(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[:, 0], R[:, 1]], axis=0).astype(np.float32)

def _pose_from_flat_16(flat_16: list[float]) -> tuple[np.ndarray, np.ndarray]:
    M = np.asarray(flat_16, dtype=np.float32).reshape(4, 4, order="F")
    R = M[:3, :3]
    t = M[3, :3]
    return R, t

def _kpts_from_skeleton_joints(joints_flat: list[float]) -> np.ndarray:
    joints = np.asarray(joints_flat, dtype=np.float32).reshape(25, 4, 4)
    return joints[C.RETARGETTING_INDICES, :3, 3]

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="批量转换动捕数据为训练格式（只生成top摄像头）")
    p.add_argument("--input-dir", type=str, required=True, help="输入目录")
    p.add_argument("--output-base-dir", type=str, default=None, help="输出基础目录")
    p.add_argument("--task-name", type=str, default=None, help="任务名称")
    p.add_argument("--format", type=str, default="auto", choices=["auto", "hdf5", "wholebody_json"], help="输入格式")
    p.add_argument("--start", type=int, default=0, help="起始索引")
    p.add_argument("--end", type=int, default=1000, help="结束索引")
    p.add_argument("--image-h", type=int, default=240, help="图像高度")
    p.add_argument("--image-w", type=int, default=320, help="图像宽度")
    p.add_argument("--zero-head-translation", action="store_true", help="头部平移置零")
    p.add_argument("--no-zero-head-translation", action="store_true", help="保留头部平移")
    p.add_argument("--pos-scale", type=float, default=1.0, help="位置缩放因子")
    p.add_argument("--cam-to-sim", type=str, default=None, help="相机到仿真的变换")
    p.add_argument("--rel-frame", type=str, default="none", choices=["none", "head0", "head_each"], help="相对参考系")
    p.add_argument("--still-head", action="store_true", help="固定头部旋转")
    p.add_argument("--head-as-origin", action="store_true", help="以头部为原点")
    p.add_argument("--no-head-as-origin", action="store_true", help="不以头部为原点")
    p.add_argument("--kpts-frame", type=str, default="wrist", choices=["wrist", "head", "world"], help="关键点参考系")
    p.add_argument("--no-shift-action", action="store_true", help="不偏移action")
    p.add_argument("--state-from-prev-action", action="store_true", help="state来自前一action")
    p.add_argument("--wholebody-aligned", action="store_true", help="使用对齐的wholebody处理")
    args = p.parse_args()

    input_dir = Path(args.input_dir.strip())
    task_name = _sanitize_task_name(args.task_name if args.task_name is not None else input_dir.name)
    output_base_dir = (
        Path(args.output_base_dir.strip())
        if args.output_base_dir is not None
        else (input_dir / "processed_hdt")
    )
    output_task_dir = output_base_dir / task_name
    output_task_dir.mkdir(parents=True, exist_ok=True)

    fmt = args.format
    if fmt == "auto":
        if list(input_dir.glob("wholebody-*.json")):
            fmt = "wholebody_json"
        elif list(input_dir.glob("*.hdf5")):
            fmt = "hdf5"
        else:
            raise RuntimeError(f"Unsupported input format in {input_dir}")

    if fmt == "hdf5":
        zero_head_translation = bool((args.zero_head_translation) and not args.no_zero_head_translation)
        head_as_origin = bool((args.head_as_origin) and not args.no_head_as_origin)
        shift_action = not bool(args.no_shift_action)
        state_from_prev_action = bool(args.state_from_prev_action)

        batch_process_hdf5_range(
            input_dir=input_dir,
            output_task_dir=output_task_dir,
            output_base_dir=output_base_dir,
            task_name=task_name,
            start_idx=args.start,
            end_idx=args.end,
            zero_head_translation=zero_head_translation,
            pos_scale=float(args.pos_scale),
            image_hw=(int(args.image_h), int(args.image_w)),
            cam_to_sim=args.cam_to_sim,
            head_as_origin=head_as_origin,
            shift_action=shift_action,
            state_from_prev_action=state_from_prev_action,
            rel_frame=args.rel_frame,
            still_head=bool(args.still_head),
        )
    else:
        use_aligned = bool(args.wholebody_aligned)
        zero_head_translation = bool((args.zero_head_translation) and not args.no_zero_head_translation)
        head_as_origin = bool((args.head_as_origin) and not args.no_head_as_origin)
        shift_action = not bool(args.no_shift_action) if use_aligned else True
        state_from_prev_action = bool(args.state_from_prev_action) if use_aligned else False
        kpts_frame = (args.kpts_frame if args.kpts_frame is not None else "wrist")

        batch_process_wholebody_json(
            input_dir=input_dir,
            output_task_dir=output_task_dir,
            output_base_dir=output_base_dir,
            task_name=task_name,
            start_idx=args.start,
            end_idx=args.end,
            use_aligned=use_aligned,
            zero_head_translation=zero_head_translation,
            pos_scale=float(args.pos_scale),
            image_resolution_hw=(args.image_h, args.image_w),
            cam_to_sim=args.cam_to_sim,
            head_as_origin=head_as_origin,
            shift_action=shift_action,
            state_from_prev_action=state_from_prev_action,
            rel_frame=args.rel_frame,
            still_head=bool(args.still_head),
            kpts_frame=kpts_frame,
        )
