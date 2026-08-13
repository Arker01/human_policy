#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HDT_DIR = _REPO_ROOT / "hdt"
if str(_HDT_DIR) not in sys.path:
    sys.path.insert(0, str(_HDT_DIR))

import hdt.constants as C


def _transform_points(points: np.ndarray, transform_mat: np.ndarray) -> np.ndarray:
    points_h = np.concatenate(
        [points, np.ones((points.shape[0], 1), dtype=points.dtype)],
        axis=1,
    )
    transformed = np.dot(transform_mat, points_h.T).T
    return transformed[:, :3]

def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n

def _rot6d_to_mat(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    R = np.stack([b1, b2, b3], axis=1).astype(np.float32)
    return R


def _action_to_eval_joints(action_128: np.ndarray) -> dict[str, np.ndarray]:
    head = action_128[C.OUTPUT_HEAD_EEF]
    lw = action_128[C.OUTPUT_LEFT_EEF]
    rw = action_128[C.OUTPUT_RIGHT_EEF]
    lk_local = action_128[C.OUTPUT_LEFT_KEYPOINTS].reshape(6, 3).astype(np.float32)
    rk_local = action_128[C.OUTPUT_RIGHT_KEYPOINTS].reshape(6, 3).astype(np.float32)
    
    waist = action_128[89:98]
    waist_mat = np.eye(4, dtype=np.float32)
    waist_mat[:3, 3] = waist[:3]
    waist_mat[:3, :3] = _rot6d_to_mat(waist[3:9])

    head_mat = np.eye(4, dtype=np.float32)
    head_mat[:3, 3] = head[:3]
    head_mat[:3, :3] = _rot6d_to_mat(head[3:9])

    lw_mat = np.eye(4, dtype=np.float32)
    lw_mat[:3, 3] = lw[:3]
    lw_mat[:3, :3] = _rot6d_to_mat(lw[3:9])

    rw_mat = np.eye(4, dtype=np.float32)
    rw_mat[:3, 3] = rw[:3]
    rw_mat[:3, :3] = _rot6d_to_mat(rw[3:9])

    lk_world = _transform_points(lk_local, lw_mat)
    rk_world = _transform_points(rk_local, rw_mat)

    return {
        "head": head_mat[:3, 3].astype(np.float32),
        "lw": lw_mat[:3, 3].astype(np.float32),
        "rw": rw_mat[:3, 3].astype(np.float32),
        "lk_world": lk_world.astype(np.float32),
        "rk_world": rk_world.astype(np.float32),
        "waist": waist_mat[:3, 3].astype(np.float32),
    }


_EARLY_JUMP_CHECK_SLICES = (
    C.OUTPUT_HEAD_EEF[0:3],
    C.OUTPUT_RIGHT_EEF[0:3],
    C.OUTPUT_NECK[0:3],
    C.OUTPUT_LEFT_EEF[0:3],
    C.OUTPUT_WAIST[0:3],
)


def _detect_valid_start_from_actions(actions, *, check_frames=20, jump_threshold_m=0.3, settle_frames=0):
    if actions.shape[0] <= 1 or check_frames <= 1 or jump_threshold_m <= 0:
        return 0

    n = min(int(check_frames), int(actions.shape[0]))
    last_bad_next_frame = -1
    for sl in _EARLY_JUMP_CHECK_SLICES:
        pos = actions[:n, sl]
        if pos.shape[1] != 3:
            continue
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        bad = np.where(step > float(jump_threshold_m))[0]
        if len(bad) > 0:
            last_bad_next_frame = max(last_bad_next_frame, int(bad[-1]) + 1)

    if last_bad_next_frame < 0:
        return 0
    valid_start = last_bad_next_frame + int(settle_frames)
    return min(valid_start, max(0, int(actions.shape[0]) - 1))


def eval_arrays_mpjpe(gt_actions: np.ndarray, pred_actions: np.ndarray, *, dirty_start_check_frames=20, dirty_start_jump_threshold_m=0.3, dirty_start_settle_frames=0) -> dict[str, float | int]:
    T = min(int(gt_actions.shape[0]), int(pred_actions.shape[0]))
    if T <= 0:
        raise ValueError("No overlapping timesteps between gt and prediction.")

    valid_start = _detect_valid_start_from_actions(
        gt_actions,
        check_frames=dirty_start_check_frames,
        jump_threshold_m=dirty_start_jump_threshold_m,
        settle_frames=dirty_start_settle_frames,
    )

    all_joint_err = []
    hand_err = []
    wrist_err = []
    head_err = []
    waist_err = []

    for t in range(valid_start, T):
        gt_j = _action_to_eval_joints(gt_actions[t])
        pr_j = _action_to_eval_joints(pred_actions[t])

        gt_hand = np.concatenate([gt_j["lk_world"], gt_j["rk_world"]], axis=0)  # (12, 3)
        pr_hand = np.concatenate([pr_j["lk_world"], pr_j["rk_world"]], axis=0)
        gt_wrist = np.stack([gt_j["lw"], gt_j["rw"]], axis=0)  # (2, 3)
        pr_wrist = np.stack([pr_j["lw"], pr_j["rw"]], axis=0)
        gt_head = gt_j["head"][None, :]
        pr_head = pr_j["head"][None, :]
        gt_waist = gt_j["waist"][None, :]
        pr_waist = pr_j["waist"][None, :]

        gt_all = np.concatenate([gt_head, gt_wrist, gt_hand, gt_waist], axis=0)  # (16, 3)
        pr_all = np.concatenate([pr_head, pr_wrist, pr_hand, pr_waist], axis=0)

        all_joint_err.append(np.linalg.norm(pr_all - gt_all, axis=1))
        hand_err.append(np.linalg.norm(pr_hand - gt_hand, axis=1))
        wrist_err.append(np.linalg.norm(pr_wrist - gt_wrist, axis=1))
        head_err.append(np.linalg.norm(pr_head - gt_head, axis=1))
        waist_err.append(np.linalg.norm(pr_waist - gt_waist, axis=1))

    all_joint_err = np.concatenate(all_joint_err, axis=0)
    hand_err = np.concatenate(hand_err, axis=0)
    wrist_err = np.concatenate(wrist_err, axis=0)
    head_err = np.concatenate(head_err, axis=0)
    waist_err = np.concatenate(waist_err, axis=0)

    return {
        "timesteps_compared": int(T - valid_start),
        "timesteps_skipped": int(valid_start),
        "joints_per_timestep": 16,
        "mpjpe_all_m": float(np.mean(all_joint_err)),
        "mpjpe_hand_m": float(np.mean(hand_err)),
        "mpjpe_wrist_m": float(np.mean(wrist_err)),
        "mpjpe_head_m": float(np.mean(head_err)),
        "mpjpe_waist_m": float(np.mean(waist_err)),
        "mpjpe_all_mm": float(np.mean(all_joint_err) * 1000.0),
        "mpjpe_hand_mm": float(np.mean(hand_err) * 1000.0),
        "mpjpe_wrist_mm": float(np.mean(wrist_err) * 1000.0),
        "mpjpe_head_mm": float(np.mean(head_err) * 1000.0),
        "mpjpe_waist_mm": float(np.mean(waist_err) * 1000.0),
    }

def eval_pair_mpjpe(gt_hdf5_path: Path, pred_hdf5_path: Path, *, dirty_start_check_frames=20, dirty_start_jump_threshold_m=0.3, dirty_start_settle_frames=0) -> dict[str, float | int]:
    with h5py.File(gt_hdf5_path, "r") as f_gt, h5py.File(pred_hdf5_path, "r") as f_pr:
        gt_actions = f_gt["action"][()]
        pr_actions = f_pr["action"][()]
    return eval_arrays_mpjpe(gt_actions, pr_actions, dirty_start_check_frames=dirty_start_check_frames, dirty_start_jump_threshold_m=dirty_start_jump_threshold_m, dirty_start_settle_frames=dirty_start_settle_frames)


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")

def _read_image_frame(ds: h5py.Dataset, t: int) -> np.ndarray:
    arr = ds[t]
    if isinstance(arr, np.ndarray) and arr.dtype == object:
        arr = arr[()]
    if isinstance(arr, np.ndarray) and arr.ndim == 1 and arr.dtype == np.uint8:
        data = arr.tobytes()
        try:
            import cv2

            img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("cv2.imdecode returned None")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        except Exception:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(data)).convert("RGB")
            return np.asarray(img)

    if isinstance(arr, np.ndarray) and arr.ndim == 3:
        if arr.shape[0] == 3:
            return np.transpose(arr, (1, 2, 0))
        return arr
    raise ValueError(f"Unsupported image dataset format: shape={getattr(arr, 'shape', None)} dtype={getattr(arr, 'dtype', None)}")

def _load_yaml(path: Path) -> dict:
    import yaml

    with open(path, "r") as f:
        return yaml.safe_load(f)

def _to_numpy(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _load_norm_stats(stats_path: Path, embodiment: str | None) -> dict:
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    if isinstance(stats, dict) and "qpos_mean" in stats:
        return {k: _to_numpy(v) for k, v in stats.items()}
    if isinstance(stats, dict) and embodiment is not None and embodiment in stats:
        return {k: _to_numpy(v) for k, v in stats[embodiment].items()}
    if isinstance(stats, dict) and embodiment is not None:
        emb = embodiment.encode() if isinstance(next(iter(stats.keys())), bytes) else embodiment
        if emb in stats:
            return {k: _to_numpy(v) for k, v in stats[emb].items()}
    if isinstance(stats, dict):
        first = next(iter(stats.values()))
        if isinstance(first, dict):
            return {k: _to_numpy(v) for k, v in first.items()}
    raise ValueError(f"Unsupported stats format: {type(stats)}")

def _make_act_policy(model_yaml: Path, ckpt_path: Path, *, device: str) -> tuple[object, dict]:
    import torch
    from hdt.policy import ACTPolicy

    cfg = _load_yaml(model_yaml)
    common = cfg.get("common", {})
    model = cfg.get("model", {})

    cameras = common.get("camera_names", [])
    args_override = dict(
        lr=1e-4,
        weight_decay=1e-4,
        lr_backbone=float(model.get("lr_backbone", 0.0)),
        backbone=str(model.get("backbone", "resnet18")),
        camera_names=list(cameras),
        enc_layers=int(model.get("enc_layers", 4)),
        dec_layers=int(model.get("dec_layers", 7)),
        nheads=int(model.get("nheads", 8)),
        hidden_dim=int(model.get("hidden_dim", 512)),
        dim_feedforward=int(model.get("dim_feedforward", 3200)),
        num_queries=int(common.get("action_chunk_size", 100)),
        kl_weight=float(model.get("kl_weight", 10.0)),
        state_dim=int(common.get("state_dim", 128)),
        action_dim=int(common.get("action_dim", 128)),
        image_feature_strategy=str(model.get("image_feature_strategy", "ACT_linear")),
        use_language_conditioning=bool(model.get("use_language_conditioning", False)),
    )

    policy = ACTPolicy(args_override)
    policy.eval()

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    policy.load_state_dict(state_dict, strict=False)
    policy.to(device)
    return policy, {"camera_names": list(cameras), "use_language_conditioning": bool(model.get("use_language_conditioning", False))}

def _check_camera_in_file(f: h5py.File, camera_names: list[str]) -> bool:
    """Check if at least one camera is available in the HDF5 file."""
    for cam in camera_names:
        key = f"observation.image.{cam}"
        if key in f:
            return True
    if 'observation.image.left' in f or 'observation.image.right' in f:
        return True
    return False

def _predict_actions_for_episode(
    policy,
    episode_path: Path,
    *,
    stats: dict,
    camera_names: list[str],
    device: str,
    max_steps: int | None,
) -> np.ndarray:
    import torch

    with h5py.File(episode_path, "r") as f:
        if not _check_camera_in_file(f, camera_names):
            raise ValueError(f"No camera data available in {episode_path}")
        
        states = f["observation.state"][()]
        T = int(states.shape[0])
        if max_steps is not None:
            T = min(T, int(max_steps))

        qpos_mean = stats["qpos_mean"].reshape(1, -1)
        qpos_std = stats["qpos_std"].reshape(1, -1)
        act_mean = stats["action_mean"].reshape(1, -1)
        act_std = stats["action_std"].reshape(1, -1)

        pred = np.zeros((T, states.shape[1]), dtype=np.float32)
        for t in range(T):
            imgs = []
            for cam in camera_names:
                key = f"observation.image.{cam}"
                if key not in f:
                    # Fallback to left or right camera if top is not available
                    if cam == 'top':
                        if 'observation.image.left' in f:
                            key = 'observation.image.left'
                        elif 'observation.image.right' in f:
                            key = 'observation.image.right'
                        else:
                            raise KeyError(f"Missing {cam} and no fallback cameras available in {episode_path}")
                    else:
                        raise KeyError(f"Missing {key} in {episode_path}")
                img = _read_image_frame(f[key], t)
                if img.shape[-1] == 4:
                    img = img[:, :, :3]
                if img.shape[0] != 240 or img.shape[1] != 320:
                    import cv2
                    img = cv2.resize(img, (320, 240))
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)  # (num_cam, H, W, 3)
            imgs_t = torch.from_numpy(imgs).to(device=device, dtype=torch.float32) / 255.0
            imgs_t = imgs_t.permute(0, 3, 1, 2).unsqueeze(0)  # (1, num_cam, 3, H, W)

            qpos = states[t : t + 1].astype(np.float32)
            qpos_n = (qpos - qpos_mean) / (qpos_std + 1e-8)
            qpos_t = torch.from_numpy(qpos_n).to(device=device, dtype=torch.float32)

            conditioning_dict = {}
            a_hat = policy(imgs_t, qpos_t, conditioning_dict=conditioning_dict)
            a0 = a_hat[0, 0].detach().float().cpu().numpy().astype(np.float32)
            a0 = a0 * act_std.reshape(-1) + act_mean.reshape(-1)
            pred[t] = a0

    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="批量计算预测 HDF5 相对于 GT HDF5 的 MPJPE，并输出平均值"
    )
    parser.add_argument("--pred-dir", type=str, default=None, help="预测 HDF5 目录（如果使用 --policy-ckpt 则可不填）")
    parser.add_argument("--gt-dir", type=str, required=True, help="GT HDF5 目录")
    parser.add_argument("--glob", type=str, default="*.hdf5", help="匹配文件的 glob，默认 *.hdf5")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="严格模式：若某个预测文件找不到同名 GT 文件则报错退出",
    )
    parser.add_argument("--policy-ckpt", type=str, default=None, help="直接加载 policy ckpt 并在 GT episode 上推理")
    parser.add_argument("--policy-config-yaml", type=str, default="/home/aigc/human_policy/hdt/configs/models/act_resnet.yaml", help="policy 模型配置 YAML（例如 hdt/configs/models/act_resnet.yaml）")
    parser.add_argument("--norm-stats", type=str, default=None, help="归一化统计量 pkl（例如 dataset_stats.pkl）")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备（ACT 默认需要 cuda）")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 episode 最多评估多少步（默认全部）")
    parser.add_argument("--dirty-start-check-frames", type=int, default=20, help="检测脏帧的初始帧数（默认20）")
    parser.add_argument("--dirty-start-jump-threshold-m", type=float, default=0.3, help="脏帧位置跳变阈值（米，默认0.3）")
    parser.add_argument("--dirty-start-settle-frames", type=int, default=0, help="脏帧后额外跳过的帧数（默认0）")
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="可选：保存批量评估结果 JSON 路径",
    )
    args = parser.parse_args()

    gt_dir = Path(args.gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt-dir not found: {gt_dir}")

    use_policy = args.policy_ckpt is not None
    if use_policy:
        if args.policy_config_yaml is None:
            raise ValueError("--policy-config-yaml is required when using --policy-ckpt")
        if args.norm_stats is None:
            raise ValueError("--norm-stats is required when using --policy-ckpt")
        gt_files = sorted(gt_dir.glob(args.glob))
        if not gt_files:
            raise RuntimeError(f"No files matched in gt-dir: {gt_dir} with glob={args.glob}")
        policy, pol_info = _make_act_policy(Path(args.policy_config_yaml), Path(args.policy_ckpt), device=args.device)
        camera_names = pol_info["camera_names"]
        if not camera_names:
            raise ValueError("camera_names is empty in policy config")
    else:
        if args.pred_dir is None:
            raise ValueError("--pred-dir is required unless using --policy-ckpt")
        pred_dir = Path(args.pred_dir)
        if not pred_dir.exists():
            raise FileNotFoundError(f"pred-dir not found: {pred_dir}")
        pred_files = sorted(pred_dir.glob(args.glob))
        if not pred_files:
            raise RuntimeError(f"No files matched in pred-dir: {pred_dir} with glob={args.glob}")

    per_file = []
    skipped = []

    if use_policy:
        stats_path = Path(args.norm_stats)
        for gt_path in gt_files:
            with h5py.File(gt_path, "r") as f:
                emb = f.attrs.get("embodiment", None)
                if isinstance(emb, bytes):
                    emb = emb.decode()
                emb = str(emb) if emb is not None else None
            stats = _load_norm_stats(stats_path, emb)
            try:
                pred_actions = _predict_actions_for_episode(
                    policy,
                    gt_path,
                    stats=stats,
                    camera_names=camera_names,
                    device=args.device,
                    max_steps=args.max_steps,
                )
            except ValueError as e:
                if "No camera data available" in str(e):
                    print(f"Skipping {gt_path.name} - {e}")
                    skipped.append(gt_path.name)
                    continue
                raise
            with h5py.File(gt_path, "r") as f_gt:
                gt_actions = f_gt["action"][()]
            metrics = eval_arrays_mpjpe(
                gt_actions, pred_actions,
                dirty_start_check_frames=args.dirty_start_check_frames,
                dirty_start_jump_threshold_m=args.dirty_start_jump_threshold_m,
                dirty_start_settle_frames=args.dirty_start_settle_frames,
            )
            metrics["file"] = gt_path.name
            per_file.append(metrics)
    else:
        for pred_path in pred_files:
            gt_path = gt_dir / pred_path.name
            if not gt_path.exists():
                if args.strict:
                    raise FileNotFoundError(f"Missing GT file for {pred_path.name}: {gt_path}")
                skipped.append(pred_path.name)
                continue
            metrics = eval_pair_mpjpe(
                gt_path, pred_path,
                dirty_start_check_frames=args.dirty_start_check_frames,
                dirty_start_jump_threshold_m=args.dirty_start_jump_threshold_m,
                dirty_start_settle_frames=args.dirty_start_settle_frames,
            )
            metrics["file"] = pred_path.name
            per_file.append(metrics)

    if not per_file:
        raise RuntimeError("No valid prediction/gt pairs were evaluated.")

    summary = {
        "num_pairs_evaluated": len(per_file),
        "num_pairs_skipped": len(skipped),
        "skipped_files": skipped,
        "avg_timesteps_compared": _safe_mean([m["timesteps_compared"] for m in per_file]),  # type: ignore[index]
        "avg_timesteps_skipped": _safe_mean([m["timesteps_skipped"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_all_m": _safe_mean([m["mpjpe_all_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_hand_m": _safe_mean([m["mpjpe_hand_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_wrist_m": _safe_mean([m["mpjpe_wrist_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_head_m": _safe_mean([m["mpjpe_head_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_waist_m": _safe_mean([m["mpjpe_waist_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_all_mm": _safe_mean([m["mpjpe_all_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_hand_mm": _safe_mean([m["mpjpe_hand_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_wrist_mm": _safe_mean([m["mpjpe_wrist_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_head_mm": _safe_mean([m["mpjpe_head_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_waist_mm": _safe_mean([m["mpjpe_waist_mm"] for m in per_file]),  # type: ignore[index]
    }

    print("===== Batch MPJPE Summary =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"summary": summary, "per_file": per_file}
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved metrics to: {out_path}")


if __name__ == "__main__":
    main()
