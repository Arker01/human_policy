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
    }


def eval_arrays_mpjpe(gt_actions: np.ndarray, pred_actions: np.ndarray) -> dict[str, float | int]:
    T = min(int(gt_actions.shape[0]), int(pred_actions.shape[0]))
    if T <= 0:
        raise ValueError("No overlapping timesteps between gt and prediction.")

    all_joint_err = []
    hand_err = []
    wrist_err = []
    head_err = []

    for t in range(T):
        gt_j = _action_to_eval_joints(gt_actions[t])
        pr_j = _action_to_eval_joints(pred_actions[t])

        gt_hand = np.concatenate([gt_j["lk_world"], gt_j["rk_world"]], axis=0)  # (12, 3)
        pr_hand = np.concatenate([pr_j["lk_world"], pr_j["rk_world"]], axis=0)
        gt_wrist = np.stack([gt_j["lw"], gt_j["rw"]], axis=0)  # (2, 3)
        pr_wrist = np.stack([pr_j["lw"], pr_j["rw"]], axis=0)
        gt_head = gt_j["head"][None, :]
        pr_head = pr_j["head"][None, :]

        gt_all = np.concatenate([gt_head, gt_wrist, gt_hand], axis=0)  # (15, 3)
        pr_all = np.concatenate([pr_head, pr_wrist, pr_hand], axis=0)

        all_joint_err.append(np.linalg.norm(pr_all - gt_all, axis=1))
        hand_err.append(np.linalg.norm(pr_hand - gt_hand, axis=1))
        wrist_err.append(np.linalg.norm(pr_wrist - gt_wrist, axis=1))
        head_err.append(np.linalg.norm(pr_head - gt_head, axis=1))

    all_joint_err = np.concatenate(all_joint_err, axis=0)
    hand_err = np.concatenate(hand_err, axis=0)
    wrist_err = np.concatenate(wrist_err, axis=0)
    head_err = np.concatenate(head_err, axis=0)

    return {
        "timesteps_compared": int(T),
        "joints_per_timestep": 15,
        "mpjpe_all_m": float(np.mean(all_joint_err)),
        "mpjpe_hand_m": float(np.mean(hand_err)),
        "mpjpe_wrist_m": float(np.mean(wrist_err)),
        "mpjpe_head_m": float(np.mean(head_err)),
        "mpjpe_all_mm": float(np.mean(all_joint_err) * 1000.0),
        "mpjpe_hand_mm": float(np.mean(hand_err) * 1000.0),
        "mpjpe_wrist_mm": float(np.mean(wrist_err) * 1000.0),
        "mpjpe_head_mm": float(np.mean(head_err) * 1000.0),
    }

def eval_pair_mpjpe(gt_hdf5_path: Path, pred_hdf5_path: Path) -> dict[str, float | int]:
    with h5py.File(gt_hdf5_path, "r") as f_gt, h5py.File(pred_hdf5_path, "r") as f_pr:
        gt_actions = f_gt["action"][()]
        pr_actions = f_pr["action"][()]
    return eval_arrays_mpjpe(gt_actions, pr_actions)


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

def _make_policy(model_yaml: Path, ckpt_path: Path, *, device: str) -> tuple[object, dict]:
    import torch

    cfg = _load_yaml(model_yaml)
    common = cfg.get("common", {})
    model = cfg.get("model", {})
    policy_class = common.get("policy_class", "ACT")

    cameras = common.get("camera_names", [])
    chunk_size = int(common.get("action_chunk_size", model.get("chunk_size", 100)))

    if policy_class == "ACT":
        from hdt.policy import ACTPolicy

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
            num_queries=chunk_size,
            chunk_size=chunk_size,
            kl_weight=float(model.get("kl_weight", 10.0)),
            state_dim=int(common.get("state_dim", 128)),
            action_dim=int(common.get("action_dim", 128)),
            image_feature_strategy=str(model.get("image_feature_strategy", "ACT_linear")),
            use_language_conditioning=bool(model.get("use_language_conditioning", False)),
        )
        policy = ACTPolicy(args_override)
    elif policy_class == "ACT_FM":
        from hdt.modeling.modeling_act_flow import ACTFlowPolicy

        args_override = dict(
            lr=1e-4,
            lr_backbone=float(model.get("lr_backbone", 0.0)),
            backbone=str(model.get("backbone", "resnet18")),
            camera_names=list(cameras),
            enc_layers=int(model.get("enc_layers", 4)),
            fm_layers=int(model.get("fm_layers", 4)),
            nheads=int(model.get("nheads", 8)),
            hidden_dim=int(model.get("hidden_dim", 512)),
            dim_feedforward=int(model.get("dim_feedforward", 3200)),
            chunk_size=chunk_size,
            state_dim=int(common.get("state_dim", 128)),
            action_dim=int(common.get("action_dim", 128)),
            image_feature_strategy=str(model.get("image_feature_strategy", "ACT_linear")),
            use_language_conditioning=bool(model.get("use_language_conditioning", False)),
            num_flow_steps=int(model.get("num_flow_steps", 10)),
            hand_eef_weight=float(model.get("hand_eef_weight", 2.0)),
            head_eef_weight=float(model.get("head_eef_weight", 0.0)),
        )
        policy = ACTFlowPolicy(args_override)
    else:
        raise ValueError(f"Unsupported policy_class for policy eval: {policy_class}")

    policy.eval()

    state_dict = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"Warning: checkpoint load had {len(missing)} missing and {len(unexpected)} unexpected keys")
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
    eval_mode: str,
    chunk_stride: int,
    temporal_decay: float,
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
        action_dim = int(act_mean.shape[1])

        def infer_chunk(t: int) -> np.ndarray:
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
                imgs.append(img)
            imgs = np.stack(imgs, axis=0)  # (num_cam, H, W, 3)
            imgs_t = torch.from_numpy(imgs).to(device=device, dtype=torch.float32) / 255.0
            imgs_t = imgs_t.permute(0, 3, 1, 2).unsqueeze(0)  # (1, num_cam, 3, H, W)

            qpos = states[t : t + 1].astype(np.float32)
            qpos_n = (qpos - qpos_mean) / (qpos_std + 1e-8)
            qpos_t = torch.from_numpy(qpos_n).to(device=device, dtype=torch.float32)

            conditioning_dict = {}
            with torch.no_grad():
                a_hat = policy(imgs_t, qpos_t, conditioning_dict=conditioning_dict)
            chunk = a_hat[0].detach().float().cpu().numpy().astype(np.float32)
            return chunk * act_std.reshape(1, -1) + act_mean.reshape(1, -1)

        if eval_mode == "chunk_rollout":
            pred = np.zeros((T, action_dim), dtype=np.float32)
            filled = np.zeros((T,), dtype=bool)
            stride = max(1, int(chunk_stride))
            for t in range(0, T, stride):
                chunk = infer_chunk(t)
                horizon = min(stride, int(chunk.shape[0]), T - t)
                pred[t : t + horizon] = chunk[:horizon]
                filled[t : t + horizon] = True
            if not np.all(filled):
                raise RuntimeError("Internal error: chunk_rollout left unfilled timesteps")
            return pred

        if eval_mode == "temporal_agg":
            pred_sum = np.zeros((T, action_dim), dtype=np.float64)
            weight_sum = np.zeros((T, 1), dtype=np.float64)
            for t in range(T):
                chunk = infer_chunk(t)
                horizon = min(int(chunk.shape[0]), T - t)
                offsets = np.arange(horizon, dtype=np.float64)
                weights = np.exp(-float(temporal_decay) * offsets).reshape(-1, 1)
                pred_sum[t : t + horizon] += chunk[:horizon].astype(np.float64) * weights
                weight_sum[t : t + horizon] += weights
            valid = weight_sum[:, 0] > 0
            if not np.all(valid):
                raise RuntimeError("Internal error: temporal_agg left unfilled timesteps")
            return (pred_sum / weight_sum).astype(np.float32)

        if eval_mode != "first_token":
            raise ValueError(f"Unsupported eval_mode: {eval_mode}")

        pred = np.zeros((T, action_dim), dtype=np.float32)
        for t in range(T):
            pred[t] = infer_chunk(t)[0]

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
    parser.add_argument("--policy-config-yaml", type=str, default=None, help="policy 模型配置 YAML（例如 hdt/configs/models/act_resnet.yaml）")
    parser.add_argument("--norm-stats", type=str, default=None, help="归一化统计量 pkl（例如 dataset_stats.pkl）")
    parser.add_argument("--device", type=str, default="cuda", help="推理设备（ACT 默认需要 cuda）")
    parser.add_argument("--max-steps", type=int, default=None, help="每个 episode 最多评估多少步（默认全部）")
    parser.add_argument("--seed", type=int, default=0, help="随机种子，用于随机采样式 policy 的可复现评估")
    parser.add_argument(
        "--eval-mode",
        choices=["first_token", "chunk_rollout", "temporal_agg"],
        default="first_token",
        help="policy 推理模式：first_token=每帧只取 chunk 第 0 个；chunk_rollout=每隔 chunk-stride 推理并使用前 chunk-stride 个；temporal_agg=重叠 chunk 加权平均",
    )
    parser.add_argument("--chunk-stride", type=int, default=10, help="chunk_rollout 模式每次执行多少个预测 action")
    parser.add_argument("--temporal-decay", type=float, default=0.01, help="temporal_agg 模式的 offset 指数衰减系数")
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="可选：保存批量评估结果 JSON 路径",
    )
    args = parser.parse_args()
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

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
        policy, pol_info = _make_policy(Path(args.policy_config_yaml), Path(args.policy_ckpt), device=args.device)
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
                    eval_mode=args.eval_mode,
                    chunk_stride=args.chunk_stride,
                    temporal_decay=args.temporal_decay,
                )
            except ValueError as e:
                if "No camera data available" in str(e):
                    print(f"Skipping {gt_path.name} - {e}")
                    skipped.append(gt_path.name)
                    continue
                raise
            with h5py.File(gt_path, "r") as f_gt:
                gt_actions = f_gt["action"][()]
            metrics = eval_arrays_mpjpe(gt_actions, pred_actions)
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
            metrics = eval_pair_mpjpe(gt_path, pred_path)
            metrics["file"] = pred_path.name
            per_file.append(metrics)

    if not per_file:
        raise RuntimeError("No valid prediction/gt pairs were evaluated.")

    summary = {
        "num_pairs_evaluated": len(per_file),
        "num_pairs_skipped": len(skipped),
        "skipped_files": skipped,
        "avg_timesteps_compared": _safe_mean([m["timesteps_compared"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_all_m": _safe_mean([m["mpjpe_all_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_hand_m": _safe_mean([m["mpjpe_hand_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_wrist_m": _safe_mean([m["mpjpe_wrist_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_head_m": _safe_mean([m["mpjpe_head_m"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_all_mm": _safe_mean([m["mpjpe_all_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_hand_mm": _safe_mean([m["mpjpe_hand_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_wrist_mm": _safe_mean([m["mpjpe_wrist_mm"] for m in per_file]),  # type: ignore[index]
        "avg_mpjpe_head_mm": _safe_mean([m["mpjpe_head_mm"] for m in per_file]),  # type: ignore[index]
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
