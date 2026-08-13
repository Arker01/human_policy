import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
import h5py
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_HDT_DIR = _REPO_ROOT / "hdt"
if str(_HDT_DIR) not in sys.path:
    sys.path.insert(0, str(_HDT_DIR))

from hdt.inference_utils import get_eef_kpts_from_prediction

HEAD_TRACKER_EEF = np.arange(58, 67)
WAIST_EEF = np.arange(89, 98)


def _rotation_6d_to_matrix_np(rot6d: np.ndarray) -> np.ndarray:
    import torch
    from pytorch3d.transforms import rotation_6d_to_matrix

    rot6d = np.asarray(rot6d, dtype=np.float32)
    if np.allclose(rot6d, 0.0):
        return np.eye(3, dtype=np.float32)
    return rotation_6d_to_matrix(torch.from_numpy(rot6d).unsqueeze(0)).squeeze(0).numpy()


def _pose9_mat_from_action(action_128: np.ndarray, indices: np.ndarray) -> np.ndarray:
    pose = np.asarray(action_128[indices], dtype=np.float32)
    pose_mat = np.eye(4, dtype=np.float32)
    pose_mat[:3, 3] = pose[:3]
    pose_mat[:3, :3] = _rotation_6d_to_matrix_np(pose[3:])
    return pose_mat


def _expand_hand_25_from_tips(hand_kpts_25: np.ndarray) -> np.ndarray:
    hand_kpts_25 = np.asarray(hand_kpts_25, dtype=np.float32)
    if hand_kpts_25.shape != (25, 3):
        raise ValueError(f"Expected hand_kpts_25 shape (25,3), got {hand_kpts_25.shape}")
    palm = hand_kpts_25[0].copy()
    out = hand_kpts_25.copy()
    for finger in range(5):
        base = finger * 5
        tip = base + 4
        tip_pos = hand_kpts_25[tip]
        if np.allclose(tip_pos, 0):
            continue
        alphas = np.linspace(0.0, 1.0, 5, dtype=np.float32)
        for j, a in enumerate(alphas):
            idx = base + j
            if idx == 0:
                out[idx] = palm
            else:
                out[idx] = palm + a * (tip_pos - palm)
    return out


def load_cmd_tuple_hdf5(path, *, full_hand=False, max_frames=None, start_step=0):
    data_list = []
    start_step = int(start_step or 0)
    if start_step < 0:
        raise ValueError("start_step must be non-negative")

    with h5py.File(path, 'r') as file:
        # Processed HDF5
        assert "/action" in file
        total_frames = file["/action"].shape[0]
        if start_step >= total_frames:
            raise ValueError(f"start_step={start_step} is outside episode length {total_frames}")
        frame_count = total_frames - start_step
        if max_frames is not None:
            frame_count = min(frame_count, int(max_frames))
        end_step = start_step + frame_count
        actions = file["/action"][start_step:end_step]
        waist_valid = file["waist_valid"][start_step:end_step].astype(bool) if "waist_valid" in file else None
        head_tracker_valid = file["head_tracker_valid"][start_step:end_step].astype(bool) if "head_tracker_valid" in file else None
        has_waist = bool(np.any(np.abs(actions[:, WAIST_EEF]) > 1e-6))
        has_head_tracker = bool(np.any(np.abs(actions[:, HEAD_TRACKER_EEF]) > 1e-6))
        for i in range(frame_count):
            action = actions[i]
            cur_cmd_dict = get_eef_kpts_from_prediction(action)
            # Format post-processed data to match expected structure
            head_mat = cur_cmd_dict['head_mat']
            left_wrist_mat = cur_cmd_dict['left_wrist_mat']
            right_wrist_mat = cur_cmd_dict['right_wrist_mat']
            left_hand_kpts = cur_cmd_dict['left_hand_kpts']
            right_hand_kpts = cur_cmd_dict['right_hand_kpts']

            if full_hand:
                left_hand_kpts = _expand_hand_25_from_tips(left_hand_kpts)
                right_hand_kpts = _expand_hand_25_from_tips(right_hand_kpts)
            
            # Create skeleton joint structures from the keypoints
            left_skeleton_joints = np.zeros((25, 4, 4))
            right_skeleton_joints = np.zeros((25, 4, 4))
            
            # Set identity matrices for each joint
            for j in range(25):
                left_skeleton_joints[j] = np.eye(4)
                right_skeleton_joints[j] = np.eye(4)
            
            # Fill in the finger positions
            for j in range(min(25, len(left_hand_kpts))):
                left_skeleton_joints[j, 3, 0:3] = left_hand_kpts[j]
            
            for j in range(min(25, len(right_hand_kpts))):
                right_skeleton_joints[j, 3, 0:3] = right_hand_kpts[j]
            
            # Construct data dictionary matching expected format
            data = {
                'head': head_mat.flatten(order="F").tolist(),
                'rightWrist': right_wrist_mat.flatten(order="F").tolist(),
                'leftWrist': left_wrist_mat.flatten(order="F").tolist(),
                'rightSkeleton': {
                    'joints': right_skeleton_joints.reshape(-1).tolist()
                },
                'leftSkeleton': {
                    'joints': left_skeleton_joints.reshape(-1).tolist()
                }
            }
            if has_waist and (waist_valid is None or waist_valid[i]):
                waist_mat = _pose9_mat_from_action(action, WAIST_EEF)
                data['waist'] = waist_mat.flatten(order="F").tolist()
            if has_head_tracker and (head_tracker_valid is None or head_tracker_valid[i]):
                head_tracker_mat = _pose9_mat_from_action(action, HEAD_TRACKER_EEF)
                data['headTracker'] = head_tracker_mat.flatten(order="F").tolist()
            data_list.append(data)

    return data_list

def _load_act_policy(model_cfg_path, chunk_size, camera_names):
    import yaml
    import torch
    from hdt.policy import ACTPolicy

    with open(model_cfg_path, "r") as fp:
        trainer_config = yaml.safe_load(fp)

    policy_config = {
        "lr": 1e-4,
        "num_queries": chunk_size,
        "kl_weight": trainer_config["model"]["kl_weight"],
        "hidden_dim": trainer_config["model"]["hidden_dim"],
        "chunk_size": chunk_size,
        "dim_feedforward": trainer_config["model"]["dim_feedforward"],
        "lr_backbone": float(trainer_config["model"]["lr_backbone"]),
        "backbone": trainer_config["model"]["backbone"],
        "enc_layers": trainer_config["model"]["enc_layers"],
        "dec_layers": trainer_config["model"]["dec_layers"],
        "nheads": trainer_config["model"]["nheads"],
        "camera_names": camera_names,
        "state_dim": 128,
        "action_dim": 128,
        "image_feature_strategy": trainer_config["model"]["image_feature_strategy"],
        "use_language_conditioning": trainer_config["model"]["use_language_conditioning"],
    }

    policy = ACTPolicy(policy_config)
    policy.eval()
    return policy

def predict_episode_to_hdf5(
    episode_hdf5_path,
    out_hdf5_path,
    ckpt_path,
    model_cfg_path,
    *,
    chunk_size=100,
    camera_name="top",
    max_steps=None,
    device=None,
):
    import torch
    import cv2
    import pickle
    from hdt.modeling.utils import make_visual_encoder

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    policy = _load_act_policy(model_cfg_path, chunk_size, [camera_name]).to(device)
    state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
    policy.load_state_dict(state_dict, strict=False)
    policy.eval()

    _, visual_preprocessor = make_visual_encoder("ACT", {})

    with h5py.File(episode_hdf5_path, "r") as root:
        embodiment = str(root.attrs.get("embodiment", "human_mocap_annotated"))
        ckpt_dir = os.path.dirname(ckpt_path)
        if os.path.basename(ckpt_dir).startswith("policy_iter_"):
            ckpt_dir = os.path.dirname(ckpt_dir)
        stats_path = os.path.join(ckpt_dir, "dataset_stats.pkl")
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"dataset_stats.pkl not found next to ckpt: {stats_path}")
        with open(stats_path, "rb") as f:
            loaded = pickle.load(f)
        norm_stats = loaded[0] if isinstance(loaded, tuple) and len(loaded) == 2 else loaded
        if embodiment not in norm_stats:
            raise KeyError(f"embodiment={embodiment} not found in dataset_stats.pkl: {list(norm_stats.keys())}")
        ns = norm_stats[embodiment]
        qpos_mean = ns["qpos_mean"].astype(np.float32)
        qpos_std = ns["qpos_std"].astype(np.float32)
        action_mean = ns["action_mean"].astype(np.float32)
        action_std = ns["action_std"].astype(np.float32)

        states = root["observation.state"][()]

        available_cams = [k[len("observation.image."):] for k in root.keys() if k.startswith("observation.image.")]
        if camera_name in available_cams:
            effective_camera = camera_name
        elif camera_name == "top" and "left" in available_cams:
            effective_camera = "left"
        elif camera_name == "top" and "right" in available_cams:
            effective_camera = "right"
        elif "top" in available_cams:
            effective_camera = "top"
        elif "left" in available_cams:
            effective_camera = "left"
        elif available_cams:
            effective_camera = available_cams[0]
        else:
            raise KeyError(f"No observation.image.* keys found in {episode_hdf5_path}")

        images = root[f"observation.image.{effective_camera}"]
        T = states.shape[0]
        if max_steps is not None:
            T = min(T, int(max_steps))

        pred_actions = np.zeros((T, 128), dtype=np.float32)

        for t in range(T):
            img = images[t]
            if len(img.shape) == 1:
                img = cv2.imdecode(img, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 3:
                if img.shape[-1] == 3:
                    img = img
                else:
                    img = img[:, :, :3]
            else:
                raise ValueError(f"Unsupported image shape at t={t}: {img.shape}")

            if img.shape[0] != 240 or img.shape[1] != 320:
                img = cv2.resize(img, (320, 240))

            img_nchw = img.transpose(2, 0, 1)[None, ...].astype(np.uint8)
            img_tensor = visual_preprocessor(img_nchw).unsqueeze(0).to(device)

            qpos_raw = states[t].astype(np.float32)
            qpos_norm = (qpos_raw - qpos_mean) / (qpos_std + 1e-6)
            qpos = torch.from_numpy(qpos_norm).unsqueeze(0).to(device)

            with torch.no_grad():
                a_hat = policy(img_tensor, qpos, conditioning_dict=None)
            if isinstance(a_hat, dict):
                raise ValueError("Policy returned a loss dict; expected trajectory predictions.")
            pred_norm = a_hat[0, 0].detach().cpu().numpy().astype(np.float32)
            pred_actions[t] = pred_norm * (action_std + 1e-6) + action_mean

    os.makedirs(os.path.dirname(out_hdf5_path) or ".", exist_ok=True)
    with h5py.File(out_hdf5_path, "w") as f_out:
        f_out.create_dataset("action", data=pred_actions, compression="gzip", compression_opts=4)
        f_out.attrs["sim"] = np.bool_(False)
        f_out.attrs["embodiment"] = "predicted"
        f_out.attrs["description"] = f"predicted_from:{os.path.basename(episode_hdf5_path)}"

def _transform_points(points, transform_mat):
    points_h = np.concatenate([points, np.ones((points.shape[0], 1), dtype=points.dtype)], axis=1)
    transformed = np.dot(transform_mat, points_h.T).T
    return transformed[:, :3]

def _action_to_eval_joints(action_128: np.ndarray) -> dict:
    import hdt.constants as C
    cmd = get_eef_kpts_from_prediction(action_128)
    head_mat = cmd["head_mat"]
    lw_mat = cmd["left_wrist_mat"]
    rw_mat = cmd["right_wrist_mat"]
    waist_mat = cmd["waist_mat"]
    lk_full = cmd["left_hand_kpts"]
    rk_full = cmd["right_hand_kpts"]
    valid_idx = C.RETARGETTING_INDICES
    lk_local = lk_full[valid_idx].astype(np.float32)
    rk_local = rk_full[valid_idx].astype(np.float32)

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

def evaluate_mpjpe(
    gt_hdf5_path: str,
    pred_hdf5_path: str,
    save_json_path: str | None = None,
    *,
    start_step=0,
    _dirty_start_check_frames=20,
    _dirty_start_jump_threshold_m=0.3,
    _dirty_start_settle_frames=0,
) -> dict:
    with h5py.File(gt_hdf5_path, "r") as f_gt, h5py.File(pred_hdf5_path, "r") as f_pr:
        gt_actions = f_gt["action"][()]
        pr_actions = f_pr["action"][()]

    start_step = int(start_step or 0)
    if start_step < 0:
        raise ValueError("start_step must be non-negative")

    gt_start = _detect_valid_start_from_actions(
        gt_actions,
        check_frames=_dirty_start_check_frames,
        jump_threshold_m=_dirty_start_jump_threshold_m,
        settle_frames=_dirty_start_settle_frames,
    )
    pr_start = _detect_valid_start_from_actions(
        pr_actions,
        check_frames=_dirty_start_check_frames,
        jump_threshold_m=_dirty_start_jump_threshold_m,
        settle_frames=_dirty_start_settle_frames,
    )
    valid_start = max(start_step, gt_start, pr_start)

    T = min(gt_actions.shape[0], pr_actions.shape[0]) - valid_start
    if T <= 0:
        raise ValueError("No overlapping timesteps between gt and prediction.")
    gt_actions = gt_actions[valid_start:valid_start + T]
    pr_actions = pr_actions[valid_start:valid_start + T]

    all_joint_err = []
    hand_err = []
    wrist_err = []
    head_err = []
    waist_err = []

    for t in range(T):
        gt_j = _action_to_eval_joints(gt_actions[t])
        pr_j = _action_to_eval_joints(pr_actions[t])

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

    metrics = {
        "timesteps_compared": int(T),
        "timesteps_skipped": int(valid_start),
        "start_step": int(start_step),
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

    if save_json_path is not None:
        Path(save_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_json_path, "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def _detect_valid_start_from_actions(actions, *, check_frames=20, jump_threshold_m=0.3, settle_frames=0):
    if actions.shape[0] <= 1 or check_frames <= 1 or jump_threshold_m <= 0:
        return 0
    _HDT_DIR_LOCAL = _REPO_ROOT / "hdt"
    import sys as _sys
    if str(_HDT_DIR_LOCAL) not in _sys.path:
        _sys.path.insert(0, str(_HDT_DIR_LOCAL))
    import hdt.constants as C

    slices = (
        C.OUTPUT_HEAD_EEF[0:3],
        C.OUTPUT_RIGHT_EEF[0:3],
        C.OUTPUT_NECK[0:3],
        C.OUTPUT_LEFT_EEF[0:3],
        C.OUTPUT_WAIST[0:3],
    )
    n = min(int(check_frames), int(actions.shape[0]))
    last_bad_next_frame = -1
    for sl in slices:
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


def _safe_mean(values):
    return float(np.mean(values)) if values else float("nan")


def _to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_norm_stats(stats_path, embodiment):
    import pickle as _pkl
    with open(stats_path, "rb") as f:
        loaded = _pkl.load(f)
    if isinstance(loaded, tuple) and len(loaded) == 2:
        loaded = loaded[0]
    if isinstance(loaded, dict) and "qpos_mean" in loaded:
        return {k: _to_numpy(v) for k, v in loaded.items()}
    if isinstance(loaded, dict) and embodiment is not None and embodiment in loaded:
        return {k: _to_numpy(v) for k, v in loaded[embodiment].items()}
    if isinstance(loaded, dict) and embodiment is not None:
        emb = embodiment.encode() if isinstance(next(iter(loaded.keys())), bytes) else embodiment
        if emb in loaded:
            return {k: _to_numpy(v) for k, v in loaded[emb].items()}
    if isinstance(loaded, dict):
        first = next(iter(loaded.values()))
        if isinstance(first, dict):
            return {k: _to_numpy(v) for k, v in first.items()}
    raise ValueError(f"Unsupported stats format: {type(loaded)}")


def predict_dir_to_hdf5(
    gt_dir,
    out_dir,
    ckpt_path,
    model_cfg_path,
    *,
    chunk_size=100,
    camera_name="top",
    max_steps=None,
    device=None,
    glob="*.hdf5",
):
    gt_files = sorted(Path(gt_dir).glob(glob))
    if not gt_files:
        raise RuntimeError(f"No files matched in {gt_dir} with glob={glob}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for gt_path in gt_files:
        out_path = Path(out_dir) / gt_path.name
        print(f"[predict] {gt_path.name} -> {out_path}", flush=True)
        predict_episode_to_hdf5(
            str(gt_path),
            str(out_path),
            ckpt_path,
            model_cfg_path,
            chunk_size=chunk_size,
            camera_name=camera_name,
            max_steps=max_steps,
            device=device,
        )


def batch_evaluate_mpjpe(
    gt_dir,
    pred_dir=None,
    *,
    ckpt=None,
    norm_stats=None,
    model_cfg=None,
    chunk_size=100,
    camera="top",
    device=None,
    glob="*.hdf5",
    out_json=None,
    dirty_start_check_frames=20,
    dirty_start_jump_threshold_m=0.3,
    dirty_start_settle_frames=0,
):
    gt_dir = Path(gt_dir)
    if not gt_dir.exists():
        raise FileNotFoundError(f"gt_dir not found: {gt_dir}")

    use_policy = ckpt is not None
    if use_policy:
        if norm_stats is None:
            stats_path = Path(ckpt).parent / "dataset_stats.pkl"
            if stats_path.exists():
                norm_stats = str(stats_path)
            else:
                raise ValueError("--norm-stats is required when using --ckpt")
        if pred_dir is None:
            import tempfile
            pred_dir = Path(tempfile.mkdtemp(prefix="mpjpe_pred_"))
        predict_dir_to_hdf5(
            str(gt_dir), str(pred_dir), ckpt, model_cfg or str(_REPO_ROOT / "hdt/configs/models/act_resnet.yaml"),
            chunk_size=chunk_size, camera_name=camera, max_steps=None, device=device, glob=glob,
        )
    else:
        if pred_dir is None:
            raise ValueError("--pred-dir is required unless --ckpt is set")

    pred_dir = Path(pred_dir)
    gt_files = sorted(gt_dir.glob(glob))
    pred_files = sorted(pred_dir.glob(glob))
    gt_names = {f.name: f for f in gt_files}

    per_file = []
    skipped = []

    for pred_path in pred_files:
        if pred_path.name not in gt_names:
            skipped.append(pred_path.name)
            continue
        gt_path = gt_names[pred_path.name]
        metrics = evaluate_mpjpe(
            str(gt_path), str(pred_path),
            start_step=0,
            _dirty_start_check_frames=dirty_start_check_frames,
            _dirty_start_jump_threshold_m=dirty_start_jump_threshold_m,
            _dirty_start_settle_frames=dirty_start_settle_frames,
        )
        metrics["file"] = pred_path.name
        per_file.append(metrics)

    for gt_path in gt_files:
        if gt_path.name not in {f.name for f in pred_files}:
            skipped.append(gt_path.name)

    if not per_file:
        raise RuntimeError("No valid prediction/gt pairs were evaluated.")

    summary = {
        "num_pairs_evaluated": len(per_file),
        "num_pairs_skipped": len(skipped),
        "skipped_files": skipped,
        "avg_timesteps_compared": _safe_mean([m["timesteps_compared"] for m in per_file]),
        "avg_timesteps_skipped": _safe_mean([m.get("timesteps_skipped", 0) for m in per_file]),
        "avg_mpjpe_all_m": _safe_mean([m["mpjpe_all_m"] for m in per_file]),
        "avg_mpjpe_hand_m": _safe_mean([m["mpjpe_hand_m"] for m in per_file]),
        "avg_mpjpe_wrist_m": _safe_mean([m["mpjpe_wrist_m"] for m in per_file]),
        "avg_mpjpe_head_m": _safe_mean([m["mpjpe_head_m"] for m in per_file]),
        "avg_mpjpe_waist_m": _safe_mean([m["mpjpe_waist_m"] for m in per_file]),
        "avg_mpjpe_all_mm": _safe_mean([m["mpjpe_all_mm"] for m in per_file]),
        "avg_mpjpe_hand_mm": _safe_mean([m["mpjpe_hand_mm"] for m in per_file]),
        "avg_mpjpe_wrist_mm": _safe_mean([m["mpjpe_wrist_mm"] for m in per_file]),
        "avg_mpjpe_head_mm": _safe_mean([m["mpjpe_head_mm"] for m in per_file]),
        "avg_mpjpe_waist_mm": _safe_mean([m["mpjpe_waist_mm"] for m in per_file]),
    }

    print("===== Batch MPJPE Summary (plot_keypoints_ys.py) =====")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if out_json is not None:
        out_path = Path(out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"summary": summary, "per_file": per_file}, f, indent=2)
        print(f"Saved metrics to: {out_path}")

    return {"summary": summary, "per_file": per_file}


def _frames_from_hdf5(input_file, *, full_hand=False, max_frames=None, draw_spine=False, start_step=0):
    datas = load_cmd_tuple_hdf5(input_file, full_hand=full_hand, max_frames=max_frames, start_step=start_step)
    
    # Prepare data for animation
    frames = []
    for data in datas:
        head_mat = np.array(data['head']).reshape(4, 4, order="F")
        right_wrist_mat = np.array(data['rightWrist']).reshape(4, 4, order="F")
        left_wrist_mat = np.array(data['leftWrist']).reshape(4, 4, order="F")
        waist_mat = np.array(data['waist']).reshape(4, 4, order="F") if 'waist' in data else None
        head_tracker_mat = np.array(data['headTracker']).reshape(4, 4, order="F") if 'headTracker' in data else None
        
        right_fingers = np.array(data["rightSkeleton"]["joints"]).reshape(25, 4, 4)[:, 3, 0:3]
        left_fingers = np.array(data["leftSkeleton"]["joints"]).reshape(25, 4, 4)[:, 3, 0:3]
        
        # Extract positions (translation vectors) from transformation matrices
        head_pos = head_mat[:3, 3]
        right_wrist_pos = right_wrist_mat[:3, 3]
        left_wrist_pos = left_wrist_mat[:3, 3]
        
        # Extract rotation axes (first 3 columns of rotation matrix)
        def get_axes(mat, scale=0.1):
            R = mat[:3, :3]
            pos = mat[:3, 3]
            x_axis = pos + R[:, 0] * scale
            y_axis = pos + R[:, 1] * scale
            z_axis = pos + R[:, 2] * scale
            return pos, x_axis, y_axis, z_axis
        
        head_pos, head_x, head_y, head_z = get_axes(head_mat)
        rw_pos, rw_x, rw_y, rw_z = get_axes(right_wrist_mat)
        lw_pos, lw_x, lw_y, lw_z = get_axes(left_wrist_mat)
        if waist_mat is not None:
            waist_pos, waist_x, waist_y, waist_z = get_axes(waist_mat)
        if head_tracker_mat is not None:
            head_tracker_pos, head_tracker_x, head_tracker_y, head_tracker_z = get_axes(head_tracker_mat)
        
        # Transform finger positions from local wrist frame to world frame
        def transform_points(points, transform_mat):
            # Convert to homogeneous coordinates
            points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
            # Transform points
            transformed = np.dot(transform_mat, points_h.T).T
            return transformed[:, :3]  # Return only xyz coordinates

        right_fingers_world = transform_points(right_fingers, right_wrist_mat)
        left_fingers_world = transform_points(left_fingers, left_wrist_mat)
        
        frame_data = {
            'positions': {
                'head': head_pos,
                'right_wrist': right_wrist_pos,
                'left_wrist': left_wrist_pos
            },
            'axes': {
                'head': (head_x, head_y, head_z),
                'right_wrist': (rw_x, rw_y, rw_z),
                'left_wrist': (lw_x, lw_y, lw_z)
            },
            'fingers': {
                'right': right_fingers_world,
                'left': left_fingers_world
            }
        }
        if waist_mat is not None:
            frame_data['positions']['waist'] = waist_pos
            frame_data['axes']['waist'] = (waist_x, waist_y, waist_z)
        if head_tracker_mat is not None:
            frame_data['positions']['head_tracker'] = head_tracker_pos
            frame_data['axes']['head_tracker'] = (head_tracker_x, head_tracker_y, head_tracker_z)
        if draw_spine and head_tracker_mat is not None:
            links = []
            if waist_mat is not None:
                links.append(('head_tracker_to_waist', np.stack([head_tracker_pos, waist_pos], axis=0)))
            links.append(('head_tracker_to_head', np.stack([head_tracker_pos, head_pos], axis=0)))
            frame_data['body_links'] = links
        frames.append(frame_data)

    return frames


def main(input_file, *, full_hand=False, max_frames=None, draw_spine=False, start_step=0):
    frames = _frames_from_hdf5(input_file, full_hand=full_hand, max_frames=max_frames, draw_spine=draw_spine, start_step=start_step)

    # Create figure
    fig = go.Figure()
    
    # Add initial positions
    colors = {
        'head': 'blue',
        'right_wrist': 'red',
        'left_wrist': 'green',
        'waist': 'orange',
        'head_tracker': 'purple',
    }
    axis_colors = ['red', 'green', 'blue']  # x, y, z axes colors
    
    def add_coordinate_frame(pos, axes, name, base_color):
        # Add position marker
        fig.add_trace(go.Scatter3d(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            mode='markers',
            name=f"{name}_position",
            marker=dict(size=8, color=base_color)
        ))
        
        # Add coordinate axes
        for i, (axis_end, color) in enumerate(zip(axes, axis_colors)):
            fig.add_trace(go.Scatter3d(
                x=[pos[0], axis_end[0]],
                y=[pos[1], axis_end[1]],
                z=[pos[2], axis_end[2]],
                mode='lines',
                name=f"{name}_{['x', 'y', 'z'][i]}_axis",
                line=dict(color=color, width=3)
            ))
    
    def add_hand_keypoints(pos, fingers, name, color):
        # Add finger keypoints
        fig.add_trace(go.Scatter3d(
            x=fingers[:, 0], y=fingers[:, 1], z=fingers[:, 2],
            mode='markers',
            name=f"{name}_fingers",
            marker=dict(size=4, color=color, opacity=0.7)
        ))

    def add_body_links(body_links):
        link_colors = {
            'head_tracker_to_waist': 'black',
            'head_tracker_to_head': 'purple',
        }
        link_names = {
            'head_tracker_to_waist': 'head_tracker_to_waist',
            'head_tracker_to_head': 'head_tracker_to_head',
        }
        for name, points in body_links:
            color = link_colors.get(name, 'black')
            label = link_names.get(name, name)
            fig.add_trace(go.Scatter3d(
                x=points[:, 0], y=points[:, 1], z=points[:, 2],
                mode='lines+markers',
                name=label,
                line=dict(color=color, width=5),
                marker=dict(size=4, color=color)
            ))

    def make_body_link_traces(body_links):
        link_colors = {
            'head_tracker_to_waist': 'black',
            'head_tracker_to_head': 'purple',
        }
        traces = []
        for name, points in body_links:
            color = link_colors.get(name, 'black')
            traces.append(go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='lines+markers',
                line=dict(color=color, width=5),
                marker=dict(size=4, color=color)
            ))
        return traces

    def add_spine(spine_points):
        fig.add_trace(go.Scatter3d(
            x=spine_points[:, 0], y=spine_points[:, 1], z=spine_points[:, 2],
            mode='lines+markers',
            name='spine',
            line=dict(color='black', width=5),
            marker=dict(size=4, color='black')
        ))
    
    # Initial frame
    first_frame = frames[0]
    parts = ['head', 'right_wrist', 'left_wrist']
    if 'waist' in first_frame['positions']:
        parts.append('waist')
    if 'head_tracker' in first_frame['positions']:
        parts.append('head_tracker')
    # Add origin coordinate frame
    
    
    for part in parts:
        pos = first_frame['positions'][part]
        axes = first_frame['axes'][part]
        add_coordinate_frame(pos, axes, part, colors[part])
    
    # Add initial hand keypoints
    add_hand_keypoints(first_frame['positions']['right_wrist'], 
                      first_frame['fingers']['right'], 
                      'right', colors['right_wrist'])
    add_hand_keypoints(first_frame['positions']['left_wrist'], 
                      first_frame['fingers']['left'], 
                      'left', colors['left_wrist'])
    if draw_spine and 'spine' in first_frame:
        add_spine(first_frame['spine'])
    if draw_spine and 'body_links' in first_frame:
        add_body_links(first_frame['body_links'])
    
    # Update layout
    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=-1.73, y=1.0, z=0.5)
            ),
            # xaxis=dict(range=[-0.3, 0.3]),
            # yaxis=dict(range=[0, 1.4]),
            # zaxis=dict(range=[-0.5, 0.1]),
            aspectratio=dict(x=1, y=1, z=1)
        ),
        title="3D Transformation Visualization",
        margin=dict(l=0, r=0, t=36, b=0),
        showlegend=True,
        updatemenus=[{
            'buttons': [
                {
                    'args': [None, {
                        'frame': {'duration': 50, 'redraw': True},
                        'fromcurrent': True,
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Play',
                    'method': 'animate'
                },
                {
                    'args': [[None], {
                        'frame': {'duration': 0, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 0}
                    }],
                    'label': 'Pause',
                    'method': 'animate'
                }
            ],
            'type': 'buttons',
            'direction': 'left',
            'showactive': True
        }],
        sliders=[{
            'currentvalue': {'prefix': 'Frame: '},
            'pad': {'t': 50},
            'len': 0.9,
            'x': 0.1,
            'xanchor': 'left',
            'y': 0,
            'yanchor': 'top',
            'steps': [{
                'args': [[str(i)], {
                    'frame': {'duration': 0, 'redraw': True},
                    'mode': 'immediate',
                    'transition': {'duration': 0}
                }],
                'label': str(i),
                'method': 'animate'
            } for i in range(len(frames))]
        }]
    )
    
    # Create animation frames
    fig_frames = []
    for i, frame in enumerate(frames):
        frame_traces = []
        
        # Add origin to each frame
        # frame_traces.append(go.Scatter3d(
        #     x=[0], y=[0], z=[0],
        #     mode='markers',
        #     marker=dict(size=8, color='black')
        # ))
        
        # Add origin axes to each frame
        # for axis_end, color in zip(origin_axes, axis_colors):
        #     frame_traces.append(go.Scatter3d(
        #         x=[0, axis_end[0]],
        #         y=[0, axis_end[1]],
        #         z=[0, axis_end[2]],
        #         mode='lines',
        #         line=dict(color=color, width=3)
        #     ))
            
        for part in parts:
            pos = frame['positions'][part]
            axes = frame['axes'][part]
            
            # Position marker
            frame_traces.append(go.Scatter3d(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                mode='markers',
                marker=dict(size=8, color=colors[part])
            ))
            
            # Coordinate axes
            for axis_end, color in zip(axes, axis_colors):
                frame_traces.append(go.Scatter3d(
                    x=[pos[0], axis_end[0]],
                    y=[pos[1], axis_end[1]],
                    z=[pos[2], axis_end[2]],
                    mode='lines',
                    line=dict(color=color, width=3)
                ))
        
        # Add finger keypoints to each frame
        frame_traces.append(go.Scatter3d(
            x=frame['fingers']['right'][:, 0],
            y=frame['fingers']['right'][:, 1],
            z=frame['fingers']['right'][:, 2],
            mode='markers',
            marker=dict(size=4, color=colors['right_wrist'], opacity=0.7)
        ))
        
        frame_traces.append(go.Scatter3d(
            x=frame['fingers']['left'][:, 0],
            y=frame['fingers']['left'][:, 1],
            z=frame['fingers']['left'][:, 2],
            mode='markers',
            marker=dict(size=4, color=colors['left_wrist'], opacity=0.7)
        ))

        if draw_spine and 'spine' in frame:
            spine_points = frame['spine']
            frame_traces.append(go.Scatter3d(
                x=spine_points[:, 0],
                y=spine_points[:, 1],
                z=spine_points[:, 2],
                mode='lines+markers',
                line=dict(color='black', width=5),
                marker=dict(size=4, color='black')
            ))
        if draw_spine and 'body_links' in frame:
            frame_traces.extend(make_body_link_traces(frame['body_links']))
        
        fig_frames.append(go.Frame(data=frame_traces, name=str(i)))
    
    fig.frames = fig_frames
    
    return fig, frames

def _save_html(fig, out_path):
    import plotly.io as pio

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    pio.write_html(fig, out_path, auto_open=False, include_plotlyjs="cdn")

def _save_mp4(fig, out_path, *, fps=20, width=960, height=720, frames=None):
    import subprocess
    import tempfile

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if frames is not None:
        if isinstance(frames, dict) and frames.get("kind") == "compare":
            _save_mp4_matplotlib_compare(frames, out_path, fps=fps, width=width, height=height)
            return
        _save_mp4_matplotlib(frames, out_path, fps=fps, width=width, height=height)
        return

    # Fallback: kaleido-based rendering (slow)
    ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode != 0:
        raise RuntimeError("ffmpeg not found in PATH. Please install ffmpeg to export plotly-frame mp4.")

    import plotly.io as pio
    try:
        _ = pio.kaleido.scope
    except Exception as e:
        raise RuntimeError("plotly kaleido is required to export video frames. Please install kaleido.") from e

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, fr in enumerate(fig.frames):
            frame_fig = go.Figure(data=fr.data, layout=fig.layout)
            png_bytes = frame_fig.to_image(format="png", width=width, height=height, scale=1)
            frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            with open(frame_path, "wb") as f:
                f.write(png_bytes)

        cmd = (
            f"ffmpeg -y -framerate {int(fps)} -i {os.path.join(tmpdir, 'frame_%06d.png')} "
            f"-c:v libx264 -pix_fmt yuv420p {out_path}"
        )
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {proc.stderr.strip()}")


def _encode_png_sequence_to_mp4(tmpdir, out_path, *, fps):
    import subprocess

    pattern = os.path.join(tmpdir, 'frame_%06d.png')
    ffmpeg = subprocess.run(["bash", "-lc", "command -v ffmpeg"], capture_output=True, text=True)
    if ffmpeg.returncode == 0:
        cmd = (
            f"ffmpeg -y -framerate {int(fps)} -i {pattern} "
            f"-vf \"scale=trunc(iw/2)*2:trunc(ih/2)*2\" "
            f"-c:v libx264 -pix_fmt yuv420p {out_path}"
        )
        proc = subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True)
        if proc.returncode == 0:
            return
        ffmpeg_err = proc.stderr.strip()
    else:
        ffmpeg_err = "ffmpeg not found"

    try:
        import cv2
    except Exception as e:
        raise RuntimeError(f"Could not encode mp4. ffmpeg error: {ffmpeg_err}; cv2 import failed: {e}") from e

    frame_paths = sorted(Path(tmpdir).glob("frame_*.png"))
    if not frame_paths:
        raise RuntimeError("No rendered frames found for mp4 encoding.")
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        raise RuntimeError(f"Could not read first rendered frame: {frame_paths[0]}")
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open cv2 VideoWriter for {out_path}. ffmpeg error: {ffmpeg_err}")
    try:
        writer.write(first)
        for frame_path in frame_paths[1:]:
            img = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError(f"Could not read rendered frame: {frame_path}")
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            writer.write(img)
    finally:
        writer.release()


def _collect_frame_points(frame):
    pts = []
    for pos in frame.get('positions', {}).values():
        pts.append(np.asarray(pos, dtype=np.float32).reshape(1, 3))
    for hand_pts in frame.get('fingers', {}).values():
        pts.append(np.asarray(hand_pts, dtype=np.float32).reshape(-1, 3))
    for _, link_pts in frame.get('body_links', []):
        pts.append(np.asarray(link_pts, dtype=np.float32).reshape(-1, 3))
    if 'spine' in frame:
        pts.append(np.asarray(frame['spine'], dtype=np.float32).reshape(-1, 3))
    if not pts:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(pts, axis=0)


def _save_mp4_matplotlib_compare(compare_frames, out_path, *, fps=20, width=960, height=720):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import subprocess
    import tempfile

    gt_frames = compare_frames["gt"]
    pred_frames = compare_frames["pred"]
    labels = compare_frames.get("labels", {"gt": "GT", "pred": "Pred"})
    T = min(len(gt_frames), len(pred_frames))
    if T <= 0:
        raise ValueError("No overlapping frames for compare video.")

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    parts = ['head', 'right_wrist', 'left_wrist']
    for optional in ('waist', 'head_tracker'):
        if optional in gt_frames[0].get('positions', {}) or optional in pred_frames[0].get('positions', {}):
            parts.append(optional)

    all_pts = np.concatenate(
        [_collect_frame_points(f) for f in gt_frames[:T]] +
        [_collect_frame_points(f) for f in pred_frames[:T]],
        axis=0,
    )
    finite = np.all(np.isfinite(all_pts), axis=1)
    all_pts = all_pts[finite]
    margin = 0.12
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    gt_colors = {
        'head': '#1f77b4',
        'right_wrist': '#d62728',
        'left_wrist': '#2ca02c',
        'waist': '#ff7f0e',
        'head_tracker': '#9467bd',
    }
    pred_colors = {
        'head': '#17becf',
        'right_wrist': '#e377c2',
        'left_wrist': '#bcbd22',
        'waist': '#8c564b',
        'head_tracker': '#7f7f7f',
    }

    def draw_motion(ax, frame, colors, label, *, linestyle, marker, alpha):
        for part in parts:
            if part not in frame.get('positions', {}):
                continue
            pos = frame['positions'][part]
            c = colors.get(part, 'black')
            ax.scatter(*pos, color=c, s=50, marker=marker, alpha=alpha, label=f"{label} {part}")
            if part in frame.get('axes', {}):
                for axis_end in frame['axes'][part]:
                    ax.plot([pos[0], axis_end[0]], [pos[1], axis_end[1]], [pos[2], axis_end[2]],
                            color=c, linewidth=1.5, linestyle=linestyle, alpha=alpha)

        for side, wrist_part in (('right', 'right_wrist'), ('left', 'left_wrist')):
            if side not in frame.get('fingers', {}):
                continue
            pts = frame['fingers'][side]
            c = colors.get(wrist_part, 'black')
            ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=12, marker=marker,
                       alpha=alpha, label=f"{label} {side} hand")

        for name, points in frame.get('body_links', []):
            ax.plot(points[:, 0], points[:, 1], points[:, 2],
                    color=colors.get('head_tracker', 'black'), linewidth=2.0,
                    linestyle=linestyle, marker=marker, markersize=3, alpha=alpha,
                    label=f"{label} {name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        from tqdm import tqdm
        for i in tqdm(range(T), desc="Rendering compare frames", unit="frame"):
            fig_m = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax = fig_m.add_subplot(111, projection='3d')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.view_init(elev=15, azim=210)
            ax.set_title(f"{labels.get('gt', 'GT')} vs {labels.get('pred', 'Pred')} | Frame {i}")
            fig_m.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.94)

            draw_motion(ax, gt_frames[i], gt_colors, labels.get('gt', 'GT'),
                        linestyle='-', marker='o', alpha=0.9)
            draw_motion(ax, pred_frames[i], pred_colors, labels.get('pred', 'Pred'),
                        linestyle='--', marker='x', alpha=0.85)

            handles, legend_labels = ax.get_legend_handles_labels()
            dedup = {}
            for h, l in zip(handles, legend_labels):
                dedup.setdefault(l, h)
            ax.legend(dedup.values(), dedup.keys(), loc='upper left', fontsize=6, markerscale=0.8)

            frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            fig_m.savefig(frame_path, dpi=dpi)
            plt.close(fig_m)

        _encode_png_sequence_to_mp4(tmpdir, out_path, fps=fps)


def _save_mp4_matplotlib(frames, out_path, *, fps=20, width=960, height=720):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    import subprocess
    import tempfile

    dpi = 100
    fig_w = width / dpi
    fig_h = height / dpi

    colors = {
        'head': 'blue',
        'right_wrist': 'red',
        'left_wrist': 'green',
        'waist': 'orange',
        'head_tracker': 'purple',
    }
    axis_colors = ['red', 'green', 'blue']
    parts = ['head', 'right_wrist', 'left_wrist']
    if frames and 'waist' in frames[0]['positions']:
        parts.append('waist')
    if frames and 'head_tracker' in frames[0]['positions']:
        parts.append('head_tracker')

    all_pts = np.concatenate(
        [np.stack([f['positions'][p] for p in parts])
         for f in frames], axis=0
    )
    margin = 0.1
    xlim = (all_pts[:, 0].min() - margin, all_pts[:, 0].max() + margin)
    ylim = (all_pts[:, 1].min() - margin, all_pts[:, 1].max() + margin)
    zlim = (all_pts[:, 2].min() - margin, all_pts[:, 2].max() + margin)

    with tempfile.TemporaryDirectory() as tmpdir:
        from tqdm import tqdm
        for i, frame in enumerate(tqdm(frames, desc="Rendering frames", unit="frame")):
            fig_m = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
            ax = fig_m.add_subplot(111, projection='3d')
            ax.set_xlim(*xlim)
            ax.set_ylim(*ylim)
            ax.set_zlim(*zlim)
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            ax.view_init(elev=15, azim=210)  # View from -x with y+ on the left.
            ax.set_title(f"Frame {i}")
            fig_m.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.95)

            for part in parts:
                pos = frame['positions'][part]
                axes_ends = frame['axes'][part]
                c = colors[part]
                ax.scatter(*pos, color=c, s=60, zorder=5, label=f"{part} position")
                for axis_end, ac, ax_name in zip(axes_ends, axis_colors, ('x', 'y', 'z')):
                    ax.plot([pos[0], axis_end[0]], [pos[1], axis_end[1]], [pos[2], axis_end[2]],
                            color=ac, linewidth=2, label=f"{part}_{ax_name}_axis")

            for side, c in (('right', colors['right_wrist']), ('left', colors['left_wrist'])):
                pts = frame['fingers'][side]
                ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], color=c, s=15, alpha=0.7,
                           label=f"{side}_fingers")

            if 'spine' in frame:
                spine = frame['spine']
                ax.plot(spine[:, 0], spine[:, 1], spine[:, 2],
                        color='black', linewidth=3, marker='o', markersize=3,
                        label='spine')

            if 'body_links' in frame:
                for name, points in frame['body_links']:
                    c = 'purple' if name == 'head_tracker_to_head' else 'black'
                    ax.plot(points[:, 0], points[:, 1], points[:, 2],
                            color=c, linewidth=3, marker='o', markersize=3,
                            label=name)

            ax.legend(loc='upper left', fontsize=6, markerscale=0.8)

            frame_path = os.path.join(tmpdir, f"frame_{i:06d}.png")
            fig_m.savefig(frame_path, dpi=dpi)
            plt.close(fig_m)

        _encode_png_sequence_to_mp4(tmpdir, out_path, fps=fps)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Plot processed data from HDF5 file')
    parser.add_argument('--file', '-f', type=str,
                        default="/home/aigc/human_policy/data/dex5_val/G1_WB_Dex5_Pickup_Pillow_ep_0093_rootfix.hdf5",
                        help='Path to the processed/predicted HDF5 file (must contain dataset: action)')
    parser.add_argument('--predict_episode', type=str, default=None,
                        help='If set, run policy on this episode HDF5 and write predictions to --out, then visualize --out')
    parser.add_argument('--out', type=str, default="/home/aigc/human-policy/data/predicted_episode.hdf5",
                        help='Output HDF5 path for predicted actions')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to model checkpoint state_dict (e.g., .../pytorch_model.bin or policy_last.ckpt)')
    parser.add_argument('--model_cfg', type=str, default="/home/aigc/human_policy/hdt/configs/models/act_resnet.yaml",
                        help='Path to model yaml (ACT config)')
    parser.add_argument('--chunk_size', type=int, default=100, help='Chunk size (num_queries) used by ACT')
    parser.add_argument('--camera', type=str, default="left", help='Camera name inside episode hdf5: observation.image.<camera>')
    parser.add_argument('--max_steps', type=int, default=None, help='Limit number of steps to predict')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu (default auto)')
    parser.add_argument('--full_hand', action='store_true', help='Expand each hand from 5 fingertip keypoints to an approximate 25-joint chain for visualization')
    parser.add_argument('--spine', action='store_true', help='Draw an approximate waist-to-head spine when action[89:98] contains waist pose data')
    parser.add_argument('--save_html', type=str, default=None, help='If set, save the interactive visualization to an HTML file')
    parser.add_argument('--save_mp4', type=str, default=None, help='If set, export the animation to an MP4 file (requires kaleido + ffmpeg)')
    parser.add_argument('--fps', type=int, default=20, help='FPS for MP4 export')
    parser.add_argument('--max_seconds', type=float, default=None, help='Limit visualization/export to the first N seconds, using --fps to convert seconds to frames')
    parser.add_argument('--start_step', type=int, default=0, help='Skip the first N timesteps when visualizing and evaluating MPJPE')
    parser.add_argument('--width', type=int, default=960, help='Frame width for MP4 export')
    parser.add_argument('--height', type=int, default=720, help='Frame height for MP4 export')
    parser.add_argument('--eval_mpjpe', action='store_true', help='Evaluate MPJPE between prediction and ground-truth episode actions')
    parser.add_argument('--gt_file', type=str, default=None, help='Ground-truth episode file for MPJPE. Default: --predict_episode')
    parser.add_argument('--metrics_out', type=str, default=None, help='Optional output JSON path for metrics')
    parser.add_argument('--compare_with_gt', action='store_true',
                        help='Visualize GT and predicted motion together. With --predict_episode, GT is --predict_episode and prediction is --out. Without --predict_episode, use --gt_file as GT and --file as prediction.')
    parser.add_argument('--gt_dir', type=str, default=None,
                        help='Directory of GT HDF5 files for batch MPJPE evaluation. If set with --ckpt, runs policy inference on each file then evaluates.')
    parser.add_argument('--pred_dir', type=str, default=None,
                        help='Directory of predicted HDF5 files for batch MPJPE evaluation (without --ckpt).')
    parser.add_argument('--glob', type=str, default='*.hdf5',
                        help='Glob pattern for matching HDF5 files inside --gt_dir / --pred_dir (default: *.hdf5)')
    parser.add_argument('--batch_out_json', type=str, default=None,
                        help='Path to save the per-file + summary MPJPE JSON from batch evaluation.')
    parser.add_argument('--norm_stats', type=str, default=None,
                        help='Path to dataset_stats.pkl (auto-resolved next to --ckpt when omitted)')
    parser.add_argument('--dirty_start_check_frames', type=int, default=20,
                        help='First N frames to scan for early-jump dirty-start detection (default 20)')
    parser.add_argument('--dirty_start_jump_threshold_m', type=float, default=0.3,
                        help='Position jump threshold in meters for dirty-start detection (default 0.3)')
    parser.add_argument('--dirty_start_settle_frames', type=int, default=0,
                        help='Extra frames to skip after a dirty-start jump (default 0)')
    args = parser.parse_args()
    max_frames = None
    if args.max_seconds is not None:
        if args.max_seconds <= 0:
            raise ValueError("--max_seconds must be positive")
        max_frames = int(np.ceil(args.max_seconds * args.fps))

    compare_frames = None

    if args.gt_dir is not None:
        batch_evaluate_mpjpe(
            args.gt_dir,
            pred_dir=args.pred_dir,
            ckpt=args.ckpt,
            norm_stats=args.norm_stats,
            model_cfg=args.model_cfg,
            chunk_size=args.chunk_size,
            camera=args.camera,
            device=args.device,
            glob=args.glob,
            out_json=args.batch_out_json,
            dirty_start_check_frames=args.dirty_start_check_frames,
            dirty_start_jump_threshold_m=args.dirty_start_jump_threshold_m,
            dirty_start_settle_frames=args.dirty_start_settle_frames,
        )
    elif args.predict_episode is not None:
        if args.ckpt is None:
            raise ValueError("--ckpt is required when --predict_episode is set")
        predict_episode_to_hdf5(
            args.predict_episode,
            args.out,
            args.ckpt,
            args.model_cfg,
            chunk_size=args.chunk_size,
            camera_name=args.camera,
            max_steps=args.max_steps,
            device=args.device,
        )
        if args.compare_with_gt:
            gt_frames = _frames_from_hdf5(args.predict_episode, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
            pred_frames = _frames_from_hdf5(args.out, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
            compare_frames = {"kind": "compare", "gt": gt_frames, "pred": pred_frames, "labels": {"gt": "GT", "pred": "Pred"}}
        fig, frames = main(args.out, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
        if args.eval_mpjpe:
            gt_file = args.gt_file if args.gt_file is not None else args.predict_episode
            metrics = evaluate_mpjpe(
                gt_file, args.out, save_json_path=args.metrics_out, start_step=args.start_step,
                _dirty_start_check_frames=args.dirty_start_check_frames,
                _dirty_start_jump_threshold_m=args.dirty_start_jump_threshold_m,
                _dirty_start_settle_frames=args.dirty_start_settle_frames,
            )
            print("===== MPJPE Metrics =====")
            for k, v in metrics.items():
                print(f"{k}: {v}")
    else:
        if args.compare_with_gt:
            if args.gt_file is None:
                raise ValueError("--gt_file is required when using --compare_with_gt without --predict_episode")
            gt_frames = _frames_from_hdf5(args.gt_file, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
            pred_frames = _frames_from_hdf5(args.file, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
            compare_frames = {"kind": "compare", "gt": gt_frames, "pred": pred_frames, "labels": {"gt": "GT", "pred": "Pred"}}
        fig, frames = main(args.file, full_hand=args.full_hand, max_frames=max_frames, draw_spine=args.spine, start_step=args.start_step)
        if args.eval_mpjpe:
            if args.gt_file is None:
                raise ValueError("--gt_file is required when using --eval_mpjpe without --predict_episode")
            metrics = evaluate_mpjpe(
                args.gt_file, args.file, save_json_path=args.metrics_out, start_step=args.start_step,
                _dirty_start_check_frames=args.dirty_start_check_frames,
                _dirty_start_jump_threshold_m=args.dirty_start_jump_threshold_m,
                _dirty_start_settle_frames=args.dirty_start_settle_frames,
            )
            print("===== MPJPE Metrics =====")
            for k, v in metrics.items():
                print(f"{k}: {v}")

    if args.gt_dir is None:
        if args.save_html is not None:
            _save_html(fig, args.save_html)
        if args.save_mp4 is not None:
            _save_mp4(fig, args.save_mp4, fps=args.fps, width=args.width, height=args.height, frames=compare_frames or frames)
        if args.save_html is None and args.save_mp4 is None:
            fig.show()
