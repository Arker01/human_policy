#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

# isaacgym must be imported before any torch import, so do it here at the top.
import isaacgym  # noqa: F401

import numpy as np

_HP_ROOT = Path(__file__).resolve().parent
_TWIST_ROOT = _HP_ROOT.parent / "TWIST"
_HDT_DIR = _HP_ROOT / "hdt"
_DETR_DIR = _HDT_DIR / "detr"

for _p in (_HP_ROOT, _HDT_DIR, _DETR_DIR, _TWIST_ROOT / "legged_gym", _TWIST_ROOT / "rsl_rl", _TWIST_ROOT / "pose"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _mat_to_quat_xyzw(mat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_matrix(mat).as_quat().astype(np.float32)


def _quat_fix_sign(q: np.ndarray) -> np.ndarray:
    q = q.copy()
    for i in range(1, len(q)):
        if np.dot(q[i - 1], q[i]) < 0:
            q[i] *= -1.0
    return q


def _quat_ang_vel_xyzw(quat: np.ndarray, fps: float) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    if len(quat) <= 1:
        return np.zeros((len(quat), 3), dtype=np.float32)
    q = _quat_fix_sign(quat)
    r = Rotation.from_quat(q)
    rv = r.as_rotvec()
    vel = np.gradient(rv, 1.0 / fps, axis=0)
    return vel.astype(np.float32)


def _interp_np(values: np.ndarray, frame_f: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    T = values.shape[0]
    frame_f = np.clip(frame_f.astype(np.float32), 0.0, float(max(T - 1, 0)))
    lo = np.floor(frame_f).astype(np.int64)
    hi = np.clip(lo + 1, 0, T - 1)
    a = (frame_f - lo).reshape((-1,) + (1,) * (values.ndim - 1))
    return values[lo] * (1.0 - a) + values[hi] * a


def _sample_quat_nearest(quat: np.ndarray, frame_f: np.ndarray) -> np.ndarray:
    idx = np.rint(np.clip(frame_f, 0.0, float(max(len(quat) - 1, 0)))).astype(np.int64)
    return quat[idx].astype(np.float32)


def _actions_to_hand_refs(actions: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    import hdt.constants as C

    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] < 89:
        raise ValueError(f"Expected actions shaped (T, >=89), got {actions.shape}")

    l_wrist = actions[:, C.OUTPUT_LEFT_EEF]
    r_wrist = actions[:, C.OUTPUT_RIGHT_EEF]
    l_local6 = actions[:, C.OUTPUT_LEFT_KEYPOINTS].reshape(actions.shape[0], 6, 3)
    r_local6 = actions[:, C.OUTPUT_RIGHT_KEYPOINTS].reshape(actions.shape[0], 6, 3)

    l_root_pos = l_wrist[:, :3].astype(np.float32)
    r_root_pos = r_wrist[:, :3].astype(np.float32)
    l_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in l_wrist], axis=0)
    r_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in r_wrist], axis=0)
    l_root_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in l_rot_m], axis=0))
    r_root_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in r_rot_m], axis=0))

    # RETARGETTING_INDICES = [palm, thumb, index, middle, ring, pinky].
    l_tip_local = l_local6[:, 1:6, :].astype(np.float32)
    r_tip_local = r_local6[:, 1:6, :].astype(np.float32)
    l_tip_world = l_root_pos[:, None, :] + np.einsum("tij,tkj->tki", l_rot_m, l_tip_local)
    r_tip_world = r_root_pos[:, None, :] + np.einsum("tij,tkj->tki", r_rot_m, r_tip_local)

    return {
        "lh_root_pos": l_root_pos,
        "lh_root_rot": l_root_rot.astype(np.float32),
        "lh_tip_world": l_tip_world.astype(np.float32),
        "rh_root_pos": r_root_pos,
        "rh_root_rot": r_root_rot.astype(np.float32),
        "rh_tip_world": r_tip_world.astype(np.float32),
        "lh_root_vel": np.gradient(l_root_pos, 1.0 / fps, axis=0).astype(np.float32),
        "rh_root_vel": np.gradient(r_root_pos, 1.0 / fps, axis=0).astype(np.float32),
        "lh_root_ang_vel": _quat_ang_vel_xyzw(l_root_rot, fps),
        "rh_root_ang_vel": _quat_ang_vel_xyzw(r_root_rot, fps),
        "length_s": np.float32(max(actions.shape[0] - 1, 1) / fps),
        "fps": np.float32(fps),
    }


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def _rot6d_to_mat_np(rot6d: np.ndarray) -> np.ndarray:
    a1 = rot6d[0:3]
    a2 = rot6d[3:6]
    b1 = _normalize(a1)
    b2 = _normalize(a2 - np.dot(b1, a2) * b1)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1).astype(np.float32)


# Isaac Gym add_lines has no line-width param; draw each segment with tiny
# perpendicular offsets to simulate thickness.
_LINE_OFFSETS = np.array([
    [ 0.000,  0.000, 0.000],
    [ 0.003,  0.000, 0.000],
    [ 0.000,  0.003, 0.000],
    [ 0.000,  0.000, 0.003],
    [-0.003,  0.000, 0.000],
    [ 0.000, -0.003, 0.000],
], dtype=np.float32)


def _build_skeleton_lines(rh_w, lh_w, rh_tips, lh_tips):
    """Return (verts, colors) for add_lines: wrist→5 fingertips, both hands, with thickness."""
    base_segs = [(rh_w, tip) for tip in rh_tips] + [(lh_w, tip) for tip in lh_tips]
    all_segs = [[p0 + d, p1 + d] for d in _LINE_OFFSETS for (p0, p1) in base_segs]
    verts = np.array(all_segs, dtype=np.float32).reshape(-1, 3)
    colors = np.tile(np.array([[0., 1., 0.]], dtype=np.float32), (verts.shape[0], 1))
    return len(all_segs), verts, colors


class HumanPolicyHandMotionLib:
    """Small TWIST-compatible motion lib backed by a human-policy action sequence."""

    def __init__(self, refs: dict[str, np.ndarray], *, device: str):
        self.refs = refs
        self.device = device
        self.env = None

    def attach_env(self, env) -> None:
        self.env = env

    def num_motions(self) -> int:
        return 1

    def sample_motions(self, n: int):
        import torch

        return torch.zeros((n,), device=self.device, dtype=torch.long)

    def sample_time(self, motion_ids):
        import torch

        return torch.zeros_like(motion_ids, dtype=torch.float)

    def get_motion_length(self, motion_ids):
        import torch

        if not torch.is_tensor(motion_ids):
            return torch.tensor(float(self.refs["length_s"]), device=self.device, dtype=torch.float)
        return torch.full(motion_ids.shape, float(self.refs["length_s"]), device=self.device, dtype=torch.float)

    def calc_motion_frame(self, motion_ids, motion_times):
        import torch

        if self.env is None:
            raise RuntimeError("HumanPolicyHandMotionLib must be attached to the env before rollout.")

        times = motion_times.detach().float().cpu().numpy()
        frame_f = times * float(self.refs["fps"])
        n = int(len(frame_f))

        zeros3 = np.zeros((n, 3), dtype=np.float32)
        zeros4 = np.zeros((n, 4), dtype=np.float32)
        zeros_dof = np.zeros((n, self.env.num_dexhand_rh_dofs), dtype=np.float32)
        zeros_forces = np.zeros((n, len(self.env._key_body_ids), 3), dtype=np.float32)
        obj_rot = np.tile(np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32), (n, 1))

        def side(prefix: str):
            root_pos = _interp_np(self.refs[f"{prefix}_root_pos"], frame_f)
            root_rot = _sample_quat_nearest(self.refs[f"{prefix}_root_rot"], frame_f)
            root_vel = _interp_np(self.refs[f"{prefix}_root_vel"], frame_f)
            root_ang_vel = _interp_np(self.refs[f"{prefix}_root_ang_vel"], frame_f)

            body_pos = np.repeat(root_pos[:, None, :], self.env.rh_body_states.shape[1], axis=1)
            body_rot = np.repeat(root_rot[:, None, :], self.env.rh_body_states.shape[1], axis=1)
            body_vel = np.zeros_like(body_pos, dtype=np.float32)
            body_ang_vel = np.zeros_like(body_pos, dtype=np.float32)

            tips = _interp_np(self.refs[f"{prefix}_tip_world"], frame_f)
            key_ids = self.env._key_body_ids.detach().cpu().numpy().astype(np.int64)
            for tip_i, body_i in enumerate(key_ids[: tips.shape[1]]):
                body_pos[:, body_i, :] = tips[:, tip_i, :]

            return (
                root_pos,
                root_rot,
                root_vel,
                root_ang_vel,
                zeros_dof,
                zeros_dof,
                body_pos.astype(np.float32),
                body_rot.astype(np.float32),
                body_vel,
                body_ang_vel,
                zeros_forces,
                zeros3,
                obj_rot,
                zeros3,
                zeros3,
            )

        def to_torch(data):
            return tuple(torch.as_tensor(x, device=self.device, dtype=torch.float32) for x in data)

        return to_torch(side("rh")), to_torch(side("lh"))


def _select_episode(gt_dir: str | None, episode_hdf5: str | None, glob_pat: str) -> Path:
    if episode_hdf5:
        return Path(episode_hdf5)
    if not gt_dir:
        raise ValueError("Provide either --episode-hdf5 or --gt-dir")
    files = sorted(Path(gt_dir).glob(glob_pat))
    if not files:
        raise FileNotFoundError(f"No files matched {glob_pat!r} in {gt_dir}")
    return files[0]


def _load_or_predict_actions(args) -> np.ndarray:
    import h5py
    from data.eval_mpjpe_batch import _make_policy, _predict_actions_for_episode

    if args.pred_hdf5:
        with h5py.File(args.pred_hdf5, "r") as f:
            return f["action"][()].astype(np.float32)

    episode = _select_episode(args.gt_dir, args.episode_hdf5, args.glob)
    if args.use_gt_actions:
        with h5py.File(episode, "r") as f:
            actions = f["action"][()].astype(np.float32)
        return actions[: args.max_steps] if args.max_steps else actions

    if not (args.policy_ckpt and args.policy_config_yaml and args.norm_stats):
        raise ValueError("Need --pred-hdf5, --use-gt-actions, or all of --policy-ckpt/--policy-config-yaml/--norm-stats")

    with h5py.File(episode, "r") as f:
        emb = f.attrs.get("embodiment", None)
        if isinstance(emb, bytes):
            emb = emb.decode()
        emb = str(emb) if emb is not None else None

    policy, pol_info = _make_policy(Path(args.policy_config_yaml), Path(args.policy_ckpt), device=args.device)
    stats = _load_norm_stats_compat(Path(args.norm_stats), emb)
    return _predict_actions_for_episode(
        policy,
        episode,
        stats=stats,
        camera_names=pol_info["camera_names"],
        device=args.device,
        max_steps=args.max_steps,
        eval_mode=args.eval_mode,
        chunk_stride=args.chunk_stride,
        temporal_decay=args.temporal_decay,
    )


def _to_numpy_compat(x) -> np.ndarray:
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _load_norm_stats_compat(stats_path: Path, embodiment: str | None) -> dict:
    import pickle

    class NumpyCompatUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if module.startswith("numpy._core"):
                module = module.replace("numpy._core", "numpy.core", 1)
            return super().find_class(module, name)

    with open(stats_path, "rb") as f:
        stats = NumpyCompatUnpickler(f).load()
    if isinstance(stats, dict) and "qpos_mean" in stats:
        return {k: _to_numpy_compat(v) for k, v in stats.items()}
    if isinstance(stats, dict) and embodiment is not None and embodiment in stats:
        return {k: _to_numpy_compat(v) for k, v in stats[embodiment].items()}
    if isinstance(stats, dict) and embodiment is not None:
        emb = embodiment.encode() if isinstance(next(iter(stats.keys())), bytes) else embodiment
        if emb in stats:
            return {k: _to_numpy_compat(v) for k, v in stats[emb].items()}
    if isinstance(stats, dict):
        first = next(iter(stats.values()))
        if isinstance(first, dict):
            return {k: _to_numpy_compat(v) for k, v in first.items()}
    raise ValueError(f"Unsupported stats format: {type(stats)}")


def _save_refs_npz(path: str, refs: dict[str, np.ndarray]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **refs)
    print(f"Saved hand GMT refs: {out}")


def _load_refs_npz(path: str) -> dict[str, np.ndarray]:
    data = np.load(path)
    return {k: data[k] for k in data.files}


def _twist_args(args):
    from isaacgym import gymapi

    device = args.gmt_device
    sim_device = device
    if device.startswith("cuda:"):
        sim_device = device
    return SimpleNamespace(
        task="dexhand_mimic_direct",
        proj_name=args.hand_gmt_proj_name,
        exptid=args.hand_gmt_exptid,
        resumeid=None,
        teacher_exptid="mimic",
        teacher_checkpoint=-1,
        eval_student=False,
        experiment_name=None,
        run_name=None,
        load_run=None,
        checkpoint=args.hand_gmt_checkpoint,
        resume=False,
        fix_action_std=False,
        num_envs=1,
        seed=args.seed,
        rows=None,
        cols=None,
        record_video=args.record_video,
        record_log=False,
        teleop_mode=False,
        no_rand=False,
        max_iterations=None,
        rl_device=device,
        device=device,
        sim_device=sim_device,
        headless=not args.viewer,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=device.startswith("cuda"),
        use_gpu_pipeline=device.startswith("cuda"),
        subscenes=0,
        num_threads=0,
    )


def _load_hand_policy(env, train_cfg, twist_args, args):
    import torch
    from legged_gym.gym_utils import task_registry
    from legged_gym.gym_utils.helpers import get_load_path
    import re

    runner, _ = task_registry.make_alg_runner(
        log_root=None,
        env=env,
        name="dexhand_mimic_direct",
        args=twist_args,
        train_cfg=train_cfg,
        init_wandb=False,
    )

    ckpt_path = args.hand_gmt_ckpt
    if ckpt_path is None:
        log_root = Path(args.hand_gmt_log_root) if args.hand_gmt_log_root else _TWIST_ROOT / "legged_gym" / "logs" / args.hand_gmt_proj_name / args.hand_gmt_exptid
        ckpt_path = get_load_path(str(log_root), checkpoint=args.hand_gmt_checkpoint)

    ckpt_path = Path(ckpt_path)
    if not ckpt_path.name.startswith("model_"):
        match = re.search(r"model_(\d+)\.pt$", ckpt_path.name)
        if match is None:
            raise ValueError(f"Hand GMT checkpoint must be model_<iter>.pt or end with model_<iter>.pt: {ckpt_path}")
        alias_dir = _HP_ROOT / "outputs" / "twist_hand_gmt_ckpt_alias"
        alias_dir.mkdir(parents=True, exist_ok=True)
        alias_path = alias_dir / f"model_{match.group(1)}.pt"
        if alias_path.exists() or alias_path.is_symlink():
            alias_path.unlink()
        alias_path.symlink_to(ckpt_path)
        ckpt_path = alias_path

    ckpt_path = str(ckpt_path)
    print(f"Loading hand GMT checkpoint: {ckpt_path}")
    runner.load(ckpt_path)
    policy = runner.get_inference_policy(device=env.device)

    normalizer = None
    if env.cfg.env.normalize_obs:
        try:
            normalizer = runner.get_normalizer(device=env.device)
        except Exception:
            print("Warning: no GMT normalizer found; using raw observations")

    return policy, normalizer


def _run_gt_only(args, env, refs, rh_wrist_seq, lh_wrist_seq,
                 rh_tips_seq, lh_tips_seq, rh_rot_seq, lh_rot_seq,
                 actions_128, has_viewer, env_ptr):
    """Visualize and optionally record the GT reference skeleton without the dexhand."""
    import time as _time
    import imageio
    import os
    import torch
    from isaacgym import gymapi, gymtorch
    import hdt.constants as C

    total_steps = len(rh_wrist_seq)

    # Hide both inspire hands by moving them far below the scene.
    with torch.no_grad():
        far = torch.tensor([0., 0., -200.], device=env.device)
        env.rh_root_states[:, :3] = far
        env.lh_root_states[:, :3] = far
        eids = torch.arange(env.num_envs, device=env.device)
        ids = torch.cat([env.rh_env_ids[eids].flatten(),
                         env.lh_env_ids[eids].flatten()]).to(torch.int32)
        env.gym.set_actor_root_state_tensor_indexed(
            env.sim, gymtorch.unwrap_tensor(env.root_states),
            gymtorch.unwrap_tensor(ids), len(ids))
    env.gym.simulate(env.sim)
    env.gym.fetch_results(env.sim, True)

    # Position viewer camera: head is origin [0,0,0].
    # Camera at (-0.1, 0, 0.25), looking at mean hand position.
    if has_viewer:
        mean_hand = ((rh_wrist_seq.mean(0) + lh_wrist_seq.mean(0)) / 2).astype(np.float32)
        cam_pos = gymapi.Vec3(-0.1, 0.0, 0.25)
        cam_target = gymapi.Vec3(float(mean_hand[0]),
                                 float(mean_hand[1]),
                                 float(mean_hand[2]))
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
        print(f"[gt_only] camera pos=[-0.1, 0.0, 0.25]  "
              f"target={list(np.round([cam_target.x, cam_target.y, cam_target.z], 3))}")

    # Video writer via viewer screenshot.
    writer = None
    tmp_png = None
    if args.out_gt_video:
        out_path = Path(args.out_gt_video)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(out_path), fps=int(args.ref_fps))
        tmp_png = str(out_path.parent / "_gt_tmp_frame.png")
        print(f"[gt_only] recording {total_steps} frames → {out_path}")

    for step_i in range(total_steps):
        if step_i % 20 == 0:
            rh_q = rh_rot_seq[step_i]   # xyzw, from refs (6D→mat→quat, no extra transform)
            lh_q = lh_rot_seq[step_i]
            print(f"[step {step_i:4d}] rh_wrist_pos={np.round(rh_wrist_seq[step_i], 4)}  "
                  f"lh_wrist_pos={np.round(lh_wrist_seq[step_i], 4)}")
            print(f"           rh_rot refs(xyzw)={np.round(rh_q, 4)}")
            print(f"           lh_rot refs(xyzw)={np.round(lh_q, 4)}")
            if actions_128 is not None and step_i < len(actions_128):
                rh_6d_raw = actions_128[step_i, C.OUTPUT_RIGHT_EEF[3:]]  # raw 6D from HDF5
                lh_6d_raw = actions_128[step_i, C.OUTPUT_LEFT_EEF[3:]]
                rh_q_raw  = _mat_to_quat_xyzw(_rot6d_to_mat_np(rh_6d_raw))
                lh_q_raw  = _mat_to_quat_xyzw(_rot6d_to_mat_np(lh_6d_raw))
                print(f"           rh_6d  HDF5 raw ={np.round(rh_6d_raw, 4)}  -> quat={np.round(rh_q_raw, 4)}")
                print(f"           lh_6d  HDF5 raw ={np.round(lh_6d_raw, 4)}  -> quat={np.round(lh_q_raw, 4)}")

        if has_viewer:
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_skeleton_lines(
                rh_wrist_seq[step_i], lh_wrist_seq[step_i],
                rh_tips_seq[step_i],  lh_tips_seq[step_i],
            )
            env.gym.add_lines(env.viewer, env_ptr, n, verts, colors)
            env.gym.step_graphics(env.sim)
            env.gym.draw_viewer(env.viewer, env.sim, True)

            if writer is not None:
                env.gym.write_viewer_image_to_file(env.viewer, tmp_png)
                img = imageio.imread(tmp_png)
                writer.append_data(img[:, :, :3] if img.shape[-1] == 4 else img)

        if args.step_delay > 0:
            _time.sleep(args.step_delay)

    if writer is not None:
        writer.close()
        if tmp_png and os.path.exists(tmp_png):
            os.unlink(tmp_png)
        print(f"[gt_only] saved: {args.out_gt_video}")


def run(args) -> None:
    if args.gt_only:
        args.use_gt_actions = True  # gt-only always uses GT, never runs policy
    if args.ref_npz:
        refs = _load_refs_npz(args.ref_npz)
        actions_128 = np.zeros((0, 128), dtype=np.float32)
    else:
        actions_128 = _load_or_predict_actions(args)
        refs = _actions_to_hand_refs(actions_128, fps=args.ref_fps)
        if args.dump_ref_npz:
            _save_refs_npz(args.dump_ref_npz, refs)
            if args.skip_gmt:
                return

    # TWIST should use dependencies from the active TWIST environment. In
    # particular, do not let human_policy/pytorch3d shadow official PyTorch3D.
    for shadow_path in (str(_HP_ROOT), str(_HDT_DIR), str(_DETR_DIR)):
        while shadow_path in sys.path:
            sys.path.remove(shadow_path)

    import torch
    import imageio

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    from legged_gym.envs import task_registry
    from legged_gym.envs.base import dexhand_mimic as dexhand_mimic_mod

    motion_lib = HumanPolicyHandMotionLib(refs, device=args.gmt_device)

    original_load_motions = dexhand_mimic_mod.DexHandMimic._load_motions

    def _load_human_policy_motion(self):
        self._motion_lib = motion_lib
        motion_lib.attach_env(self)

    dexhand_mimic_mod.DexHandMimic._load_motions = _load_human_policy_motion
    try:
        twist_args = _twist_args(args)
        env_cfg, train_cfg = task_registry.get_cfgs(name="dexhand_mimic_direct")
        env_cfg.env.num_envs = 1
        env_cfg.env.rand_reset = False
        env_cfg.env.training = False
        env_cfg.env.play = False
        env_cfg.noise.add_noise = False
        env_cfg.domain_rand.domain_rand_general = False
        env_cfg.domain_rand.randomize_friction = False
        env_cfg.domain_rand.push_robots = False
        env_cfg.domain_rand.action_delay = False
        env_cfg.env.record_video = args.record_video
        # Our motion lib returns zeros_dof (no real joint angles from human policy),
        # so the physics-simulated finger tips won't match the reference at init.
        # Disable pose_termination to avoid immediate termination from this mismatch.
        env_cfg.env.pose_termination = False
        # Use a simple plane instead of trimesh to skip the 5s ground-mesh build
        # and remove the table-like terrain surface in the viewer.
        if not args.show_table:
            env_cfg.terrain.mesh_type = None  # no ground at all, avoids finger-ground collisions

        env, _ = task_registry.make_env(name="dexhand_mimic_direct", args=twist_args, env_cfg=env_cfg)
        env.reset_idx(torch.arange(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
        obs = env.get_observations()

        policy, normalizer = _load_hand_policy(env, train_cfg, twist_args, args)

        total_steps = args.rollout_steps
        if total_steps is None:
            total_steps = int(float(refs["length_s"]) / env.dt)
        total_steps = min(total_steps, int(env.max_episode_length))
        print(f"[rollout] motion_length={refs['length_s']:.2f}s  dt={env.dt:.4f}s  "
              f"max_episode_length={int(env.max_episode_length)}  planned_steps={total_steps}")

        writer = None
        if args.record_video:
            out_video = Path(args.out_video)
            out_video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(out_video), fps=args.video_fps)

        # Pre-compute per-step ref wrist positions and tip world positions for viz.
        ref_t = np.arange(total_steps, dtype=np.float32) / float(refs["fps"])
        ref_frame_f = ref_t * float(refs["fps"])
        rh_wrist_seq = _interp_np(refs["rh_root_pos"], ref_frame_f)            # (T,3)
        lh_wrist_seq = _interp_np(refs["lh_root_pos"], ref_frame_f)            # (T,3)
        rh_tips_seq  = _interp_np(refs["rh_tip_world"], ref_frame_f)           # (T,5,3)
        lh_tips_seq  = _interp_np(refs["lh_tip_world"], ref_frame_f)           # (T,5,3)
        rh_rot_seq   = _sample_quat_nearest(refs["rh_root_rot"], ref_frame_f)  # (T,4) xyzw
        lh_rot_seq   = _sample_quat_nearest(refs["lh_root_rot"], ref_frame_f)  # (T,4) xyzw

        # Print frame-0 ref vs env wrist to check coordinate alignment.
        print(f"[ref frame0] rh_wrist_pos={rh_wrist_seq[0]}  lh_wrist_pos={lh_wrist_seq[0]}")
        print(f"[env frame0] rh_root_pos ={env.rh_root_states[0, :3].cpu().numpy()}  "
              f"lh_root_pos={env.lh_root_states[0, :3].cpu().numpy()}")

        has_viewer = args.viewer and env.viewer is not None
        _env_ptr = env.envs[0]

        def _draw_ref_skeleton(step_i: int) -> None:
            if not has_viewer:
                return
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_skeleton_lines(
                rh_wrist_seq[step_i], lh_wrist_seq[step_i],
                rh_tips_seq[step_i], lh_tips_seq[step_i],
            )
            env.gym.add_lines(env.viewer, _env_ptr, n, verts, colors)

        import time as _time

        # ── GT-only mode: hide the dexhands, record from head-position camera ──
        if args.gt_only:
            _run_gt_only(args, env, refs, rh_wrist_seq, lh_wrist_seq,
                         rh_tips_seq, lh_tips_seq, rh_rot_seq, lh_rot_seq,
                         actions_128 if len(actions_128) > 0 else None,
                         has_viewer, _env_ptr)
            return

        action_log = []
        for step_i in range(total_steps):
            _draw_ref_skeleton(step_i)

            # Periodic ref vs env wrist position print for coordinate-frame debugging.
            if step_i % 20 == 0:
                print(f"[step {step_i:4d}] ref rh_wrist={rh_wrist_seq[step_i]}  "
                      f"env rh_root={env.rh_root_states[0, :3].cpu().numpy()}")
                print(f"           ref lh_wrist={lh_wrist_seq[step_i]}  "
                      f"env lh_root={env.lh_root_states[0, :3].cpu().numpy()}")

            with torch.no_grad():
                pol_obs = normalizer.normalize(obs.detach()) if normalizer is not None else obs.detach()
                gmt_action = policy(pol_obs, hist_encoding=True)
            action_log.append(gmt_action.detach().cpu().numpy()[0].astype(np.float32))
            obs, _, _, done, _ = env.step(gmt_action.detach())

            if args.step_delay > 0:
                _time.sleep(args.step_delay)

            if writer is not None and step_i % args.video_stride == 0:
                imgs = env.render_record(mode="rgb_array")
                if imgs is not None and len(imgs) > 0:
                    writer.append_data(imgs[0])
            if bool(done[0].item()) and step_i > 1:
                # diagnose termination reason
                reasons = []
                rh_vel = torch.norm(env.rh_root_states[0, 7:10]).item()
                lh_vel = torch.norm(env.lh_root_states[0, 7:10]).item()
                if rh_vel > 5.0:
                    reasons.append(f"rh_vel_too_large={rh_vel:.2f}")
                if lh_vel > 5.0:
                    reasons.append(f"lh_vel_too_large={lh_vel:.2f}")
                if bool(env.time_out_buf[0].item()):
                    reasons.append("timeout/motion_end")
                if hasattr(env, '_pose_termination') and env._pose_termination:
                    # check finger tip distances if available
                    try:
                        rh_body_pos = env.rh_body_states[0, :, 0:3] - env.rh_body_states[0, 0:1, 0:3]
                        tar_rh = env._ref_rh_body_pos[0] - env._ref_rh_root_pos[0:1]
                        dists = torch.norm(tar_rh - rh_body_pos, dim=-1)
                        tip_ids = {
                            'thumb': env._thumb_tip_id,
                            'index': env._index_tip_id,
                            'middle': env._middle_tip_id,
                            'ring': env._ring_tip_id,
                            'pinky': env._pinky_tip_id,
                        }
                        for name, tid in tip_ids.items():
                            d = dists[tid].item()
                            thresh = getattr(env, f'_{name}_termination_dist', None)
                            if thresh is not None and d > thresh:
                                reasons.append(f"{name}_tip_dist={d:.3f}>{thresh:.3f}")
                    except Exception:
                        reasons.append("pose_termination(details unavailable)")
                if not reasons:
                    reasons.append("unknown")
                print(f"[rollout] done at step {step_i}/{total_steps}  reason: {', '.join(reasons)}")
                break
        else:
            print(f"[rollout] completed all {total_steps} steps normally")

        if writer is not None:
            writer.close()

        out_actions = Path(args.out_actions)
        out_actions.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_actions,
            gmt_actions=np.asarray(action_log, dtype=np.float32),
            human_policy_actions=actions_128.astype(np.float32),
            ref_fps=np.asarray(args.ref_fps, dtype=np.float32),
        )
        print(f"Saved GMT actions: {out_actions}")
        if args.record_video:
            print(f"Saved video: {args.out_video}")
    finally:
        dexhand_mimic_mod.DexHandMimic._load_motions = original_load_motions


def main() -> None:
    p = argparse.ArgumentParser(description="Run TWIST hand GMT from human_policy wrist/finger predictions.")
    p.add_argument("--pred-hdf5", type=str, default=None, help="HDF5 containing an action dataset; bypasses policy inference.")
    p.add_argument("--ref-npz", type=str, default=None, help="Precomputed hand reference npz from --dump-ref-npz; bypasses human-policy/HDF5 loading.")
    p.add_argument("--dump-ref-npz", type=str, default=None, help="Save extracted hand references before running GMT.")
    p.add_argument("--skip-gmt", action="store_true", help="With --dump-ref-npz, stop after writing references.")
    p.add_argument("--episode-hdf5", type=str, default=None, help="Single GT episode HDF5 used for human-policy inference or --use-gt-actions.")
    p.add_argument("--gt-dir", type=str, default=None, help="GT directory; first file matching --glob is used unless --episode-hdf5 is set.")
    p.add_argument("--glob", type=str, default="*.hdf5")
    p.add_argument("--use-gt-actions", action="store_true", help="Use episode action dataset instead of running human-policy.")
    p.add_argument("--policy-ckpt", type=str, default=None)
    p.add_argument("--policy-config-yaml", type=str, default=None)
    p.add_argument("--norm-stats", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda:0", help="Device for human-policy inference.")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-mode", choices=["first_token", "chunk_rollout", "temporal_agg"], default="first_token")
    p.add_argument("--chunk-stride", type=int, default=10)
    p.add_argument("--temporal-decay", type=float, default=0.01)

    p.add_argument("--hand-gmt-proj-name", type=str, default="dexhand_mimic_direct")
    p.add_argument("--hand-gmt-exptid", type=str, default="multi")
    p.add_argument("--hand-gmt-log-root", type=str, default=None)
    p.add_argument("--hand-gmt-ckpt", type=str, default=None, help="Direct path to a TWIST hand GMT model_*.pt checkpoint.")
    p.add_argument("--hand-gmt-checkpoint", type=int, default=-1)
    p.add_argument("--gmt-device", type=str, default="cuda:0")
    p.add_argument("--ref-fps", type=float, default=30.0)
    p.add_argument("--rollout-steps", type=int, default=None)
    p.add_argument("--record-video", action="store_true")
    p.add_argument("--viewer", action="store_true", help="Run Isaac Gym with viewer/graphics device. Requires a display or virtual display.")
    p.add_argument("--step-delay", type=float, default=0.0, help="Seconds to sleep after each rollout step (e.g. 0.3 to slow down viewer).")
    p.add_argument("--show-table", action="store_true", help="Keep the trimesh terrain (table surface). Default: no ground.")
    p.add_argument("--gt-only", action="store_true", help="Visualize GT reference skeleton only; skip inspire hand and GMT policy.")
    p.add_argument("--out-gt-video", type=str, default=None, help="Path to save GT-only skeleton video (requires --viewer).")
    p.add_argument("--out-video", type=str, default=str(_HP_ROOT / "outputs" / "twist_hand_gmt_bridge.mp4"))
    p.add_argument("--video-fps", type=int, default=30)
    p.add_argument("--video-stride", type=int, default=8)
    p.add_argument("--out-actions", type=str, default=str(_HP_ROOT / "outputs" / "twist_hand_gmt_bridge_actions.npz"))
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
