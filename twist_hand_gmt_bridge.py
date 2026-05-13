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
_INSPIRE_ASSET_ROOT = _TWIST_ROOT / "assets" / "inspire_hand"

for _p in (_HP_ROOT, _HDT_DIR, _DETR_DIR, _TWIST_ROOT / "legged_gym", _TWIST_ROOT / "rsl_rl", _TWIST_ROOT / "pose"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_INSPIRE_DOF_NAMES = [
    "index_proximal_joint",
    "index_intermediate_joint",
    "middle_proximal_joint",
    "middle_intermediate_joint",
    "pinky_proximal_joint",
    "pinky_intermediate_joint",
    "ring_proximal_joint",
    "ring_intermediate_joint",
    "thumb_proximal_yaw_joint",
    "thumb_proximal_pitch_joint",
    "thumb_intermediate_joint",
    "thumb_distal_joint",
]
_INSPIRE_TIP_NAMES = ["thumb_tip", "index_tip", "middle_tip", "ring_tip", "pinky_tip"]
_INSPIRE_DOF_LOWER = np.array([0., 0., 0., 0., 0., 0., 0., 0., -0.1, 0., 0., 0.], dtype=np.float32)
_INSPIRE_DOF_UPPER = np.array([1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.7, 1.3, 0.5, 0.8, 1.2], dtype=np.float32)


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


def _gradient_or_zeros(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) <= 1:
        return np.zeros_like(values, dtype=np.float32)
    return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


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
    head = actions[:, C.OUTPUT_HEAD_EEF]
    l_local6 = actions[:, C.OUTPUT_LEFT_KEYPOINTS].reshape(actions.shape[0], 6, 3)
    r_local6 = actions[:, C.OUTPUT_RIGHT_KEYPOINTS].reshape(actions.shape[0], 6, 3)

    head_pos = head[:, :3].astype(np.float32)
    head_rot_m = np.stack([_rot6d_to_mat_np(x[3:9]) for x in head], axis=0)
    head_rot = _quat_fix_sign(np.stack([_mat_to_quat_xyzw(m) for m in head_rot_m], axis=0))

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
    l_full_local = np.stack([_expand_hand_25_from_retarget6(x) for x in l_local6], axis=0)
    r_full_local = np.stack([_expand_hand_25_from_retarget6(x) for x in r_local6], axis=0)
    l_full_world = l_root_pos[:, None, :] + np.einsum("tij,tkj->tki", l_rot_m, l_full_local)
    r_full_world = r_root_pos[:, None, :] + np.einsum("tij,tkj->tki", r_rot_m, r_full_local)

    return {
        "head_pos": head_pos,
        "head_rot": head_rot.astype(np.float32),
        "head_forward": np.einsum("tij,j->ti", head_rot_m, np.array([1., 0., 0.], dtype=np.float32)).astype(np.float32),
        "head_axes": (head_pos[:, None, :] + 0.1 * np.transpose(head_rot_m, (0, 2, 1))).astype(np.float32),
        "lh_root_pos": l_root_pos,
        "lh_root_rot": l_root_rot.astype(np.float32),
        "lh_tip_world": l_tip_world.astype(np.float32),
        "lh_full_world": l_full_world.astype(np.float32),
        "rh_root_pos": r_root_pos,
        "rh_root_rot": r_root_rot.astype(np.float32),
        "rh_tip_world": r_tip_world.astype(np.float32),
        "rh_full_world": r_full_world.astype(np.float32),
        "lh_root_vel": _gradient_or_zeros(l_root_pos, fps),
        "rh_root_vel": _gradient_or_zeros(r_root_pos, fps),
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
    # Match pytorch3d.transforms.rotation_6d_to_matrix used by hdt.
    return np.stack([b1, b2, b3], axis=0).astype(np.float32)


def _expand_hand_25_from_retarget6(local6: np.ndarray) -> np.ndarray:
    """Approximate plot_keypoints full-hand mode from [palm, 5 fingertips]."""
    local6 = np.asarray(local6, dtype=np.float32)
    if local6.shape != (6, 3):
        raise ValueError(f"Expected retarget hand keypoints shape (6,3), got {local6.shape}")
    out = np.zeros((25, 3), dtype=np.float32)
    palm = local6[0]
    out[0] = palm
    alphas = np.linspace(0.0, 1.0, 5, dtype=np.float32)
    for finger in range(5):
        tip = local6[finger + 1]
        base = finger * 5
        for joint_i, a in enumerate(alphas):
            out[base + joint_i] = palm + a * (tip - palm)
    return out


def _inspire_urdf(side: str) -> Path:
    if side == "rh":
        return _INSPIRE_ASSET_ROOT / "inspire_hand_right.urdf"
    if side == "lh":
        return _INSPIRE_ASSET_ROOT / "inspire_hand_left.urdf"
    raise ValueError(f"Unknown hand side: {side}")


def _side_prefix(side: str) -> str:
    return "R_" if side == "rh" else "L_"


def _tips_world_to_local(root_pos: np.ndarray, root_rot_xyzw: np.ndarray, tips_world: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    root_m = Rotation.from_quat(root_rot_xyzw).as_matrix().astype(np.float32)
    return np.einsum("tji,tkj->tki", root_m, tips_world - root_pos[:, None, :]).astype(np.float32)


class _PytorchKinematicsInspireIK:
    def __init__(self, side: str, *, device: str, lr: float, iters: int, smooth_w: float, reg_w: float):
        import torch
        import pytorch_kinematics as pk

        self.torch = torch
        self.side = side
        self.prefix = _side_prefix(side)
        self.device = torch.device(device if device.startswith("cuda") and torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.iters = iters
        self.smooth_w = smooth_w
        self.reg_w = reg_w
        self.chain = pk.build_chain_from_urdf(_inspire_urdf(side).read_text())
        self.chain = self.chain.to(dtype=torch.float32, device=self.device)
        pk_names = [n.split("_", 1)[1] for n in self.chain.get_joint_parameter_names()]
        self.twist_to_pk = torch.tensor([_INSPIRE_DOF_NAMES.index(n) for n in pk_names], device=self.device, dtype=torch.long)
        self.lower = torch.as_tensor(_INSPIRE_DOF_LOWER, device=self.device, dtype=torch.float32)
        self.upper = torch.as_tensor(_INSPIRE_DOF_UPPER, device=self.device, dtype=torch.float32)
        self.tip_names = [self.prefix + name for name in _INSPIRE_TIP_NAMES]

    def _fk_tips(self, q_twist):
        q_pk = q_twist[:, self.twist_to_pk]
        ret = self.chain.forward_kinematics(q_pk)
        return self.torch.stack([ret[name].get_matrix()[:, :3, 3] for name in self.tip_names], dim=1)

    def solve(self, target_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        target = torch.as_tensor(target_local, device=self.device, dtype=torch.float32)
        q_prev = torch.zeros((1, len(_INSPIRE_DOF_NAMES)), device=self.device, dtype=torch.float32)
        out = []
        errs = []
        for t in range(target.shape[0]):
            q = q_prev.detach().clone().requires_grad_(True)
            opt = torch.optim.Adam([q], lr=self.lr)
            q_ref = q_prev.detach()
            for _ in range(self.iters):
                q_clamped = torch.max(torch.min(q, self.upper), self.lower)
                pred = self._fk_tips(q_clamped)
                tip_loss = torch.mean(torch.linalg.norm(pred[0] - target[t], dim=-1))
                smooth_loss = torch.mean((q_clamped - q_ref) ** 2)
                reg_loss = torch.mean(q_clamped ** 2)
                loss = tip_loss + self.smooth_w * smooth_loss + self.reg_w * reg_loss
                opt.zero_grad()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    q.clamp_(self.lower, self.upper)
            q_prev = torch.max(torch.min(q.detach(), self.upper), self.lower)
            err = torch.mean(torch.linalg.norm(self._fk_tips(q_prev)[0] - target[t], dim=-1))
            out.append(q_prev[0].detach().cpu().numpy().astype(np.float32))
            errs.append(float(err.detach().cpu()))
        return np.stack(out, axis=0), np.asarray(errs, dtype=np.float32)


class _PinocchioInspireIK:
    def __init__(self, side: str, *, iters: int, damping: float, step: float, smooth_w: float, reg_w: float):
        import pinocchio as pin

        self.pin = pin
        self.side = side
        self.prefix = _side_prefix(side)
        self.iters = iters
        self.damping = damping
        self.step = step
        self.smooth_w = smooth_w
        self.reg_w = reg_w
        self.model = pin.buildModelFromUrdf(str(_inspire_urdf(side)))
        self.data = self.model.createData()
        self.tip_ids = [self.model.getFrameId(self.prefix + name) for name in _INSPIRE_TIP_NAMES]
        self.lower = _INSPIRE_DOF_LOWER.astype(np.float64)
        self.upper = _INSPIRE_DOF_UPPER.astype(np.float64)

    def _fk_tips(self, q: np.ndarray) -> np.ndarray:
        pin = self.pin
        pin.framesForwardKinematics(self.model, self.data, q)
        tips = []
        for fid in self.tip_ids:
            tips.append(self.data.oMf[fid].translation.copy())
        return np.stack(tips, axis=0)

    def _jacobian(self, q: np.ndarray) -> np.ndarray:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        rows = []
        for fid in self.tip_ids:
            jac = pin.computeFrameJacobian(
                self.model,
                self.data,
                q,
                fid,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            rows.append(jac[:3, :])
        return np.concatenate(rows, axis=0)

    def solve(self, target_local: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        targets = np.asarray(target_local, dtype=np.float64)
        q_prev = np.zeros((len(_INSPIRE_DOF_NAMES),), dtype=np.float64)
        out = []
        errs = []
        eye = np.eye(len(_INSPIRE_DOF_NAMES), dtype=np.float64)
        for target in targets:
            q = q_prev.copy()
            for _ in range(self.iters):
                pred = self._fk_tips(q)
                err = (target - pred).reshape(-1)
                jac = self._jacobian(q)
                lhs = jac.T @ jac + (self.damping ** 2 + self.smooth_w + self.reg_w) * eye
                rhs = jac.T @ err - self.smooth_w * (q - q_prev) - self.reg_w * q
                dq = np.linalg.solve(lhs, rhs)
                q = np.clip(q + self.step * dq, self.lower, self.upper)
                if np.mean(np.linalg.norm((target - self._fk_tips(q)), axis=-1)) < 1e-4:
                    break
            q_prev = q
            out.append(q.astype(np.float32))
            errs.append(float(np.mean(np.linalg.norm(target - self._fk_tips(q), axis=-1))))
        return np.stack(out, axis=0), np.asarray(errs, dtype=np.float32)


def _resolve_ik_backend(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import pytorch_kinematics  # noqa: F401
        return "pytorch"
    except Exception:
        pass
    try:
        import pinocchio  # noqa: F401
        return "pinocchio"
    except Exception:
        pass
    raise RuntimeError("No IK backend available. Install pytorch_kinematics or pinocchio, or pass --ik-backend none.")


def _add_ik_refs(refs: dict[str, np.ndarray], args) -> dict[str, np.ndarray]:
    if args.ik_backend == "none":
        return refs
    if all(k in refs for k in ("rh_dof_pos", "lh_dof_pos", "rh_dof_vel", "lh_dof_vel")) and not args.force_recompute_ik:
        print("[ik] using cached rh/lh dof refs from input refs")
        return refs

    backend = _resolve_ik_backend(args.ik_backend)
    print(f"[ik] solving Inspire hand IK with backend={backend}")

    def solve_side(side: str, ref_prefix: str):
        targets = _tips_world_to_local(
            refs[f"{ref_prefix}_root_pos"],
            refs[f"{ref_prefix}_root_rot"],
            refs[f"{ref_prefix}_tip_world"],
        )
        if backend == "pytorch":
            solver = _PytorchKinematicsInspireIK(
                side,
                device=args.ik_device,
                lr=args.ik_lr,
                iters=args.ik_iters,
                smooth_w=args.ik_smooth_weight,
                reg_w=args.ik_reg_weight,
            )
        elif backend == "pinocchio":
            solver = _PinocchioInspireIK(
                side,
                iters=args.ik_iters,
                damping=args.ik_damping,
                step=args.ik_step,
                smooth_w=args.ik_smooth_weight,
                reg_w=args.ik_reg_weight,
            )
        else:
            raise ValueError(f"Unsupported IK backend: {backend}")
        q, err = solver.solve(targets)
        print(f"[ik] {ref_prefix}: mean_tip_err={err.mean():.5f}m  max_tip_err={err.max():.5f}m")
        return q

    refs = dict(refs)
    refs["rh_dof_pos"] = solve_side("rh", "rh").astype(np.float32)
    refs["lh_dof_pos"] = solve_side("lh", "lh").astype(np.float32)
    refs["rh_dof_vel"] = _gradient_or_zeros(refs["rh_dof_pos"], float(refs["fps"]))
    refs["lh_dof_vel"] = _gradient_or_zeros(refs["lh_dof_pos"], float(refs["fps"]))
    refs["ik_backend"] = np.asarray(backend)
    return refs


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


def _point_cross_segments(point: np.ndarray, radius: float) -> list[tuple[np.ndarray, np.ndarray]]:
    point = np.asarray(point, dtype=np.float32)
    axes = np.eye(3, dtype=np.float32) * float(radius)
    return [(point - axis, point + axis) for axis in axes]


def _full_hand_segments(joints25: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    joints25 = np.asarray(joints25, dtype=np.float32)
    segs = []
    for finger in range(5):
        base = finger * 5
        for joint_i in range(4):
            segs.append((joints25[base + joint_i], joints25[base + joint_i + 1]))
    return segs


def _build_gt_debug_lines(head_pos, rh_full, lh_full, cam_pos=None, viz_offset=None):
    """Draw head point, camera Z+ line, and approximate full-hand finger chains."""
    segs: list[tuple[np.ndarray, np.ndarray]] = []
    cols: list[np.ndarray] = []
    if viz_offset is None:
        viz_offset = np.zeros(3, dtype=np.float32)
    viz_offset = np.asarray(viz_offset, dtype=np.float32)
    head_pos = np.asarray(head_pos, dtype=np.float32) + viz_offset
    rh_full = np.asarray(rh_full, dtype=np.float32) + viz_offset
    lh_full = np.asarray(lh_full, dtype=np.float32) + viz_offset

    def add(seg, color):
        segs.append((np.asarray(seg[0], dtype=np.float32), np.asarray(seg[1], dtype=np.float32)))
        cols.append(np.asarray(color, dtype=np.float32))

    for seg in _point_cross_segments(head_pos, 0.09):
        add(seg, [0.1, 0.35, 1.])
    for seg in _point_cross_segments(head_pos, 0.045):
        add(seg, [1.0, 0.0, 0.0])

    if cam_pos is not None:
        cam_pos = np.asarray(cam_pos, dtype=np.float32)
        add((cam_pos, cam_pos + np.array([0.0, 0.0, 0.25], dtype=np.float32)), [1.0, 1.0, 0.0])

    for joints, color in ((rh_full, [0., 1., 0.]), (lh_full, [0., 0.85, 1.])):
        for seg in _full_hand_segments(joints):
            add(seg, color)
        for joint in joints:
            for seg in _point_cross_segments(joint, 0.007):
                add(seg, color)

    verts = np.array(segs, dtype=np.float32).reshape(-1, 3)
    colors = np.repeat(np.array(cols, dtype=np.float32), 2, axis=0)
    return len(segs), verts, colors


def _lookat_from_head_wrists(head, left_wrist, right_wrist, distance_scale=2.5):
    H = np.asarray(head, dtype=np.float32)
    L = np.asarray(left_wrist, dtype=np.float32)
    R = np.asarray(right_wrist, dtype=np.float32)

    target = (H + L + R) / 3.0
    left_axis = L - R
    left_axis = left_axis / (np.linalg.norm(left_axis) + 1e-8)
    wrist_mid = (L + R) / 2.0
    up_axis = H - wrist_mid
    up_axis = up_axis / (np.linalg.norm(up_axis) + 1e-8)
    forward_axis = np.cross(left_axis, up_axis)
    forward_axis = forward_axis / (np.linalg.norm(forward_axis) + 1e-8)

    view_vec = -1.73 * forward_axis + 1.00 * left_axis + 0.50 * up_axis
    view_vec = view_vec / (np.linalg.norm(view_vec) + 1e-8)
    body_radius = max(np.linalg.norm(H - target), np.linalg.norm(L - target), np.linalg.norm(R - target))
    eye = target + distance_scale * body_radius * view_vec
    return eye.astype(np.float32), target.astype(np.float32)


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
            dof_pos = _interp_np(self.refs[f"{prefix}_dof_pos"], frame_f) if f"{prefix}_dof_pos" in self.refs else zeros_dof
            dof_vel = _interp_np(self.refs[f"{prefix}_dof_vel"], frame_f) if f"{prefix}_dof_vel" in self.refs else zeros_dof

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
                dof_pos.astype(np.float32),
                dof_vel.astype(np.float32),
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


def _run_gt_only(args, env, refs, head_pos_seq,
                 rh_wrist_seq, lh_wrist_seq, rh_tips_seq, lh_tips_seq,
                 rh_full_seq, lh_full_seq, rh_rot_seq, lh_rot_seq,
                 actions_128, has_viewer, env_ptr):
    """Visualize and optionally record the GT reference skeleton without the dexhand."""
    import time as _time
    import imageio
    import os
    import torch
    from isaacgym import gymapi, gymtorch
    import hdt.constants as C

    total_steps = len(rh_wrist_seq)
    head_z_max = float(np.max(head_pos_seq[:, 2])) if len(head_pos_seq) else 0.0
    viz_offset = np.array([0.0, 0.0, 0.0 if head_z_max > 0.5 else 1.0], dtype=np.float32)
    print(f"[gt_only] head_z_max={head_z_max:.4f}; viz_offset={viz_offset.tolist()}")

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

    if has_viewer:
        try:
            env.gym.set_light_parameters(
                env.sim, 0,
                gymapi.Vec3(0.9, 0.9, 0.9),
                gymapi.Vec3(0.35, 0.35, 0.35),
                gymapi.Vec3(-0.3, 0.2, -1.0),
            )
        except Exception as exc:
            print(f"Warning: failed to set Isaac Gym light: {exc}")
        head_viz0 = head_pos_seq[0].astype(np.float32) + viz_offset
        lh_wrist_viz0 = lh_wrist_seq[0].astype(np.float32) + viz_offset
        rh_wrist_viz0 = rh_wrist_seq[0].astype(np.float32) + viz_offset
        viewer_cam_pos0, cam_target0 = _lookat_from_head_wrists(head_viz0, lh_wrist_viz0, rh_wrist_viz0)
        cam_pos = gymapi.Vec3(float(viewer_cam_pos0[0]), float(viewer_cam_pos0[1]), float(viewer_cam_pos0[2]))
        cam_target = gymapi.Vec3(float(cam_target0[0]), float(cam_target0[1]), float(cam_target0[2]))
        env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
        print(f"[gt_only] camera pos={list(np.round(viewer_cam_pos0, 3))}  "
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
            print(f"           rh_fingertips_xyz={np.round(rh_tips_seq[step_i], 4)}")
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
            head_viz = head_pos_seq[step_i].astype(np.float32) + viz_offset
            lh_wrist_viz = lh_wrist_seq[step_i].astype(np.float32) + viz_offset
            rh_wrist_viz = rh_wrist_seq[step_i].astype(np.float32) + viz_offset
            viewer_cam_pos, cam_target_np = _lookat_from_head_wrists(head_viz, lh_wrist_viz, rh_wrist_viz)
            cam_pos = gymapi.Vec3(float(viewer_cam_pos[0]), float(viewer_cam_pos[1]), float(viewer_cam_pos[2]))
            cam_target = gymapi.Vec3(float(cam_target_np[0]), float(cam_target_np[1]), float(cam_target_np[2]))
            env.gym.viewer_camera_look_at(env.viewer, None, cam_pos, cam_target)
            env.gym.clear_lines(env.viewer)
            n, verts, colors = _build_gt_debug_lines(
                head_pos_seq[step_i],
                rh_full_seq[step_i], lh_full_seq[step_i],
                cam_pos=viewer_cam_pos,
                viz_offset=viz_offset,
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

    if not args.gt_only:
        refs = _add_ik_refs(refs, args)
        if args.hand_control_mode == "ik" and ("rh_dof_pos" not in refs or "lh_dof_pos" not in refs):
            raise ValueError("--hand-control-mode ik needs IK refs; use --ik-backend auto/pytorch/pinocchio or a --ref-npz with rh/lh_dof_pos.")
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
    original_reset_dofs = dexhand_mimic_mod.DexHandMimic._reset_dofs

    def _load_human_policy_motion(self):
        self._motion_lib = motion_lib
        motion_lib.attach_env(self)

    dexhand_mimic_mod.DexHandMimic._load_motions = _load_human_policy_motion
    if args.exact_ik_reset and "rh_dof_pos" in refs and "lh_dof_pos" in refs:
        def _reset_dofs_exact_ik(self, env_ids):
            from isaacgym import gymtorch
            import torch

            self.rh_dof_pos[env_ids] = self._ref_rh_dof_pos[env_ids]
            self.rh_dof_vel[env_ids] = self._ref_rh_dof_vel[env_ids]
            self.lh_dof_pos[env_ids] = self._ref_lh_dof_pos[env_ids]
            self.lh_dof_vel[env_ids] = self._ref_lh_dof_vel[env_ids]
            rh_env_ids_int32 = self.rh_env_ids[env_ids].to(dtype=torch.int32)
            lh_env_ids_int32 = self.lh_env_ids[env_ids].to(dtype=torch.int32)
            dexhand_multi_env_ids_int32 = torch.concat([rh_env_ids_int32.flatten(), lh_env_ids_int32.flatten()])
            self.gym.set_dof_state_tensor_indexed(
                self.sim,
                gymtorch.unwrap_tensor(self.dof_state),
                gymtorch.unwrap_tensor(dexhand_multi_env_ids_int32),
                len(dexhand_multi_env_ids_int32),
            )

        dexhand_mimic_mod.DexHandMimic._reset_dofs = _reset_dofs_exact_ik
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
        if not args.show_table:
            env_cfg.terrain.mesh_type = "plane" if args.gt_only else None

        env, _ = task_registry.make_env(name="dexhand_mimic_direct", args=twist_args, env_cfg=env_cfg)
        env.reset_idx(torch.arange(env.num_envs, device=env.device), torch.zeros(env.num_envs, device=env.device, dtype=torch.long))
        obs = env.get_observations()

        policy, normalizer = (None, None)
        if not args.gt_only and args.hand_control_mode == "gmt":
            policy, normalizer = _load_hand_policy(env, train_cfg, twist_args, args)

        total_steps = args.rollout_steps
        if total_steps is None:
            total_steps = int(float(refs["length_s"]) / env.dt)
        total_steps = min(total_steps, int(env.max_episode_length))
        if args.max_steps is not None:
            total_steps = min(total_steps, int(args.max_steps))
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
        head_pos_seq = _interp_np(refs["head_pos"], ref_frame_f) if "head_pos" in refs else np.zeros((total_steps, 3), dtype=np.float32)
        rh_wrist_seq = _interp_np(refs["rh_root_pos"], ref_frame_f)            # (T,3)
        lh_wrist_seq = _interp_np(refs["lh_root_pos"], ref_frame_f)            # (T,3)
        rh_tips_seq  = _interp_np(refs["rh_tip_world"], ref_frame_f)           # (T,5,3)
        lh_tips_seq  = _interp_np(refs["lh_tip_world"], ref_frame_f)           # (T,5,3)
        rh_full_seq  = _interp_np(refs["rh_full_world"], ref_frame_f) if "rh_full_world" in refs else np.zeros((total_steps, 25, 3), dtype=np.float32)
        lh_full_seq  = _interp_np(refs["lh_full_world"], ref_frame_f) if "lh_full_world" in refs else np.zeros((total_steps, 25, 3), dtype=np.float32)
        rh_rot_seq   = _sample_quat_nearest(refs["rh_root_rot"], ref_frame_f)  # (T,4) xyzw
        lh_rot_seq   = _sample_quat_nearest(refs["lh_root_rot"], ref_frame_f)  # (T,4) xyzw

        # Print frame-0 ref vs env wrist to check coordinate alignment.
        print(f"[ref frame0] rh_wrist_pos={rh_wrist_seq[0]}  lh_wrist_pos={lh_wrist_seq[0]}")
        print(f"[env frame0] rh_root_pos ={env.rh_root_states[0, :3].cpu().numpy()}  "
              f"lh_root_pos={env.lh_root_states[0, :3].cpu().numpy()}")
        if "rh_dof_pos" in refs and "lh_dof_pos" in refs:
            print(f"[ik frame0] rh_dof={np.round(refs['rh_dof_pos'][0], 4)}")
            print(f"[ik frame0] lh_dof={np.round(refs['lh_dof_pos'][0], 4)}")

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
            _run_gt_only(args, env, refs, head_pos_seq,
                         rh_wrist_seq, lh_wrist_seq, rh_tips_seq, lh_tips_seq,
                         rh_full_seq, lh_full_seq, rh_rot_seq, lh_rot_seq,
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

            if args.hand_control_mode == "gmt":
                with torch.no_grad():
                    pol_obs = normalizer.normalize(obs.detach()) if normalizer is not None else obs.detach()
                    hand_action = policy(pol_obs, hist_encoding=True)
            else:
                frame_f = np.asarray([step_i * env.dt * float(refs["fps"])], dtype=np.float32)
                rh_q = _interp_np(refs["rh_dof_pos"], frame_f)
                lh_q = _interp_np(refs["lh_dof_pos"], frame_f)
                q = np.concatenate([rh_q, lh_q], axis=-1)
                default = torch.cat([env.default_rh_dof_pos, env.default_lh_dof_pos], dim=0).detach().cpu().numpy()[None]
                action_np = (q - default) / float(env.cfg.control.action_scale)
                hand_action = torch.as_tensor(action_np, device=env.device, dtype=torch.float32)

            action_log.append(hand_action.detach().cpu().numpy()[0].astype(np.float32))
            obs, _, _, done, _ = env.step(hand_action.detach())

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
            hand_actions=np.asarray(action_log, dtype=np.float32),
            gmt_actions=np.asarray(action_log, dtype=np.float32) if args.hand_control_mode == "gmt" else np.zeros((0, 0), dtype=np.float32),
            human_policy_actions=actions_128.astype(np.float32),
            rh_ik_dof_pos=refs.get("rh_dof_pos", np.zeros((0, len(_INSPIRE_DOF_NAMES)), dtype=np.float32)),
            lh_ik_dof_pos=refs.get("lh_dof_pos", np.zeros((0, len(_INSPIRE_DOF_NAMES)), dtype=np.float32)),
            hand_control_mode=np.asarray(args.hand_control_mode),
            ref_fps=np.asarray(args.ref_fps, dtype=np.float32),
        )
        print(f"Saved hand actions: {out_actions}")
        if args.record_video:
            print(f"Saved video: {args.out_video}")
    finally:
        dexhand_mimic_mod.DexHandMimic._load_motions = original_load_motions
        dexhand_mimic_mod.DexHandMimic._reset_dofs = original_reset_dofs


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
    p.add_argument("--hand-control-mode", choices=["gmt", "ik"], default="gmt",
                   help="gmt: track references with the trained GMT policy. ik: directly send IK dof targets as actions.")
    p.add_argument("--ik-backend", choices=["auto", "pytorch", "pinocchio", "none"], default="auto",
                   help="IK solver backend. auto prefers pytorch_kinematics, then pinocchio.")
    p.add_argument("--ik-device", type=str, default=None,
                   help="Torch device for --ik-backend pytorch. Defaults to --gmt-device.")
    p.add_argument("--ik-iters", type=int, default=120)
    p.add_argument("--ik-lr", type=float, default=0.03)
    p.add_argument("--ik-damping", type=float, default=1e-3)
    p.add_argument("--ik-step", type=float, default=0.7)
    p.add_argument("--ik-smooth-weight", type=float, default=1e-2)
    p.add_argument("--ik-reg-weight", type=float, default=1e-4)
    p.add_argument("--force-recompute-ik", action="store_true",
                   help="Recompute IK even if rh/lh dof refs already exist in --ref-npz.")
    p.add_argument("--no-exact-ik-reset", dest="exact_ik_reset", action="store_false",
                   help="Keep TWIST's reset dof randomization instead of exactly setting frame-0 IK dofs.")
    p.set_defaults(exact_ik_reset=True)
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
    if args.ik_device is None:
        args.ik_device = args.gmt_device
    run(args)


if __name__ == "__main__":
    main()
