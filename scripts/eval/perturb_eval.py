"""MPJPE under visual distribution shift -- ST-WAM's evaluation axes, our metric.

All three papers report the latent/DINO benefit only under visual shift, never in-domain.
Our val set is same task / same pillow / same scene, so an in-domain MPJPE can't see it.
This applies ST-WAM's real-world shift list (background, lighting, object appearance,
compound) plus LIBERO-Plus's camera and sensor-noise axes to the INPUT IMAGE ONLY --
ground truth actions are untouched -- and reports MPJPE degradation per checkpoint.

Was /tmp/perturb_eval.py. The perturbation functions and the measurement loop are
unchanged; only the entry point moved from a hard-coded 4-checkpoint list to one
checkpoint per invocation, so the 8 ablation runs can be evaluated one per GPU in
parallel (see run_perturb_all.sh). Numbers stay comparable across checkpoints because
the noise draw is seeded by frame index, not by run order.

  python scripts/eval/perturb_eval.py --name ab4 \
      --ckpt ab4_mixed_w1.0_h45_ckpt \
      --cfg hdt/configs/models/act_with_future_dino.yaml \
      --out /tmp/perturb_ab4.json
"""
import os, sys, pickle, glob, json, argparse
import numpy as np, h5py, cv2, torch

# This file lives at <repo>/scripts/eval/, so derive the repo root instead of pinning
# it to one machine.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
import plot_keypoints_ys as P
from hdt.modeling.utils import make_visual_encoder

DEV = "cuda"
STRIDE = 3           # subsample frames; ~84 per episode is plenty


# ---- perturbations (uint8 RGB HxWx3) -------------------------------------------------
def clean(im, rs):
    return im

def light_dim(im, rs):
    return np.clip(im.astype(np.float32) * 0.5, 0, 255).astype(np.uint8)

def light_bright(im, rs):
    return np.clip(im.astype(np.float32) * 1.6 + 25, 0, 255).astype(np.uint8)

def contrast(im, rs):
    m = im.astype(np.float32).mean()
    return np.clip((im.astype(np.float32) - m) * 0.5 + m, 0, 255).astype(np.uint8)

def obj_appearance(im, rs):
    """ST-WAM 'object appearance': hue shift preserves geometry, changes colour."""
    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + 60) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

def background(im, rs):
    """Texture the low-saturation / peripheral region, keeping the centre workspace."""
    h, w = im.shape[:2]
    tex = rs.randint(0, 255, (h // 8, w // 8, 3), dtype=np.uint8)
    tex = cv2.resize(tex, (w, h), interpolation=cv2.INTER_NEAREST)
    mask = np.zeros((h, w), np.float32)
    mask[: int(h * 0.35), :] = 1.0                    # upper third = background wall
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    return (im * (1 - mask) + tex * mask).astype(np.uint8)

def sensor_noise(im, rs):
    return np.clip(im.astype(np.float32) + rs.randn(*im.shape) * 25, 0, 255).astype(np.uint8)

def blur(im, rs):
    return cv2.GaussianBlur(im, (9, 9), 3)

def camera_shift(im, rs):
    """LIBERO-Plus 'camera': 8% translate + 6% zoom, edge-replicated."""
    h, w = im.shape[:2]
    M = np.float32([[1.06, 0, 0.08 * w], [0, 1.06, -0.05 * h]])
    return cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

def occlusion(im, rs):
    im = im.copy()
    h, w = im.shape[:2]
    im[int(h * 0.55):int(h * 0.80), int(w * 0.30):int(w * 0.55)] = 128
    return im

def compound(im, rs):
    return sensor_noise(obj_appearance(background(light_dim(im, rs), rs), rs), rs)

PERTURBS = [("clean", clean), ("light_dim", light_dim), ("light_bright", light_bright),
            ("contrast", contrast), ("obj_appear", obj_appearance), ("background", background),
            ("noise", sensor_noise), ("blur", blur), ("camera", camera_shift),
            ("occlusion", occlusion), ("compound", compound)]


def joints(a128):
    j = P._action_to_eval_joints(a128)
    return np.concatenate([j["head"][None], np.stack([j["lw"], j["rw"]]),
                           j["lk_world"], j["rk_world"], j["waist"][None]], axis=0)


def run_ckpt(name, ckpt_dir, cfg, val_dir, ckpt_file="policy_last.ckpt"):
    policy = P._load_act_policy(cfg, 100, ["top"]).to(DEV)
    sd = torch.load(os.path.join(ckpt_dir, ckpt_file), map_location=DEV, weights_only=True)
    policy.load_state_dict(sd, strict=False)
    policy.eval()
    _, prep = make_visual_encoder("ACT", {})
    with open(os.path.join(ckpt_dir, "dataset_stats.pkl"), "rb") as f:
        loaded = pickle.load(f)
    ns = loaded[0] if isinstance(loaded, tuple) else loaded

    err = {p: [] for p, _ in PERTURBS}
    for path in sorted(glob.glob(os.path.join(val_dir, "*.hdf5"))):
        with h5py.File(path, "r") as root:
            s = ns[str(root.attrs.get("embodiment", "dex5"))]
            qm, qs = s["qpos_mean"].astype(np.float32), s["qpos_std"].astype(np.float32)
            am, ast = s["action_mean"].astype(np.float32), s["action_std"].astype(np.float32)
            states, gt_act = root["observation.state"][()], root["action"][()]
            cams = [k[len("observation.image."):] for k in root if k.startswith("observation.image.")]
            images = root[f"observation.image.{'top' if 'top' in cams else cams[0]}"]
            start = P._detect_valid_start_from_actions(gt_act, check_frames=20,
                                                       jump_threshold_m=0.3, settle_frames=0)
            for t in range(start, states.shape[0], STRIDE):
                img = images[t]
                if img.ndim == 1:
                    img = cv2.cvtColor(cv2.imdecode(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                if img.shape[0] != 240 or img.shape[1] != 320:
                    img = cv2.resize(img, (320, 240))
                q = torch.from_numpy((states[t].astype(np.float32) - qm) / (qs + 1e-6)).unsqueeze(0).to(DEV)
                gtj = joints(gt_act[t])
                rs = np.random.RandomState(t)   # same noise draw across checkpoints
                for pname, fn in PERTURBS:
                    pim = fn(img, rs if pname in ("background", "noise", "compound") else rs)
                    it = prep(pim.transpose(2, 0, 1)[None].astype(np.uint8)).unsqueeze(0).to(DEV)
                    with torch.no_grad():
                        a = policy(it, q, conditioning_dict=None)[0, 0].cpu().numpy()
                    pj = joints(a * (ast + 1e-6) + am)
                    err[pname].append(np.linalg.norm(pj - gtj, axis=1).mean())
        print(f"  [{name}] {os.path.basename(path)[-22:]}", flush=True)
    return {p: float(np.mean(v) * 1000) for p, v in err.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--ckpt", required=True, help="checkpoint dir (holds policy_last.ckpt + dataset_stats.pkl)")
    ap.add_argument("--cfg", required=True, help="model yaml the run was trained with")
    ap.add_argument("--val_dir", default=os.path.join(REPO, "data/dex5_val"))
    ap.add_argument("--ckpt_file", default="policy_last.ckpt")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = run_ckpt(a.name, a.ckpt, a.cfg, a.val_dir, a.ckpt_file)
    json.dump({a.name: res}, open(a.out, "w"), indent=1)
    print(f"== {a.name}: " + "  ".join(f"{k}={v:.1f}" for k, v in res.items()), flush=True)
