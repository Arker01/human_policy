"""ST-WAM's frame-triplet diagnosis, run on OUR data and OUR encoders.

ST-WAM (Sec. 3) measures, over triplets of (clean frame, visually-shifted same-state
frame, different-state frame): (a) same-state cosine similarity under shift, and
(b) how often the shifted frame is ranked closer to the state-matched clean frame than
to the different-state frame. They get DINOv3 = 0.904 / 95.2% vs Wan-VAE = 0.686 / 60.0%.

Was /tmp/triplet_diag.py. Two things changed:
  - added the V-JEPA 2.1 video teacher as a third encoder, because this is exactly the
    measurement that motivated it -- DINOv2-S scored 0.517 same-state vs a 0.480
    different-state floor, i.e. an almost information-free target. If V-JEPA does not
    beat that margin here, training against it is not worth a GPU-day.
  - the different-state reference is now printed per encoder AND used for the ranking,
    unchanged in spirit, but a video teacher is fed a 2-frame clip (t-stride, t) rather
    than a single frame, matching how the training loop feeds it.

Usage: triplet_diag.py [ckpt_dir_for_the_resnet_trunk]
"""
import sys, glob, os
import numpy as np, h5py, cv2, torch

# Same as perturb_eval.py: derive the repo root from this file's location.
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "data"))
sys.path.insert(0, os.path.join(REPO, "scripts/eval"))
import plot_keypoints_ys as P
from hdt.detr.models.detr_vae import FrozenPatchTargetEncoder
from hdt.modeling.utils import make_visual_encoder
from perturb_eval import PERTURBS

DEV = "cuda"
DIFF_GAP = 60          # frames apart = "different state" reference
STRIDE = 12
CLIP_STRIDE = 4        # must match future_dino.clip_stride in the vjepa yaml
TRUNK_CKPT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "ab4_mixed_w1.0_h45_ckpt")

dino = FrozenPatchTargetEncoder("dinov2_vits14").to(DEV).eval()
vjepa = FrozenPatchTargetEncoder("vjepa2_1_vitb", target_resolution_hw=[224, 304]).to(DEV).eval()
_, prep = make_visual_encoder("ACT", {})

policy = P._load_act_policy(os.path.join(REPO, "hdt/configs/models/act_with_future_dino.yaml"),
                            100, ["top"]).to(DEV)
policy.load_state_dict(torch.load(os.path.join(TRUNK_CKPT, "policy_last.ckpt"),
                                  map_location=DEV, weights_only=True), strict=False)
policy.eval()
resnet = policy.model.backbones[0].to(DEV).eval()

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def _norm_batch(ims):
    """list of HxWx3 uint8 -> normalized [N,3,224,308] on DEV, the trunk's geometry."""
    x = torch.from_numpy(np.stack(ims).transpose(0, 3, 1, 2)).float().to(DEV) / 255.0
    x = torch.nn.functional.interpolate(x, size=(224, 308), mode="bilinear", align_corners=False)
    return (x - MEAN) / STD


def feat_dino(clip):
    with torch.no_grad():
        f = dino(_norm_batch(clip[-1:]))        # image teacher sees only the last frame
    f = f[0] if isinstance(f, (tuple, list)) else f
    return f.flatten().float().cpu().numpy()


def feat_vjepa(clip):
    with torch.no_grad():
        f = vjepa(_norm_batch(clip).unsqueeze(0))   # [1, K, 3, H, W]
    return f.flatten().float().cpu().numpy()


def feat_resnet(clip):
    x = prep(clip[-1].transpose(2, 0, 1)[None].astype(np.uint8)).to(DEV)
    with torch.no_grad():
        f = resnet(x)
    while isinstance(f, (tuple, list)):
        f = f[0]
    if isinstance(f, dict):
        f = list(f.values())[-1]
    return f.flatten().float().cpu().numpy()


def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


ENCS = [("DINOv2-S (旧 target)", feat_dino),
        ("V-JEPA 2.1 B (新 target)", feat_vjepa),
        ("resnet18 trunk (策略看的)", feat_resnet)]
same = {e: {p: [] for p, _ in PERTURBS} for e, _ in ENCS}
rank = {e: {p: [] for p, _ in PERTURBS} for e, _ in ENCS}
diffref = {e: [] for e, _ in ENCS}

for path in sorted(glob.glob(os.path.join(REPO, "data/dex5_val/*.hdf5")))[:5]:
    with h5py.File(path, "r") as root:
        cams = [k[len("observation.image."):] for k in root if k.startswith("observation.image.")]
        images = root[f"observation.image.{'top' if 'top' in cams else cams[0]}"]
        T = images.shape[0]

        def get(t):
            im = images[max(0, t)]
            if im.ndim == 1:
                im = cv2.cvtColor(cv2.imdecode(im, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            if im.shape[0] != 240 or im.shape[1] != 320:
                im = cv2.resize(im, (320, 240))
            return im

        def clip_at(t):
            return [get(t - CLIP_STRIDE), get(t)]

        for t in range(20, T - DIFF_GAP, STRIDE):
            clip_c, clip_d = clip_at(t), clip_at(t + DIFF_GAP)
            rs = np.random.RandomState(t)
            for ename, ef in ENCS:
                fc, fd = ef(clip_c), ef(clip_d)
                s_diff = cos(fc, fd)
                diffref[ename].append(s_diff)
                for pname, fn in PERTURBS:
                    if pname == "clean":
                        continue
                    s_same = cos(fc, ef([fn(im, np.random.RandomState(t)) for im in clip_c]))
                    same[ename][pname].append(s_same)
                    rank[ename][pname].append(1.0 if s_same > s_diff else 0.0)
    print("done", path.split("/")[-1][-22:], flush=True)

print(f"\n{'':<14}" + "".join(f"{e:>34}" for e, _ in ENCS))
print(f"{'perturbation':<14}" + "".join(f"{'same-state cos':>18}{'rank-correct':>16}" for _ in ENCS))
for pname, _ in PERTURBS:
    if pname == "clean":
        continue
    row = f"{pname:<14}"
    for ename, _ in ENCS:
        row += f"{np.mean(same[ename][pname]):18.3f}{np.mean(rank[ename][pname])*100:15.1f}%"
    print(row)
print("\nMEAN".ljust(14) + "".join(
    f"{np.mean([v for p, vs in same[e].items() if p != 'clean' for v in vs]):18.3f}"
    f"{np.mean([v for p, vs in rank[e].items() if p != 'clean' for v in vs]) * 100:15.1f}%"
    for e, _ in ENCS))
print("different-state cos (基线,越低越好)".ljust(14) +
      "".join(f"  {e}: {np.mean(diffref[e]):.3f}" for e, _ in ENCS))
