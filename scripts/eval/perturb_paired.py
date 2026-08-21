"""Paired per-episode comparison of two checkpoints on the perturbation suite.

Reads the `.per_ep.json` files perturb_eval.py writes alongside its `--out`.

Why paired and not "mean +- SEM": the raw SEM over 10 episodes is ~4.6mm on a ~42mm
mean, because the episodes genuinely differ in difficulty -- a hard episode is hard for
every checkpoint. That between-episode variance is common mode and cancels, because
every checkpoint sees the SAME 10 episodes, the SAME frames (STRIDE 3 from the same
detected start) and the SAME perturbation noise draw (RandomState seeded by frame index,
not by run). So the quantity with a usable error bar is the per-episode difference, not
the difference of the means. They have the identical mean; only the spread differs, and
for these checkpoints the paired spread is roughly an order of magnitude smaller.

Reported per axis: mean diff, its SEM, and how many of the 10 episodes agree in sign.
10/10 or 9/10 with |mean| > 2*SEM is a real difference; anything near 5/10 is noise.

  python scripts/eval/perturb_paired.py /tmp/errbar_ab4_dex5_val.per_ep.json \
                                        /tmp/errbar_ab7_dex5_val.per_ep.json
"""
import sys, json
import numpy as np

AXES = ["clean", "background", "noise", "blur", "camera", "occlusion", "compound"]


def load(path):
    d = json.load(open(path))
    name = next(iter(d))
    return name, {k: np.array(v, dtype=float) for k, v in d[name].items()}


def main(pa, pb):
    na, a = load(pa)
    nb, b = load(pb)
    n = len(next(iter(a.values())))
    print(f"{nb} - {na}   (paired over {n} episodes, negative = {nb} better)")
    print(f"{'axis':<12}{na:>9}{nb:>9}{'diff':>9}{'SEM':>8}   sign")
    for ax in AXES:
        if ax not in a or ax not in b:
            continue
        d = b[ax] - a[ax]
        sem = d.std(ddof=1) / np.sqrt(n)
        win = int((d < 0).sum())
        sig = "*" if abs(d.mean()) > 2 * sem else " "
        print(f"{ax:<12}{a[ax].mean():9.1f}{b[ax].mean():9.1f}"
              f"{d.mean():+9.1f}{sem:8.1f}{sig}  {win}/{n}")
    # Relative degradation vs each checkpoint's own clean, per episode: the robustness
    # number the whole ablation is judged on, so it needs the same paired treatment.
    pert = [x for x in AXES if x != "clean"]
    ra = np.mean([a[x] / a["clean"] - 1 for x in pert if x in a], axis=0) * 100
    rb = np.mean([b[x] / b["clean"] - 1 for x in pert if x in b], axis=0) * 100
    d = rb - ra
    sem = d.std(ddof=1) / np.sqrt(n)
    sig = "*" if abs(d.mean()) > 2 * sem else " "
    print(f"{'rel.degrad%':<12}{ra.mean():9.1f}{rb.mean():9.1f}"
          f"{d.mean():+9.1f}{sem:8.1f}{sig}  {int((d < 0).sum())}/{n}")
    print("  * = |mean diff| > 2 SEM")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
