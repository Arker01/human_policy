"""Merge per-checkpoint perturb_eval.py outputs into one table.

Two numbers per run matter and they say different things:
  - the absolute MPJPE on each axis (how bad it gets), and
  - degradation vs that run's OWN clean number (how much of the damage is the shift
    rather than the run just being worse everywhere). The second is the one the
    staircase is designed to compare, since clean MPJPE is known to be flat.

Usage: perturb_report.py /tmp/perturb_ab_*_dex5_val.json
"""
import json, sys, glob

AXES = ["clean", "light_dim", "light_bright", "contrast", "obj_appear", "background",
        "noise", "blur", "camera", "occlusion", "compound"]

files = [f for a in sys.argv[1:] for f in sorted(glob.glob(a))]
if not files:
    sys.exit("no result files")

res = {}
for f in files:
    res.update(json.load(open(f)))
names = sorted(res)

print("absolute MPJPE (mm)")
print(f"{'axis':<14}" + "".join(f"{n.split('_')[0]:>9}" for n in names))
for ax in AXES:
    print(f"{ax:<14}" + "".join(f"{res[n].get(ax, float('nan')):9.2f}" for n in names))

print("\ndegradation vs own clean (%)")
print(f"{'axis':<14}" + "".join(f"{n.split('_')[0]:>9}" for n in names))
for ax in AXES:
    if ax == "clean":
        continue
    print(f"{ax:<14}" + "".join(
        f"{(res[n][ax] / res[n]['clean'] - 1) * 100:8.1f}%" for n in names))
print(f"{'MEAN':<14}" + "".join(
    f"{sum(res[n][a] / res[n]['clean'] - 1 for a in AXES if a != 'clean') / (len(AXES) - 1) * 100:8.1f}%"
    for n in names))
