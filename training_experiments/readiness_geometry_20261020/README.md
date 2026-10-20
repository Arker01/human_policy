# 2026-10-20 readiness / geometry training snapshot

This directory preserves the exact 41 Python source files used for the four
100,000-step runs. The local and 4090-168-aigc copies were compared by SHA256
before versioning and matched. It is separate from the deployed `hdt/` tree.

Variants: `visual_w01` (future visual loss 0.1), `visual_w1` (1.0),
`readiness` (readiness auxiliary loss 0.1), `geometry` (geometry auxiliary loss
0.1). All use the existing replace50 split, frozen V-JEPA encoder, state mask
98:128, chunk size 100, seed 0, and global batch size 64.

The unchanged launchers contain the original machine paths. The actual completed
runs, data, and checkpoints remain at:

- Training server: `/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_readiness_geometry`
- Local experiment mirror: `/home/nerv/qingyaoxu/human_policy_training_experiments/20261020_readiness_geometry`
- Local deployment bundles: `pick_bottle_0930/b0930_20261020_*_ckpt`

`src/run_experiment.py` uses the original server jobs manifest; its source is
also archived as `configs/original_jobs.json`. Dependencies `cet`, `pytorch3d`,
and `assets` resolve to the enclosing repository. Python 3.11 and the server's
`human_policy` environment were used for training.

Weights and datasets are transferred separately, not stored in Git.
`checkpoint_sha256.json` identifies the exact final weights. The four bundle
inference YAMLs and training configs are versioned with the deployment branch.
See `README_HAT_ScaleBFM_nerv.md` for the existing online service command.
The deployed service loads all required action-network weights and ignores
Future-DINO / readiness auxiliary heads. Readiness probability gating is not
enabled by these deployment commands.
