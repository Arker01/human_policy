# Isolated object-centric HAT experiment

This directory contains the new Object Head, object-token fusion, label
manifest tools, an isolated HAT source snapshot, tests, experiment preparation,
and launcher. The normal repository HAT and deployment paths are unchanged.

The active experiment uses:

- a frozen released V-JEPA 2.1 backbone, with all HAT/action/state/fusion heads
  randomly initialized and trained without an old HAT checkpoint;
- a shared Object Head for Omega and Human RGB;
- an explicit image-space object-center to detected hand-region relation head
  for both domains, plus metric future wrist-displacement supervision;
- the shared original 3D action head for both domains, with Human action loss
  weight 0.2;
- a gated object residual initialized at 0.5; uncalibrated Presence is excluded
  from both the gate multiplier and fusion context;
- Action/HAT LR 1e-5 and object/fusion/relation LR 1e-4;
- Future-DINO disabled;
- no detector/segmenter or extra input at inference.

Six runs are defined in jobs.json under the run root: omega_original,
omega_human_original, and omega_human_random for both wipe_table and
classify_gadgets. Omega-only runs use 100k updates. Mixed runs use 200k updates
with 50/50 domain sampling, preserving approximately 100k Robot updates.

Run artifacts are under:

/mnt/nvme0n1/human_policy_ckpt/object_transfer_scratch_20261103

Each status JSON records the launcher command and isolated source root. The six
active jobs were started from source revision 14ef598. See AUDIT.md for
verified interfaces, label provenance, checks, and current limitations.
