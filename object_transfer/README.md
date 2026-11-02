# Isolated object-centric HAT experiment

This directory contains the new Object Head, object-token fusion, label
manifest tools, an isolated HAT source snapshot, tests, experiment preparation,
and launcher. The normal repository HAT and deployment paths are unchanged.

The active experiment uses:

- the original global V-JEPA/HAT visual and state paths;
- a shared Object Head for Omega and Human RGB;
- metric wrist-relative-to-object motion loss for both domains;
- the shared original 3D action head for both domains, with Human action loss
  weight 0.2;
- a gated object residual initialized near zero;
- no detector/segmenter or extra input at inference.

Six runs are defined in jobs.json under the run root: omega_original,
omega_human_original, and omega_human_random for both wipe_table and
classify_gadgets. Omega-only runs use 80k updates. Mixed runs use 160k updates
with 50/50 domain sampling, preserving approximately 80k Robot updates.

Run artifacts are under:

/mnt/nvme0n1/human_policy_ckpt/object_transfer_20261102

Each status JSON records the launcher command and isolated source root. The six
active jobs were started from source revision cf7586d; a7ab94b adds
documentation only. See AUDIT.md for verified interfaces, label provenance,
checks, and current limitations.
