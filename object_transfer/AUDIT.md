# Object-transfer implementation audit

Date: 2026-11-02

## Scope and decision override

The implementation is isolated on branch experiment/object-transfer-20261102
and under object_transfer/. The existing HAT package, deployment configuration,
checkpoints, and controller are not modified.

The latest experiment request intentionally overrides section 0/5 of
hat_object_centric.md: Human action supervision is retained at weight 0.2,
instead of being removed. Robot and Human losses are averaged within each
domain and combined as robot_mean + 0.2 * human_mean; adding Human samples
therefore does not numerically dilute the Robot loss.

## Located training and data path

- Training entry: object_transfer/src/hdt/main.py
- Dataset/HDF5 loader: object_transfer/src/hdt/data_utils_hdt.py
- ACT policy/loss: object_transfer/src/hdt/policy.py
- HAT/DETR insertion point:
  object_transfer/src/hdt/detr/models/detr_vae.py
- Omega source:
  /mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat/{task}
- Geometry-aligned Human source:
  /mnt/nvme1n1/human_policy_derived/human_geometry_aligned_20261102/{task}
- Baseline initialization:
  /mnt/nvme0n1/human_policy_ckpt/omega_HOME_4tasks_geometry_noleftwrist_w0_20261025/checkpoints/{task}_ckpt/policy_iter_100000_seed_0/pytorch_model.bin
- New run root:
  /mnt/nvme0n1/human_policy_ckpt/object_transfer_20261102

The old checkpoint loads with strict=False; only the new Object Head and gated
fusion parameters are missing and initialized from scratch.

## Verified model interface

The real V-JEPA feature used by the Object Head is [B, 768, 14, 19].
The existing global visual path remains intact and produces the existing DETR
action hidden [B, 100, 512]. The new branch predicts a 14x19 spatial
distribution, normalized UV, presence probability, and pooled 768-D object
token. A gated residual is applied to the existing action hidden before the
unchanged 128-D keypoint action head. The gate starts at
sigmoid(-6) = 0.0024726.

~~~mermaid
flowchart LR
    RGB[RGB 240x320] --> V[V-JEPA 768x14x19]
    V --> G[Existing global visual path]
    V --> O[Object heatmap / presence head]
    State[Current 128-D state] --> H[Existing HAT hidden]
    G --> H
    O --> F[Gated residual fusion]
    H --> F
    F --> A[Existing 100x128 action chunk]
~~~

Three real smoke runs exercised the following batch contract:

| Run | RGB | State | Action target | Object grid | Domains |
| --- | --- | --- | --- | --- | --- |
| wipe mixed random | [64,1,3,240,320] | [64,128] | [64,100,128] | [64,14,19] | Robot + Human |
| classify mixed random | same | same | same | same | Robot + Human |
| wipe Omega-only | same | same | same | same | Robot |

Positions/actions remain in the existing normalized HAT training interface;
the relative-wrist auxiliary loss first de-normalizes the metric wrist xyz
using the matching per-domain statistics. Object UV is normalized image
coordinates in the exact 320x240 network input.

## Label evidence and limits

The existing background-ablation foreground mask is a union of object, hand,
and arm and is not valid object-instance truth. It is not used. The offline
manifest instead uses high-confidence detector refresh boxes:

- wipe: cloth/rag box centers;
- classify: gadget box centers only near hand/arm interaction.

These labels are sparse and are box centers, not masks or true contact points.
Invalid frames are masked from localization loss. Counts validated by
check_labels.py:

| Task/domain | Episodes | Frames | Valid points |
| --- | ---: | ---: | ---: |
| wipe Robot | 187 | 137080 | 1886 |
| wipe Human | 75 | 20722 | 286 |
| classify Robot | 200 | 382008 | 2817 |
| classify Human | 75 | 26552 | 298 |

The manifest is stored separately at
/mnt/nvme1n1/human_policy_derived/object_transfer_labels_relation_20261102;
original
HDF5 files are not rewritten. No detector or segmenter is used at inference.

## Training checks

- Unit tests: 5 passed.
- Smoke runs completed forward, validation, backward, and checkpoint save.
- Both Object Head and fusion have finite nonzero first-step gradients.
- Mixed validation logs contain separate Robot and Human action/relative-wrist
  losses; Omega-only logs correctly report a zero Human component.
- Random-background Human RGB is read through human_image_override_root, while
  aligned Human state/action remains the source of supervision.
- Train/validation file lists are split by episode; background variants reuse
  those same lists and cannot cross the split.

## Known limitations / TODO

- Labels are intentionally sparse; they do not justify claims about contact
  point accuracy.
- No phase head is enabled because a reliable shared phase label was not found.
- Wrist-to-image projection is not fused because the two camera projection
  chains have not passed a common pixel-error audit.
- Training completion and offline B0/B1/new-model action comparisons remain
  pending. A successful smoke run is not evidence of task success.

