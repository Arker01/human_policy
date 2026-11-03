# Object-transfer implementation audit

Date: 2026-11-03

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
- New run root:
  /mnt/nvme0n1/human_policy_ckpt/object_transfer_scratch_20261103

No HAT/action checkpoint is loaded. The DETR/HAT encoder, decoder, state path,
action head, Object Head, gated fusion, and relation head are all randomly
initialized. The general-purpose V-JEPA 2.1 image backbone retains its released
pretrained weights and remains frozen, matching the existing HAT visual
interface.

## Verified model interface

The real V-JEPA feature used by the Object Head is [B, 768, 14, 19].
The existing global visual path remains intact and produces the existing DETR
action hidden [B, 100, 512]. The new branch predicts a 14x19 spatial
distribution, normalized UV, presence probability, and pooled 768-D object
token. A gated residual is applied to the existing action hidden before the
unchanged 128-D keypoint action head. Because the action model itself starts
from scratch, the fusion gate starts half open: sigmoid(0) = 0.5.
The uncalibrated Presence output neither multiplies the gate nor enters the
fusion context.

~~~mermaid
flowchart LR
    RGB[RGB 240x320] --> V[V-JEPA 768x14x19]
    V --> G[Existing global visual path]
    V --> O[Object heatmap / presence head]
    State[Current 128-D state] --> H[Existing HAT hidden]
    G --> H
    O --> F[Gated residual fusion]
    H --> F
    F --> R[Object-to-hand image relation head]
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
When both boxes are reliable, a second label stores the normalized image-space
vector from object center to detected hand/arm/gripper-region center. This is
a region-center proxy, not a calibrated wrist projection. Invalid frames are
masked from localization/relation losses. Counts validated by check_labels.py:

| Task/domain | Episodes | Frames | Object points | Object-hand relations |
| --- | ---: | ---: | ---: | ---: |
| wipe Robot | 187 | 137080 | 1886 | 1593 |
| wipe Human | 75 | 20722 | 286 | 277 |
| classify Robot | 200 | 382008 | 2817 | 2817 |
| classify Human | 75 | 26552 | 298 | 298 |

The manifest is stored separately at
/mnt/nvme1n1/human_policy_derived/object_transfer_labels_relation_20261102;
original
HDF5 files are not rewritten. No detector or segmenter is used at inference.

## Training checks

- Unit tests: 7 passed.
- Smoke runs completed forward, validation, backward, and checkpoint save.
- Launch manifests assert initialization=scratch_action_model,
  pretrained=null, and contain no --load_pretrained_path argument.
- Action/state/transformer LR is 1e-5; Object Head, fusion, gate, and relation
  LR is 1e-4. Runtime CSVs confirm both values.
- Future-DINO is disabled; its previous zero-weight head and target-image
  loading are absent.
- Object Head and fusion have finite nonzero first-step gradients. The relation
  head is checked on the first batch containing a valid relation; all six
  formal runs produced finite nonzero relation-head gradients by step 3.
- Mixed validation logs contain separate Robot and Human action/relative-wrist
  losses; Omega-only logs correctly report a zero Human component.
- Random-background Human RGB is read through human_image_override_root, while
  aligned Human state/action remains the source of supervision.
- Train/validation file lists are split by episode; background variants reuse
  those same lists and cannot cross the split.
- The previous checkpoint-finetune runs are stopped and preserved separately
  under /mnt/nvme0n1/human_policy_ckpt/object_transfer_20261102. They are not
  resumed or used to initialize the scratch runs.

## Known limitations / TODO

- Labels are intentionally sparse; they do not justify claims about contact
  point accuracy.
- Future wrist displacement supplies the motion-direction auxiliary signal.
  The explicit current-frame object-to-hand relation uses detector-region
  centers; it must not be reported as exact wrist-to-object geometry.
- No phase head is enabled because a reliable shared phase label was not found.
- Wrist-to-image projection is not fused because the two camera projection
  chains have not passed a common pixel-error audit.
- Training completion and offline B0/B1/new-model action comparisons remain
  pending. A successful smoke run is not evidence of task success.

