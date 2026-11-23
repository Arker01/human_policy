# G1 / Dex3 conversion geometry

These are unmodified official Unitree URDFs used by
`scripts/preprocess/convert_unitree_dex3.py`. Meshes are unnecessary for forward
kinematics and are not included; these copies are not standalone rendering assets.
Exact upstream revisions, URLs and SHA256 hashes are in `sources.json`.

- Full robot: [unitree_ros](https://github.com/unitreerobotics/unitree_ros/blob/3ae893d905d11795ddf1075825ee5010364a1109/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf), BSD 3-Clause (`LICENSE`).
- Fingertip offsets: [xr_teleoperate](https://github.com/unitreerobotics/xr_teleoperate/tree/817fb00c63cde15e5f24a0f8fa08e1e33ed89d3b/assets/unitree_hand), Apache 2.0 (`LICENSE.xr_teleoperate`).

## Representation

`action[t]` is FK of the upstream **command** `action[t]`.
`observation.state[t]` is FK of the upstream measured `observation.state[t]`.
The converter applies no additional temporal shift, downsampling, joint clipping,
or normalization. Joint names from `meta/info.json` determine the mapping: the
left hand lists middle before index; the right hand lists index before middle.

Pose blocks follow `hdt/constants.py`:

| Indices | Meaning |
| --- | --- |
| 0:9 | Fixed nominal head reference: zero translation, identity rotation |
| 10:28 | Left wrist-local points: wrist origin, thumb, index, middle, ring, little |
| 30:39 | Right `wrist_yaw_link` pose |
| 40:58 | Right wrist-local points in the same order |
| 58:67 | Unobserved head tracker: zero translation, identity rotation placeholder |
| 80:89 | Left `wrist_yaw_link` pose |
| 89:98 | Fixed nominal torso pose |
| 98:128 | Zero: full-body robot configuration was not measured |

Positions are in metres, with pelvis/torso axes (forward X, left Y, up Z). The
origin is the **nominal URDF d435 mounting position**, `[0.05366, 0.01753, 0.47387]`
in the neutral pelvis frame. This only translates the origin; it does not apply
the d435 optical rotation. The dataset supplies no base/waist/head trajectory or
external-camera extrinsics. Legs and waist are held at neutral for FK; the head
reference is fixed to the torso. These assumptions are recorded on every HDF5.
They are not measured head tracking or calibrated camera poses.

Rotations use the first **two rows** of the rotation matrix, as in
`pytorch3d.transforms.matrix_to_rotation_6d` and `hdt/action_representation.py`.
The raw G1 wrist axes are preserved. Finger points use the same wrist axes;
fingertips are the distal-link origin plus the official fixed tip transform.

Dex3 has no ring or little fingers. Those slots are zero placeholders;
`hand_keypoint_valid` and `representation_valid` identify available channels.
`head_tracker_valid` and `waist_valid` are false. The existing trainer does not
automatically use these audit masks to change its loss; missing channels remain
constant targets. The raw 28-D joints are stored under `source.action` and
`source.observation.state`, not squeezed into a different robot's qpos layout.

## Images

| Source video | HDF5 key |
| --- | --- |
| `observation.images.cam_left_high` | `observation.image.left` |
| `observation.images.cam_right_high` | `observation.image.right` |
| `observation.images.cam_left_wrist` | `observation.image.left_wrist` |
| `observation.images.cam_right_wrist` | `observation.image.right_wrist` |

Apple has two head cameras; stacking and pouring also have both wrist cameras.
Training uses **one head view** with `common.camera_names: [top]` in the model
YAML. Every converted episode exposes `observation.image.top` as an HDF5 hard
link to `observation.image.left`, whose provenance is `cam_left_high`.
`observation.image_length.top` likewise links to the left-head JPEG lengths.
These are aliases, with no second copy of the image data. Attributes
`training_camera=top`, `training_camera_source=observation.images.cam_left_high`
and `camera_aliases` make this choice explicit. Current observations, history
and future-image targets all follow the same configured camera list.
The additional stored cameras serve inspection/export; the dataset JSON selects
episodes, while the **model YAML** selects training cameras. Native 30Hz timing
is preserved.

Frames are cut from the concatenated v3 video files using per-episode camera
metadata. Decoded PTS must match every requested frame; missing frames cause a
failure instead of padding or silently shortening an episode.

Default images are 320x240, JPEG quality 80, stored as padded `(T,max_bytes)` uint8
arrays with actual lengths in `observation.image_length.<cam>`.
**The project's dataloader decodes JPEGs without BGR2RGB**, including current,
history and future images. To supply RGB tensors to that reader, this converter
passes RGB arrays directly to `cv2.imencode`. Consequently `cv2.imdecode` yields
RGB directly; do not apply BGR2RGB when reading these files. Standard image
viewers/Pillow interpret these JPEG byte streams with red and blue reversed.
The convention is recorded in the `image_decode_color_order` attribute.

## Corrupt telemetry

Upstream measured hand positions contain three isolated impossible values:

| Task | Episode | Frame | Joint | Original radians |
| --- | --- | --- | --- | --- |
| pick_apple | 103 | 211 | kRightHandMiddle0 | 428.894806 |
| pick_apple | 193 | 444 | kRightHandMiddle0 | -1151.636353 |
| pouring | 115 | 54 | kRightHandMiddle0 | -2991.563232 |

Only these observation samples, detected by `abs(q)>2*pi`, are interpolated from
their adjacent valid samples before FK. `source.observation.state` preserves the
original values; `observation.state_joint_repaired` records the exact locations.
Commands are unchanged. Boundary/consecutive corrupt samples fail conversion.

The upstream Apple task label says `Pick up the red cup on the table.` despite
the dataset name. The converter preserves the upstream label in `description`;
language-conditioned training should review that label first.
