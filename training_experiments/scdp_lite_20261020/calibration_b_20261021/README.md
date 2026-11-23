# Offline Vision Pro screen-recording equivalent camera, scheme B

Only additions inside this directory. Original RGB, pose JSON, HDF, policy and
shared training environment were not modified. CPU detector/optimization runs in
the remote directory's own `venv/`. This does not enable SCDP training.

Primary result: `human_new_equivalent_camera.json`, calibrated for stretched
320x240 stored images from the new-human recordings. This is an **equivalent
screen-recording projection**, not a physical camera calibration.

Fitted parameters: fx=173.54497, fy=224.12747, cx=194.04046, cy=164.56746.
The JSON contains both directions of the full fixed optical/head transform.
It expects original ARKit head coordinates, positive camera Z forward, metres.
The fitted additional pose time offset is +0.09742822 seconds: video frame i
corresponds to pose timestamp[i]+delta. The existing trimmed video/JSON alignment
is retained; frameIndex of the original recording is not reset again.

The 1920x1080 recording is stretched to 320x240 by the existing data converter.
Consequently fx and fy need not match. Encoder resize/crop is a separate mapping
and must still be applied by SCDP. Extrinsics in this result map ARKit HEAD to
optical coordinates; they are not directly the HAT KP-frame extrinsics:

```
T_optical_from_kp = T_optical_from_head @ inverse(T_V_from_head_at_video_time) @ T_V_from_kp
```

No wrist pose is used to construct this camera transform. The original ARKit head
must be used, with its temporal alignment; a head-mounted tracker's pose has a
different fixed frame. Integration into the training loader and inference timing
has not been validated in this task.

Automatic correspondences use MediaPipe HandLandmarker in native-aspect 960x540
recording frames, mapped to 320x240. Only detector score >=0.8, valid ARKit wrists
and continuous nearby pose windows are retained. Five fingertip mappings were
also tested (ARKit 4/9/14/19/24 to MediaPipe 4/8/12/16/20); their joint fit did not
meet the requested accuracy. The selected model is explicitly a **right-wrist
fit**, with no claim that all hand landmarks are calibrated.

Fit recordings: 155,156,158,160,165,170,180,185, with 119 accepted wrist frames.
Initial held-out recordings: 157,161,174,190, with 76 accepted wrist frames.
After selecting wrist fitting, parameters were locked. A fresh holdout used
167,176,182,188,195,202, with 85 accepted wrist frames, without refitting:

| Set | Mean px | Median px | P90 px | Within 5 px |
|---|---:|---:|---:|---:|
| Fit | 2.847 | 2.544 | 5.303 | 86.6% |
| Initial holdout | 3.826 | 3.469 | 5.960 | 77.6% |
| Fresh holdout | 3.933 | 3.566 | 6.801 | 72.9% |

Fresh holdout fingertip diagnostic mean is 13.103 px; it fails the all-landmark
criterion. Anatomical correspondence, occlusion and ARKit/detector discrepancies
have not been separated. No manually annotated ground-truth test was performed.
Several overlays were visually inspected, which is not equivalent to independent
manual annotation. The left hand was not observed in the accepted correspondences.

Leaving each fit recording out gives wrist projection mean changes of 0.30–0.84
px on the fresh holdout. Intrinsic values vary more (fx 153.6–182.3; fy 201.1–235.7;
cx 187.4–201.8; cy 155.6–171.2), so these parameters should not be interpreted as
uniquely recovered physical optics. Time offsets range 93.98–100.16 ms.

Outputs:

- `human_new_equivalent_camera.json`: reusable scoped candidate, transforms/time.
- `validation.json`: fresh holdout and leave-one-recording-out results.
- `fresh_wrist_overlays/`: selected held-out RGB frames, green detection/red projection.
- `fresh_holdout_overlays/`: full hand diagnostic overlays, including failed fingers.
- `detections_*.json`: automatic 2D correspondences and source paths.
- `*_residuals.npz`: per-point residuals for the fitted models.

The old-session raw-recording probe failed (time/geometry parameters hit bounds).
Those raw pose/video pairs are not the exact `/tmp/0907_input` files referenced
by the original HAT HDF metadata, so this result must not be blindly applied to
`human_old`. Robot/ZED and left-hand calibration remain outside the validated scope.

Reproduce on 4090-168-aigc, from this directory:

```bash
OMP_NUM_THREADS=4 venv/bin/python detect_hands.py --episodes 155 156 158 160 165 170 180 185 157 161 174 190 --frames 40
OPENBLAS_NUM_THREADS=4 venv/bin/python fit_camera.py --detections detections_155_156_158_160_165_170_180_185_157_161_174_190.json --train 155 156 158 160 165 170 180 185 --wrist-only --name wrist_optimized_time
OMP_NUM_THREADS=4 venv/bin/python detect_hands.py --episodes 167 176 182 188 195 202 --frames 35
OPENBLAS_NUM_THREADS=4 venv/bin/python validate_camera.py
venv/bin/python test_geometry.py
```
