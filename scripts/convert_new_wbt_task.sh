#!/bin/bash
# Full 3-step conversion pipeline for one new UnifoLM_WBT task.
# Usage: convert_new_wbt_task.sh <task_name> <hand_mode: brainco|inspire>
set -euo pipefail

TASK="$1"
HAND="$2"
PY=/home/aigc/miniconda/envs/human_policy/bin/python3
ROOT=/root/shengyin/human_policy
DATASET=/root/shengyin/DATASETS/UnifoLM_WBT/$TASK
XQY_DIR=/root/shengyin/DATASETS/XQY_PKL
OUT_DIR=/root/shengyin/DATASETS/human_policy/convert_UnifoLM_WBT

cd "$ROOT"

if [ "$HAND" = "brainco" ]; then
  FINGER_ARGS="--local-frame-correction brainco_rx90_ry180"
  FINGER_SUBDIR="finger_keypoints_rx90_ry180"
  HDF5_SUBDIR="hdf5_XQY3_rx90_ry180"
elif [ "$HAND" = "inspire" ]; then
  FINGER_ARGS="--reverse --inspire-thumb-mode swap"
  FINGER_SUBDIR="finger_keypoints_reverse_thumb_swap"
  HDF5_SUBDIR="hdf5_XQY3_reverse_thumb_swap"
else
  echo "Unknown hand mode: $HAND" >&2
  exit 1
fi

echo "===== $TASK ($HAND) : step0 resample XQY_PKL 60hz->30hz ====="
$PY resample_xqy_pkl_60hz_to_30hz.py \
  --dataset "$DATASET" \
  --xqy-dir "$XQY_DIR"

echo "===== $TASK : step1 compute_finger_keypoints ====="
$PY compute_finger_keypoints.py \
  --dataset "$DATASET" \
  $FINGER_ARGS \
  --out-dir "$DATASET/$FINGER_SUBDIR"

echo "===== $TASK : step2 convert_unifolm_to_hdf5 ====="
$PY convert_unifolm_to_hdf5.py \
  --dataset "$DATASET" \
  --mode XQY3 \
  --xqy3-wrist-rot-frame raw \
  --finger-dir "$DATASET/$FINGER_SUBDIR" \
  --xqy-dir "$XQY_DIR" \
  --out-dir "$DATASET/$HDF5_SUBDIR"

echo "===== $TASK : step3 translate_hdf5_head_to_origin ====="
$PY translate_hdf5_head_to_origin.py \
  --hdf5-dir "$DATASET/$HDF5_SUBDIR" \
  --xqy-prefix "$TASK" \
  --out-dir "$OUT_DIR" \
  --overwrite

echo "===== $TASK : cleanup intermediate step2 hdf5 dir ====="
rm -rf "$DATASET/$HDF5_SUBDIR"

echo "===== $TASK : DONE ====="
