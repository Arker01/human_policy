#!/usr/bin/env bash
# Future-DINO world head (WAM) training run.
#
# There was no launcher for this before -- which is a large part of why the head was
# never actually trained. Everything Future-DINO-specific lives in MODEL_CFG
# (hdt/configs/models/act_with_future_dino.yaml); the flags below only override it.
#
# lambda_world sweep (doc S7): pass the weight as $2, e.g.
#   scripts/train/train_future_dino.sh 0 0.0     # baseline, head attached but silent
#   scripts/train/train_future_dino.sh 1 0.1
#   scripts/train/train_future_dino.sh 2 0.3
#   scripts/train/train_future_dino.sh 3 1.0
# Ablations (doc S9): append 'shuffled' or 'current' as $3.
#
# Sanity check the first ~50 lines of output: train/future_dino_cosine_loss must fall
# below its ~1.0 starting value and future_dino_effective_weight must equal the
# configured weight from step 0. If either stays flat, the head is dead again.
#
# No warmup: this is from-scratch training, and the measured world-loss contribution to
# the shared trunk gradient is ~1% of the trajectory contribution at weight=0.3, so
# there is nothing to ramp in. See act_with_future_dino.yaml warmup_steps.
set -euo pipefail

gpu="${1:-0}"
weight="${2:-0.3}"
ablation="${3:-none}"
dataset_json="${4:-/home/aigc/human_policy/pillow_robot.json}"

cd /home/aigc/human_policy

PY=/home/aigc/miniconda/envs/human_policy/bin/python
# Defaults to pillow_robot.json (data/dex5_train + data/dex5_val) so this run is a
# like-for-like A/B against train_pillow_robot_ckpt, whose only difference is the
# absent world head. The previous default pointed at /tmp/episodes_45_95_train.json,
# which no longer exists.
DATASET_JSON="$dataset_json"
MODEL_CFG=hdt/configs/models/act_with_future_dino.yaml
BASE_DIR=/home/aigc/human_policy/data

exptid="future_dino_w${weight}_${ablation}"
log_path="logs/${exptid}.log"
mkdir -p "$(dirname "$log_path")"

args=(
  hdt/main.py
  --batch_size 64
  --num_epochs 100000
  --lr 1e-5
  --chunk_size 100
  --exptid "$exptid"
  --dataset_json_path "$DATASET_JSON"
  --model_cfg_path "$MODEL_CFG"
  --base_dir "$BASE_DIR"
  --cond_mask_prob 0.0
  --no_wandb
  --use_future_dino_head
  --future_dino_weight "$weight"
  --future_dino_warmup_steps 0
  --future_dino_horizon 16
  --future_dino_ablation "$ablation"
)

echo "[$(date -Is)] exptid=$exptid gpu=$gpu weight=$weight ablation=$ablation" | tee "$log_path"
env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${args[@]}" 2>&1 | tee -a "$log_path"
