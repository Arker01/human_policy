#!/usr/bin/env bash
set -euo pipefail

gpu="$1"
exptid="$2"
lr="$3"
query0_weight="$4"
log_path="$5"

cd /home/aigc/human_policy

PY=/home/aigc/miniconda/envs/human_policy/bin/python
DATASET_JSON=/tmp/episodes_45_95_train.json
MODEL_CFG=hdt/configs/models/act_resnet.yaml
BASE_DIR=/home/aigc/human_policy/data

mkdir -p "$(dirname "$log_path")"

args=(
  hdt/main.py
  --batch_size 64
  --num_epochs 100000
  --lr "$lr"
  --chunk_size 100
  --exptid "$exptid"
  --dataset_json_path "$DATASET_JSON"
  --model_cfg_path "$MODEL_CFG"
  --base_dir "$BASE_DIR"
  --cond_mask_prob 0.0
  --no_wandb
)

if [[ "$query0_weight" != "0" ]]; then
  args+=(--query0_extra_weight "$query0_weight")
fi

echo "[$(date -Is)] Starting exptid=$exptid gpu=$gpu lr=$lr query0_weight=$query0_weight" | tee "$log_path"
env CUDA_VISIBLE_DEVICES="$gpu" "$PY" "${args[@]}" 2>&1 | tee -a "$log_path"
