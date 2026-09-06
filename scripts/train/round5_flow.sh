#!/usr/bin/env bash
# Round 5: EgoWAM's 3D point-flow world head on the resnet18 trunk.
#
# One arm only, exactly the paper's setup: same trunk / action head / data mixture
# as ab1 and ab4, the ONLY change is the world target (none -> 3D flow).
#
#   r5_flow   resnet18 + flow lambda=1.0, no DINO head
#
# Compares against ab1 (lambda=0, clean 44.78) and ab4 (DINO, clean 43.49). The
# clean axis is the point: it has not moved in 22 runs, and flow is the only thing
# EgoWAM claims moves it (20-30% in-domain).
#
# Cost: 2.89 it/s on this trunk -> 100k steps ~= 10h.
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json
CFG_DIR=hdt/configs/models
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt
FLOW_DIR=/mnt/nvme0n1/human_policy_ckpt/flow_target_dex5
mkdir -p logs/round5 "$CKPT_ROOT"

# Same guard as round 2/4: a half-written target set mid-run is a confusing failure.
n=$(ls "$FLOW_DIR"/*.flow.h5 2>/dev/null | wc -l)
if [ "$n" -lt 207 ]; then echo "flow target 不全: $n/207 in $FLOW_DIR"; exit 1; fi

gpu=0
exptid=r5_flow
mkdir -p "$CKPT_ROOT/${exptid}_ckpt"
[ -e "${exptid}_ckpt" ] || ln -s "$CKPT_ROOT/${exptid}_ckpt" "${exptid}_ckpt"

log="logs/round5/${exptid}.log"
echo "[$(date -Is)] gpu=$gpu $exptid flow_dir=$FLOW_DIR" | tee "$log"

CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" hdt/main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-5 \
  --chunk_size 100 \
  --exptid "$exptid" \
  --dataset_json_path "$MIXED" \
  --model_cfg_path "$CFG_DIR/act_with_future_flow.yaml" \
  --base_dir "$BASE_DIR" \
  --cond_mask_prob 0.0 \
  --no_wandb \
  --use_future_flow_head \
  --future_flow_weight 1.0 \
  --future_flow_dir "$FLOW_DIR" \
  >> "$log" 2>&1 &

echo "  -> pid $! log $log"
