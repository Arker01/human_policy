#!/usr/bin/env bash
# Single run: Future-DINO head with a V-JEPA 2.1 VIDEO teacher instead of DINOv2.
#
# This is the "next rung" of the staircase in ablation_8gpu.sh: everything matches
# ab4 (mixed 1:1, lambda=1.0, horizon 45, l2 target norm) EXCEPT the teacher, so
# ab4 -> this run isolates "does a temporally-trained target encoder beat an
# image-only one". The motivation is measured, not aesthetic: under camera
# perturbation DINOv2-S scores cosine 0.517 between two views of the SAME state vs
# a 0.480 floor for two DIFFERENT states -- the target space we have been training
# against is nearly information-free exactly where we need generalization.
#
# One factor cannot be held fixed: a video teacher physically cannot be fed a single
# frame (its patch embed fuses tubelet=2 frames), so clip_frames goes 1 -> 2. That
# co-change is unavoidable and is called out in docs/FUTURE_DINO_CHANGES.md.
#
# Unlike the 8-GPU round, checkpoints land on /mnt/nvme0n1 from the start via a
# pre-created symlink -- main.py:253 builds ckpt_dir as a relative "<exptid>_ckpt"
# and main.py:367 only mkdir's it when it is not already a directory, so a symlink
# planted here is honoured. That is what kept the 2026-08-16 round from finishing.
#
# Usage: train_vjepa_teacher.sh [gpu] [exptid]
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt

gpu="${1:-0}"
exptid="${2:-vj0_mixed_w1.0_h45_vjepa}"
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json
CFG=hdt/configs/models/act_with_future_dino_vjepa.yaml

# The teacher weights are ~1.6GB and are fetched once into the torch hub cache.
# Downloading them inside the training process works, but a half-finished download
# on a shared machine is a confusing failure mode, so check up front.
WEIGHTS="$HOME/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt"
if [ ! -s "$WEIGHTS" ]; then
  echo "缺少 V-JEPA 2.1 权重: $WEIGHTS"
  echo "先下载: curl -L -o '$WEIGHTS' https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt"
  exit 1
fi

# Point <exptid>_ckpt at nvme before main.py ever looks at it.
mkdir -p "$CKPT_ROOT/${exptid}_ckpt" || { echo "无法创建 $CKPT_ROOT/${exptid}_ckpt"; exit 1; }
if [ ! -e "${exptid}_ckpt" ]; then
  ln -s "$CKPT_ROOT/${exptid}_ckpt" "${exptid}_ckpt"
elif [ ! -L "${exptid}_ckpt" ]; then
  echo "${exptid}_ckpt 已存在且不是软链接 -- 会写到 / 上，先处理掉再跑"; exit 1
fi

mkdir -p logs/vjepa
log="logs/vjepa/${exptid}.log"
echo "[$(date -Is)] gpu=$gpu exptid=$exptid cfg=$(basename $CFG) ckpt-> $(readlink -f ${exptid}_ckpt)" | tee "$log"

CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" hdt/main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-5 \
  --chunk_size 100 \
  --exptid "$exptid" \
  --dataset_json_path "$MIXED" \
  --model_cfg_path "$CFG" \
  --base_dir /home/aigc/human_policy/data \
  --cond_mask_prob 0.0 \
  --no_wandb \
  --use_future_dino_head \
  --future_dino_weight 1.0 \
  --future_dino_warmup_steps 0 \
  --future_dino_horizon 45 \
  --future_dino_ablation none \
  >> "$log" 2>&1 &

echo "  -> pid $! log $log"
