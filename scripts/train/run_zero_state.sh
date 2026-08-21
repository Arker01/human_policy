#!/usr/bin/env bash
# ab4 rerun with the dex5 robot-configuration block hidden from the policy input.
#
# ab4 is the best checkpoint so far (mixed 1:1 human+robot, lambda 1.0, horizon 45,
# DINOv2-S teacher, l2 target norm). This run changes exactly ONE thing: normalized
# qpos[98:128] is forced to 0, so the model can no longer read the block that only
# the robot half of the data has.
#
# What that block is: dex5 stores robot_q_current[0:26] there (root position 3 +
# root quaternion 4 + the first 19 of 29 joint angles), while every human episode
# has it at raw 0. Measured on ab4: replacing it with the training mean costs
# +36% MPJPE, so the policy really does use it -- this run asks whether that
# dependence is worth the deployment cost of having to reproduce the block exactly
# on the real robot.
#
# 98 rather than 100: dims 98 and 99 are already identically 0 in both embodiments,
# so the two ranges are equivalent; 98:128 just matches the block boundary.
#
# GPU 7 is the one the round-2 launcher deliberately left free.
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt

GPU="${1:-7}"
EXPTID="${2:-r3_ab4_nostate}"
# The config must already carry model.zero_state_dims, because eval rebuilds from it
# (the mask is a ckpt buffer and strict=False would drop it silently). Two exist:
#   act_with_future_dino_nostate.yaml  -- ab4 arm  (resnet18 trunk)
#   act_input_vjepa_nostate.yaml       -- vjepa arm (frozen V-JEPA 2.1 ViT-B trunk)
CFG="${3:-hdt/configs/models/act_with_future_dino.yaml}"

mkdir -p logs/zero_state "$CKPT_ROOT/${EXPTID}_ckpt"
[[ -e "${EXPTID}_ckpt" ]] || ln -s "$CKPT_ROOT/${EXPTID}_ckpt" "${EXPTID}_ckpt"

log="logs/zero_state/${EXPTID}.log"
echo "[$(date -Is)] gpu=$GPU exptid=$EXPTID cfg=$(basename "$CFG") zero_state_dims=98:128" | tee "$log"

CUDA_VISIBLE_DEVICES="$GPU" nohup "$PY" hdt/main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-5 \
  --chunk_size 100 \
  --exptid "$EXPTID" \
  --dataset_json_path "$MIXED" \
  --model_cfg_path "$CFG" \
  --base_dir "$BASE_DIR" \
  --cond_mask_prob 0.0 \
  --no_wandb \
  --use_future_dino_head \
  --future_dino_weight 1.0 \
  --future_dino_warmup_steps 0 \
  --future_dino_horizon 45 \
  --future_dino_ablation none \
  --zero_state_dims 98:128 \
  >> "$log" 2>&1 &

echo "  -> pid $! log $log"
