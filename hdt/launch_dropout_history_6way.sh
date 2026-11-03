#!/usr/bin/env bash
set -euo pipefail

# Isolated six-way launcher.  Re-running it leaves live jobs alone; completed or
# interrupted jobs resume from the latest accelerator checkpoint in their own dir.
WORKTREE=/mnt/nvme0n1/human_policy_ckpt/dropout_history_6way_20261103/worktree
EXPERIMENT=/mnt/nvme0n1/human_policy_ckpt/dropout_history_6way_20261103
BASELINE=/mnt/nvme0n1/human_policy_ckpt/bg_ablation_5way_20261029
PYTHON=/home/aigc/miniconda/envs/human_policy/bin/python

test -f "$EXPERIMENT/teacher_cache/wipe_table/manifest.json"
test -f "$EXPERIMENT/teacher_cache/classify_gadgets/manifest.json"
mkdir -p "$EXPERIMENT/checkpoints" "$EXPERIMENT/logs"
cd "$WORKTREE"

launch_one() {
  local name=$1 task=$2 gpu=$3 pid_file="$EXPERIMENT/logs/$1.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  nohup env CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u hdt/main.py \
    --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size 100 --seed 0 \
    --exptid "$EXPERIMENT/checkpoints/$name" \
    --dataset_json_path "$BASELINE/configs/${task}.random_omega_only.dataset.json" \
    --model_cfg_path "hdt/configs/dropout_history/${name}.yaml" \
    --base_dir /mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat \
    --cond_mask_prob 0.0 --no_wandb --human_slow_down_factor 4 \
    --use_future_dino_head --future_dino_weight 0.0 \
    --future_dino_warmup_steps 0 --future_dino_horizon 45 \
    --future_dino_ablation none --robot_target_fps 30 \
    --action_representation original --lag_aug --visual_aug \
    --zero_state_dims 98:128 \
    >> "$EXPERIMENT/logs/${name}.log" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  echo "$name launched on GPU $gpu pid=$(cat "$pid_file")"
}

launch_one wipe_D wipe_table 1
launch_one gadgets_D classify_gadgets 2
launch_one wipe_HP wipe_table 3
launch_one wipe_DHP wipe_table 4
launch_one gadgets_HP classify_gadgets 5
launch_one gadgets_DHP classify_gadgets 6
