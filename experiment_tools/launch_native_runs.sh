#!/usr/bin/env bash
set -euo pipefail

EXP=/mnt/nvme0n1/human_policy_ckpt/sail_ablation_20261109
WORKTREE="$EXP/worktree"
PYTHON=/home/aigc/miniconda/envs/human_policy/bin/python

launch_one() {
  local name=$1 dataset=$2 model=$3 variant=$4 chunk=$5 gpu=$6
  local ckpt="$EXP/checkpoints/${name}_ckpt"
  local pid_file="$EXP/status/${name}.pid"
  if [[ -e "$ckpt/policy_last.ckpt" ]]; then
    echo "$name complete; refusing to overwrite $ckpt"
    return
  fi
  if [[ -f "$pid_file" ]] && kill -0 "$(<"$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(<"$pid_file")"
    return
  fi
  if [[ -d "$ckpt" ]] && find "$ckpt" -mindepth 1 -print -quit | grep -q .; then
    echo "$name has a non-empty checkpoint directory; refusing implicit resume/overwrite" >&2
    return 1
  fi
  mkdir -p "/tmp/sail_ablation_${name}" "$EXP/cache/$name"
  nohup env CUDA_VISIBLE_DEVICES="$gpu" HDT_EXPERIMENT="$variant" \
    PYTHONPATH="$WORKTREE:$WORKTREE/hdt" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false TMPDIR="/tmp/sail_ablation_${name}" \
    MPLCONFIGDIR="$EXP/cache/$name" \
    "$PYTHON" -u "$WORKTREE/hdt/main.py" \
      --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size "$chunk" --seed 0 \
      --exptid "$EXP/checkpoints/$name" \
      --dataset_json_path "$EXP/configs/$dataset" \
      --model_cfg_path "$EXP/configs/$model" \
      --base_dir /mnt/nvme0n1/DATASETS \
      --cond_mask_prob 0.0 --no_wandb --human_slow_down_factor 4 \
      --use_future_dino_head --future_dino_weight 0.0 \
      --future_dino_warmup_steps 0 --future_dino_horizon 45 \
      --future_dino_ablation none --robot_target_fps 30 \
      --action_representation original --lag_aug --visual_aug \
      --zero_state_dims 98:128 \
      >> "$EXP/logs/${name}.log" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  printf '%s\n' "$(date --iso-8601=seconds) name=$name variant=$variant gpu=$gpu pid=$(<"$pid_file")" \
    > "$EXP/status/${name}.launch.txt"
  echo "$name launched on GPU $gpu pid=$(<"$pid_file")"
}

launch_one walk_sonic_native_d0 walk_sonic_native_d0.dataset.json model_sonic_native.yaml sonic_native 167 0
launch_one turn_sonic_native_d0 turn_sonic_native_d0.dataset.json model_sonic_native.yaml sonic_native 167 1
launch_one walk_hat_native_d0 walk_d0.dataset.json model_no_dropout.yaml hat_native 100 2
launch_one turn_hat_native_d0 turn_d0.dataset.json model_no_dropout.yaml hat_native 100 3
