#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/mnt/nvme0n1/human_policy_ckpt/classify_retrain_20261105/worktree
EXPERIMENT=/mnt/nvme0n1/human_policy_ckpt/classify_retrain_20261105
PYTHON=/home/aigc/miniconda/envs/human_policy/bin/python
MANIFEST="$EXPERIMENT/data/classify_gadgets.first_object_open_boundary_post5.dataset.json"

test -f "$MANIFEST"
mkdir -p "$EXPERIMENT/checkpoints" "$EXPERIMENT/logs"
cd "$WORKTREE"

launch_one() {
  local name=$1 gpu=$2 config=$3
  local pid_file="$EXPERIMENT/logs/${name}.pid"
  local checkpoint="$EXPERIMENT/checkpoints/${name}_ckpt"
  local log="$EXPERIMENT/logs/${name}.log"
  local tmpdir="/tmp/classify_post5_${name}"
  mkdir -p "$tmpdir"
  if [[ -e "$checkpoint" || -e "$log" || -e "$pid_file" ]]; then
    echo "refusing to reuse existing output for $name" >&2
    return 1
  fi
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HDT_EXPERIMENT=geometry \
    PYTHONPATH="$WORKTREE:$WORKTREE/hdt" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    TMPDIR="$tmpdir" \
    "$PYTHON" -u hdt/main.py \
    --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size 100 --seed 0 \
    --exptid "$EXPERIMENT/checkpoints/$name" \
    --dataset_json_path "$MANIFEST" \
    --model_cfg_path "$config" \
    --base_dir /mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat \
    --cond_mask_prob 0.0 --no_wandb --human_slow_down_factor 4 \
    --use_future_dino_head --future_dino_weight 0.0 \
    --future_dino_warmup_steps 0 --future_dino_horizon 45 \
    --future_dino_ablation none --robot_target_fps 30 \
    --action_representation original --lag_aug --visual_aug \
    --zero_state_dims 98:128 \
    >> "$log" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  echo "$name launched on physical GPU $gpu pid=$(cat "$pid_file")"
}

launch_one gadgets_first_object_post5_D_geometry 0 \
  hdt/configs/classify_retrain_20261105/gadgets_D_trimmed.yaml
launch_one gadgets_first_object_post5_noD_geometry 1 \
  hdt/configs/classify_retrain_20261105/gadgets_post5_noD.yaml
