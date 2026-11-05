#!/usr/bin/env bash
set -euo pipefail

WORKTREE=/mnt/nvme0n1/human_policy_ckpt/classify_retrain_20261105/worktree
EXPERIMENT=/mnt/nvme0n1/human_policy_ckpt/classify_retrain_20261105
BASELINE=/mnt/nvme0n1/human_policy_ckpt/bg_ablation_5way_20261029
PYTHON=/home/aigc/miniconda/envs/human_policy/bin/python
SOURCE_MANIFEST="$BASELINE/configs/classify_gadgets.random_omega_only.dataset.json"
TRIM_MANIFEST="$EXPERIMENT/data/classify_gadgets.placement_trim.dataset.json"

mkdir -p "$EXPERIMENT/checkpoints" "$EXPERIMENT/logs" "$EXPERIMENT/data"
if [[ ! -f "$TRIM_MANIFEST" ]]; then
  "$PYTHON" "$WORKTREE/hdt/build_classify_placement_trim.py" \
    --source-manifest "$SOURCE_MANIFEST" \
    --output-dir "$EXPERIMENT/data/placement_trim" \
    --output-manifest "$TRIM_MANIFEST" \
    --settle-frames 15 \
    > "$EXPERIMENT/logs/build_trim.log" 2>&1
fi
cd "$WORKTREE"

launch_one() {
  local name=$1 manifest=$2 config=$3 gpu=$4 use_lag_aug=$5
  local pid_file="$EXPERIMENT/logs/$1.pid"
  local -a augmentation_args=(--visual_aug)
  if [[ "$use_lag_aug" == true ]]; then
    augmentation_args+=(--lag_aug)
  fi
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
    echo "$name already running pid=$(cat "$pid_file")"
    return
  fi
  nohup env \
    CUDA_VISIBLE_DEVICES="$gpu" \
    HDT_EXPERIMENT=geometry \
    PYTHONPATH="$WORKTREE:$WORKTREE/hdt" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
    TOKENIZERS_PARALLELISM=false \
    "$PYTHON" -u hdt/main.py \
    --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size 100 --seed 0 \
    --exptid "$EXPERIMENT/checkpoints/$name" \
    --dataset_json_path "$manifest" \
    --model_cfg_path "$config" \
    --base_dir /mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat \
    --cond_mask_prob 0.0 --no_wandb --human_slow_down_factor 4 \
    --use_future_dino_head --future_dino_weight 0.0 \
    --future_dino_warmup_steps 0 --future_dino_horizon 45 \
    --future_dino_ablation none --robot_target_fps 30 \
    --action_representation original "${augmentation_args[@]}" \
    --zero_state_dims 98:128 \
    >> "$EXPERIMENT/logs/${name}.log" 2>&1 < /dev/null &
  echo $! > "$pid_file"
  echo "$name launched on GPU $gpu pid=$(cat "$pid_file")"
}

launch_one gadgets_DH_ordered "$SOURCE_MANIFEST" \
  hdt/configs/classify_retrain_20261105/gadgets_DH_ordered.yaml 5 false
launch_one gadgets_D_trimmed "$TRIM_MANIFEST" \
  hdt/configs/classify_retrain_20261105/gadgets_D_trimmed.yaml 6 true
