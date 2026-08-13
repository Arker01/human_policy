#!/usr/bin/env bash
set -euo pipefail

cd /home/aigc/human_policy

PY=/home/aigc/miniconda/envs/human_policy/bin/python
DATASET_JSON=/tmp/episodes_45_95_train.json
MODEL_CFG=hdt/configs/models/act_resnet.yaml
BASE_DIR=/home/aigc/human_policy/data

mkdir -p logs

common_args=(
  --batch_size 64
  --num_epochs 100000
  --chunk_size 100
  --dataset_json_path "$DATASET_JSON"
  --model_cfg_path "$MODEL_CFG"
  --base_dir "$BASE_DIR"
  --cond_mask_prob 0.0
  --no_wandb
)

nohup env CUDA_VISIBLE_DEVICES=0 "$PY" hdt/main.py \
  "${common_args[@]}" \
  --lr 1e-5 \
  --exptid train_episodes_45_95_nomask_q0w05_100k \
  --query0_extra_weight 0.5 \
  > logs/run_100k_nomask_q0w05.log 2>&1 &
echo $! > logs/run_100k_nomask_q0w05.pid

nohup env CUDA_VISIBLE_DEVICES=1 "$PY" hdt/main.py \
  "${common_args[@]}" \
  --lr 1e-5 \
  --exptid train_episodes_45_95_nomask_q0w1_100k \
  --query0_extra_weight 1.0 \
  > logs/run_100k_nomask_q0w1.log 2>&1 &
echo $! > logs/run_100k_nomask_q0w1.pid

nohup env CUDA_VISIBLE_DEVICES=2 "$PY" hdt/main.py \
  "${common_args[@]}" \
  --lr 1e-5 \
  --exptid train_episodes_45_95_nomask_q0w2_100k \
  --query0_extra_weight 2.0 \
  > logs/run_100k_nomask_q0w2.log 2>&1 &
echo $! > logs/run_100k_nomask_q0w2.pid

nohup env CUDA_VISIBLE_DEVICES=3 "$PY" hdt/main.py \
  "${common_args[@]}" \
  --lr 3e-6 \
  --exptid train_episodes_45_95_nomask_lr3e6_100k \
  > logs/run_100k_nomask_lr3e6_ft.log 2>&1 &
echo $! > logs/run_100k_nomask_lr3e6_ft.pid

echo "Started 100k nomask sweep:"
cat logs/run_100k_nomask_q0w05.pid
cat logs/run_100k_nomask_q0w1.pid
cat logs/run_100k_nomask_q0w2.pid
cat logs/run_100k_nomask_lr3e6_ft.pid
