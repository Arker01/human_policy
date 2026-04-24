#!/bin/bash

# 训练配置
batch_size=64
learning_rate=1e-5
chunk_size=100
expt_id="pick_place_processed_only"

# 数据和模型配置
dataset_json_path="pick_place_processed_only.json"
model_cfg_path="hdt/configs/models/act_resnet.yaml"
base_dir="/data1/zxlei/dataset"

# 输出路径
output_dir="/data1/zxlei/model/${expt_id}_ckpt"
mkdir -p "$output_dir"

# 启动训练
echo "Starting training with processed-only pick and place dataset..."
echo "Using base directory: $base_dir"
echo "Output directory: $output_dir"

python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 50000 \
    --lr $learning_rate \
    --chunk_size $chunk_size \
    --exptid "$expt_id" \
    --dataset_json_path "$dataset_json_path" \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --no_wandb
