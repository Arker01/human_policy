#!/bin/bash

# 训练参数设置
BATCH_SIZE=64
LEARNING_RATE=1e-4
CHUNK_SIZE=100
EXPT_ID="/data1/zxlei/model/convert2"
DATASET_JSON="/home/embodied/human-policy/data/convert2_with_val.json"
MODEL_CFG="/home/embodied/human-policy/hdt/configs/models/act_resnet.yaml"
BASE_DIR="/data1/zxlei/dataset/part2/"

# 创建模型保存目录
mkdir -p "$EXPT_ID"

# 运行训练命令
python /home/embodied/human-policy/hdt/main.py \
    --batch_size $BATCH_SIZE \
    --num_epochs 30000 \
    --lr $LEARNING_RATE \
    --chunk_size $CHUNK_SIZE \
    --exptid "$EXPT_ID" \
    --dataset_json_path "$DATASET_JSON" \
    --model_cfg_path "$MODEL_CFG" \
    --base_dir "$BASE_DIR" \
    --no_wandb