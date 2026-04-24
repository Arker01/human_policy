#!/bin/bash

# 从 convert2_1000 模型微调 - data1_legacy数据
BATCH_SIZE=64
LEARNING_RATE=5e-5  # 微调学习率
CHUNK_SIZE=100
EXPT_ID="/data1/zxlei/model/data1_legacy_from_1000"
DATASET_JSON="/home/embodied/human-policy/data/data1_legacy.json"
MODEL_CFG="/home/embodied/human-policy/hdt/configs/models/act_resnet.yaml"
BASE_DIR="/data1/zxlei/dataset/"
LOAD_PRETRAINED_PATH="/data1/zxlei/model/convert2_with_pick_ckpt/policy_last.ckpt"

# 创建模型保存目录
mkdir -p "$EXPT_ID"

# 运行训练命令
python /home/embodied/human-policy/hdt/main.py \
    --batch_size $BATCH_SIZE \
    --num_epochs 10000 \
    --lr $LEARNING_RATE \
    --chunk_size $CHUNK_SIZE \
    --exptid "$EXPT_ID" \
    --dataset_json_path "$DATASET_JSON" \
    --model_cfg_path "$MODEL_CFG" \
    --base_dir "$BASE_DIR" \
    --load_pretrained_path "$LOAD_PRETRAINED_PATH" \
    --no_wandb