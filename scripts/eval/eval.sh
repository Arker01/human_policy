#!/bin/bash

# 评估 convert2_with_pick 模型
GT_DIR="/home/aigc/human_policy/data/all_episodes"
POLICY_CKPT="/home/aigc/human_policy/train_all_episodes_finetune_finetune_all_episodes_ckpt/policy_last.ckpt"
POLICY_CONFIG="/home/aigc/human_policy/hdt/configs/models/act_resnet.yaml"
NORM_STATS="/home/aigc/human_policy/train_all_episodes_finetune_finetune_all_episodes_ckpt/dataset_stats.pkl"
OUT_JSON="all_episodes_mpjpe_eval_3.json"

DEVICE="cuda:0"

# 运行评估
python /home/aigc/human_policy/data/eval_mpjpe_batch.py \
    --gt-dir "$GT_DIR" \
    --policy-ckpt "$POLICY_CKPT" \
    --policy-config-yaml "$POLICY_CONFIG" \
    --norm-stats "$NORM_STATS" \
    --device "$DEVICE" \
    --out-json "$OUT_JSON"