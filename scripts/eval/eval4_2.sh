#!/bin/bash

# 评估 convert2_with_pick 模型
GT_DIR="/data1/zxlei/dataset/part2/convert2"
POLICY_CKPT="/data1/zxlei/model/data1_legacy_from_1000-fine_ckpt/policy_last.ckpt"
POLICY_CONFIG="/home/embodied/human-policy/hdt/configs/models/act_resnet.yaml"
NORM_STATS="/data1/zxlei/model/convert2_with_pick_ckpt/dataset_stats.pkl"
OUT_JSON="/data1/zxlei/model/data1_legacy_from_1500_mpjpe_eval_3.json"

DEVICE="cuda:1"

# 运行评估
python /home/embodied/human-policy/data/eval_mpjpe_batch.py \
    --gt-dir "$GT_DIR" \
    --policy-ckpt "$POLICY_CKPT" \
    --policy-config-yaml "$POLICY_CONFIG" \
    --norm-stats "$NORM_STATS" \
    --device "$DEVICE" \
    --out-json "$OUT_JSON"