# 生成IK
# Pinocchio，会自动切到 human_policy conda 环境
python3 human_policy/batch_ik_cache.py \
  /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --backend pinocchio

# Pytorch，会自动切到 twist conda 环境
CUDA_VISIBLE_DEVICES=4 python3 human_policy/batch_ik_cache.py \
  /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --backend pytorch \
  --ik-device cuda:0

###


#!/usr/bin/env bash
set -euo pipefail

cd /media/magic-4090/DATA1/shengyin/human_policy

export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:${PATH}
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:${LD_LIBRARY_PATH:-}

COMMON_ARGS=(
  --gt-dir /media/magic-4090/DATA1/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16
  --policy-ckpt /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/policy_last.ckpt
  --policy-config-yaml /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/act_resnet.yaml
  --norm-stats /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/dataset_stats.pkl
  --device cuda:0
  --eval-mode first_token
  --hand-gmt-ckpt /media/magic-4090/DATA1/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt
  --gmt-device cuda:0
  --viewer
)

CUDA_VISIBLE_DEVICES=0 python twist_hand_gmt_bridge.py \
  "${COMMON_ARGS[@]}" \
  --hand-control-mode gmt \
  --ik-backend pytorch \
  --out-actions /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/eval-result/twist_hand_gmt_act_resnet_gmt_pytorch_actions.npz

CUDA_VISIBLE_DEVICES=0 python twist_hand_gmt_bridge.py \
  "${COMMON_ARGS[@]}" \
  --hand-control-mode gmt \
  --ik-backend pinocchio \
  --out-actions /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/eval-result/twist_hand_gmt_act_resnet_gmt_pinocchio_actions.npz

CUDA_VISIBLE_DEVICES=0 python twist_hand_gmt_bridge.py \
  "${COMMON_ARGS[@]}" \
  --hand-control-mode ik \
  --ik-backend pytorch \
  --out-actions /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/eval-result/twist_hand_gmt_act_resnet_ik_pytorch_actions.npz

CUDA_VISIBLE_DEVICES=0 python twist_hand_gmt_bridge.py \
  "${COMMON_ARGS[@]}" \
  --hand-control-mode ik \
  --ik-backend pinocchio \
  --out-actions /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/eval-result/twist_hand_gmt_act_resnet_ik_pinocchio_actions.npz
