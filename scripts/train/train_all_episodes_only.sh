#!/bin/bash

# 训练配置
batch_size=64
learning_rate=1e-5
chunk_size=100
expt_id="train_all_episodes_only"

# 数据和模型配置
model_cfg_path="hdt/configs/models/act_resnet.yaml"
base_dir="/home/aigc/human_policy/data"

# 输出路径
output_dir="/home/aigc/human_policy/data/${expt_id}_ckpt"
mkdir -p "$output_dir"

# ==========================================
# 训练配置: 仅用 all_episodes
# ==========================================
cat > /tmp/all_episodes_train.json << 'EOF'
{
  "train": [
    {
      "dataset_path": "/home/aigc/human_policy/data/all_episodes",
      "type": "human",
      "start_idx": 10,
      "end_idx": 95
    }
  ],
  "val": [
    {
      "dataset_path": "/home/aigc/human_policy/data/all_episodes",
      "type": "human",
      "start_idx": 0,
      "end_idx": 10
    }
  ]
}
EOF

echo "=========================================="
echo "仅用 all_episodes 训练"
echo "=========================================="
echo "  Train: all_episodes files 10-94 (85 files)"
echo "  Val:   all_episodes files 0-9  (10 files)"
echo "  lr=$learning_rate, 50000 epoch"
echo "  Output: $output_dir"
echo "=========================================="

CUDA_VISIBLE_DEVICES=2 python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 50000 \
    --lr $learning_rate \
    --chunk_size $chunk_size \
    --exptid "$expt_id" \
    --dataset_json_path /tmp/all_episodes_train.json \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --no_wandb

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "模型输出: $output_dir"
echo "=========================================="
