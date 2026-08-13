#!/bin/bash

# 训练配置
batch_size=64
learning_rate=1e-5
finetune_lr=5e-6
chunk_size=100
expt_id="train_all_episodes_finetune"

# 数据和模型配置
model_cfg_path="hdt/configs/models/act_resnet.yaml"
base_dir="/home/aigc/human_policy/data"

# 输出路径
output_dir="/home/aigc/human_policy/data/${expt_id}_ckpt"
mkdir -p "$output_dir"

finetune_expt_id="${expt_id}_finetune_all_episodes"
finetune_output_dir="/home/aigc/human_policy/data/${finetune_expt_id}_ckpt"
mkdir -p "$finetune_output_dir"

# ==========================================
# 阶段1: 用 processed + ego_last 预训练
# ==========================================
cat > /tmp/pick_place_base_stage1.json << 'EOF'
{
  "train": [
    {
      "dataset_path": "processed/106-grasping-zedbox-human_2024-11-13_15-27-43",
      "type": "human"
    },
    {
      "dataset_path": "processed/107-grasping-chocolate-human_2024-11-13_15-15-09",
      "type": "human"
    },
    {
      "dataset_path": "processed/108-grasping-water-human _2024-11-13_15-00-20",
      "type": "human"
    },
    {
      "dataset_path": "processed/109-grasping-mtn-human_2024-11-13_14-40-07",
      "type": "human"
    },
    {
      "dataset_path": "processed/110-picking-dynamixel-human_2024-11-13_17-34-13",
      "type": "human"
    },
    {
      "dataset_path": "processed/112-picking-brownbox-human_2024-11-13_20-42-14",
      "type": "human"
    },
    {
      "dataset_path": "processed/113-picking-blackcube-human_2024-11-13_22-09-05",
      "type": "human"
    },
    {
      "dataset_path": "processed/1403-human_pick_color_pad_left_2025-01-13_13-05-06",
      "type": "human"
    },
    {
      "dataset_path": "processed/402-pick_on_color_pad_right-2025_01_09-16_36_15",
      "type": "human"
    },
    {
      "dataset_path": "processed/104-lars-grasping_2024-11-08_15-23-40",
      "type": "robot"
    },
    {
      "dataset_path": "processed/105_lars-grasping_2024-11-08_15-47-04",
      "type": "robot"
    },
    {
      "dataset_path": "processed/111-picking-colorful-toycube_2024-11-13_20-25-34",
      "type": "robot"
    },
    {
      "dataset_path": "processed/302-grasp_coke_random-2024_12_09-21_39_30",
      "type": "robot"
    },
    {
      "dataset_path": "processed/303-grasp_coke_random-2024_12_12-19_13_53",
      "type": "robot"
    },
    {
      "dataset_path": "processed/304-grasp_coke_random-2024_12_12-19_58_36",
      "type": "robot"
    },
    {
      "dataset_path": "processed/1401_grasping_three_items_2025-01-23_19-24-05",
      "type": "robot"
    },
    {
      "dataset_path": "/home/aigc/.cache/modelscope/hub/datasets/arker01/human_policy/convert_ego_last",
      "type": "human",
      "start_idx": 0,
      "end_idx": 800
    }
  ],
  "val": [
    {
      "dataset_path": "processed/903-picking-val-2024_11_18-18_58_16",
      "type": "mixed"
    },
    {
      "dataset_path": "/home/aigc/.cache/modelscope/hub/datasets/arker01/human_policy/convert_ego_last",
      "type": "human",
      "start_idx": 800,
      "end_idx": 1000
    }
  ]
}
EOF

# ==========================================
# 阶段2: 在 all_episodes 新数据集上 Finetune
# ==========================================
cat > /tmp/all_episodes_finetune.json << 'EOF'
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

# ==========================================
# 阶段1: 预训练
# ==========================================
echo "=========================================="
echo "阶段1: processed + ego_last 预训练..."
echo "=========================================="
echo "Output directory: $output_dir"
echo "Learning rate: $learning_rate"
echo "Config: /tmp/pick_place_base_stage1.json"

CUDA_VISIBLE_DEVICES=1 python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 50000 \
    --lr $learning_rate \
    --chunk_size $chunk_size \
    --exptid "$expt_id" \
    --dataset_json_path /tmp/pick_place_base_stage1.json \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --no_wandb

# ==========================================
# 阶段2: 在 all_episodes 上 Finetune
# ==========================================
echo ""
echo "=========================================="
echo "阶段2: 在 all_episodes 上 Finetune..."
echo "=========================================="

latest_ckpt="/home/aigc/human_policy/train_all_episodes_finetune_ckpt/policy_last.ckpt"
if [ ! -f "$latest_ckpt" ]; then
    echo "Error: No checkpoint found at $latest_ckpt"
    exit 1
fi
echo "Loading pretrained model from: $latest_ckpt"
echo "Using finetune dataset: /home/aigc/human_policy/data/all_episodes"
echo "  - train: files 10-95 (85 files)"
echo "  - val:   files 0-9  (10 files)"
echo "Output directory: $finetune_output_dir"
echo "Learning rate: $finetune_lr"

CUDA_VISIBLE_DEVICES=1 python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 20000 \
    --lr $finetune_lr \
    --chunk_size $chunk_size \
    --exptid "$finetune_expt_id" \
    --dataset_json_path /tmp/all_episodes_finetune.json \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --load_pretrained_path "$latest_ckpt" \
    --no_wandb

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "基础模型: $output_dir"
echo "Finetune模型: $finetune_output_dir"
