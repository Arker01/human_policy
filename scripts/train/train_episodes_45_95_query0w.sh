#!/bin/bash
cd /home/aigc/human_policy

# 训练配置
batch_size=64
learning_rate=1e-5
chunk_size=100
expt_id="train_episodes_45_95_query0w5"

# 数据和模型配置
model_cfg_path="hdt/configs/models/act_resnet.yaml"
base_dir="/home/aigc/human_policy/data"

# 输出路径
output_dir="/home/aigc/human_policy/data/${expt_id}_ckpt"
mkdir -p "$output_dir"

# ==========================================
# 训练配置: 仅用 wholebody-45 ~ wholebody-95（共51个，最一致的"捡起后继续往前走放到桌上"任务阶段）
# train/val 用显式 file_list（按episode编号每隔5个抽1个做val），
# 避免 start_idx/end_idx 依赖目录字符串排序（"wholebody-10" 会排在 "wholebody-2" 前面）导致的切分偏差
# ==========================================
cat > /tmp/episodes_45_95_query0w_train.json << 'EOF'
{
  "train": [
    {
      "dataset_path": "/home/aigc/human_policy/data/all_episodes",
      "type": "human",
      "file_list": [
        "wholebody-45_unified_V.hdf5",
        "wholebody-46_unified_V.hdf5",
        "wholebody-47_unified_V.hdf5",
        "wholebody-48_unified_V.hdf5",
        "wholebody-50_unified_V.hdf5",
        "wholebody-51_unified_V.hdf5",
        "wholebody-52_unified_V.hdf5",
        "wholebody-53_unified_V.hdf5",
        "wholebody-55_unified_V.hdf5",
        "wholebody-56_unified_V.hdf5",
        "wholebody-57_unified_V.hdf5",
        "wholebody-58_unified_V.hdf5",
        "wholebody-60_unified_V.hdf5",
        "wholebody-61_unified_V.hdf5",
        "wholebody-62_unified_V.hdf5",
        "wholebody-63_unified_V.hdf5",
        "wholebody-65_unified_V.hdf5",
        "wholebody-66_unified_V.hdf5",
        "wholebody-67_unified_V.hdf5",
        "wholebody-68_unified_V.hdf5",
        "wholebody-70_unified_V.hdf5",
        "wholebody-71_unified_V.hdf5",
        "wholebody-72_unified_V.hdf5",
        "wholebody-73_unified_V.hdf5",
        "wholebody-75_unified_V.hdf5",
        "wholebody-76_unified_V.hdf5",
        "wholebody-77_unified_V.hdf5",
        "wholebody-78_unified_V.hdf5",
        "wholebody-80_unified_V.hdf5",
        "wholebody-81_unified_V.hdf5",
        "wholebody-82_unified_V.hdf5",
        "wholebody-83_unified_V.hdf5",
        "wholebody-85_unified_V.hdf5",
        "wholebody-86_unified_V.hdf5",
        "wholebody-87_unified_V.hdf5",
        "wholebody-88_unified_V.hdf5",
        "wholebody-90_unified_V.hdf5",
        "wholebody-91_unified_V.hdf5",
        "wholebody-92_unified_V.hdf5",
        "wholebody-93_unified_V.hdf5",
        "wholebody-95_unified_V.hdf5"
      ]
    }
  ],
  "val": [
    {
      "dataset_path": "/home/aigc/human_policy/data/all_episodes",
      "type": "human",
      "file_list": [
        "wholebody-49_unified_V.hdf5",
        "wholebody-54_unified_V.hdf5",
        "wholebody-59_unified_V.hdf5",
        "wholebody-64_unified_V.hdf5",
        "wholebody-69_unified_V.hdf5",
        "wholebody-74_unified_V.hdf5",
        "wholebody-79_unified_V.hdf5",
        "wholebody-84_unified_V.hdf5",
        "wholebody-89_unified_V.hdf5",
        "wholebody-94_unified_V.hdf5"
      ]
    }
  ]
}
EOF

echo "=========================================="
echo "仅用 episodes 45-95 训练"
echo "=========================================="
echo "  Train: 41 episodes (45-95 范围内, 排除每隔5个抽出的val)"
echo "  Val:   10 episodes (49,54,59,64,69,74,79,84,89,94, 均匀分布)"
echo "  lr=$learning_rate, 50000 epoch"
echo "  Output: $output_dir"
echo "=========================================="

CUDA_VISIBLE_DEVICES=1 python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 50000 \
    --lr $learning_rate \
    --chunk_size $chunk_size \
    --exptid "$expt_id" \
    --dataset_json_path /tmp/episodes_45_95_query0w_train.json \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --query0_extra_weight 5.0 \
    --no_wandb

echo ""
echo "=========================================="
echo "训练完成!"
echo "=========================================="
echo "模型输出: $output_dir"
echo "=========================================="
