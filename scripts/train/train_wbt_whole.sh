#!/bin/bash

# 训练配置
batch_size=64
learning_rate=1e-5
finetune_lr=1e-6
chunk_size=100
expt_id="train_wbt_whole"

# 数据和模型配置
model_cfg_path="hdt/configs/models/act_resnet.yaml"
base_dir="/home/aigc/human_policy/data"

# 输出路径
output_dir="/home/aigc/human_policy/data/${expt_id}_ckpt"
mkdir -p "$output_dir"

# 创建临时目录结构，按机器人类型分组
# 注意：目录名必须包含 'robot' 字样，以便数据加载代码正确识别embodiment类型
echo "创建临时目录结构..."
TEMP_DIR="/tmp/wbt_sorted_robot"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR/robot_Dex5"
mkdir -p "$TEMP_DIR/robot_Inspire"
mkdir -p "$TEMP_DIR/robot_Brainco"

# 根据文件名中的机器人类型创建符号链接
echo "按机器人类型分类数据..."
for file in "$base_dir/convert_UnifoLM_WBT"/*.hdf5; do
    filename=$(basename "$file")
    if [[ "$filename" == *"Dex5"* ]]; then
        ln -s "$file" "$TEMP_DIR/robot_Dex5/"
    elif [[ "$filename" == *"Inspire"* ]]; then
        ln -s "$file" "$TEMP_DIR/robot_Inspire/"
    elif [[ "$filename" == *"Brainco"* ]]; then
        ln -s "$file" "$TEMP_DIR/robot_Brainco/"
    fi
done

# 统计各类数据数量
DEX5_COUNT=$(ls "$TEMP_DIR/robot_Dex5"/*.hdf5 2>/dev/null | wc -l)
INSPIRE_COUNT=$(ls "$TEMP_DIR/robot_Inspire"/*.hdf5 2>/dev/null | wc -l)
BRAINCO_COUNT=$(ls "$TEMP_DIR/robot_Brainco"/*.hdf5 2>/dev/null | wc -l)

echo "Dex5: $DEX5_COUNT files"
echo "Inspire: $INSPIRE_COUNT files"
echo "Brainco: $BRAINCO_COUNT files"

# 计算训练/验证划分
DEX5_TRAIN_END=$((DEX5_COUNT * 9 / 10))
INSPIRE_TRAIN_END=$((INSPIRE_COUNT * 9 / 10))
BRAINCO_TRAIN_END=$((BRAINCO_COUNT * 9 / 10))

# 创建训练配置
cat > /tmp/wbt_whole_train.json << EOF
{
  "train": [
    {
      "dataset_path": "$TEMP_DIR/robot_Dex5",
      "type": "robot",
      "start_idx": 0,
      "end_idx": $DEX5_TRAIN_END
    },
    {
      "dataset_path": "$TEMP_DIR/robot_Inspire",
      "type": "robot",
      "start_idx": 0,
      "end_idx": $INSPIRE_TRAIN_END
    },
    {
      "dataset_path": "$TEMP_DIR/robot_Brainco",
      "type": "robot",
      "start_idx": 0,
      "end_idx": $BRAINCO_TRAIN_END
    },
    {
      "dataset_path": "convert_whole_last",
      "type": "human",
      "start_idx": 0,
      "end_idx": 135
    }
  ],
  "val": [
    {
      "dataset_path": "$TEMP_DIR/robot_Dex5",
      "type": "robot",
      "start_idx": $DEX5_TRAIN_END,
      "end_idx": $DEX5_COUNT
    },
    {
      "dataset_path": "$TEMP_DIR/robot_Inspire",
      "type": "robot",
      "start_idx": $INSPIRE_TRAIN_END,
      "end_idx": $INSPIRE_COUNT
    },
    {
      "dataset_path": "$TEMP_DIR/robot_Brainco",
      "type": "robot",
      "start_idx": $BRAINCO_TRAIN_END,
      "end_idx": $BRAINCO_COUNT
    },
    {
      "dataset_path": "convert_whole_last",
      "type": "human",
      "start_idx": 135,
      "end_idx": 150
    }
  ]
}
EOF

echo ""
echo "=========================================="
echo "训练配置信息"
echo "=========================================="
echo "训练集:"
echo "  - WBT_Dex5: $DEX5_TRAIN_END个episode [type: robot]"
echo "  - WBT_Inspire: $INSPIRE_TRAIN_END个episode [type: robot]"
echo "  - WBT_Brainco: $BRAINCO_TRAIN_END个episode [type: robot]"
echo "  - convert_whole_last: 135个episode [type: human]"
echo "验证集:"
echo "  - WBT_Dex5: $((DEX5_COUNT - DEX5_TRAIN_END))个episode [type: robot]"
echo "  - WBT_Inspire: $((INSPIRE_COUNT - INSPIRE_TRAIN_END))个episode [type: robot]"
echo "  - WBT_Brainco: $((BRAINCO_COUNT - BRAINCO_TRAIN_END))个episode [type: robot]"
echo "  - convert_whole_last: 15个episode [type: human]"
echo "总训练数据: $((DEX5_TRAIN_END + INSPIRE_TRAIN_END + BRAINCO_TRAIN_END + 135))个episode"
echo "总验证数据: $((DEX5_COUNT - DEX5_TRAIN_END + INSPIRE_COUNT - INSPIRE_TRAIN_END + BRAINCO_COUNT - BRAINCO_TRAIN_END + 15))个episode"
echo "=========================================="

# 开始训练
CUDA_VISIBLE_DEVICES=2 python hdt/main.py \
    --batch_size $batch_size \
    --num_epochs 50000 \
    --lr $learning_rate \
    --chunk_size $chunk_size \
    --exptid "$expt_id" \
    --dataset_json_path /tmp/wbt_whole_train.json \
    --model_cfg_path "$model_cfg_path" \
    --base_dir "$base_dir" \
    --no_wandb

# 清理临时目录
rm -rf "$TEMP_DIR"

echo ""
echo "=========================================="
echo "训练完成!"
echo "输出目录: $output_dir"
echo "=========================================="