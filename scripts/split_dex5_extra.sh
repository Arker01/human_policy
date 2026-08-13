#!/bin/bash

# Python解释器路径
PYTHON="/home/aigc/miniconda/envs/human_policy/bin/python"

# 目录配置
DATA_DIR="/home/aigc/human_policy/data/07_dex5_extra"
TRAIN_DIR="${DATA_DIR}_train"
VAL_DIR="${DATA_DIR}_val"

# 处理后的目录
TRAIN_PROCESSED_DIR="${DATA_DIR}_train_processed"
VAL_PROCESSED_DIR="${DATA_DIR}_val_processed"

# 清理旧目录
rm -rf "$TRAIN_DIR" "$VAL_DIR" "$TRAIN_PROCESSED_DIR" "$VAL_PROCESSED_DIR"

# 创建输出目录
mkdir -p "$TRAIN_DIR"
mkdir -p "$VAL_DIR"

# 获取所有hdf5文件并排序
files=($(ls "$DATA_DIR"/*.hdf5 | sort))
total_files=${#files[@]}

echo "总文件数: $total_files"

# 按90%/10%划分
train_count=$((total_files * 9 / 10))
val_count=$((total_files - train_count))

echo "训练集: $train_count 个文件"
echo "验证集: $val_count 个文件"

# 创建符号链接
for ((i=0; i<total_files; i++)); do
    file="${files[$i]}"
    filename=$(basename "$file")
    
    if [ $i -lt $train_count ]; then
        ln -s "$file" "$TRAIN_DIR/$filename"
    else
        ln -s "$file" "$VAL_DIR/$filename"
    fi
done

# 验证
echo ""
echo "验证结果:"
echo "训练集文件数: $(ls "$TRAIN_DIR"/*.hdf5 | wc -l)"
echo "验证集文件数: $(ls "$VAL_DIR"/*.hdf5 | wc -l)"

echo ""
echo "训练集文件列表:"
ls "$TRAIN_DIR"/*.hdf5 | head -5
echo "..."

echo ""
echo "验证集文件列表:"
ls "$VAL_DIR"/*.hdf5

# ==========================================
# 处理数据：填充后颈和腰部位置
# ==========================================
echo ""
echo "=========================================="
echo "处理训练集数据..."
echo "=========================================="
$PYTHON /home/aigc/human_policy/scripts/fill_neck_waist_relative.py \
    --input-dir "$TRAIN_DIR" \
    --output-dir "$TRAIN_PROCESSED_DIR"

echo ""
echo "=========================================="
echo "处理验证集数据..."
echo "=========================================="
$PYTHON /home/aigc/human_policy/scripts/fill_neck_waist_relative.py \
    --input-dir "$VAL_DIR" \
    --output-dir "$VAL_PROCESSED_DIR"

# ==========================================
# 设置embodiment字段
# ==========================================
echo ""
echo "=========================================="
echo "设置训练集embodiment字段..."
echo "=========================================="
$PYTHON /home/aigc/human_policy/scripts/fill_embodiment_wbt.py \
    --input-dir "$TRAIN_PROCESSED_DIR"

echo ""
echo "=========================================="
echo "设置验证集embodiment字段..."
echo "=========================================="
$PYTHON /home/aigc/human_policy/scripts/fill_embodiment_wbt.py \
    --input-dir "$VAL_PROCESSED_DIR"

# ==========================================
# 最终验证
# ==========================================
echo ""
echo "=========================================="
echo "最终验证结果:"
echo "=========================================="
echo "训练集(原始): $(ls "$TRAIN_DIR"/*.hdf5 | wc -l) 个文件"
echo "验证集(原始): $(ls "$VAL_DIR"/*.hdf5 | wc -l) 个文件"
echo "训练集(处理后): $(ls "$TRAIN_PROCESSED_DIR"/*.hdf5 | wc -l) 个文件"
echo "验证集(处理后): $(ls "$VAL_PROCESSED_DIR"/*.hdf5 | wc -l) 个文件"

# 验证一个样本文件
echo ""
echo "验证样本文件:"
$PYTHON -c "
import h5py
import os

sample_file = os.listdir('$TRAIN_PROCESSED_DIR')[0]
f = h5py.File('$TRAIN_PROCESSED_DIR/' + sample_file, 'r')
action = f['action'][()]
emb = f.attrs.get('embodiment', 'NOT FOUND')

print(f'文件: {sample_file}')
print(f'Action shape: {action.shape}')
print(f'Embodiment: {emb}')
print(f'后颈数据: {action[0, 58:67]}')
print(f'腰部数据: {action[0, 89:98]}')
f.close()
"