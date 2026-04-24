# 训练启动指南

本文档详细说明如何使用训练脚本启动模型训练，包括脚本结构、数据配置、参数设置等内容。

## 目录结构

```
human-policy/
├── scripts/
│   ├── train/         # 训练脚本
│   ├── finetune/      # 微调脚本
│   └── eval/          # 评估脚本
├── hdt/
│   ├── configs/       # 模型配置文件
│   ├── data_utils_hdt.py  # 数据加载工具
│   └── main.py        # 训练主入口
├── data/              # 数据处理脚本
└── *.json             # 数据集配置文件
```

## 训练脚本说明

### 1. 训练脚本分类

**训练脚本** (`scripts/train/`):
- `train_convert2.sh` - 训练 convert2 数据集
- `train_convert2_1000.sh` - 使用 1000 条 convert2 数据训练
- `train_convert2_1500.sh` - 使用 1500 条 convert2 数据训练
- `train_convert2_with_pick.sh` - 包含 pick and place 数据的训练
- `train_pick_place_mixed.sh` - 混合人类和机器人 pick and place 数据训练
- `train_pick_place_mixed_translated.sh` - 使用平移后的 processed 数据训练
- `train_pick_place_mixed_with_ego.sh` - 包含 convert_ego 数据的训练
- `train_pick_place_processed_only.sh` - 只使用 processed 数据训练

**微调脚本** (`scripts/finetune/`):
- `finetune_data1_from_1000.sh` - 基于 1000 条数据训练的模型微调 data1 数据
- `finetune_data1_from_1500.sh` - 基于 1500 条数据训练的模型微调 data1 数据
- `finetune_20000.sh` - 在 20000 轮模型基础上微调

**评估脚本** (`scripts/eval/`):
- 多个评估脚本，用于评估不同模型的性能

## 数据配置文件

### 1. 配置文件格式

数据配置文件使用 JSON 格式，包含训练集和验证集的数据集路径。

**基本结构：**
```json
{
  "train": [
    {
      "dataset_path": "path/to/dataset",
      "type": "human" or "robot",
      "start_idx": 0,  // 可选，数据起始索引
      "end_idx": 100   // 可选，数据结束索引
    }
  ],
  "val": [
    {
      "dataset_path": "path/to/val/dataset",
      "type": "mixed"
    }
  ]
}
```

### 2. 现有配置文件

- `pick_place_mixed.json` - 混合人类和机器人 pick and place 数据
- `pick_place_mixed_translated.json` - 使用平移后的 processed 数据
- `pick_place_mixed_with_ego.json` - 包含 convert_ego 数据
- `pick_place_processed_only.json` - 只使用 processed 数据

## 如何启动训练

### 1. 基本步骤

1. **选择训练脚本**：根据需要选择合适的训练脚本
2. **配置参数**：根据需求修改脚本中的参数
3. **启动训练**：运行脚本启动训练

### 2. 脚本参数说明

训练脚本中的主要参数：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `batch_size` | 批次大小 | 32 |
| `learning_rate` | 学习率 | 1e-5 |
| `chunk_size` | 序列长度 | 16 |
| `num_epochs` | 训练轮次 | 30000 |
| `expt_id` | 实验ID | 脚本名称 |
| `dataset_json_path` | 数据配置文件 | 对应配置文件 |
| `model_cfg_path` | 模型配置文件 | hdt/configs/models/act_resnet.yaml |
| `base_dir` | 数据基础目录 | /data1/zxlei/dataset |

### 3. 启动示例

**启动只使用 processed 数据的训练：**
```bash
cd /home/embodied/human-policy
bash scripts/train/train_pick_place_processed_only.sh
```

**启动包含 convert_ego 数据的训练：**
```bash
cd /home/embodied/human-policy
bash scripts/train/train_pick_place_mixed_with_ego.sh
```

## 模型配置

### 1. 模型配置文件

模型配置文件位于 `hdt/configs/models/` 目录，主要配置包括：

- `act_resnet.yaml` - ACT 模型配置，使用 ResNet 作为视觉编码器

**关键配置参数：**
- `camera_names`: 相机名称，默认为 `['top']`
- `image_resolution_hw`: 图像分辨率，默认为 `[240, 320]`
- `backbone`: 视觉编码器，默认为 `resnet18`
- `kl_weight`: KL 损失权重，默认为 10

### 2. 相机配置

训练代码会自动处理相机数据：
- 当指定的相机不存在时，会尝试使用其他相机（left → right）
- 如果完全没有相机数据，会跳过该文件

## 常见问题及解决方案

### 1. 数据加载错误

**问题**：`KeyError: "None of the expected cameras (top, left, right) found in ...hdf5"`

**解决方案**：检查数据文件是否包含相机数据，或修改模型配置中的 `camera_names`。

### 2. 内存不足

**问题**：`CUDA out of memory`

**解决方案**：
- 减小 `batch_size`
- 使用 `CUDA_VISIBLE_DEVICES` 指定其他 GPU
- 减小 `chunk_size`

### 3. 训练速度慢

**解决方案**：
- 增加 `batch_size`（如果内存允许）
- 使用多 GPU 训练
- 调整 `learning_rate`

### 4. 评估效果差

**可能原因**：
- 数据量不足
- 相机配置不匹配
- 模型过拟合

**解决方案**：
- 增加训练数据
- 确保相机配置正确
- 调整正则化参数
- 使用数据增强

## 输出目录

训练结果会保存在：
- 模型检查点：`/data1/zxlei/model/{expt_id}_ckpt/`
- 训练日志：控制台输出

## 注意事项

1. **数据路径**：确保 `base_dir` 参数指向正确的数据目录
2. **GPU 选择**：如果有多个 GPU，可以使用 `CUDA_VISIBLE_DEVICES` 环境变量指定
3. **模型配置**：根据数据特点调整模型配置文件
4. **训练时间**：训练可能需要较长时间，建议在后台运行

## 联系信息

如果遇到问题，请联系项目维护者。
