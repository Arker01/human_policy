# Flow Matching 训练数据配置与启动命令

本文档针对 `/root/shengyin/human_policy` 新引入的 `ACT_FM` flow matching policy，说明如何组织当前这组数据，并给出预训练和 fine-tune 的启动命令。

## 1. 训练设计

推荐分两阶段训练：

| 阶段 | 训练数据 | 验证数据 | 目标 |
| --- | --- | --- | --- |
| 预训练 | `/root/shengyin/DATASETS/human_policy/convert_ego` + `/root/shengyin/DATASETS/PH2D` 中选定的 folder | `convert_ego` holdout + PH2D val folder | 学到通用视觉-状态到动作的 flow matching policy |
| Fine-tune | `/root/shengyin/DATASETS/human_policy/convert_whole` + `/root/shengyin/DATASETS/UnifoLM_WBT/human_policy_real` | 两个数据源各留一小段 holdout | 适配最终真实任务/whole-body 数据分布 |

训练入口读取 JSON 时，所有 `dataset_path` 都是相对 `--base_dir` 的路径。这里统一把：

```bash
--base_dir /root/shengyin/DATASETS
```

所以 JSON 中应写成：

```json
"human_policy/convert_ego"
"PH2D/<folder_name>"
"human_policy/convert_whole"
"UnifoLM_WBT/human_policy_real"
```

当前 loader 支持：

- 字符串：直接加载该目录下所有 `.hdf5`
- 字典：可用 `dataset_path`、`start_idx`、`end_idx` 切分 episode
- `type` 字段目前主要用于标注/可读性，实际采样时更依赖 hdf5 内的 `embodiment` attr

## 3. Fine-tune 数据 JSON

建议新建：

`/root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json`

`convert_whole` 当前约 150 个 episode，`human_policy_real` 当前约 13 个 episode。建议每个数据源都留少量验证集：

```json
{
  "train": [
    {
      "dataset_path": "human_policy/convert_whole",
      "type": "human",
      "start_idx": 0,
      "end_idx": 120
    },
    {
      "dataset_path": "UnifoLM_WBT/human_policy_real",
      "type": "human",
      "start_idx": 0,
      "end_idx": 10
    }
  ],
  "val": [
    {
      "dataset_path": "human_policy/convert_whole",
      "type": "human",
      "start_idx": 120,
      "end_idx": 150
    },
    {
      "dataset_path": "UnifoLM_WBT/human_policy_real",
      "type": "human",
      "start_idx": 10,
      "end_idx": 13
    }
  ]
}
```

如果 fine-tune 数据很少且只关心最终效果，可以把更多 episode 放进 `train`，但仍建议保留至少 2-3 条 validation，方便判断是否过拟合或数据读取异常。

## 4. 模型配置

Flow matching 使用已有配置：

```bash
/root/shengyin/human_policy/hdt/configs/models/act_flow.yaml
```

关键项：

```yaml
common:
  policy_class: ACT_FM
  state_dim: 128
  action_dim: 128
  camera_names: ['top']

model:
  backbone: resnet18
  image_feature_strategy: ACT_linear
  enc_layers: 4
  fm_layers: 4
  num_flow_steps: 10
  hand_eef_weight: 2.0
  head_eef_weight: 2.0
```

如果某些数据没有 `top` camera，loader 会 fallback 到 `left` 或 `right`。如果想强制使用别的相机，需要改 `camera_names`。

## 5. 启动预训练

单卡直接跑：

```bash
cd /root/shengyin/human_policy

python hdt/main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --no_wandb
```

使用 accelerate 单卡配置：

```bash
cd /root/shengyin/human_policy/hdt

accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --no_wandb
```

accelerate launch --config_file ./1_gpu.yaml main.py   --batch_size 64   --num_epochs 100000   --lr 1e-4   --chunk_size 100   --seed 0   --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2   --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json   --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml   --base_dir /root/shengyin/DATASETS   --human_slow_down_factor 4   --no_wandb

输出目录会自动变成：

```bash
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_ckpt
```

常用 checkpoint：

```bash
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_ckpt/policy_last.ckpt
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_ckpt/policy_iter_10000_seed_0/pytorch_model.bin
```

## 6. 启动 Fine-tune

从预训练的 `policy_last.ckpt` fine-tune：

```bash
cd /root/shengyin/human_policy

python hdt/main.py \
  --batch_size 32 \
  --num_epochs 30000 \
  --lr 5e-5 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --load_pretrained_path /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_ckpt/policy_last.ckpt \
  --no_wandb
```

使用 accelerate：

```bash
cd /root/shengyin/human_policy/hdt

accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 5e-5 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --load_pretrained_path /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_ckpt/policy_last.ckpt \
  --no_wandb
```

输出目录：

```bash
/root/shengyin/outputs/act_fm_finetune_whole_wbt_ckpt
```

## 7. 训练监控与验证

训练时重点看这些指标：

- `val/loss`：总 loss
- `val/fm`：flow matching velocity MSE
- `val/hand_eef_loss`：左右手 EEF 相关 loss
- `val/head_eef_loss`：头部 EEF 相关 loss

日志和曲线会写到 checkpoint 目录：

```bash
metrics.csv
metrics.png
dataset_stats.pkl
```

回溯评估已有 checkpoints：

```bash
cd /root/shengyin/human_policy/hdt

accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 32 \
  --num_epochs 30000 \
  --lr 5e-5 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --eval_ckpts \
  --no_wandb
```

## 8. JIT trace 导出

fine-tune 完成后导出 traced policy：

```bash
cd /root/shengyin/human_policy/hdt

accelerate launch --config_file ./1_gpu.yaml main.py \
  --batch_size 1 \
  --num_epochs 30000 \
  --lr 5e-5 \
  --chunk_size 100 \
  --seed 0 \
  --exptid /root/shengyin/outputs/act_fm_finetune_whole_wbt \
  --dataset_json_path /root/shengyin/human_policy/data/act_fm_finetune_whole_wbt.json \
  --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml \
  --base_dir /root/shengyin/DATASETS \
  --human_slow_down_factor 4 \
  --val_and_jit_trace \
  --no_wandb
```

导出文件：

```bash
/root/shengyin/outputs/act_fm_finetune_whole_wbt_ckpt/policy_traced.pt
```

注意：`num_flow_steps` 在 trace 后会固定。如果要改推理步数，需要先改 `act_flow.yaml`，再重新 trace。

## 9. 建议的默认超参

| 参数 | 预训练 | Fine-tune | 说明 |
| --- | --- | --- | --- |
| `chunk_size` | 100 | 100 | 沿用当前 ACT 脚本习惯 |
| `batch_size` | 64 | 32 | fine-tune 数据少，batch 可小一点 |
| `lr` | `1e-4` | `5e-5` | 加载预训练时代码会启用 warmup scheduler |
| `num_epochs` | 100000 | 30000 | 这里的 epoch 实际是训练 step 上限 |
| `human_slow_down_factor` | 4 | 4 | 当前 human trajectory 默认插值放慢 |
| `cond_mask_prob` | 0.1 | 0.1 | 默认即可 |

如果显存不够，优先降低 `batch_size`；如果训练明显过拟合，减少 fine-tune step 或降低学习率到 `1e-5`。
