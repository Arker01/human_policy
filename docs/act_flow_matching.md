# ACT + Flow Matching (ACT_FM) 使用说明

## 架构概述

```
image + qpos
     │
     ▼
Backbone (ResNet18/DINOv2)
     │
     ▼
TransformerEncoder (enc_layers 层)  ←── context tokens
     │
     ▼
FlowMatchingHead (fm_layers 层 TransformerDecoder)
  · xt tokens (noisy action) 作为 query
  · context tokens 作为 key/value
  · 输出 velocity field v(xt, t, context)
```

与原始 ACT 的区别：
- **去掉** CVAE encoder / KL 散度损失
- **去掉** Transformer decoder 的 query embedding（改为 action token）
- **新增** Flow Matching head，通过 ODE 生成动作

---

## Loss 组成

训练时共 **2 部分** loss：

| 名称 | 含义 | 默认权重 | 对应 indices |
|------|------|----------|-------------|
| `fm` | Flow Matching MSE，预测 velocity 误差 `‖v_pred − v_target‖²`，只在非 padding 时刻计算 | ×1 | 全部 128 维 |
| `hand_eef_loss` | 左右手 EEF 维度的 FM loss | ×`hand_eef_weight`(2.0) | `OUTPUT_LEFT_EEF`(80-88) + `OUTPUT_RIGHT_EEF`(30-38) |
| `head_eef_loss` | 头部 pos(3) + rot(6) 维度的 FM loss | ×`head_eef_weight`(0.0) | `OUTPUT_HEAD_EEF`(0-8) |
| `loss` | 总 loss = `fm + hand_eef_loss × hand_eef_weight + head_eef_loss × head_eef_weight` | — | — |

> **注意**：没有 `kl` loss（Flow Matching 本身不需要 VAE 潜变量）。

### Flow Matching 训练细节

```
x0 ~ N(0, I)          # 随机噪声
x1 = a_gt             # ground truth action chunk
t  ~ U[0, 1]          # 随机采样时刻

xt       = (1 - t) * x0 + t * x1   # 线性插值
v_target = x1 - x0                  # 目标速度场

v_pred = FlowHead(xt, t, context)
loss   = MSE(v_pred, v_target)  [on non-pad timesteps]
```

### 推理（Euler ODE）

```
x ~ N(0, I)
dt = 1 / num_flow_steps
for i in range(num_flow_steps):
    t = i * dt
    v = FlowHead(x, t, context)
    x = x + v * dt
return x   # predicted action chunk
```

---

## 配置文件

`hdt/configs/models/act_flow.yaml`

```yaml
common:
  policy_class: ACT_FM
  state_dim: 128
  action_dim: 128
  camera_names: ['top']

model:
  enc_layers: 4          # context TransformerEncoder 层数
  fm_layers: 4           # FlowMatchingHead Decoder 层数
  nheads: 8
  hidden_dim: 512
  dim_feedforward: 3200
  lr_backbone: 1e-5
  backbone: resnet18
  image_feature_strategy: ACT_linear
  use_language_conditioning: False
  num_flow_steps: 10     # 推理时 Euler ODE 步数
```

---

## 训练命令

### 单卡（调试用）

```bash
cd hdt
python main.py \
  --model_cfg_path configs/models/act_flow.yaml \
  --chunk_size 64 \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --seed 0 \
  --exptid exp/act_fm_v1 \
  --dataset_json_path configs/datasets/<your_dataset>.json \
  --no_wandb
```

### 多卡（accelerate）

```bash
cd hdt
accelerate launch --config_file ./accelerator_setup.yaml main.py \
  --model_cfg_path configs/models/act_flow.yaml \
  --chunk_size 64 \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --seed 0 \
  --exptid exp/act_fm_v1 \
  --dataset_json_path configs/datasets/<your_dataset>.json \
  --no_wandb
```

### 从预训练模型 finetune

```bash
accelerate launch --config_file ./accelerator_setup.yaml main.py \
  --model_cfg_path configs/models/act_flow.yaml \
  --chunk_size 64 \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --seed 0 \
  --exptid exp/act_fm_finetune \
  --dataset_json_path configs/datasets/<your_dataset>.json \
  --load_pretrained_path <path_to_pretrained>/pytorch_model.bin \
  --no_wandb
```

加载预训练时会自动启用 warmup scheduler（1e-7 → 1e-4，每 1000 步 ×10）。

---

## 验证 & 导出

```bash
# 验证最新 checkpoint 并导出 JIT traced 模型
cd hdt
accelerate launch --config_file ./accelerator_setup.yaml main.py \
  --model_cfg_path configs/models/act_flow.yaml \
  --chunk_size 64 \
  --batch_size 1 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --seed 0 \
  --exptid exp/act_fm_v1 \
  --dataset_json_path configs/datasets/<your_dataset>.json \
  --no_wandb \
  --val_and_jit_trace
```

导出的 traced 模型保存在 `exp/act_fm_v1_ckpt/policy_traced.pt`。

> **注意**：Flow Matching 推理内部有 for 循环，JIT trace 时循环会被展开（固定步数）。
> `num_flow_steps` 一旦 trace 后不能在线修改，需要重新 trace。

---

## 回溯验证所有 checkpoint

```bash
cd hdt
accelerate launch --config_file ./accelerator_setup.yaml main.py \
  --model_cfg_path configs/models/act_flow.yaml \
  --chunk_size 64 \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-4 \
  --seed 0 \
  --exptid exp/act_fm_v1 \
  --dataset_json_path configs/datasets/<your_dataset>.json \
  --no_wandb \
  --eval_ckpts
```

结果写入 `exp/act_fm_v1_ckpt/retro_metrics.csv` 和 `retro_metrics.png`。

---

## 注意事项

### 超参建议

| 参数 | 建议值 | 说明 |
|------|--------|------|
| `chunk_size` | 64 | 与 ACT 对齐，可调 |
| `num_flow_steps` | 10 | 推理步数，越多越准但越慢；5~20 均可 |
| `fm_layers` | 4 | FlowHead decoder 层数，可从 4 开始 |
| `enc_layers` | 4 | 与原 ACT encoder 对齐 |
| `lr` | 1e-4 | 同 ACT |
| `batch_size` | 64 | 同 ACT |

### 与 ACT 对比训练

- ACT 的 val loss 看 `val/l1` 和 `val/kl`
- ACT_FM 的 val loss 看 `val/fm`、`val/hand_eef_loss`、`val/head_eef_loss`
- 两者 `val/loss` 均为总 loss，可直接比较趋势（数值量纲不同，不要直接比大小）

### 常见问题

**Q: 推理速度比 ACT 慢？**
A: FM 需要 `num_flow_steps` 次前向，ACT 只需 1 次。可以降低 `num_flow_steps`（5步通常够用）。

**Q: loss 一开始很高？**
A: 正常，FM loss 的量纲取决于动作空间的 scale，建议先看 loss 下降趋势而非绝对值。

**Q: 想换 backbone（DINOv2）？**
A: 修改 yaml 中的 `backbone: dinov2_vits14` 和 `image_feature_strategy: linear`，其余不变。

---

## 相关文件

```
hdt/
├── modeling/
│   └── modeling_act_flow.py   # 核心模型：ACTFlowMatching + ACTFlowPolicy
├── configs/models/
│   └── act_flow.yaml          # 配置文件
└── main.py                    # 训练入口（已添加 ACT_FM 分支）
```
