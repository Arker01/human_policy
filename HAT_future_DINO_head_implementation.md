# HAT 增加 Future-DINO World Head：实现方案

## 1. 目标

在现有 HAT 的未来轨迹预测任务之外，增加一个 **Future-DINO World Head**，要求模型同时预测未来时刻的 DINO patch features。

现有任务：

```text
当前 egocentric RGB
+ 当前/历史的人体、手腕、指尖状态
→ 未来 body / wrist / fingertip 轨迹
```

修改后：

```text
当前 egocentric RGB
+ 当前/历史的人体、手腕、指尖状态
→ 未来 body / wrist / fingertip 轨迹
→ 未来 egocentric 图像的 DINO patch features
```

训练目标：

\[
\mathcal{L}
=
\mathcal{L}_{traj}
+
\lambda_{world}\mathcal{L}_{DINO}
\]

第一阶段只把 Future-DINO Head 当作 **训练辅助监督**。推理时可以删除该 head，保持原 HAT 的输入输出和运行速度不变。

---

## 2. 数据集需要增加的内容

现有每条样本大致包含：

```text
I_t                         当前 egocentric 图像
X_{t-k:t}                   历史 body / wrist / fingertip 状态
τ_{t:t+H}                   未来轨迹标签
```

增加：

```text
I_{t+H}                     预测窗口终点的未来图像
Z_{t+H} = DINO(I_{t+H})     未来 DINO patch feature 标签
```

### 2.1 第一版只预测单个未来时刻

先预测轨迹窗口终点 `t + H` 的 DINO feature，不要一开始预测完整未来 feature 序列。

理由：

- 实现简单；
- 显存开销较小；
- 避免同时引入多时间尺度和视频预测问题；
- 足以验证 world supervision 是否能改善 HAT。

后续确认有效后，再考虑预测多个 horizon：

```text
t + H/3
 t + 2H/3
 t + H
```

### 2.2 离线预计算 Future-DINO target

使用与 HAT 当前视觉输入相同的冻结 DINO encoder，提前为未来图像计算 patch features：

```python
with torch.no_grad():
    future_tokens = dino_encoder(future_image)
    future_tokens = remove_cls_and_register_tokens(future_tokens)
    future_tokens = tokenwise_normalize(future_tokens)
```

推荐保存为：

```text
[N, P, D]

P: patch 数量，例如 16 × 16 = 256
D: DINO feature 维度，例如 768
```

这样训练 HAT 时不需要重复运行未来图像的 DINO encoder。

### 2.3 图像增强必须保持时空一致

当前图像 `I_t` 和未来图像 `I_{t+H}` 必须使用相同的几何增强参数：

- 相同 crop；
- 相同 resize；
- 相同 horizontal flip；
- 相同相机内参变换。

否则 patch-level loss 会把随机裁剪差异误认为真实世界变化。

颜色增强可以分别使用，但第一版建议也保持一致，减少无关变量。

---

## 3. 网络结构修改

整体结构：

```text
Current egocentric image
          ↓
      Frozen DINO
          ↓
 Current DINO patch tokens ───────────────┐
                                          │
Body / wrist / fingertip history ─────┐   │
                                      ↓   ↓
                                HAT shared trunk
                                  ↙           ↘
                       Trajectory Head     Future-DINO Head
                              ↓                  ↓
                    Future trajectories    Future DINO tokens
```

关键要求：

> Future-DINO Head 必须读取 HAT 的共享 trunk feature，确保 world prediction 的梯度真正作用于轨迹预测所使用的共享表征。

不要额外建立一套与 HAT 独立的 future predictor，否则它不会明显改善 HAT 本身。

---

## 4. Future-DINO Head 设计

### 4.1 推荐第一版：one-step Transformer decoder

输入：

```text
current_dino_tokens: [B, P, D_dino]
hat_memory:          [B, N, D_hidden]
```

输出：

```text
pred_future_dino:    [B, P, D_dino]
```

建议使用当前 DINO patch token 作为 decoder query，而不是只使用一个 global token。

原因：

- 保留图像空间布局；
- 允许模型学习每个区域未来如何变化；
- 比直接从一个 global feature 回归全部 patch 更容易训练。

### 4.2 模块示例

```python
import torch
import torch.nn as nn


class FutureDINOHead(nn.Module):
    def __init__(
        self,
        dino_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 4,
        num_heads: int = 8,
        num_patches: int = 256,
    ):
        super().__init__()

        self.query_proj = nn.Linear(dino_dim, hidden_dim)

        self.position_embedding = nn.Parameter(
            torch.randn(1, num_patches, hidden_dim) * 0.02
        )

        self.horizon_embedding = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            batch_first=True,
            norm_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.output_proj = nn.Linear(hidden_dim, dino_dim)

    def forward(
        self,
        current_dino_tokens: torch.Tensor,
        hat_memory: torch.Tensor,
    ) -> torch.Tensor:
        """
        current_dino_tokens: [B, P, D_dino]
        hat_memory:          [B, N, D_hidden]
        return:              [B, P, D_dino]
        """

        queries = self.query_proj(current_dino_tokens)
        queries = (
            queries
            + self.position_embedding
            + self.horizon_embedding
        )

        future_tokens = self.decoder(
            tgt=queries,
            memory=hat_memory,
        )

        return self.output_proj(future_tokens)
```

### 4.3 不建议第一版直接照搬 diffusion / flow matching

第一版建议直接回归未来 feature：

```text
shared HAT representation
→ one-step Future-DINO decoder
→ future DINO feature
```

暂时不要加入：

- diffusion；
- flow matching；
- 多步 Euler sampling；
- autoregressive feature generation。

原因是当前目的只是验证 auxiliary world supervision 是否改善 HAT，而不是建立完整的生成式世界模型。

---

## 5. HAT Forward 修改

伪代码：

```python
def forward(
    self,
    image,
    body_history,
    hand_history,
):
    # 1. 当前图像视觉编码
    with torch.no_grad():
        current_dino = self.dino_encoder(image)
        current_dino = self.process_dino_tokens(current_dino)

    visual_tokens = self.visual_projection(current_dino)
    body_tokens = self.body_encoder(body_history)
    hand_tokens = self.hand_encoder(hand_history)

    # 2. 原 HAT 输入
    input_tokens = torch.cat(
        [
            visual_tokens,
            body_tokens,
            hand_tokens,
            self.trajectory_queries.expand(image.shape[0], -1, -1),
        ],
        dim=1,
    )

    # 3. 共享 trunk
    shared_features = self.hat_transformer(input_tokens)

    # 4. 原轨迹预测
    future_trajectory = self.trajectory_head(shared_features)

    # 5. 新增 future-DINO 预测
    future_dino = self.future_dino_head(
        current_dino_tokens=current_dino,
        hat_memory=shared_features,
    )

    return {
        "trajectory": future_trajectory,
        "future_dino": future_dino,
    }
```

---

## 6. Loss 设计

### 6.1 总损失

```python
loss = trajectory_loss + lambda_world * world_loss
```

### 6.2 Future-DINO loss

第一版建议使用：

```text
Cosine loss + 少量 Smooth L1 loss
```

实现：

```python
import torch
import torch.nn.functional as F


def future_dino_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    huber_weight: float = 0.1,
) -> torch.Tensor:
    """
    predicted: [B, P, D]
    target:    [B, P, D]
    """

    predicted_norm = F.normalize(predicted, dim=-1)
    target_norm = F.normalize(target.detach(), dim=-1)

    cosine_loss = 1.0 - (
        predicted_norm * target_norm
    ).sum(dim=-1).mean()

    huber_loss = F.smooth_l1_loss(
        predicted,
        target.detach(),
    )

    return cosine_loss + huber_weight * huber_loss
```

训练部分：

```python
outputs = model(
    image=current_image,
    body_history=body_history,
    hand_history=hand_history,
)

trajectory_loss = compute_trajectory_loss(
    outputs["trajectory"],
    future_trajectory_gt,
)

world_loss = future_dino_loss(
    outputs["future_dino"],
    future_dino_gt,
)

loss = trajectory_loss + lambda_world * world_loss
```

---

## 7. World loss 权重

不要直接固定一个权重。至少测试：

```text
λ_world = 0
λ_world = 0.1
λ_world = 0.3
λ_world = 1.0
```

同时记录：

- trajectory loss；
- future-DINO loss；
- trajectory head gradient norm；
- world head gradient norm；
- shared trunk gradient norm。

目标是：

```text
world loss 能影响 shared trunk，
但不能在训练早期完全压过 trajectory loss。
```

如果两个 loss 数量级差异很大，可以使用：

- loss normalization；
- gradient norm balancing；
- warm-up world loss。

简单 warm-up：

```python
world_weight = lambda_world * min(
    1.0,
    current_step / warmup_steps,
)
```

---

## 8. 训练流程

如果从现有 HAT checkpoint 开始，推荐分三步：

### Stage 1：只训练 Future-DINO Head

```text
冻结：DINO、HAT trunk、trajectory head
训练：Future-DINO Head
```

目的：

- 验证数据和loss没有问题；
- 确认future feature可以被预测；
- 避免随机初始化的world head破坏已有HAT。

### Stage 2：解冻 HAT trunk 最后几层

```text
冻结：DINO、HAT前部层、trajectory head可选
训练：Future-DINO Head + HAT最后若干层
```

目的：让world supervision开始修改共享表征。

### Stage 3：联合微调

```text
冻结：DINO
训练：HAT trunk + trajectory head + Future-DINO Head
```

使用较小学习率：

```text
HAT pretrained parameters: 1× learning rate
Future-DINO Head:           5× 或 10× learning rate
```

---

## 9. 必须做的消融实验

### 9.1 核心实验

| 方法 | 说明 |
|---|---|
| 原始 HAT | 仅轨迹预测 |
| HAT + Future-DINO | 核心方法 |
| HAT + Current-DINO Reconstruction | 排除“任意辅助任务都有效” |
| HAT + Shuffled Future-DINO | 排除仅由正则化导致的提升 |

### 9.2 Shuffled Future-DINO

将batch中的未来DINO标签随机打乱：

```python
permutation = torch.randperm(batch_size)
shuffled_target = future_dino_gt[permutation]
```

如果结果是：

```text
Future-DINO > Current Reconstruction ≈ Shuffled Future-DINO
```

才能比较有力地说明模型确实利用了未来世界变化，而不是仅仅因为多了一个辅助loss。

---

## 10. 评价指标

不能只看 Future-DINO prediction loss。

### 10.1 HAT轨迹指标

- body position error；
- body rotation error；
- wrist position error；
- wrist rotation error；
- fingertip position error；
- 长时间 rollout drift。

### 10.2 下游任务指标

保持后端 controller 完全不变，只替换 HAT：

- 任务成功率；
- 抓取成功率；
- 新物体成功率；
- 新位置成功率；
- 新背景成功率；
- 接触前失败率；
- 接触后失败率。

### 10.3 DINO prediction 指标

- cosine similarity；
- Smooth L1 error；
- changed-region feature error；
- static-background feature error。

重点看：模型是否只预测了大面积静态背景，而没有预测真正发生变化的手和物体区域。

---

## 11. 头部 egocentric 图像的风险

单头部相机下，Future-DINO Head可能主要学习：

- 头部相机运动；
- 整体任务阶段；
- 物体的大致位置变化；
- 背景与视角变化。

它未必能准确表示：

- 指尖和物体的毫米级距离；
- 精确接触关系；
- 抓取力；
- 物体滑动；
- 被手遮挡后的局部状态。

但HAT本身同时有 wrist / fingertip trajectory supervision，因此合理分工是：

```text
Future-DINO Head：
学习物体、环境、任务阶段和整体未来变化

Trajectory Head：
学习手腕、指尖和身体的精细几何运动
```

因此第一阶段不需要要求Future-DINO Head独立解决精细手部控制。

---

## 12. 推荐的最小可行版本

第一版只做以下内容：

1. 冻结现有DINO encoder；
2. 离线计算 `t + H` 时刻的DINO patch target；
3. 在HAT共享trunk后增加4层Transformer Future-DINO decoder；
4. 预测单个未来时刻的DINO patch features；
5. 使用 cosine + Smooth L1 loss；
6. 测试 `λ_world = 0.1 / 0.3 / 1.0`；
7. 推理时删除Future-DINO Head；
8. 与原始HAT、current reconstruction、shuffled future三组对比。

最终核心问题只有一个：

> 在数据、backbone、controller和训练步数都相同的情况下，Future-DINO supervision 是否改善 HAT 的轨迹预测和下游泛化成功率？

如果答案是否定的，就不继续增加更复杂的多步world model；如果答案是肯定的，再扩展到multi-horizon、变化区域加权和将预测latent提供给后端controller。
