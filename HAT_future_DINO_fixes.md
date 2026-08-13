# Future-DINO (WAM) 修复记录

对应设计文档：`HAT_future_DINO_head_implementation.md`

**改动范围约束**：只改 Future-DINO 这个新增功能相关的代码。加这个功能之前就存在的逻辑
（`head_loss` / `waist_loss` / `query0_loss`、`OUTPUT_NECK` 置零、dirty-start 裁剪、
train/val 数字排序、MPJPE 16 关节等）一律没动。所有新代码都在 `future_dino.enabled`
开关后面，关掉时（默认）行为和改之前逐字节一致 —— 这一点由测试
`test_disabled_path_is_unchanged` 和 `collate_fn` 的 5-tuple 分支覆盖。

---

## 一、先说结论：原来的代码为什么等于没跑

7/31 那次 100k step 的训练里，Future-DINO 头**一次梯度都没收到**。证据：checkpoint 里
78 个 `future_dino_head.*` 张量，其 LayerNorm 权重在 100000 步之后仍然精确等于初始化值
（`==1.0` / `==0.0`，max_dev `0.000e+00`），而 trunk 同类参数已经漂了 `1.95e-01`；
`optimizer.bin` 里 341 个参数只有 261 个有 state（AdamW 的 state 是 lazy 创建的）。

根因是四个独立的 bug 叠在一起，任何一个单独存在都足以让这个功能变成死代码。

---

## 二、逐项修复

### P0-a　dataloader 根本没产出 `future_image`（**这一条就是主因**）

`read_one()` / `collate_fn()` 只返回 5 个元素，`future_image` 永远是 `None`，
`detr_vae.py` 里整个 Future-DINO 分支从来没进去过。文档 §2 的离线 target 预计算也完全没实现。

**改**（`hdt/data_utils_hdt.py`）：

- `EpisodicDataset.__init__` 新增 `future_image_enabled` / `future_horizon` 两个开关参数。
- 新增 `_read_cam_images_at()` / `_resolve_cam_name()`：按任意时间戳解码所有相机。
  未来帧是监督 target，所以没有 `cond_mask` 置零分支。
- `read_one()` 在开启时读 `t+H` 帧，返回 6-tuple，并在 `conditioning_dict` 里带一个
  `future_valid` 标量。
- `collate_fn()` 按 tuple 长度自动分支；关闭时走原来的 5-tuple。
- `load_data()` 透传这两个参数。

顺带处理了三件设计文档要求、但实现里没有的事：

1. **episode 末尾要 mask，不能 clamp。** `t+H` 越界时仍然读最后一帧，但 `future_valid=0`，
   loss 里整条样本被屏蔽。直接 clamp 会教模型"未来是静止的"，在 pick-and-place 这种
   末尾就是静止的数据上尤其有害。
2. **human 数据的 horizon 要按 `slow_down_factor` 缩放。** human episode 的 action chunk 被
   插值压缩过：64 个 action 实际只覆盖 `64/slow_down_factor` 个原始帧。视觉 horizon 不跟着缩，
   world target 就跑到 trunk 正在预测的动作区间之外去了。
3. **当前帧和未来帧共用同一次数据增强**（文档 §2.3）。现在把两帧 concat 成一个 tensor 走
   *一次* `visual_preprocessor` + `training_transforms`；torchvision v2 每次调用只采样一组
   ColorJitter 参数，所以两帧拿到完全相同的光度变换（已实测 `torch.allclose == True`）。
   代码里没有几何增强，所以几何这块无需同步。

> 注：文档 §2.2 要求把 DINO target **离线预计算**成 `[N, P, D]` 存盘。这里做的是在线计算
> （每步多跑一次冻结 DINOv2 前向）。功能等价，代价是训练慢一些；好处是不用管缓存失效，
> 也不用为每个 horizon / 每次数据增强各存一份。如果吞吐成为瓶颈再改离线。

### P0-b　world loss 的梯度到不了共享 trunk

```python
hat_memory = src.flatten(2).permute(0, 2, 1)   # 旧
```

`src` 是 `get_features_and_pos()` 的输出，也就是 **backbone → input_proj 之后、
`self.transformer` 之前**。文档 §3 明确要求"Future-DINO Head 必须读取 HAT 的共享 trunk
feature，确保 world prediction 的梯度真正作用于轨迹预测所使用的共享表征"——这里直接违反了。

实测（只对 world loss 做 backward，按模块统计 `p.grad.abs().sum()`）：

| 模块 | 修复前 | 修复后 |
|---|---|---|
| `backbones` | 2.87e+03 | 1.23e+03 |
| `future_dino_head` | 2.87e+03 | 9.71e+02 |
| `input_proj` | 1.24e+02 | 9.99e+01 |
| **`transformer`（共享 trunk）** | **无梯度** | **6.88e+02** |
| `query_embed` | 无梯度 | 9.65e-03 |
| `encoder` | 无梯度 | 1.91e+00 |

**改**（`hdt/detr/models/detr_vae.py`）：`hat_memory = hs`，也就是 action head 消费的
那个 trunk 输出。这条由 `test_world_gradient_reaches_trunk()` 断言守住。

### P0-c　warmup step 传不进去，`effective_weight` 永远是 0

```python
if hasattr(policy, 'model') and hasattr(policy.model, 'set_training_step'):   # 旧
```

accelerate 下 `policy` 是 `DistributedDataParallel`，`policy.model` 不是 ACT 模型，
`hasattr` 直接 False → step 永远是 0 → `warmup_factor = 0` → `effective_weight = 0`。
RDT 路径更彻底：`HumanDiffusionTransformer` 压根没定义过 `_training_step`，
`get_current_step()` 恒返回 0。

**改**：
- `hdt/main.py` `forward_pass()`：先 `policy.module if hasattr(policy, 'module')` 解包 DDP，
  再对 `inner` 和 `inner.model` 都尝试 `set_training_step`。
- `hdt/modeling/modeling_hdt.py`：`__init__` 里初始化 `self._training_step = 0`，补上
  `set_training_step()`。
- `hdt/policy.py`：把 `future_dino_effective_weight` 也写进 `loss_dict`。
  **这条曲线是这次事故最直接的哨兵** —— 它要是一直贴着 0，头就又死了。

由 `test_warmup_reaches_model_through_forward_pass()` 覆盖，含 DDP 包装场景。

### P1　target encoder 必须冻结，而且不能是 trunk 自己的 backbone

旧实现里 query 和 target 都来自 `self.backbones[0]`，也就是**正在被训练的 resnet18**。
teacher 跟着 student 一起动，world loss 完全可以靠把特征塌缩掉降下去，而不是靠预测未来。
`detr_vae.py` 那份 `FutureDINOLoss` 甚至没有 `.detach()` target（`modeling_hdt.py` 那份有）。

按你说的：**trunk 不必换成 DINO，resnet18 继续联合训练没问题，只有喂给 Future-DINO 头的
target 必须冻住。** 所以：

**改**（`hdt/detr/models/detr_vae.py`）：新增 `FrozenPatchTargetEncoder`。

- trunk backbone 保持 `resnet18` + `lr_backbone: 1e-5`，照旧训练，**没动**。
- 另起一个**独立冻结**的 encoder，只负责产出 query tokens（第 t 帧）和 target tokens（第 t+H 帧）。
  默认 `dinov2_vits14`（384 维，权重已缓存在 `~/.cache/torch/hub/`）；离线机器可以退回
  `resnet18` / `resnet34`（冻结 ImageNet 权重，target 质量差些，但"冻结"这个关键性质还在）。
- query 和 target 出自**同一个**冻结 encoder，所以在同一个特征空间里，头只需要建模"变化量"。
- `forward_features()['x_norm_patchtokens']` 天然已经去掉 CLS 和 register token（文档 §2.2）。
- 输入不是 patch_size 整数倍时内部 bilinear resize（真实路径上 `ACTPolicy.transform`
  已经给到 224×308 = 16×22 patch，是 no-op；这只为直接调 encoder 的测试/探针兜底）。
- **故意不注册进 module tree**（父模块用一个普通 list 持有）：22M 参数不进 `state_dict`、
  不进 optimizer、不进 DDP 通信桶。device 放置在 forward 里 lazy 完成。
  由 `test_target_encoder_is_frozen_and_off_ledger()` 断言。

loss 侧另外加了：target 上硬 `.detach()`（双保险），以及 `normalize_target`
（逐 token L2 归一化，文档 §2.2 `tokenwise_normalize`，让 Huber 项 scale-free）。

### P2　patch 数量静默截断

```python
min_patches = min(pred_future_dino.shape[1], future_target.shape[1])   # 旧
```

注释写着 "Align shapes"，实际是在悄悄丢 patch。`num_patches` 默认 150，而真实 patch 数是
352，等于永远只监督前 150 个。

**改**：`num_patches` 改为"上界"语义（位置表大小，默认 1024，forward 时切片），
超了就 assert 报错而不是截断；pred / target 形状不一致也 assert。

### P3　RDT 路径：world head 能直接看到答案

```python
hat_memory = torch.cat([hat_memory, state_action_traj], dim=1)   # 旧
```

`state_action_traj` 里装的是**加噪后的 ground-truth 未来动作**。把它喂给 world head，
头可以直接读答案，world loss 降了但 trunk 什么也没学到。

**改**（`hdt/modeling/modeling_hdt.py`）：只保留 `img_cond`。

### 其他

- `data/diag_z_shortcut.py:90`：`DETRVAE.forward` 现在返回 4 个值，这里还在按 3 个解包，
  一跑就 `ValueError`。改成取 `[0]`。这行是被新功能改坏的，所以算在范围内。
- `hdt/configs/models/rdt_with_future_dino.yaml`：`enabled` 默认改成 `false`（见下方"未做"）。
- 新增 `scripts/train/train_future_dino.sh` —— **之前根本没有这个功能的启动脚本**，
  这本身就是"训练从没跑过"的一条旁证。

---

## 三、新增测试

`hdt/test_future_dino_e2e.py`。

原有的 `test_act_future_dino.py` / `test_future_dino.py` 是**假阳性**：它们手工构造
`future_image` 直接调模型，绕开了 dataloader 和 `forward_pass`，所以在功能实际已死的
100k 步训练期间它们一直是绿的。新测试专门打真实训练路径上断掉的那几处：

| 测试 | 守住的东西 |
|---|---|
| `test_collate_returns_future_tuple` | dataloader 真的产出 `future_image` + `future_valid`；5-tuple 老路径不变 |
| `test_world_gradient_reaches_trunk` | **共享 transformer trunk 真的收到 world 梯度**（P0-b） |
| `test_target_encoder_is_frozen_and_off_ledger` | teacher 冻结，且不在 `state_dict` / optimizer 里 |
| `test_warmup_reaches_model_through_forward_pass` | 0 → 0.15 → 0.30，DDP 包装下也成立（P0-c） |
| `test_invalid_future_is_masked_out` | 越界样本 loss 为 0 |
| `test_disabled_path_is_unchanged` | 关掉时 loss dict 和原来完全一样 |

三个测试文件现在全绿。另外在真实数据上验证过完整一步：

```
step=    0  loss=106.63  l1=0.6557  fd=1.0568  cos=0.9768  w=0.000
step=  500  loss= 98.47  l1=0.6564  fd=1.0554  cos=0.9771  w=0.150
step= 1000  loss= 96.39  l1=0.6551  fd=1.0593  cos=0.9803  w=0.300
total-loss grad: transformer=1.015e+05  future_dino_head=5.429e+02
```

dataloader 也在 `data/dex5_val/` 真实 hdf5 上验证过：`start_ts=221`（episode 长 222）
正确给出 `future_valid=0`。

---

## 四、怎么跑

```bash
# lambda_world 扫描（文档 §7）
scripts/train/train_future_dino.sh 0 0.0     # baseline，头挂着但不出力
scripts/train/train_future_dino.sh 1 0.1
scripts/train/train_future_dino.sh 2 0.3
scripts/train/train_future_dino.sh 3 1.0

# 消融（文档 §9）
scripts/train/train_future_dino.sh 0 0.3 shuffled   # 打乱 t -> t+H 配对
scripts/train/train_future_dino.sh 1 0.3 current    # 退化成当前帧重建
```

开跑后头 50 行必须确认两件事，否则就是又死了：

1. `future_dino_effective_weight` 从 step 0 起就等于配置的 weight（不是 0）；
2. `train/future_dino_cosine_loss` 从 ~1.0 开始往下掉。

新增的 config 项都在 `hdt/configs/models/act_with_future_dino.yaml`：
`target_encoder` / `horizon` / `num_patches` / `normalize_target` / `ablation`。
对应 CLI 覆盖：`--future_dino_horizon`、`--future_dino_ablation`。

---

## 五、没做的部分（明确说明）

1. **RDT / HDT 路径没有真正修好。** 它的 memory 仍然是 pre-trunk 的 `img_cond` 而不是
   RDT block 的输出，world 梯度到不了 diffusion trunk，文档 §3 在这条路径上依然不满足；
   而且 `current_dino_tokens = image_embeds` 是已经 `.detach()` 过的，视觉编码器拿不到梯度。
   真实的训练跑的全是 ACT，所以我只修了它明确的两个 bug（warmup step、动作泄漏），
   并把 config 默认置为 `enabled: false` + 加了醒目注释。要用 RDT 需要先把 trunk 输出暴露出来。

2. **`metrics.csv` 表头 bug 没动。** `_append_metrics_csv()` 只在文件不存在时写一次表头，
   之后 union 了 fieldnames 却从不重写磁盘上的表头，导致 8 列表头 / 16 列数据，
   `_plot_metrics` 永远画不出 `train/*` 曲线。这是**加这个功能之前就有的** bug，
   会影响所有 run 的 CSV 格式，按你的要求没碰。一行修法：在 fieldnames 扩张时把整个
   CSV 重写一遍。在此之前，warmup 和 world loss 请直接看训练 stdout。

3. **文档 §8 的三阶段训练（先冻 trunk 只训头 → 解冻末层 → 联合微调，头用 5–10× lr）没实现，
   而且不该实现。** §8 原文的前提是"**如果从现有 HAT checkpoint 开始**，推荐分三步"——
   那是一份 finetune 配方。我们是从头训练（`scripts/train/run_100k_worker.sh` 不带
   `--load_pretrained_path`，只有 `scripts/finetune/*` 带），前提不成立，所以不适用。
   现在是单阶段联合训练。详见下面第七节。

4. **DINO target 没有离线预计算**（见 P0-a 下方的注）。

5. **两份 `FutureDINOHead` / `FutureDINOLoss` 仍然是重复实现**（`detr_vae.py` 一份、
   `modeling_hdt.py` 一份，维度不同）。合并会动到 RDT 路径，在 RDT 没修好之前先不动。

---

## 六、顺带澄清的一件事（和这次修复无关）

你记得"7 月初 <2cm、7/31 变成 5-6cm"，怀疑是 DINO 换 resnet18 导致的。查下来不是：

- 四个 7 月 checkpoint **全都是 resnet18**（`layer4.1.conv2.weight` 形状 `(512,512,3,3)`），
  没有任何一次 run 用过 DINO backbone。
- 7/31 **没有退化**。同一个 eval set（`data/dex5_val`，`--max-steps 200`）：
  robot-only 7/16 = 47.57 mm，mixed 7/16 = 47.45 mm，future 7/31 = 47.77 mm，相差 0.3 mm 以内。
  训练曲线也几乎重合（final loss 0.4974 vs 0.4969）—— 7/31 那次实际上就是 7/16 mixed run
  重跑一遍，外加一个死掉的头。
- 真正的原因是**测的数据集不一样**。同一个 `train_pillow_ckpt`：
  `human_train`（训练集）**23.12 mm** / `human_val` 39.43 mm / `dex5_val` 47.45 mm。
  记忆里的 ~2cm 是在**训练集**上测的，5-6cm 是在**验证集**上测的。

顺带一个发现：train 23mm vs val 39mm，接近 2 倍的差距，是实打实的过拟合。
这本身就是给 world 辅助监督的一个正当理由。

---

## 七、warmup 已经默认关掉（`warmup_steps: 1000` → `0`）

**结论：不需要 warmup，从头训练直接开就行。** 这一条是量出来的，不是推的。

文档 §7 要求 warmup 是**有条件**的：原文是"如果两个 loss 数量级差异很大，可以使用：
loss normalization / gradient norm balancing / warm-up world loss"，目标是"world loss 能影响
shared trunk，但不能在训练早期完全压过 trajectory loss"，而它要求记录的判据是
**shared trunk gradient norm**。所以就量这一个数。

用真实 config（`act_with_future_dino.yaml`）+ 真实数据（`data/dex5_val`，batch 8，chunk 100），
在 **step 0**（head 完全随机初始化，最坏情况）分别只 backward 轨迹项和 world 项，
统计 `model.transformer.*`（共享 trunk）的梯度范数，5 个 batch：

| | trajectory loss | world loss (λ=1) |
|---|---|---|
| loss 值 | 76 ~ 107 | 1.03 ~ 1.06 |
| **共享 trunk 梯度范数** | **4.08e+02** | **1.44e+01** |

换算成实际配置的 λ：

| λ_world | world / trajectory 的 trunk 梯度比 |
|---|---|
| 0.1 | 0.35% |
| **0.3（默认）** | **1.06%** |
| 1.0 | 3.53% |

即使 λ=1.0，world 项对 trunk 的贡献也只有轨迹项的 3.5%，**根本压不过谁**。
§7 那个"数量级差异很大"的触发条件不成立，warmup 只会白白推迟 head 开始学习。

另外 warmup 那套说辞里"保护已经训好的 trunk"的理由在我们这里也不存在——
我们是**从头训练**，`scripts/train/run_100k_worker.sh` 没有 `--load_pretrained_path`，
7 月那几次 100k step 的 run 全是 from scratch。文档 §8 的三阶段是给"从现有 checkpoint 开始"
准备的，跟我们无关。

改动：

- `hdt/configs/models/act_with_future_dino.yaml`：`warmup_steps: 1000` → `0`，注释里写清测量数字；
- `scripts/train/train_future_dino.sh`：`--future_dino_warmup_steps 1000` → `0`。

warmup 的代码路径**保留**（`warmup_steps: 0` 时 `effective_weight` 从 step 0 就是配置的 weight），
以后要是把 λ 调到远大于 1，可以直接把它调回来。三个测试文件改完全部通过。
