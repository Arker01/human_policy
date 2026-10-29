# HAT 增加 Horizon Head（可执行长度 / 可靠性曲线）：实现方案

本文给 codex 作为实现依据。方法名：**Offline Predictability Learning**。
目标是把部署侧写死的 `min_chunk_execution_fraction=0.1` 换成大脑自己预测的可靠执行长度。

> 这是基于已验证组件的组合设计，**不是某篇论文已验证过的完整方法**。所以第 3 节是一个
> 硬性前置门槛，不通过就停下来报告，不要直接进第 5 节训练。

---

## 1. 目标

现有：

```text
RGB + State → HAT → Action Chunk (100 步 @30Hz)
部署端恒定执行前 10 帧（min_chunk_execution_fraction=0.1 × 100）
```

修改后：

```text
RGB + State → HAT → Action Chunk (100 步)
                  → Horizon Head → 可靠性曲线 r_t(k), k=1..100
部署端：K = max{ k : r_t(k) >= τ }，执行 K 帧
```

**预测可靠性曲线，不是预测标量 K。** 阈值 τ 留在部署侧当旋钮，可以不重训就 A/B；
烘进标签里就做不到了。

---

## 2. 硬性约束

1. **不许改动现有真机推理代码和任何已有 checkpoint。** 全部改动放独立副本，做法完全照抄
   上一轮实验：
   - 参照目录：`training_experiments/readiness_geometry_20261020/`（本地归档）
     与 `/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_readiness_geometry/`（训练机）
   - 本轮新建：`/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_horizon_head/`
   - 收尾时对原始源码做 SHA-256 校验并记录，和上一轮一样。
2. 走同一个 `HDT_EXPERIMENT` 环境变量开关机制（`hdt/auxiliary_tasks.py:VARIANT`），
   新 variant 名 `horizon`。`baseline` 路径必须逐比特不变。
3. 训练配置、数据、编码器全部沿用当前被测模型，不要换：

```text
--batch_size 32 --nproc_per_node 2    # 全局 batch 64
--num_epochs 100000 --lr 1e-5 --chunk_size 100 --seed 0
--dataset_json_path  .../configs/replace50.json
--model_cfg_path     .../configs/baseline.yaml
--human_slow_down_factor 4 --robot_target_fps 30
--action_representation original
--use_future_dino_head --future_dino_weight 0.0
--lag_aug --visual_aug --zero_state_dims 98:128 --cond_mask_prob 0.0
```

4. **教师模型（冻结）**：
   `/mnt/nvme0n1/human_policy_ckpt/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt`
5. 数据：`/mnt/nvme0n1/human_policy_ckpt/bottle0930_8runs_20261003_data/`
   —— human_old 152 条、human_new 50 条（30 Hz），robot_new 79 条（50 Hz → 降采样到 30 Hz）。
   `replace50.json` 划分：train 206 条 / val 25 条。

---

## 3. 【前置门槛】重锚后的误差曲线必须先量出来

### 3.1 为什么必须先做这一步

教师模型在**绝对坐标系**下的右腕误差（val，25 条）：

| chunk step k | 1 | 10 | 20 | 30 | 60 | 100 |
|---|---:|---:|---:|---:|---:|---:|
| dex5 | **3.5** | 5.1 | 6.5 | 7.1 | 9.9 | 14.9 |
| human | 2.1 | 2.2 | 2.5 | 2.9 | 4.1 | 5.2 |

k=1 就有 3.5 cm。这不是衰减，是 `--lag_aug` 造成的**站定偏置**：
`lag_aug_prob=0.5`、输入腕位沿过去轨迹回退 `U[2cm, 8cm]`、再叠随机 3D 偏移
（`offset_clip_m=0.08`），而 action 标签一帧不动 —— 模型被显式禁止抄 state 的腕部通道。

而部署侧 `chunk_start_reference_continuity=true` + `--relative-anchor-mode actual`
本来就会把 chunk 重新锚到实测状态，**这个站定偏置大部分会被吃掉**。
若直接用绝对误差当标签，每个 k 都被那 3.5 cm 顶满，`r_t(k)` 会对所有 t 长得一样，
Horizon Head 只能学到一个常数。

误差是模长，不能直接相减，必须按部署方式对齐后重算。

### 3.2 要产出的东西

写一个独立脚本（不进训练路径），用冻结教师在 **val 25 条 + train 206 条**上跑一遍，输出：

1. **重锚后（形状）误差曲线**，按 embodiment 分开：

   对时刻 `t` 的预测 chunk `Â_t`（反归一化到绝对量）与真值 `A_t`，定义
   ```
   shape_err_t(k) = ‖ (q̂_t(k) − q̂_t(0)) − (q_t(k) − q_t(0)) ‖
   ```
   其中 `q(·)` 取**右手场景系指尖**（直接复用 `auxiliary_tasks.scene_tips`，手维取下标 1，
   5 个指尖取均值）。这一个量同时覆盖腕位、腕朝向和开合，且它决定抓取成败。
   另外单独记右腕位置 `action[30:33]` 和右腕朝向角误差（`rotation_6d_rows` + 迹公式）。

   以 step 0 为对齐参考，等价于 `chunk_start_reference_continuity` 的行为。

2. **分阶段分布**。阶段定义直接复用方案 2 的开合检测器
   （`auxiliary_tasks.transition_labels`，右手 close onset），按 `onset − t` 分箱：
   `>60, 40-60, 25-40, 15-25, 8-15, 0-8, <0`。

3. **动态范围统计**：对 `ε = 2/3/5 cm`，算 `H_t(ε) = max{k : max_{j<=k} shape_err_t(j) <= ε}`
   的分位数，按 embodiment × 阶段分箱列出。

### 3.3 通过门槛（不满足就停）

在 **dex5** 上，`ε = 3 cm` 的 `H_t` 必须满足：

- 远段（`onset − t > 60`）中位数 **>= 2 ×** 近段（`0 <= onset − t < 15`）中位数；
- 全体帧的 p25 与 p75 至少相差 **15 帧**。

不满足就把统计结果写进 `results/stage0_anchored_error.json` 并**停止**，在对话里报告
"动态范围不足，Horizon Head 只会学到常数"，不要继续训练。

> 参考：绝对坐标系下已经量到的分阶段一致性（ε=3 cm）是远段中位 51 帧 / 近段 10–39 帧，
> 所以预期是能过的，但必须在重锚后复核。

---

## 4. 标签定义

### 4.1 两个信号都要，不能只用一致性

**信号 A：与示教的形状残差**（第 3.2 节的 `shape_err_t(k)`）。
**信号 B：相邻时刻 chunk 的重叠一致性**。取重规划间隔 `Δ = 10`（= `--replan-frames 10`）：

```
cons_t(k) = ‖ (q̂_t(k) − q̂_t(Δ)) − (q̂_{t+Δ}(k−Δ) − q̂_{t+Δ}(0)) ‖     k >= Δ
cons_t(k) = 0                                                          k <  Δ
```

两边都在交接点上对齐，理由同 3.1。`k < Δ` 置 0：那几帧在任何新 chunk 落地前就已经执行完了。

**必须两个都用。** 实测 ε=3 cm 时有 **18% 的帧整条 chunk 都"一致"，而同一批帧 err@30 是
7.1 cm** —— 不同时刻的模型稳定地预测同一条错误轨迹，一致性会很高。只看一致性会把这 18%
标成"全程可靠"。

### 4.2 可靠性标签

```python
reliable_t(k) = 1  iff  max_{j<=k} shape_err_t(j) <= EPS_RES
                  and   max_{j<=k} cons_t(j)      <= EPS_CONS
```

前缀形式 ⇒ 关于 `k` 天然单调非增。默认 `EPS_RES = EPS_CONS = 0.03`（米）。

**标签缓存里要同时存 ε = 2/3/5 cm 三套**，这样换容差不用重跑教师。

派生量 `K_t = max{k : reliable_t(k) = 1}`（全 0 时为 0），只用于日志和评估，不作为训练目标。

### 4.3 只在 dex5 上监督

human 数据慢放 4 倍（`--human_slow_down_factor 4`，`_read_action_chunk` 的 human 分支经
`interpolate_128dim_action` 上采样 4 倍），100 个 chunk 步只覆盖 25 原始帧 ≈ 0.83 s 真实运动。
慢放对动作预测是好事（human 误差因此很平：2.1 → 5.2 cm），但**对这个标签就没有方差了**
—— 实测 human 一致视界中位恒为 10。而采样器 `P_ROBOT=0.5`，等于一半 batch 在教常数。

所以：`horizon_mask = (embodiment == 'dex5')`，human 样本该头损失为 0。
human 侧标签仍然算、仍然存，只用于诊断对比。

### 4.4 离线标注的实现方式

**一次顺序扫描，不要缓存全部 chunk。** 对每条 episode 按逻辑帧顺序推进，维护一个长度 11 的
`deque` 存最近的预测（只存右手需要的量：5×3 指尖 + 3 腕位 + 6D 朝向 = 27 维 × 100 步），
就能同时算 A 和 B，内存 O(1)。

- 规模：231 条 × 平均 ~300 逻辑帧 ≈ 7 万次前向，batch 16 单卡几分钟级，不需要多卡。
- 产物：每条 episode 一个 `.npz`，键为
  `reliable_eps002 / reliable_eps003 / reliable_eps005`，形状 `(n_frames, 100)` 的 `uint8`，
  外加 `shape_err (n,100) float16`、`cons (n,100) float16`、`K_eps003 (n,) int16`。
- 存放：`<isolated>/labels/<episode_stem>.npz`，另写一份
  `labels/manifest.json` 记录教师 ckpt 的 SHA-256、`Δ`、各 ε、以及每条 episode 的帧数。
  训练启动时校验 manifest 与当前 `replace50.json` 一致，不一致直接报错退出。

---

## 5. 网络改动

全部改动集中在和上一轮相同的四个文件，改法照抄 `readiness` 分支。

### 5.1 `hdt/auxiliary_tasks.py`：新增 Head 与损失

```python
class HorizonHead(nn.Module):
    """逐 chunk-step 的可靠性生存曲线。

    直接读解码器输出 hs（[B, num_queries, hidden]），它本来就带 per-chunk-step 表示，
    不用再造一套 query。输出构造成单调非增，和前缀标签的单调性一致。
    """
    def __init__(self, dim):
        super().__init__()
        self.trunk = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU())
        self.hazard = nn.Linear(128, 1)    # 每步的非负"风险"增量
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, hs):
        h = self.trunk(hs)
        hazard = F.softplus(self.hazard(h)).squeeze(-1)      # [B, T] >= 0
        return self.bias - torch.cumsum(hazard, dim=-1)      # logits，关于 k 单调非增
```

单调靠构造保证，不要靠损失去"鼓励"单调 —— 部署侧要取 `max{k : r(k) >= τ}`，
非单调曲线会让这个取法有歧义。

```python
def horizon_loss(logits, target, valid):
    """valid = (~is_pad) & (embodiment == dex5)，形状 [B, T]。"""
    element = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    return (element * valid).sum() / valid.sum().clamp_min(1)
```

### 5.2 `hdt/detr/models/detr_vae.py`

- `DETRVAE.__init__`：在 `readiness_head` 旁边加
  `self.horizon_head = HorizonHead(hidden_dim) if self.experiment_variant == 'horizon' else None`，
  以及 `self.horizon_logits = None`。
- `forward`：在 `hs = self.transformer(...)` 之后、`a_hat = self.action_head(hs)` 旁边加
  `if self.horizon_head is not None: self.horizon_logits = self.horizon_head(hs)`。
  放在公共路径上（不要只在训练分支），否则推理时取不到。

### 5.3 `hdt/data_utils_hdt.py`

- `dataset_aux()` 里，`VARIANT == 'horizon'` 时从 `.npz` 读 `reliable_eps00X[raw]`
  得到 `(100,)` 的标签，连同 `horizon_valid`（标量：是否 dex5）一起塞进 `conditioning_dict`。
  **human 分支的时间轴换算照抄现有 `hand_event_weights` 的做法**
  （`linspace(raw, raw+available-1, available*factor)` 取 floor），human 只做诊断，但索引要对。
- `collate_fn` 的 allow-list 里加 `horizon_target`、`horizon_valid`。
  这个 allow-list 会静默丢掉没列名的键，漏了就会训出一个没有损失的死头。

### 5.4 `hdt/policy.py`

```python
elif self.model.experiment_variant == 'horizon':
    logits = self.model.horizon_logits
    target = conditioning_dict['horizon_target']
    valid  = (~is_pad) & conditioning_dict['horizon_valid'][:, None].bool()
    hz = horizon_loss(logits, target, valid)
    loss_dict['horizon_loss'] = hz
    loss_dict['horizon_valid_frac'] = valid.float().mean()
    loss_dict['horizon_pos_frac'] = (target * valid).sum() / valid.sum().clamp_min(1)
    loss_dict['horizon_K_pred'] = ((logits > 0).float() * valid).sum(-1).mean()
    loss_dict['horizon_K_true'] = ((target > .5).float() * valid).sum(-1).mean()
    loss_dict['loss'] = loss_dict['loss'] + 0.1 * hz
```

权重先用 **0.1**。注意上一轮的教训：`readiness` 用 0.1 时该头梯度范数 0.108，而
`action_head` 是 10.8，差两个数量级，对主干几乎没有塑形作用。**这里是可以接受的** ——
Horizon Head 的目的就是读出一个量，不是改造表示。但要把梯度范数记进日志，别误判。

### 5.5 `hdt/main.py`

- `policy_config['experiment_variant'] = VARIANT`（已有）。
- 第 0 步梯度检查的 group 列表里加 `'horizon_head'`，并加
  `if inner_model.experiment_variant == 'horizon': assert groups['horizon_head'] > 0`。
  这是上一轮抓住"死头"的机制，必须保留。

---

## 6. 校准（必须分阶段做）

训练完在 **val 25 条**上做，不要用训练集。

1. 对每个阈值 `τ ∈ {0.5, 0.7, 0.8, 0.9, 0.95}`，算 `K̂_t = max{k : r_t(k) >= τ}`。
2. **实际覆盖率**：在 `K̂_t` 处真实的 `max_{j<=K̂} shape_err_t(j) <= EPS_RES` 的帧占比。
   这是"0.94 到底是不是 94%"的检验。
3. **分阶段分别校准**。实测误差分布随阶段整体平移（远段 err@10 3.2 cm → 近段 5 cm，
   ε=3 cm 一致视界远段 51 帧 → 近段 10–39 帧），用全局校准会恰好在抓取那一段偏掉。
   阶段用方案 2 的 close-onset 距离分箱，产出一张 `τ → 分阶段覆盖率` 表。
4. 选 τ 的准则：**覆盖率 >= 0.9 的前提下 K̂ 的中位数最大**。写进
   `results/calibration.json`，部署侧读这个文件，不要把数字硬编码进代码。

---

## 7. 部署接线

### 7.1 生产侧 `hdt/online_hat_service.py`

- `action_chunk_to_targets()`（约 324–366 行）构造 chunk dict 的地方，新增两个字段：
  - `reliability`: `float[frame_count]`，即 `r_t(k)`；
  - `execution_frames`: `int`，按 `calibration.json` 的 τ 算好的 `K̂`。
- 加载器 `_load_state_dict()` 是 `strict=False` 且只对 `missing_keys` 报错，所以
  **不带 horizon 头的旧 ckpt 不受影响**；反之新 ckpt 跑旧配置时 horizon 头会被忽略并打印一行。
- 新增 CLI 开关 `--horizon-head {off,publish}`，默认 `off`：`off` 时一个字段都不写，
  wire 格式与现在逐字节一致。

### 7.2 消费侧 `ScaleBFM/ScaleBridge`

`scalebridge/env/motion_tracking_hat4_online.py::_chunk_execution_fraction()`（约 1883 行）
已经有一条自适应路径（`focus_adaptive.execution_frames`，约 1894–1896 行），按同样的形状插入：

```python
if bool(self.cfg.get("horizon_head_enabled", False)) and chunk is not None:
    frames = chunk.get("execution_frames")
    if frames is not None:
        return min(1.0, self._guard_execution_frames(chunk, int(frames))
                        / int(chunk["frame_count"]))
```

字段缺失或开关关闭 ⇒ 完全走现有逻辑。新增 cfg 默认值：

```yaml
horizon_head_enabled: false
horizon_min_frames: 10            # 下限，等于现在的 replan 间隔
horizon_min_displacement_m: 0.01  # 位移护栏
```

### 7.3 位移护栏（必须有，且不要塞进标签）

实测 dex5 腕速中位 0.11 m/s（10 帧 = 3.7 cm），但 **p10 只有 0.025 m/s（10 帧 = 0.8 cm）**。
有相当一批帧在执行窗口内位移是亚厘米级，即使预测全对，小脑的跟踪特性也可能几乎不动。

这是部署侧护栏，不是标签问题 —— 不要把它混进回归目标里：

```python
def _guard_execution_frames(self, chunk, frames):
    """执行到「可靠性跌破阈值」或「计划累计位移达到下限」中更晚的那个。"""
    d_min = float(self.cfg.get("horizon_min_displacement_m", .01))
    floor = int(self.cfg.get("horizon_min_frames", 10))
    path = np.linalg.norm(np.diff(chunk_right_wrist_xyz, axis=0), axis=-1).cumsum()
    k_disp = int(np.searchsorted(path, d_min)) + 1
    return int(np.clip(max(frames, k_disp, floor), floor, int(chunk["frame_count"])))
```

位移下限管一头、可靠性上限管另一头，各自独立可调。

---

## 8. 验收标准

必须全部满足并各自给出数字：

1. **Stage 0 门槛**（第 3.3 节）通过，统计写入 `results/stage0_anchored_error.json`。
2. 第 0 步梯度检查里 `horizon_head > 0`，`action_head > 0` 且全部有限。
3. 训练满 100,000 步，`val/horizon_loss` 下降，`val/horizon_K_pred` 与
   `val/horizon_K_true` 同向移动。
4. **val 上的预测质量**：`K̂` 对 `K` 的 Spearman 相关 >= 0.5，MAE 以帧为单位报出；
   并给出"按 `K̂` 执行时实际形状误差 <= ε"的覆盖率（校准后的 τ 下应 >= 0.9）。
5. **主任务不许退化**：用上一轮同一套口径（`src/evaluate_experiment.py` 的
   `prior_z_zero_no_future_actions` 固定验证 + `src/mpjpe_comparison.py`）对比原模型，
   dex5 的 query0 右腕误差和全身 MPJPE 退化不得超过噪声水平
   （参照：四次独立重训的 query0 腕误差都落在 3.4–3.6 cm，所以超过 0.2 cm 就要解释）。
6. 原始源码 SHA-256 与改动前一致；`--horizon-head off` 时 wire 格式不变；
   `horizon_head_enabled: false` 时 ScaleBridge 行为不变（补一个单测，参照
   `tests/test_hat_online_chunk_hold.py`）。

---

## 9. 已知坑（都是实测踩过的，别重复）

1. **k=1 就有 3.5 cm，不是 bug。** 是 `--lag_aug` 的设计代价（半数样本输入腕位被回退
   2–8 cm 而标签不动）。不要试图"修"它，也不要用绝对误差做标签（见 3.1）。
2. **一致性高 ≠ 对。** ε=3 cm 下 18% 的帧整条 chunk 一致但 err@30 = 7.1 cm。
   残差项必须参与，且权重不能小于一致性项。
3. **一致性信号在 k=Δ 处是阶跃，不是平滑衰减。** 实测 dex5 ε=2 cm：中位 10、p75 25、p90 81，
   双峰且在 10 处截尾。单调生存曲线的参数化（第 5.1 节）就是为了吃下这个形状。
4. **不要用曲率/匀速外推当标签。** 实测 ε=2 cm 的匀速外推视界 human 中位 5 帧、robot 7 帧，
   几乎没有动态范围；且在 human 上**非单调**（静止 20 帧 → 快速伸手 5 帧 → 接近物体回到 8–11 帧），
   会让机器人在搬运段重规划最频繁，方向是反的。
5. **不要依赖 `focus_adaptive_near_distance_m=0.045` 这个现成触发器做阶段判断。**
   实测这个阈值在 robot 数据上是 close onset 前 **88 帧（2.9 s）** 就被跨过，在 human 上只有
   13 帧 —— 在真机上它几乎全程处于 near 模式，等价于现在的 0.1。阶段判断用方案 2 的
   开合检测器。
6. **`collate_fn` 的 allow-list 会静默丢键**（`data_utils_hdt.py` 约 1229 行附近有注释写了
   这个坑）。加新键必须同时改 allow-list，并靠第 0 步梯度断言兜底。
7. **这个头不治"大脑看到瓶子还在前面却预测已到位"。** 那是预测偏差，缩短执行长度只是更频繁
   地重规划，而每个新计划带同样的偏差。那件事的杠杆是到位门控
   （`auxiliary_tasks.gate_hand_commands`，目前尚未接入）。两件事分开评，别混进同一个指标。

---

## 10. 预期收益（先对齐期望）

按实测的分阶段一致性（远段 ε=3 cm 中位 51 帧 / 近段 10–39 帧），这个头的主要作用是
**在搬运段把执行拉长**，减少 prefix 被反复打断导致的停滞；**不是**在抓取段缩短
（那里本来就已经压在 10 帧的下限上）。

所以评估指标应该看运动连续性和搬运段推进速度，不要指望抓取成功率因此变好。
