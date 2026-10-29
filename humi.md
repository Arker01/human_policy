按你现在的目标，我建议只做一个**新增模式**，不要动原数据、不要覆盖现有 HAT pipeline，也先不要改 ACT loss。

### 要改成什么

现有数据照旧存：

$$
K_t,K_{t+1},...,K_{t+H}
$$

读取一个 sample 后，**临时在 dataloader/preprocess 里构造新的 action label**。

新模式定义为：

$$
\boxed{
A_{t+i}=T_t^{-1}T_{t+i},\quad i=1...H
}
$$

也就是**整个 future chunk 都相对于当前时刻 \(t\) 的 KP pose 表示**。

position：

$$
\Delta p_{t+i}=p_{t+i}-p_t
$$

rotation：

$$
\Delta R_{t+i}=R_t^{-1}R_{t+i}
$$

注意：**不是相邻帧相减**，不是

$$
p_{t+i}-p_{t+i-1}.
$$

这就是我们刚才讨论的 HuMI/UMI-style relative chunk。

---

### 代码修改计划

1. **原始 dataset 完全不改、不重新生成、不覆盖。**

读取原来的 image、current KP、future KP 后，再在线做：

```python
current_kp = sample["kp"][t]
future_kp = sample["kp"][t+1:t+H+1]

relative_future_kp = transform_to_current_frame(
    current_kp,
    future_kp
)
```

原始文件保持原样。

2. **新增 config 开关**

例如：

```yaml
action_representation: original
```

或：

```yaml
action_representation: relative_chunk
```

旧实验完全可以复现。

3. **输入暂时不改**

继续：

```text
image + current KP
        ↓
       HAT
```

不要现在加 DOF-pos，也不要删 KP。

4. **输出 shape 不改**

仍然：

```python
[B, H, action_dim]
```

只是 `relative_chunk` 模式下，每一个 future KP 表示：

```text
相对于当前 t 时刻 KP 的位移/旋转
```

所以 ACT 网络结构基本不用改。

5. **loss 完全保持你现在 HAT 的 loss**

先继续现有的 L1 / wrist weighting。

不要第一版加 cumulative loss，不要换 Flow Matching。

这样实验最干净：

> **只改变 action representation，其余全部保持一致。**

6. **normalization 必须重新统计**

原 dataset 不变，但 `relative_chunk` action 经过转换后，要单独计算：

```text
relative_action_mean
relative_action_std
```

保存成新的 stats 文件，例如：

```text
stats_original.pkl
stats_relative_chunk.pkl
```

绝对不能复用旧 action normalization。

7. **inference 时 inverse transform**

网络预测：

$$
\hat A_{1:H}
$$

以后，需要通过 anchor 恢复成 ScaleBFM 真正需要的 KP：

$$
\hat T_{t+i}=T_{\rm anchor}\hat A_{t+i}
$$

position 即：

$$
\hat p_{t+i}
=
p_{\rm anchor}+\Delta\hat p_{t+i}.
$$

---

### 最关键：部署时 anchor 怎么选

第一次预测：

$$
T_{\rm anchor}=T_{\rm robot,current}
$$

之后不要每次都无脑用 robot actual KP 重新 anchor。

维护：

```python
scheduled_anchor
```

新 chunk 来时优先接到**当前计划应该执行到的位置**：

$$
T_{\rm anchor}=T_{\rm scheduled}
$$

而不是 ScaleBFM 实际只走到的：

$$
T_{\rm actual}.
$$

这样如果 ScaleBFM 少走 2–3 cm，计划不会跟着退回来，tracking error 可以自然累积，把它从 deadzone 拉出来。

同时加 safety：

```python
if distance(scheduled_kp, actual_kp) > threshold:
    reanchor / stop / replan
```

阈值真机回来以后再定。

当前部署实现：

- `dataset_stats.pkl` 的 `action_representation=relative_chunk` 会自动启用
  inverse，不需要额外手工开关；`original` checkpoint 完全沿用旧路径。
- 第一次以 robot actual pose 为 anchor；之后根据相邻 RobotState 的
  `timestamp_ns`，从上一条 absolute scheduled chunk 选出当前时刻对应帧，
  不使用容易受推理抖动影响的固定行号。
- 相机/状态断档超过上一 chunk 的 horizon，或者 opening origin/heading 尚未
  锁定时，自动回到 actual pose 重新建立计划链。
- `--scheduled-anchor-max-error-m N` 可设置安全阈值。它比较 head、左右腕和
  waist 的最大位置误差；超限后本轮立即 actual re-anchor。默认 `0`，表示真机
  定阈值前不擅自启用，但误差和 anchor 来源仍会写入 inference record。
- wire protocol 会携带 `action_representation`。ScaleBFM 对 `relative_chunk`
  默认使用 `relative_chunk_min_execution_fraction: 0.0`，按 replan 频率接收，
  避免旧模式的 2.2 秒 chunk hold 丢掉大部分 scheduled updates；RTC、phase
  alignment 和 absolute target 消费路径保持不变。

---

### Evaluation 也不要直接比较 relative 数值

validation 时：

1. network 输出 relative chunk；
2. inverse-transform 回实际 future KP；
3. 再算你现在的：

$$
\text{position error / rotation error}
$$

这样才能和你当前 **HAT 1–2 cm** baseline 公平比较。

---

### 最后你给 Codex 的任务可以概括成

> **新增 `relative_chunk` action representation，不修改或重新生成任何已有 dataset。Dataset load 后根据 current KP 在线把 future KP 转成相对于当前 pose 的 relative chunk；ACT architecture 和现有 HAT loss 保持不变；为新 representation 单独统计 normalization；inference 时恢复到 absolute/scheduled KP；保留原模式供 ablation。部署端增加 scheduled-anchor，而不是每个新 chunk 都重新 anchor 到 robot actual pose。**

我建议**第一版就只做这些**。不要同时改 loss、视觉、DOF input、网络结构，否则最后不知道到底是哪一项起作用。
