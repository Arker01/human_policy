# HuMI relative chunk（已实现）

这是一个可选 action representation，不修改原 HDF5，不改 ACT 结构和 loss。
训练时启用：

```bash
--action_representation relative_chunk
```

也可写进 model yaml：

```yaml
data:
  action_representation: relative_chunk
```

不配置时保持 `original`，用于复现旧实验。

## 表示定义

现有数据照旧存：

$$
K_t,K_{t+1},...,K_{t+H}
$$

读取 sample 后，dataloader 在内存中构造新的 action label：

$$
\boxed{
A_{t+i}=T_t^{-1}T_{t+i},\quad i=1...H
}
$$

整个 future chunk 都相对同一个当前时刻 $t$，不做相邻帧差分。

position 在 dataset 坐标系中直接相减：

$$
\Delta p_{t+i}=p_{t+i}-p_t
$$

rotation 使用：

$$
\Delta R_{t+i}=R_t^{-1}R_{t+i}
$$

两侧手指 keypoint 块也使用 `future - current`。5 个 pose 块是 head、
left/right wrist、neck 和 waist。`state/action[98:128]` 是机器人构型尾部，
不是 KP pose，保持原来的绝对值，以保持 128 维接口。

## 训练数据路径

1. 原始 dataset 不重新生成，也不覆盖。
2. 输入仍是 image + current KP，不增加 DOF-pos，不删 KP。
3. 输出 shape 仍是 `[B, H, 128]`。
4. loss 仍是现有 L1 / wrist weighting，没有 cumulative loss 或 Flow Matching。
5. `dataset_stats.pkl` 使用转换后的所有有效 future step 重新统计
   `action_mean/action_std`，并写入 `action_representation=relative_chunk`元数据。

## Evaluation 和部署

网络输出会先反归一化，再通过 anchor 恢复绝对 KP：

$$
\hat T_{t+i}=T_{\rm anchor}\hat A_{t+i}
$$

$$
\hat p_{t+i}=p_{\rm anchor}+\Delta\hat p_{t+i}
$$

inverse transform 已接入 `data/eval_mpjpe_batch.py`、
`scripts/eval/perturb_eval.py` 和 `hdt/online_hat_service.py`。HuMI 的 query 0 是
`t+1`，评测会与 GT `t+1` 对齐后再算绝对坐标系下的 MPJPE。

部署端第一个 chunk 用 robot actual pose 启动，之后用上一个 chunk 在
replan 时刻的 scheduled pose 做 anchor，避免 robot 短时 lag 持续改写计划坐标系。

实际选择不是固定取 `replan_frames - 1`：服务根据相邻 RobotState 的
`timestamp_ns` 从上一条 absolute chunk 选择对应 scheduled row。首次、opening
坐标尚未锁定、时间倒退或超过上一 chunk horizon 时使用 actual pose 重新建立
计划链。

可选安全阈值：

```bash
--scheduled-anchor-max-error-m 0.10
```

它比较 scheduled 与 actual 的 head、左右腕、waist 位置，最大误差超限时本轮
立即 actual re-anchor。默认 `0`（阈值关闭，等待真机标定），但 anchor 来源、
scheduled row 和逐部位误差始终记录在 `inference_inputs.jsonl`。

chunk 协议携带 `action_representation`。ScaleBFM 收到 `relative_chunk` 时默认
令 `relative_chunk_min_execution_fraction=0.0`，按 HAT replan 节拍消费，而不是
沿用 original checkpoint 的长 chunk hold；RTC 和 phase alignment 继续作用于
已经恢复出的 absolute chunk。

## 与 30fps、lag、vis 同时使用

机器人遥操 HDF5 如果明确带有 `fps > 30` attr，dataloader 默认使用与
`scripts/preprocess/resample_sonic.py` 相同的 nearest-frame 规则在线建立 30fps 时间轴。
因此 0908 使用的 50Hz `b0907_robot_coupled_all` 之后也会自动按 30fps 训练。
复现旧实验时可显式传 `--no_robot_fps_resample`。

HuMI + lag + vis 可同时开：

```bash
--action_representation relative_chunk --lag_aug --visual_aug
```

lag 只扰动模型看到的 current state，HuMI label 仍以未扰动的演示 pose 为 anchor；
vis 只改输入图像，三者不会互相改写。
