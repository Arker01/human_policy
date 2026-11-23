# G1 Dex3 三任务：优化视觉数据重新训练

2026-11-05 16:44:42 UTC 启动三个独立训练。参考实验为
`../unitree_g1_dex3_hat_20261031/`。本次使用
`/mnt/nvme0n1/lirui_backup/public_data/` 下的 `episodes_g1_fov125` 数据。

| 任务 | 数据子目录 | GPU | 训练 PID | Train / Val episodes | 日志 |
| --- | --- | ---: | ---: | ---: | --- |
| block_stacking | block | 2 | 2219680 | 271 / 30 | [block_stacking.log](logs/block_stacking.log) |
| pick_apple | pick_apple | 3 | 2219681 | 181 / 20 | [pick_apple.log](logs/pick_apple.log) |
| pouring | pour | 4 | 2219682 | 280 / 31 | [pouring.log](logs/pouring.log) |

训练配置沿用上次：V-JEPA 2.1 ViT-B，128 维 state/action，单 `top` 相机，
batch 64，100000 optimizer steps，lr 1e-5，chunk 100，seed 0，30 fps，
`action_representation=original`。保留原 lag/visual augmentation 及默认参数，
Future-DINO head 保留、weight 0、horizon 45，禁用语言条件，`cond_mask_prob=0`。
从相同预训练视觉编码器及新策略初始化。

训练源码逐文件 SHA256 与上次 manifest 一致，包含原 validation 隔离修复。
旧实验的模型 YAML 后来被注释编辑过，本次从原 baseline YAML 恢复，
SHA256 与上次 manifest 记录完全一致。逐项还原入口及配置/输出路径后，
完整训练命令与对应旧任务命令完全一致。证据保存于 [manifest.json](manifest.json)。

## 双手输入与监督

按用户要求保留双手，尤其是左手操作的 pick_apple、block_stacking。
`OUTPUT_LEFT_EEF=80:89`、`OUTPUT_LEFT_KEYPOINTS=10:28` 全部保持输入与 loss。
唯一的 state 置零范围仍为 `98:128`，该范围是机器人配置尾部，与左手无交集。
原 ACT loss 的 neck `58:67` mask 保留。

核查上次 manifest 对应源码后，没有发现整块左手 mask，因此没有额外更改 loss。
本次直接执行源码快照中的 ACT loss，已确认左手 EEF 和关键点梯度均为正、
state_keep_mask 的全部左手维度为 1，输入左手值原样传入模型。
见 [left_hand_verification.json](left_hand_verification.json) 和
[check_left_hand.py](check_left_hand.py)。

## 数据检查

共 813 episodes、555352 frames。所有文件的 state/action 均与上次原数据逐值完全
一致，且为有限 float32；每个任务的 train/val episode 成员及顺序完整保留，互不重叠。
已检查全部轨迹的图像别名、视觉处理元数据、文件大小及首/中/末帧 JPEG 解码，
并在每个任务抽取完整 episode 校验 processing report 的 SHA256。
见 [data_preflight.json](data_preflight.json)。

## 输出与查看状态

Checkpoint 根目录：
`/mnt/nvme0n1/human_policy_ckpt/unitree_g1_dex3_hat_20261105_fov125/`。
本目录 [checkpoints](checkpoints) 是指向该根目录的软链接，任务各存入
`block_stacking_ckpt`、`pick_apple_ckpt`、`pouring_ckpt`。
保留每 10000 步保存完整恢复状态及结束时保存 `policy_last.ckpt` 的机制。

```bash
cd /home/aigc/human_policy
/home/aigc/miniconda/envs/human_policy/bin/python \
  training_experiments/unitree_g1_dex3_hat_20261105_fov125/manage.py status
```

`status/<task>.json` 记录 supervisor、训练 PID、退出状态；
`status_summary.json` 保存最近一次查询。三个 supervisor 均独立于启动终端运行。
启动器拒绝重复启动、覆盖已有训练输出或占用其他训练的 GPU。

启动检查通过：三组均已完成首轮验证和超过 100 步训练，loss 全部有限，
无异常堆栈。实际进程命令与 manifest 一致，实际归一化统计与上次完全一致。
见 [startup_verification.json](startup_verification.json) 和
[runtime_configuration_verification.json](runtime_configuration_verification.json)。

沿用的 CSV writer 在 train 行新增列时未改写最初的 val-only 表头；读取原始
`metrics.csv` 时需按后续行的排序列重建 train/lr 字段。启动检查已正确解析，
没有为此改变训练源码或参数。
