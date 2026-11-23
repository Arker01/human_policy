# G1 Dex3 视觉优化重训：补充核查记录

本次实际运行位于 [../unitree_g1_dex3_hat_20261105_fov125/](../unitree_g1_dex3_hat_20261105_fov125/README.md)。
独立会话已于 2026-11-05 16:44:42 UTC 启动用户要求的三个训练：

| 任务 | GPU | 训练 PID |
| --- | ---: | ---: |
| block_stacking | 2 | 2219680 |
| pick_apple | 3 | 2219681 |
| pouring | 4 | 2219682 |

本会话准备期间发现上述运行，已核实其全部训练命令、数据配置、源码及模型
SHA256 与本次要求和 20261031 原记录一致，归一化统计与旧训练逐值一致。
GPU 检查阻止了重复启动；本目录未启动训练，也未创建额外 checkpoint 目录。
`manifest.json` 已标记 `using_existing_runs`，`manage.py launch` 会拒绝重复启动，
`manage.py status` 转到实际运行查询。

补充检查结果：

- [configs/data_audit.json](configs/data_audit.json)：全部 813 个新 HDF5 的 SHA256
  与视觉处理报告一致，所有动作/状态与旧数据逐值一致，train/val 成员及顺序一致，
  每个 episode 首/中/末图像可解码且已更新，共检查 2439 张图像。
- [configs/left_hand_audit.json](configs/left_hand_audit.json)：执行原源码的模型 forward，
  左手 state 原值进入 encoder 和 decoder；执行原 ACT loss 的 backward，
  左腕 `80:89` 和左手指 `10:28` 的每个动作维度均有正梯度。
  state 仅屏蔽 `98:128` 构型尾部，原 neck loss mask 保留。
- [startup_verification.json](startup_verification.json)：三组实际进程命令核对通过，
  均超过 200 步且首轮验证完成，指标全部有限。

实际权重目录：`/mnt/nvme0n1/human_policy_ckpt/unitree_g1_dex3_hat_20261105_fov125/`。
模型从相同视觉预训练权重和新策略初始化，保持 batch 64、100000 步、lr 1e-5、
chunk 100、seed 0，以及全部原 augmentation 和 Future-DINO 参数。

当前实际训练状态：

```bash
/home/aigc/miniconda/envs/human_policy/bin/python \
  /home/aigc/human_policy/training_experiments/unitree_g1_dex3_hat_20261105_fov125/manage.py status
```
