# Unitree G1 Dex3 三任务标准 HAT 训练

用户已选择沿用标准 HAT 配置。参考记录为最近完成的标准 HAT run：
`b0930_20261004_robotonly_nostate_vjepa_w0_lagvis`，来源
`experiments/bottle0930_16runs_20261004/jobs.json` 及对应训练日志。
`manifest.json` 保存完整原记录、启动参数、权重及源码 SHA256。

三组于 **2026-10-31 16:47（Asia/Shanghai）** 启动。

| 任务 | GPU | 训练 PID | 训练 / 验证 episodes | 日志 |
| --- | ---: | ---: | ---: | --- |
| 堆积木 | 1 | 1519958 | 271 / 30 | [block_stacking.log](logs/block_stacking.log) |
| 抓苹果 | 2 | 1519959 | 181 / 20 | [pick_apple.log](logs/pick_apple.log) |
| 倒水 | 3 | 1519960 | 280 / 31 | [pouring.log](logs/pouring.log) |

## 配置

- V-JEPA 2.1 ViT-B，128 维 state/action，`camera_names: [top]`。
- batch 64，100000 optimizer steps，学习率 1e-5，chunk size 100，seed 0。
- `action_representation=original`，30fps，归一化后屏蔽 state 的 98:128 维。
- 保留相同 lag augmentation 和 visual augmentation，包括原来的默认参数。
- Future-DINO 模块保留、horizon 45、weight 0.0，与参考记录一致。
- 禁用语言条件，`cond_mask_prob=0.0`；初始化方式与参考相同：预训练视觉编码器＋新策略训练。
- 模型 YAML 原样复制至 [configs/model.yaml](configs/model.yaml)；各任务的 train/val JSON 单独快照。
- 完整训练参数已逐项与参考记录比对；任务数据路径、模型配置快照路径、输出路径和入口源码快照路径按本次实验替换。

本轮保留现有 ACT loss，没有加入前面讨论的三指 loss mask 改动。
独立源码快照位于 `source/hdt/`；模型训练不依赖后续对工作目录源码的编辑。

## 训练循环修复

启动前发现原 `hdt/main.py` 的 validation 循环覆盖了外层训练变量 `data`，
导致执行 validation 后的第一次训练更新会误用最后一个验证 batch。
本轮已将 validation 的局部变量改为 `val_data`，不改变模型配置或训练超参数。
补丁见 [training_loop_fix.patch](training_loop_fix.patch)。
回归测试 `tests/test_training_validation_isolation.py` 已通过，验证训练只使用
训练 batch、validation 只使用验证 batch。修复已包含在本次源码快照中。

## 输出与状态

Checkpoint 实际存储在：
`/mnt/nvme0n1/human_policy_ckpt/unitree_g1_dex3_hat_20261031/`。
本目录 [checkpoints](checkpoints) 为指向该目录的软链接。

每个任务的 `*_ckpt/` 独立保存 `dataset_stats.pkl`、`metrics.csv`、每 10000 步
的完整恢复状态及结束时的 `policy_last.ckpt`。全部统计使用新任务的 `dex3` 数据。

```bash
cd /home/aigc/human_policy
/home/aigc/miniconda/envs/human_policy/bin/python \
  training_experiments/unitree_g1_dex3_hat_20261031/manage.py status
```

`status/<task>.json` 保存 supervisor/training PID 和最终退出状态；
`status_summary.json` 为最后一次查询快照。
训练进程已脱离启动终端，启动工具结束后继续运行。
启动器会拒绝重复启动、覆盖旧输出或占用其他计算进程使用的 GPU。

初始检查记录见 [startup_verification.json](startup_verification.json)：
三组已通过模型权重加载、单头部相机配置、数据划分、Dex3 归一化统计、
首轮验证和超过 100 步的实际训练检查，无异常堆栈或非有限指标。
