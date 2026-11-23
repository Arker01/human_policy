# 拿起三脚架并立在桌上：G1 Dex3 HAT

原始 ZIP 已完整解压到同目录：
`/mnt/nvme0n1/lirui_backup/public_data/pick_up_a_tripod_and_stand_it_on_the_desk/`。
解压时读取并校验 ZIP 条目 CRC，保留全部 RGB、depth、lidar、JSON 和演示视频。

转换数据：
`/mnt/nvme0n1/lirui_backup/public_data/pick_up_a_tripod_and_stand_it_on_the_desk_hat/`。
共 40 个 HDF5、24528 帧；固定 seed 42 按 episode 划分 train 36 / val 4，无重叠。
`data/g1_dex3_tripod` 是上述目录的软链接。数据清单见
[configs/tripod.json](configs/tripod.json)，完整转换记录位于数据目录的
`conversion_report.json`，每个输出记录 SHA256。

## 输入、动作和双手

- `embodiment=dex3`，`fps=30.0`。
- `observation.image.top` 硬链接到 `observation.image.head`，来自原始
  `color/frame_*.jpg` 头部 RGB。保留完整画面，640×480 缩放至 320×240。
  原始数据没有相机内参，因此未套用另一批数据的 FOV125 重映射。
- JPEG 与已有 Dex3 数据使用同一颜色约定：训练器直接 `cv2.imdecode` 得到 RGB。
  当前图像和 Future-DINO 目标都使用 `camera_names: [top]`。
- `observation.state` 和 `action` 均为 `(T,128)` float32；前者使用测量关节，
  后者使用同帧 `sol_q + left_angles + right_angles`，不额外移帧，不离线归一化。
- 与最近双手 Dex3 训练保持同一 URDF FK、手腕坐标、名义头部参考点。
  额外的腿部、IMU、里程计和遥操头/手姿态保存在 `source.*`，不混入旧机器人
  的 `98:128` 构型定义。头/腰采用原 Dex3 转换的固定参考约定。
- Dex3 的拇指、食指、中指均保留；无名指和小指为零占位。
  左手 DDS 电机顺序为 thumb/middle/index，右手为 thumb/index/middle；
  可核对 [Unitree 控制器枚举](https://github.com/unitreerobotics/xr_teleoperate/blob/main/teleop/robot_control/robot_hand_unitree.py)。
- 左右手输入和动作监督均保留。唯一 state 置零范围为 `98:128`；原 neck loss
  `58:67` mask 保留。没有沿用其他行走实验的左腕 mask 或消融 loss。

## 训练配置和验证

复用最近双手 Dex3 实验 `../unitree_g1_dex3_hat_20261105_fov125/` 的源码快照和
模型 YAML；两者 SHA256 全部匹配。训练命令仅替换入口快照、数据和输出路径，
参数逐项还原后与上次相同，见 [manifest.json](manifest.json)。

V-JEPA 2.1 ViT-B，ACT 128 维，batch 64，100000 optimizer steps，lr 1e-5，
chunk 100，seed 0，30 Hz，original action；lag/visual augmentation 保留，
Future-DINO weight 0、horizon 45；语言关闭、cond_mask_prob 0；
视觉编码器使用原预训练权重，策略从头初始化。

[preflight_report.json](preflight_report.json) 检查通过：

- 全量 FK、双手变化、来源数组、相机别名、帧数和旋转有效性。
- 真实 8-worker train/val batch64 加载、训练集归一化和所有 episode 的末尾 padding。
- 真实加载器的图像张量与头部 RGB 解码一致。
- 完整 V-JEPA ACT 和 DINO teacher 的 CPU 前向/反向，319 个参数梯度有限。
- 双手状态在 CVAE encoder 和 decoder 中保持；双手每个动作维度均有非零 loss 梯度。
- 现有独立 URDF 几何和异常遥测测试：6 passed。

## 排队与输出

GPU 忙时，独立 supervisor 每 20 秒检查一次；连续两次检查空闲且无 compute
进程的 GPU 才启动，不中断已有任务。启动前再次校验源码、配置和全部 HDF5 哈希；
文件锁防止重复运行，已有输出会拒绝覆盖。

```bash
/home/aigc/miniconda/envs/human_policy/bin/python \
  training_experiments/g1_dex3_tripod_hat_20261109/manage.py status
```

状态：[status.json](status.json)。排队日志：`logs/queue.log`。
训练开始后日志：`logs/train.log`。
权重目录：`/mnt/nvme0n1/human_policy_ckpt/g1_dex3_tripod_hat_20261109/tripod_ckpt/`。

可复用的转换器：`scripts/preprocess/convert_g1_raw_dex3.py`。再次运行会核验已有
输出；`--verify-only` 仅核验 HDF5 并更新清单/报告。
