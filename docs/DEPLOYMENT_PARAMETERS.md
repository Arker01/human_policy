# 9 月 7 日以来新增参数：HAT / HuMI ↔ ScaleBFM

这份说明以 **2026-09-06 的仓库状态为基线**，整理当前本地代码中从
2026-09-07 起新增或改变行为的、需要用户配置的参数。内部协议字段、日志字段和
不会被部署命令读取的实现细节没有列入。

涉及文件：

- `human_policy/hdt/main.py`、`human_policy/hdt/online_hat_service.py`
- `human_policy/hdt/configs/models/*_hist.yaml`
- `ScaleBFM/ScaleBridge/scalebridge/config/env/motion_tracking_hat4_online.yaml`
- `ScaleBFM/ScaleBridge/scalebridge/config/simulator/real_world_inspire.yaml`

## 先用哪套配置

### original checkpoint

- HAT 的 `dataset_stats.pkl` 没有 `action_representation`，或明确为 `original`。
- ScaleBFM 保持 `min_chunk_execution_fraction: 0.66`。
- `relative_chunk_min_execution_fraction` 对这种 checkpoint 不生效。

### HuMI checkpoint

- 训练时使用 `relative_chunk`，部署时由 stats 中的
  `action_representation: relative_chunk` 自动识别，不需要给在线服务再传一个
  representation 参数。
- ScaleBFM 使用 `relative_chunk_min_execution_fraction: 0.0`，按 HAT 的 replan
  节拍接收 scheduled chunk；不要把它改回 0.66，除非在做明确的 A/B 实验。
- `--scheduled-anchor-max-error-m` 先保持 `0`（只记录误差，不自动重锚）；标定稳定后
  才考虑例如 `0.10`。

两套都建议保持 HAT `--fps 30`、`--replan-frames 10`，以及 ScaleBFM 的 RTC
配置不变。

## 1. HAT 训练参数

训练入口仍是 `hdt/main.py`，例如：

```bash
python hdt/main.py \
  --batch_size 32 --num_epochs 200 --exptid humi_r1 \
  --dataset_json_path <dataset.json> \
  --model_cfg_path hdt/configs/models/act_with_future_dino_nostate_hist.yaml \
  --action_representation relative_chunk \
  --lag_aug --visual_aug
```

### 表示、时间轴和历史观测

| 参数 | 默认值 | 用途 |
| --- | ---: | --- |
| `--action_representation {original,relative_chunk}` | 配置文件值，否则 `original` | `relative_chunk` 把每个 future action 都表示为相对同一 anchor 的位姿差；HuMI 部署依赖 stats 中写入的同名元数据。旧实验用 `original`。也可在 model YAML 的 `data.action_representation` 设置。 |
| `--robot_target_fps` | `30` | 对带有 `fps` 属性且标记为 robot 的高帧率 HDF5，按 nearest-frame 规则映射到统一逻辑时间轴。图像、state、history、label 共用这条时间轴。 |
| `--no_robot_fps_resample` | 关闭 | 复现旧训练；禁用高于 target FPS 的 robot 数据重采样。正常新训练不要使用。 |
| `model.history.enabled` | `false`（旧 YAML） | 开启真实观测历史。当前帧始终是 slot 0。 |
| `model.history.offsets` | `[]` | 以 **30 fps 逻辑帧**表示的过去帧，例如 `[5, 10]` 是约 0.17 s、0.33 s 前；必须为正数且不能重复。 |
| `model.history.token_pool` | `2` | 历史图像 token 的空间池化倍率；当前帧不池化。改动会改变模型结构，训练和部署 YAML 必须一致。 |

仓库提供了带历史观测的结构模板：
`hdt/configs/models/act_with_future_dino_nostate_hist.yaml` 和
`hdt/configs/models/act_input_vjepa_nostate_hist.yaml`。带 history 训练出的
checkpoint，在线服务必须加载同一份 YAML；不能换成无 history 的 YAML。

### `lag_aug`：模拟执行落后

这些参数只作用于训练 split 的 **输入 state**，不会改 action label；`history` 与
`lag_aug` 不能同时开启，因为历史帧没有做同样的扰动。

| 参数 | 默认值 | 用途 |
| --- | ---: | --- |
| `--lag_aug` | 关闭 | 沿演示轨迹把模型看到的 wrist state 向后取样，模拟 BFM 跟不上真人。 |
| `--lag_aug_prob` | `0.5` | 一个 sample 施加轨迹回退的概率。 |
| `--lag_aug_min_m` / `--lag_aug_max_m` | `0.02` / `0.08` m | 回退距离范围；必须满足 min ≤ max。 |
| `--lag_aug_search_frames` | `450` | 向过去搜索匹配位移的最大逻辑帧数。 |
| `--lag_aug_offset_prob` | `0.25` | 额外随机 3D wrist 偏移的概率。 |
| `--lag_aug_offset_sigma_m` | `0.02` m | 随机偏移每轴标准差。 |
| `--lag_aug_offset_clip_m` | `0.08` m | 随机偏移的范数上限；设为 `0` 表示不裁剪。 |
| `--lag_aug_finger_prob` | `0.25` | 让 fingertip block 使用不同回退量的概率，用于模拟手指与手腕不同步。 |
| `--lag_aug_finger_scale` | `2.0` | 手指回退量为 `U(0, scale) × wrist_lag`；大于 1 允许手指更慢。 |

### `visual_aug`：模拟视角/构图变化

| 参数 | 默认值 | 用途 |
| --- | ---: | --- |
| `--visual_aug` | 关闭 | 对训练图像做 RandomResizedCrop，输出分辨率不变，迫使模型依赖目标而不是固定背景位置。 |
| `--visual_aug_scale_min` / `--visual_aug_scale_max` | `0.75` / `1.0` | 保留面积比例范围。 |
| `--visual_aug_ratio_jitter` | `0.9` | 原始宽高比的抖动范围为 `(native×jitter, native/jitter)`。 |

`visual_aug` 不能和预计算固定图像网格的 `future_flow` target 同时使用。HuMI、
`lag_aug`、`visual_aug` 三者可以一起使用（但仍不能启用 history + lag_aug）。

## 2. HAT 在线服务参数

启动入口是 `python -m hdt.online_hat_service`。典型命令：

```bash
python -m hdt.online_hat_service \
  --config <model.yaml> --checkpoint <checkpoint.pt> --stats <dataset_stats.pkl> \
  --embodiment h1_inspire --image-source <module>:<get_images> \
  --fps 30 --replan-frames 10
```

| 参数 | 默认值 | 用途与建议 |
| --- | ---: | --- |
| `--fps` | 从 `common.chunk_fps` 读取 | chunk 的真实播放/采样频率，不是显示帧率。若 YAML 没有 `common.chunk_fps`，必须显式传入。应与训练频率一致，通常为 `30`。 |
| `--allow-fps-mismatch` | 关闭 | 允许 `--fps` 与 checkpoint 训练频率不同。会改变参考轨迹速度，只用于有意的速度 A/B；不要用来绕过错误配置。 |
| `--heading-frame {initial,world}` | `initial` | `initial` 将位置/旋转转到开场 root heading frame，匹配训练数据；`world` 保留 9 月 8 日前的世界朝向行为，只有复现旧模型时使用。 |
| `--state-ood-limit` | `10.0` σ | 启动前检查 policy state 的标准化值；任一维超过阈值就拒绝启动。降低可更早暴露坐标/手部映射错误；提高前应先看启动时打印的最差维度。 |
| `--scheduled-anchor-max-error-m` | `0`（关闭） | 仅对 `relative_chunk` 生效。scheduled anchor 与实际 head/双腕/waist 的最大位置误差超过阈值时，本轮改用 actual pose 重锚。推荐先保持 0，标定稳定后再试 `0.10`。 |

已有但经常需要配套确认的选项：`--replan-frames`（默认 10）、`--origin-frames`
（默认 10）、`--state-tail`、`--wrist-frame-conversion`。它们不是本次新增重点，
但必须与训练数据和 ScaleBFM 的 frame convention 一致。

`relative_chunk` 的 representation 来自 `dataset_stats.pkl`；旧 stats 没有该字段
时协议按 `original` 处理。在线服务会把 anchor 来源、scheduled row 和误差写入
`inference_inputs.jsonl`，可先用记录调阈值，不要盲调机器人动作。

## 3. ScaleBFM 在线环境参数

配置文件：`ScaleBridge/scalebridge/config/env/motion_tracking_hat4_online.yaml`。

### 启动与 chunk 消费

| 参数 | 当前配置 | 用途 |
| --- | ---: | --- |
| `prestart_arm_pose` | `stand` | R1 前的 FK hold 手型；可选 `raised`、`stand`、`sonic_0904`。`sonic_0904` 是 9 月 4 日 SONIC 开场全身姿态。它不消费 live HAT chunk。 |
| `startup_hand_timeout_s` | `8.0` s | 首个 HAT state 前等待双手反馈为 physically open 的最长时间。超时表示启动链路有问题，不应直接放宽。 |
| `min_chunk_execution_fraction` | `0.66` | `original` chunk 至少执行的长度比例；100 帧、30 Hz 时约 2.2 s，用于防止普通 checkpoint 反复播放 chunk 开头。设 0 才是每次 replan 都可换。 |
| `relative_chunk_min_execution_fraction` | `0.0` | `relative_chunk` 专用执行下限。HuMI 使用 0，让每次 scheduled replan 都能被消费；只有 A/B 才设成 0.66。 |
| `chunk_end_margin_s` | `0.5` s | 当前 chunk 剩余这么多时间时提前释放 hold，给下一个 chunk 留出接收和 blend 时间。 |

### RTC（Receding Trajectory Correction）

| 参数 | 当前配置 | 用途 |
| --- | ---: | --- |
| `rtc_enabled` | `true` | 在换 chunk 时把旧 chunk 尚未执行的 future 与新 chunk 的实际播放前缀对齐混合。设 `false` 可复现旧的 hold + `chunk_blend_s` 路径。 |
| `rtc_prefix_frames` | `10` | RTC 处理的前缀总帧数。 |
| `rtc_hard_prefix_frames` | `2` | 前几帧 100% 保留旧 chunk，避免瞬时跳变。 |
| `rtc_blend_frames` | `8` | 旧 → 新的渐变帧数。必须满足 `hard + blend == prefix`。 |

### 接缝和手指实验开关

以下三组是 9 月 9 日 grasp failure 实验开关，当前默认保持旧行为：

| 参数 | 当前配置 | 用途 |
| --- | ---: | --- |
| `chunk_start_reference_continuity` | `false` | 新 chunk 在 latency frame 之后向前搜索与 outgoing reference 最连续的起点。要求 `chunk_phase_alignment: true`；开启后牺牲严格 source-timestamp clock 换接缝连续性。 |
| `continuity_search_ahead_s` | `1.0` s | 上一项的 forward search 窗口；只在 continuity 开启时生效。 |
| `finger_wrist_gate` | `false` | wrist 与目标相差较大时冻结手指 closing，opening 始终放行，用于避免手腕滞后时提前抓取。 |
| `finger_wrist_gate_engage_m` | `0.06` m | 误差超过 6 cm 进入 gate。 |
| `finger_wrist_gate_release_m` | `0.035` m | 误差回到 3.5 cm 以下解除 gate；必须满足 `0 < release ≤ engage`。 |
| `finger_chunk_source` | `playback` | `playback` 让手指跟随 body 的 accepted chunk、hold、RTC；`latest` 让手指直接跟随最新收到的 chunk，约每次 replan 更新，但更容易提前闭合。 |

建议一次只打开一个实验开关；`finger_chunk_source: latest` 若确实需要，通常与
`finger_wrist_gate: true` 一起做小范围测试。

## 4. 真机 simulator 参数

配置文件：`ScaleBridge/scalebridge/config/simulator/real_world_inspire.yaml`。

| 参数 | 当前配置 | 用途 |
| --- | ---: | --- |
| `state_watchdog_hold_last_target` | `True` | Robot state/NIC 暂时冻结时保持最后一个全身目标，而不是掉到 damping；用于保留 SONIC 风格 hold。 |
| `hand_ik_mode` | `coupled_motor` | `coupled_motor` 约束真实 Inspire 六个物理执行器；`independent_joint` 保留旧的 12-DOF IK，仅用于 A/B。 |

这两个参数位于 simulator，不是 `env.config`；命令行覆盖要写成：

```bash
simulator.config.state_watchdog_hold_last_target=true \
simulator.config.hand_ik_mode=coupled_motor
```

## 5. 参数组合和排错顺序

1. 先确认 HAT 启动日志的 `chunk fps`、`heading frame`、state OOD 最差维度；FPS
   不匹配时先修 YAML/命令，不要直接加 `--allow-fps-mismatch`。
2. original 只看 `min_chunk_execution_fraction`；HuMI 只看
   `relative_chunk_min_execution_fraction`，两者不要互相替代。
3. 如果 HuMI 的 scheduled anchor 误差持续升高，先检查 opening origin、heading、
   wrist frame conversion 和 RobotState timestamp，再考虑设置
   `--scheduled-anchor-max-error-m`。
4. 如果出现 chunk 接缝跳变，先保持 RTC 开启；确认 `10 = 2 + 8` 后，再单独试
   `chunk_start_reference_continuity`。
5. 如果是提前闭手而不是 body 接缝问题，只试 `finger_wrist_gate` 或
   `finger_chunk_source`，不要同时修改 chunk hold、RTC 和 phase alignment。
