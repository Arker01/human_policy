# HAT / HuMI + ScaleBFM 真机启动（四个终端）

旧说明：`/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge/docs/hat_real_stage4.md`。
以下按本机历史启动命令及当前代码整理，四个终端均在 nerv 工作站运行。
前置：SteamVR 已运行，腰部和地面两个 Vive tracker 在线；机器人端 ZED sender
已向 `tcp://192.168.123.164:5555` 发送图像。

## 1. Vive sender（scalebridge）

```bash
source /home/nerv/miniconda3/etc/profile.d/conda.sh
conda activate scalebridge
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge/scalebridge/utils/vive_sender
python tcp_send_raw_tracker_array.py \
  127.0.0.1 tracker_1 tracker_2 \
  --port 5000 --hz 250 --status-every 250
```

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge/scalebridge/utils/vive_sender

/home/nerv/miniconda3/envs/scalebridge/bin/python tcp_send_raw_tracker_array.py \
  127.0.0.1 tracker_1 tracker_2 \
  --port 5000 --hz 250 --status-every 250
```


当前真机在线配置按高度自动识别地面/腰部，不依赖 tracker_1、tracker_2 的排列。
sender 等第 4 个终端的接收端启动后建立连接。

## 2. ScaleBFM 底层控制器（C++，无需 conda）

```bash
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge/third_party/unitree_sdk2/build/bin
./g1_29dof_controller enp4s0
```

## 3. HAT / HuMI 在线推理（twist_qyx）

用 `HAT_BUNDLE` 和 `HAT_CONFIG` 选择权重，checkpoint、stats、YAML 必须属于
同一次训练。下面列了几组可选赋值；若全部执行，最后一组赋值生效。

```bash
source /home/nerv/miniconda3/etc/profile.d/conda.sh
conda activate twist_qyx
cd /home/nerv/qingyaoxu/human_policy
export PYTHONPATH=/home/nerv/qingyaoxu/human_policy:/home/nerv/qingyaoxu/human_policy/hdt:/home/nerv/qingyaoxu/human_policy/hdt/detr:/home/nerv/qingyaoxu/ScaleBFM/ScaleBridge

HAT_BUNDLE=pick_bottle_0930/b0930_20261003_oldhuman_vjepa_w0_ckpt
HAT_CONFIG=act_input_vjepa_w0.yaml

HAT_BUNDLE=pick_bottle_0930/b0930_20261003_oldhuman_humi_ckpt
HAT_CONFIG=act_input_humi.yaml

HAT_BUNDLE=pick_bottle_0930/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt
HAT_CONFIG=act_input_vjepa_w0_nostate_lagvis.yaml

HAT_BUNDLE=pick_bottle_0930/b0930_20261004_oldhuman_nostate_humi_ckpt
HAT_CONFIG=act_input_humi_nostate.yaml

HAT_BUNDLE=pick_bottle_0930/b0930_20261004_oldhuman_nostate_humi_ckpt
HAT_CONFIG=act_input_humi_nostate.yaml

HAT_BUNDLE=pick_bottle_0930/b0930_20261004_replace50_nostate_humi_lagvis_ckpt
HAT_CONFIG=act_input_humi_nostate_lagvis.yaml

/home/nerv/miniconda3/envs/twist_qyx/bin/python -m hdt.online_hat_service \
  --config "$HAT_BUNDLE/$HAT_CONFIG" \
  --checkpoint "$HAT_BUNDLE/policy_last.ckpt" \
  --stats "$HAT_BUNDLE/dataset_stats.pkl" \
  --embodiment dex5 \
  --image-source hdt.zed2i_image_source:get_images \
  --state-tail full \
  --fps 30 --replan-frames 10 \
  --wrist-frame-conversion inspire_hand_base \
  --heading-frame initial \
  --scheduled-anchor-max-error-m 0 \
  --relative-anchor-mode actual
```

常用替换（改上面两个变量，再执行同一条 python 命令）：

| 模型 | HAT_BUNDLE | HAT_CONFIG |
|---|---|---|
| 旧 pillow HAT | `BEST_CKPT` | `act_with_future_dino.yaml` |
| 旧 pillow nostate | `BEST_CKPT_ab4_nostate` | `act_with_future_dino_nostate.yaml` |
| 新 bottle HuMI | `pick_bottle_0930/b0930_20261003_oldhuman_humi_ckpt` | `act_input_humi.yaml` |
| 新 bottle HuMI + nostate + lag/vis | `pick_bottle_0930/b0930_20261004_oldhuman_nostate_humi_lagvis_ckpt` | `act_input_humi_nostate_lagvis.yaml` |

其他 bottle 权重与推理 YAML 对照见 [pick_bottle_0930/README.md](pick_bottle_0930/README.md)。

- **HuMI**：从 stats 自动识别 `relative_chunk`，在线命令不用加 `--action_representation`。
- **nostate**：配套 YAML 的 `model.zero_state_dims: "98:128"` 自动屏蔽状态尾部；保留 `--state-tail full`。
- **lag / vis**：训练增强，在线推理不用加 `--lag_aug`、`--visual_aug`；使用对应训练的权重即可。
- **history**：自动读取配套 YAML 的历史配置，启动入口相同。

HAT 加载完成后等第 4 个终端发送机器人状态，尚无 chunk 输出是正常的。

## 4. ScaleBFM 模型与真机跟踪（scalebridge）

以下两份均为 4090-168 上对应训练组截至 2026-10-11 最新保存的
`model_14000.pt`，已在本机 4080 SUPER 导出 TensorRT；各目录也保留原始 checkpoint。
第 4 个终端选择其中一条执行。新旧 focus 指训练标签版本，两条命令沿用当前在线
focus 与手部交接逻辑；参考变速和精细跟踪奖励属于训练机制，在线无需附加开关。

### 新 focus＋新机制（2026-10-11，14,000 轮）

```bash
source /home/nerv/miniconda3/etc/profile.d/conda.sh
conda activate scalebridge
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_v3_new_settings_20261011/model_14000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false \
  'simulator.config.hand_thumb_rot_invert_sides=[left,right]' \
  'simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]' \
  env.config.min_chunk_execution_fraction=0.1 \
  env.config.chunk_start_reference_continuity=True \
  env.config.focus_handoff_enabled=true \
  env.config.focus_handoff_finger_mode=continuity/hold_progress
```

### 旧 focus＋新机制（2026-10-11，14,000 轮）

```bash
source /home/nerv/miniconda3/etc/profile.d/conda.sh
conda activate scalebridge
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_v2_new_settings_20261011/model_14000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false \
  env.config.min_chunk_execution_fraction=0.1 \
  env.config.chunk_start_reference_continuity=True \
  'simulator.config.hand_thumb_rot_invert_sides=[left,right]' \
  'simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]' \
  env.config.focus_handoff_enabled=true \
  env.config.focus_handoff_finger_mode=continuity/hold_progress
```

### 原有七月底训练权重（20,000 轮）

```bash
source /home/nerv/miniconda3/etc/profile.d/conda.sh
conda activate scalebridge
cd /home/nerv/qingyaoxu/ScaleBFM/ScaleBridge
/home/nerv/miniconda3/envs/scalebridge/bin/python scalebridge/run.py \
  --config-name=hat4_online \
  simulator=real_world_inspire_online \
  agent.config.checkpoint=/home/nerv/qingyaoxu/ScaleBFM/MyScaleBFM/ScaleTrack/releases/hat4_focus_release_v1/model_20000_scalebridge_tensorrt_4080s.pt \
  env.config.reference_forcing=false \
  'simulator.config.hand_thumb_rot_invert_sides=[left,right]' \
  'simulator.config.hand_fk_thumb_rot_invert_sides=[left,right]' \
  env.config.focus_handoff_enabled=true \
  env.config.focus_handoff_finger_mode=continuity/hold_progress

```

当前配置已设置：original 的 `min_chunk_execution_fraction=0.66`，
HuMI 的 `relative_chunk_min_execution_fraction=0.0`，会按动作表示自动选用；
换 HAT / HuMI 权重时，第 4 个终端命令相同。

操作：第一次 **R2** 标定 Vive，第二次 **R2** 建立底层通信；看到
`Accepted chunk`、`READY` 后再按 **R1** 开始闭环。

## 5. 在线部署参数说明

这里整理此前关于腕部补偿、闭手门控、chunk hold、RTC 和过渡混合的参数答复，
并按当前代码补充 `actual` 锚点模式。表中的 ScaleBFM 默认值来自
`scalebridge/config/env/motion_tracking_hat4_online.yaml`；命令行覆盖优先于 YAML。

第 3 个终端的 HAT 参数用 `--参数 值`；第 4 个终端的环境参数用
`env.config.参数=值`；真机后端参数用 `simulator.config.参数=值`。

### 5.1 HAT 坐标系、预测频率和解码锚点（第 3 个终端）

| 参数 | 默认值 / 本文启动值 | 作用 |
|---|---|---|
| `--fps` | 读取 YAML 的 `common.chunk_fps`；本文显式设 `30` | 预测轨迹的播放频率，应匹配训练采样频率；不是摄像头显示帧率。YAML 没有该字段时必须显式传入。 |
| `--replan-frames` | `10` | 每隔这么多帧重新预测。30 Hz、10 帧对应约 0.333 秒一次，不代表机器人每次必须播放完整 chunk。 |
| `--allow-fps-mismatch` | 关闭 | 允许显式播放频率与 YAML 不同，会改变轨迹速度。 |
| `--origin-frames` | `10` | 用开场若干状态平均头部世界位置，建立位置原点；设 `0` 不减头部原点。 |
| `--heading-frame` | `initial` | 将位置和旋转转到开场 root 的朝向坐标系，匹配训练约定；`world` 保留世界朝向。`actual` 模式下仍然生效，本文保留 `initial`。 |
| `--wrist-frame-conversion` | 默认 `none`；本文 `inspire_hand_base` | 输入将 G1 `wrist_yaw_link` 转到训练用的 Inspire 手基座，输出再逆变换回 G1 手腕。包含位置偏移和旋转，`actual` 模式下仍然生效。 |
| `--state-tail` | `full` | 填充状态尾部；`qpos-only` 清空尾部 root 位姿、保留关节，`zero` 清空整个尾部。nostate 权重由配套 YAML 再屏蔽 `98:128`，本文仍传 `full`。 |
| `--state-ood-limit` | `10.0` σ | 启动时检查状态标准化值，超限拒绝启动；后续超限只打印告警。 |
| `--relative-anchor-mode` | 默认 `scheduled`；本文 `actual` | 仅对 HuMI `relative_chunk` 生效。`scheduled` 延续 HAT 自己的上一条预测计划；`actual` 每轮以本轮输入观测对应的实测位置和姿态恢复预测，不继承旧计划误差。 |
| `--scheduled-anchor-max-error-m` | `0` | 仅在 `relative_chunk` + `scheduled` 时生效。设正值后，计划与实测 head/双腕/waist 的最大位置误差超过该值时改用实测锚点；`0` 不启用距离阈值。在 `actual` 模式下此参数不起作用，可以保留 `0`。 |

`actual` 的锚点在推理前从本轮 `policy_state_raw` 复制，和送入模型的当前观测
使用同一条 RobotState、同一个位置原点和初始朝向。恢复动作时不会取预测返回后的
最新实测位姿；模型推理期间机器人移动，也不会改变已经保存的锚点。

只需在 HAT 命令中切换：

```bash
--heading-frame initial \
--scheduled-anchor-max-error-m 0 \
--relative-anchor-mode actual
```

要复现原来的计划接续方式，将最后一项改成 `--relative-anchor-mode scheduled`。

新增 `--relative-anchor-mode real`：先按本轮输入观测解码，再在推理完成时刻
分别插值新旧轨迹，对每个发布的位姿做
`T_old(t_a) @ inverse(T_new(t_a)) @ T_new(t)` 对齐。它继续上一条计划，
并消掉新轨迹接缝处的常数偏移；不是每轮退回实测的 `actual` 模式。
当前模型第 0 行是 `t+1`，插值遵循这一时间定义，超出轨迹区间时保持端点。
手指点仍在腕部局部坐标中，用接缝差分接续，不随世界位姿变换旋转。

训练统计现已取消 `std >= 0.01` 的下限，恒定通道 std 保留为 0；归一化除法
仍使用数值 epsilon。现有八个 HuMI checkpoint 目录中另有
`dataset_stats_true_std.pkl`，使用 69 条 dex5 训练集重算的完整 128 维 action std，
保留原 mean、输入 qpos 统计及其它 embodiment。两项同时启用需将启动参数改为：

```bash
--stats <checkpoint目录>/dataset_stats_true_std.pkl \
--relative-anchor-mode real
```

原 `dataset_stats.pkl` 保留以便对照。旧 checkpoint 使用新 action 尺度属于部署
消融，不等于按新统计重训；`real` 也不保证消除模型预测的错误运动趋势。

`--relative-anchor-mode real_wrist` 保留 `real` 的双手腕和手指接续；head 与
waist/root 的全部位置、姿态改为本轮输入 RobotState 的实测值，并在整条 chunk
内保持该实测姿态。head 来自 ScaleBridge 的当前关节 FK，root 来自同一帧的
实测根状态。它不叠加模型对躯干的相对预测；这些实测躯干字段也写入内部计划
与推理记录，避免发送值和下一轮沿用的计划不一致。
此处 `scheduled` 指本地现有实现，不表示已复现公开 HuMI 的完整 SE(3) 接缝归零。

### 5.2 腕部落点补偿（第 4 个终端）

| `env.config` 参数 | 当前默认值 | 作用 |
|---|---|---|
| `wrist_compensation` | `none` | `none` 不补偿；`offset` 给左右腕位置目标加补偿。在线只支持这两种，不支持旧的 `root_override`。不改变 root/torso 目标，也不直接修改腕部旋转目标。 |
| `wrist_compensation_gain` | `0.1` | 大于 0 时，在 FOCUS 内每个控制步累加 `gain × (原始腕目标 − 实测腕位置)`；左右手分别积分。设 `0` 则使用限幅后的 root 位置误差作前馈补偿。 |
| `wrist_compensation_clamp_m` | `0.15` m | 每只手、每个坐标分量的补偿限幅，不是整个三维向量的范数上限。 |
| `wrist_compensation_fade_s` | `0.4` s | 进入和离开 FOCUS 时，已应用补偿淡入、淡出的时间。 |

实测腕位置来自 Vive root 与实测关节 FK。FOCUS 值达到 `0.5` 后开始积分；
短暂离开 FOCUS 时保留积分值、淡出应用量。R1 前、定位暂停或参考被冻结时清空积分。
记录写入第 4 个终端输出目录的 `wrist_compensation_trace.jsonl`。

若要启用，在第 4 个终端命令末尾加：

```bash
env.config.wrist_compensation=offset \
env.config.wrist_compensation_gain=0.1 \
env.config.wrist_compensation_clamp_m=0.15 \
env.config.wrist_compensation_fade_s=0.4
```

`actual` 只改变 HAT 解码锚点，**不会自动关闭或开启腕补偿**。若配置为 `offset`，
ScaleBFM 仍会在恢复后的腕目标上加上述补偿。

### 5.3 腕部到位后才能闭手：`finger_wrist_gate`

| `env.config` 参数 | 当前默认值 | 作用 |
|---|---|---|
| `finger_wrist_gate` | `false` | 开启后，左右手分别按腕部位置误差判断是否阻止继续闭手；张开始终允许。 |
| `finger_wrist_gate_engage_m` | `0.06` m | 实测腕与当前播放参考腕的位置误差大于 6 cm 时，进入阻挡状态。 |
| `finger_wrist_gate_release_m` | `0.035` m | 已阻挡时，误差小于 3.5 cm 才解除；要求 `0 < release ≤ engage`。 |
| `finger_chunk_source` | `playback` | `playback`：手指跟随身体当前 accepted chunk 的 hold、RTC 和过渡混合。`latest`：手指跟随最新收到的 chunk 的源时间轴，绕过身体的 chunk hold。 |

在 3.5～6 cm 之间保持原先的阻挡状态，避免反复开关。因此不是“每次闭手都必须
小于 3.5 cm”。门控比较实测 FK 腕和当前播放参考的腕位置（腕补偿前），用拇指与
食指指尖间距判断是否正在闭合。它不检查腕部旋转误差。

启用门控时，在第 4 个终端命令末尾加：

```bash
env.config.finger_wrist_gate=true \
env.config.finger_wrist_gate_engage_m=0.06 \
env.config.finger_wrist_gate_release_m=0.035
```

HAT、HuMI 的 `scheduled` 和 `actual` 模式都受此开关影响。`actual` 不是“手已到位”
的保证，预测目标与实测腕仍可能存在误差，门控仍可能阻挡闭手。

### 5.4 chunk 消费、接入时间和过渡

| `env.config` 参数 | 当前默认值 | 作用 |
|---|---|---|
| `min_chunk_execution_fraction` | `0.66` | 普通 `original` chunk 的最短执行比例。100 帧、30 Hz 时约 2.2 秒后才换下一条。 |
| `relative_chunk_min_execution_fraction` | `0.0` | HuMI `relative_chunk` 的专用执行比例，每次 replan 都可接入新 chunk。即使 HAT 选 `actual`，协议仍是 `relative_chunk`，仍用这一项。 |
| `chunk_end_margin_s` | `0.5` s | 当前 chunk 将结束前提前释放执行 hold，留出新 chunk 接入时间。 |
| `chunk_phase_alignment` | `true` | 按预测所用源状态时间计算延迟帧，接入时跳过已经过时的前缀。 |
| `chunk_start_reference_continuity` | `false` | 开启后，从延迟帧向前搜索更连续的接入帧，并移动播放时钟；要求 `chunk_phase_alignment=true`。 |
| `continuity_search_ahead_s` | `1.0` s | 上一项的向前搜索窗口，仅在该项开启时生效。 |
| `chunk_blend_s` | `0.22` s | 每次正常换 chunk 时的新旧轨迹过渡混合；位置插值，旋转 SLERP。RTC 开启时仍有这一层。 |
| `takeover_blend_s` | `0.75` s | R1 开始播放时，从保持姿态的关节目标渐变到策略目标。 |
| `chunk_alignment` | `fixed` | 旧协议坐标恢复的固定/跟随模式。当前 HuMI chunk 携带显式 `frame_heading_world_wxyz`，直接用生产端原点和朝向恢复世界坐标，不用该项重新估计朝向。 |

### 5.5 RTC 前缀连接

| `env.config` 参数 | 当前默认值 | 作用 |
|---|---|---|
| `rtc_enabled` | `true` | 在换 chunk 时，将旧 chunk 尚未执行的未来接到新 chunk 实际播放的前缀。 |
| `rtc_prefix_frames` | `10` | RTC 前缀总帧数。 |
| `rtc_hard_prefix_frames` | `2` | 前 2 帧完全保留旧 chunk。 |
| `rtc_blend_frames` | `8` | 后 8 帧逐渐从旧 chunk 过渡到新 chunk；必须满足 `hard + blend == prefix`。 |

HuMI 不会自动绕过 RTC 或 `chunk_blend_s`。**`actual` 保证本轮 HAT 解码不继承
旧计划锚点，但 ScaleBFM 播放时仍会用旧 chunk 作短暂平滑过渡。** 两者作用不同。
HAT 记录里的 `action_anchor` 也不是经过 RTC、混合或腕补偿后的最终执行参考。

### 5.6 启动保持和真机后端

| 参数 | 当前默认值 | 作用 |
|---|---|---|
| `env.config.reference_forcing` | `false` | 本文使用实测状态闭环，不把参考 root 状态强制写成机器人观测。 |
| `env.config.prestart_arm_pose` | `stand` | R1 前保持的姿态，可选 `stand`、`raised`、`sonic_0904`；R1 前不播放 live HAT 的未来运动。 |
| `env.config.startup_hand_motor_target` | `[1,1,1,1,1,0]` | 首个 HAT 状态前双手的 DDS 六电机目标，顺序为小指、无名指、中指、食指、拇指弯曲、拇指旋转。 |
| `env.config.startup_hand_timeout_s` | `8.0` s | 等待双手新鲜反馈匹配上述目标的最长时间，每维容差为 `0.05`。 |
| `simulator.config.state_watchdog_hold_last_target` | `true` | 机器人状态/NIC 暂时冻结时保持最后一个全身目标。 |
| `simulator.config.hand_ik_mode` | `coupled_motor` | 按真实 Inspire 六个物理执行器约束做 IK；`independent_joint` 是旧的 12-DOF 模式。 |

当前在线后端用 R1 控制播放。HuMI 按 R1 后的新状态生成首个启动 chunk，收到它之前
继续保持；定位暂停时不推进 HAT 的计划链。上述腕补偿、闭手门控、RTC 等开关仍
按自身配置生效，不会随 `--relative-anchor-mode` 自动切换。

2026-10-05 单臂 dex5 输入排查：左手指尖在训练中几乎恒定，但在线重建值可偏离训练均值8–9σ，导致HuMI右臂外绕被压小。新增 `--left-hand-state-mode train_mean` 仅将模型encoder左指尖10:28输入恢复至训练均值；实测raw记录、右手、姿态和relative action anchor保留实际值。默认仍为 `actual`，双臂任务应保留实测输入。配对离线干预在三个起步输入上将外绕从0.16–1.12cm恢复至7.79–16.04cm，尚未复验自主抓瓶。

已修训练lag增强的relative标签anchor，使它与增强后的输入状态一致，绝对演示future不变；这不会追溯修改现有checkpoint。ScaleBridge的 `env.finger_chunk_source=latest` 已修为未跳帧的原始预测时钟，避免身体continuity搜索提前手指闭合；默认仍为playback。完整证据与限制见 `analysis_outputs/root_cause_20261005/diagnosis.md`。


1010

现在这套 pipeline，用你说的 “实际到了 8 cm，但 target 已经到了 12 cm” 来讲：

1．HAT 一直看当前画面和机器人当前状态，预测下一段动作。
每个 chunk 同时包含手腕怎么走、手指怎么动。按那次的参数，一段预测长 5 秒，约每 0.5 秒重新预测一次。

2．还没进入抓取 FOCUS 时，手腕继续走，手指不允许收紧。
手腕按原来的 continuity 接 chunk。手指从一开始就限制收紧；现在松手可以直接执行。

3．FOCUS 入口是从 HAT 预测里找的。
比如它预测：“手腕到 8 cm 的时候，手指开始收缩到 0.1。”系统就把那个手腕位置、方向当成抓取入口。结合两个 chunk 确认入口。

这里没有识别‘瓶子已经被手包围了’。入口只是模型预测开始收紧时的手腕姿态。

4．实际手腕接近这个入口，才解锁抓取。
目前要求距离在 3 cm 内、方向误差在 10° 内。满足后先保持手腕目标，等一个使用到位后状态生成的新 chunk，再开始抓取 FOCUS。

5．进入 FOCUS 后，手腕和手指目前是分别这样接的。

假设旧 target 是 12 cm，实际手腕到了 8 cm，新 chunk 预测：

- 手腕：8、8.2、8.4
- 手指：逐渐握紧

手腕会变成：12、12.2、12.4。也就是接到旧 target 上，保留新预测里的增量。

手指则由你刚加的参数决定：

- off：直接执行新 chunk 的手指预测。
- continuity：接到上一条手指目标，再执行新 chunk 的变化。
- hold_progress：保留已收紧的程度，新预测更紧才继续收。

目前默认仍是 off，需要启动参数选另外两种。

6．解锁之后，不再持续检查“手腕是不是还在抓取入口”。
它会继续执行新的手腕、手指预测。系统靠实际手指反馈判断抓取完成：达到设定的收紧条件并保持 1 秒，或者 5 秒超时，就结束抓取 FOCUS。这也不是判断瓶子真的抓住了，只是判断手指动作完成了。


所以你最关心的两个“知道不知道”，目前答案是：它不知道手是否真正包围了瓶子，也不知道瓶子是否真正抓住了；它现在用预测入口决定何时允许收紧，用手指反馈决定收紧是否完成。