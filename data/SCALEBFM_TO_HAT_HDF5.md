# ScaleBFM 实机采集数据转 HAT HDF5

本文说明如何处理新版 ScaleBFM 遥操采集目录。目标产物是 `human_policy` 使用的 128 维 HDF5，以及与每个 episode 一一对应的原始 ego MP4。

## 1. 结论

新版 ScaleBFM 采集器已经按 SONIC 的 LeRobot schema 记录 HDF5 转换所需数据，可以复用 SONIC 的 coupled-motor 手部映射。转换不需要 latency、motion token、encoder input 或 planner 字段。

采集结束后先保留完整原始目录，不要只复制 Parquet。一个可转换 episode 必须同时具有：

- `data/.../episode_XXXXXX.parquet`
- `videos/.../observation.images.ego_view/episode_XXXXXX.mp4`
- 若使用 Vive root 平移：对应的 `tracker_data/*.jsonl`
- 根目录下的 `meta/info.json`、`meta/tasks.jsonl`、`meta/episodes.jsonl`

## 2. 原始数据位置与结构

ScaleBFM 默认输出：

```text
/home/nerv/qingyaoxu/GR00T-WholeBodyControl/outputs/
  pico_scalebfm_YYYYMMDD_HHMMSS/
```

关键结构：

```text
pico_scalebfm_YYYYMMDD_HHMMSS/
├── data/chunk-000/episode_000000.parquet
├── videos/chunk-000/observation.images.ego_view/episode_000000.mp4
├── tracker_data/*.jsonl
├── scalebfm_data/chunk-000/episode_000000.jsonl
└── meta/
```

`scalebfm_data` 是原始 ScaleBFM sidecar，转换时通常不用，但必须保留用于排查和重新对齐。

建议把完整目录传到 4090-168：

```bash
rsync -a --info=progress2 \
  /home/nerv/qingyaoxu/GR00T-WholeBodyControl/outputs/pico_scalebfm_YYYYMMDD_HHMMSS/ \
  4090-168:/root/shengyin/human_policy/data/raw_scalebfm/pico_scalebfm_YYYYMMDD_HHMMSS/
```

## 3. 转换前必须检查的内容

每个 Parquet 必须包含以下列和维度：

| 列 | 维度 | 用途 |
|---|---:|---|
| `observation.state` | 43 | 29DoF 身体状态；转换脚本提取标准 29DoF |
| `observation.root_orientation` | 4 | 机器人 IMU root 四元数，顺序 wxyz |
| `observation.inspire.left_position` | 6 | 左手六电机实测反馈 |
| `observation.inspire.right_position` | 6 | 右手六电机实测反馈 |
| `timestamp` | 1 | episode 相对时间；Vive 同步需要 |
| `task_index` | 1 | HDF5 description |

还要检查：

1. 左右手反馈不是全零、不是常数，并且没有大段 `telemetry_fresh=0`。
2. MP4 可解码，视频帧数与 Parquet 行数一致。
3. 若使用 Vive，tracker JSONL 必须包含 `sonic_start_unix`，且覆盖对应 episode 的时间范围。
4. Parquet、MP4、tracker 必须来自同一次采集，禁止跨 run 拼接。

当前 ScaleBFM recorder 的写入策略是：手部反馈、PICO pose 或相机图像任一缺失时，该帧不写入。因此落盘的 Parquet 行和 MP4 帧按同一个 `add_frame` 配对；这是软件帧同步，相机原始时间戳另存在 `observation.camera_timestamp.ego_view`。

## 4. 手部 coupled-motor 处理

不要从 `teleop.pico_controls` 或手部 command 推算手指；必须使用六电机实测反馈：

```text
observation.inspire.left_position
observation.inspire.right_position
```

六电机顺序：

```text
[pinky, ring, middle, index, thumb_bend, thumb_rot]
```

转换脚本 `convert_sonic_to_hat_hdf5.py` 已使用 `inspire_motor_mapping.py` 的 coupled LUT：

```text
motor6 feedback -> coupled 12DoF -> Inspire URDF FK -> 5 fingertips
```

因此不需要为 ScaleBFM 单独实现一套手部映射。

## 5. 两种转换方式

工作目录和 Python：

```bash
cd /root/shengyin/human_policy
PY=/root/miniconda3/envs/isaaclab_sonic/bin/python
```

### 5.1 不使用 Vive：支撑脚 FK 估计 root 平移

新版 ScaleBFM 目录本身就是标准 LeRobot dataset，可直接传给普通转换脚本：

```bash
$PY convert_sonic_to_hat_hdf5.py \
  --dataset /root/shengyin/human_policy/data/raw_scalebfm/pico_scalebfm_YYYYMMDD_HHMMSS \
  --output-dir /root/shengyin/human_policy/data/scalebfm_YYYYMMDD_hat_rootfix \
  --jpeg-quality 50
```

只转换一个 episode：

```bash
$PY convert_sonic_to_hat_hdf5.py \
  --dataset /root/shengyin/human_policy/data/raw_scalebfm/pico_scalebfm_YYYYMMDD_HHMMSS \
  --output-dir /root/shengyin/human_policy/data/scalebfm_YYYYMMDD_hat_rootfix \
  --episode 0 \
  --jpeg-quality 50
```

### 5.2 使用 Vive：Tracker 平移 + 支撑脚 FK 补洞

`convert_sonic_vive_to_hat_hdf5.py` 目前要求下面的包装结构，不能把原始 ScaleBFM 目录直接作为 `--dataset-root`：

```text
<dataset-root>/
├── usable_dataset/
│   ├── data/...
│   ├── videos/...
│   └── meta/quality_manifest.jsonl
└── tracker_data/*.jsonl
```

处理新数据时应先：

1. 保留原始目录不动。
2. 在新的 curated 目录下建立 `usable_dataset`，复制或硬链接合格的 Parquet、MP4 和 meta 文件。
3. 给每个 curated episode 生成一条 `quality_manifest.jsonl`，至少正确填写 `source_run`、`source_episode`、`curated_episode`、`length` 和 `tracker_file`。
4. 把 tracker JSONL 放入 curated 根目录的 `tracker_data/`。
5. 确认 episode 重编号后，Parquet、MP4 和 manifest 的 `curated_episode` 完全一致。

然后运行：

```bash
$PY convert_sonic_vive_to_hat_hdf5.py \
  --dataset-root /root/shengyin/human_policy/data/scalebfm_YYYYMMDD_curated \
  --output-dir /root/shengyin/human_policy/data/scalebfm_YYYYMMDD_hat_rootfix_coupled_motor \
  --output-prefix ScaleBFM_Dex5_YYYYMMDD \
  --auto-waist-tracker \
  --jpeg-quality 50
```

这里的 `coupled_motor` 指六电机反馈通过 coupled LUT 转为手部 12DoF；它与是否使用 Vive 是两个独立概念。普通转换脚本也已经使用相同的 coupled-motor 手部映射。

## 6. HDF5 与 MP4 的对应关系

转换脚本会把 MP4 解码为 JPEG 帧并写进 HDF5 的 `observation.image.*`，但不会生成一个新的外部 MP4。原始 MP4 必须继续保留，并建立一一对应关系：

```text
episode_000000.parquet
episode_000000.mp4
ScaleBFM_Dex5_YYYYMMDD_ep_0000_rootfix.hdf5
```

如果要求在输出目录中同时交付 HDF5 和 MP4，应把原始 MP4复制或硬链接为同前缀文件：

```text
ScaleBFM_Dex5_YYYYMMDD_ep_0000_rootfix.hdf5
ScaleBFM_Dex5_YYYYMMDD_ep_0000_ego.mp4
```

不要重新编码 MP4，除非确认帧数和时间基准不会改变。

## 7. 转换后验收

每个 episode 至少检查：

- HDF5 中 `observation.state.shape == (T, 128)`。
- HDF5 中 `action.shape == (T, 128)`，且 `action[t] == state[t+1]`，最后一帧重复。
- HDF5 图像帧数、Parquet 行数、MP4 帧数均为 `T`。
- 左右手 fingertip 区域不是全零，手开合时连续变化。
- Root 朝向连续；使用 Vive 时检查 `root_translation_vive_valid` 和 fallback 时长。
- HDF5 属性中 `inspire_transform == dds_motor6_coupled_lut_v1_open1`。
- 输出文件保留 `source_parquet`、`source_video`，Vive 版本还要保留 `source_vive_tracker`。

任何一个 episode 出现手部反馈缺失、视频帧数不一致或 tracker 对错 episode，都应先隔离，不要直接混入训练集。

## 8. 本流程明确不使用的数据

以下字段即使存在，也不参与当前 HAT HDF5 转换：

- `action.motion_token`
- `teleop.encoder_mode`
- `teleop.encoder_input`
- encoder/motion latency
- planner 内部状态
- 录制的 `action.wbc`（HDF5 action 由下一帧 state 构造）

