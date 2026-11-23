# SONIC 真机遥操 → HAT 128 维 HDF5 转换说明

本目录是 `sonic_real_teleop_20260903_curated` 数据集转成 human_policy/HAT 训练格式的产物。

- 转换脚本：`human_policy/convert_sonic_to_hat_hdf5.py`
- 输入：`sonic_real_teleop_20260903_curated/usable_dataset`（LeRobot 格式，46 episode，50 FPS，单目 ego 相机 480×640）
- 输出：本目录 46 个 `SONIC_Dex5_20260903_ep_XXXX_rootfix.hdf5`，共 63,152 帧，约 1.4 GB
- 复现命令：

```bash
PYTHONPATH=/root/shengyin/TWIST /root/miniconda3/envs/isaaclab_sonic/bin/python \
  human_policy/convert_sonic_to_hat_hdf5.py \
  --output-dir human_policy/data/sonic_real_teleop_20260903_hat_rootfix --overwrite
```

## 转换流程

### 1. 关节角

SONIC 的 `observation.state` 是 43 维。取 `[:22]` 与 `[29:36]` 拼成 G1 的 29 自由度 body 关节（丢掉中间两段 7 维手部块）。

### 2. root 朝向

`observation.root_orientation`（IMU，wxyz）→ xyzw，做符号连续化、σ=1 的高斯平滑、归一化，再用前 20 帧的平均 yaw 反向旋转，使 episode 起始朝向对齐到 +X。

### 3. root 平移（关键，非真值）

原始数据**没有记录 base 平移**，脚本用**支撑脚运动学里程计**恢复：

1. 用 TWIST 的 `KinematicsModel`（`TWIST/assets/g1inspire/g1_29dof_inspire_hand.xml`）在 root 平移置零、只给朝向的条件下做 FK，得到双踝位姿；踝关节沿本体系偏移 `[0.035, 0, -0.030]` 得到脚底接触点。
2. 选 z 更低的脚为支撑脚，带 12 mm 滞回、3 帧去抖、最小 8 帧切换间隔。
3. 假设支撑脚在世界系静止，反解 pelvis 的 xy；z 取 `-支撑脚底高度`（即踩地约束）。支撑脚切换时把锚点续接到上一帧位置，保证连续。
4. 对结果做 σ=1.5 高斯平滑，并把首帧 xy 归零。

同时写出逐帧 `root_odometry_stance`（支撑脚索引）和 `root_odometry_confidence`（由双脚高度差映射到 0.70–1.0）。

> 该 root 是估计值，不是外部定位真值。全量统计：位移中位数 0.38 m（0.13–0.53），路径长度中位数 1.12 m，z 波动中位数 0.034 m，支撑脚切换 1–4 次。

### 4. 关键点 FK

拿到 root 位姿后，再做一次完整 FK，取 `head_mocap`、`L_hand_base_link`、`R_hand_base_link`（以及双踝）。旋转统一转成 6D（旋转矩阵前两行）。

### 5. 坐标原点

沿用旧 rootfix 约定：**前 10 帧头部位置的均值**作为 (0,0,0)，头/双腕/root 全部减去该偏移。原始偏移记录在 attr `head_origin_source_mean` / `head_origin_translation`。

### 6. 手指

`observation.inspire.{left,right}_position`（6 维 SDK 反馈）→ 按旧流程做「0–4 通道开合反转 + 拇指 pitch/yaw 互换」映射到 12 维关节角，再用 pinocchio 对 `TWIST/assets/inspire_hand/inspire_hand_{left,right}.urdf` 做 FK 取 5 个指尖。**指尖坐标是手掌基座局部系**（`plot_keypoints.py` 会用腕部位姿把它变换到世界系），槽位 0 留 0，指尖填 1–5。

### 7. 128 维布局

| 区间 | 内容 |
|---|---|
| `0:9` | head 位置 + 6D 旋转 |
| `10:28` | 左手 6×3 关键点（局部系） |
| `30:39` | 右腕位置 + 6D 旋转 |
| `40:58` | 右手 6×3 关键点（局部系） |
| `80:89` | 左腕位置 + 6D 旋转 |
| `89:98` | root 位置 + 6D 旋转（`plot_keypoints.py` 里当作 waist 画） |
| `100:126` | qpos：root xyz + quat(wxyz) + 前 19 个 body 关节 |
| 其余 | 0 |

`action[t] = state[t+1]`，最后一帧重复。

### 8. 图像

用 OpenCV 逐帧解码 ego mp4，JPEG 质量 50 编码，存成变长 uint8 行（`observation.image.left`）。SONIC 只有一路相机，所以 `observation.image.right` 和 `observation.image.ego_view` 是指向同一 dataset 的**硬链接**——训练时不要当成双目用。

### 9. 其他 dataset / attrs

- `waist_valid`：全 True
- attrs：`sim=False`、`embodiment=dex5`、`fps=50`、`description`（任务名 `pickpillow0903`）、`rootfix_method`、`root_odometry_is_ground_truth=False`、各项里程计诊断量、源文件路径

## 顺带改动的两个文件

- `human_policy/pytorch3d/__init__.py`、`transforms.py`：仓库里原本只有空的 `pytorch3d/` 目录，`plot_keypoints.py` 导入失败。补了 `rotation_6d_to_matrix` / `matrix_to_rotation_6d` 两个与 PyTorch3D 约定一致的实现。
- `human_policy/data/plot_keypoints.py`：本机没装 ffmpeg，MP4 导出必失败。加了 OpenCV `VideoWriter`（mp4v）兜底，两处 `--save_mp4` 分支各一份。

## 已知的数据特性（非转换 bug）

- **左手全程不动**：46 个 episode 的 `observation.inspire.left_position` 原始值都恒为 ≈1.0（张开），逐通道变化 < 0.03。转换后左手指尖时间标准差 < 0.15 mm。右手正常，各通道变化幅度 0.4–1.0。这是采集时只用右手做 pickpillow 任务的结果。
- root 平移是估计值，见上文第 3 节。

## 可视化

```bash
PYTHONPATH=/root/shengyin/human_policy /root/miniconda3/envs/isaaclab_sonic/bin/python \
  human_policy/data/plot_keypoints.py \
  --file <episode>.hdf5 --full_hand --spine \
  --save_mp4 out.mp4 --fps 50 --width 960 --height 720
```

**注意 `--max_seconds`**：不加就渲染整段；加了会按 `ceil(max_seconds * fps)` 截断帧数。本目录里的 `SONIC_Dex5_20260903_ep_0000_keypoints.mp4` 是用 `--max_seconds 1` 生成的，所以只有 50 帧 / 1 秒。完整版 `SONIC_Dex5_20260903_ep_0000_keypoints_full.mp4` = 1582 帧 / 31.6 s / 960×720。

**`--spine` 对本数据无效果**：`plot_keypoints.py:395` 要求 head_tracker 通道（`58:67`）非零才画脊柱线，而本转换没有写 head_tracker（该数据源没有独立头部 tracker），所以加不加 `--spine` 输出一样。waist（即 root，`89:98`）本身是会画出来的。

## 2026-09-04：Vive tracker root 版本

09-04 数据在机器人腰部增加了 Vive tracker。对应转换没有修改上面的 09-03 脚部 FK 脚本，而是新增了一条独立流程：

- 主转换脚本：`human_policy/convert_sonic_vive_to_hat_hdf5.py`
- 复用的 FK、手指和 HDF5 封装代码：`human_policy/convert_sonic_to_hat_hdf5.py`
- 输入数据：`sonic_real_teleop_20260904_curated`
- Vive 原始日志：`sonic_real_teleop_20260904_curated/tracker_data/*.jsonl`
- episode 与 tracker 对应关系：`sonic_real_teleop_20260904_curated/usable_dataset/meta/quality_manifest.jsonl`
- 输出数据：`human_policy/data/sonic_real_teleop_20260904_hat_rootfix`
- 转换报告：`human_policy/data/sonic_real_teleop_20260904_hat_rootfix/conversion_report.json`

复现命令：

```bash
PYTHONPATH=/root/shengyin/human_policy:/root/shengyin/TWIST \
  /root/miniconda3/envs/isaaclab_sonic/bin/python \
  human_policy/convert_sonic_vive_to_hat_hdf5.py --overwrite
```

### Vive root XYZ 处理流程

1. 使用腰部的 `tracker1`，按 `sonic_start_unix + parquet timestamp` 将 120 Hz tracker 日志同步到 SONIC 的 50 Hz 帧。
2. SteamVR 是 Y-up，先变换为 Z-up：`(x, y, z)_vive -> (x, -z, y)_zup`。
3. 用 tracker 全程干净的 episode 0、5、9 求一个**仅旋转、不缩放**的水平对齐角；本批数据为 `14.385°`，转换到 `X 前、Y 左、Z 上`的初始朝向坐标系。
4. tracker 帧必须同时满足：时间附近有真实采样、相对 FK 的 XY 残差不超过 25 cm、Z 残差不超过 12 cm、帧间速度不超过 2 m/s。未通过的米级跳点和丢帧不会写入训练 root。
5. 坏段不是直接把绝对位置切换成 FK，而是使用连续偏差融合：

   ```text
   correction_valid = Vive_xyz - FK_xyz
   correction_all   = 对有效 correction 跨缺口插值并平滑
   root_xyz          = FK_xyz + correction_all
   ```

   例如丢失前 Vive 已前进 10 cm、FK 只有 8 cm，则保留 `+2 cm` 偏差；缺口内 FK 若继续前进 5 cm，输出变成约 15 cm，不会跳回 8 cm。Vive 恢复后，偏差再连续对准恢复后的 Vive 锚点。
6. root 旋转仍来自机器人 IMU；得到 root 位姿后，头部、双腕、手指 FK、前 10 帧头部归零、128 维布局和图像封装均与 09-03 流程一致。

实际训练用的 root 写在 `observation.state/action[89:98]` 和 qpos 对应位置。额外保留：

- `root_translation_vive_valid`：每帧是否有通过门控的 Vive 测量。
- `root_translation_vive_clean`：通过门控的 Vive root；无效帧为 NaN，仅用于检查，不作为训练 state。
- `root_odometry_stance` / `root_odometry_confidence`：用于缺口补偿的脚部 FK 诊断。
- attr `root_translation_source=vive_tracker1_gated_with_stance_fk_gap_fill`。
- attr `root_rotation_source=robot_imu`。

09-04 共输出 11 条、13,529 帧，约 298 MB。episode 0、5、9 的 Vive 全程有效；其他 episode 的有效率约 81%–94%，但 episode 3 只有 17.9%，最长连续缺口 18.24 秒，因此 episode 3 的 root 大部分仍由 FK 维持，不应视为 Vive Ground Truth。

## 2026-09-06 至 09-07：bottle-to-basin 新数据

输入数据为 `pick_bottle/sonic_real_teleop_20260906_07_curated`，任务是将瓶子放入盆中。该批数据经过筛选后有 103 条可用 episode、92,204 帧；对应的 103 个 HDF5 输出在 `human_policy/data/sonic_real_teleop_20260906_07_hat_rootfix`，报告为同目录下的 `conversion_report.json`。

处理仍沿用 09-04 的 Vive root、脚部 FK 缺口补偿、IMU root 旋转、手指 FK 和 128 维布局。由于两次采集的 tracker 编号发生变化，转换脚本启用了自动选择：逐 episode 将两个 tracker 的同步相对旋转与机器人 IMU 比较，选择误差更小且通过有效性门控的通道作为腰部 tracker。结果为：09-06 的 60 条使用 `tracker2`，09-07 的 43 条使用 `tracker1`。选择结果、序列号和运动诊断都写入 HDF5 attrs 与转换报告。

复现命令：

```bash
PYTHONPATH=/root/shengyin/human_policy:/root/shengyin/TWIST \
  /root/miniconda3/envs/isaaclab_sonic/bin/python \
  human_policy/convert_sonic_vive_to_hat_hdf5.py \
  --dataset-root pick_bottle/sonic_real_teleop_20260906_07_curated \
  --output-dir human_policy/data/sonic_real_teleop_20260906_07_hat_rootfix \
  --output-prefix SONIC_Dex5_20260906_07 \
  --calibrate-by-source-run --auto-waist-tracker --overwrite
```

关键点视频示例（episode 0）：

```bash
PYTHONPATH=/root/shengyin/human_policy \
  /root/miniconda3/envs/isaaclab_sonic/bin/python \
  human_policy/data/plot_keypoints.py \
  -f human_policy/data/sonic_real_teleop_20260906_07_hat_rootfix/SONIC_Dex5_20260906_07_ep_0000_rootfix.hdf5 \
  --full_hand --spine \
  --save_mp4 human_policy/data/sonic_real_teleop_20260906_07_hat_rootfix/SONIC_Dex5_20260906_07_ep_0000_keypoints.mp4 \
  --fps 20 --width 960 --height 720
```
