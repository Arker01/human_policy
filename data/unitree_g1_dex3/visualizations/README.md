# G1 Dex3 任务可视化视频

每个任务随机选取 3 条完整轨迹（随机种子 42）。
由 `data/plot_keypoints_ys.py` 绘制 `action` 的三维位姿和关键点；960×720，30 fps。
全部视频已经逐帧解码，帧数与源 HDF5 一致。

| 任务 | Episode | 时长（秒） | MP4 |
| --- | --- | ---: | --- |
| block_stacking | episode_000012 | 38.33 | [打开视频](block_stacking/episode_000012.mp4) |
| block_stacking | episode_000057 | 30.63 | [打开视频](block_stacking/episode_000057.mp4) |
| block_stacking | episode_000140 | 31.03 | [打开视频](block_stacking/episode_000140.mp4) |
| pick_apple | episode_000006 | 49.60 | [打开视频](pick_apple/episode_000006.mp4) |
| pick_apple | episode_000028 | 51.40 | [打开视频](pick_apple/episode_000028.mp4) |
| pick_apple | episode_000163 | 19.57 | [打开视频](pick_apple/episode_000163.mp4) |
| pouring | episode_000012 | 14.97 | [打开视频](pouring/episode_000012.mp4) |
| pouring | episode_000057 | 12.27 | [打开视频](pouring/episode_000057.mp4) |
| pouring | episode_000140 | 11.97 | [打开视频](pouring/episode_000140.mp4) |

选择清单、源路径、完整导出参数及验证结果见 `sample_videos.json`。

## 真实相机画面及同步对比

新增每条轨迹的两个完整 MP4，均为 30 fps，并与关键点视频逐帧对应：
- `_real.mp4`：直接从源 HDF5 的 JPEG 解码，按相机标签拼图；堆积木/倒水有头部左右及左右手腕四路，抓苹果有头部左右两路。
- `_compare.mp4`：左侧为相机画面，右侧为此前导出的 `action` 关键点，1920×760。

图像按 HDF5 标记的 RGB 解码约定读取，避免红蓝通道颠倒。所有相机帧均保留，未重新定位下载目录中的视频。

| 任务 | Episode | 实拍视频 | 同帧并排对比 |
| --- | --- | --- | --- |
| block_stacking | episode_000012 | [实拍](block_stacking/episode_000012_real.mp4) | [并排对比](block_stacking/episode_000012_compare.mp4) |
| block_stacking | episode_000057 | [实拍](block_stacking/episode_000057_real.mp4) | [并排对比](block_stacking/episode_000057_compare.mp4) |
| block_stacking | episode_000140 | [实拍](block_stacking/episode_000140_real.mp4) | [并排对比](block_stacking/episode_000140_compare.mp4) |
| pick_apple | episode_000006 | [实拍](pick_apple/episode_000006_real.mp4) | [并排对比](pick_apple/episode_000006_compare.mp4) |
| pick_apple | episode_000028 | [实拍](pick_apple/episode_000028_real.mp4) | [并排对比](pick_apple/episode_000028_compare.mp4) |
| pick_apple | episode_000163 | [实拍](pick_apple/episode_000163_real.mp4) | [并排对比](pick_apple/episode_000163_compare.mp4) |
| pouring | episode_000012 | [实拍](pouring/episode_000012_real.mp4) | [并排对比](pouring/episode_000012_compare.mp4) |
| pouring | episode_000057 | [实拍](pouring/episode_000057_real.mp4) | [并排对比](pouring/episode_000057_compare.mp4) |
| pouring | episode_000140 | [实拍](pouring/episode_000140_real.mp4) | [并排对比](pouring/episode_000140_compare.mp4) |

逐帧解码、时间戳、分辨率、帧率及抽样 RGB 差异检查结果见 `real_world_videos.json`。
导出脚本：`scripts/preprocess/export_unitree_dex3_reference_videos.py`。
