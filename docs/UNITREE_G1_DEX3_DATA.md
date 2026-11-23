# Unitree G1 Dex3 数据转换与训练接入

原始目录：
`https-huggingface-co-datasets-unitreerobotics-g1-3/outputs/datasets/`。

转换脚本：`scripts/preprocess/convert_unitree_dex3.py`。
详细的坐标系、手指槽位、模型来源、图像解码和异常修复约定见
[assets/g1_dex3/README.md](../assets/g1_dex3/README.md)。

## 输出

| 任务 | Episodes | Frames | 训练 / 验证 episodes | 数据目录 | 数据配置 |
| --- | ---: | ---: | ---: | --- | --- |
| 堆积木 | 301 | 281196 | 271 / 30 | `data/unitree_g1_dex3/block_stacking/episodes/` | `configs/datasets/unitree_g1_dex3_block_stacking.json` |
| 抓苹果 | 201 | 152569 | 181 / 20 | `data/unitree_g1_dex3/pick_apple/episodes/` | `configs/datasets/unitree_g1_dex3_pick_apple.json` |
| 倒水 | 311 | 121587 | 280 / 31 | `data/unitree_g1_dex3/pouring/episodes/` | `configs/datasets/unitree_g1_dex3_pouring.json` |
| 合计 | 813 | 555352 | 732 / 81 | 同上 | `configs/datasets/unitree_g1_dex3_all.json` |

每条轨迹单独一个 `episode_XXXXXX.hdf5`，包含训练所需
`action (T,128)`、`observation.state (T,128)`、JPEG 图像及
`embodiment=dex3`、`fps=30.0` 属性。所有原始轨迹和帧均保留。
原始 28 维指令和观测另存 `source.*`，包括原始时间戳及帧索引。

按 episode 用随机种子 42 固定划分约 90% / 10%，没有帧级泄漏；JSON
使用绝对目录和显式 `file_list`，所有文件只存一份。
各任务的完整清单、帧数、文件大小和修复数量见
`data/unitree_g1_dex3/conversion_report.json`。

本次转换完成：813 / 813 条、555352 / 555352 帧，0 失败；HDF5 合计
19194575729 bytes（约 17.88 GiB，含单头部相机字段别名），初始转换用时约 359 秒。

已完成的检查：

- 全部 813 条的形状、数值、旋转、相机帧数及首/中/末帧 JPEG 解码检查。
- 全量原始关节、时间戳及索引与 Parquet 精确一致；3 个异常观测的修复位置和结果已核对。
- 全部 train/val 文件覆盖完整且互不重叠；10 路末段相机帧重新从源视频定位、编码后字节一致。
- 独立 URDF 矩阵计算及异常修复测试：6 passed。
- 真实 `EpisodicDataset` 的 original / relative_chunk、历史帧、未来帧和批量拼接读取检查。

对应报告位于数据根目录的 `source_integrity_report.json`、
`split_and_video_validation.json` 和 `dataloader_smoke_test.json`。
相机画面及对应 FK 关键点预览：
[preview.png](../data/unitree_g1_dex3/preview.png)。

## 复现和验证

```bash
/home/aigc/miniconda/envs/human_policy/bin/python \
  scripts/preprocess/convert_unitree_dex3.py --workers 12

/home/aigc/miniconda/envs/human_policy/bin/python \
  scripts/preprocess/convert_unitree_dex3.py --verify-only

/home/aigc/miniconda/envs/human_policy/bin/python -m pytest -q \
  tests/test_unitree_dex3_conversion.py
```

转换器先写 `.partial`，检查通过后原子重命名；已完成且转换参数一致的
文件会校验后跳过。不兼容的旧输出会报错，不覆盖。
源数据只读；转换结果写在当前项目内。

图像默认 320x240、JPEG quality 80。复现时如要改这些参数，请指定新的
`--output-root` 和 `--config-root`，避免与已有输出混用。

## 训练接入

按 [TRAINING.md](../TRAINING.md) 使用已有的 128 维配置，只需将
`--dataset_json_path` 指向上表对应 JSON。现有 `camera_names: [top]` 会自动
读取头部左相机。相机列表来自模型 YAML 的 `common.camera_names`，与数据 JSON
的 train/val 划分分开配置：

```yaml
common:
  camera_names: [top]
```

全部 813 条 HDF5 已显式提供 `observation.image.top`，它是
`observation.image.left` 的硬链接，源视频为 `observation.images.cam_left_high`
（头部左相机）。JPEG 长度字段 `observation.image_length.top` 同样使用硬链接，
图像数据没有额外复制。当前帧、历史帧和 future targets 均使用这个单一头部视角。
HDF5 中保留的其他相机用于检查与导出；使用 `[top]` 时不会读取它们作为训练输入。
全量相机字段及实际 dataloader 验证结果见
`data/unitree_g1_dex3/training_camera_report.json`。

推荐继续使用 `--zero_state_dims 98:128` 配置：这批数据没有全身机器人
状态，尾部已置零；原始 28 个关节不等同于旧 Dex5 的全身 qpos 定义。
无需额外重采样，数据原生为 30Hz。

头/腰固定姿态、缺少的两根手指均已明确标记。有效性掩码用于检查，现有
训练器不会自动用它们屏蔽 loss。若要使用语言条件，先核对上游 Apple
任务的错误标签（其原始标签写的是拿红杯子，转换时保留原文）。

## 训练字段完整性审核

全量 813 条、555352 帧通过审核，适用范围为 **128 维 ACT、EE 控制、
`camera_names: [top]`、关闭语言条件**。本轮审核只读 HDF5。

| 项目 | 审核结果 |
| --- | --- |
| `action` / `observation.state` | 全部 `(T,128)` float32，数值有限，128 维布局及 6D 旋转有效 |
| 原始关节与索引 | `source.*` 与上游 Parquet 精确一致；测量 state 和命令 action 分别转换，未额外移帧 |
| 时间轴 | state/action/头部图像帧数一致，时间戳单调、原生 30Hz |
| 属性 | `embodiment=dex3`、数值 `fps=30.0`、任务文本及来源信息完整 |
| 零值与有效性标记 | 未观测的 98:128 尾部、缺失手指及保留维度符合约定；三个遥测异常的修复位置完整保留 |
| 划分 | 三个任务及合并 JSON 全覆盖，训练 732 条、验证 81 条，互不重叠 |
| 归一化 | 实际训练入口按 dex3 计算统计；mean/std 均有限，std 下限 0.01；验证集使用训练集统计 |
| batch | 三任务独立配置及合并配置均通过真实 8-worker train/val 加载，chunk size 100 |
| 序列末尾 | original / relative_chunk 均检查全部 813 条的末尾动作和 padding，未发生越界或错位 |
| 附加图像路径 | 历史帧、未来四帧视频片段、future_valid 及其 collate 均通过 |
| 模型接口 | 实际 batch 通过 ResNet ACT 的 CPU 前向与反向，loss 及 261 个参数梯度均有限；未更新权重 |

完整报告：`data/unitree_g1_dex3/training_compatibility_report.json`。
审核脚本：`scripts/preprocess/check_unitree_dex3_training.py`，可分别运行
`--phase schema` 与 `--phase runtime`；runtime 需要允许本地多进程通信。

这些检查验证训练加载与模型接口。头/腰仍为固定参考值、两根缺失手指仍为零；
审计有效性掩码不会自动改变 ACT 的 loss（neck loss 在 policy 中单独置零）。
语言条件训练还需要核对 Apple 标签并准备任务语言 embedding。
`qpos` 控制所需的 `old_action` 和可选 flow 分支的预计算 target 不在此数据格式中。
