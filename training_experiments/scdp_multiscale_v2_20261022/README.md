# SCDP-Lite 五尺度 v2

独立实验目录：`training_experiments/scdp_multiscale_v2_20261022`。
v1、原 HAT、训练数据、相机标定 sidecar、部署模型均不修改。
参考原 SCDP 的多尺度轨迹视觉采样和 FiLM 思路：https://scdp-project.github.io/ 。
仍是一轮 coarse → project/sample → refine，没有 Diffusion 去噪循环。

## 特征来源与尺寸

冻结的 V-JEPA 2.1 ViT-B 正常执行一次，通过临时 forward hook 读取第 3、6、9 层 block 的真实 tokens，使用对应预训练 `norms_block`；第 12 层沿用原 Backbone 的最终输出。
不更改 `out_layers`、Backbone 权重、原 HAT 使用的视觉 tokens 或 Decoder。

| 尺度 | 中间层（从 1 起） | 新增空间 stride | 特征图 `[B,C,H,W]` |
| --- | --- | --- | --- |
| P0 | 3 | 1 | `[B,128,14,19]` |
| P1 | 6 | 2 | `[B,128,7,10]` |
| P2 | 9 | 4 | `[B,128,4,5]` |
| P3 | 12 | 8 | `[B,128,2,3]` |
| P4 | 12 | 16 | `[B,128,1,2]` |

各尺度使用独立可训练 1×1 lateral、GroupNorm、GELU 和 depthwise 3×3 卷积；P1–P4 分别执行 1–4 次 stride-2 下采样，P0 使用 stride-1 3×3 卷积。
没有特征插值或上采样。P3/P4 虽共享第 12 层来源，但空间聚合范围和网格不同，不是重复采样同一张特征图。最细尺度仍是原始 14×19 patch 网格，不声称恢复更细图像空间信息。

每手沿未来轨迹均匀选择 8 点，投影和有效性掩码与 v1 相同。每个尺度按有效轨迹点取平均，再拼接成 2×5×128=1280 维 context，通过原形式的零初始化 FiLM（隐层 256）调制 `[B,100,512]`。最终只替换有效手的腕 xyz，其他动作通道原样保留。
下采样 3×3、stride-2、padding-1 的坐标中心变换单独计算，不能直接在每层复用同一个归一化 grid。

## 训练

配置：`configs/bases.json`、`configs/experiment.json`。

三组均从原底座重新开始训练新增五尺度模块，不加载 v1 conditioner，不使用过拟合测试权重。
全局 batch 64、AdamW LR 1e-5、weight_decay 1e-4、clip 0.1、seed 0、100000 步、每 1000 步保存；数据、增强、归一化、mask、原 ACT/CVAE loss 均沿用 v1。
两个 geometry 底座保留原 0.1 geometry loss 及其 hand-event weights；20261004 原底座不新增该 loss。

仅使用 GPU 6/7，三组并行共享：

- `original_20261004`：GPU 6,7，DDP 每进程 batch 32，与其 v1 的两卡设置一致。
- `geometry`：GPU 6，batch 64，与其 v1 单卡设置一致。
- `geometry_no_left_wrist`：GPU 7，batch 64，与其 v1 单卡设置一致。

共享 GPU 只影响运行时间。启动器遇到占用或重复结果目录会拒绝启动，不会杀其他进程。

```bash
cd /home/aigc/human_policy
/home/aigc/miniconda/envs/human_policy/bin/python training_experiments/scdp_multiscale_v2_20261022/src/launch_v2.py
```

FiLM 最后一层为零时，首步只有 FiLM 输出层有非零梯度，这是初始化的预期行为。第二步检查五个 pyramid level 都有非零且有限的梯度；冻结 Backbone/HAT 无梯度。
日志：`logs/<base>.log`；状态：`logs/<base>_status.json`；模型：`checkpoints/<base>/step_<N>.pt`。

## 评估

统一实现：`src/evaluate_protocol.py`。不向模型传 GT 动作，使用 prior 推理。

1. **Query-0 MPJPE**：使用原 `data/eval_mpjpe_batch.py` 的 16 个 scene joints、dirty-start 默认规则、机器人 30 Hz 取帧、qpos 归一化 epsilon 1e-8、输出反归一化与 per-episode macro average。此指标不平均未来 100 个 query。
2. **合手前 0.5 秒腕 xyz**：沿用既有 GT 手指开合检测器的 closing onset 作为抓取代理；窗口按原录制时间计算，不宣称是物理接触时刻。在 query-0 上统计相机 xyz MAE 和腕位置 L2（mm）。只统计有标定且对应手有效的帧，报告计数。旧人类批次没有有效标定，不能报告其相机轴误差。
3. **瓶子平移响应**：只接受真实、经检查仅瓶子移动的成对图像；固定模型状态和相机条件。未提供合法成对数据时明确标为 pending，不将全图平移称为瓶子平移。

两个新 geometry 的 v1 保留到第 6000 步，v2 到第 6000 步时自动在完整验证集上比较 baseline/v1/v2；原底座的 v1 已完成 100000 步，v2 完成时自动比较。步数不同会拒绝 v1/v2 对比。
所有 v2 完成后自动执行完整 query-0 与合手前窗口评估，结果分别保存在 `results/<base>/`。

```bash
SCDP_BASE_VARIANT=geometry CUDA_VISIBLE_DEVICES=6 \
  /home/aigc/miniconda/envs/human_policy/bin/python \
  training_experiments/scdp_multiscale_v2_20261022/src/evaluate_protocol.py \
  --checkpoint training_experiments/scdp_multiscale_v2_20261022/checkpoints/geometry/step_6000.pt \
  --reference training_experiments/scdp_geometry_20261022/checkpoints/geometry/step_6000.pt
```

瓶子成对数据的可选 JSON 清单为数组，每项包含 `episode`（已有 HDF 绝对路径）、`frame_a`/`frame_b`（原始帧号）、`bottle_uv_a`/`bottle_uv_b`（320×240 坐标）、`bottle_only_change_verified: true`、可选 `query_index`（默认 0）。通过 `--pairs <manifest>` 指定。脚本拒绝头、腕、手指或腰位置变化超过 5 mm / 旋转 6D 分量变化超过 0.01 的配对；仍需独立确认场景只移动瓶子。输出腕位移、投影位移及沿瓶子位移方向的 response gain。

## 验证

18 项单元测试包含：原投影/裁剪/畸变/左右手 mask/回退、五尺度尺寸、真实下采样、stride 坐标对齐、零初始化、全部新增尺度梯度。
`src/check_variant.py` 使用真实 checkpoint + 人类/机器人数据检查原模型冻结、关闭时严格等价、geometry loss、no-left-wrist 输入不变性和五尺度梯度。
`--overfit-steps 100 --batch-size 8 --lr .001 --workers 0` 只做固定 mini-batch 学习检查，不保存或复用该权重。过拟合结果不是验证集收益。
