# 训练指南：拿到新数据 / 新任务之后怎么跑

本文是统一训练入口，包含数据要求、训练命令、默认 30fps 转换、HuMI relative chunk、
lag/vis 增强，以及评测和部署。新增功能的配置见 [§5.7](#57-humi与lagvis增强)。

这份文档面向「手上有一批新数据或一个新任务，想训一个能用的策略」的人。
只写目前验过、跑得出结果的那条路；被证伪的分支在最后一节列了「别踩」。

结论先放这里：**这一周所有改动里，唯一站得住的是把输入主干换成冻结的 V-JEPA 2.1 ViT-B**
（扰动鲁棒性 9.5% → 4.1%，是全表唯一通过显著性检验的改动）。
输出端那个 Future-DINO「world 头」在 resnet18 主干上有点用（2.4pt），
在 V-JEPA 主干上**完全没用**（λ=0 和 λ=1 打平，11 个轴无一显著）。
所以新任务的默认选择是：**V-JEPA 主干 + world 头关掉**。

---

## 一、三个推荐配置（该选哪个）

| | 配置 | 用什么 model cfg | 参考 ckpt | 速度 (RTX 4090) | 什么时候选它 |
|---|---|---|---|---|---|
| **首选** | V-JEPA 主干，world 头 λ=0 | `hdt/configs/models/act_input_vjepa.yaml` + `--future_dino_weight 0.0` | `BEST_CKPT_vjepa_w0/` | 2.05 it/s，10 万步 ≈ **14h** | 视觉条件跟训练时不完全一致（换场景/换光照/换相机位）时唯一靠谱的一档 |
| 次选 | resnet18 主干 + Future-DINO 头 λ=1.0 / horizon 45 | `hdt/configs/models/act_with_future_dino.yaml` | `BEST_CKPT/`（ab4） | 2.89 it/s，10 万步 ≈ **9.6h** | 卡不够、要快、部署环境跟训练画面基本一致；ckpt 也小（390MB vs 679MB） |
| 部署省事版 | ab4 + 遮掉 `qpos[98:128]` | `hdt/configs/models/act_with_future_dino_nostate.yaml` + `--zero_state_dims 98:128` | `BEST_CKPT_ab4_nostate/` | 2.89 it/s ≈ 9.6h | 真机上凑不齐那 30 维机器人构型输入时。**注意它只在 resnet18 上有效，V-JEPA 上不叠加** |

三者在**干净画面**上分不出高低（10 集验证集能分辨的最小差约 2mm，三者都在这个噪声里）。
分高低的是扰动套件，见第六节。

> **V-JEPA 那档为什么写「λ=0」而不是「不挂头」**：`act_input_vjepa.yaml` 里带着 `future_dino` 段，
> 已交付的 `BEST_CKPT_vjepa_w0` 就是「头挂着、权重 0、不进梯度」跑出来的（机制同 ab1）。
> 训练时 DINOv2-S teacher 仍会被调用（日志里 `future_dino_loss` 有数、`future_dino_effective_weight: 0.000`），
> 所以**训练机上仍然需要 DINOv2-S 权重**；推理时不需要。想彻底不挂头就自己复制一份 yaml 把 `future_dino` 段删掉，
> 但那样架构就跟已交付 ckpt 不完全一致了。

---

## 二、环境

```bash
conda activate human_policy        # 本机路径 /home/aigc/miniconda/envs/human_policy
cd /home/aigc/human_policy
```

torch 2.5.1+cu124 / python 3.11。全新机器按 [docs/INSTALL.md](./docs/INSTALL.md) 装。

**离线权重（缺了就建不出模型，报错在 torch.hub 那一步）：**

| 用途 | 路径 | 大小 | 谁需要 |
|---|---|---|---|
| V-JEPA 2.1 ViT-B 代码 | `~/.cache/torch/hub/facebookresearch_vjepa2_204698b45b3712590f06245fbfba32d3be539812` | 6MB | 首选档 |
| V-JEPA 2.1 ViT-B 权重 | `~/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt` | 1.6GB | 首选档 |
| DINOv2-S teacher | `~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth` | 88MB | 挂 world 头的都要（含 λ=0） |
| Wan2.2 VAE | `~/.cache/huggingface/wan22_vae/diffusion_pytorch_model.safetensors` | 2.8GB | 只有 dual 那条废弃分支要 |

联网能自己下；换机器直接拷这几个目录也行。
加载成功的日志长这样，**看不到就是没生效，结果不作数**：

```
V-JEPA2 teacher vjepa2_1_vitb: loaded ... (0 unused keys, 0 pos_embed skipped)
Future-DINO Head enabled: target_encoder=dinov2_vits14 (dim=384, frozen), weight=0.0, ...
```

---

## 三、数据要求

### 3.1 每个 episode 是一个 hdf5

必需的三个 key（`T` = 帧数）：

| key | 形状 | 说明 |
|---|---|---|
| `action` | `(T, 128)` float32 | 128 维人体中心表示，**归一化前的原始值** |
| `observation.state` | `(T, 128)` float32 | 同上 |
| `observation.image.<cam>` | 见下 | 相机名走 `top` → 没有就退 `left` → 再退 `right` |

必需的 attr：**`embodiment`**（如 `human` / `dex5`）。
没有这个 attr 时代码按文件路径里有没有 `human` 猜，**别依赖这个猜测**——
归一化统计量是**按 embodiment 分开算**的，猜错等于把两种数据混着归一化，全盘偏掉。

机器人遥操数据还应写入数值型 `fps` attr。训练 dataloader 默认把明确标记为
`fps > 30` 的机器人 episode 在线映射到 30fps，图像、state、action、history 和 future
target 共用同一条时间轴。旧数据没有 `fps` 时按30fps处理；只在复现旧的高帧率实验时传
`--no_robot_fps_resample`。离线转换仍可使用 `scripts/preprocess/resample_sonic.py`，其目标帧率默认也是30。

图像有两种存法，代码都认：

- **未压缩**：`(T, 240, 320, 3)` uint8（现有人类数据是这种）
- **JPEG 压缩**：`(T, 变长)` uint8，读的时候 `cv2.imdecode` + `BGR2RGB`（现有机器人数据是这种，省 3 倍空间）

样本在 `example_data/`，一条人类一条机器人，附 [example_data/README.md](./example_data/README.md)
说明两种格式的三个差别。128 维怎么切（哪几维是头/手腕/腰）看
[hdt/constants.py](./hdt/constants.py) 和 `data/plot_keypoints_ys.py::_action_to_eval_joints`。

### 3.2 `state[98:128]` 这一块

只有机器人数据有内容（`robot_q_current[0:26]`：根位置 3 + 根四元数 4 + 29 个关节角的前 19 个），
人类数据整块是 0。如果你的新数据里机器人侧凑不齐这 30 维，用第一节的「部署省事版」，
把它整块置 0 去训 —— 实测跟给全状态**打平**（不是更好，是不掉）。

### 3.3 转换脚本

现成的转换器（都输出上面这个格式）：

| 输入 | 脚本 |
|---|---|
| ARKit JSON（iPhone 采的人类数据） | `arkit_to_128.py` |
| Mocap | `mocap_to_128.py`、`data/batch_convert_mocap.py` |
| mp4 | `mp4_to_hdf5.py` |
| ZED 相机原始录制 | `human_data/` 采集 + `cet/post_process_zed.py` |

上游 UnifoLM_WBT 那批机器人数据的转换结果在 `data/convert_UnifoLM_WBT*/`，
下载脚本是 `scripts/d.py`。转完建议先跑一遍
`python data/plot_keypoints_ys.py` 目视检查，再进训练。

### 3.4 数据量参考

现在这套结论是在 **人类 93 条 + 机器人 93 条（1:1），每条约 200 帧** 上跑出来的，
10 万步 batch 64 收敛。验证集人类 11 条、机器人 10 条。

**10 集验证集只能分辨约 2mm 的差**（配对口径下）。新任务如果要比较配置，验证集尽量多划一些，
否则跑再多 run 也说不清。

### 3.5 人机比例

`ab0`（只有机器人 93 条）→ `ab1`（加人类 93 条）相对退化 17.3% → 11.9%，
**−5.4pt，是全周最大的单笔收益**。有人类数据就一定要加，比例按 1:1 配。

---

## 四、数据集 JSON

训练用一个 JSON 指定 train/val 目录：

```json
{
  "train": [
    { "dataset_path": "/home/aigc/human_policy/data/human_train",    "type": "human" },
    { "dataset_path": "/home/aigc/human_policy/data/dex5_train_s93", "type": "robot" }
  ],
  "val": [
    { "dataset_path": "/home/aigc/human_policy/data/human_val", "type": "human" },
    { "dataset_path": "/home/aigc/human_policy/data/dex5_val",  "type": "robot" }
  ]
}
```

- `dataset_path` 写绝对路径最省事（相对路径会跟 `--base_dir` 拼接）
- 可选 `start_idx` / `end_idx` 截取，或 `file_list` 显式给文件名列表
- **`type` 字段代码根本不读**，只是给人看的；真正决定归一化分组的是 hdf5 里的 `embodiment` attr
- 目录里的文件按**数字**排序（`wholebody-2` 在 `wholebody-10` 前面），不是字典序 —— 这里以前踩过一次坑，
  字典序会造出有偏的 train/val 划分

现成的参考：`pickup_pillow_mixed_1to1.json`（混合）、`pillow_robot_s93.json`（纯机器人）。

---

## 五、训练

### 5.1 首选：V-JEPA 主干

```bash
cd /home/aigc/human_policy
CUDA_VISIBLE_DEVICES=0 /home/aigc/miniconda/envs/human_policy/bin/python hdt/main.py \
  --batch_size 64 \
  --num_epochs 100000 \
  --lr 1e-5 \
  --chunk_size 100 \
  --exptid my_task_vjepa \
  --dataset_json_path /home/aigc/human_policy/my_task.json \
  --model_cfg_path hdt/configs/models/act_input_vjepa.yaml \
  --base_dir /home/aigc/human_policy/data \
  --cond_mask_prob 0.0 \
  --no_wandb \
  --use_future_dino_head \
  --future_dino_weight 0.0 \
  --future_dino_warmup_steps 0 \
  --future_dino_horizon 45 \
  --future_dino_ablation none
```

### 5.2 次选：resnet18 + Future-DINO 头（ab4 配方）

把上面两行换掉：

```
  --model_cfg_path hdt/configs/models/act_with_future_dino.yaml \
  --future_dino_weight 1.0 \
```

### 5.3 部署省事版（遮状态）

在 5.2 的基础上换 cfg 并加一行：

```
  --model_cfg_path hdt/configs/models/act_with_future_dino_nostate.yaml \
  --zero_state_dims 98:128 \
```

日志里必须出现这一行，**看不到就是没生效**：

```
[state ablation] normalized qpos[98:128] forced to 0 (30/128 dims), matching the human episodes
```

现成的封装脚本：`scripts/train/round4_vjepa.sh`（V-JEPA 各臂）、
`scripts/train/ablation_8gpu.sh`（8 卡消融）、`scripts/train/run_zero_state.sh`（遮状态）。
照着改 `RUNS=()` 里的数据 JSON 和 exptid 就能跑新任务。

### 5.4 参数说明（只列会动的）

| 参数 | 现在用的值 | 说明 |
|---|---|---|
| `--num_epochs` | 100000 | **其实是 iteration 数，不是 epoch**。到这个步数就停 |
| `--batch_size` | 64 | 单张 4090 24GB 跑得下（含 V-JEPA 主干） |
| `--chunk_size` | 100 | action chunk 长度 |
| `--lr` | 1e-5 | 常数 LR；**注意用 `--load_pretrained_path` 时这个值会被忽略**，见 5.6 |
| `--cond_mask_prob` | 0.0 | 语言条件 dropout。最近所有 run 都是 0.0（默认值是 0.1） |
| `--human_slow_down_factor` | 4（默认） | 对人类 action 轨迹插值放慢，独立于机器人 30fps 转换。传 `0` 按两边平均长度自动算 |
| `--future_dino_horizon` | 45 | world 头的未来偏移，按转换后的逻辑时间轴计帧；人类 episode 会按 slow_down_factor 折算 |
| `--action_representation` | `original` | 可选 `relative_chunk`，HuMI 标签表示，见 §5.7 |
| `--robot_target_fps` | 30（默认） | 明确带 `fps > 30` 的机器人数据默认在线降采样；不修改原文件 |
| `--no_robot_fps_resample` | 关 | 复现旧高帧率实验时关闭机器人帧率转换 |
| `--lag_aug` | 关 | 仅训练集扰动输入腕部 state，模拟执行滞后，见 §5.7 |
| `--visual_aug` | 关 | 仅训练集增加随机裁剪、缩放和位移，见 §5.7 |
| `--seed` | 0 | 恢复训练时必须跟 ckpt 里的一致，否则直接 assert 失败 |
| `--base_dir` | `data/` | 跟 JSON 里的相对路径拼接；JSON 写绝对路径时拼接结果就是那个绝对路径。不给这个参数会退到默认的 `hdt/../data/recordings/processed`，而代码对 base_dir 有 `assert os.path.exists`，目录不存在直接挂 |
| `--no_wandb` | 开 | wandb 在代码里被硬关了（公开发布），指标只落 csv |

自动清脏起始帧默认开着（`--dirty_start_check_frames 20 --dirty_start_jump_threshold_m 0.3`）：
开头 20 帧里如果某个末端位置单帧跳超过 0.3m，就从最后一次跳之后开始取样。
启动日志会打印 `Auto-trimmed dirty starts for N/M ... episodes`。数据本来就干净的话不影响。

### 5.5 产出、恢复、磁盘

ckpt 目录是 **`<exptid>_ckpt`，相对当前工作目录**（所以要在仓库根目录启动）：

```
my_task_vjepa_ckpt/
├── policy_last.ckpt            # 最终权重，部署用这个
├── dataset_stats.pkl           # 归一化统计量，必须跟 ckpt 配套
├── policy_iter_10000_seed_0/   # 每 1 万步一存（accelerate save_state）
├── metrics.csv                 # 每 1000 步一行验证指标
└── metrics.png
```

- **自动恢复**：重跑同一条命令会自动找 `policy_iter_*` 里步数最大的接着训，不用加参数
- **磁盘**：每个 `policy_iter_*` 约 **1.15GB**（`pytorch_model.bin` 408MB + `optimizer.bin` 815MB）。
  8 个 run × 10 个存档 = 92GB，2026-08-16 就是这么把根分区撑爆、8 个 run 全挂的。
  长跑请挂看门狗：`scripts/train/prune_ckpts.sh 2 120 'my_task*_ckpt'`（只留最新 2 个）
- **大盘子**：本机 ckpt 都软链到 `/mnt/nvme0n1/human_policy_ckpt/`（根分区只有 141G 空余），
  照 `round4_vjepa.sh` 里 `mkdir + ln -s` 的做法来
- **多卡**：单个 run 就是单卡（8 张卡的用法是同时跑 8 个不同配置，不是数据并行）。
  真要数据并行走 `accelerate launch`，见根 README

### 5.6 从已有 ckpt 微调（新任务的另一条路）

```bash
... --exptid my_new_task_ft --load_pretrained_path BEST_CKPT_vjepa_w0/policy_last.ckpt
```

三个坑：

1. **`--lr` 会被忽略**。给了 `--load_pretrained_path` 就会切到 finetune 调度器，
   写死 `max_lr=1e-6`、1000 步 warmup（代码里 `config.get('finetune_lr', 1e-6)`，
   而 config 里根本没有这个 key）。要改就改 `hdt/main.py:558`
2. **加载是 `strict=False`**，架构对不上会**静默跳过**不匹配的权重。
   微调前确认 model cfg 跟原 ckpt 训练时的那份一致
3. **exptid 要用新的**，否则会被当成「恢复训练」去读旧 ckpt 目录

坦白说：最近所有出结论的 run 都是**从零训 10 万步**，微调这条路最近没验过
（`scripts/finetune/` 是 5 月的老脚本，路径都是旧机器的）。新任务如果数据够（每边 ~90 条），
建议直接从零训。

---

### 5.7 HuMI与lag/vis增强

HuMI 是可选的 action 表示；lag 和 vis 是可独立启用的训练增强。它们可与 §5.1–5.3
的配方组合使用，目前不能把实现验证当成训练收益结论。

在原训练命令末尾加入以下参数，即可同时开启：

```bash
--action_representation relative_chunk --lag_aug --visual_aug
```

为方便区分模型和归一化统计量，使用新的实验名，例如
`--exptid fruitpick_humi_lagvis`。也可以只加入其中一项做消融。
机器人高帧率数据默认转换到 30fps，无需额外开关；数据的 `fps` attr 要准确（见 §3.1）。
已启动的训练进程不会自动应用后续代码修改。

**HuMI relative chunk 的定义。** 原 HDF5 继续存绝对 pose/keypoint，dataloader
读取后在内存中构造整段 future 相对于同一个当前时刻 $t$ 的标签：

$$
\Delta p_{t+i}=p_{t+i}-p_t,\qquad
\Delta R_{t+i}=R_t^{-1}R_{t+i},\qquad i=1\ldots H.
$$

position 在 dataset 坐标系中直接相减，rotation 使用相对旋转；这里按上面的两条公式实现。
每个 future step 都相对同一 anchor，不做相邻帧差分。两侧手指 keypoint 块也用
`future - current`。转换覆盖 head、左右 wrist、neck 和 waist 五个 pose 块；
机器人构型尾部 `action[98:128]` 保持绝对值。

输入仍为 image + current KP，输出为 `[B, H, 128]`，ACT 结构及现有 L1 / wrist weighting
保持一致。人类轨迹继续按 `human_slow_down_factor` 插值放慢，HuMI chunk step $i$
对应当前帧之后 $i/\text{factor}$ 个源帧；机器人 step $i$ 对应 30fps 逻辑时间轴之后 $i$ 帧。

HuMI 也可通过 model yaml 配置；CLI 优先于 YAML，未指定时使用 `original`：

```yaml
data:
  action_representation: relative_chunk
```

每次新训练按转换后的全部有效 future step 单独统计 `action_mean/action_std`，保存到
该实验的 `dataset_stats.pkl`，并记录 `action_representation`、`robot_target_fps`。
HuMI 模型必须配套使用这份 stats，不能复用绝对 action 的统计量。

**lag 和 vis 的行为。** `--lag_aug` 复用腕部回溯和偏移实现，只扰动模型看到的当前 state；
HuMI 标签以未扰动的当前演示 pose 为 anchor。默认以 0.5 概率回溯 2–8cm（搜索 450 个源帧），
以 0.25 概率增加随机位置偏移（每轴 sigma 2cm，范数上限 8cm），
以 0.25 概率让指尖采用不同的滞后时刻。

`--visual_aug` 加入 RandomResizedCrop，默认保留面积 0.75–1.0、原始宽高比上下约 10% 的扰动，
输出分辨率不变。同一个 sample 的 current/history/future 图像共享裁剪。
两项增强仅在训练集启用，验证集不做这些扰动。

现有实现的组合限制：lag 不能与 observation history 同时开启；vis 不能与预计算
Future-Flow target 同时开启，因为裁剪后图像与固定 flow anchor 网格会不一致。
HuMI + lag + vis 三项可同时使用。

**评测和部署。** HuMI 输出先反归一化，再用 anchor 恢复绝对 KP：

$$
\hat p_{t+i}=p_{\rm anchor}+\Delta\hat p_{t+i},\qquad
\hat R_{t+i}=R_{\rm anchor}\Delta\hat R_{t+i}.
$$

`data/eval_mpjpe_batch.py` 和 `scripts/eval/perturb_eval.py` 根据 stats 元数据自动 inverse，
再在绝对坐标系计算 MPJPE。HuMI query 0 是未来第一步，机器人按下一逻辑帧对齐；
当前评测脚本人类 query 0 按下一源帧对齐，与训练中的插值时间存在差异，比较人类结果时需注意。
训练日志中的 `val/l1` 仍是归一化标签空间的 loss，不直接作为跨表示的厘米级误差比较。

`hdt/online_hat_service.py` 根据 stats 自动 inverse。第一个 chunk 用机器人真实 pose
启动，后续按相邻 RobotState 的 source timestamp 差值，在上一 chunk 中选择对应的 scheduled pose
做 anchor；query 0 对应未来第一步，因此 scheduled row 使用逻辑帧偏移减 1。时间超出上一 chunk
范围时回退到当前实测 pose。可通过 `--scheduled-anchor-max-error-m` 配置位置误差重锚阈值，
默认 0 表示关闭该阈值。实际部署效果仍需真机验证；详细参数见
[docs/DEPLOYMENT_PARAMETERS.md](./docs/DEPLOYMENT_PARAMETERS.md)。

---

## 六、评估

### 6.1 别只看 val loss

8 个不同配置的 `val/l1` **全是 0.096**，干净画面上分不出任何高低。
判决靠扰动套件：只扰动输入图像（GT 动作不动），测 MPJPE 退化。

```bash
python scripts/eval/perturb_eval.py \
  --name my_task_vjepa \
  --ckpt my_task_vjepa_ckpt \
  --cfg hdt/configs/models/act_input_vjepa.yaml \
  --val_dir data/dex5_val \
  --out /tmp/perturb_my_task.json
```

11 个轴：clean / light_dim / light_bright / contrast / obj_appear / background /
noise / blur / camera / occlusion / compound。
`--cfg` **必须是这个 ckpt 训练时用的那一份**，否则 `strict=False` 会静默丢掉权重和 buffer，
结果看起来像大幅退化，其实是配置错了。

### 6.2 比较两个 ckpt 要用配对口径

```bash
python scripts/eval/perturb_paired.py /tmp/perturb_A.per_ep.json /tmp/perturb_B.per_ep.json
```

10 集的原始 SEM 是 ±4.6mm（均值才 42mm），因为难的集对谁都难 —— 那部分方差是共模的。
所有 ckpt 看的是同样 10 集、同样帧、同样噪声种子，所以要看**逐集之差**，误差棒小一个数量级。
判读：10/10 或 9/10 同号且 |mean| > 2×SEM 才是真的差；接近 5/10 就是噪声。

**必须同时看绝对 mm 和相对退化。** 只看比值会被分母骗：负对照 `wv3`、`r4_vjepa_shuf` 两次
「相对退化更好」都是因为自己的 clean 更差，按绝对 mm 它们是输的。这个坑踩过两次。

批量跑多个 ckpt：`scripts/eval/run_perturb_errbars.sh`。

### 6.3 常规 MPJPE

```bash
python data/eval_mpjpe_batch.py \
  --gt-dir data/dex5_val \
  --policy-ckpt my_task_vjepa_ckpt/policy_last.ckpt \
  --policy-config-yaml hdt/configs/models/act_input_vjepa.yaml \
  --norm-stats my_task_vjepa_ckpt/dataset_stats.pkl \
  --out-json my_task_vjepa_ckpt/mpjpe_val.json
```

不加 `--out-json` 就只打印在终端，事后什么都不剩。

单条 episode 的可视化 / 出视频：`data/plot_keypoints_ys.py`。

### 6.4 成功率

**只能上真机测。** 仿真里的渲染差距比任何一个扰动轴都大，MPJPE 和扰动套件都只是代理指标。

---

## 七、新任务的完整流程（照着做）

1. 把新数据转成 §3 的格式，**写对 `embodiment` attr**
2. 划 train/val 目录。val 尽量 > 10 集
3. 人类数据和机器人数据配到 **1:1**（数量对齐，多的那边抽样）
4. 写一个 dataset JSON（§4）
5. 用 §5.1 的命令训 10 万步（V-JEPA 档约 14h/卡）
6. 先看 `metrics.png` 的 `val/l1` 有没有正常下降（参考值：1 万步 ~0.14，10 万步更低）
7. 跑 §6.1 扰动套件，跟 `BEST_CKPT_vjepa_w0/perturb_result.json` 对照量级
8. 真机测成功率
9. 交付时把 `policy_last.ckpt` + `dataset_stats.pkl` + **训练时那份 model yaml** 一起打包，
   照 `BEST_CKPT_vjepa_w0/README.md` 的格式写清楚依赖

---

## 八、踩坑清单

1. **model yaml 必须跟 ckpt 配套。** 加载全程 `strict=False`，配错了不会报错，只会静默丢权重/buffer。
   `zero_state_dims` 的 mask 是个 ckpt buffer —— 用错 yaml 会把它丢掉，模型被喂进从没见过的完整状态
2. **`dataset_stats.pkl` 必须跟 ckpt 配套。** 换一个数字全错。它是训练集算出来的，按 embodiment 分组
3. **`--num_epochs` 是步数不是 epoch**
4. **磁盘**：不清 ckpt 会把根分区撑爆，见 §5.5
5. **`type` 字段是摆设**，真正生效的是 hdf5 的 `embodiment` attr
6. **相机名**：找不到 `top` 会自动退 `left`/`right`，三个都没有直接 KeyError
7. **恢复训练的 seed 必须一致**，否则 assert 失败
8. **比较两个配置时**：先问差多少、10 集能不能分辨（约 2mm），再问显著不显著（§6.2）
9. **wandb 在代码里被硬关**（`mode = "disabled"`），别指望它，看 `metrics.csv`

---

## 九、别再跑的分支（已经证伪，省得重复）

| 分支 | 结论 |
|---|---|
| world 头 λ=0.3（ab2） | 权重太轻，等于没开 |
| DINOv2 **ViT-B** teacher（ab5） | 13.5%，明显差于 ViT-S 的 9.5% |
| LayerNorm target 归一化（ab6） | 10.6%，差于 l2 |
| 只用机器人数据（ab0） | 17.3%，差得最多 |
| V-JEPA 主干上挂 world 头（λ=1 vs λ=0） | 11 个轴无一显著，**纯开销** |
| V-JEPA 主干 + 遮状态（r3_vjepa_nostate） | 全轴略差、无一显著，**两个改进不叠加** |
| DINO + Wan VAE 双 target（wv1/wv2/dual） | resnet18 上收益被自己的负对照反超；30h/run，最贵 |
| 打乱时序的负对照（ab7 / wv3 / r4_vjepa_shuf） | 「相对退化更好」是分母造的假象，绝对 mm 输 |

细节和完整实验记录：
[docs/NEXT_RUNS.md](./docs/NEXT_RUNS.md)（决策与判据）、
[docs/FUTURE_DINO_CHANGES.md](./docs/FUTURE_DINO_CHANGES.md)（实现与扰动套件）、
各 `BEST_CKPT*/README.md`（单个 ckpt 的交付说明）。

> 注：根目录的 `TRAINING_GUIDE.md` 是 2026-05 的旧文档，路径和默认值都已过时，以本文为准。
