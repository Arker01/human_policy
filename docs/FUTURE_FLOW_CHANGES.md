# EgoWAM 3D flow world head：改了什么

配套 `docs/FUTURE_DINO_CHANGES.md`。那份记的是第一个 world 头（DINO），这份记第三个（3D flow）。
第二个（Wan VAE）在 `docs/FUTURE_VAE_CHANGES.md`。

**为什么做**：clean 轴一整周没动过。排除 ab0（只有机器人数据，49.1mm），22 个 run 的 clean MPJPE
全落在 40.6–44.8mm，连 V-JEPA 那 1.7mm 都不显著（配对 −1.3±1.8，5/10）。我们推动的全是 OOD 轴。
[EgoWAM](https://arxiv.org/abs/2607.08436) 做的正是这个受控实验——固定 trunk / action head / 数据配比，
**只换 world target**——结论是 DINO 只涨 OOD（最多 4×），**3D flow 涨 in-domain 20–30%**，两者互补。
我们自己的数据完全对得上前半句：ab1→ab4 赚的 2.4pt 全在 background/compound，clean 纹丝不动。
**所以这是唯一瞄准 clean 轴的改动。**

主干选 resnet18 不选 V-JEPA：resnet18 是唯一证明过 world 头有空间的主干（2.4pt），
而且跟 ab1(λ=0)/ab4(DINO) 严格可比，正好复刻 EgoWAM 的受控设计。
`r4_vjepa_w0` 已经证明 V-JEPA 上挂 world 头全轴打平，没有 headroom。

---

## 原则：全部是加法，默认关

跟 `future_vae` 一条路：所有新代码都在 `enabled: false` 时不执行，老 run 逐比特不变。
下面每一处都标了"老 run 走哪条路"。

---

## 1. 新增文件

### `hdt/detr/models/future_flow.py`

`FutureFlowHead` + `FutureFlowLoss` + `build_2d_sincos_pos_embed`。

**头**：1200 个**可学 anchor query**（叠 30×40 二维正弦位置编码）cross-attend 到
`hat_memory [B,100,512]`，4 层 TransformerDecoder，输出 `[B,1200,300]` → reshape `[B,100,1200,3]`。

- **为什么用可学 query 而不是图像 token**（这是跟 `FutureDINOHead` 的结构性差别）：
  resnet18 output stride 32，240×320 只出 8×10 = 80 个 token，凑不出 1200 个 anchor。
  可学 query 还让头跟主干分辨率解耦，换 V-JEPA（14×19）不用改一行
- **位置编码用固定正弦不用可学**：anchor 本身**就是**几何栅格，相邻 anchor 带相邻 3D 点，
  固定表把这个度量结构白送给 decoder，不用它自己从头发现
- **output_proj 零初始化**：场景大部分静态、target 大部分接近 0 米，零初始化让头一开始就
  对背景 anchor 给出正确答案，同时共享主干上的初始 world 梯度为 0 —— 就是这个性质让 DINO 头
  退掉了 warmup
- **不上 flow matching / diffusion**：`HAT_future_DINO_head_implementation.md:232` 明写
  「不建议第一版直接照搬 diffusion / flow matching……当前目的只是验证 auxiliary world
  supervision 是否改善 HAT」。EgoWAM 的头是 flow-matching decoder，那是 v2 的事

**loss**：masked Huber，不是 cosine。`FutureDINOLoss` 用 cosine 是因为特征只有方向有意义，
而位移**幅值就是全部内容**（"枕头往左动了 4cm"），cosine 会把它扔掉。
`huber_beta=0.01` 把二次/线性拐点放在 1cm，真实操作位移大致就在这个量级：
真运动走光滑的二次段，tracker 离群点走线性段。

三个 mask 相乘：
| mask | 形状 | 作用 |
|---|---|---|
| `anchor_valid` | `[B,P]` | 位移没过阈值的 anchor 丢弃（阈值见 §3.3，不是 EgoWAM 那个 2mm/10mm）。场景大部分静态，这是唯一能让 loss 不被"预测零"主导的东西 |
| `flow_valid` | `[B,K]` | 第 k 步掉到 episode 外面，**或者超出预处理实际存的 39 步**（见 §3.4） |
| `future_valid` | `[B]` | 样本的未来帧被 clamp 过；DINO 路已经在 `conditioning_dict` 里传这个 flag |

诊断量 `flow_valid_frac`：**掉到 ~0 就说明移动阈值对这批数据不对，头在拿空 mask 训练。**

### `hdt/configs/models/act_with_future_flow.yaml`

照 `act_with_future_dino.yaml` 写，resnet18 主干 + `future_flow` 段。
`weight: 1.0` 跟 ab4/r4 一致，这样跟那些 run 的差别只归因于 target 本身。

### `scripts/preprocess/flow_target.py`

离线造 target，跑在独立的 `track4world` conda env 里。详细设计和三个实测结论见 §3。

---

## 2. 改动的文件

### `hdt/detr/models/detr_vae.py`

| 位置 | 改动 | 老 run 走哪条路 |
|---|---|---|
| import | `from .future_flow import FutureFlowHead, FutureFlowLoss` | 纯 import |
| `DETRVAE.__init__` 签名 | 加 `future_flow_config=None`（插在 `future_vae_config` 后、`zero_state_dims` 前，都是关键字参数所以位置无关） | 默认 None |
| `__init__` 尾部 | 新增 `future_flow` 初始化块 | `enabled` 缺省 false → 五个 `self.use_future_flow_*` 全是关/0 |
| 新方法 `_future_flow_forward` | 新增 | 不被调用 |
| `forward()` 尾部 | 新增 flow 块，把 `flow_*` 键折进同一个返回 dict | `use_future_flow_head` false → 跳过 |
| `build_ACT_model` | 读 `args.future_flow_config` 并传下去 | `getattr(..., None)` |

**两个刻意的设计**：

1. **flow 块放在 `use_future_dino_head` 那个 if 的外面。** DINO 和 VAE 头都需要 dataloader
   给的未来**帧**和一个冻结 teacher；flow 头两个都不需要（target 离线算好了），
   所以它能完全独立启用 —— `r5_flow` 臂就是一个 DINO 头都没挂的 flow-only 臂。
   相应地这里**没有** `assert self.use_future_dino_head`（`future_vae` 有）。

2. **flow-only 时伪造一个全零的 DINO dict。** `policy.py:85-87` 在 dict 非 None 之后
   无条件读 `cosine_loss`/`huber_loss`/`loss`。为了**不动那段老代码**，flow-only 臂给它递一个
   `effective_weight=0.0` 的零三元组：贡献恰好是 0，而 `future_dino_*` 三个标量在日志里记 0
   也是老实的读数 —— 这个臂确实没有 DINO 头。

3. **没有 `'current'` ablation**（DINO/VAE 都有）。位移的"当前值"恒等于 0，而零初始化的输出层
   本来就坐在那个答案上，这个臂什么都测不出来。`'shuffled'` 才是这里有意义的负对照。

4. **`'shuffled'` 用同一个 permutation 同时打乱 target 和它的两个 mask。**
   两个 mask 描述的是**那个** flow field 哪些 anchor 真动了、伸到 episode 多远，
   落在原处就会把样本 i 的 mask 配到样本 j 的 field 上 —— 那是个**自相矛盾**的 target，
   不是"错的"target，就不是我们要的对照了。`future_valid` 不打乱：它是 DINO 路的帧 clamp flag，
   不属于这个 target。

### `hdt/policy.py`

在 VAE 块后面加一个平行的 flow 块（`if 'flow_loss' in future_dino_loss_dict:`），
记 5 个标量并把 `flow_weight * flow_loss` 加进总 loss。**老代码一行没动。**

### `hdt/data_utils_hdt.py`

| 位置 | 改动 | 老 run 走哪条路 |
|---|---|---|
| 模块级 | 新增 `flow_target_filename()` | 纯函数 |
| `EpisodicDataset.__init__` | 加 4 个参数 + 初始化块 | `flow_target_enabled` 默认 False |
| 新方法 `_read_flow_target` | 新增 | 不被调用 |
| `read_one` 尾部 | `if self.flow_target_enabled:` 塞 3 个键进 `conditioning_dict` | false → 跳过 |
| `collate_fn` | 新增 flow 三键的 stack | `'flow_target' in conditioning_list[0]` false → 跳过 |
| `load_data` | 加 4 个参数并透传 | 默认值等于关 |

**三个容易踩的点**：

1. **`collate_fn` 是白名单。** 它按 `KEYWORDS_LIST` 逐键搬，没列进去的键**静默丢掉**——
   这正是一个 flow 臂能"训练时 world loss 是空的、却一个报错都没有"的路径。所以必须显式加。

2. **人类 episode 的 horizon 要换算。** h5 按**原始帧偏移** 1..R 存，头按**chunk step** 1..K 索引，
   两者对人类不是一回事：人类 chunk 被 `slow_down_factor` 时间压缩过，chunk step k 落在
   原始偏移 k/factor。跟 DINO 路对它那个单一 horizon 做的修正（`data_utils_hdt.py:486`）同理，
   只是一次对 K 步全做。

3. **`flow_target` 走 `conditioning_dict` 不走返回元组。** `read_one` 的 5 元组/6 元组契约
   因此完全不变，flow-only 臂仍然走原来的 5 元组那条路。

### `hdt/main.py`

| 位置 | 改动 |
|---|---|
| CLI 覆写段 | 4 个 `--future_flow_*` 的 yaml 覆写（照 `future_vae` 的写法） |
| config 解析段 | `future_flow_cfg` / `_enabled` / `_dir` / `_grid_hw` / `_horizon` + 启用时打印 |
| `policy_config` | `if future_flow_enabled: policy_config['future_flow_config'] = ...` |
| `load_data` 调用 | 透传 4 个 flow 参数 |
| argparse | `--use_future_flow_head` / `--future_flow_weight` / `--future_flow_ablation` / `--future_flow_dir` |

两个启用时的 assert：`target_dir` 必须存在；`horizon` 必须等于 `chunk_size`
（头 cross-attend 到 `hat_memory`，它的 token k **就是** chunk step k，两条轨迹是同一套索引）。

---

## 3. Target 定义

想要的东西一句话：`target[t,k,g] = R_t^T · (P_g(t+k) − P_g(t))`，即 anchor g 这个场景点
从帧 t 到帧 t+k 的 3D 位移，**表达在帧 t 的相机系里** → 静态背景 ≈ 0，只有真在动的东西
（手、枕头、夹爪）带信号。这就是 EgoWAM 的 desideratum D3。

Anchor = 帧 t 上固定的 30×40 像素栅格（stride 8）取格心，1200 个（EgoWAM 是 28×40=1120，
同 stride）。人类 clip 掐头去尾各 20 帧。这些都照 EgoWAM。

**跟 EgoWAM 的两处结构性偏离**（计划里就写了的）：

1. **target 离线算，不用 train 时的 teacher。** dense 3D tracker 在训练循环里跑不动，
   而且跟 DINO/VAE 特征不同，flow target **不依赖任何我们在训的权重**，缓存起来零损失。
2. **头直接回归轨迹，不用 flow-matching decoder。** 理由见 §1。

### 3.1 三条实测结论，每一条都推翻了计划里的一个假设

计划是按 Track4World 的 README 和 demo.py 写的，**跑起来量了之后三条都不对**。
每一条如果照原假设写下去，都会静默产出一个错的 target（不报错）：

| 假设 | 实测 | 怎么量的 |
|---|---|---|
| `flow_2d/flow_3d` 是**相邻帧** pairwise flow（demo.py 的变量名叫 `all_pairwise_flows_2d`） | 是 **query→帧 i 的长程 flow** | `mean\|flow_2d[i] − identity\|` = 0.01, 0.43, 1.65, 2.20 … 9.35, 9.38 px：单调上升然后饱和，相邻 pair 不会是这个形状 |
| `flow_3d` 是位移 | 是**绝对 3D 位置**（~1.85m，等于场景深度），而且在**各自帧自己的相机系**里 | `flow_3d[0]` vs `points[0]` 差 13.15mm，vs `world_points[0]` 差 84.58mm |
| 用模型输出的 `camera_poses` 把位置转回帧 t 系 | **精度不够**：这样稳定完，静态背景在 k=20 还有 16mm 中位数，而且带明显左右梯度 | 直接量静态背景残差；顺便确认了 c2w 是对的约定（16.1mm vs w2c 的 40.8mm） |

**长程 flow 这条是好消息**：窗口内不需要 chaining，(t, t+k) 直接就有对应关系，
计划里担心的"100 步链式累积漂移"在窗口内不存在。

### 3.2 ego-motion 怎么去掉（不用相机位姿）

`flow_3d` 给的是**跨帧的对应点**，那帧 t+k → 帧 t 的刚体变换就可以**直接拟合**，
不需要相信模型的位姿输出：

- **robust weighted Procrustes + IRLS**（`robust_align_batch`，sigma=1cm，6 轮，带 det 反射修正）。
  逻辑就是 D3 本身：**静态的大多数定义变换，手是离群点，它的残差正好就是我们要的位移。**
  背景中位数 16mm → **8mm**
- 但还剩一层：机器人在走的时候，背景在 k=20 还有 ~30mm，形状是一个**光滑的低频场**
  （沙发是平面 → 刚体拟合病态；视差放大深度误差）。物体运动是**局部**的，所以再做一次
  **robust 二次空间去趋势**（`detrend`，基 `[1,x,y,x²,y²,xy]`，同样 IRLS），减掉这个场。
  t=110 的保留比例 0.23 → 0.16，mask 从"糊满整块地板"变成"贴着夹爪和枕头"
- **去趋势是加在 target 上的，不只是加在 mask 上。** 理由：EgoWAM 有 Aria 的真位姿，
  我们没有，减掉这个残差场是让我们**更靠近**它那个理想 target（纯物体运动），不是更远
- 试过一个假设：残差梯度是不是来自"两次拟合经同一个 gauge 复合"。改成 a→a+k 直接拟合后
  数字基本没变（0.257 vs 0.259）→ **不是复合造成的**，是大视角变化下 `flow_3d` 本身的性质。
  直接拟合还是留下了，因为它少一层、更简单

### 3.3 移动阈值：按噪声底定，不用 EgoWAM 的 2mm/10mm

EgoWAM 的 2mm 在我们这份素材上**不可用**：噪声底就在 1cm 量级，2mm 会把 ~99% 的 anchor
标成"在动"，mask 等于没有。分辨率扫过 240×320 / 448×336 / 640×480（k=10 时 7.6 / 7.3 / 9.6mm）
→ 这个底是模型在这份素材上的**固有精度**，不是分辨率不够。

**固定绝对阈值也不行**：机器人背景 p50 从 k=1 的 2.1mm 涨到 k=39 的 19.1mm，
`frac>25mm` 在 k=39 到了 0.439。但信噪比是**随 k 变好**的（p90/p50 从 3.1× 涨到 5.7×），
所以缩短 horizon 是错的解法。**正确的解法是相对每一步自己的背景中位数去切**：

- `MOVE_THRESHOLD_M = 0.025`，含义是"**高出该 (t,k) 步自己背景中位数 25mm**"
- 人机**同一个值**：这个限制来自 tracker 精度，跟哪个本体在动无关
- 效果：`frac` 0.439 → 0.257；两条 example episode 上 kept 与 dropped 的 `|d|` 差 **8×**
  （robot kept 40mm vs dropped 5.1mm）

### 3.4 kmax = 39，第 40–100 步被 mask（不是 clamp）

一个窗口 `L = stride + kmax = 16 + 39 = 55` 帧，240×320 下峰值显存 ~10–24GB，装得下；
101 帧装不下。窗口在帧 q 上 query，只提供帧 q 像素的轨迹，所以 `L = stride + kmax`
保证 `[q, q+stride)` 里每个 anchor 帧都从**单个窗口**拿到完整 kmax 步 ——
**窗口之间不 chaining，所以没有跨窗口累积漂移**，代价是 overlap 区的重复计算。

h5 里只有 R=39 个原始偏移，头要预测 100 步，所以 dataloader 里 **step > R 的直接 mask 掉**：

- 不 clamp。clamp 会静默声称"第 39 步以后什么都不动"，那是个**错的** target，
  比一个**缺失的** target 坏
- **人类一点不损失**：`slow_down_factor=4`，chunk step 100 落在原始偏移 25 < 39
- 机器人 39/100 有效，人类 100/100 有效（实测打印就是这两个数）

### 3.5 输出

每 episode 一个 h5：`flow_target` fp16 `[T,39,1200,3]`、`flow_valid` bool `[T,39]`、
`anchor_valid` bool `[T,1200]`，加 attrs（source / embodiment / grid_hw / image_hw /
horizon_raw_offsets / move_threshold_m / trim / units）。h5 的行 t **就是**原始帧 t
（人类 trim 掉的头尾重新 pad 回去），所以 dataloader 不需要做任何索引换算。
约 60MB/episode，207 个 episode ≈ 12GB，落 `/mnt/nvme0n1`。

机器人 JPEG 解出来是 480×640，**预处理里 resize 到 240×320** 再跑 tracker，
这样"anchor g"在人机两端指的是同一个像素。

`--metric_scale` 只在 DA3 backbone 下支持（README:200），MoGe/Pi3 出的是相对尺度 →
必须用 DA3，否则米制阈值没意义、人机两端也对不齐。

---

## 4. 环境

`track4world` 独立 conda env（`/home/aigc/Track4World/install_min.sh`），**只给离线预处理用，
训练环境一行不改**。比 README 的安装窄了三块：

- **不装 Grounded-SAM-2 / SAM2 权重**：README:152 说分割那步是「for better visualization,
  especially to clearly separate foreground and background objects」。我们要的是原始 flow，不是图
- **不装 gradio / viser / open3d / pycolmap / evo**：demo UI + 位姿 benchmark
- **只下 DA3 权重**（metric scale 是 DA3 独有，README:200）

用 `-c conda-forge --override-channels` 建 env：defaults 频道要交互式接受 ToS，
为一个一次性 env 去改全局 conda 配置不值得。

两个装的时候踩到的：`moviepy==1.0.3` **不是可选的**（DA3 的 `api.py` 模块级
`import moviepy.editor`）；`utils3d` 要当**目录**clone 进去，repo 本身就是那个 package。

---

## 5. 判据（先写死，免得事后找解释）

基线 = **ab1**（resnet18 λ=0，`BEST_CKPT_baseline`，clean 44.78）和 **ab4**（DINO，clean 43.49）。
全部用 `scripts/eval/perturb_paired.py` 的配对口径，**同时看绝对 mm 和相对退化**——
比值指标被坑过两次（wv3、r4_vjepa_shuf），分母变差会让比值假性好看。

- **主判据在 clean 轴**（EgoWAM 声称 flow 起作用的地方）：`r5_flow − ab1` 的 clean 要
  **< −2mm 且配对 8/10 同号**才算真的。EgoWAM 报 20–30%，对应 44.8 → 31–36mm；
  我们只要拿到 −2mm 就已经是全周 clean 轴第一次显著移动
- **负对照**：`r5_flow_shuf − r5_flow` 的 clean 打平 → 收益不来自"预测运动"，
  跟 ab7/wv3 一个性质，要在文档里照实写
- **互补性**：`r5_flow_dino` 要同时在 clean（对 ab4）和 compound（对 r5_flow）上赢，
  才算 EgoWAM 的互补成立
- 10 集配对能分辨的最小差约 **2mm**，比这更小的差加再多 run 也说不清

## 6. 要跑的臂

负对照**必须同批跑** —— 前两轮 ab7 和 wv3 的教训是，负对照晚 30h 落地不如跟正臂同时落地。

| GPU | run | 配置 | 问什么 |
|---|---|---|---|
| 0 | `r5_flow` | resnet18 + flow λ1.0，无 DINO 头 | flow 单独顶不顶用（对 ab1） |
| 1 | `r5_flow_shuf` | 同上，target+mask 批内 randperm | **负对照**：收益是不是来自"预测运动" |
| 2 | `r5_flow_dino` | resnet18 + flow λ1.0 + DINO λ1.0 | EgoWAM 说互补，验证（对 ab4） |

## 7. 状态

- [x] `future_flow.py`（头 + loss），自测通过
- [x] `detr_vae.py` / `policy.py` / `data_utils_hdt.py` / `main.py` 接线，合成数据跑通
- [x] `act_with_future_flow.yaml`
- [x] `track4world` env
- [x] `scripts/preprocess/flow_target.py`（按 §3.1 的实测重写过一遍）
- [x] 在 `example_data/` 两条上验证 target：静态背景低、夹爪/枕头/手高、量级在厘米，
      kept 与 dropped 差 8×；训练侧读回路径也验了（robot 39/100、human 100/100，
      loss 见到 `flow_valid_frac` 0.144、gt 幅值 37.5mm）
- [ ] 全量预处理（207 集）
- [ ] smoke test + 确认老 run 逐比特不变
- [ ] 三个臂
