# Future-DINO / 世界模型这条线:改了什么

一句话背景:这个仓库原本是 ACT-CVAE 策略(resnet18 看图 + qpos → 未来 100 步动作)。
学弟加了一个 "Future-DINO 头":让模型顺便预测 **未来画面的 DINO 特征**,当成额外的训练监督,
指望策略的视觉编码器因此学得更通用。改动只在这条新功能上,原有代码没动。

---

## 一、原来是什么样

| | 状态 |
|---|---|
| Future-DINO 头 | 代码在,但**从来没真正训练过** —— `train_pillow_future_ckpt` 里 78 个 `future_dino_head` 张量从 1 万步到 9 万步逐比特没变(max\|Δ\| = 0.000e+00),同期 transformer 变了 2.196e-02。等于挂了个死零件 |
| 启动脚本 | 没有。这也是它没被训起来的主要原因 |
| 训练数据 | 只有机器人 37 条(`pillow_robot.json`),93 条人类数据闲置 |
| 评测 | `data/plot_keypoints_ys.py`,每帧喂**真实** qpos,只读 `a_hat[0,0]`(下一帧),单集测试 |

## 二、现在是什么样

### 1. 头真的在训了

新建 `scripts/train/train_future_dino.sh`。跑出两个 10 万步对照:world loss 从 1.0114 掉到 0.1077,
对照组(λ=0)死死钉在 1.0203。**λ 从第 0 步就生效,不再是死零件。**

### 2. 发现原来的评测指标看不见任何东西

- 每帧喂真实姿态、只看下一帧 → 30fps 下手腕一帧才动 4mm,**"原地不动"这个假模型在 k=0 就赢**(6.97mm vs 47.78mm);
  但到部署长度 k=99 就崩(331mm vs 81mm),交叉点在 k≈8。所以那个 8mm 的漂亮数字不代表策略好。
- 学弟报的 8.13mm 那条(ep_0131)在**训练集**里;35.17mm 那条(ep_0093)在验证集里,而且是 10 条里**最好**的一条。
  批量重跑:训练集 8.60mm(37 条)/ 验证集 49.41mm(10 条),**5.7 倍泛化差距**。
- 约 5cm 的验证误差**不是新引入的退化**,这个项目一直是这个水平。

### 3. 新增两套能看见东西的评测(在 `/tmp`,要用可以挪进仓库)

- `perturb_eval.py` —— 只扰动输入图像(GT 不动),10 个轴:调暗/调亮/对比度/换色/换背景/噪声/模糊/相机平移缩放/遮挡/复合。
- `triplet_diag.py` —— 照 ST-WAM 的方法,直接量我们自己两个编码器的特征稳不稳。
- `img_ablation.py` —— 把图像换成全黑/别的集的图/冻结第一帧,看策略到底有没有在看图。

**测出来的关键结论:**

| 结论 | 证据 |
|---|---|
| **人类数据买的是鲁棒性,不是精度** | clean 下 mixed 49.97 vs robot-only 50.15(等于没差),但换背景 53.20 vs 59.25、加噪声 50.82 vs 55.12、动相机 50.06 vs 52.96。平均退化 **10.6% → 6.9%** |
| Future-DINO 有效但很弱 | 11 个扰动里 8 个 λ=0.3 更好,赢的全在视觉 shift 轴上(背景 26.5% → 22.4%),clean/光照/对比度上是 0。约 2–4% 相对,不是论文里的 4× |
| **我们的 world target 本身是短板** | 相机扰动下 DINOv2-S 同状态余弦只有 0.517,而"完全不同状态"的基线是 0.480 —— 几乎没信息量。同场景下 resnet18 是 0.818。这正是 EgoWAM 说的"DINO 特征钉在图像网格上" |
| 策略确实在看图,看的是几何不是颜色 | 换色相 0 影响;但图换成全黑 +164%、换成别集的图 +280%、冻结第一帧 +142%。mixed 明显更抗(+45%) |
| clean MPJPE 看不见 world model 收益 | 这不是"没用",是指标照不出来。所以评测口径改成扰动套件 |

### 4. 数据扩了

- 机器人 pickup pillow 从 37 条 → 可用 **609 条**(`data/convert_UnifoLM_WBT_inspire_rootfix`,已验证和老 pipeline 数值完全一致)
- 按 1:1 抽 93 条建 `data/dex5_train_s93/`(seed=0,已排除 10 条 val),配 93 条人类数据
- 新增 `pillow_robot_s93.json`、`pickup_pillow_mixed_1to1.json`
- val 沿用原来的 `dex5_val` 10 条,保证和历史数字可比

### 5. 代码改动(只动新功能)

| 文件 | 改了什么 |
|---|---|
| `hdt/detr/models/detr_vae.py` | `FutureDINOLoss` 增加 `target_norm: 'l2' \| 'layernorm'`,默认 `l2`,老 run 逐比特不变。LayerNorm 是 EgoWAM/RAE 的做法(L2 会丢掉 token 模长,而模长携带"这个 patch 有多重要") |
| `hdt/configs/models/act_with_future_dino_vitb.yaml` | 新建,只差一行:teacher 换 DINOv2 **ViT-B** |
| `hdt/configs/models/act_with_future_dino_ln.yaml` | 新建,只差一行:`target_norm: layernorm` |
| `scripts/train/train_future_dino.sh` | 新建。默认数据集改成 `pillow_robot.json`(原来指向已不存在的 `/tmp/episodes_45_95_train.json`) |
| `scripts/train/ablation_8gpu.sh` | 新建。8 卡消融,见下 |
| `scripts/train/prune_ckpts.sh` | 新建。每个 run 只留最新 2 个 checkpoint |

### 6. 换 teacher:V-JEPA 2.1(视频 teacher)

**为什么换。** 上面第 3 节量出来的:相机扰动下 DINOv2-S 同状态余弦 0.517,不同状态基线 0.480 —— 我们一直在拟合的
那个 target 空间,在最需要泛化的地方几乎没信息。DINOv2 是纯图像自监督,token 钉在图像网格上(EgoWAM 的 D3)。
V-JEPA 2 是在视频上训的,2.1 这一版专门优化 **时序一致的 dense 特征**。

> **⚠️ 先读这段再决定要不要花一张卡一天。** 这个 config 建起来时,理由里有一条是我读错论文得出的:
> **ω-0 并不预测 V-JEPA latent** —— 它回归的是冻结 **Wan** 的 latent,V-JEPA 2.1 在它那里编码的是**当前**观测
> (输入侧)。把那个输入编码器从 V-JEPA 换成 Wan,他们成功率掉 15.5 个点。所以 ω-0 的证据支持的是
> **"V-JEPA 当输入编码器"**,不是当 target。另外两条也是反向的:
> - DexWM 做过 target 编码器横评(DINOv2 / DINOv3 / Web-SSL / **V-JEPA 2** / SigLIP 2),结论 **DINOv2 综合最好**。
> - ST-WAM 是 VAE + DINO **两个都预测**,而 **VAE 是主力**(去掉 VAE:LIBERO-Plus 72.8→39.7,比纯 VAE 基线 51.5 还低;
>   去掉 DINO:→63.5)。它表里的 "DINO Future Only" 那一行 = **我们现在这套** = 最差的一行。它给 DINO 的权重只有 0.02。
>
> 再加上我们自己 8 卡消融的结果:**换更大的 DINOv2 ViT-B teacher 反而更差**(平均退化 9.5%→13.5%),
> 说明 teacher 容量不是能出钱的那根轴。代码留着,但真正值得做的是**输入侧**那个实验。

**一个变量没法固定住:** 视频 teacher 的 patch embed 会把 `tubelet`(=2)帧融成一组 token,所以**物理上不可能喂单帧**,
`clip_frames` 必须 1 → 2。这是唯一的连带改动。取 2 既是下限也是最省的:只有 1 个时间组,每相机 14×19 = **266** token,
比 DINOv2-S 的 352 还少,而 tubelet 卷积照样看到了运动。query 端是截止到 t 的 clip,target 端是截止到 t+H 的 clip,
两边同类型,头还是只建模"变化"。

**已验证:** 3 步真实训练跑通(`Clip frames / stride: 2 / 4`,teacher 加载 `ema_encoder` 0 个权重缺失,
`val/future_dino_cosine_loss` 正常);用老的 `act_with_future_dino.yaml` 回归跑 exit=0、`clip_frames=1`、
`val/l1: 0.552` `val/kl: 7.612` 与 vjepa run 完全一致 —— 说明改动只落在 teacher 这条路上,DINOv2 路径逐比特没变。

| 文件 | 改了什么 |
|---|---|
| `hdt/detr/models/detr_vae.py` | `FrozenPatchTargetEncoder` 加 `VJEPA2_MODELS` 表(2.1 ViT-B/L + 2.0 L/H,各自的 hub 入口、权重文件、checkpoint key、维度)、`VJEPA2_HUB_REF`(钉死 SHA)、`VJEPA2_WEIGHT_URL`;新增 `expects_clip` / `tubelet_size` / `_snap_to_patch_grid` / `target_resolution_hw`;`forward` 加 vjepa2 分支(吃 `[N,K,C,H,W]`,内部 permute 成 `[N,C,K,H,W]`)。`DETRVAE.__init__` 加 `future_dino_clip_frames` 和 teacher/clip 一致性断言;`DETRVAE.forward` 加 clip 模式的 `enc_input` 拼装(`[B,2K,n_cam,C,H,W]` → `[2·B·n_cam,K,C,H,W]`)。**`clip_frames==1` 时走的还是原来那条 4-D 路,分支外的代码没动** |
| `hdt/data_utils_hdt.py` | 加 `future_clip_frames` / `future_clip_stride` 两个参数和 `_read_cam_clip_ending_at`;`read_one` 在 `>1` 时读两段 clip,和当前帧一起**一次** `visual_preprocessor` + `training_transforms`(保证 teacher 和 trunk 看到同样的增广)。`load_data()` 转发这两个参数 |
| `hdt/main.py` | 从 config 解析并打印 `clip_frames` / `clip_stride`,转发给 `load_data` |
| `hdt/configs/models/act_with_future_dino_vjepa.yaml` | 新建。只差 teacher:`target_encoder: vjepa2_1_vitb`、`clip_frames: 2`、`clip_stride: 4`(30fps 下两帧差 133ms)、`target_resolution_hw: [224, 304]`、`num_patches: 1024`。其余 λ=1.0 / horizon=45 / l2 与 **ab4 完全一致**,所以 ab4 → 这个 run 是干净的单因子对比 |
| `scripts/train/train_vjepa_teacher.sh` | 新建单卡启动脚本。开跑前先检查 1.6G 权重是否下完;`<exptid>_ckpt` 预先软链到 `/mnt/nvme0n1/human_policy_ckpt/`(`main.py:367` 只在不是目录时才 mkdir,软链会被沿用),避免再写满 `/` |

**两个要知道的细节:**

1. **权重要自己下。** 上游 `src/hub/backbones.py` 里 `VJEPA_BASE_URL = "http://localhost:8300"  # for testing`,
   真 URL 被注释掉了 —— `pretrained=True` 什么都下不到。所以我们 `pretrained=False` 建模型,再自己从
   `https://dl.fbaipublicfiles.com/vjepa2/<file>.pt` 取权重。另外 checkpoint key 不统一:2.1 的 B/L 是
   `ema_encoder`,2.0 和 2.1 的 giant 是 `target_encoder`,表里逐个写死了。
   也**没有**升 `transformers`(本机 4.49,HF 的 VJEPA2 要 ≥4.52),走 `torch.hub` 就绕开了,不碰正在跑的 8 个 job 的环境。
2. **teacher 输入被 resize 了两次**(240×320 → 224×308 → 224×304)。V-JEPA 是 patch 16,308/16 = 19.25 除不尽,
   224×304 是最近的 16 的倍数,差 4 像素。宁可多一次 resize,也不动 `hdt/policy.py` 里原有的 transform。

### 7. 顺手查清的两件事(结论,没写代码)

以下是**读了正文之后**的版本。第一版是照摘要写的,有几处错,已就地改掉。

- **那几篇工作能不能在仿真里测?** 都有仿真,只是分量不同:ST-WAM 仿真是主场(LIBERO 98.7%、
  LIBERO-Plus 72.8、RoboTwin 2.0 92.8%,真机 25.8%→61.5%);**EgoWAM 附录 D 有 RoboTwin 跨机器人迁移**;
  **DexWM 有 RoboCasa,每任务 50 次**(Reach 72 / Place 28 / Grasp 58,真机另有 12 次抓取 10 成);
  只有 ω-0 的评测是纯真机(它的仿真只用来给 SONIC 回放数据做 grounding)。
  **关键不是"世界模型测不了仿真",而是"拿真机画面训的策略测不了仿真"。**
  我们仓库里有 ALOHA dm_control 仿真(`sim_test/`)和 H1+Inspire MuJoCo 仿真(`assets/h1_inspire_sim/`、
  `cet/sim_mujoco.py`),但没有 pillow 场景、没有 Dex5 手,而且真实画面 → MuJoCo 渲染的差距比我们任何一个
  扰动轴都大。**成功率这个数只能上真机**;想在仿真里做的话,只能用 `cet/mujoco_rollout_replay.py`
  回放预测出来的 chunk,看运动学上能不能执行、平不平滑。
- **ω-0 的 RTC 加不进来,但理由不是我一开始说的那个。** 我原来写"RTC 是用来藏 ~160ms 推理延迟的" —— **正文
  通篇没有任何延迟数字**,也没这么 frame。它的实际目的是**receding-horizon chunk 接缝的连贯性**:训练时把前 M 个
  (M∈[0,8]) 噪声 latent 换成干净的,loss 只算后半段;部署时用上一个 chunk 的尾巴当锚。去掉 RTC 成功率 79.1→71.8。
  加不进来的真实原因只有一条:**我们的 ACT-CVAE 头是一次前向,没有可以做 inpainting 的去噪循环**。
  而它要解决的接缝问题我们仓库本来就有解 —— `sim_test/eval_sim.py:126-232` 的 temporal ensembling,
  只是评测脚本 `plot_keypoints_ys.py` 没用(它只读 `a_hat[0,0]`)。
- **一条对我们有利的:打乱时序这个负对照,四篇论文一个都没做过。** EgoWAM 完全没有破坏 (o_t, s_{t+T}) 配对的
  控制组;ST-WAM 最接近的是"保留语义分支、去掉未来目标"(参数量对齐,62.9 vs 72.8),是**去掉**不是**打乱**。
  所以"收益到底来不来自未来"在文献里是空白,而我们测了 —— 见下面「扰动套件判决」。

## 三、正在跑的 8 卡消融

阶梯设计,每一行只比上一行改**一个**东西,所以相邻两行的差就是那个因子的作用。

| GPU | 数据 | λ | horizon | teacher | target norm | 读法 |
|---|---|---|---|---|---|---|
| 0 | robot 93 | 0.0 | – | S | l2 | 对照:没有人类数据 |
| 1 | mixed 1:1 | 0.0 | – | S | l2 | 锚点:只加人类数据 |
| 2 | mixed 1:1 | 0.3 | 16 | S | l2 | 现在的设置 |
| 3 | mixed 1:1 | 1.0 | 16 | S | l2 | 2→3 = λ |
| 4 | mixed 1:1 | 1.0 | 45 | S | l2 | 3→4 = horizon(原来 16 帧是按 chunk=64 调的,我们跑 chunk=100) |
| 5 | mixed 1:1 | 1.0 | 45 | **B** | l2 | 4→5 = teacher 大小 |
| 6 | mixed 1:1 | 1.0 | 45 | S | **LN** | 4→6 = target 归一化 |
| 7 | mixed 1:1 | 1.0 | 45 | S | l2 + **shuffled** | 4→7 = **负对照**。如果打乱时序也一样好,说明前面全是安慰剂 |

`ab0` 用的是**同样那 93 条**机器人数据(不是 609 条),这样 `ab1 − ab0` 才是干净的"人类数据值多少"。

**注意:阶梯设计假设因子之间不交互。先看 ab7 —— 如果 shuffled 最后和 ab4 一样好,中间所有对比都不作数。**

阶梯的下一级(等有空卡再跑,`scripts/train/train_vjepa_teacher.sh`):

| run | 数据 | λ | horizon | teacher | target norm | 读法 |
|---|---|---|---|---|---|---|
| vj0 | mixed 1:1 | 1.0 | 45 | **V-JEPA 2.1 B(视频)** | l2 | 4→vj0 = **teacher 是不是短板**(连带 `clip_frames` 1→2,躲不开) |

## 三之二、扰动套件判决(8 个 run 全部跑完 100k 步)

8 个 run 于 2026-08-17 11:39–16:44 全部干净跑完 100000 步。clean 指标一如预期分不出东西
(`val/l1` 全是 0.096,连负对照也一样),唯一在 clean 上看得见的是 ab0→ab1:
eef 0.284→0.267、head 0.135→0.120、waist 0.152→0.126。

判决口径是扰动套件:`data/dex5_val` 10 集 × 11 个轴,每个 checkpoint 约 9200 次前向。
脚本已从 `/tmp` 挪进 `scripts/eval/`(`perturb_eval.py` 单卡单 run / `run_perturb_all.sh` 8 卡并行 / `perturb_report.py` 汇总),
原始结果在 `/tmp/perturb_ab_*_dex5_val.json`。

**绝对 MPJPE(mm)**

| 轴 | ab0 robot | ab1 mixed λ0 | ab2 λ.3h16 | ab3 λ1h16 | ab4 λ1h45 | ab5 ViT-B | ab6 LN | ab7 shuf |
|---|---|---|---|---|---|---|---|---|
| clean | 49.15 | 44.78 | 43.72 | 43.35 | 43.49 | 44.07 | 43.30 | 43.82 |
| background | 73.60 | 54.14 | 51.52 | 48.82 | **48.61** | 51.10 | 48.70 | 50.12 |
| noise | 60.29 | 50.04 | 49.18 | 48.04 | 48.15 | 51.75 | 48.30 | 48.49 |
| blur | 56.60 | 51.80 | 51.98 | 50.27 | 49.91 | 52.36 | 50.11 | **49.80** |
| camera | 51.95 | 44.65 | 44.51 | 44.79 | 44.71 | 45.01 | 44.38 | 44.51 |
| occlusion | 54.19 | 50.37 | 49.28 | 48.23 | 48.56 | 49.42 | 48.49 | **47.69** |
| compound | 83.78 | 69.48 | 66.17 | 62.65 | **61.79** | 72.97 | 64.54 | 63.92 |

(光照/对比度/色相四个轴所有 run 都在 ±2% 内,略。)

**相对自己 clean 的平均退化:** ab0 **17.3%** → ab1 11.9% → ab2 11.9% → ab3 10.2% → ab4 **9.5%**;
ab5 13.5%,ab6 10.6%,**ab7 9.8%**。

1. **人类数据是最大的一笔:** 17.3% → 11.9%,全部来自 background(49.8%→20.9%)、noise、compound。
2. **λ 要给足:** 0.3 完全没用(11.9%→11.9%,学弟原来那个设置等于白挂),1.0 才动(10.2%),horizon 16→45 再到 9.5%。
   注:EgoWAM 的世界 loss 权重也是 λ=1。
3. **两个"改进"是负的:** teacher 换 DINOv2 ViT-B 9.5%→**13.5%**(compound 61.8→73.0mm);LayerNorm 归一化 10.6%。都别用。
4. **负对照打平了 —— 最重要的一条。** ab7 把 target 在 batch 内 `randperm` 打乱(`detr_vae.py:633`,时序配对彻底破坏),
   平均退化 9.8% vs ab4 的 9.5%。按第三节事先写好的判据,**"预测未来"拿不到这批数据的支持**:
   这个辅助 loss 的收益来自"去回归 DINO 特征"本身(相当于特征蒸馏正则),不来自"猜对了哪一帧"。
   有余地的地方:ab4 在 background(11.8% vs 14.4%)和 compound(42.1% vs 45.9%)这两个最该赢的轴上确实领先 2.6–3.8pt,
   是 blur/occlusion 上输回去把均值拉平的。但只有 10 集、脚本只存了均值没存方差,**这 2.6pt 是真的还是噪声现在说不了**,
   要定论得带 per-episode 误差棒重跑(约 20 分钟)。
5. **拿去用就用 `ab4_mixed_w1.0_h45_ckpt`**(混合数据 + λ=1.0 + horizon 45)。

## 三之三、第二轮:7 卡(2026-08-18 03:30 开跑)

`scripts/train/round2_7gpu.sh`。两个互不相干的问题塞进一次启动,第 7 张卡故意空着做探针和诊断。

### A. 输入编码器对比(GPU 0-1)—— 学姐要的第 2 件事

**ab4 本身就是 resnet18 那一臂**(而且是 `lr_backbone=1e-5` 端到端训的),所以只新起两个 run:

| GPU | run | 输入编码器 |
|---|---|---|
| 0 | `r2_in_dinov2` | 冻结 DINOv2-S(384 维,16×22 格) |
| 1 | `r2_in_vjepa` | 冻结 V-JEPA 2.1 ViT-B(768 维,14×19 格) |
| — | `ab4`(已有) | resnet18,**在训** |

**一个消不掉的混淆:** 两个新编码器都是冻结的(`backbone.py` 里跑在 `no_grad` 下,这是原有 `DINOv2BackBone` 就
有的设计,不是这次引入的),而 resnet18 是训练的。**所以新臂输了不等于编码器差**,只能说"冻结的它 打不过 会训的 resnet18"。

**另一个细节:单帧怎么喂进一个视频编码器。** V-JEPA 的 patch embed 是 Conv3d,要吃 `tubelet`(=2)帧,
而 trunk 手上只有当前帧,所以把当前帧**复制**成 2 帧 —— tubelet 卷积看到的是零运动。三件已核过的事:

1. **没有别的选择。** 发布的 2.1 ViT-B checkpoint 是按视频模型建的(`num_frames=64`、`is_video=True`、`PatchEmbed3D`)。
   上游代码里确实有一条 4-D 图像分支,但那条路要求模型建成 `num_frames=1`(2D `PatchEmbed`),而那样预训练的
   Conv3d 权重根本装不进去。实测直接喂 `[1,3,224,304]` 报 `ValueError: not enough values to unpack (expected 5, got 4)`。
2. **复制帧不是土办法,它就是标准的 3D→2D 核折叠。** 同一帧喂两遍,Conv3d 的输出恒等于"把时间维两片卷积核加起来"
   的那个 Conv2d。实测最大绝对误差 1.9e-05(纯浮点噪声)。所以相对于任何"正规的图像模式",这样喂**没有额外损失**。
3. **但不能说"ω-0 也是这么做的"** —— 这是我之前写过头的一句。论文里能查到的只有:
   "the policy receives only a single current image at each decision step"、
   "We use the frozen V-JEPA 2.1 **image** encoder as the visual encoder"。
   **tubelet / 时间维怎么处理,正文一个字没写。** 唯一的旁证是它对当前观测的 visual token 用的是 **2D RoPE**
   ("we apply 2D RoPE according to their spatial patch coordinates"),而未来视频 query 才用 3D RoPE ——
   说明它确实把当前帧当纯图像用,但具体怎么把视频骨干降到一帧,是论文没交代的那块。
   我们这边保留了 3D RoPE、只是时间维只有 1 组,这是一处和它对不齐的实现差异。

无论如何,V-JEPA 预训练里时序那一半在这个槽位上是闲置的。

**先验是这个实验不会赢:** DexWM 自己做过编码器横评(DINOv2 / DINOv3 / Web-SSL / **V-JEPA 2** / SigLIP 2),
DINOv2 第一;没有任何一篇报过 V-JEPA 在这个槽位上打赢 DINOv2 或打赢一个会训的 resnet。跑它是为了量,不是因为知道答案。

### B. VAE + DINO 双 target(GPU 2-6)—— 学姐要的第 1 件事

ST-WAM 是 **VAE 和 DINO 两路都回归**,而且 **VAE 是主力**(去掉 VAE 72.8→39.7,去掉 DINO 72.8→63.5,权重比 1.0 : 0.02)。
我们整套 ab0..ab7 只有 DINO 那半 —— 正好是他们说贡献最小的那半。**`ab4` 就是这次的 "DINO only" 臂,不重训。**

| GPU | run | λ_vae | λ_dino | VAE target 归一化 | ablation | 读作 |
|---|---|---|---|---|---|---|
| 2 | `r2_wv0_vaeonly` | 1.0 | **0.0** | raw | none | 只有 VAE |
| 3 | `r2_wv1_stwam` | 1.0 | 0.02 | raw | none | ST-WAM 的配比 |
| 4 | `r2_wv2_equal` | 1.0 | 1.0 | raw | none | 两路等权 |
| 5 | `r2_wv3_shuf` | 1.0 | 0.02 | raw | **两路都 shuffled** | 负对照 |
| 6 | `r2_wv4_l2norm` | 1.0 | 0.02 | **l2** | none | target 空间那个旋钮 |
| — | `ab4`(已有) | 0 | 1.0 | — | none | 只有 DINO |

**wv3 是第一个要看的。** 上一轮 DINO-only 的 shuffled 负对照跟 ab4 打平(9.8% vs 9.5%),
也就是说那点收益来自"多回归一遍特征"这个正则,不来自预测未来。如果 wv1 赢了 ab4 但 wv3 又追平 wv1,
那 VAE 这半也一样,这里没有世界模型可言。

λ_dino=0.0 的臂**仍然把 DINO 头挂着**(权重 0,静默),沿用 ab0/ab1 的做法:参数量和优化器状态跨臂完全一致,唯一的差别就是那一路 loss 有没有进总和。

### 关于 Wan VAE 的几件实测

- **Wan 是 VAE 不是 DINO 式编码器。** 它是 3D causal video autoencoder,目标是**重建像素**,
  latent 里保留了纹理/光照/精确颜色 —— 正好是 DINO 丢掉的那些。所以它是**第二个 target 物种**,不是替代品。
- **权重:** `Wan-AI/Wan2.2-TI2V-5B-Diffusers` 的 `vae/`,2.8GB,本地在 `~/.cache/huggingface/wan22_vae`。
  选 2.2 不选 2.1:2.2 是 16× 空间压缩,224×304 → 14×19 = **266** token(维度 48),和 DINOv2-S 的 352 同数量级;
  2.1 是 8×,会给出 1064 token,头的预算就不可比了。
- **单帧怎么过 4× 时间压缩:** causal VAE 下 T=1 进、T'=1 出,时间压缩在这里根本不参与。已验证输出 `[B,48,1,14,19]`。
- **bf16 是钉死的**,不跟 trunk 的 dtype 走:fp32 相对 bf16 只多出 0.016 的 latent 最大绝对误差(对一个回归 target 无意义),
  却要两倍的时间和显存。分块 32 帧编码,峰值 4.0GiB;不分块 128 帧一次要 12.6GiB。
- **latent 要按 checkpoint 自带的 `latents_mean`/`latents_std` 做逐通道标准化**,否则 48 个通道的 std 从 0.35 到 1.69,
  Huber 项会被少数几个通道吃掉。
- **VAE 那路的 loss 默认和 DINO 那路不一样,是故意的:** `normalize_target: false` + `huber_weight: 1.0`。
  latent 已经逐通道标准化过了,再把每个 48 维 token 单位化会把**模长**扔掉 —— 对一个重建码来说模长是信号;
  而且预测重建码是回归问题,预测 DINO 方向主要是角度问题,所以回归项在这里该当主角。`wv4` 把这个旋钮翻过来,让它变成量出来的而不是拍出来的。

### 代价与进度

Wan 编码器是这一步里最贵的东西:224×304 bf16 下 128 帧 **+0.51s/step**,叠在 ab4 的 0.35s/step 上。
实测 wv* 约 **1.07 it/s → 26 小时**;输入编码器那两臂 1.95–2.13 it/s → **13–14 小时**(对比 ab4 的 2.89 it/s / 9h50m)。

### 代码改动(还是只动新功能)

| 文件 | 改了什么 |
|---|---|
| `hdt/detr/models/backbone.py` | 新增 `VJEPABackBone`(把当前帧复制成 tubelet 帧喂视频编码器,输出 `[B,768,14,19]`,契约和 `DINOv2BackBone` 一样);`build_backbone` 加两行 `elif args.backbone.startswith('vjepa')`。**resnet18 / dinov2 两条路一行没动** |
| `hdt/detr/models/detr_vae.py` | `FrozenPatchTargetEncoder` 加 `WAN_VAE_MODELS` / `WAN_DTYPE` / `WAN_ENCODE_CHUNK` 和 `wan_vae` 分支 + `_forward_wan_vae`(反 ImageNet 归一化 → [-1,1] → snap 到 16 的倍数 → 分块 encode → 逐通道标准化 → `[N,266,48]`),加 `fixed_dtype` 属性。`DETRVAE` 加 `future_vae_config` 和一整套并列的 `future_vae_*` 成员 + 新方法 `_future_vae_forward`。**DINO 那条路整块没动**(故意不抽公共函数),ab0..ab7 仍是逐比特可复现的基线 |
| `hdt/policy.py` | `future_dino_loss_dict` 里出现 `vae_loss` 时才多加四个 `future_vae_*` 日志项和一项 loss。模型返回值还是那个四元组,没改契约 |
| `hdt/main.py` | 新增 `--use_future_vae_head` / `--future_vae_weight` / `--future_vae_ablation` / `--future_vae_normalize_target`,以及 config 解析与打印。**故意没有 `--future_vae_horizon`**:一个样本只有一帧未来,horizon 跟 DINO 那路共用 |
| `hdt/configs/models/act_input_dinov2.yaml`<br>`hdt/configs/models/act_input_vjepa.yaml` | 新建。除 `backbone` 外每一项都和 ab4 相同(λ=1.0 / horizon=45 / teacher=dinov2_vits14) |
| `hdt/configs/models/act_with_future_vae.yaml` | 新建。两个 target 物种各自一段;非世界头的部分与 ab4 一致 |
| `scripts/train/round2_7gpu.sh` | 新建。7 个 run,checkpoint 目录预先软链到 `/mnt/nvme0n1`,开跑前检查 Wan 权重在不在 |

**已验证:** 三个新 config 各自 3 步真实训练 exit=0;双 target 臂 step 0 打出
`val/future_dino_loss: 1.092`(w=0.02)和 `val/future_vae_loss: 1.570`(w=1.0),两路都在动、没有 NaN。

## 三之四、换一台机器怎么把这个 ckpt 跑起来

权重**不在 git 里**(`.gitignore` 里 `*_ckpt` 和 `BEST_CKPT` 都挡掉了),要手动拷。

1. `git clone` 这个仓库。
2. 把 `BEST_ab4_mixed_w1.0_h45/` 整个目录拷过去(390MB,里面四个文件是自洽的)。
   放哪都行,下面用 `$CKPT` 指代。
3. 推理只需要其中三个文件,`perturb_result.json` 只是成绩单:

```bash
python data/eval_mpjpe_batch.py \
  --policy-ckpt        "$CKPT/policy_last.ckpt" \
  --policy-config-yaml "$CKPT/act_with_future_dino.yaml" \
  --norm-stats         "$CKPT/dataset_stats.pkl" \
  --gt-dir  <验证集 hdf5 目录> \
  --device  cuda:0 \
  --out-json out.json
```

**三件容易踩的:**

- `dataset_stats.pkl` **必须**跟这个 ckpt 配对用。换一份归一化统计,输出就全是错的,而且不会报错。
- config 里的 `future_dino` 那一段推理时用不到(world 头不参与前向出 action),但**别删** ——
  `_load_act_policy` 按这个 yaml 建模型,少一段会导致 `load_state_dict` 对不上。
  已经是 `strict=False` 加载,所以多出来的 world 头权重会被忽略,这是预期行为。
- 只有做 world 头训练时才需要 teacher 权重:DINOv2 走 `torch.hub` 自己下;
  Wan VAE 那 2.8GB **不会自动下**,默认找 `~/.cache/huggingface/wan22_vae`,
  放在别处就设 `WAN_VAE_DIR` 环境变量。**纯推理这两个都不需要。**

## 三之五、`--zero_state_dims`:把机器人自身构型那一块挡掉(2026-08-18 16:35 开跑)

**为什么要挡。** 128 维状态里 `98:128` 这一块只有 dex5 有内容(`robot_q_current[0:26]`:
根位置 3 + 根四元数 4 + 29 个关节角的前 19 个),人类数据这一块原始值全是 0。
在 ab4 上量过:把这一块换成训练均值,MPJPE 掉 +36%,说明模型真的在读它。
问题是真机部署要一模一样地复现这一块,代价不小 —— 这一版就是问:这 36% 值不值。

**为什么是在归一化之后置 0,不是原始值置 0。** 人类那一半的统计是 mean 0 / std 0.01,
所以人类样本进模型时那一块本来就正好是 0。dex5 若写原始 0,按 dex5 自己的统计会落到 −59σ,
反而更糟。只有归一化之后置 0,两个 embodiment 才真正对齐,这才是"跟人类数据一样"。

**为什么写 `98:128` 而不是 `100:128`。** 第 98、99 维在两个 embodiment 里都恒为 0,两个区间等价;
`98:128` 只是对齐了块边界。共 30/128 维。

**改了什么(全部是新增、默认关闭,ab0~ab7 和 r2_\* 逐比特不受影响):**

- `hdt/main.py`:新增 `--zero_state_dims "lo:hi"`,解析成 `(lo, hi)` 塞进 `policy_config`。
  默认 `None` = 整个状态不动。
- `hdt/detr/models/detr_vae.py`:`DETRVAE.__init__` 多一个 `zero_state_dims=None` 参数,
  据此注册 buffer `state_keep_mask`;`forward` 里在 `bs, _ = qpos.shape` 之后立刻
  `qpos = qpos * self.state_keep_mask`。放这个位置的意思是 CVAE encoder 的 `qpos_embed`
  和 decoder 的 `input_proj_robot_state` **两条路都看不到**被挡的维度。
  写成 buffer 而不是普通属性,是为了让区间跟着 ckpt 走:真机那边加载权重就自动带上 mask,
  不需要知道该留哪几维空着。
- `scripts/train/run_zero_state.sh`(新文件):ab4 的命令原样照抄,只多 `--zero_state_dims 98:128`。

**评测侧也必须带 mask(踩过一次的坑):** mask 是 ckpt 里的 buffer,
`load_state_dict(strict=False)` 会**静默丢掉**它。用 `act_with_future_dino.yaml` 建模型来评,
模型就会被喂进它从没见过的完整状态,看起来像大幅退化,其实是配置错了。所以另加两处:

- `hdt/configs/models/act_with_future_dino_nostate.yaml`(新文件):ab4 的 config + `model.zero_state_dims: "98:128"`。
- `data/plot_keypoints_ys.py::_load_act_policy`:读 `model.zero_state_dims` 塞进 `policy_config`。
  纯新增,key 不存在时为 `None`,对之前所有 ckpt 行为逐比特不变。
  评测日志里应该能看到 `[state ablation] normalized qpos[98:128] forced to 0`,看不到就是没生效。

**结论(100k 步跑完,2026-08-18 21:11,9h49m @ 2.89 it/s):挡掉这一块反而更好。**
dex5_val 扰动套件,单位 mm:

| run | clean | noise | blur | camera | occlusion | background | compound |
|---|---|---|---|---|---|---|---|
| ab4(给全部状态) | 43.5 | 48.2 | 49.9 | 44.7 | 48.6 | 48.6 | 61.8 |
| r3_ab4_nostate | **41.0** | **44.0** | **44.7** | **41.0** | **44.1** | 48.7 | 63.3 |

clean −2.5mm(−5.8%),七个轴里五个更好,只有 compound +1.5mm。
**对真机的意义:98:128 那 30 维不用复现了** —— 之前在 ab4 上量到的 +36% 只说明 ab4 学会了依赖它,
不说明这信息是必需的;不给它去训,模型自己就找到了更好的解。

## 三之六、误差棒:哪些差是真的,哪些是噪声(2026-08-20)

之前全表每个 run 每个轴只有一个数,于是 2–3mm 的差一律说不清。补了两件小东西:

**1. `scripts/eval/perturb_eval.py`:多存一份 per-episode 均值。**
`run_ckpt` 原来只返回 `{轴: 全部帧的均值}`,现在返回 `({轴: 同一个均值}, {轴: [10 个 episode 的均值]})`。
**headline 均值的算法一个字没改**(还是那个 flat 的逐帧 list 取平均),所以已经报出去的每一个数都还成立 ——
这一点是**实测验证过的**:重跑 ab4 / r2_in_vjepa / r3_vjepa_nostate,新的 `--out` json 跟
`BEST_CKPT*/perturb_result.json` 里存的**逐位相同**。per-episode 那份写到旁边的
`<out>.per_ep.json`,不塞进主 json——因为 `perturb_report.py` 是 glob 主 json 的,多一个顶层 key 它会炸。

**2. `scripts/eval/perturb_paired.py`(新文件):配对比较。**
关键点:**不能看「均值 ± SEM」**。10 集的原始 SEM 是 ±4.6mm(均值才 42mm),因为 episode 本身难易差别巨大,
而这个方差是**共模**的 —— 难的那集对谁都难。而所有 ckpt 看的是**同样 10 集、同样的帧**
(STRIDE 3 从同一个检测起点)、**同样的扰动噪声**(`RandomState` 用帧号做种子,不是用 run 序号),
所以共模方差会抵消。有意义的量是**逐集之差**的 SEM,不是「两个均值之差」。两者均值完全一样,只有误差棒不同,
而这批 ckpt 上配对的误差棒小一个数量级(±0.3–3mm)。

判据:`|均值差| > 2×SEM` 记一个 `*`,再看 10 集里符号一致的有几集。10/10 或 9/10 且带 `*` 才算真差异,
5/10 附近就是噪声。**用法:`python scripts/eval/perturb_paired.py A.per_ep.json B.per_ep.json`**
(A 是基线),`scripts/eval/run_perturb_errbars.sh` 一把跑完关键的 7 个 ckpt。

### 结论:五条差异,只有两条站得住

单位 mm,负号 = 后者更好;`*` = 超过 2×SEM。

| 比较 | clean | background | compound | 相对退化 | 判决 |
|---|---|---|---|---|---|
| r2_in_vjepa − ab4（换主干） | −1.3±1.8 | +0.2±3.6 | **−11.1±4.6\*** | **−9.9±3.6\* (8/10)** | **真的** |
| ab7 − ab4（world 头打乱时序） | +0.5±0.5 | **+1.5±0.4\* (1/10)** | **+1.9±0.8\* (3/10)** | −0.4±0.8 (6/10) | **要改口径,见下** |
| wv3 − wv2（双 target 打乱） | **+1.3±0.6\*** | −0.5±1.2 | −0.8±1.0 | **−2.5±1.0\* (8/10)** | **是比值的假象** |
| r3_ab4_nostate − ab4（遮 98:128） | −1.9±1.9 | +0.5±1.9 | +2.3±3.3 | +1.9±1.6 (2/10) | **不显著** |
| r3_vjepa_nostate − r2_in_vjepa | +0.6±1.7 | +2.6±1.4 | +1.5±2.6 | +3.7±3.0 (3/10) | 不显著（确认不叠加） |

**要改口径的两条:**

- **「world 头的收益不来自预测未来」这句话下调成「大部分不来自」。** ab7 在**整体**相对退化上确实跟 ab4 打平
  (−0.4±0.8,6/10),这条没变;但**在最难的两个轴上 ab4 显著赢**:background +1.5±0.4(10 集里 9 集 ab4 更好)、
  compound +1.9±0.8。也就是说打乱时序确实丢了东西,只是只值 1.5–1.9mm,而且只在重度视觉偏移下才看得见 ——
  跟「换主干」那 11.1mm 完全不是一个量级。
- **「wv3 反超所有正经 VAE 臂」是相对退化这个比值造出来的。** 按绝对 mm,wv3 在 clean / noise / camera /
  occlusion 四个轴上**显著更差**(+1.1 到 +2.0,基本都是 1–2/10);它相对退化好看(−2.5±1.0),
  纯粹因为分母(自己的 clean)本来就更差。**教训:比值类指标必须跟绝对值一起看**,不然负对照会白捡便宜。

**掉下去的两条:** `r3_ab4_nostate` 那 2.5mm(clean −1.9±1.9,只有 6/10)**不显著** ——
遮掉 98:128 到底涨不涨精度,10 集分辨不出来。但**它的部署价值一点没变**:关键结论是「遮掉不变差」,
这条比「涨 2.5mm」弱得多也够用了,那 30 维照样不必在真机上复现。

## 四、踩过的坑,别再踩

1. **磁盘会满。** accelerate 每个 checkpoint 存 `optimizer.bin`(815M) + `pytorch_model.bin`(408M) = 1.15G。
   8 run × 10 个 = 92G,`/` 放不下。2026-08-16 那轮 8 个 run 全部在 ~50k/100k 步被杀。
   现在靠 `prune_ckpts.sh` 只留最新 2 个,峰值 18G。
2. **`metrics.csv` 列序是错的**(`hdt/main.py:29` 老 bug,没改):`_append_metrics_csv` 每次重算 `fieldnames`
   取并集,但表头只在建文件时写一次。结果 val 行 11 列、train 行 22 列共用同一个表头。
   **正确读法:只取列数等于表头的行(=val 行),按表头对齐。** 按位置硬解会解出 `val/l1=0.0000` 这种鬼数。
3. **clean val/l1 别当判据。** 8 个配置全挤在 0.1160–0.1167(极差 0.0007,纯噪声),负对照还是最低的那个。
   这个指标就是照不出 world model 的收益,判决要看扰动套件。
4. **`--load_pretrained_path` 不是 resume**,不恢复 optimizer 和 step 计数,跑挂了只能从头来。
5. **V-JEPA 的 `pretrained=True` 是坏的**(上游把权重 URL 换成了 `http://localhost:8300`),必须自己下 1.6G 权重;
   而且 checkpoint key 分两种(`ema_encoder` / `target_encoder`),看第二节第 6 条。

## 五、还没做的

- ~~误差棒~~ —— 见第三之六节,2026-08-20 已补,关键的 7 个 ckpt 都跑了配对比较。
  **仍然欠的是扩验证集**:10 集是所有结论的置信度天花板,配对之后能分辨的最小差异大约是 2mm(clean 轴),
  比这更小的差(比如 nostate 那条)加再多 run 也说不清
- **第二轮跑完要做的评测**:`scripts/eval/run_perturb_all.sh` 目前把 8 个 ab run 和它们各自的 yaml 写死在里面,
  第二轮那 7 个 run 得照着加进去(每个 run 必须配它**训练时**用的那个 yaml)
- ~~V-JEPA 当 target~~ / ~~VAE target~~ / ~~输入编码器对比~~ —— 见第三之三节,2026-08-18 已开跑
- 扰动套件三个脚本:`perturb_eval.py` 已挪进 `scripts/eval/`;`triplet_diag.py` 也挪了并加了 V-JEPA 对比(**还没跑**);
  `img_ablation.py` 仍在 `/tmp`
- 3D motion flow target(EgoWAM 说 in-domain 涨点要靠这个,DINO 只涨 OOD)—— 需要 dense 3D point tracker,工程量大
- 609 条机器人数据只用了 93 条。"609 条纯机器人 vs 93+93 混合"是另一个问题,等这轮跑完补
- DexWM 的手部关键点辅助 loss **已排除** —— 他们需要那条是因为没有 action head,我们的 action head 本来就直接监督手部关键点,重复了
