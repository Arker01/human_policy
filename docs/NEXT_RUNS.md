# 下一批要跑什么（2026-08-20，执行中）

**进度：** 表里的 1 号（`r4_vjepa_w0`）已在 GPU 0 上跑，2.0 it/s，约 13.9h，预计当日 20:45 前后完；
下面「不是训练的两件」里的**误差棒已经补完**（7 个 ckpt，GPU 1-7，结论见文末，已改口径）。
2/3/4 号仍未启动，要跑请点名。

## 先看现状：手上已有的结论

`data/dex5_val` 10 集 × 11 轴，MPJPE(mm)。「相对退化」= 相对自己 clean 的平均退化，衡量纯鲁棒性。

| run | clean | compound | 相对退化 | 状态 |
|---|---|---|---|---|
| ab4（resnet18 + DINO 头 λ1 h45） | 43.5 | 61.8 | 9.5% | 已交付 `BEST_CKPT` |
| ab7（同上，**打乱时序负对照**） | 43.8 | 63.9 | 9.8% | **和 ab4 打平** |
| r2_in_vjepa（主干换 V-JEPA） | 41.8 | **50.4** | **4.1%** | 已交付 `BEST_CKPT_vjepa` |
| r3_ab4_nostate（ab4 + 遮 98:128） | **41.0** | 63.3 | 10.0% | 已交付 `BEST_CKPT_ab4_nostate` |
| r3_vjepa_nostate（V-JEPA + 遮 98:128） | 42.7 | 51.1 | 4.1% | **刚测完，两个改进不叠加** |
| wv2（resnet18 + DINO&VAE 双 target 等权） | 40.9 | 58.2 | 8.5% | wv 组最好 |
| wv3（**两个 target 都打乱，负对照**） | 42.3 | 57.4 | 7.6% | **比所有正经 VAE 臂都好** |

两条已经站住的事实：

1. **V-JEPA 主干是全周唯一把鲁棒性砍半的改动**（9.5% → 4.1%），且赢在最难的 compound 轴。
2. **两轮独立的负对照（ab7、wv3）都打平或反超** —— world 头的收益不来自「预测未来」，
   而来自「去回归一个冻结编码器的特征」这件事本身（正则/蒸馏效应）。

刚补上的第三条：**遮状态和换主干不叠加**。r3_vjepa_nostate 相对 r2_in_vjepa 是
clean 41.8→42.7、compound 50.4→51.1、退化 4.1%→4.1%，即**略微变差、鲁棒性完全不变**。
遮 98:128 在 resnet18 上赚的那 2.5mm，在 V-JEPA 主干上消失了 ——
说明那 2.5mm 是「弱主干靠状态走捷径、遮掉反而逼它看图」，V-JEPA 本来就不走这条捷径。
**部署含义不变**：真机仍然不必复现那 30 维，只是别指望它再送精度。

---

## 候选 run

按「问题的价值 ÷ 成本」排序。速度按已测得的 it/s 估。

| 优先级 | run 名 | 配置 | 回答什么问题 | 卡时 |
|---|---|---|---|---|
| **1** | `r4_vjepa_w0` | V-JEPA 主干，`--future_dino_weight 0.0`（world 头整个关掉） | **换了好主干之后，world 头还值不值得挂？** | ~10h |
| **2** | `r4_vjepa_shuf` | V-JEPA 主干，`--future_dino_ablation shuffled` | 如果还值得挂，是不是因为预测未来？ | ~14h |
| **3** | `r4_vjepa_dual` | V-JEPA 主干 + DINO 1.0 & Wan VAE 1.0 双 target | 全周最好的输入端 + 最好的输出端能不能叠加 | ~30h |
| 4 | `r4_vjepa_dual_shuf` | 同上，两个 target 都打乱 | 3 的负对照。**只有 3 赢了才需要跑** | ~30h |

### 1. `r4_vjepa_w0` —— 应该最先跑

配置文件已备好的话直接改 `--future_dino_weight 0.0` 即可，不需要挂 DINO teacher，
所以比其它几个都快（估 2.0+ it/s，10 小时出头）。

**判据（先写死，免得事后找解释）：**
- 若 `vjepa_w0` 的相对退化 ≈ 4.1%（±0.5pt）→ **world 头在好主干上纯属开销，整条线可以收了**，
  3 和 4 都不必跑，直接把 `BEST_CKPT_vjepa` 简化成无 world 头版本重训一遍交付。
- 若明显差于 4.1%（比如回到 7% 以上）→ world 头确实在做事，继续跑 2 去问它做的是不是「预测未来」。

这个 run 的价值在于它是**唯一能证伪整条 world model 路线**的实验。ab7 和 wv3 只说明
「不是因为时序」，没说明「头本身有没有用」——因为打乱之后头还在，正则效应还在。
λ=0 才是把头拿掉。

### 2. `r4_vjepa_shuf`

只有 1 显示 world 头有用时才跑。配置 = `act_input_vjepa.yaml` + `--future_dino_ablation shuffled`。

### 3. `r4_vjepa_dual` —— 你提的那个组合

配置已经建好在 `hdt/configs/models/act_input_vjepa_dual.yaml`
（V-JEPA 主干 + `future_dino` λ1.0 h45 + `future_vae` wan22_vae λ1.0）。
启动命令：

```bash
CUDA_VISIBLE_DEVICES=<gpu> python hdt/main.py \
  --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size 100 \
  --exptid r4_vjepa_dual \
  --dataset_json_path pickup_pillow_mixed_1to1.json \
  --model_cfg_path hdt/configs/models/act_input_vjepa_dual.yaml \
  --base_dir data --cond_mask_prob 0.0 --no_wandb \
  --use_future_dino_head --future_dino_weight 1.0 --future_dino_warmup_steps 0 \
  --future_dino_horizon 45 --future_dino_ablation none \
  --use_future_vae_head --future_vae_weight 1.0
```

**为什么排第三而不是第一：** 它假设 world 头有用，而这个前提恰恰是 1 要检验的。
如果 1 说头没用，这 30 小时就白烧了。而且 wv 组已经给了个不好的信号——
双 target 在 resnet18 上的收益（8.5%）也被自己的负对照（7.6%）反超。

实测速度：起过一次，**1.10 s/it（0.91 it/s）→ 10 万步约 30 小时**，是所有候选里最贵的。

---

## 不是训练、但比上面几个都便宜的两件

| 事 | 成本 | 状态 |
|---|---|---|
| **per-episode 误差棒** | 实际 ~25 分钟 | **✅ 已完成**，见下 |
| **扩验证集** | 看有多少现成数据 | 仍然欠。10 集配对之后能分辨的最小差约 2mm，比这更小的差加再多 run 也说不清 |

---

## 误差棒结果（2026-08-20 补完，改了三处口径）

做法在 `docs/FUTURE_DINO_CHANGES.md` 第三之六节。要点：**不能看「均值 ± SEM」**——
10 集的原始 SEM 是 ±4.6mm（均值才 42mm），但那是共模的（难的集对谁都难），而所有 ckpt 看的是
同样 10 集、同样帧、同样噪声种子，所以要看**逐集之差**，误差棒小一个数量级。
`scripts/eval/perturb_paired.py A.per_ep.json B.per_ep.json`。

先确认没破坏任何已报的数：重跑 ab4 / r2_in_vjepa / r3_vjepa_nostate，
新 json 跟 `BEST_CKPT*/perturb_result.json` **逐位相同**。

| 比较 | clean | compound | 相对退化 | 判决 |
|---|---|---|---|---|
| r2_in_vjepa − ab4 | −1.3±1.8 | **−11.1±4.6\*** | **−9.9±3.6\* (8/10)** | **真的，全表唯一** |
| ab7 − ab4 | +0.5±0.5 | **+1.9±0.8\* (3/10)** | −0.4±0.8 (6/10) | 见下 |
| wv3 − wv2 | **+1.3±0.6\*** | −0.8±1.0 | **−2.5±1.0\* (8/10)** | 比值假象 |
| r3_ab4_nostate − ab4 | −1.9±1.9 | +2.3±3.3 | +1.9±1.6 (2/10) | **不显著** |
| r3_vjepa_nostate − r2_in_vjepa | +0.6±1.7 | +1.5±2.6 | +3.7±3.0 (3/10) | 不显著（确认不叠加） |

**三处改口径：**

1. **「world 头收益不来自预测未来」→「大部分不来自」。** ab7 在整体相对退化上确实打平（−0.4±0.8），
   但 background（+1.5±0.4，9/10 是 ab4 赢）和 compound（+1.9±0.8）两个最难的轴上**显著输**。
   打乱时序确实丢了东西，只值 1.5–1.9mm，且只在重度视觉偏移下才现形。
2. **「wv3 反超所有正经 VAE 臂」是比值造的假象。** 按绝对 mm，wv3 在 clean/noise/camera/occlusion
   四个轴上显著更差（+1.1~+2.0）；它相对退化好看纯粹因为分母（自己的 clean）更差。
   **比值指标必须跟绝对值一起看**，否则负对照白捡便宜。
3. **nostate 那 2.5mm 不显著。** 部署价值不变——它靠的是「遮掉不变差」，不是「涨」。

**对 1 号 run 判据的影响：** 原判据「相对退化 ≈4.1%（±0.5pt）」**太紧，作废**。
r2_in_vjepa 自己在配对口径下的相对退化是 7.2%，而 vjepa 臂之间的差要超过 ~3pt 才显著（见上表末行 ±3.0）。
**新判据：`r4_vjepa_w0` 的相对退益比 r2_in_vjepa 差 3pt 以上（且配对 8/10 同号）才算「world 头有用」，
否则算打平 → 整条 world model 线可以收，2/3/4 都不必跑。**
另外别只看相对退化，**必须同时看 compound 的绝对 mm**，就是被第 2 条坑过的那个教训。
