# G1_WBT_Brainco_Collect_Plates_Into_Dishwasher 实际处理流程

记录 `compute_finger_keypoints.py` 和 `convert_unifolm_to_hdf5.py` 对
`/root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/data`
这份数据到底用了什么 mode、什么顺序。结论基于：磁盘上现存目录的时间戳、
`/root/.bash_history` 里的实际命令、以及最终产物 HDF5 里写入的 attrs（比日志和命令行更可信，
因为 attrs 是脚本运行时直接写进文件的）。

---

## 结论（当前生产数据实际使用的参数）

最终喂给训练、落在
`/root/shengyin/DATASETS/human_policy/convert_UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_XXXX.hdf5`
（300 个 episode，`ep_0000`–`ep_0299`）的数据，是下面三步串起来的结果：

### Step 1 — `compute_finger_keypoints.py`

```bash
python3 human_policy/compute_finger_keypoints.py \
  --dataset DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --local-frame-correction brainco_rx90_ry180 \
  --out-dir DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/finger_keypoints_rx90_ry180
```

- `hand_type`：自动识别为 `brainco`（路径里含 "Brainco"）。
- `finger_frame`：`wrist`（默认，手腕/手掌坐标系下的相对坐标，不是世界坐标）。
- `reverse` / `zero_hand`：均为 `False`，走正常 SDK→FK 映射。
- `local_frame_correction`：**`brainco_rx90_ry180`**（先绕 X 转 90°，再绕 Y 转 180°，
  修正 Revo2 URDF 手指方向与 policy 手腕坐标约定不一致的问题）。
- 输出：`finger_keypoints_rx90_ry180/ep_XXXX.pkl`，300 个文件，验证 `local_frame_correction`
  字段确实是 `brainco_rx90_ry180`（见下方“验证方式”）。

### Step 2 — `convert_unifolm_to_hdf5.py`

```bash
python3 human_policy/convert_unifolm_to_hdf5.py \
  --dataset DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY3 \
  --xqy3-wrist-rot-frame raw \
  --finger-dir DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/finger_keypoints_rx90_ry180 \
  --out-dir DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY3_rx90_ry180
```

- `--mode XQY3`：wrist 位置优先取 pkl 里显式的 wrist_pos 字段，否则退回
  `L/R_hand_base_link`；wrist 旋转取 `left/right_wrist_rot`。
- `--xqy3-wrist-rot-frame raw`：`left/right_wrist_rot` 按原始四元数使用，不做
  yaw_link 对齐变换。
- `--align-finger-frame none`：不再对指尖做二次坐标系对齐。
- 输出目录 `hdf5_XQY3_rx90_ry180` 目前磁盘上是**空的**——300 个 episode 的 hdf5
  在跑完 Step 3 之后被清理掉了，只留下了产物被消费的痕迹（下方 attrs 里的
  `head_origin_input` 路径就是指向这个已经被清空的目录）。

### Step 3 — `translate_hdf5_head_to_origin.py`（不在用户点名的两个脚本里，但是流水线的必要一环）

```bash
python3 human_policy/translate_hdf5_head_to_origin.py \
  --hdf5-dir DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher/hdf5_XQY3_rx90_ry180 \
  --xqy-prefix G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --out-dir DATASETS/human_policy/convert_UnifoLM_WBT
```

- 在前 20 帧（`--search-frames 20`）里找一个 10 帧（`--stable-window 10`）窗口，
  使头部位置方差最小（"最稳定"的一段），把这段的头部位置均值当作 `head_origin`。
- 把 `observation.state`/`action` 里的头部、双腕、qpos 根位置全部减去 `head_origin`，
  相当于把每个 episode 的起始头部位置平移到坐标原点。
- 最终文件名：`<xqy-prefix>_ep_XXXX.hdf5`，即
  `G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_0000.hdf5` … `..._ep_0299.hdf5`。

---

## 完整顺序总结

```
compute_finger_keypoints.py  (--local-frame-correction brainco_rx90_ry180)
        │  输出 finger_keypoints_rx90_ry180/ep_XXXX.pkl
        ▼
convert_unifolm_to_hdf5.py   (--mode XQY3 --xqy3-wrist-rot-frame raw)
        │  输出 hdf5_XQY3_rx90_ry180/{ep_idx}.hdf5   (128 维 state/action + 双目 JPEG)
        ▼
translate_hdf5_head_to_origin.py  (头部平移到原点 + 稳定窗口选取)
        │  输出 convert_UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_XXXX.hdf5
        ▼
   最终训练用 HDF5（300 episodes）
```

---

## 验证方式（怎么确认上面的结论，而不是猜的）

1. **最终 HDF5 的 attrs 直接写明了 mode**：

   ```python
   with h5py.File(".../convert_UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_0000.hdf5") as f:
       f.attrs["wrist_mode"]              # "XQY3"
       f.attrs["xqy3_wrist_rot_frame"]     # "raw"
       f.attrs["align_finger_frame"]       # "none"
       f.attrs["input_finger_frame"]       # "wrist"
       f.attrs["embodiment"]               # "human_mocap_xqy3"
       f.attrs["head_origin_input"]        # ".../hdf5_XQY3_rx90_ry180/0.hdf5"
       f.attrs["head_origin_window"]       # [0, 10]
   ```

2. **`finger_keypoints_rx90_ry180/ep_0000.pkl` 的 `local_frame_correction` 字段**
   确实是 `"brainco_rx90_ry180"`，`finger_frame` 是 `"wrist"`，`reverse`/`zero_hand`
   都是 `False`。

3. **`/root/shengyin/DATASETS/human_policy/convert_UnifoLM_WBT/convert_head_origin_batch.log`**
   里 `===== G1_WBT_Brainco_Collect_Plates_Into_Dishwasher =====` 这一段，300 行记录
   `input` 全部是 `.../hdf5_XQY3_rx90_ry180/{0..299}.hdf5`，`output` 全部是
   `convert_UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher_ep_{0000..0299}.hdf5`，
   和 attrs 交叉印证一致。

4. **对比数据**：`/root/shengyin/DATASETS/human_policy/convert_UnifoLM_WBT/` 目录下
   同名任务的 301 个文件（300 个 `_ep_XXXX.hdf5` + 1 个调试用的
   `_ep_0000_head_origin.html`），episode 数、attrs、head_origin 数值都和上面对得上，
   可以确认这就是这份 Dishwasher 数据最终对应的产出，不是别的任务混进来的。

---

## 探索过程中试过、但没有成为最终产物的其他 mode

从 `/root/.bash_history` 和各 `hdf5_*` 目录的时间戳看，这份数据在定下 `XQY3 +
brainco_rx90_ry180` 之前，做了大量对比实验，每个都只转了 1～13 个 episode 用
`data/plot_keypoints.py` 生成 mp4/html 做可视化排查，从未跑满 300 个 episode：

| 目录 | mode / 特殊参数 | episode 数 |
|---|---|---|
| `hdf5_EE` | `EE` | 13 |
| `hdf5_EE_V1_eval_tmp` / `hdf5_EE_V2_eval_tmp` | `EE_V1` / `EE_V2` | 各 1 |
| `hdf5_real-EE` / `hdf5_real-EE-rot` | `real-EE` / `real-EE-rot` | 各 1 |
| `hdf5_real-EE_aligned` / `hdf5_real-EE-rot_aligned` | 同上 + `--align-finger-frame xqy3_raw` | 各 1 |
| `hdf5_real-EE_worldtips` / `hdf5_real-EE-rot_worldtips` | 同上 + world-frame 指尖（`finger_keypoints_world_real-EE`） | 各 1 |
| `hdf5_XQY1` | `XQY1` | 5 |
| `hdf5_XQY2` | `XQY2` | 5 |
| `hdf5_XQY3` | `XQY3`（未加 rx90_ry180 修正） | 4 |
| `hdf5_XQY3_zero_hand` | `XQY3` + `finger_keypoints_zero`（`--zero-hand`，排除手指姿态干扰单独看手腕） | 1 |
| `hdf5_XQY4` / `hdf5_XQY4_*_eval_tmp` | `XQY4`（在 XQY3 基础上加固定手腕修正矩阵），试过 `raw`/`yaw_link_aligned` 两种 `--xqy3-wrist-rot-frame` | 各 1~3 |
| `hdf5_XQY5` / `hdf5_XQY5_pure90_eval_tmp` | `XQY5`（另一套固定手腕修正矩阵） | 各 1 |
| `hdf5_XQY_PH2D` | `XQY_PH2D`（指尖方向对齐到 PH2D 参考数据集） | 1 |

按目录创建时间排序，大致的排查顺序是：
`EE → XQY1 → XQY2 → XQY3 → XQY_PH2D → XQY4 → XQY5 →`
（对比 `--xqy3-wrist-rot-frame raw` vs `yaw_link_aligned`）→
`EE_V1/EE_V2/real-EE/real-EE-rot`（各种 world-tips / aligned 变体）→
`XQY3 + finger_keypoints_zero`（隔离手指姿态用于纯手腕验证）→
最终定为 `XQY3 + finger_keypoints_rx90_ry180`，跑满 300 集，再经
`translate_hdf5_head_to_origin.py` 产出最终 300 个训练 HDF5。

---

## 附：两个脚本支持的全部 mode / 参数速查

### `compute_finger_keypoints.py`

- `--hand-type`：`auto`（按路径含 Brainco/Inspire 判断）/ `brainco` / `inspire`
- `--finger-frame`：`wrist`（默认）/ `world`
- `--wrist-pose-source`（仅 `--finger-frame world` 时用）：`real-ee` / `xqy3_raw`
- `--reverse`：仅对 Inspire 生效，翻转开合通道
- `--zero-hand`：忽略 `hand_state`，手部关节全部置零算 FK（用于纯手腕验证）
- `--local-frame-correction`：`none` / `brainco_rx90` / `brainco_rx90_ry180`
  （仅 Brainco 生效，修正 Revo2 URDF 指尖方向约定）

### `convert_unifolm_to_hdf5.py`

- `--mode`（必填）：`EE` / `EE_V1` / `EE_V2` / `real-EE` / `real-EE-rot` /
  `XQY1` / `XQY2` / `XQY3` / `XQY4` / `XQY5` / `XQY_PH2D`
  （腕部位置/旋转的来源和修正各不相同，详见脚本头部 docstring）
- `--xqy3-wrist-rot-frame`：`raw`（默认）/ `yaw_link_aligned`，只影响
  `XQY3`/`XQY4`/`XQY5` 里 `left/right_wrist_rot` 的解释方式
- `--align-finger-frame`：`none`（默认）/ `xqy3_raw`，是否把指尖从
  `xqy3_raw` 手腕系重新投影到当前 mode 的手腕系
- 128 维 `observation.state` 布局：`[0:9]` 头部 EEF，`[10:28]` 左手指尖，
  `[30:39]` 右腕 EEF，`[40:58]` 右手指尖，`[80:89]` 左腕 EEF，`[100:126]` qpos

---

## Inspire 手的流程：确实比 Brainco 简单

Brainco（Dishwasher）为了让指尖朝向对得上，中间做了一大圈 `brainco_rx90` /
`brainco_rx90_ry180` 的试验（见上面“探索过程”表格），才定下最终的
`local_frame_correction`。Inspire 这边**没有等价的“探索-定型”过程**：`compute_finger_keypoints.py`
里 `local_frame_correction` 参数对 Inspire 直接不生效（`apply_local_frame_correction`
只在 `hand_type=="brainco"` 时才应用旋转修正，Inspire 传什么值都会被忽略/报错），
真正需要调的只有两个开关：

- `--reverse`：翻转开合通道（`inspire_reverse_mode` 默认就是 `fingers_thumb`，
  代表食指/中指/无名指/小指/拇指开合五个通道整体做 `1 - x` 翻转，拇指侧摆通道不变）
- `--inspire-thumb-mode swap`：把 SDK 里的 `thumb_pitch`/`thumb_yaw` 两个通道互换

确认这两个开关就够的证据：5 个 Inspire 任务里，除最早的
`G1_WB_Dex5_Collect_Clothes`（第一次摸索 Inspire 参数，留下了
`hdf5_EE_reverse`、`hdf5_real-EE_reverse`、`hdf5_XQY3_current_tmp`、
`hdf5_XQY3_yaw_current_tmp` 等一堆一次性对比目录）之外，其余 4 个任务磁盘上都
**只有** `finger_keypoints_reverse_thumb_swap` / `hdf5_XQY3_reverse_thumb_swap`
这一套目录，没有任何其它 mode 的尝试痕迹——说明参数组合一旦在第一个任务上验证过，
后面就直接照搬，不用再逐个任务重新试错。

### Step 1 — `compute_finger_keypoints.py`（Inspire）

```bash
python3 human_policy/compute_finger_keypoints.py \
  --dataset DATASETS/UnifoLM_WBT/<Inspire 任务目录> \
  --hand-type inspire \
  --reverse \
  --inspire-thumb-mode swap \
  --out-dir DATASETS/UnifoLM_WBT/<Inspire 任务目录>/finger_keypoints_reverse_thumb_swap
```

用 `ep_0000.pkl` 实测确认到的字段（以 `G1_WB_Dex5_Collect_Clothes` 为例）：

```text
hand_type: inspire
finger_frame: wrist
reverse: True
inspire_reverse_mode: fingers_thumb   # 默认值，没有额外指定
inspire_thumb_mode: swap
inspire_thumb_yaw_sign: normal        # 默认值，没有额外指定
local_frame_correction: none          # Inspire 不适用，恒为 none
```

### Step 2 — `convert_unifolm_to_hdf5.py`（Inspire）

和 Brainco 用的**同一个 mode**：

```bash
python3 human_policy/convert_unifolm_to_hdf5.py \
  --dataset DATASETS/UnifoLM_WBT/<Inspire 任务目录> \
  --mode XQY3 \
  --xqy3-wrist-rot-frame raw \
  --finger-dir DATASETS/UnifoLM_WBT/<Inspire 任务目录>/finger_keypoints_reverse_thumb_swap \
  --out-dir DATASETS/UnifoLM_WBT/<Inspire 任务目录>/hdf5_XQY3_reverse_thumb_swap
```

最终 hdf5 attrs 实测（`G1_WB_Dex5_Collect_Clothes_ep_0000.hdf5`）：
`wrist_mode=XQY3`、`xqy3_wrist_rot_frame=raw`、`align_finger_frame=none`，
和 Dishwasher 一模一样——**Inspire 和 Brainco 在 Step 2 上完全没有差异**，
差异只在 Step 1 的手指参数（`local_frame_correction` vs `reverse`+`thumb_mode`）。

### Step 3 — 同一个 `translate_hdf5_head_to_origin.py`，Inspire/Brainco 不区分

### 五个 Inspire 任务的落地情况

| 任务（`meta/info.json` episode 数） | Step1 finger pkl 数 | Step2 hdf5 数（磁盘现状，已清理） | 最终 `convert_UnifoLM_WBT` 产出数 |
|---|---|---|---|
| `G1_WB_Dex5_Collect_Clothes`（336） | 336 | 0（已清理） | 336 |
| `G1_WB_Dex5_Pickup_Pillow`（609） | 609 | 0（已清理） | 609 |
| `G1_WB_Dex5_Put_Clothes_into_Washing_Machine`（457） | 457 | 0（已清理） | 457 |
| `G1_WBT_Inspire_Put_Clothes_Into_Basket`（300） | 300 | 0（已清理） | **106**（不完整，只转了一部分） |
| `G1_WBT_Inspire_Put_Clothes_into_Washing_Machine`（300） | 300 | 0（已清理） | 300 |

`G1_WBT_Inspire_Put_Clothes_Into_Basket` 目前只有 106/300 个 episode 落到了
`convert_UnifoLM_WBT`，如果要用这份数据训练，需要注意先把剩下的 194 个 episode
按同样的 Step1→Step2→Step3 补跑完。
