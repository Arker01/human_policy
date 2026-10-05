# example_data —— 两个格式样本（pick pillow）

一条人类、一条机器人，都是 pick pillow 任务，都取自训练实际用的 val 目录，
所以拿它们跑出来的数能跟已报的结果对上。

| 文件 | 来源 | `embodiment` | 帧数 | 大小 |
|---|---|---|---|---|
| `human_pickup_pillow_wholebody-91.hdf5` | `data/human_val/wholebody-91_unified_V.hdf5` | `human` | 193 | 45MB |
| `robot_inspire_pickup_pillow_ep_0093.hdf5` | `data/dex5_val/G1_WB_Dex5_Pickup_Pillow_ep_0093_rootfix.hdf5` | `dex5` | 222 | 13MB |

## 关于「Inspire」这个名字

机器人这条的文件名里写的是 `Dex5`，但它的上游就是 Inspire 那批 pillow 数据 ——
文件属性 `head_origin_input` 指向
`.../UnifoLM_WBT/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly/G1_WB_Dex5_Pickup_Pillow/...`，
`scripts/d.py` 下的也是 `unitreerobotics/G1_WBT_Inspire_Pickup_Pillow_MainCamOnly`。
即**上游 release 名叫 Inspire_Pickup_Pillow，里面这个子集是 Dex5 手**，转换后按 `embodiment=dex5` 存。
所以「pick pillow inspire 的机器人数据」= `data/dex5_*` 这批，没有另一份。
（真正标 Inspire 的另有 4 个任务，都是衣服/饮料，没有 pillow：见
`data/convert_UnifoLM_WBT_inspire_rootfix/`。pillow 只有 Dex5 843 条 + Brainco 56 条。）

## 两种格式的三个差别（会咬人的地方）

```
human : action(193,128) observation.state(193,128) observation.image.top(193,240,320,3) uint8
        + head_tracker_valid(193,) waist_valid(193,)
robot : action(222,128) observation.state(222,128) observation.image.left(222,~32k)
        + observation.image.right(222,~33k)
```

1. **相机名和编码都不一样。** 人类是 `observation.image.top`，**未压缩** `(T,240,320,3)`；
   机器人是 `left`/`right`，**JPEG 压缩**成 `(T,变长)` 的 uint8，要 `cv2.imdecode` 再 `BGR2RGB`。
   代码里的通用写法是「有 top 用 top，否则用第一个相机」，见 `scripts/eval/perturb_eval.py`。
2. **`state[98:128]` 只有机器人有内容。** 实测这两条：机器人 30 维里 26 维非零，
   人类**整块全 0**。这就是 `--zero_state_dims 98:128` 挡掉的那一块
   （`robot_q_current[0:26]`：根位置 3 + 根四元数 4 + 29 个关节角的前 19 个）。
   要复现 `BEST_CKPT_ab4_nostate` 就是把这 30 维置 0，两种数据长得一样。
3. **人类多两个 valid 掩码**（`head_tracker_valid` / `waist_valid`），机器人没有。

## 怎么读

```python
import h5py, cv2, numpy as np
with h5py.File("example_data/robot_inspire_pickup_pillow_ep_0093.hdf5", "r") as f:
    print(dict(f.attrs))                      # description / embodiment / rootfix_method ...
    state  = f["observation.state"][()]        # (T,128) 归一化前的原始值
    action = f["action"][()]                   # (T,128)
    raw = f["observation.image.left"][5]       # 变长 uint8 = JPEG 字节
    img = cv2.cvtColor(cv2.imdecode(raw, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
```

128 维怎么切（哪几维是头/手腕/膝/腰）看 `data/plot_keypoints_ys.py::_action_to_eval_joints`。
**注意归一化按 embodiment 分开做**，统计量在各个 `*_ckpt/dataset_stats.pkl` 里，
必须跟 ckpt 配套用。
