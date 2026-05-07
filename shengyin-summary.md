# Shengyin Human Policy Summary

## Paths

Repo:

```bash
/root/shengyin/human_policy
```

Main training entry:

```bash
/root/shengyin/human_policy/hdt/main.py
```

MPJPE eval script:

```bash
/root/shengyin/human_policy/data/eval_mpjpe_batch.py
```

UnifoLM WBT -> human_policy HDF5 converter:

```bash
/root/shengyin/human_policy/convert_unifolm_to_hdf5.py
```

Pretrain dataset JSON:

```bash
/root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json
```

Common 903 validation GT:

```bash
/root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16
```

## Config Files

ACT Flow Matching, ResNet-18:

```bash
/root/shengyin/human_policy/hdt/configs/models/act_flow.yaml
```

ACT Flow Matching, DINOv2:

```bash
/root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml
```

ACT ResNet config used by earlier baseline scripts:

```bash
/root/shengyin/human_policy/hdt/configs/models/act_resnet.yaml
```

Single-GPU accelerate config:

```bash
/root/shengyin/human_policy/hdt/1_gpu.yaml
```

## Training Commands

Run from:

```bash
cd /root/shengyin/human_policy/hdt
```

ResNet-18 ACT_FM pretrain with action position/query embedding:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./1_gpu.yaml main.py --batch_size 64 --num_epochs 100000 --lr 1e-4 --chunk_size 100 --seed 0 --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml --base_dir /root/shengyin/DATASETS --human_slow_down_factor 4 --no_wandb
```

DINOv2 ACT_FM pretrain with action position/query embedding:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file ./1_gpu.yaml main.py --batch_size 64 --num_epochs 100000 --lr 1e-5 --chunk_size 100 --seed 0 --exptid /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos --dataset_json_path /root/shengyin/human_policy/data/act_fm_pretrain_convert_ego_ph2d.json --model_cfg_path /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml --base_dir /root/shengyin/DATASETS --human_slow_down_factor 4 --no_wandb
```

Checkpoint output rule:

```text
--exptid <PATH>  ->  <PATH>_ckpt
```

Training saves:

```text
<exptid>_ckpt/policy_iter_<ITER>_seed_<SEED>/pytorch_model.bin
<exptid>_ckpt/policy_last.ckpt
<exptid>_ckpt/dataset_stats.pkl
<exptid>_ckpt/metrics.csv
<exptid>_ckpt/metrics.png
```

The code saves full accelerate checkpoints every 10000 training iterations.

## UnifoLM WBT HDF5 Conversion

新的独立转换脚本：

```bash
/root/shengyin/human_policy/convert_unifolm_to_hdf5.py
```

它不依赖旧的 `convert_to_hdf5.py` 或 `convert_to_hdf5_inspire.py`。输入数据：

- finger keypoints：`<dataset>/finger_keypoints/ep_XXXX.pkl`
- XQY pkl：`/root/shengyin/DATASETS/XQY_PKL/<xqy-prefix>_ep_XXXX.pkl`
- 原始 parquet：`<dataset>/data/chunk-*/*.parquet`
- 视频：从 `<dataset>/meta/episodes` 自动读取左右相机文件和 timestamp

输出目录默认是：

```text
<dataset>/hdf5_<mode>
```

输出 HDF5 结构可直接给 human_policy 训练：

```text
observation.state        (T, 128) float32
action                   (T, 128) float32
observation.image.left   JPEG-compressed uint8
observation.image.right  JPEG-compressed uint8
```

四个 wrist/head 模式：

```text
EE:
  head:  XQY pkl 的 head_mocap，默认比 head_link 更符合真实头部高度；需要复现旧行为可传 `--head-link head_link`
  wrist: 原始 ee_state；position 直接作为 world position 使用，不再当作 head-relative；rotation 是全局 Euler 直接用

XQY1:
  head:  XQY pkl 的 head_mocap
  wrist position: L/R_hand_base_link
  wrist rotation: L/R_hand_base_link。finger keypoints 是 hand-base local frame，不能用 wrist_yaw_link，否则 thumb 会被转到 palm 下方

XQY2:
  head:  XQY pkl 的 head_mocap
  wrist position: L/R_hand_base_link
  wrist rotation: roll-link 的 roll + pitch-link 的 pitch + yaw-link 的 yaw 重新组装

XQY3:
  head:  XQY pkl 的 head_mocap
  wrist position: 优先 pkl 里的 wrist-pos 字段；没有时回退 L/R_hand_base_link
  wrist rotation: pkl 里的 left_wrist_rot/right_wrist_rot，xyzw quaternion。当前数据里它和 L/R_hand_base_link rotation 完全一致，默认直接使用 raw；不要对齐到 wrist_yaw_link

XQY_PH2D:
  head:  XQY pkl 的 head_mocap
  wrist position: L/R_hand_base_link
  wrist rotation: left/right_wrist_yaw_link
  finger keypoints: 用 PH2D 参考文件首尾帧的五指 world direction 作为模板，并按时间插值；再用当前 wrist rotation 反算回 local keypoints 写入 HDF5。这个模式强制保证可视化时五指全局朝向关系匹配 `/root/shengyin/DATASETS/PH2D/402-pick_on_color_pad_right-2025_01_09-16_36_15/processed_episode_0.hdf5` 的首尾帧关系，同时保留 Brainco FK 的每根手指长度
```

Brainco plates 数据集常用命令：

```bash
cd /root/shengyin

/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode EE \
  --max-episodes 5

/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY1 \
  --max-episodes 5

/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY2 \
  --max-episodes 5

/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY3 \
  --max-episodes 5

/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY_PH2D \
  --max-episodes 5
```

只转一个 episode 调试：

```bash
/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY1 \
  --episode 0 \
  --overwrite
```

如果你已经用旧版本生成过 HDF5，修正 head/rotation 后需要加 `--overwrite` 重新生成：

```bash
/root/miniconda3/envs/human_policy/bin/python human_policy/convert_unifolm_to_hdf5.py \
  --dataset /root/shengyin/DATASETS/UnifoLM_WBT/G1_WBT_Brainco_Collect_Plates_Into_Dishwasher \
  --mode XQY1 \
  --overwrite
```

如果 XQY pkl 文件前缀和 dataset basename 不一致，例如需要读取：

```text
/root/shengyin/DATASETS/XQY_PKL/G1_WB_Dex5_Collect_Clothes_ep_0000.pkl
```

就显式传：

```bash
--xqy-prefix G1_WB_Dex5_Collect_Clothes
```

如果只想快速检查 state，不抽视频：

```bash
--no-images
```

## Loss Curves

Training/validation loss curves are saved automatically:

```bash
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos_ckpt/metrics.csv
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos_ckpt/metrics.png

/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/metrics.csv
/root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/metrics.png
```

Important: `val/loss` is not the same as MPJPE. Use MPJPE to compare motion prediction quality.

For ACT_FM, loss keys include:

```text
val/fm
val/loss
val/hand_eef_loss
val/head_eef_loss
train/fm
train/loss
train/hand_eef_loss
train/head_eef_loss
```

## MPJPE Eval Commands

Run from:

```bash
cd /root/shengyin/human_policy
```

ResNet actionpos, final checkpoint:

```bash
python data/eval_mpjpe_batch.py --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 --policy-ckpt /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos_ckpt/policy_iter_100000_seed_0/pytorch_model.bin --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_flow.yaml --norm-stats /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos_ckpt/dataset_stats.pkl --device cuda:0 --seed 0 --eval-mode first_token --out-json /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_1e-4_actionpos_ckpt/mpjpe_903_seed0_step100000.json
```

DINOv2 actionpos, best checkpoint on 903 MPJPE:

```bash
python data/eval_mpjpe_batch.py --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 --policy-ckpt /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/policy_iter_70000_seed_0/pytorch_model.bin --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml --norm-stats /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/dataset_stats.pkl --device cuda:0 --seed 0 --eval-mode first_token --out-json /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/mpjpe_903_seed0_step70000.json
```

Alternative eval modes:

```bash
--eval-mode chunk_rollout --chunk-stride 10
--eval-mode temporal_agg --temporal-decay 0.01
```

Meaning:

```text
first_token: every frame runs policy once and uses only predicted chunk[0].
chunk_rollout: runs policy every k frames and uses chunk[0:k].
temporal_agg: every frame predicts a full chunk; overlapping predictions are exponentially weighted and averaged.
```

Current observation: chunk rollout and temporal aggregation did not systematically improve 903 MPJPE. Best current 903 result is DINOv2 actionpos at step 70000 with first-token eval.

## Human Policy 接 TWIST 手部 GMT

入口脚本：

```bash
/root/shengyin/human_policy/twist_hand_gmt_bridge.py
```

先用 human policy 生成手部参考轨迹：

```bash
cd /root/shengyin/human_policy
/root/miniconda3/envs/human_policy/bin/python twist_hand_gmt_bridge.py \
  --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --policy-ckpt /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/policy_iter_70000_seed_0/pytorch_model.bin \
  --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_flow_dinov2.yaml \
  --norm-stats /root/shengyin/outputs/act_fm_pretrain_convert_ego_ph2d_dinov2_1e-5_actionpos_ckpt/dataset_stats.pkl \
  --device cuda:0 \
  --eval-mode first_token \
  --dump-ref-npz /root/shengyin/human_policy/outputs/twist_hand_refs.npz \
  --skip-gmt
```

然后 headless 跑手部 GMT：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --ref-npz /root/shengyin/human_policy/outputs/twist_hand_refs.npz \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_bridge_actions.npz
```

有显示器或虚拟显示器时，开 viewer 看可视化：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --ref-npz /root/shengyin/human_policy/outputs/twist_hand_refs.npz \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --viewer \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_viewer_actions.npz
```

需要录视频时：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --ref-npz /root/shengyin/human_policy/outputs/twist_hand_refs.npz \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --viewer \
  --record-video \
  --video-stride 2 \
  --out-video /root/shengyin/human_policy/outputs/twist_hand_gmt_bridge.mp4 \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_video_actions.npz
```

说明：纯 headless 不要加 `--record-video`。录视频需要 `--viewer`，并且机器要有显示器或虚拟显示器。

### ACT + ResNet 版本

headless 一阶段：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --policy-ckpt /data1/zxlei/model/convert2_with_pick_ckpt/policy_last.ckpt \
  --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_resnet.yaml \
  --norm-stats /data1/zxlei/model/convert2_with_pick_ckpt/dataset_stats.pkl \
  --device cuda:0 \
  --eval-mode first_token \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_act_resnet_actions.npz
```

### magic-4090
CUDA_VISIBLE_DEVICES=0 python twist_hand_gmt_bridge.py \
  --gt-dir /media/magic-4090/DATA1/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --policy-ckpt /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/policy_last.ckpt \
  --policy-config-yaml /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/act_resnet.yaml \
  --norm-stats /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/dataset_stats.pkl \
  --device cuda:0 \
  --eval-mode first_token \
  --hand-gmt-ckpt /media/magic-4090/DATA1/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --viewer \
  --out-actions /media/magic-4090/DATA1/shengyin/human_policy/ruili-result/0506/eval-result/twist_hand_gmt_act_resnet_actions.npz

viewer 一阶段：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --policy-ckpt /data1/zxlei/model/convert2_with_pick_ckpt/policy_last.ckpt \
  --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_resnet.yaml \
  --norm-stats /data1/zxlei/model/convert2_with_pick_ckpt/dataset_stats.pkl \
  --device cuda:0 \
  --eval-mode first_token \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --viewer \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_act_resnet_viewer_actions.npz
```

录视频一阶段：

```bash
cd /root/shengyin/human_policy
export PATH=/root/miniconda3/envs/twist/bin:/usr/local/cuda-12.4/bin:$PATH
export LD_LIBRARY_PATH=/root/miniconda3/envs/twist/lib:/usr/local/cuda-12.4/lib64:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=0 /root/miniconda3/envs/twist/bin/python twist_hand_gmt_bridge.py \
  --gt-dir /root/shengyin/DATASETS/PH2D/903-picking-val-2024_11_18-18_58_16 \
  --policy-ckpt /data1/zxlei/model/convert2_with_pick_ckpt/policy_last.ckpt \
  --policy-config-yaml /root/shengyin/human_policy/hdt/configs/models/act_resnet.yaml \
  --norm-stats /data1/zxlei/model/convert2_with_pick_ckpt/dataset_stats.pkl \
  --device cuda:0 \
  --eval-mode first_token \
  --hand-gmt-ckpt /root/shengyin/human_policy/hand_GMT/dexhand_mimic_direct_newkpkd_model_50000.pt \
  --gmt-device cuda:0 \
  --viewer \
  --record-video \
  --video-stride 2 \
  --out-video /root/shengyin/human_policy/outputs/twist_hand_gmt_act_resnet.mp4 \
  --out-actions /root/shengyin/human_policy/outputs/twist_hand_gmt_act_resnet_video_actions.npz
```

## Current MPJPE Results On 903

Action position/query embedding improved ACT_FM substantially.

```text
ResNet 1e-4 actionpos, step100000:
  all 57.02 mm
  hand 63.65 mm
  wrist 44.97 mm
  head 1.61 mm

DINOv2 1e-5 actionpos, step70000:
  all 53.71 mm
  hand 59.97 mm
  wrist 41.86 mm
  head 2.21 mm
```

For the DINOv2 actionpos run, step 100000 is not the best by 903 MPJPE. Step 70000 is better.

## ACT vs ACT_FM In This Codebase

ACT:

- Uses `policy_class: ACT`.
- Uses DETR VAE style model in `hdt/detr/models/detr_vae.py`.
- Has learned action query embeddings:

```python
self.query_embed = nn.Embedding(num_queries, hidden_dim)
```

- Predicts the action chunk directly.
- Training loss is mainly L1 plus optional KL / EEF terms depending on config.

ACT_FM:

- Uses `policy_class: ACT_FM`.
- Main implementation:

```bash
/root/shengyin/human_policy/hdt/modeling/modeling_act_flow.py
```

- Training samples random Gaussian noise `x0`, random time `t`, and linearly interpolates:

```python
xt = (1 - t) * x0 + t * actions
v_target = actions - x0
```

- The model predicts velocity `v_pred`.
- Inference starts from Gaussian noise and integrates Euler steps:

```python
x = torch.randn(B, chunk_size, action_dim)
for i in range(num_flow_steps):
    x = x + v * dt
```

- Current default `num_flow_steps` is 10 in the YAML config.
- We added action position/query embedding to `FlowMatchingHead` so action chunk tokens have explicit identities. This made the model closer to ACT and PI-style action-token sequence modeling.

Important ACT_FM implementation detail after the fix:

```python
self.action_query_embed = nn.Embedding(chunk_size, hidden_dim)
out = self.decoder(x, ctx, query_pos=query_pos)
```

Without this, chunk tokens had no explicit position identity, and MPJPE was much worse.

## Notes

- Dataset JSON field `type` is currently not the main behavior switch in loading; actual embodiment comes from HDF5 attrs.
- If config camera is `top`, loader/eval can fall back to `left` then `right` when top is missing.
- For fair ACT_FM actionpos testing, train from scratch. Loading old non-actionpos checkpoints leaves the new embedding randomly initialized.
- When using `--load_pretrained_path`, current train code uses `strict=False` and has a hardcoded warmup scheduler branch that does not fully respect the CLI `--lr`.
