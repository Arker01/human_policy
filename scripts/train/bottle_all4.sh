#!/usr/bin/env bash
# pick-bottle 新任务：目前最好的三个配置 + 今天新加的 3D flow，两版数据各跑一遍。
#
# 全部是 new training（不是 finetune）：不传 --load_pretrained_path，权重随机初始化，
# lr 保持 1e-5。（传了那个 flag 会走 main.py:558 的 finetune_lr=1e-6，把 --lr 悄悄覆盖掉。）
#
# 两版数据的区别只在 sonic 机器人半边的时间降采样（人类半边两版完全相同）：
#
#   30hz  50->30 Hz，843 帧/条，30.4k 帧，人机帧比 3.4:1
#         和仓库既有语料的 30fps 约定一致（docs/FUTURE_DINO_CHANGES.md），
#         人类 dt=33ms、机器人 dt=33ms，chunk=100 在两边是同样的 3.3s。
#         剩下的差异（2.35 vs 老 dex5 的 5.36 mm/帧）是采集时确实更慢，真实差异，不抹。
#   10hz  50->10 Hz，281 帧/条，10.1k 帧，人机帧比 1.15:1
#         换成对齐「每帧位移」和帧数平衡：7.17 mm/帧 vs 人类 6.68，条长 281 vs 245，
#         整体形状最接近老 pillow 语料（1.06:1）。代价是机器人 dt=100ms 和人类 33ms 不一致。
#
#   gpu0-3  b30_*   30Hz 版（主）
#   gpu4-7  b10_*   10Hz 版（对照）
#
# flow 臂依赖离线 flow 目标，必须先跑 scripts/preprocess/run_flow_target_bottle.sh，
# 下面有条数守卫；另外六个臂不依赖它。
#
# 用法： bash scripts/train/bottle_all4.sh [all|<gpu 号>]
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
CFG_DIR=hdt/configs/models
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt
FLOW_DIR=/mnt/nvme0n1/human_policy_ckpt/flow_target_bottle
mkdir -p logs/bottle "$CKPT_ROOT"

DINO_COMMON="--use_future_dino_head --future_dino_warmup_steps 0 --future_dino_horizon 45 --future_dino_ablation none"

# gpu | exptid | 数据版本 | model_cfg | 额外 flag
RUNS=(
  "0|b30_vjepa_w0   |30hz|$CFG_DIR/act_input_vjepa.yaml              |$DINO_COMMON --future_dino_weight 0.0"
  "1|b30_ab4        |30hz|$CFG_DIR/act_with_future_dino.yaml         |$DINO_COMMON --future_dino_weight 1.0"
  "2|b30_ab4_nostate|30hz|$CFG_DIR/act_with_future_dino_nostate.yaml |$DINO_COMMON --future_dino_weight 1.0 --zero_state_dims 98:128"
  "3|b30_flow       |30hz|$CFG_DIR/act_with_future_flow.yaml         |--use_future_flow_head --future_flow_weight 1.0 --future_flow_dir FLOWDIR"
  "4|b10_vjepa_w0   |10hz|$CFG_DIR/act_input_vjepa.yaml              |$DINO_COMMON --future_dino_weight 0.0"
  "5|b10_ab4        |10hz|$CFG_DIR/act_with_future_dino.yaml         |$DINO_COMMON --future_dino_weight 1.0"
  "6|b10_ab4_nostate|10hz|$CFG_DIR/act_with_future_dino_nostate.yaml |$DINO_COMMON --future_dino_weight 1.0 --zero_state_dims 98:128"
  "7|b10_flow       |10hz|$CFG_DIR/act_with_future_flow.yaml         |--use_future_flow_head --future_flow_weight 1.0 --future_flow_dir FLOWDIR"
)

only="${1:-all}"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu exptid tag mcfg extra <<<"$spec"
  exptid="$(echo "$exptid" | xargs)"; mcfg="$(echo "$mcfg" | xargs)"; tag="$(echo "$tag" | xargs)"
  [[ "$only" != "all" && "$only" != "$gpu" ]] && continue

  MIXED="/home/aigc/human_policy/bottle_mixed_${tag}.json"
  if [ ! -s "$MIXED" ]; then echo "缺少数据配置: $MIXED"; exit 1; fi

  if [[ "$extra" == *FLOWDIR* ]]; then
    # flow 目标必须齐：训练时缺一条目标 = 那条 episode 的 world loss 静默变 0
    want=0
    for d in bottle_human_train bottle_human_val "bottle_robot_${tag}_train" "bottle_robot_${tag}_val"; do
      want=$((want + $(ls "$BASE_DIR/$d"/*.hdf5 2>/dev/null | wc -l)))
    done
    # 共享目录里还有另一版机器人的目标，所以只能查「这一版需要的那些」是否都在。
    # 用 dataloader 自己那份 flow_target_filename（hdt/data_utils_hdt.py:50）算名字，
    # 保证守卫和训练时的查找逻辑是同一套。
    have=$("$PY" - "$BASE_DIR" "$FLOW_DIR" "$tag" <<'PYEOF'
import os, sys, glob
sys.path.insert(0, '/home/aigc/human_policy')
from hdt.data_utils_hdt import flow_target_filename
base, flow_dir, tag = sys.argv[1], sys.argv[2], sys.argv[3]
dirs = ['bottle_human_train', 'bottle_human_val',
        f'bottle_robot_{tag}_train', f'bottle_robot_{tag}_val']
n = 0
for d in dirs:
    for f in sorted(glob.glob(os.path.join(base, d, '*.hdf5'))):
        p = os.path.join(flow_dir, flow_target_filename(f))
        if os.path.exists(p) and os.path.getsize(p) > 0:
            n += 1
print(n)
PYEOF
)
    if [ "$have" -lt "$want" ]; then
      echo "[$exptid] flow target 不全: $have/$want in $FLOW_DIR -- 先跑 scripts/preprocess/run_flow_target_bottle.sh"
      continue
    fi
    extra="${extra/FLOWDIR/$FLOW_DIR}"
  fi

  mkdir -p "$CKPT_ROOT/${exptid}_ckpt"
  [ -e "${exptid}_ckpt" ] || ln -s "$CKPT_ROOT/${exptid}_ckpt" "${exptid}_ckpt"

  log="logs/bottle/${exptid}.log"
  echo "[$(date -Is)] gpu=$gpu $exptid data=$tag cfg=$(basename "$mcfg")" | tee "$log"

  # flow 臂 batch 64 在 24G 卡上差约 600MB，碎片化导致的；expandable_segments
  # 就够了，不用降 batch（降了就和另外六个臂不可比）。对非 flow 臂无影响。
  CUDA_VISIBLE_DEVICES="$gpu" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    nohup "$PY" hdt/main.py \
    --batch_size 64 \
    --num_epochs 100000 \
    --lr 1e-5 \
    --chunk_size 100 \
    --exptid "$exptid" \
    --dataset_json_path "$MIXED" \
    --model_cfg_path "$mcfg" \
    --base_dir "$BASE_DIR" \
    --cond_mask_prob 0.0 \
    --no_wandb \
    $extra \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 25   # stagger: 多个进程同时抢 torch.hub cache 会卡死
done
