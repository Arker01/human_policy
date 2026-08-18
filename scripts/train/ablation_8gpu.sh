#!/usr/bin/env bash
# 8-GPU Future-DINO ablation, one run per GPU.
#
# Staircase design: each run changes exactly ONE thing from the run above it, so
# every adjacent pair answers one question. This buys 7 factors out of 8 slots at
# the cost of assuming the factors don't interact -- if run7 (the shuffled negative
# control) comes out as good as run4, none of the middle comparisons mean anything.
#
#  gpu  data          lambda  horizon  target_enc  target_norm  ablation | reads as
#   0   robot 93        0.0      -         -           -          -      | control: no human data
#   1   mixed 1:1       0.0      -         -           -          -      | anchor: human data only
#   2   mixed 1:1       0.3     16        S           l2        none     | current setting
#   3   mixed 1:1       1.0     16        S           l2        none     | 2->3 = lambda
#   4   mixed 1:1       1.0     45        S           l2        none     | 3->4 = horizon
#   5   mixed 1:1       1.0     45        B           l2        none     | 4->5 = teacher size
#   6   mixed 1:1       1.0     45        S      layernorm      none     | 4->6 = target norm
#   7   mixed 1:1       1.0     45        S           l2      shuffled   | 4->7 = is it just a regularizer?
#
# runs 0/1 keep the head attached at weight 0 (silent) rather than switching to
# act_resnet.yaml, so parameter count and optimizer state stay identical and the
# only difference is whether the world loss contributes anything.
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
CFG_DIR=hdt/configs/models
ROBOT=/home/aigc/human_policy/pillow_robot_s93.json
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json

mkdir -p logs/ablation8

# gpu | exptid | dataset_json | model_cfg | weight | horizon | ablation
RUNS=(
  "0|ab0_robot_w0.0        |$ROBOT|$CFG_DIR/act_with_future_dino.yaml     |0.0|16|none"
  "1|ab1_mixed_w0.0        |$MIXED|$CFG_DIR/act_with_future_dino.yaml     |0.0|16|none"
  "2|ab2_mixed_w0.3_h16    |$MIXED|$CFG_DIR/act_with_future_dino.yaml     |0.3|16|none"
  "3|ab3_mixed_w1.0_h16    |$MIXED|$CFG_DIR/act_with_future_dino.yaml     |1.0|16|none"
  "4|ab4_mixed_w1.0_h45    |$MIXED|$CFG_DIR/act_with_future_dino.yaml     |1.0|45|none"
  "5|ab5_mixed_w1.0_h45_vitb|$MIXED|$CFG_DIR/act_with_future_dino_vitb.yaml|1.0|45|none"
  "6|ab6_mixed_w1.0_h45_ln |$MIXED|$CFG_DIR/act_with_future_dino_ln.yaml  |1.0|45|none"
  "7|ab7_mixed_w1.0_h45_shuf|$MIXED|$CFG_DIR/act_with_future_dino.yaml    |1.0|45|shuffled"
)

only="${1:-all}"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu exptid djson mcfg weight horizon ablation <<<"$spec"
  exptid="$(echo "$exptid" | xargs)"; mcfg="$(echo "$mcfg" | xargs)"
  [[ "$only" != "all" && "$only" != "$gpu" ]] && continue

  log="logs/ablation8/${exptid}.log"
  echo "[$(date -Is)] gpu=$gpu exptid=$exptid w=$weight h=$horizon abl=$ablation cfg=$(basename $mcfg) data=$(basename $djson)" | tee "$log"

  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" hdt/main.py \
    --batch_size 64 \
    --num_epochs 100000 \
    --lr 1e-5 \
    --chunk_size 100 \
    --exptid "$exptid" \
    --dataset_json_path "$djson" \
    --model_cfg_path "$mcfg" \
    --base_dir "$BASE_DIR" \
    --cond_mask_prob 0.0 \
    --no_wandb \
    --use_future_dino_head \
    --future_dino_weight "$weight" \
    --future_dino_warmup_steps 0 \
    --future_dino_horizon "$horizon" \
    --future_dino_ablation "$ablation" \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 20   # stagger so 8 processes don't hit the dinov2 hub cache at once
done

wait
