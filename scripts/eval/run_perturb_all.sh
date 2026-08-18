#!/usr/bin/env bash
# Run the perturbation suite on all 8 ablation checkpoints, one per GPU, then aggregate.
#
# Each checkpoint must be evaluated with the yaml it was TRAINED with -- ab5 has a
# ViT-B teacher and ab6 a LayerNorm target, and those change the world-head shapes.
# (The world head is not used for the action prediction we measure, but loading with
# the wrong config would silently drop weights via strict=False, so keep them matched.)
#
# Usage: run_perturb_all.sh [val_dir]   (default data/dex5_val)
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
CFG_DIR=hdt/configs/models
VAL_DIR="${1:-/home/aigc/human_policy/data/dex5_val}"
TAG="$(basename "$VAL_DIR")"
OUT=/tmp/perturb_ab
mkdir -p logs/perturb

RUNS=(
  "0|ab0_robot_w0.0        |$CFG_DIR/act_with_future_dino.yaml"
  "1|ab1_mixed_w0.0        |$CFG_DIR/act_with_future_dino.yaml"
  "2|ab2_mixed_w0.3_h16    |$CFG_DIR/act_with_future_dino.yaml"
  "3|ab3_mixed_w1.0_h16    |$CFG_DIR/act_with_future_dino.yaml"
  "4|ab4_mixed_w1.0_h45    |$CFG_DIR/act_with_future_dino.yaml"
  "5|ab5_mixed_w1.0_h45_vitb|$CFG_DIR/act_with_future_dino_vitb.yaml"
  "6|ab6_mixed_w1.0_h45_ln |$CFG_DIR/act_with_future_dino_ln.yaml"
  "7|ab7_mixed_w1.0_h45_shuf|$CFG_DIR/act_with_future_dino.yaml"
)

pids=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu name cfg <<<"$spec"
  name="$(echo "$name" | xargs)"; cfg="$(echo "$cfg" | xargs)"
  log="logs/perturb/${name}_${TAG}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/eval/perturb_eval.py \
    --name "$name" --ckpt "${name}_ckpt" --cfg "$cfg" \
    --val_dir "$VAL_DIR" --out "${OUT}_${name}_${TAG}.json" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "gpu=$gpu $name -> $log"
done

echo "waiting for ${#pids[@]} jobs ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$fail job(s) failed"

"$PY" scripts/eval/perturb_report.py "${OUT}"_*_"${TAG}".json
