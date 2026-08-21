#!/usr/bin/env bash
# Re-run the perturbation suite on the 7 checkpoints the conclusions actually rest on,
# to get the per-episode spread that the original sweep never stored.
#
# Why only these 7: every claim in docs/NEXT_RUNS.md is a difference between two of
# them, and three of those differences are 2-3mm -- ab4 vs ab7 (is the world head's
# gain about predicting the future?), ab4 vs r3_ab4_nostate (is the masked state worth
# 2.5mm?), r2_in_vjepa vs r3_vjepa_nostate (do the two improvements stack?). With one
# number per checkpoint none of the three can be told from noise. The other 10 runs are
# either superseded or differ by enough that the spread would not change the reading.
#
# The headline mean is unchanged by design (perturb_eval.py still averages the same flat
# per-frame list), so this does NOT invalidate any already-reported figure -- comparing
# the new --out json against the stored perturb_result.json is the check that it didn't.
#
# GPU 0 is skipped: r4_vjepa_w0 is training on it.
#
# Usage: run_perturb_errbars.sh [val_dir]   (default data/dex5_val)
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
CFG_DIR=hdt/configs/models
VAL_DIR="${1:-/home/aigc/human_policy/data/dex5_val}"
TAG="$(basename "$VAL_DIR")"
OUT=/tmp/errbar
mkdir -p logs/perturb

# gpu | name | ckpt dir | config it was TRAINED with (must match, or strict=False
#                                                     silently drops weights/buffers)
RUNS=(
  "1|ab4              |ab4_mixed_w1.0_h45_ckpt      |$CFG_DIR/act_with_future_dino.yaml"
  "2|ab7              |ab7_mixed_w1.0_h45_shuf_ckpt |$CFG_DIR/act_with_future_dino.yaml"
  "3|r2_in_vjepa      |r2_in_vjepa_ckpt             |$CFG_DIR/act_input_vjepa.yaml"
  "4|r3_ab4_nostate   |r3_ab4_nostate_ckpt          |$CFG_DIR/act_with_future_dino_nostate.yaml"
  "5|r3_vjepa_nostate |r3_vjepa_nostate_ckpt        |$CFG_DIR/act_input_vjepa_nostate.yaml"
  "6|r2_wv2           |r2_wv2_equal_ckpt            |$CFG_DIR/act_with_future_vae.yaml"
  "7|r2_wv3           |r2_wv3_shuf_ckpt             |$CFG_DIR/act_with_future_vae.yaml"
)

pids=()
for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu name ckpt cfg <<<"$spec"
  name="$(echo "$name" | xargs)"; ckpt="$(echo "$ckpt" | xargs)"; cfg="$(echo "$cfg" | xargs)"
  log="logs/perturb/errbar_${name}_${TAG}.log"
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" scripts/eval/perturb_eval.py \
    --name "$name" --ckpt "$ckpt" --cfg "$cfg" \
    --val_dir "$VAL_DIR" --out "${OUT}_${name}_${TAG}.json" \
    > "$log" 2>&1 &
  pids+=($!)
  echo "gpu=$gpu $name -> $log"
done

echo "waiting for ${#pids[@]} jobs ..."
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=$((fail+1)); done
echo "$fail job(s) failed"
grep -h "^== " logs/perturb/errbar_*_"${TAG}".log
