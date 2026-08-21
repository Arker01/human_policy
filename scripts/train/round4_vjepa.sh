#!/usr/bin/env bash
# Round 4: the three remaining arms on the V-JEPA trunk (docs/NEXT_RUNS.md #2/#3/#4).
#
# GPU 0 is skipped -- r4_vjepa_w0 (#1, world head off) is already training there since
# 2026-08-20T06:47, ~13.9h. Together the four arms answer the whole question at once
# instead of serially:
#
#   w0        lambda=0          -- head removed          -> does the head earn its keep?
#   shuf      dino shuffled     -- head kept, time broken -> is it about the future?
#   dual      dino + Wan VAE    -- best input + best output
#   dual_shuf both shuffled     -- dual's negative control
#
# Running #4 without waiting for #3 is deliberate: a control that lands at the same
# time as its arm is worth more than one that lands 30h later, and on the resnet18
# trunk the dual control (wv3) is exactly where the surprise was.
#
# Cost: dual/dual_shuf measured at 1.10 s/it -> ~30h each; shuf matches r2_in_vjepa's
# 1.92 it/s -> ~14.5h. All three fit on separate GPUs, so wall clock is ~30h.
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json
CFG_DIR=hdt/configs/models
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt
mkdir -p logs/round4 "$CKPT_ROOT"

# Same guard as round 2: a half-written 2.8GB VAE mid-run is a confusing failure.
WAN=/home/aigc/.cache/huggingface/wan22_vae/diffusion_pytorch_model.safetensors
if [ ! -s "$WAN" ]; then echo "缺少 Wan2.2 VAE 权重: $WAN"; exit 1; fi

# gpu | exptid | model_cfg | dino_ablation | vae_flags
RUNS=(
  "1|r4_vjepa_shuf     |$CFG_DIR/act_input_vjepa.yaml      |shuffled|"
  "2|r4_vjepa_dual     |$CFG_DIR/act_input_vjepa_dual.yaml |none    |--use_future_vae_head --future_vae_weight 1.0"
  "3|r4_vjepa_dual_shuf|$CFG_DIR/act_input_vjepa_dual.yaml |shuffled|--use_future_vae_head --future_vae_weight 1.0 --future_vae_ablation shuffled"
)

only="${1:-all}"
for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu exptid mcfg dabl vaeflags <<<"$spec"
  exptid="$(echo "$exptid" | xargs)"; mcfg="$(echo "$mcfg" | xargs)"; dabl="$(echo "$dabl" | xargs)"
  [[ "$only" != "all" && "$only" != "$gpu" ]] && continue

  mkdir -p "$CKPT_ROOT/${exptid}_ckpt"
  [ -e "${exptid}_ckpt" ] || ln -s "$CKPT_ROOT/${exptid}_ckpt" "${exptid}_ckpt"

  log="logs/round4/${exptid}.log"
  echo "[$(date -Is)] gpu=$gpu $exptid cfg=$(basename $mcfg) dino_abl=$dabl vae='$vaeflags'" | tee "$log"

  CUDA_VISIBLE_DEVICES="$gpu" nohup "$PY" hdt/main.py \
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
    --use_future_dino_head \
    --future_dino_weight 1.0 \
    --future_dino_warmup_steps 0 \
    --future_dino_horizon 45 \
    --future_dino_ablation "$dabl" \
    $vaeflags \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 25   # stagger: several processes hitting the torch.hub cache at once wedges it
done
