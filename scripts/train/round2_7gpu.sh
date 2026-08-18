#!/usr/bin/env bash
# Round 2, 7 runs on 7 GPUs. Two independent questions, one launcher.
#
# ---------------------------------------------------------------------------
# Question A (GPU 0-1): what should the trunk LOOK THROUGH?
#   ab4 already is the resnet18 arm -- trained end to end at lr_backbone 1e-5.
#   These two swap only the input encoder, everything else identical to ab4.
#   Confound that cannot be removed: both new encoders are frozen (they run under
#   no_grad in backbone.py, inherited from the pre-existing DINOv2BackBone) while
#   resnet18 trains. So a loss to ab4 is NOT proof the encoder is worse.
#   Published expectation is that this does not win: DexWM's own encoder sweep
#   (DINOv2 / DINOv3 / Web-SSL / V-JEPA 2 / SigLIP 2) put DINOv2 first, and no
#   paper has V-JEPA beating DINOv2 or a trained resnet in this slot.
#
# Question B (GPU 2-6): does adding a VAE target to the DINO one help?
#   ab4 is the DINO-only arm (lambda_dino 1.0, no VAE) -- do not retrain it.
#   ST-WAM regresses both species and reports the VAE half is the load-bearing
#   one (72.8 -> 63.5 without DINO, 72.8 -> 39.7 without VAE, ratio 1.0 : 0.02).
#   Our entire ab0..ab7 suite only had the half they say does the least.
#
#   gpu  arm    lambda_vae  lambda_dino  vae target norm  ablation | reads as
#    2   wv0       1.0          0.0            raw          none   | VAE only
#    3   wv1       1.0          0.02           raw          none   | ST-WAM ratio
#    4   wv2       1.0          1.0            raw          none   | both, equal
#    5   wv3       1.0          0.02           raw        shuffled | negative control
#    6   wv4       1.0          0.02           l2           none   | target-norm knob
#
#   wv3 is the one to read first. On the DINO-only suite the shuffled control TIED
#   with ab4 (9.8% vs 9.5% mean degradation), i.e. the gain was distillation, not
#   future prediction. If wv1 beats ab4 but wv3 matches wv1, the same is true of
#   the VAE and nothing here is a world model either.
#
# GPU 7 is left free on purpose: it is where the perturbation probes and any
# interactive diagnostics run while these are training.
#
# Cost: the Wan VAE encoder is the expensive part -- +0.51s/step for 128 frames at
# 224x304 in bf16, on top of ab4's 0.35s/step. Expect ~24h for the wv* arms and
# ~14-20h for the input-encoder arms, vs ab4's 9h50m.
#
# Checkpoints go straight to /mnt/nvme0n1 via a pre-created symlink: main.py builds
# ckpt_dir as a relative "<exptid>_ckpt" and only mkdir's it when it is not already
# a directory, so a symlink planted here is honoured. / has 156G free and these
# would fill it.
#
# Usage: round2_7gpu.sh [all|<gpu>]
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/human_policy/bin/python
BASE_DIR=/home/aigc/human_policy/data
CFG_DIR=hdt/configs/models
CKPT_ROOT=/mnt/nvme0n1/human_policy_ckpt
MIXED=/home/aigc/human_policy/pickup_pillow_mixed_1to1.json

mkdir -p logs/round2 "$CKPT_ROOT"

# The Wan VAE is not fetched on demand -- a half-written 2.8GB file mid-run is a
# confusing failure, so fail loudly here instead.
WAN=/home/aigc/.cache/huggingface/wan22_vae/diffusion_pytorch_model.safetensors
if [ ! -s "$WAN" ]; then
  echo "缺少 Wan2.2 VAE 权重: $WAN"; exit 1
fi

# gpu | exptid | model_cfg | dino_weight | vae_flags
RUNS=(
  "0|r2_in_dinov2  |$CFG_DIR/act_input_dinov2.yaml     |1.0 |"
  "1|r2_in_vjepa   |$CFG_DIR/act_input_vjepa.yaml      |1.0 |"
  "2|r2_wv0_vaeonly|$CFG_DIR/act_with_future_vae.yaml  |0.0 |--use_future_vae_head --future_vae_weight 1.0"
  "3|r2_wv1_stwam  |$CFG_DIR/act_with_future_vae.yaml  |0.02|--use_future_vae_head --future_vae_weight 1.0"
  "4|r2_wv2_equal  |$CFG_DIR/act_with_future_vae.yaml  |1.0 |--use_future_vae_head --future_vae_weight 1.0"
  "5|r2_wv3_shuf   |$CFG_DIR/act_with_future_vae.yaml  |0.02|--use_future_vae_head --future_vae_weight 1.0 --future_vae_ablation shuffled"
  "6|r2_wv4_l2norm |$CFG_DIR/act_with_future_vae.yaml  |0.02|--use_future_vae_head --future_vae_weight 1.0 --future_vae_normalize_target 1"
)

only="${1:-all}"

for spec in "${RUNS[@]}"; do
  IFS='|' read -r gpu exptid mcfg dweight vaeflags <<<"$spec"
  exptid="$(echo "$exptid" | xargs)"; mcfg="$(echo "$mcfg" | xargs)"; dweight="$(echo "$dweight" | xargs)"
  [[ "$only" != "all" && "$only" != "$gpu" ]] && continue

  # wv3's negative control has to break the pairing for BOTH species, otherwise the
  # intact DINO half would still be learning the real future.
  dabl=none
  [[ "$exptid" == "r2_wv3_shuf" ]] && dabl=shuffled

  # pre-create the checkpoint dir on the big disk and point the relative name at it
  mkdir -p "$CKPT_ROOT/${exptid}_ckpt"
  [ -e "${exptid}_ckpt" ] || ln -s "$CKPT_ROOT/${exptid}_ckpt" "${exptid}_ckpt"

  log="logs/round2/${exptid}.log"
  echo "[$(date -Is)] gpu=$gpu $exptid cfg=$(basename $mcfg) dino_w=$dweight dino_abl=$dabl vae='$vaeflags'" | tee "$log"

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
    --future_dino_weight "$dweight" \
    --future_dino_warmup_steps 0 \
    --future_dino_horizon 45 \
    --future_dino_ablation "$dabl" \
    $vaeflags \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 25   # stagger: 7 processes hitting the dinov2 hub cache at once wedges it
done

wait
