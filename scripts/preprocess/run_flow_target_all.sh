#!/usr/bin/env bash
# Full-corpus flow targets for round 5 (docs/FUTURE_FLOW_CHANGES.md).
#
# 207 episodes (human_train 93 + human_val 11 + dex5_train_s93 93 + dex5_val 10),
# round-robin over 8 free GPUs. Each shard loads its own Track4World/DA3 copy;
# they are independent processes with no shared state.
#
# The episode paths must be EXACTLY the strings the dataloader builds
# (data_utils_hdt.py: os.path.join(task_dir, fn)), because flow_target_filename()
# hashes the path into the output name. dex5_train_s93 is a directory of symlinks
# into convert_UnifoLM_WBT_inspire_rootfix/ -- do NOT resolve them, abspath in
# flow_target_filename does not resolve either, so the two sides agree only if we
# feed the symlink directory.
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/track4world/bin/python
# Under human_policy_ckpt/ rather than at the nvme root because the root is
# root-owned and we have no password; this is the same device (the one that
# matters -- / is a spinning disk and the dataloader needs ~92MB/s random read).
OUT=/mnt/nvme0n1/human_policy_ckpt/flow_target_dex5
DATA=/home/aigc/human_policy/data
NSHARD=8
mkdir -p logs/flow_target "$OUT"

# Same guard as round 2/4: a half-downloaded checkpoint mid-run is a confusing failure.
CKPT=/home/aigc/Track4World/checkpoints/track4world_da3.pth
if [ ! -s "$CKPT" ]; then echo "缺少 Track4World DA3 权重: $CKPT"; exit 1; fi

EPS=("$DATA/human_train" "$DATA/human_val" "$DATA/dex5_train_s93" "$DATA/dex5_val")

only="${1:-all}"
for s in $(seq 0 $((NSHARD - 1))); do
  [[ "$only" != "all" && "$only" != "$s" ]] && continue
  log="logs/flow_target/shard${s}.log"
  echo "[$(date -Is)] gpu=$s shard=$s/$NSHARD -> $OUT" | tee "$log"

  CUDA_VISIBLE_DEVICES="$s" nohup "$PY" scripts/preprocess/flow_target.py \
    --episodes "${EPS[@]}" \
    --out_dir "$OUT" \
    --num_shards "$NSHARD" --shard "$s" \
    --skip_existing \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 15   # stagger: 8 processes building DA3 at once thrashes the weight load
done
