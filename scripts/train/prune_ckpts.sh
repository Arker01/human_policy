#!/usr/bin/env bash
# Keep only the newest N policy_iter_* checkpoints per run directory.
#
# Why this exists: the 2026-08-16 ablation launch filled / at ~50k/100k steps and
# killed all 8 runs. accelerate saves optimizer.bin (815MB) next to
# pytorch_model.bin (408MB), so every checkpoint costs ~1.15GB -- 8 runs x 10
# checkpoints = 92GB, which does not fit. The ablation only needs the final
# checkpoint per run, so keeping the newest 2 (one spare in case the newest is a
# partial write) caps peak usage at ~18GB.
#
# Deliberately a separate watchdog rather than a change to main.py's save logic,
# which is pre-existing code.
#
# Usage: prune_ckpts.sh [keep] [interval_sec] [pattern]
set -uo pipefail

keep="${1:-2}"
interval="${2:-120}"
pattern="${3:-ab*_ckpt}"
cd /home/aigc/human_policy

while true; do
  for d in $pattern; do
    [ -d "$d" ] || continue
    # sort numerically by step so "policy_iter_100000" beats "policy_iter_90000"
    mapfile -t ck < <(ls -d "$d"/policy_iter_*_seed_* 2>/dev/null \
      | sed -E 's/.*policy_iter_([0-9]+)_seed.*/\1 &/' | sort -k1,1nr | cut -d' ' -f2-)
    n=${#ck[@]}
    (( n <= keep )) && continue
    for ((i=keep; i<n; i++)); do
      # never touch something accelerate may still be writing
      if [ -n "$(find "${ck[$i]}" -newermt '-300 seconds' -print -quit 2>/dev/null)" ]; then
        continue
      fi
      rm -rf "${ck[$i]}" && echo "[$(date -Is)] pruned ${ck[$i]}"
    done
  done
  avail=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if (( avail < 8 )); then
    echo "[$(date -Is)] WARNING: only ${avail}G left on /"
  fi
  sleep "$interval"
done
