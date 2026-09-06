#!/usr/bin/env bash
# pick-bottle 任务的 3D flow 目标（docs/FUTURE_FLOW_CHANGES.md）。
#
# 和 run_flow_target_all.sh 是同一件事，只换了 episode 目录和输出目录。
# 30Hz 和 10Hz 两版机器人数据共用一个输出目录：输出名是 episode 路径的 hash，
# 两版的目录名不同所以不会撞；而人类那两个目录在两版里是同一份，共用一个目录
# 就只算一次，省掉一半人类侧的计算和约 6GB 磁盘。
#
# episode 路径必须和 dataloader 拼出来的字符串完全一致
# （data_utils_hdt.py: os.path.join(task_dir, fn)），因为 flow_target_filename()
# 是对路径做 hash 得到输出名，且 abspath 不解析 symlink。下面六个目录都是
# symlink 目录，不要用 realpath 展开，否则训练时一条目标都命中不了。
set -uo pipefail

cd /home/aigc/human_policy
PY=/home/aigc/miniconda/envs/track4world/bin/python
OUT=/mnt/nvme0n1/human_policy_ckpt/flow_target_bottle
DATA=/home/aigc/human_policy/data
NSHARD=8
mkdir -p logs/flow_target_bottle "$OUT"

CKPT=/home/aigc/Track4World/checkpoints/track4world_da3.pth
if [ ! -s "$CKPT" ]; then echo "缺少 Track4World DA3 权重: $CKPT"; exit 1; fi

EPS=("$DATA/bottle_human_train"      "$DATA/bottle_human_val" \
     "$DATA/bottle_robot_30hz_train" "$DATA/bottle_robot_30hz_val" \
     "$DATA/bottle_robot_10hz_train" "$DATA/bottle_robot_10hz_val")

for d in "${EPS[@]}"; do
  if [ ! -d "$d" ]; then echo "缺少 episode 目录: $d"; exit 1; fi
done

TOTAL=0
for d in "${EPS[@]}"; do TOTAL=$((TOTAL + $(ls "$d"/*.hdf5 2>/dev/null | wc -l))); done
echo "episodes: $TOTAL -> $OUT"

only="${1:-all}"
for s in $(seq 0 $((NSHARD - 1))); do
  [[ "$only" != "all" && "$only" != "$s" ]] && continue
  log="logs/flow_target_bottle/shard${s}.log"
  echo "[$(date -Is)] gpu=$s shard=$s/$NSHARD -> $OUT" | tee "$log"

  CUDA_VISIBLE_DEVICES="$s" nohup "$PY" scripts/preprocess/flow_target.py \
    --episodes "${EPS[@]}" \
    --out_dir "$OUT" \
    --num_shards "$NSHARD" --shard "$s" \
    --skip_existing \
    >> "$log" 2>&1 &

  echo "  -> pid $! log $log"
  sleep 15   # stagger: 8 个进程同时建 DA3 会把权重加载卡住
done
