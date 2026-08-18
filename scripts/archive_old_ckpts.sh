#!/usr/bin/env bash
# Move historical checkpoint dirs off / (436G, was 100% full) onto /mnt/nvme0n1 (2.2T free),
# leaving a symlink behind so every existing path keeps working.
#
# Non-destructive by construction: rsync first, verify byte count AND file count match,
# only then remove the original. Anything that fails verification is left untouched and
# reported. Skips the ab*_ckpt dirs of the currently-running ablation.
set -uo pipefail

SRC=/home/aigc/human_policy
# /mnt/nvme0n1 itself is root-owned, so this must live under the one dir that was
# chowned to aigc. Bail loudly rather than silently failing every rsync.
DST=/mnt/nvme0n1/human_policy_ckpt/archive
mkdir -p "$DST" || { echo "无法创建 $DST -- 需要 sudo chown aigc:aigc 上层目录"; exit 1; }
[ -w "$DST" ] || { echo "$DST 不可写"; exit 1; }
cd "$SRC"

ok=0; skip=0; fail=0
for d in *_ckpt; do
  [ -d "$d" ] || continue
  [ -L "$d" ] && { echo "SKIP  $d (已是软链接)"; skip=$((skip+1)); continue; }
  case "$d" in ab[0-7]_*) echo "SKIP  $d (本轮消融正在写)"; skip=$((skip+1)); continue;; esac

  sz=$(du -sb "$d" | cut -f1); n=$(find "$d" -type f | wc -l)
  rsync -a --quiet "$d/" "$DST/$d/" || { echo "FAIL  $d (rsync)"; fail=$((fail+1)); continue; }
  sz2=$(du -sb "$DST/$d" | cut -f1); n2=$(find "$DST/$d" -type f | wc -l)

  if [ "$sz" = "$sz2" ] && [ "$n" = "$n2" ]; then
    rm -rf "$d" && ln -s "$DST/$d" "$d"
    echo "OK    $d  ($(numfmt --to=iec $sz), $n 文件)  -> 软链接"
    ok=$((ok+1))
  else
    echo "FAIL  $d  源 $sz/$n vs 目标 $sz2/$n2 -- 原目录保留未动"
    fail=$((fail+1))
  fi
done
echo "---- 搬走 $ok, 跳过 $skip, 失败 $fail"
df -h / | tail -1
