#!/usr/bin/env bash
# download_mimic_subset.sh — 断点续传下载 MIMIC-IV-ECG 子集。
# 用法:
#   bash download_mimic_subset.sh urls_20000.txt mimic_raw_20k [log.txt]
set -euo pipefail

URLS="${1:?用法: download_mimic_subset.sh <urls.txt> <dest_dir> [log.txt]}"
DEST="${2:?缺少目标目录}"
LOG="${3:-$DEST/download.log}"

mkdir -p "$DEST"

echo "[download] urls=$URLS dest=$DEST log=$LOG"
echo "[download] 若 403, 请在浏览器登录 PhysioNet 后重试, 或改用:"
echo "            wget -c -i '$URLS' -P '$DEST' --user=<physionet用户名> --ask-password"

wget -c -i "$URLS" -P "$DEST" \
  --wait=0.2 --random-wait --tries=3 --timeout=30 \
  -o "$LOG"

n_hea=$(find "$DEST" -maxdepth 1 -name '*.hea' | wc -l)
n_dat=$(find "$DEST" -maxdepth 1 -name '*.dat' | wc -l)
echo "[download] done. .hea=$n_hea .dat=$n_dat"
