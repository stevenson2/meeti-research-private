#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cleanup_mimic_raw.py — 安全删除已处理/已校验的 MIMIC 原始 .hea/.dat。

安全规则:
  1. 默认 --dry-run, 只打印不删除;
  2. 只删除 validated_paths_*.txt 中列出的 stem;
  3. 一个 stem 必须 .hea/.dat 同时存在才删除;
  4. 若提供 --processed-manifest, 只删除 processed_ok=1 的记录, 双保险;
  5. 删除清单写 deleted_files_*.csv 留痕。

用法(试运行):
  python cleanup_mimic_raw.py --raw-dir mimic_raw_20k --validated validated_paths_20k.txt

用法(正式执行):
  python cleanup_mimic_raw.py --raw-dir mimic_raw_20k \
      --validated validated_paths_20k.txt \
      --processed-manifest processed_20k.csv --yes
"""
import argparse
import csv
import time
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--validated", required=True,
                    help="validate_mimic_subset.py 输出的 validated_paths_*.txt")
    ap.add_argument("--processed-manifest", default=None,
                    help="可选; 处理脚本输出, 只删 processed_ok=1 的记录")
    ap.add_argument("--yes", action="store_true", help="实际删除; 不加则为 dry-run")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    valid_path = Path(args.validated)
    stems = set(p.strip() for p in valid_path.read_text(encoding="utf-8").splitlines()
                if p.strip())

    if args.processed_manifest:
        pm = pd.read_csv(args.processed_manifest)
        if "processed_ok" not in pm.columns:
            raise SystemExit("processed_manifest 缺少 processed_ok 列")
        ok_stems = set(Path(x).name for x in
                       pm.loc[pm["processed_ok"] == 1, "stem"].astype(str))
        stems = stems & ok_stems
        print(f"[cleanup] processed_ok 过滤后待处理 stem: {len(stems)}")
    else:
        print(f"[cleanup] ⚠ 未提供 processed-manifest, 将删除全部已校验原始文件")

    pairs = []
    missing = 0
    for stem in sorted(stems):
        hea = raw / f"{stem}.hea"
        dat = raw / f"{stem}.dat"
        if hea.exists() and dat.exists():
            pairs.append((stem, hea, dat))
        else:
            missing += 1

    total_bytes = sum(hea.stat().st_size + dat.stat().st_size
                      for _, hea, dat in pairs)
    print(f"[cleanup] raw_dir={raw}")
    print(f"[cleanup] stem 候选={len(stems)} 成对存在={len(pairs)} "
          f"缺失跳过={missing}")
    print(f"[cleanup] 可释放={total_bytes/1024/1024:.1f} MB")

    if not args.yes:
        print("[cleanup] DRY-RUN: 未删除任何文件。确认后加 --yes 执行")
        return

    deleted = []
    for stem, hea, dat in pairs:
        hea.unlink(missing_ok=True)
        dat.unlink(missing_ok=True)
        deleted.append({"stem": stem, "hea": str(hea), "dat": str(dat),
                        "deleted_at": time.strftime("%Y-%m-%d %H:%M:%S")})

    out = raw / f"deleted_files_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(deleted).to_csv(out, index=False)
    print(f"[cleanup] 已删除 {len(deleted)} 条记录原始文件, "
          f"释放 {total_bytes/1024/1024:.1f} MB")
    print(f"[cleanup] 删除清单: {out}")


if __name__ == "__main__":
    main()
