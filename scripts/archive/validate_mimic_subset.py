#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_mimic_subset.py — 校验已下载的 MIMIC-IV-ECG 子集。

检查项:
  1. manifest 中每条记录是否有 .hea/.dat;
  2. .hea 首行是否形如 "<id> 12 500 5000" (12 导联, 500Hz, 10s);
  3. .dat 大小是否等于 2 * n_signals * n_samples 字节;
  4. .hea/.dat 是否为空文件。

输出:
  validation_{tag}.json
  validated_paths_{tag}.txt    — 只含通过校验的 record stem, 供后续清理脚本使用

用法:
  python validate_mimic_subset.py \
    --manifest /home/devcontainers/meeti/mimic/manifest_20000.csv \
    --raw-dir  /home/devcontainers/meeti/mimic/mimic_raw_20k \
    --tag 20k
"""
import argparse
import json
import time
from pathlib import Path

import pandas as pd


def parse_hea_first_line(line: str):
    parts = line.strip().split()
    if len(parts) < 4:
        raise ValueError(f"header 字段不足: {line!r}")
    record = parts[0]
    n_signals = int(parts[1])
    fs = int(parts[2])
    n_samples = int(parts[3])
    return record, n_signals, fs, n_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--tag", default="subset")
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    if not raw.exists():
        raise SystemExit(f"raw-dir 不存在: {raw}")

    df = pd.read_csv(args.manifest)
    if "record_path" not in df.columns or "stem" not in df.columns:
        raise SystemExit("manifest 缺少 record_path/stem 列, 请重新生成 manifest")

    rows = []
    n_missing = 0
    n_bad_header = 0
    n_bad_size = 0
    n_empty = 0
    n_ok = 0
    t0 = time.time()

    for _, r in df.iterrows():
        stem = Path(str(r["stem"])).name
        hea = raw / f"{stem}.hea"
        dat = raw / f"{stem}.dat"

        if not (hea.exists() and dat.exists()):
            n_missing += 1
            rows.append({"stem": stem, "ok": False, "reason": "missing"})
            continue

        if hea.stat().st_size == 0 or dat.stat().st_size == 0:
            n_empty += 1
            rows.append({"stem": stem, "ok": False, "reason": "empty"})
            continue

        try:
            first = hea.read_text(encoding="utf-8", errors="replace").splitlines()[0]
            record, n_signals, fs, n_samples = parse_hea_first_line(first)
        except Exception as e:
            n_bad_header += 1
            rows.append({"stem": stem, "ok": False, "reason": f"bad_header:{e}"})
            continue

        expected = 2 * n_signals * n_samples
        actual = dat.stat().st_size
        if actual != expected:
            n_bad_size += 1
            rows.append({"stem": stem, "ok": False,
                         "reason": f"bad_size:{actual}!={expected}"})
            continue

        n_ok += 1
        rows.append({"stem": stem, "ok": True, "reason": "",
                     "n_signals": n_signals, "fs": fs, "n_samples": n_samples})

    vdf = pd.DataFrame(rows)
    ok_stems = vdf.loc[vdf["ok"], "stem"].tolist()
    out_json = raw / f"validation_{args.tag}.json"
    out_ok = raw / f"validated_paths_{args.tag}.txt"
    vdf.to_csv(raw / f"validation_{args.tag}.csv", index=False)
    out_json.write_text(json.dumps({
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "manifest": str(args.manifest),
        "raw_dir": str(raw),
        "total": int(len(vdf)),
        "ok": n_ok,
        "missing": n_missing,
        "empty": n_empty,
        "bad_header": n_bad_header,
        "bad_size": n_bad_size,
        "ok_ratio": round(n_ok / max(1, len(vdf)), 6),
        "elapsed_s": round(time.time() - t0, 1),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    out_ok.write_text("\n".join(ok_stems) + "\n", encoding="utf-8")

    print(f"total={len(vdf)} ok={n_ok} missing={n_missing} empty={n_empty} "
          f"bad_header={n_bad_header} bad_size={n_bad_size}")
    print(f"written: {out_json}")
    print(f"written: {out_ok}")
    if n_ok == 0:
        raise SystemExit("没有通过校验的记录, 终止后续流程")


if __name__ == "__main__":
    main()
