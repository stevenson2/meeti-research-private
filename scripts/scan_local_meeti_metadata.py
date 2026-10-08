#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scan_local_meeti_metadata.py — 扫描本地 MEETI 目录中已下载的 .mat 元数据。

用于回答: 现在到底下载了什么、哪些 study 有 report/LLM 文本、字段是否完整。
不会下载/修改任何文件。

用法:
  python scan_local_meeti_metadata.py \
    --root "/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/new_papers/dataests/MEETI" \
    --out /home/devcontainers/meeti/docs/state/local_meeti_metadata_manifest.csv \
    [--limit 2000]   # 冒烟测试
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

KNOWN_KEYS = {
    "id", "signal", "report", "LLM_Interpretation",
    *[f"featuredb_lead_{x}" for x in
      ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]],
}


def safe_read_mat(path: Path):
    try:
        d = loadmat(str(path), squeeze_me=False, struct_as_record=True)
    except Exception as e:
        return None, str(e)
    keys = [k for k in d.keys() if not k.startswith("__")]
    row = {
        "path": str(path),
        "size_bytes": int(path.stat().st_size),
        "n_keys": len(keys),
        "keys": ",".join(keys),
        "has_id": "id" in keys,
        "has_signal": "signal" in keys,
        "has_report": "report" in keys,
        "has_llm": "LLM_Interpretation" in keys,
        "n_featuredb_leads": sum(1 for k in keys if k.startswith("featuredb_lead_")),
        "unknown_keys": ",".join(k for k in keys if k not in KNOWN_KEYS),
    }
    # 轻量提取文本字段
    for field in ["id", "report", "LLM_Interpretation"]:
        if field in keys:
            try:
                v = np.asarray(d[field], dtype=object).ravel()
                row[field] = str(v[0])[:500] if len(v) else ""
            except Exception:
                row[field] = ""
        else:
            row[field] = ""
    if "signal" in keys:
        try:
            row["signal_shape"] = str(np.asarray(d["signal"]).shape)
        except Exception:
            row["signal_shape"] = "?"
    else:
        row["signal_shape"] = ""
    return row, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="/home/devcontainers/meeti/docs/state/local_meeti_metadata_manifest.csv")
    ap.add_argument("--limit", type=int, default=None, help="最多扫描文件数")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"目录不存在: {root}")

    t0 = time.time()
    rows = []
    errors = 0
    n_files = 0
    for mat in root.rglob("*.mat"):
        n_files += 1
        if args.limit and n_files > args.limit:
            break
        row, err = safe_read_mat(mat)
        if row is None:
            errors += 1
            rows.append({"path": str(mat), "error": err})
        else:
            rows.append(row)
        if n_files % 1000 == 0:
            print(f"  scanned {n_files}, errors={errors}, "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "root": str(root),
        "scanned_mat_files": n_files,
        "read_errors": errors,
        "total_rows": int(len(df)),
        "n_with_id": int(df["has_id"].sum()) if "has_id" in df else 0,
        "n_with_signal": int(df["has_signal"].sum()) if "has_signal" in df else 0,
        "n_with_report": int(df["has_report"].sum()) if "has_report" in df else 0,
        "n_with_llm": int(df["has_llm"].sum()) if "has_llm" in df else 0,
        "n_with_featuredb": int((df["n_featuredb_leads"] > 0).sum()) if "n_featuredb_leads" in df else 0,
        "elapsed_s": round(time.time() - t0, 1),
    }
    out_summary = out.with_suffix(".json")
    out_summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"manifest: {out}")
    print(f"summary:  {out_summary}")


if __name__ == "__main__":
    main()
