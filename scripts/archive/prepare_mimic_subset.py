#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_mimic_subset.py — 从 MIMIC-IV-ECG record_list.csv 生成 MEETI 蒸馏用下载清单。

功能:
  1. 自动识别 record_list.csv 的 path/subject_id/study_id 列;
  2. 默认按 subject 去重后随机抽 N 条;
  3. 若提供 machine_measurements.csv, 则按机器报告关键词做 normal/abnormal 分层
     (默认 60% 正常倾向 / 40% 异常倾向, 不足时随机补齐);
  4. 输出:
       manifest_{N}.csv   — 选中记录清单 + 报告分类 + 报告文本
       urls_{N}.txt       — wget -i 可用 URL (.hea/.dat 各一条)
       manifest_{N}.json  — 统计信息

用法:
  python prepare_mimic_subset.py \
    --record-list /home/devcontainers/meeti/mimic/record_list.csv \
    --machine-csv  /home/devcontainers/meeti/mimic/machine_measurements.csv \
    --n 20000 --seed 42 \
    --out-dir /home/devcontainers/meeti/mimic

只读输入文件, 所有输出写到 out-dir, 不修改输入。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://physionet.org/files/mimic-iv-ecg/1.0"

# 机器报告关键词。报告字段为 report_0..report_17。
ABNORMAL_KEYWORDS = [
    "atrial fibrillation", "atrial flutter", "afib", "a fib",
    "supraventricular tachycardia", "ventricular tachycardia",
    "ventricular fibrillation", "premature ventricular",
    "premature atrial", "infarct", "infarction", "ischemia",
    "ischemic", "atrial enlargement", "left bundle branch block",
    "right bundle branch block", "bifascicular block", "trifascicular block",
    "first degree", "second degree", "third degree", "av block",
    "pacemaker", "paced rhythm", "prolonged qt", "short qt",
    "st elevation", "st depression", "t wave inversion",
]
NORMAL_KEYWORDS = [
    "normal sinus rhythm", "sinus rhythm",
    "normal ecg", "within normal limits",
]


def find_col(df: pd.DataFrame, names: tuple, extra_ok: bool = False):
    for c in df.columns:
        if str(c).strip().lower() in names:
            return c
    if extra_ok:
        for c in df.columns:
            if all(s in str(c).strip().lower() for s in names):
                return c
    raise SystemExit(f"未找到列, 实际列: {df.columns.tolist()}")


def classify_reports(report_text: pd.Series) -> pd.Series:
    """返回 normal / abnormal / unknown 三分类。大小写不敏感。"""
    text = report_text.fillna("").astype(str).str.lower()
    is_abn = np.zeros(len(text), dtype=bool)
    for kw in ABNORMAL_KEYWORDS:
        is_abn |= text.str.contains(kw, regex=False, na=False)

    is_normal_hint = np.zeros(len(text), dtype=bool)
    for kw in NORMAL_KEYWORDS:
        is_normal_hint |= text.str.contains(kw, regex=False, na=False)

    cls = pd.Series("unknown", index=text.index, dtype="object")
    cls[is_abn] = "abnormal"
    cls[~is_abn & is_normal_hint] = "normal"
    return cls


def build_record_path_url(path: str) -> str:
    p = str(path).strip().rstrip("/")
    if not p.startswith("files/"):
        raise SystemExit(f"记录 path 不符合预期: {p!r}")
    return f"{BASE_URL}/{p}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--record-list", required=True)
    ap.add_argument("--machine-csv", default=None,
                    help="可选; 提供时做 normal/abnormal 分层")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--normal-ratio", type=float, default=0.6,
                    help="目标 normal 记录比例, 不足时随机补齐")
    ap.add_argument("--out-dir", default="/home/devcontainers/meeti/mimic")
    ap.add_argument("--max-per-subject", type=int, default=1,
                    help="每个 subject 最多保留几条, 默认 1 避免同患者重复")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    rec = pd.read_csv(args.record_list, low_memory=False)
    print(f"[1/5] record_list: {len(rec):,} rows, columns={rec.columns.tolist()}",
          flush=True)

    path_col = find_col(rec, ("path",), extra_ok=True)
    subj_col = find_col(rec, ("subject_id",), extra_ok=True)
    study_col = find_col(rec, ("study_id",), extra_ok=True)

    rec = rec.rename(columns={path_col: "record_path",
                              subj_col: "subject_id",
                              study_col: "study_id"})
    rec["record_path"] = rec["record_path"].astype(str).str.strip().str.rstrip("/")
    rec = rec[rec["record_path"].str.startswith("files/")]
    print(f"[1/5] 有效 files/ 路径: {len(rec):,}", flush=True)

    # ---------- 可选: 机器报告分类 ----------
    if args.machine_csv:
        mach = pd.read_csv(
            args.machine_csv, low_memory=False,
            usecols=lambda c: c in ("study_id",) or str(c).startswith("report_"),
        )
        report_cols = [c for c in mach.columns if str(c).startswith("report_")]
        if not report_cols:
            raise SystemExit("machine_measurements.csv 没有 report_* 列")
        # 机器报告通常一行一条 study; 若有重复取第一行
        mach = mach.drop_duplicates(subset="study_id", keep="first")
        mach["report_text"] = (
            mach[report_cols].astype(str).fillna("").agg(" | ".join, axis=1)
        )
        mach["report_text"] = mach["report_text"].str.strip(" |")
        mach = mach[["study_id", "report_text"]]
        rec = rec.merge(mach, on="study_id", how="left")
        rec["report_text"] = rec["report_text"].fillna("")
        rec["report_class"] = classify_reports(rec["report_text"])
    else:
        rec["report_text"] = ""
        rec["report_class"] = "unknown"

    print("[2/5] 报告分类分布:",
          rec["report_class"].value_counts(dropna=False).to_dict(), flush=True)

    # ---------- 每 subject 去重 ----------
    if args.max_per_subject > 0:
        rec = (
            rec.sort_values(["subject_id", "record_path"])
            .groupby("subject_id", sort=False)
            .head(args.max_per_subject)
        )
        print(f"[3/5] 每 subject 最多 {args.max_per_subject} 条后: {len(rec):,}",
              flush=True)

    # ---------- 分层抽样 ----------
    rng = np.random.default_rng(args.seed)
    if args.machine_csv:
        n_normal = int(round(args.n * args.normal_ratio))
        n_abnormal = args.n - n_normal
        avail_normal = rec[rec["report_class"] == "normal"]
        avail_abnormal = rec[rec["report_class"] == "abnormal"]
        n_norm_use = min(n_normal, len(avail_normal))
        n_abn_use = min(n_abnormal, len(avail_abnormal))
        chosen = []
        if n_norm_use:
            chosen.append(avail_normal.sample(n=n_norm_use, random_state=args.seed))
        if n_abn_use:
            chosen.append(avail_abnormal.sample(n=n_abn_use, random_state=args.seed + 1))
        chosen = pd.concat(chosen) if chosen else avail_normal.iloc[0:0]

        need = args.n - len(chosen)
        if need > 0:
            already = set(zip(chosen["subject_id"], chosen["record_path"]))
            rest = rec[~rec.set_index(["subject_id", "record_path"]).index.isin(already)]
            # 优先 unknown, 再全体随机补齐
            unk = rest[rest["report_class"] == "unknown"]
            fill = unk.sample(n=min(need, len(unk)), random_state=args.seed + 2)
            if len(fill) < need:
                rest2 = rest.drop(fill.index)
                fill = pd.concat([fill, rest2.sample(n=need - len(fill),
                                                     random_state=args.seed + 3)])
            chosen = pd.concat([chosen, fill])
        chosen = chosen.sample(frac=1.0, random_state=args.seed).head(args.n)
    else:
        chosen = rec.sample(n=min(args.n, len(rec)), random_state=args.seed)

    print(f"[4/5] 选中: {len(chosen):,} 条; "
          f"分类分布={chosen['report_class'].value_counts().to_dict()}", flush=True)

    # ---------- URL 与 manifest ----------
    rows = []
    urls = []
    for _, row in chosen.iterrows():
        p = str(row["record_path"]).rstrip("/")
        if p.endswith(".hea") or p.endswith(".dat"):
            base = p[:-4] if p.endswith((".hea", ".dat")) else p
            stem_url = f"{BASE_URL}/{base}"
        else:
            stem_url = f"{BASE_URL}/{p}"
        urls.append(stem_url + ".hea")
        urls.append(stem_url + ".dat")
        rows.append({
            "record_path": p,
            "stem": p[:-4] if p.endswith((".hea", ".dat")) else p,
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "report_class": row.get("report_class", "unknown"),
            "report_text": row.get("report_text", ""),
            "url_hea": stem_url + ".hea",
            "url_dat": stem_url + ".dat",
        })

    manifest = pd.DataFrame(rows)
    manifest.to_csv(out / f"manifest_{args.n}.csv", index=False)
    with open(out / f"urls_{args.n}.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(urls) + "\n")

    summary = {
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "source_record_list": str(args.record_list),
        "machine_csv": str(args.machine_csv) if args.machine_csv else None,
        "n_requested": args.n,
        "n_selected": int(len(manifest)),
        "n_urls": len(urls),
        "est_mb": round(len(manifest) * 120 / 1024, 1),
        "class_dist": chosen["report_class"].value_counts().to_dict(),
        "outputs": {
            "manifest": str(out / f"manifest_{args.n}.csv"),
            "urls": str(out / f"urls_{args.n}.txt"),
            "summary": str(out / f"manifest_{args.n}.json"),
        },
    }
    with open(out / f"manifest_{args.n}.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[5/5] 输出: manifest_{args.n}.csv / urls_{args.n}.txt "
          f"(预计 {summary['est_mb']} MB, 耗时 {time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
