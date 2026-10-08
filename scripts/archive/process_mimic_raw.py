#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_mimic_raw.py — MIMIC 原始 .hea/.dat → 学生 250 点拍窗 + 教师 5s 段缓存。
输出全部写到 /home/devcontainers/meeti/data/processed_small。
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ/'pc_tools/ecg_dl'))
from eval_deploy_match import corrected_deployment_chain, extract_beats_deploy

LABEL_MAP = {"normal": 0, "abnormal": 1, "unknown": 2}


def process_record(rec_path: Path, save_teacher: bool):
    rec = wfdb.rdrecord(str(rec_path.with_suffix('')))
    sig = rec.p_signal  # (5000,12) float64
    if sig.shape != (5000, 12):
        raise RuntimeError(f"unexpected signal shape {sig.shape}")
    lead2 = sig[:, 1].astype(np.float64)
    s250 = corrected_deployment_chain(lead2, 500)
    # 对齐/截断为预期长度 2500
    n = int(len(lead2) * 250 / 500)
    s250 = s250[:n]
    try:
        from wfdb.processing import xqrs_detect
        peaks = xqrs_detect(s250.astype(np.float64), fs=250, verbose=False)
    except Exception:
        # 回退: 对 8-25Hz 带通后的平方信号做峰值检测
        from scipy.signal import butter, filtfilt, find_peaks
        b, a = butter(2, [8/125, 25/125], btype='band')
        y = filtfilt(b, a, s250)
        peaks, _ = find_peaks(y*y, distance=100, height=np.quantile(y*y, 0.6))
    beats = extract_beats_deploy(s250, np.asarray(peaks, dtype=np.int64), "ptb")
    if save_teacher:
        seg0 = sig[:2500].astype(np.float32)
        seg1 = sig[2500:5000].astype(np.float32)
    else:
        seg0 = seg1 = None
    return beats.astype(np.float32), seg0, seg1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="/home/devcontainers/meeti/data/selected8k/selected_manifest.csv")
    ap.add_argument("--raw-dir", default="/mnt/d/MIMIC_ECG_raw")
    ap.add_argument("--out-dir", default="/home/devcontainers/meeti/data/processed_small")
    ap.add_argument("--save-teacher", action="store_true")
    ap.add_argument("--max-records", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_dir)
    df = pd.read_csv(args.manifest)
    if "record_path" not in df.columns:
        raise SystemExit("manifest 缺少 record_path")

    beats_all, labels_all, rids, sids, report_texts = [], [], [], [], []
    teacher_segs = [] if args.save_teacher else None
    processed = []
    n_done = n_skip = n_fail = 0
    t0 = time.time()

    for i, row in df.iterrows():
        if args.max_records and n_done >= args.max_records:
            break
        stem = Path(str(row["record_path"])).name
        rec_path = raw / stem
        if not (rec_path.with_suffix(".hea").exists() and rec_path.with_suffix(".dat").exists()):
            n_skip += 1
            continue
        try:
            beats, seg0, seg1 = process_record(rec_path, args.save_teacher)
        except Exception as e:
            n_fail += 1
            if n_fail <= 20:
                print(f"fail {stem}: {e}", flush=True)
            continue
        if len(beats) == 0:
            n_skip += 1
            continue
        beats_all.append(beats)
        lab = LABEL_MAP.get(str(row.get("report_class", "unknown")), 2)
        labels_all.append(np.full(len(beats), lab, dtype=np.int8))
        rids.append(np.full(len(beats), int(row.get("study_id", 0)), dtype=np.int64))
        sids.append(np.full(len(beats), str(row.get("subject_id", "")), dtype=object))
        report_texts.append(str(row.get("report", "")))
        if args.save_teacher:
            teacher_segs.append(np.stack([seg0, seg1], axis=0))
        processed.append({"stem": stem, "study_id": row.get("study_id"),
                          "subject_id": row.get("subject_id"),
                          "n_beats": int(len(beats)), "processed_ok": 1,
                          "report_class": row.get("report_class", "unknown")})
        n_done += 1
        if n_done % 500 == 0:
            print(f"done={n_done} skip={n_skip} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)

    if not beats_all:
        print("no processed records", flush=True)
        return

    beats = np.concatenate(beats_all).astype(np.float32)
    labels = np.concatenate(labels_all).astype(np.int8)
    rids = np.concatenate(rids).astype(np.int64)
    np.save(out/"beats.npy", beats)
    np.save(out/"labels_weak.npy", labels)
    np.save(out/"record_ids.npy", rids)
    pd.DataFrame({"subject_id": [r["subject_id"] for r in processed for _ in range(r["n_beats"])]}).to_csv(out/"beat_meta.csv", index=False)
    if teacher_segs is not None:
        np.save(out/"teacher_segments.npy", np.concatenate(teacher_segs, axis=0).astype(np.float32))
    pm = pd.DataFrame(processed)
    pm.to_csv(out/"processed_manifest.csv", index=False)
    summary = {"processed_records": n_done, "skipped": n_skip, "failed": n_fail,
               "n_beats": int(len(beats)),
               "weak_label_dist": pd.Series(labels).value_counts().to_dict(),
               "elapsed_s": round(time.time()-t0, 1)}
    (out/"summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(summary, flush=True)


if __name__ == "__main__":
    main()
