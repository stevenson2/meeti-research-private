#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3: process MIMIC raw .hea/.dat -> beat-centered 12-lead windows.

Outputs (all inside --out-dir, default data/processed_small/m3_beat_centered):
  beat_windows.npy       (n_beats, 12, 2500) float32, original 500 Hz,
                         center +/-2.5 s around the R peak, edge zero-padded
  beat_positions.npy     (n_beats,) int64, R peak position in the 250 Hz chain
  record_ids.npy         (n_beats,) int64, MIMIC study_id
  beat_meta.csv          per-beat metadata incl. weak label / padding / mask
  student_beats_lead2.npy (n_beats, 250) firmware-chain lead-II beat (zero where invalid)
  student_beat_mask.npy  (n_beats,) bool: strict 250-pt window fits the 250 Hz chain
  summary.json
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from eval_deploy_match import (  # noqa: E402
    corrected_deployment_chain, extract_beats_deploy,
)

WINDOW_SAMPLES = 2500   # 5 s at 500 Hz
HALF = WINDOW_SAMPLES // 2
BEAT_SAMPLES = 250
BEAT_HALF = BEAT_SAMPLES // 2
RAW_FS = 500
TARGET_FS = 250


def detect_peaks(s250):
    xqrs_peaks = None
    try:
        from wfdb.processing import xqrs_detect
        xqrs_peaks = np.asarray(
            xqrs_detect(s250.astype(np.float64), fs=TARGET_FS, verbose=False),
            dtype=np.int64).ravel()
    except Exception:
        xqrs_peaks = None
    # XQRS occasionally returns <3 peaks on noisy MIMIC records; use the
    # band-pass energy fallback in that case.
    if xqrs_peaks is not None and len(xqrs_peaks) >= 3:
        return xqrs_peaks
    from scipy.signal import butter, filtfilt, find_peaks
    b, a = butter(2, [8 / 125.0, 25 / 125.0], btype='band')
    y = filtfilt(b, a, s250)
    fallback, _ = find_peaks(y * y, distance=100,
                             height=np.quantile(y * y, 0.6))
    fallback = np.asarray(fallback, dtype=np.int64).ravel()
    n_xqrs = 0 if xqrs_peaks is None else len(xqrs_peaks)
    if len(fallback) >= n_xqrs:
        return fallback
    return (xqrs_peaks if n_xqrs else fallback)


def firmware_zscore(beat):
    beat = beat.astype(np.float64)
    mu = float(np.mean(beat))
    var_pop = float(np.var(beat))
    std_pop = math.sqrt(var_pop) if var_pop > 0 else 0.0
    if std_pop < 1e-6:
        std_pop = 1.0
    return ((beat - mu) / std_pop).astype(np.float32)


def read_record(raw_dir, stem):
    rec = wfdb.rdrecord(str(raw_dir / stem))
    sig = np.asarray(rec.p_signal, dtype=np.float32)
    if sig.shape[0] != 5000 or sig.shape[1] != 12:
        raise RuntimeError(f'unexpected signal shape {sig.shape}')
    return sig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',
                    default='/home/devcontainers/meeti/data/selected8k/m3_1000_manifest.csv')
    ap.add_argument('--raw-dir', default='/mnt/d/MIMIC_ECG_raw_v2')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/data/processed_small/m3_beat_centered')
    ap.add_argument('--max-records', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0,
                    help='debug: process only first N records')
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.manifest).head(args.max_records)
    if args.limit:
        df = df.head(args.limit)

    t0 = time.time()
    records, n_fail, n_skip = [], 0, 0
    total_beats = 0
    print(f'pass1: detect peaks for {len(df)} records', flush=True)
    for i, row in df.iterrows():
        stem = str(Path(str(row['record_path'])).name)
        if not (raw / f'{stem}.hea').exists() or not (raw / f'{stem}.dat').exists():
            n_skip += 1
            continue
        try:
            sig = read_record(raw, stem)
            lead2_250 = corrected_deployment_chain(
                sig[:, 1].astype(np.float64), RAW_FS)
            peaks = detect_peaks(lead2_250)
            # Keep only beats whose 5 s raw window (p500 +/- 1250) fully fits
            # the 10 s record.  This also guarantees the strict 250-pt student
            # beat fits the 250 Hz chain.
            peaks = peaks[(peaks * 2 - HALF >= 0) &
                          (peaks * 2 + HALF <= 5000)]
            if len(peaks) == 0:
                n_skip += 1
                continue
            records.append({
                'stem': stem,
                'study_id': int(row['study_id']),
                'subject_id': str(row['subject_id']),
                'report_class': str(row.get('report_class', 'unknown')),
                'record_path': str(row['record_path']),
                'peaks': peaks,
            })
            total_beats += len(peaks)
        except Exception as exc:
            n_fail += 1
            if n_fail <= 20:
                print(f'fail {stem}: {exc}', flush=True)
        if (i + 1) % 100 == 0:
            print(f'pass1 {i + 1}/{len(df)} records={len(records)} '
                  f'beats={total_beats} fail={n_fail} skip={n_skip} '
                  f'elapsed={time.time() - t0:.0f}s', flush=True)

    print(f'pass1 done: records={len(records)} beats={total_beats} '
          f'fail={n_fail} skip={n_skip} elapsed={time.time() - t0:.0f}s',
          flush=True)

    positions = np.zeros(total_beats, dtype=np.int64)
    record_ids = np.zeros(total_beats, dtype=np.int64)
    student_beats = np.zeros((total_beats, BEAT_SAMPLES), dtype=np.float32)
    student_mask = np.zeros(total_beats, dtype=bool)
    beat_windows = np.lib.format.open_memmap(
        out / 'beat_windows.npy', mode='w+', dtype=np.float32,
        shape=(total_beats, 12, WINDOW_SAMPLES))
    meta_rows = []
    beat_idx = 0
    t1 = time.time()
    print('pass2: fill beat-centered windows', flush=True)
    for ri, rec in enumerate(records):
        stem = rec['stem']
        try:
            sig = read_record(raw, stem)
            lead2_250 = corrected_deployment_chain(
                sig[:, 1].astype(np.float64), RAW_FS)
        except Exception as exc:
            n_fail += 1
            print(f'pass2 fail {stem}: {exc}', flush=True)
            continue
        n250 = len(lead2_250)
        for li, p250 in enumerate(rec['peaks']):
            p250 = int(p250)
            p500 = p250 * 2
            lo = p500 - HALF
            hi = p500 + HALF
            pad_before = 0
            pad_after = 0
            s0 = lo
            s1 = hi
            seg = sig[s0:s1].astype(np.float32)
            if seg.shape[0] != WINDOW_SAMPLES:
                raise RuntimeError(f'window length {seg.shape[0]}')
            beat_windows[beat_idx] = seg.T

            valid = (p250 - BEAT_HALF >= 0 and p250 + BEAT_HALF <= n250)
            if valid:
                sb = lead2_250[p250 - BEAT_HALF:p250 + BEAT_HALF]
                student_beats[beat_idx] = firmware_zscore(sb)
                student_mask[beat_idx] = True

            positions[beat_idx] = p250
            record_ids[beat_idx] = rec['study_id']
            meta_rows.append({
                'beat_idx': beat_idx,
                'record_path': rec['record_path'],
                'stem': stem,
                'study_id': rec['study_id'],
                'subject_id': rec['subject_id'],
                'report_class': rec['report_class'],
                'r_peak_250': p250,
                'r_peak_500': p500,
                'pad_before': int(pad_before),
                'pad_after': int(pad_after),
                'student_beat_valid': int(valid),
            })
            beat_idx += 1
        if (ri + 1) % 100 == 0:
            print(f'pass2 {ri + 1}/{len(records)} filled={beat_idx}/{total_beats} '
                  f'elapsed={time.time() - t1:.0f}s', flush=True)

    beat_windows.flush()
    del beat_windows
    np.save(out / 'beat_positions.npy', positions)
    np.save(out / 'record_ids.npy', record_ids)
    np.save(out / 'student_beats_lead2.npy', student_beats)
    np.save(out / 'student_beat_mask.npy', student_mask)
    meta = pd.DataFrame(meta_rows)
    meta.to_csv(out / 'beat_meta.csv', index=False)

    lab = meta['report_class'].map({'normal': 0, 'abnormal': 1}).fillna(2).astype(int)
    summary = {
        'manifest': args.manifest,
        'raw_dir': str(raw),
        'out_dir': str(out),
        'records_seen': int(len(records)),
        'records_failed': int(n_fail),
        'records_skipped': int(n_skip),
        'n_beats': int(total_beats),
        'student_valid_beats': int(student_mask.sum()),
        'weak_label_dist': lab.value_counts().to_dict(),
        'window_shape': [int(total_beats), 12, WINDOW_SAMPLES],
        'window_sampling_hz': RAW_FS,
        'window_policy': 'strict full-fit only, no padding',
        'r_peak_chain_hz': TARGET_FS,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2), flush=True)
    print(f'done -> {out} elapsed={time.time() - t0:.1f}s', flush=True)


if __name__ == '__main__':
    main()
