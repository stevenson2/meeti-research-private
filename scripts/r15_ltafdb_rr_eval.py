#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B4 eval: RR-entropy channel on ltafdb (third external domain).

Samples up to N windows per class per record from the B4 segments.npz,
reads ECG2 (index 1) at 128 Hz, resamples to 100 Hz, computes the A4
rr_entropy feature, and reports AUC/AP + recall/spec at the B1 thresholds
(theta_R90=2.1418, theta_youden=1.9808; fit on PTB folds1-9 + afdb fit
records, never on ltafdb).

Outputs: runs/r15_ltafdb_rr_eval/result.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r14_rr_baseline import detect_rr, features_from_rr  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')
SEG = ROOT / 'runs/r15_ltafdb_prep/segments.npz'
OUT = ROOT / 'runs/r15_ltafdb_rr_eval'
THETA_R90 = 2.1418
THETA_YOUDEN = 1.9808


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(42)
    per_class = 60
    ent, yy, out_rec = [], [], []
    for r in sorted(set(rec)):
        idx = np.where(rec == r)[0]
        obj = wfdb.rdrecord(str(LTAFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        for L in (1, 0):
            cand = idx[lab[idx] == L]
            if len(cand) == 0:
                continue
            take = rng.choice(cand, min(per_class, len(cand)), replace=False)
            for i in take:
                w = sig[start[i]:start[i] + 1280]
                if len(w) < 1280:
                    continue
                w100 = resample_poly(w, 25, 32)
                rr, _ = detect_rr(w100)
                ent.append(features_from_rr(rr)['rr_entropy'])
                yy.append(L)
                out_rec.append(r)
        print('done', r, 'total', len(yy), flush=True)
    ent = np.array(ent, float)
    yy = np.array(yy, int)
    ok = np.isfinite(ent)
    res = {
        'arm': 'r15_ltafdb_rr_eval',
        'domain': 'ltafdb (AFIB vs N; VT/AFL/J/other excluded by prep)',
        'channel': 'ECG2 (index 1), 128->100 Hz',
        'n_windows': int(len(yy)), 'n_pos': int(yy.sum()),
        'n_valid': int(ok.sum()),
        'auc': round(float(roc_auc_score(yy[ok], ent[ok])), 4),
        'ap': round(float(average_precision_score(yy[ok], ent[ok])), 4),
        'per_record_windows': per_class,
        'thresholds_source': 'B1 fit (PTB folds1-9 + afdb fit records)',
        'op_points': {},
    }
    for name, th in [('theta_R90', THETA_R90), ('theta_youden', THETA_YOUDEN)]:
        yp = ent[ok] >= th
        res['op_points'][name] = {
            'theta': th,
            'recall': round(float((yp & (yy[ok] == 1)).sum())
                            / max(int((yy[ok] == 1).sum()), 1), 4),
            'spec': round(float(((~yp) & (yy[ok] == 0)).sum())
                          / max(int((yy[ok] == 0).sum()), 1), 4),
        }
    (OUT / 'result.json').write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
