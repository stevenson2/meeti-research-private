#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B1 evidence: RR-shape comparison (sinus arrhythmia vs AF vs ectopy).

Diagnostic only (no gate retuning).  For the device recordings, afdb AF/N
windows and PTB-XL AF/non-AF records it reports SDNN / RMSSD / pNN50 /
RMSSD-over-SDNN / lag-1 autocorrelation / derivative smoothness, to test the
hypothesis that ecg_real_052's high RR entropy is respiratory sinus
arrhythmia (smooth, autocorrelated) rather than AF (abrupt, uncorrelated).
"""
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r14_rr_baseline import detect_rr, features_from_rr  # noqa: E402
from r15_laneA_rr_gate import DEVICE, device_peak_times  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r15_laneA_rr_gate'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'


def metrics(rr):
    rr = np.asarray(rr, float)
    rr = rr[(rr >= 0.30) & (rr <= 2.00)]
    if len(rr) < 6:
        return None
    d = np.diff(rr)
    dd = np.diff(d)
    ac1 = (float(np.corrcoef(rr[:-1], rr[1:])[0, 1])
           if len(rr) > 3 and np.std(rr[:-1]) > 0 and np.std(rr[1:]) > 0
           else np.nan)
    f = features_from_rr(rr)
    return {
        'n_rr': int(len(rr)),
        'median_rr': round(float(np.median(rr)), 4),
        'sdnn': round(float(rr.std(ddof=1)), 4),
        'rmssd': round(float(np.sqrt(np.mean(d ** 2))), 4),
        'pnn50': round(float(np.mean(np.abs(d) > 0.05)), 4),
        'rmssd_over_sdnn': round(float(np.sqrt(np.mean(d ** 2))
                                      / max(rr.std(ddof=1), 1e-9)), 4),
        'autocorr_lag1': round(ac1, 4) if np.isfinite(ac1) else None,
        'smoothness_ddrr_over_drr': round(float(np.mean(np.abs(dd))
                                              / max(np.mean(np.abs(d)), 1e-9)), 4),
        'rr_entropy': round(float(f['rr_entropy']), 4),
    }


def agg(rows, key):
    v = np.array([r[key] for r in rows if r and r[key] is not None
                  and np.isfinite(r[key])], float)
    if len(v) == 0:
        return None
    return {'n': int(len(v)), 'median': round(float(np.median(v)), 4),
            'p10': round(float(np.percentile(v, 10)), 4),
            'p90': round(float(np.percentile(v, 90)), 4)}


def main():
    out = {'arm': 'r15_b1_rr_shape', 'note': 'diagnostic only'}
    # device whole-recording RR
    dev = {}
    for name in DEVICE:
        pk = device_peak_times(name)
        rr = np.diff(pk['deploy_xqrs250'])  # seconds
        dev[name] = metrics(rr)
    out['device_whole_record'] = dev

    # afdb eval records: sample windows
    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    fit_recs = list(np.load(OUT / 'features.npz')['fit_recs'])
    eval_recs = [r for r in sorted(set(rec)) if r not in fit_recs]
    rng = np.random.default_rng(7)
    af_rows, n_rows = [], []
    for r in eval_recs:
        idx = np.where(rec == r)[0]
        obj = wfdb.rdrecord(str(AFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        for L, bag in ((1, af_rows), (0, n_rows)):
            cand = idx[lab[idx] == L]
            if len(cand) == 0:
                continue
            take = rng.choice(cand, min(150, len(cand)), replace=False)
            for i in take:
                w = sig[start[i]:start[i] + 2500]
                if len(w) < 2500:
                    continue
                rr, _ = detect_rr(resample_poly(w, 2, 5))
                m = metrics(rr)
                if m:
                    bag.append(m)
    out['afdb_eval'] = {'AF': {k: agg(af_rows, k) for k in
                               ['sdnn', 'rmssd', 'pnn50', 'rmssd_over_sdnn',
                                'autocorr_lag1', 'smoothness_ddrr_over_drr',
                                'rr_entropy']},
                        'N': {k: agg(n_rows, k) for k in
                              ['sdnn', 'rmssd', 'pnn50', 'rmssd_over_sdnn',
                               'autocorr_lag1', 'smoothness_ddrr_over_drr',
                               'rr_entropy']}}

    # PTB train sample
    tr = np.load(ROOT / 'runs/r14_af_prep/ptbxl_af_train.npz')
    x, y = tr['x'], tr['y_af'].astype(int)
    rng = np.random.default_rng(8)
    af_rows2, n_rows2 = [], []
    for L, bag in ((1, af_rows2), (0, n_rows2)):
        idx = np.where(y == L)[0]
        take = rng.choice(idx, min(400, len(idx)), replace=False)
        for i in take:
            rr, _ = detect_rr(x[i])
            m = metrics(rr)
            if m:
                bag.append(m)
    out['ptb_train'] = {'AF': {k: agg(af_rows2, k) for k in
                               ['sdnn', 'rmssd', 'pnn50', 'rmssd_over_sdnn',
                                'autocorr_lag1', 'smoothness_ddrr_over_drr',
                                'rr_entropy']},
                        'nonAF': {k: agg(n_rows2, k) for k in
                                  ['sdnn', 'rmssd', 'pnn50', 'rmssd_over_sdnn',
                                   'autocorr_lag1', 'smoothness_ddrr_over_drr',
                                   'rr_entropy']}}
    (OUT / 'rr_shape.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
