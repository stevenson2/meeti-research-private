#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A6: ltafdb channel / window-length sensitivity (low priority).

Compares ECG1 (index 0) vs ECG2 (index 1) and 10 s / 30 s / 60 s windows on
the ltafdb third external domain.  Features: rr_entropy + the A1 shape /
normalized-variability family.  Thresholds are fitted on afdb fit windows of
the same length (channel 1) and never on ltafdb; an oracle ltafdb threshold
(max spec at recall >= 0.90 on ltafdb itself) is reported as a diagnostic.

Disclosures: ltafdb N segments are scarce and AF-enriched; 60 s windows
require 6 consecutive same-label 10 s grid windows (shorter N segments drop).

Outputs: runs/r16_ltafdb_sensitivity/result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import train_theta_at_recall  # noqa: E402
from r14_rr_baseline import detect_rr  # noqa: E402
from r16_rr_features import features_from_rr_ext  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')
AFDB = Path('/home/devcontainers/ecg_data/afdb')
LTA_SEG = ROOT / 'runs/r15_ltafdb_prep/segments.npz'
AFDB_SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
OUT = ROOT / 'runs/r16_ltafdb_sensitivity'
FEATS = ['rr_entropy', 'cv', 'mean_abs_drr_norm', 'rmssd', 'sdnn', 'sd1',
         'katz_fd', 'range_norm', 'autocorr1', 'pnn50', 'disten_m2']


def seg_index(path):
    seg = np.load(path)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    by = {}
    for r, s, l in zip(rec, start, lab):
        by.setdefault(str(r), {})[int(s)] = int(l)
    return by


def candidates(labmap, fs, L, n_samp):
    win = 10 * fs
    n_need = L // 10
    out = {1: [], 0: []}
    for s, l in labmap.items():
        if l not in out:
            continue
        ok = all(labmap.get(s - k * win) == l for k in range(1, n_need))
        if ok and s - (n_need - 1) * win >= 0 and s + win <= n_samp:
            out[l].append(s - (n_need - 1) * win)
    return out


def feats_for_sig(w, fs, up, down):
    w100 = resample_poly(w, up, down)
    rr, _ = detect_rr(w100)
    f = features_from_rr_ext(rr)
    return {k: f.get(k, np.nan) for k in FEATS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-class', type=int, default=20)
    ap.add_argument('--per-class-60', type=int, default=10)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # ---- afdb fit thresholds per window length (channel 1) ----
    af_by = seg_index(AFDB_SEG)
    fit_recs = sorted(set(np.load(ROOT / 'runs/r16_rr_shape/afdb_features.npz')[
        'afdb_rec'][np.load(ROOT / 'runs/r16_rr_shape/afdb_features.npz')[
            'afdb_split'] == 0].tolist()))
    af_fit = {}
    for L in (10, 30, 60):
        rows, ys = [], []
        rng = np.random.default_rng(42)
        for r in fit_recs:
            obj = wfdb.rdrecord(str(AFDB / r))
            sig = np.asarray(obj.p_signal, np.float64)[:, 1]
            n_samp = len(sig)
            cand = candidates(af_by[r], 250, L, n_samp)
            for lab in (1, 0):
                if not cand[lab]:
                    continue
                take = rng.choice(cand[lab], min(args.per_class, len(cand[lab])),
                                  replace=False)
                for s in take:
                    w = sig[s:s + L * 250]
                    if len(w) < L * 250:
                        continue
                    rows.append(feats_for_sig(w, 250, 2, 5))
                    ys.append(lab)
        y = np.array(ys, int)
        af_fit[L] = {'n': int(len(y)), 'pos': int(y.sum()), 'features': {}}
        for k in FEATS:
            s = np.array([r[k] for r in rows], float)
            m = np.isfinite(s)
            if m.sum() < 30 or len(np.unique(y[m])) < 2:
                continue
            auc = roc_auc_score(y[m], s[m])
            sign = 1.0 if auc >= 0.5 else -1.0
            th = train_theta_at_recall(y[m], sign * s[m], 0.90)
            af_fit[L]['features'][k] = {
                'auc': round(float(auc), 4), 'sign': sign,
                'theta_R90': round(float(th), 4) if th is not None else None}
        print(f'afdb fit {L}s n={len(y)} pos={int(y.sum())}', flush=True)

    # ---- ltafdb: read each record once, both channels ----
    lta_by = seg_index(LTA_SEG)
    records = sorted(lta_by)
    rng = np.random.default_rng(42)
    store = {}
    for ch in (0, 1):
        for L in (10, 30, 60):
            store[(ch, L)] = {'rows': [], 'y': [], 'rec': []}
    for ri, r in enumerate(records):
        obj = wfdb.rdrecord(str(LTAFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)
        n_samp = len(sig)
        for ch in (0, 1):
            for L in (10, 30, 60):
                cand = candidates(lta_by[r], 128, L, n_samp)
                pc = args.per_class_60 if L == 60 else args.per_class
                for lab in (1, 0):
                    if not cand[lab]:
                        continue
                    take = rng.choice(cand[lab], min(pc, len(cand[lab])),
                                      replace=False)
                    for s in take:
                        w = sig[s:s + L * 128, ch]
                        if len(w) < L * 128:
                            continue
                        store[(ch, L)]['rows'].append(
                            feats_for_sig(w, 128, 25, 32))
                        store[(ch, L)]['y'].append(lab)
                        store[(ch, L)]['rec'].append(r)
        print('ltafdb', r, ri + 1, '/', len(records), flush=True)

    result = {
        'arm': 'r16_ltafdb_sensitivity',
        'domain': 'ltafdb (AFIB vs N; VT/AB/SVTA/SBR etc excluded by prep)',
        'afdb_fit_thresholds': af_fit,
        'ltafdb': {},
        'disclosures': [
            'ltafdb AF-enriched and N-scarce (R15-B4: 371,193 AF / 298,319 N '
            'windows); 60 s windows require 6 consecutive same-label 10 s '
            'grid windows, so short N segments drop out',
            'thresholds fitted on afdb fit windows of the same length '
            '(channel 1); ltafdb never used for threshold selection',
            'oracle thresholds are post-hoc diagnostics',
            'ECG1/ECG2 are hardware channels of the same recording; both are '
            'reported, ECG2 was the R15-B4 default',
        ],
    }
    for (ch, L), d in store.items():
        y = np.array(d['y'], int)
        if len(y) == 0:
            continue
        entry = {'n': int(len(y)), 'pos': int(y.sum()),
                 'n_records': int(len(set(d['rec']))), 'features': {}}
        for k in FEATS:
            s = np.array([r[k] for r in d['rows']], float)
            m = np.isfinite(s)
            if m.sum() < 30 or len(np.unique(y[m])) < 2:
                continue
            auc = float(roc_auc_score(y[m], s[m]))
            ap_ = float(average_precision_score(y[m], s[m]))
            sign = 1.0 if auc >= 0.5 else -1.0
            ss = sign * s
            orc = train_theta_at_recall(y[m], ss[m], 0.90)
            fit_th = af_fit[L]['features'].get(k, {}).get('theta_R90')
            row = {'auc': round(auc, 4), 'ap': round(ap_, 4), 'sign': sign}
            if fit_th is not None:
                fl = m & (ss >= fit_th)
                row['afdb_theta_R90'] = fit_th
                row['op_at_afdb_theta'] = {
                    'recall': round(float((fl & (y == 1)).sum())
                                    / max(int((y == 1).sum()), 1), 4),
                    'spec': round(float(((~fl) & (y == 0)).sum())
                                  / max(int((y == 0).sum()), 1), 4),
                    'n_flag': int(fl.sum())}
            if orc is not None:
                fl = m & (ss >= orc)
                row['oracle_theta_R90'] = round(float(orc), 4)
                row['op_at_oracle'] = {
                    'recall': round(float((fl & (y == 1)).sum())
                                    / max(int((y == 1).sum()), 1), 4),
                    'spec': round(float(((~fl) & (y == 0)).sum())
                                  / max(int((y == 0).sum()), 1), 4)}
            entry['features'][k] = row
        result['ltafdb'][f'ecg{ch + 1}_{L}s'] = entry
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps({k: {'n': v['n'], 'pos': v['pos'],
                          'rr_entropy_auc': v['features'].get(
                              'rr_entropy', {}).get('auc'),
                          'cv_auc': v['features'].get('cv', {}).get('auc')}
                      for k, v in result['ltafdb'].items()},
                     indent=2, ensure_ascii=False))
    print('written', OUT / 'result.json')


if __name__ == '__main__':
    main()
