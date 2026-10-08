#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16 annotated-QRS sensitivity for the RR gate (detector-quality control).

The primary R16 gate uses the R14-A4 PT100 detector, whose ltafdb quality is
poor (r16_detector_qc: p90 |beat-count error| = 25).  This script repeats the
30 s RR features using the DATABASE QRS annotations (afdb .qrs, ltafdb .qrs)
to separate detector failure from gate failure.  Device 052 keeps the
deploy-chain XQRS (no database annotations exist for the device).

Evaluated gate (post-hoc cell from r16_posthoc_gate):
  Stage1 rr_entropy >= theta1 (train recall 0.99, afdb fit)
  Stage2 feature >= theta2 (composite train recall 0.99, afdb fit)
Acceptance: 052 M4 events=0 AND afdb eval recall>=0.90 AND ltafdb recall>=0.90.

Outputs: runs/r16_rr_shape_gate/annot_qrs_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r16_rr_features import features_from_rr_ext  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r16_rr_shape_gate/annot_qrs_sensitivity.json'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')
AFDB_SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
LTA_SEG = ROOT / 'runs/r15_ltafdb_prep/segments.npz'
FEATS = ['rr_entropy', 'mean_abs_drr_norm', 'cv', 'rmssd', 'sdnn', 'sd1',
         'katz_fd', 'range_norm', 'autocorr1', 'pnn50']
M4_CONFIRM, M4_COOLDOWN = 5, 20
STAGE1, STAGE1_TARGET, COMPOSITE_TARGET = 'rr_entropy', 0.99, 0.99


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def seg_index(path):
    seg = np.load(path)
    by = {}
    for r, s, l in zip(seg['rec'], seg['start'], seg['label'].astype(int)):
        by.setdefault(str(r), {})[int(s)] = int(l)
    return by


def rr_from_ann(qrs, fs, a, b):
    m = (qrs >= a) & (qrs < b)
    pk = qrs[m]
    if len(pk) < 3:
        return np.array([])
    rr = np.diff(pk) / fs
    return rr[(rr >= 0.30) & (rr <= 2.00)]


def build_annot(dst, seg_path, fs, records, per_class, seed=42):
    by = seg_index(seg_path)
    rng = np.random.default_rng(seed)
    rows, ys, recs, starts = [], [], [], []
    for r in records:
        if r not in by:
            continue
        ann = wfdb.rdann(str(dst / r), 'qrs')
        qrs = ann.sample.astype(np.int64)
        win = 10 * fs
        labmap = by[r]
        for L in (1, 0):
            cand = [s for s, l in labmap.items()
                    if l == L and s - 2 * win >= 0
                    and labmap.get(s - win) == L
                    and s - 2 * win < qrs.max()]
            if not cand:
                continue
            take = rng.choice(np.array(cand), min(per_class, len(cand)),
                              replace=False)
            for s in np.sort(take):
                rr = rr_from_ann(qrs, fs, s - 2 * win, s + win)
                f = features_from_rr_ext(rr)
                rows.append({k: f.get(k, np.nan) for k in FEATS})
                ys.append(L)
                recs.append(r)
                starts.append(int(s))
        print('annot', dst.name, r, len(ys), flush=True)
    return rows, np.array(ys, int), np.array(recs), np.array(starts)


def th_valid(y, s, target):
    m = np.isfinite(s)
    ys, ss = np.asarray(y)[m], np.asarray(s)[m]
    best = None
    for th in np.unique(ss):
        if float(((ss >= th) & (ys == 1)).sum()) / max(
                int((ys == 1).sum()), 1) >= target - 1e-12:
            best = float(th)
    return best


def rec_all(y, s, th):
    fl = np.isfinite(s) & (s >= th)
    return float((fl & (y == 1)).sum()) / max(int((y == 1).sum()), 1)


def op(y, s, th):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fl = np.isfinite(s) & (s >= th)
    return {'recall': round(float((fl & (y == 1)).sum())
                            / max(int((y == 1).sum()), 1), 4),
            'spec': round(float(((~fl) & (y == 0)).sum())
                          / max(int((y == 0).sum()), 1), 4),
            'n_flag': int(fl.sum())}


def main():
    cache = ROOT / 'runs/r16_rr_shape_gate/annot_features.npz'
    afnpz = np.load(ROOT / 'runs/r16_rr_shape/afdb_features.npz')
    fit_recs = sorted(set(afnpz['afdb_rec'][afnpz['afdb_split'] == 0].tolist()))
    eval_recs = sorted(set(afnpz['afdb_rec'][afnpz['afdb_split'] == 1].tolist()))
    if cache.exists():
        d = np.load(cache)
        F = {k: d['F_' + k] for k in FEATS}
        E = {k: d['E_' + k] for k in FEATS}
        L = {k: d['L_' + k] for k in FEATS}
        yf, ye, yl = d['yf'], d['ye'], d['yl']
    else:
        Xf, yf, _, _ = build_annot(AFDB, AFDB_SEG, 250, fit_recs, 600)
        Xe, ye, _, _ = build_annot(AFDB, AFDB_SEG, 250, eval_recs, 600)
        lt_recs = sorted({str(x) for x in np.load(LTA_SEG)['rec']})
        Xl, yl, _, _ = build_annot(LTAFDB, LTA_SEG, 128, lt_recs, 30)
        F = {k: np.array([r[k] for r in Xf], float) for k in FEATS}
        E = {k: np.array([r[k] for r in Xe], float) for k in FEATS}
        L = {k: np.array([r[k] for r in Xl], float) for k in FEATS}
        np.savez_compressed(
            cache, yf=yf, ye=ye, yl=yl,
            **{f'F_{k}': F[k] for k in FEATS},
            **{f'E_{k}': E[k] for k in FEATS},
            **{f'L_{k}': L[k] for k in FEATS})
    # device 052 paired 30 s from deploy-chain XQRS
    sys.path.insert(0, str(ROOT / 'scripts'))
    from r15_laneA_rr_gate import device_peak_times  # noqa
    pk = device_peak_times('ecg_real_052')
    t = pk['deploy_xqrs250']
    starts = np.arange(20.0, pk['dur_s'] - 10.0 + 1e-9, 1.0)
    d052 = {}
    for k in FEATS:
        vals = []
        for s in starts:
            rr = rr_from_ann(t * 250, 250, (s - 20) * 250, (s + 10) * 250)
            vals.append(features_from_rr_ext(rr).get(k, np.nan))
        d052[k] = np.array(vals, float)
    th1 = th_valid(yf, F[STAGE1], STAGE1_TARGET)
    m1f = np.isfinite(F[STAGE1]) & (F[STAGE1] >= th1)
    m1e = np.isfinite(E[STAGE1]) & (E[STAGE1] >= th1)
    m1l = np.isfinite(L[STAGE1]) & (L[STAGE1] >= th1)
    m1d = np.isfinite(d052[STAGE1]) & (d052[STAGE1] >= th1)
    out = {'arm': 'r16_annot_qrs_sensitivity',
           'note': 'database QRS annotations replace PT100; device 052 keeps '
                   'deploy XQRS',
           'n': {'fit': int(len(yf)), 'fit_pos': int(yf.sum()),
                 'eval': int(len(ye)), 'eval_pos': int(ye.sum()),
                 'ltafdb': int(len(yl)), 'ltafdb_pos': int(yl.sum())},
           'theta1': round(float(th1), 4),
           'stage1': {'afdb_eval_recall': round(rec_all(ye, E[STAGE1], th1), 4),
                      'ltafdb_recall': round(rec_all(yl, L[STAGE1], th1), 4),
                      '052_events': int(len(m4_alarms(m1d)))},
           'candidates': {}}
    for k in FEATS:
        if k == STAGE1:
            continue
        sf = F[k]
        m = np.isfinite(sf)
        if m.sum() < 50 or len(np.unique(yf[m])) < 2:
            continue
        auc = roc_auc_score(yf[m], sf[m])
        sign = 1.0 if auc >= 0.5 else -1.0
        sf, se, sl, sd = [sign * np.asarray(x, float)
                          for x in (F[k], E[k], L[k], d052[k])]
        val = m1f & np.isfinite(sf)
        Pval = max(int((val & (yf == 1)).sum()), 1)
        th2 = None
        for th in np.unique(sf[np.isfinite(sf)]):
            if float(((val & (sf >= th) & (yf == 1)).sum())) / Pval \
                    >= COMPOSITE_TARGET - 1e-12:
                th2 = float(th)
        if th2 is None:
            continue
        fl = m1e & np.isfinite(se) & (se >= th2)
        fll = m1l & np.isfinite(sl) & (sl >= th2)
        fld = m1d & np.isfinite(sd) & (sd >= th2)
        row = {'fit_auc': round(float(auc), 4), 'theta2': round(th2, 4),
               'afdb_eval': op(ye, m1e * se, th2),
               'ltafdb': op(yl, m1l * sl, th2),
               'device_052': {'n_flag': int(fld.sum()),
                              'm4_alarms': int(len(m4_alarms(fld))),
                              'max_score': (round(float(np.nanmax(sd)), 4)
                                            if np.isfinite(sd).any() else None)}}
        row['pass'] = bool(row['device_052']['m4_alarms'] == 0
                           and row['afdb_eval']['recall'] >= 0.90
                           and row['ltafdb']['recall'] >= 0.90)
        out['candidates'][k] = row
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False)[:4000])


if __name__ == '__main__':
    main()
