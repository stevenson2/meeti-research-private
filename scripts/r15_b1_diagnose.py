#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B1 diagnosis: why does the RR-only device gate fail on ecg_real_052?

No gate retuning: this script only characterises the blocker.
  - evaluates every A4 feature + the pooled LR combo on both device
    recordings at their own fit-side Youden/R90 thresholds;
  - compares the ecg_real_052 RR series against rec_latest and afdb AF
    segments (autocorrelation / smoothness / RMSSD-SDNN ratio) to test the
    respiratory-sinus-arrhythmia hypothesis.
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import train_theta_at_recall, youden_threshold  # noqa: E402
from r14_rr_baseline import detect_peaks, detect_rr, features_from_rr  # noqa: E402
from r15_laneA_rr_gate import (DEVICE, FEATURES, PRIMARY, afdb_features,  # noqa: E402
                               device_peak_times, m4_alarms, window_features)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r15_laneA_rr_gate'
PTB_FEAT = ROOT / 'runs/r14_rr_baseline/features.npz'


def fit_theta(sc, y):
    m = np.isfinite(sc)
    th_y = float(youden_threshold(y[m], sc[m]))
    th_r = train_theta_at_recall(y[m], sc[m], 0.90)
    return {'youden': th_y, 'R90': (float(th_r) if th_r else None)}


def eval_scores(sc, y, th):
    m = np.isfinite(sc)
    yp = np.zeros(len(sc), bool)
    yp[m] = sc[m] >= th
    rec = float((yp & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
    spec = float(((~yp) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
    return {'recall': round(rec, 4), 'spec': round(spec, 4),
            'n_flag': int(yp.sum())}


def device_flags(X, th, meta, feat):
    sc = X[feat]
    m = np.isfinite(sc)
    flags = np.zeros(len(sc), bool)
    flags[m] = sc[m] >= th
    alarms = m4_alarms(flags)
    return {'n_flag': int(flags.sum()), 'n_valid': int(m.sum()),
            'm4_alarms': int(len(alarms)),
            'max_score': round(float(np.nanmax(sc)), 4) if m.any() else None,
            'median_score': round(float(np.nanmedian(sc)), 4) if m.any() else None,
            'alarm_starts_s': [round(float(meta[i]['start_s']), 1)
                               for i in alarms[:8]]}


def rr_diagnostics(peaks, fs):
    rr = np.diff(peaks) / fs
    rr = rr[(rr >= 0.3) & (rr <= 2.0)]
    d = np.diff(rr)
    dd = np.diff(d)
    ac = [float(np.corrcoef(rr[:-k], rr[k:])[0, 1]) for k in (1, 2, 3, 5, 10)
          if len(rr) > k + 2]
    return {
        'n_rr': int(len(rr)), 'median_s': round(float(np.median(rr)), 4),
        'sdnn': round(float(rr.std(ddof=1)), 4),
        'rmssd': round(float(np.sqrt(np.mean(d ** 2))), 4),
        'pnn50': round(float(np.mean(np.abs(d) > 0.05)), 4),
        'cv': round(float(rr.std(ddof=1) / rr.mean()), 4),
        'rmssd_over_sdnn': round(float(np.sqrt(np.mean(d ** 2))
                                      / max(rr.std(ddof=1), 1e-9)), 4),
        'mean_abs_drr': round(float(np.mean(np.abs(d))), 4),
        'mean_abs_ddrr': round(float(np.mean(np.abs(dd))), 4),
        'smoothness_ddrr_over_drr': round(float(np.mean(np.abs(dd))
                                              / max(np.mean(np.abs(d)), 1e-9)), 4),
        'autocorr_lags_1_2_3_5_10': [round(a, 4) for a in ac],
        'rr_entropy_whole': round(float(features_from_rr(rr)['rr_entropy']), 4),
    }


def main():
    # pooled fit features
    pf = np.load(PTB_FEAT)
    ptb = {k: pf[f'tr_{k}'].astype(float) for k in FEATURES}
    ptb_y = pf['y_tr'].astype(int)
    af = np.load(OUT / 'features.npz')
    Xa = {k: af[f'afdb_{k}'].astype(float) for k in FEATURES}
    rec_a, y_a = af['afdb_rec'], af['afdb_y'].astype(int)
    fit_recs = list(af['fit_recs']); eval_recs = list(af['eval_recs'])
    fit_mask = np.isin(rec_a, fit_recs); eval_mask = np.isin(rec_a, eval_recs)

    # orient features by pooled fit AUC
    for k in FEATURES:
        sc = np.concatenate([ptb[k], Xa[k][fit_mask]])
        m = np.isfinite(sc)
        if roc_auc_score(ptb_y[m[:len(ptb_y)]] if False else
                         np.concatenate([ptb_y, y_a[fit_mask]])[m], sc[m]) < 0.5:
            ptb[k] = -ptb[k]; Xa[k] = -Xa[k]

    # LR combo on pooled fit
    def design(Xd, med=None):
        Xm = np.column_stack([Xd[k] for k in FEATURES])
        ind = np.column_stack([np.isnan(Xm[:, j]).astype(float)
                               for j in range(Xm.shape[1])])
        if med is None:
            med = np.nanmedian(Xm, axis=0)
        return np.hstack([np.where(np.isnan(Xm), med, Xm), ind]), med

    Xf, med = design({k: np.concatenate([ptb[k], Xa[k][fit_mask]])
                      for k in FEATURES})
    mu, sd = Xf.mean(0), Xf.std(0); sd[sd == 0] = 1.0
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    yfit = np.concatenate([ptb_y, y_a[fit_mask]])
    lr.fit((Xf - mu) / sd, yfit)
    fit_lr = lr.predict_proba((Xf - mu) / sd)[:, 1]
    Xe, _ = design({k: Xa[k][eval_mask] for k in FEATURES}, med)
    eval_lr = lr.predict_proba((Xe - mu) / sd)[:, 1]

    # device windows
    dev = {}
    for name in DEVICE:
        pk = device_peak_times(name)
        dev[name] = {'peaks': pk, 'diag': rr_diagnostics(pk['deploy_xqrs250'], 250.0),
                     'det': {}}
        for det in ('deploy_xqrs250', 'pt100'):
            X, meta = window_features(pk[det], pk['dur_s'], 1.0)
            Xl = dict(X)
            Xl['lr_combo'] = lr.predict_proba(
                (design(X, med)[0] - mu) / sd)[:, 1]
            dev[name]['det'][det] = {'X': Xl, 'meta': meta}

    # evaluate every feature + LR combo on both sides
    rows = {}
    for feat in FEATURES + ['lr_combo']:
        if feat == 'lr_combo':
            fit_sc, eval_sc = fit_lr, eval_lr
        else:
            fit_sc = np.concatenate([ptb[feat], Xa[feat][fit_mask]])
            eval_sc = Xa[feat][eval_mask]
        th = fit_theta(fit_sc, yfit)
        row = {'fit_theta': {k: (round(v, 4) if v is not None else None)
                             for k, v in th.items()},
               'afdb_eval': {}, 'device': {}}
        for key, t in th.items():
            if t is None:
                continue
            row['afdb_eval'][key] = eval_scores(eval_sc, y_a[eval_mask], t)
            row['device'][key] = {
                n: {det: device_flags(dev[n]['det'][det]['X'], t,
                                      dev[n]['det'][det]['meta'], feat)
                    for det in ('deploy_xqrs250', 'pt100')}
                for n in dev}
        rows[feat] = row

    out = {
        'arm': 'r15_b1_diagnose',
        'blocker': 'ecg_real_052 respiratory sinus arrhythmia: RR entropy '
                   'median 1.82 / max 2.45 overlaps the AF threshold 1.98-2.14',
        'feature_device_afdb_table': rows,
        'rr_diagnostics': {n: dev[n]['diag'] for n in dev},
        'note': 'no gate retuning; all numbers are diagnostics only',
    }
    (OUT / 'diagnosis.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'rr_diagnostics': out['rr_diagnostics']},
                     indent=2, ensure_ascii=False))
    for feat, row in rows.items():
        devtxt = {}
        for key in ('youden', 'R90'):
            if key in row['device']:
                devtxt[key] = {n: {d: row['device'][key][n][d]['m4_alarms']
                                   for d in row['device'][key][n]}
                               for n in row['device'][key]}
        print(feat, 'theta', row['fit_theta'], 'afdb', row['afdb_eval'],
              'device_alarms', devtxt, flush=True)


if __name__ == '__main__':
    main()
