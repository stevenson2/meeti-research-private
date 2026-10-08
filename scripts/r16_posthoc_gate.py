#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16 post-hoc gate sensitivity (clearly labelled NOT pre-registered).

The pre-registered A2 two-stage gate used a composite train recall target of
0.90 and FAILED (train->eval threshold drift).  This script re-runs the SAME
pre-registered structure (Stage 1 = rr_entropy, Stage 2 = shape / normalized
variability feature) with more permissive TRAIN-SIDE recall targets
(Stage-1 target 0.99; Stage-2 composite target 0.95-0.99) and reports whether
the three pre-registered acceptance lines can be met.

Acceptance lines (unchanged):
  1. ecg_real_052 event-level FP=0 (M4 1-of-5, cooldown 20; both the deploy
     XQRS and the PT100 detector are checked)
  2. afdb held-out 11 records AF recall >= 0.90
  3. ltafdb sampled AF recall >= 0.90
rec_latest is disclosed separately.

Selection bias is disclosed: the threshold targets are permissive values from
the R16 pre-registration vocabulary, but the specific (feature, target) cell
is chosen after seeing the eval/device outcomes -> this is a post-hoc
feasibility result, NOT a deployment claim.  A fresh pre-registration and
device AF positives (A7) are required before deployment.

Outputs: runs/r16_rr_shape_gate/posthoc_two_stage_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r16_rr_features import SHAPE_FEATURES  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DATA = ROOT / 'runs/r16_rr_shape'
OUT = ROOT / 'runs/r16_rr_shape_gate'
M4_CONFIRM, M4_COOLDOWN = 5, 20
STAGE1 = 'rr_entropy'
STAGE1_TARGET = 0.99
COMPOSITE_TARGETS = (0.95, 0.97, 0.98, 0.99)
CANDIDATES = ['mean_abs_drr_norm', 'cv', 'rmssd', 'sdnn', 'sd1', 'katz_fd',
              'range_norm', 'iqr_norm', 'rmssd_over_sdnn', 'sd1_sd2_ratio',
              'pnn50', 'pnn20', 'tpr', 'autocorr1', 'autocorr2', 'autocorr3',
              'drr_sign_entropy', 'trend_slope_norm', 'disten_m2',
              'fuzzyen_m1', 'fuzzyen_m2', 'smoothness', 'higuchi_fd',
              'dfa_alpha1', 'sd2', 'sampen_m1', 'sampen_m2_r02',
              'sampen_m2_r025', 'entropy_norm', 'drr_entropy_norm', 'cosen',
              'drr_entropy', 'sd1_sd2_ratio']


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


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
    y = np.asarray(y, int)
    s = np.asarray(s, float)
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


def dev_eval(score, stage1, th2, starts):
    fl = stage1 & np.isfinite(score) & (score >= th2)
    return {'n_windows': int(len(score)), 'n_flag': int(fl.sum()),
            'm4_alarms': int(len(m4_alarms(fl))),
            'max_score': (round(float(np.nanmax(score)), 4)
                          if np.isfinite(score).any() else None),
            'median_score': (round(float(np.nanmedian(score)), 4)
                             if np.isfinite(score).any() else None),
            'theta2_margin_abs': (round(float(th2 - np.nanmax(score)), 4)
                                  if np.isfinite(score).any() else None),
            'm4_alarm_starts_s': [round(float(starts[i]), 1)
                                  for i in m4_alarms(fl)[:10]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    af = dict(np.load(DATA / 'afdb_features.npz'))
    lt = dict(np.load(DATA / 'ltafdb_features.npz'))
    ptb = dict(np.load(DATA / 'ptb_features.npz'))
    dv = np.load(DATA / 'device_features.npz')
    yaf = af['afdb_y'].astype(int)
    sp = af['afdb_split'].astype(int)
    ylt = lt['ltafdb_y'].astype(int)
    starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
    starts_r = dv['rec_latest_deploy_xqrs250_paired_start']

    result = {
        'arm': 'r16_posthoc_gate_sensitivity',
        'status': 'POST-HOC (not pre-registered); feasibility only',
        'structure': 'Stage1 rr_entropy >= theta1 -> Stage2 feature >= theta2',
        'stage1_target': STAGE1_TARGET,
        'composite_targets': list(COMPOSITE_TARGETS),
        'acceptance': ('052 M4 events=0 (deploy XQRS and PT100) AND afdb eval '
                       'recall>=0.90 AND ltafdb recall>=0.90'),
        'window_lengths': {},
        'disclosures': [
            'the (feature, target) cell is selected after seeing eval/device '
            'outcomes -> post-hoc, multiplicity over 34 features x 4 targets',
            'thresholds themselves are TRAIN-side only (no eval/device tuning)',
            'no device AF positives; 052 silence is necessary, not sufficient',
            'NaN windows count as misses in recall (R15 convention)',
            'rec_latest is disclosure only',
        ],
    }
    for wl in ('10s', '30s'):
        if wl == '10s':
            s1f = np.concatenate([ptb['ptb_rr_entropy'],
                                  af['afdb10_rr_entropy'][sp == 0]])
            yf = np.concatenate([ptb['ptb_y'].astype(int), yaf[sp == 0]])
            s1e, ye = af['afdb10_rr_entropy'][sp == 1], yaf[sp == 1]
            s1l, yl = lt['ltafdb10_rr_entropy'], ylt
            s1d = dv['ecg_real_052_deploy_xqrs250_paired10_rr_entropy']
            s1dpt = dv['ecg_real_052_pt100_paired10_rr_entropy']
            s1r = dv['rec_latest_deploy_xqrs250_paired10_rr_entropy']
            pref = 'afdb10_'
            lpref = 'ltafdb10_'
            dpref = 'ecg_real_052_deploy_xqrs250_paired10_'
            dptpref = 'ecg_real_052_pt100_paired10_'
            rpref = 'rec_latest_deploy_xqrs250_paired10_'
        else:
            s1f = af['afdb30_rr_entropy'][sp == 0]
            yf = yaf[sp == 0]
            s1e, ye = af['afdb30_rr_entropy'][sp == 1], yaf[sp == 1]
            s1l, yl = lt['ltafdb30_rr_entropy'], ylt
            s1d = dv['ecg_real_052_deploy_xqrs250_paired30_rr_entropy']
            s1dpt = dv['ecg_real_052_pt100_paired30_rr_entropy']
            s1r = dv['rec_latest_deploy_xqrs250_paired30_rr_entropy']
            pref = 'afdb30_'
            lpref = 'ltafdb30_'
            dpref = 'ecg_real_052_deploy_xqrs250_paired30_'
            dptpref = 'ecg_real_052_pt100_paired30_'
            rpref = 'rec_latest_deploy_xqrs250_paired30_'
        th1 = th_valid(yf, s1f, STAGE1_TARGET)
        m1f = np.isfinite(s1f) & (s1f >= th1)
        m1e = np.isfinite(s1e) & (s1e >= th1)
        m1l = np.isfinite(s1l) & (s1l >= th1)
        m1d = np.isfinite(s1d) & (s1d >= th1)
        m1dpt = np.isfinite(s1dpt) & (s1dpt >= th1)
        m1r = np.isfinite(s1r) & (s1r >= th1)
        wl_res = {
            'theta1': round(float(th1), 4),
            'stage1_train_recall_valid': round(float(
                (m1f & (yf == 1)).sum()) / max(
                int(((yf == 1) & np.isfinite(s1f)).sum()), 1), 4),
            'stage1_afdb_eval_recall': round(rec_all(ye, s1e, th1), 4),
            'stage1_ltafdb_recall': round(rec_all(yl, s1l, th1), 4),
            'stage1_052_flags': int(m1d.sum()),
            'stage1_052_events': int(len(m4_alarms(m1d))),
            'candidates': {},
        }
        for k in CANDIDATES:
            sf = (np.concatenate([ptb['ptb_' + k], af[pref + k][sp == 0]])
                  if wl == '10s' else af[pref + k][sp == 0])
            se = af[pref + k][sp == 1]
            sl = lt[lpref + k]
            sd = dv[dpref + k]
            sdpt = dv[dptpref + k]
            sr = dv[rpref + k]
            m = np.isfinite(sf)
            if m.sum() < 50 or len(np.unique(yf[m])) < 2:
                continue
            auc = roc_auc_score(yf[m], sf[m])
            sign = 1.0 if auc >= 0.5 else -1.0
            sf, se, sl, sd, sdpt, sr = [sign * np.asarray(x, float)
                                        for x in (sf, se, sl, sd, sdpt, sr)]
            row = {'sign': sign, 'fit_auc': round(float(auc), 4),
                   'targets': {}}
            for ctgt in COMPOSITE_TARGETS:
                val = m1f & np.isfinite(sf)
                Pval = max(int((val & (yf == 1)).sum()), 1)
                th2 = None
                for th in np.unique(sf[np.isfinite(sf)]):
                    if float(((val & (sf >= th) & (yf == 1)).sum())) \
                            / Pval >= ctgt - 1e-12:
                        th2 = float(th)
                if th2 is None:
                    continue
                fld = m1d & np.isfinite(sd) & (sd >= th2)
                fldpt = m1dpt & np.isfinite(sdpt) & (sdpt >= th2)
                flr = m1r & np.isfinite(sr) & (sr >= th2)
                d052 = dev_eval(sd, m1d, th2, starts)
                d052_pt = dev_eval(sdpt, m1dpt, th2, starts)
                drec = dev_eval(sr, m1r, th2, starts_r)
                row['targets'][str(ctgt)] = {
                    'theta2': round(th2, 4),
                    'afdb_eval': op(ye, m1e * se, th2),
                    'ltafdb': op(yl, m1l * sl, th2),
                    'device_052_deploy': d052,
                    'device_052_pt100': d052_pt,
                    'device_rec_latest': drec,
                    'pass': bool(d052['m4_alarms'] == 0
                                 and d052_pt['m4_alarms'] == 0
                                 and op(ye, m1e * se, th2)['recall'] >= 0.90
                                 and op(yl, m1l * sl, th2)['recall'] >= 0.90),
                }
            wl_res['candidates'][k] = row
        result['window_lengths'][wl] = wl_res

    # summary of passing cells
    passes = []
    for wl, wv in result['window_lengths'].items():
        for k, kv in wv['candidates'].items():
            for t, tv in kv['targets'].items():
                if tv['pass']:
                    passes.append({
                        'window': wl, 'feature': k, 'composite_target': t,
                        'theta1': wv['theta1'], 'theta2': tv['theta2'],
                        'afdb_recall': tv['afdb_eval']['recall'],
                        'ltafdb_recall': tv['ltafdb']['recall'],
                        '052_events_deploy': tv['device_052_deploy'][
                            'm4_alarms'],
                        '052_events_pt100': tv['device_052_pt100'][
                            'm4_alarms'],
                        '052_margin_deploy': tv['device_052_deploy'][
                            'theta2_margin_abs'],
                        'rec_latest_events': tv['device_rec_latest'][
                            'm4_alarms'],
                    })
    result['n_pass_cells'] = len(passes)
    result['pass_cells'] = sorted(
        passes, key=lambda x: (-x['afdb_recall'], -x['ltafdb_recall']))
    (OUT / 'posthoc_two_stage_sensitivity.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps({'n_pass_cells': len(passes),
                      'top': result['pass_cells'][:8]},
                     indent=2, ensure_ascii=False))
    print('written', OUT / 'posthoc_two_stage_sensitivity.json')


if __name__ == '__main__':
    main()
