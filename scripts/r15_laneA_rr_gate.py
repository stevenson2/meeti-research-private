#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B1: RR-only Lane A device-domain gate (zero training).

Lane A main channel = RR irregularity (rr_entropy), threshold fitted on the
joint TRAIN-side distribution (PTB-XL folds1-9 + afdb fit records), device
domain used for acceptance only (never for threshold selection).

Pre-registered acceptance:
  (1) both device normal recordings event-level FP=0 under the M4 protocol
      (1-of-5 causal group, cooldown 20 units), at the fitted threshold;
  (2) afdb held-out records AF recall >= 0.90 at the same threshold.

Device R-peaks come from the firmware deployment chain
(deployment_chain + causal HP0.5@250 -> wfdb XQRS @250 Hz), i.e. the same
beat-position sequence the firmware/PC driver already produces -- no new
on-device detector.  A PT100 detector at 100 Hz is reported as sensitivity.

Disclosure: only 2 device normal recordings exist (no device AF positives);
a PASS proves the normal side is silent, sensitivity claims come from
afdb/PTB-XL.
"""
import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import (spec_at_recall, train_theta_at_recall,  # noqa: E402
                       youden_threshold)
from r14_rr_baseline import (detect_peaks, detect_rr,  # noqa: E402
                             features_from_rr)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r15_laneA_rr_gate'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
PTB_FEAT = ROOT / 'runs/r14_rr_baseline/features.npz'
ELECTRON = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                'ecg-programme-tju-flex.electron-master')
DEVICE = {
    'ecg_real_052': ELECTRON / 'pc_tools/ecg_dl/data/real/ecg_real_052.ecgr',
    'rec_latest': ELECTRON / 'rec_latest.ecgr',
}
FEATURES = ['rmssd', 'sdnn', 'rr_entropy', 'drr_entropy', 'sampen_m1',
            'cosen']
PRIMARY = 'rr_entropy'
WIN_S = 10.0
MIN_RR = 6
FS100 = 100


def read_ecgr(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 18)[0]
    dur = struct.unpack_from('<I', raw, 14)[0]
    x = np.frombuffer(raw, dtype='<i2', count=n, offset=32).astype(np.float64)
    return x / 8000.0, n / dur, dur


def deploy_chain250(path):
    sys.path.insert(0, str(ELECTRON / 'pc_tools/ecg_dl'))
    from eval_deploy_match import causal_hp_05_fs250, deployment_chain
    x, fs, dur = read_ecgr(path)
    ch = causal_hp_05_fs250(deployment_chain(x, fs))
    return ch, fs, dur


def features_for_rr(rr):
    f = features_from_rr(rr)
    return {k: f.get(k, np.nan) for k in FEATURES}, len(rr)


def window_features(peak_times, dur_s, stride_s=1.0):
    """10 s sliding windows; RR intervals with both peaks inside."""
    starts = np.arange(0.0, max(dur_s - WIN_S + stride_s, stride_s), stride_s)
    rows, meta = [], []
    for s in starts:
        m = (peak_times >= s) & (peak_times < s + WIN_S)
        pk = peak_times[m]
        rr = np.diff(pk)
        rr = rr[(rr >= 0.30) & (rr <= 2.00)]
        f, n = features_for_rr(rr)
        if n < MIN_RR:
            f = {k: np.nan for k in FEATURES}
        rows.append(f)
        meta.append({'start_s': round(float(s), 2), 'n_rr': int(n)})
    X = {k: np.array([r[k] for r in rows], float) for k in FEATURES}
    return X, meta


def afdb_features(max_per_class, seed):
    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(seed)
    rows = {k: [] for k in FEATURES}
    out_rec, out_lab, out_nrr = [], [], []
    for r in sorted(set(rec)):
        idx = np.where(rec == r)[0]
        obj = wfdb.rdrecord(str(AFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        for L in (1, 0):
            cand = idx[lab[idx] == L]
            if len(cand) == 0:
                continue
            take = rng.choice(cand, min(max_per_class, len(cand)),
                              replace=False)
            for i in take:
                w = sig[start[i]:start[i] + 2500]
                if len(w) < 2500:
                    continue
                w100 = resample_poly(w, 2, 5)
                rr, _ = detect_rr(w100)
                f, n = features_for_rr(rr)
                if n < MIN_RR:
                    f = {k: np.nan for k in FEATURES}
                for k in FEATURES:
                    rows[k].append(f[k])
                out_rec.append(r)
                out_lab.append(int(L))
                out_nrr.append(int(n))
        print(f'afdb features {r}: total {len(out_lab)}', flush=True)
    X = {k: np.array(v, float) for k, v in rows.items()}
    return X, np.array(out_rec), np.array(out_lab, int), np.array(out_nrr, int)


def device_peak_times(name):
    """Primary: deploy-chain XQRS @250 Hz; sensitivity: PT100 @100 Hz."""
    from wfdb.processing import xqrs_detect
    ch250, fs, dur = deploy_chain250(DEVICE[name])
    r250 = xqrs_detect(ch250.astype(np.float64), fs=250, verbose=False)
    t_primary = r250 / 250.0
    ch100 = resample_poly(ch250, 2, 5)
    p100 = detect_peaks(ch100)
    t_sens = p100 / float(FS100)
    return {'deploy_xqrs250': t_primary, 'pt100': t_sens,
            'dur_s': float(dur), 'fs_raw': float(fs),
            'n_peaks_xqrs': int(len(r250)), 'n_peaks_pt100': int(len(p100))}


def m4_alarms(flags, confirm=5, cooldown=20):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def fit_theta(scores, y):
    m = np.isfinite(scores)
    th_youden = youden_threshold(y[m], scores[m])
    th_r90 = train_theta_at_recall(y[m], scores[m], 0.90)
    out = {'n_fit': int(m.sum()), 'pos_fit': int(y[m].sum()),
           'theta_youden': round(float(th_youden), 4),
           'theta_R90': round(float(th_r90), 4) if th_r90 else None}
    for key, th in [('youden', th_youden), ('R90', th_r90)]:
        if th is None:
            continue
        yp = scores[m] >= th
        rec = float((yp & (y[m] == 1)).sum()) / max(int((y[m] == 1).sum()), 1)
        spec = float(((~yp) & (y[m] == 0)).sum()) / max(
            int((y[m] == 0).sum()), 1)
        out[f'fit_R_{key}'] = round(rec, 4)
        out[f'fit_spec_{key}'] = round(spec, 4)
    return out


def eval_afdb(scores, y, theta):
    m = np.isfinite(scores)
    yp = np.zeros(len(scores), bool)
    yp[m] = scores[m] >= theta
    rec = float((yp & (y == 1)).sum()) / max(int((y == 1).sum()), 1)
    spec = float(((~yp) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)
    return {'n': int(len(y)), 'n_valid': int(m.sum()),
            'recall': round(rec, 4), 'spec': round(spec, 4),
            'n_flag': int(yp.sum())}


def eval_device(X, theta, rec_meta):
    scores = X[PRIMARY]
    flags = np.zeros(len(scores), bool)
    m = np.isfinite(scores)
    flags[m] = scores[m] >= theta
    alarms = m4_alarms(flags)
    return {'n_windows': int(len(scores)),
            'n_valid': int(m.sum()),
            'n_flag': int(flags.sum()),
            'flag_frac_valid': round(float(flags.sum()) / max(int(m.sum()), 1), 4),
            'max_score': round(float(np.nanmax(scores)), 4) if m.any() else None,
            'median_score': round(float(np.nanmedian(scores)), 4) if m.any() else None,
            'm4_alarms': int(len(alarms)),
            'm4_alarm_starts_s': [round(float(rec_meta[i]['start_s']), 2)
                                  for i in alarms[:10]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-per-class', type=int, default=600)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--stride-s', type=float, default=1.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- fit side: PTB-XL folds1-9 (A4 features) ----
    pf = np.load(PTB_FEAT)
    ptb = {k: pf[f'tr_{k}'].astype(float) for k in FEATURES}
    ptb_y = pf['y_tr'].astype(int)
    ptb_rec = np.array(['ptb'] * len(ptb_y))

    # ---- fit/eval side: afdb record-level parity split ----
    Xa, rec_a, y_a, nrr_a = afdb_features(args.max_per_class, args.seed)
    records = sorted(set(rec_a))
    fit_recs = [r for i, r in enumerate(records) if i % 2 == 0]
    eval_recs = [r for i, r in enumerate(records) if i % 2 == 1]
    fit_mask = np.isin(rec_a, fit_recs)
    eval_mask = np.isin(rec_a, eval_recs)
    np.savez_compressed(
        OUT / 'features.npz',
        **{f'afdb_{k}': v for k, v in Xa.items()},
        afdb_rec=rec_a, afdb_y=y_a, afdb_nrr=nrr_a,
        ptb_y=ptb_y, fit_recs=np.array(fit_recs), eval_recs=np.array(eval_recs))

    # ---- pooled fit scores ----
    fit_scores = np.concatenate([ptb[PRIMARY], Xa[PRIMARY][fit_mask]])
    fit_y = np.concatenate([ptb_y, y_a[fit_mask]])
    fit_domain = np.concatenate([ptb_rec, rec_a[fit_mask]])
    # sign orientation (higher = AF)
    if np.isfinite(fit_scores).sum() > 10 and roc_auc_score(
            fit_y[np.isfinite(fit_scores)],
            fit_scores[np.isfinite(fit_scores)]) < 0.5:
        for k in FEATURES:
            ptb[k] = -ptb[k]
            Xa[k] = -Xa[k]
        fit_scores = -fit_scores
    single = {}
    for k in FEATURES:
        sc = np.concatenate([ptb[k], Xa[k][fit_mask]])
        mm = np.isfinite(sc)
        a = (round(float(roc_auc_score(fit_y[mm], sc[mm])), 4)
             if len(np.unique(fit_y[mm])) > 1 else None)
        ft = fit_theta(sc, fit_y)
        ev = {}
        for th in ('youden', 'R90'):
            if ft[f'theta_{th}'] is not None:
                ev[th] = eval_afdb(Xa[k][eval_mask], y_a[eval_mask],
                                   ft[f'theta_{th}'])
        single[k] = {'fit_auc': a, 'fit': ft, 'afdb_eval': ev}

    # ---- LR combo on the pooled fit set ----
    def design(Xd, med=None):
        Xm = np.column_stack([Xd[k] for k in FEATURES])
        ind = np.column_stack([np.isnan(Xm[:, j]).astype(float)
                               for j in range(Xm.shape[1])])
        if med is None:
            med = np.nanmedian(Xm, axis=0)
        return np.hstack([np.where(np.isnan(Xm), med, Xm), ind]), med

    Xf, med = design({k: np.concatenate([ptb[k], Xa[k][fit_mask]])
                      for k in FEATURES})
    mu, sd = Xf.mean(0), Xf.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    lr.fit((Xf - mu) / sd, fit_y)
    fit_lr = lr.predict_proba((Xf - mu) / sd)[:, 1]
    Xe, _ = design({k: Xa[k][eval_mask] for k in FEATURES}, med)
    eval_lr = lr.predict_proba((Xe - mu) / sd)[:, 1]
    lr_fit = fit_theta(fit_lr, fit_y)
    lr_eval = {th: eval_afdb(eval_lr, y_a[eval_mask],
                             lr_fit[f'theta_{th}'])
               for th in ('youden', 'R90') if lr_fit[f'theta_{th}'] is not None}

    # ---- device domain ----
    fit_primary = fit_theta(fit_scores, fit_y)
    th_r90 = fit_primary['theta_R90']
    th_youden = fit_primary['theta_youden']
    device = {}
    for name in DEVICE:
        pk = device_peak_times(name)
        entry = {k: v for k, v in pk.items() if not k.endswith('_xqrs250')
                 and not k.endswith('pt100')}
        for det in ('deploy_xqrs250', 'pt100'):
            X, meta = window_features(pk[det], pk['dur_s'], args.stride_s)
            entry[det] = {'peak_source': det,
                          'n_peaks': pk['n_peaks_xqrs'] if det ==
                          'deploy_xqrs250' else pk['n_peaks_pt100'],
                          'primary': eval_device(X, th_r90, meta),
                          'youden': eval_device(X, th_youden, meta)}
        device[name] = entry

    afdb_eval_r90 = eval_afdb(Xa[PRIMARY][eval_mask], y_a[eval_mask], th_r90)
    device_alarms0 = all(
        device[n]['deploy_xqrs250']['primary']['m4_alarms'] == 0
        for n in device)
    device_flags0 = all(
        device[n]['deploy_xqrs250']['primary']['n_flag'] == 0
        for n in device)
    device_fp0 = bool(device_alarms0)
    verdict = bool(device_fp0 and afdb_eval_r90['recall'] >= 0.90)

    result = {
        'arm': 'r15_laneA_rr_gate',
        'protocol': ('RR-only Lane A: rr_entropy threshold fitted on PTB-XL '
                     'folds1-9 + afdb fit records (record-parity split); '
                     'device domain acceptance only; 10 s sliding windows '
                     '(1 s stride), >=6 RR; M4 protocol 1-of-5 cooldown 20'),
        'primary_feature': PRIMARY,
        'thresholds': {'theta_youden': round(float(th_youden), 4),
                       'theta_R90': round(float(th_r90), 4)},
        'fit': fit_theta(fit_scores, fit_y),
        'single_feature': single,
        'lr_combo': {'fit': lr_fit, 'afdb_eval': lr_eval,
                     'coef': [round(float(c), 4) for c in lr.coef_[0]]},
        'afdb_split': {'fit_records': fit_recs, 'eval_records': eval_recs,
                       'n_fit_windows': int(fit_mask.sum()),
                       'n_eval_windows': int(eval_mask.sum()),
                       'n_eval_af': int(y_a[eval_mask].sum())},
        'afdb_eval_primary': afdb_eval_r90,
        'afdb_eval_youden': eval_afdb(Xa[PRIMARY][eval_mask],
                                      y_a[eval_mask], th_youden),
        'device': device,
        'device_afhead_not_used': True,
        'acceptance': {
            'criterion_1_device_fp0': 'both recordings 0 M4 event alarms at '
                                      'theta_R90 (primary device R-peaks = '
                                      'deploy-chain XQRS); raw flags reported '
                                      'separately',
            'criterion_2_afdb_recall': 'afdb held-out records AF recall '
                                       '>= 0.90 at theta_R90',
            'device_m4_alarms0_pass': bool(device_alarms0),
            'device_raw_flags0': bool(device_flags0),
            'device_fp0_pass': bool(device_fp0),
            'afdb_recall_pass': bool(afdb_eval_r90['recall'] >= 0.90),
            'pass': verdict,
        },
        'disclosure': ('only 2 device normal recordings (no device AF '
                       'positives): PASS proves normal-side silence; '
                       'sensitivity from afdb/PTB-XL; RR detector quality '
                       'cross-checked in R14-A4 (500 Hz XQRS); 10 s windows '
                       'give ~10 RR intervals, entropy variance disclosed '
                       '(Lake 2011 >=12 beats)'),
    }
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: result[k] for k in
                      ['thresholds', 'fit', 'afdb_split',
                       'afdb_eval_primary', 'afdb_eval_youden',
                       'acceptance']}, indent=2, ensure_ascii=False), flush=True)
    for n in device:
        print(n, json.dumps({d: device[n][d]['primary']
                             for d in ('deploy_xqrs250', 'pt100')},
                            ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
