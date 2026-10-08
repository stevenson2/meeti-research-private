#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A4: zero-training RR-irregularity baseline on fold-10 lead II.

Detector (corrected): Pan-Tompkins style -- bandpass 5-20 Hz, derivative,
squaring, 150 ms moving integration, scipy find_peaks on the integrated
signal (distance 0.25 s, height 0.25*max), then refine to the local maximum
of the band-passed signal.  The first version of this script used plain
find_peaks on the 5-15 Hz signal and massively over-detected (median 26 vs
13 true beats per 10 s window, median RR 0.27 s); it is archived as
result_naive_findpeaks_INVALID.json and replaced here.  Detector agreement
is cross-checked against wfdb XQRS on the 500 Hz records (see --validate).

Features: {rmSSD, SDRR, RR Shannon entropy (normalized), dRR Shannon
entropy, SampEn(m=1, r=0.25*SD), COSEn simplified = SampEn + ln(2r) -
ln(meanRR)}.

Reported on fold-10 (AF vs rest): single-feature AUC + logistic-regression
combination fitted on folds1-9 (no fold-10 fitting).  R-peak quality is NOT
manually validated on PTB-XL (vision tool unavailable this session); a
quantitative 500 Hz XQRS cross-check is provided instead.  This arm
establishes the feasibility coordinate of the RR confirmation channel only
-- it is not a deployment gate.

Outputs: runs/r14_rr_baseline/{features.npz,result.json,spotcheck_20.png,
detector_crosscheck_500hz.json}
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

DATA = Path('/home/devcontainers/meeti/runs/r14_af_prep')
OUT = Path('/home/devcontainers/meeti/runs/r14_rr_baseline')
FS = 100.0
BANDPASS = (5.0, 20.0)
MIN_RR = 0.30
MAX_RR = 2.00
FEATURES = ['rmssd', 'sdnn', 'rr_entropy', 'drr_entropy', 'sampen_m1',
            'cosen']


def bandpass(x):
    b, a = butter(3, [BANDPASS[0] / (FS / 2), BANDPASS[1] / (FS / 2)],
                  btype='band')
    return filtfilt(b, a, x)


def detect_peaks(x):
    """Pan-Tompkins style R-peak detection on one 10 s record (100 Hz)."""
    f = bandpass(np.asarray(x, np.float64))
    d = np.diff(f, prepend=f[0])
    integ = uniform_filter1d(d * d, size=15, mode='nearest')
    mx = float(integ.max())
    if not np.isfinite(mx) or mx <= 0:
        return np.array([], np.int64)
    peaks, _ = find_peaks(integ, distance=int(0.25 * FS), height=0.25 * mx)
    refined = []
    for q in peaks:
        lo, hi = max(0, q - 10), min(len(f), q + 11)
        refined.append(lo + int(np.argmax(f[lo:hi])))
    return np.array(sorted(set(refined)), np.int64)


def detect_rr(x):
    peaks = detect_peaks(x)
    if len(peaks) < 3:
        return np.array([]), peaks
    rr = np.diff(peaks) / FS
    rr = rr[(rr >= MIN_RR) & (rr <= MAX_RR)]
    return rr, peaks


def shannon_entropy(v, bins):
    h, _ = np.histogram(v, bins=bins)
    p = h.astype(np.float64)
    tot = p.sum()
    if tot <= 0:
        return np.nan
    p = p[p > 0] / tot
    return float(-(p * np.log2(p)).sum())


def sampen(x, m=1, r=None):
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.25 * np.std(x, ddof=0)
    if not np.isfinite(r) or r <= 0:
        return np.nan

    def count(mm):
        c = 0
        for i in range(n - mm):
            vi = x[i:i + mm]
            for j in range(i + 1, n - mm):
                if np.max(np.abs(vi - x[j:j + mm])) <= r:
                    c += 1
        return c

    b, a = count(m), count(m + 1)
    if b <= 0 or a <= 0:
        return np.nan
    return float(-np.log(a / b))


def features_from_rr(rr):
    out = {k: np.nan for k in FEATURES}
    out['n_rr'] = len(rr)
    if len(rr) < 3:
        return out
    drr = np.diff(rr)
    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(drr ** 2)))
    mean_rr = float(np.mean(rr))
    r = 0.25 * sdnn
    se = sampen(rr, m=1, r=r) if sdnn > 0 else np.nan
    out.update(
        rmssd=rmssd, sdnn=sdnn,
        rr_entropy=shannon_entropy(rr, np.arange(0.30, 2.01, 0.05)),
        drr_entropy=shannon_entropy(np.abs(drr), np.arange(0.0, 0.401, 0.02)),
        sampen_m1=se,
        cosen=(se + np.log(2 * r) - np.log(mean_rr)
               if np.isfinite(se) and r > 0 else np.nan),
        mean_rr=mean_rr, hr=60.0 / mean_rr,
        qc_hr_ok=(30.0 <= 60.0 / mean_rr <= 220.0),
        qc_tachy_suspect=bool(mean_rr < 0.35),
    )
    return out


def extract(x):
    rows = []
    t0 = time.time()
    for i in range(len(x)):
        rr, _ = detect_rr(x[i])
        rows.append(features_from_rr(rr))
        if (i + 1) % 5000 == 0:
            print(f'extracted {i+1}/{len(x)} elapsed={time.time()-t0:.0f}s',
                  flush=True)
    keys = ['rmssd', 'sdnn', 'rr_entropy', 'drr_entropy', 'sampen_m1',
            'cosen', 'mean_rr', 'hr', 'n_rr', 'qc_hr_ok', 'qc_tachy_suspect']
    return {k: np.array([r.get(k, np.nan) for r in rows], np.float64)
            for k in keys}


def auc_safe(y, s):
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return None, int(m.sum())
    return round(float(roc_auc_score(y[m], s[m])), 4), int(m.sum())


def youden(y, s):
    m = np.isfinite(s)
    o = np.argsort(s[m])
    ys, ss = y[m][o], s[m][o]
    P, N = int(ys.sum()), int(len(ys) - ys.sum())
    tp = P - np.cumsum(ys)
    fp = N - np.cumsum(1 - ys)
    j = tp / max(P, 1) - fp / max(N, 1)
    i = int(np.argmax(j[:-1])) + 1
    return float(ss[min(i, len(ss) - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spotcheck', action='store_true')
    ap.add_argument('--validate', type=int, default=0,
                    help='cross-check N records against 500 Hz XQRS')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    tr = np.load(DATA / 'ptbxl_af_train.npz')
    te = np.load(DATA / 'ptbxl_af_test.npz')
    x_tr, x_te = tr['x'], te['x']
    y_tr, y_te = tr['y_af'].astype(int), te['y_af'].astype(int)

    print('extracting train features', flush=True)
    f_tr = extract(x_tr)
    print('extracting test features', flush=True)
    f_te = extract(x_te)
    np.savez_compressed(
        OUT / 'features.npz',
        **{f'tr_{k}': v for k, v in f_tr.items()},
        **{f'te_{k}': v for k, v in f_te.items()},
        y_tr=y_tr, y_te=y_te, ecg_tr=tr['ecg_ids'], ecg_te=te['ecg_ids'])

    single = {}
    for k in FEATURES:
        a, n = auc_safe(y_te, f_te[k])
        a_tr, n_tr = auc_safe(y_tr, f_tr[k])
        single[k] = {'auc_fold10': a, 'n_valid_test': n,
                     'auc_train': a_tr, 'n_valid_train': n_tr}

    # logistic regression combination, fitted on folds1-9 only
    Xtr = np.column_stack([f_tr[k] for k in FEATURES])
    Xte = np.column_stack([f_te[k] for k in FEATURES])
    med = np.nanmedian(Xtr, axis=0)
    ind_tr = np.column_stack([np.isnan(Xtr[:, j]).astype(float)
                              for j in range(Xtr.shape[1])])
    ind_te = np.column_stack([np.isnan(Xte[:, j]).astype(float)
                              for j in range(Xte.shape[1])])
    Xtr_f = np.hstack([np.where(np.isnan(Xtr), med, Xtr), ind_tr])
    Xte_f = np.hstack([np.where(np.isnan(Xte), med, Xte), ind_te])
    mu, sd = Xtr_f.mean(0), Xtr_f.std(0)
    sd[sd == 0] = 1.0
    lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=2000)
    lr.fit((Xtr_f - mu) / sd, y_tr)
    p_tr = lr.predict_proba((Xtr_f - mu) / sd)[:, 1]
    p_te = lr.predict_proba((Xte_f - mu) / sd)[:, 1]
    theta_lr = youden(y_tr, p_tr)
    yp = p_te >= theta_lr
    rec = float((yp & (y_te == 1)).sum()) / max(int((y_te == 1).sum()), 1)
    spec = float(((~yp) & (y_te == 0)).sum()) / max(
        int((y_te == 0).sum()), 1)

    # RR entropy confirmator, sign-oriented by train AUC
    ent_tr, ent_te = f_tr['rr_entropy'], f_te['rr_entropy']
    a_ent_tr, _ = auc_safe(y_tr, ent_tr)
    if a_ent_tr is not None and a_ent_tr < 0.5:
        ent_tr, ent_te = -ent_tr, -ent_te
    a_ent, n_ent = auc_safe(y_te, ent_te)
    theta_ent = youden(y_tr, ent_tr)
    mte = np.isfinite(ent_te)
    yp = ent_te[mte] >= theta_ent
    rec_ent = float((yp & (y_te[mte] == 1)).sum()) / max(
        int((y_te[mte] == 1).sum()), 1)
    spec_ent = float(((~yp) & (y_te[mte] == 0)).sum()) / max(
        int((y_te[mte] == 0).sum()), 1)

    quality = {}
    for split, f in [('train', f_tr), ('test', f_te)]:
        n_rr, hr = f['n_rr'], f['hr']
        quality[split] = {
            'coverage_ge3_rr': round(float((n_rr >= 3).mean()), 4),
            'n_rr_median': float(np.median(n_rr)),
            'n_rr_p5_p95': [float(np.percentile(n_rr, 5)),
                            float(np.percentile(n_rr, 95))],
            'hr_median': round(float(np.nanmedian(hr)), 2),
            'hr_outside_30_220_frac': round(float(np.mean(
                (~np.isfinite(hr)) | (hr < 30) | (hr > 220))), 4),
            'tachy_suspect_frac_medRR_lt_0.35': round(
                float(np.mean(f['qc_tachy_suspect'])), 4),
        }

    result = {
        'arm': 'r14_rr_baseline',
        'detector': ('Pan-Tompkins style: bandpass 5-20 Hz -> diff -> square '
                     '-> 150 ms moving integration -> scipy find_peaks '
                     '(dist 0.25 s, height 0.25*max) -> local-max refine'),
        'detector_version_note': ('v1 (plain find_peaks on 5-15 Hz) was '
                                  'invalid (over-detection: median 26 vs 13 '
                                  'true beats); archived as '
                                  'result_naive_findpeaks_INVALID.json'),
        'protocol': ('fold-10 lead II 10 s; RR filter [0.3,2.0] s; '
                     'single-feature AUC + LR combo fitted on folds1-9 only'),
        'disclosure': ('R-peak quality not manually validated on PTB-XL '
                       '(vision tool unavailable); 500 Hz XQRS cross-check '
                       'reported. 10 s window => few RR intervals, entropy '
                       'estimates noisy (Lake 2011: >=12 beats for stable '
                       'COSEn). Feasibility coordinate for the RR '
                       'confirmation channel, not a deployment gate.'),
        'n_train': int(len(x_tr)), 'n_test': int(len(x_te)),
        'af_prevalence_test': round(float(y_te.mean()), 4),
        'single_feature': single,
        'lr_combo': {
            'auc_fold10': round(float(roc_auc_score(y_te, p_te)), 4),
            'ap_fold10': round(float(average_precision_score(y_te, p_te)), 4),
            'theta_tr_youden': round(theta_lr, 4),
            'R_at_theta_tr': round(rec, 4), 'spec_at_theta_tr': round(spec, 4),
            'coef': [round(float(c), 4) for c in lr.coef_[0]],
            'feature_order': FEATURES + [f'isnan_{k}' for k in FEATURES],
        },
        'rr_entropy_confirmator': {
            'feature': 'rr_entropy (normalized Shannon, sign-oriented by '
                       'train AUC)',
            'auc_fold10': a_ent, 'n_valid_test': n_ent,
            'theta_tr_youden': round(theta_ent, 4),
            'R_at_theta_tr': round(rec_ent, 4),
            'spec_at_theta_tr': round(spec_ent, 4),
        },
        'quality': quality,
    }
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)

    if args.validate:
        validate_500hz(args.validate)
    if args.spotcheck:
        make_spotcheck(x_te, y_te, f_te)


def validate_500hz(n_rec):
    """Cross-check PT100 against wfdb XQRS on the 500 Hz records."""
    import wfdb
    from wfdb.processing import xqrs_detect
    from scipy.signal import resample_poly
    from project_paths import P as _PP  # 统一路径
    ptbxl = Path(_PP['ptbxl_copy'])
    df = pd.read_csv(ptbxl / 'ptbxl_database.csv', index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(eval)
    df['y_af'] = df['scp_codes'].apply(lambda d: 1 if 'AFIB' in d else 0)
    te = df[df['strat_fold'] == 10]
    rng = np.random.default_rng(5)
    k = n_rec // 2
    sel = np.concatenate([
        rng.choice(te[te.y_af == 1].index.to_numpy(), k, replace=False),
        rng.choice(te[te.y_af == 0].index.to_numpy(), k, replace=False)])
    rows = []
    for eid in sel:
        row = df.loc[eid]
        sig, _ = wfdb.rdsamp(str(ptbxl / row['filename_hr']))
        s = sig[:, 1].astype(np.float64)
        try:
            p500 = xqrs_detect(s, fs=500, verbose=False)
        except Exception as e:
            print('xqrs500 failed', e, flush=True)
            p500 = np.array([])
        s100 = resample_poly(s, 1, 5)
        p100 = detect_peaks(s100)
        rows.append((int(row['y_af']), len(p500), len(p100),
                     float(np.median(np.diff(p500) / 500.))
                     if len(p500) > 1 else np.nan,
                     float(np.median(np.diff(p100) / 100.))
                     if len(p100) > 1 else np.nan))
    r = np.array(rows)
    d = r[:, 1] - r[:, 2]
    ok = np.isfinite(r[:, 3]) & np.isfinite(r[:, 4])
    out = {
        'n': int(len(r)),
        'beat_count_diff_median': float(np.median(d)),
        'beat_count_within1_frac': round(float((np.abs(d) <= 1).mean()), 3),
        'beat_count_within2_frac': round(float((np.abs(d) <= 2).mean()), 3),
        'beat_count_exact_frac': round(float((d == 0).mean()), 3),
        'median_rr_corr': round(float(np.corrcoef(r[ok, 3], r[ok, 4])[0, 1]), 4)
        if ok.sum() > 2 else None,
        'note': 'reference = wfdb XQRS on 500 Hz lead II',
    }
    (OUT / 'detector_crosscheck_500hz.json').write_text(
        json.dumps(out, indent=2), encoding='utf-8')
    print('VALIDATE', json.dumps(out, indent=2), flush=True)


def make_spotcheck(x, y, f):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(7)
    idx_af = rng.choice(np.where(y == 1)[0], 10, replace=False)
    idx_no = rng.choice(np.where(y == 0)[0], 10, replace=False)
    idx = np.concatenate([idx_af, idx_no])
    fig, axes = plt.subplots(5, 4, figsize=(18, 14))
    for ax, i in zip(axes.ravel(), idx):
        sig = x[i]
        rr, peaks = detect_rr(sig)
        fsig = bandpass(sig)
        ax.plot(sig, lw=0.7, color='0.5')
        ax.plot(fsig, lw=0.7, color='tab:blue')
        ax.plot(peaks, fsig[peaks], 'rv', ms=4)
        ttl = (f'idx{i} y={y[i]} n_rr={len(rr)} hr={60/np.mean(rr):.0f}'
               if len(rr) else f'idx{i} y={y[i]} no rr')
        ax.set_title(ttl, fontsize=8)
        ax.set_xticks([])
    fig.suptitle('R14-A4 spot check: 10 AF + 10 non-AF (PT100 detector)')
    fig.tight_layout()
    fig.savefig(OUT / 'spotcheck_20.png', dpi=90)
    print('saved spotcheck', flush=True)


if __name__ == '__main__':
    main()
