#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16: extended RR features (irregularity + shape) for Lane A.

Reuses the R14-A4 detector/feature family (features_from_rr) and adds
shape-aware descriptors intended to separate smooth respiratory sinus
arrhythmia from the abrupt RR irregularity of AF.

All features are computed on an RR interval sequence in seconds (already
range-filtered by the caller).  Orientation (higher = AF) is decided at
fit time by the caller; nothing here is thresholded.

Feature groups
--------------
irregularity baseline : rmssd, sdnn, rr_entropy, drr_entropy, cosen,
                        sampen_m1 (identical semantics to R14-A4)
shape                 : lag-1/2/3 autocorrelation, RMSSD/SDNN, Poincare
                        SD1/SD2 (+ratios), pNN50/pNN20, turning-point ratio,
                        dRR sign-alternation entropy, RR trend slope,
                        SampEn(m=2), FuzzyEn(m=1/2), DistEn(m=2),
                        Katz/Higuchi fractal dimension, DFA alpha1,
                        smoothness (mean|ddRR| / mean|dRR|), normalized
                        entropy/range/IQR/mean|dRR|.

Short-window caveat: a 10 s window has ~10-13 RR intervals; entropy /
sample-entropy estimates are high variance (Lake 2011: >=12 beats for COSEn).
Callers must report per-window validity rates and the 30 s comparison.
"""
import numpy as np

EPS = 1e-12

BASE_FEATURES = ['rmssd', 'sdnn', 'rr_entropy', 'drr_entropy', 'sampen_m1',
                 'cosen']
SHAPE_FEATURES = [
    'autocorr1', 'autocorr2', 'autocorr3',
    'rmssd_over_sdnn', 'sd1', 'sd2', 'sd1_sd2_ratio', 'sd2_sd1_ratio',
    'pnn50', 'pnn20', 'tpr', 'drr_sign_entropy', 'trend_slope_norm',
    'sampen_m2_r02', 'sampen_m2_r025', 'fuzzyen_m1', 'fuzzyen_m2',
    'disten_m2', 'katz_fd', 'higuchi_fd', 'dfa_alpha1', 'smoothness',
    'entropy_norm', 'drr_entropy_norm', 'range_norm', 'iqr_norm',
    'mean_abs_drr_norm', 'cv',
]
ALL_FEATURES = BASE_FEATURES + SHAPE_FEATURES


# ---------------------------------------------------------------- helpers
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
            for j in range(i + 1, n - mm + 1):
                if np.max(np.abs(vi - x[j:j + mm])) <= r:
                    c += 1
        return c

    b, a = count(m), count(m + 1)
    if b <= 0 or a <= 0:
        return np.nan
    return float(-np.log(a / b))


def fuzzyen(x, m=2, r=None, n_exp=2.0):
    """Fuzzy entropy, exponential membership exp(-ln2*(d/r)^n)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < m + 2:
        return np.nan
    if r is None:
        r = 0.2 * np.std(x, ddof=0)
    if not np.isfinite(r) or r <= 0:
        return np.nan

    def phi(mm):
        nvec = n - mm + 1
        if nvec < 2:
            return np.nan
        Y = np.lib.stride_tricks.sliding_window_view(x, mm)
        d = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
        S = np.exp(-np.log(2.0) * (d / r) ** n_exp)
        np.fill_diagonal(S, 0.0)
        return float(S.sum() / (nvec * (nvec - 1)))

    pm, pm1 = phi(m), phi(m + 1)
    if not np.isfinite(pm) or not np.isfinite(pm1) or pm <= 0 or pm1 <= 0:
        return np.nan
    return float(np.log(pm) - np.log(pm1))


def disten(x, m=2, n_bins=10):
    """Distribution entropy of the Chebyshev distance matrix (Li 2015)."""
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < m + 2:
        return np.nan
    Y = np.lib.stride_tricks.sliding_window_view(x, m)
    d = np.max(np.abs(Y[:, None, :] - Y[None, :, :]), axis=2)
    iu = np.triu_indices(len(Y), k=1)
    dd = d[iu]
    if len(dd) < 4 or dd.max() <= dd.min():
        return np.nan
    hist, _ = np.histogram(dd, bins=n_bins, range=(dd.min(), dd.max()))
    p = hist.astype(np.float64)
    p = p[p > 0] / p.sum()
    return float(-(p * np.log2(p)).sum() / np.log2(n_bins))


def katz_fd(x):
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < 3:
        return np.nan
    x = x / max(np.mean(np.abs(x)), EPS)
    steps = np.sqrt(1.0 + np.diff(x) ** 2)
    L = float(np.sum(steps))
    d = float(np.max(np.sqrt(np.arange(n, dtype=float) ** 2
                             + (x - x[0]) ** 2)))
    if L <= 0 or d <= 0:
        return np.nan
    den = np.log10(d / L) + np.log10(n)
    if abs(den) < EPS:
        return np.nan
    return float(np.log10(n) / den)


def higuchi_fd(x, kmax=4):
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < 2 * kmax:
        return np.nan
    lk = []
    for k in range(1, kmax + 1):
        lm = []
        for m in range(k):
            idx = np.arange(m, n, k)
            if len(idx) < 2:
                continue
            lmk = float(np.sum(np.abs(np.diff(x[idx]))))
            norm = (n - 1) / (len(idx) - 1) / k
            lm.append(lmk * norm)
        if lm:
            lk.append(float(np.mean(lm)))
    if len(lk) < 2 or min(lk) <= 0:
        return np.nan
    kk = np.arange(1, len(lk) + 1, dtype=float)
    return float(np.polyfit(np.log(1.0 / kk), np.log(lk), 1)[0])


def dfa_alpha1(x, scales=(4, 6, 8)):
    x = np.asarray(x, np.float64)
    n = len(x)
    if n < max(scales) + 1:
        return np.nan
    y = np.cumsum(x - np.mean(x))
    f = []
    used = []
    for s in scales:
        if n < s + 1:
            continue
        nseg = n // s
        if nseg < 1:
            continue
        rms = []
        for i in range(nseg):
            seg = y[i * s:(i + 1) * s]
            t = np.arange(s, dtype=float)
            coef = np.polyfit(t, seg, 1)
            resid = seg - np.polyval(coef, t)
            rms.append(float(np.mean(resid ** 2)))
        f.append(np.sqrt(np.mean(rms)))
        used.append(s)
    if len(f) < 2 or min(f) <= 0:
        return np.nan
    return float(np.polyfit(np.log(used), np.log(f), 1)[0])


def _autocorr(x, lag):
    x = np.asarray(x, np.float64)
    if len(x) < lag + 4:
        return np.nan
    a, b = x[:-lag], x[lag:]
    if np.std(a) < EPS or np.std(b) < EPS:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def features_from_rr_ext(rr):
    """Extended feature dict for one RR sequence (seconds)."""
    rr = np.asarray(rr, np.float64)
    n = len(rr)
    out = {k: np.nan for k in ALL_FEATURES}
    out.update(n_rr=int(n), mean_rr=np.nan, median_rr=np.nan, hr=np.nan)
    if n < 3:
        return out
    drr = np.diff(rr)
    mean_rr = float(np.mean(rr))
    med_rr = float(np.median(rr))
    sdnn = float(np.std(rr, ddof=1))
    rmssd = float(np.sqrt(np.mean(drr ** 2)))
    r025 = 0.25 * sdnn
    se1 = sampen(rr, m=1, r=r025) if sdnn > 0 else np.nan
    rr_ent = shannon_entropy(rr, np.arange(0.30, 2.01, 0.05))
    drr_ent = shannon_entropy(np.abs(drr), np.arange(0.0, 0.401, 0.02))
    out.update(
        n_rr=int(n), mean_rr=mean_rr, median_rr=med_rr, hr=60.0 / mean_rr,
        rmssd=rmssd, sdnn=sdnn, rr_entropy=rr_ent, drr_entropy=drr_ent,
        sampen_m1=se1,
        cosen=(se1 + np.log(2 * r025) - np.log(mean_rr)
               if np.isfinite(se1) and r025 > 0 else np.nan),
        autocorr1=_autocorr(rr, 1), autocorr2=_autocorr(rr, 2),
        autocorr3=_autocorr(rr, 3),
        rmssd_over_sdnn=(rmssd / sdnn if sdnn > EPS else np.nan),
        pnn50=float(np.mean(np.abs(drr) > 0.05)),
        pnn20=float(np.mean(np.abs(drr) > 0.02)),
        mean_abs_drr_norm=(float(np.mean(np.abs(drr))) / mean_rr),
        cv=sdnn / mean_rr,
        range_norm=(float(np.max(rr) - np.min(rr)) / med_rr),
        iqr_norm=(float(np.percentile(rr, 75)
                        - np.percentile(rr, 25)) / med_rr),
        entropy_norm=(rr_ent / np.log2(n) if rr_ent == rr_ent and n > 1
                      else np.nan),
        drr_entropy_norm=(drr_ent / np.log2(max(len(drr), 2))
                          if drr_ent == drr_ent else np.nan),
    )
    # Poincare
    if n >= 4:
        sd1 = float(np.sqrt(0.5) * np.std(drr, ddof=1))
        sd2_sq = 2.0 * sdnn ** 2 - 0.5 * np.std(drr, ddof=1) ** 2
        sd2 = float(np.sqrt(max(sd2_sq, 0.0)))
        out.update(sd1=sd1, sd2=sd2,
                   sd1_sd2_ratio=(sd1 / sd2 if sd2 > EPS else np.nan),
                   sd2_sd1_ratio=(sd2 / sd1 if sd1 > EPS else np.nan))
    # turning point ratio / sign runs
    if n >= 3:
        s = np.sign(drr)
        s = s[s != 0]
        out['tpr'] = (float(np.mean(s[1:] != s[:-1])) if len(s) >= 2
                      else np.nan)
        if len(s) >= 2:
            p_pos = float(np.mean(s > 0))
            p_neg = 1.0 - p_pos
            ps = np.array([p_pos, p_neg])
            ps = ps[ps > 0]
            out['drr_sign_entropy'] = float(-(ps * np.log2(ps)).sum())
    # trend slope (1/beat, normalized by mean RR)
    if n >= 3 and mean_rr > EPS:
        slope = float(np.polyfit(np.arange(n, dtype=float), rr, 1)[0])
        out['trend_slope_norm'] = slope / mean_rr
    # entropy family (m=2) and shape descriptors
    if n >= 4:
        out['sampen_m2_r02'] = sampen(rr, m=2, r=0.2 * sdnn) if sdnn > 0 else np.nan
        out['sampen_m2_r025'] = sampen(rr, m=2, r=0.25 * sdnn) if sdnn > 0 else np.nan
        out['fuzzyen_m1'] = fuzzyen(rr, m=1, r=0.2 * sdnn) if sdnn > 0 else np.nan
        out['fuzzyen_m2'] = fuzzyen(rr, m=2, r=0.2 * sdnn) if sdnn > 0 else np.nan
        out['disten_m2'] = disten(rr, m=2, n_bins=10)
        out['katz_fd'] = katz_fd(rr)
    out['higuchi_fd'] = higuchi_fd(rr, kmax=4)
    out['dfa_alpha1'] = dfa_alpha1(rr)
    if len(drr) >= 2:
        madrr = float(np.mean(np.abs(drr)))
        maddr = float(np.mean(np.abs(np.diff(drr))))
        out['smoothness'] = maddr / madrr if madrr > EPS else np.nan
    return out


def feature_matrix(dicts, keys=None):
    keys = keys or ALL_FEATURES
    return {k: np.array([d.get(k, np.nan) for d in dicts], float)
            for k in keys}


def meta_matrix(dicts):
    return {k: np.array([d.get(k, np.nan) for d in dicts], float)
            for k in ('n_rr', 'mean_rr', 'median_rr', 'hr')}


if __name__ == '__main__':
    rng = np.random.default_rng(0)
    # smooth sinus-arrhythmia-like oscillation
    t = np.arange(20)
    sinus = 0.8 + 0.10 * np.sin(2 * np.pi * t / 6.0) + rng.normal(0, 0.005, 20)
    af = 0.8 + rng.normal(0, 0.15, 20)
    for name, rr in [('sinus', sinus), ('af', af)]:
        f = features_from_rr_ext(rr)
        print(name, {k: (round(v, 4) if isinstance(v, float)
                         and np.isfinite(v) else v)
                     for k, v in f.items()})
