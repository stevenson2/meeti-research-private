#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9 data construction: normalization / form-alignment fixes for the 5s
single-lead record student (post-R8a diagnosis).

Evidence chain (runs/R9_LITERATURE_GAP_ANALYSIS.md + runs/r9_probe_scale_stats):
  P1  MIMIC 5s windows are RAW mV (per-window std median 0.119, p10-p90
      0.069-0.202) while MEETI/PTB/device beat-concatenation windows are
      per-beat z-scored by the deploy chain (per-window std exactly 1.0,
      mean 0).  R8a mixed an ~8x amplitude gap with heteroscedastic spread
      into one encoder.
  Literature convention (ECG-FM, SemiSegECG, PLITA, LiteHeart 2511.02851):
      resample + (bandpass) + PER-SIGNAL Z-SCORE.  Our own teacher probe
      fed the FM z-scored windows - the KD targets were computed on
      normalized inputs the student never saw.

Arms (data-level fixes; training recipe otherwise identical to frozen R8a):
  norm : per-window z-score of the MIMIC beat-centered windows (the only
         un-normalized domain).  Eval windows (MEETI/PTB/device) are already
         exactly unit-scale, so the transform is numerically identity there
         and no evaluation caliber changes.
  form : norm + rebuild the MIMIC training windows into the SAME 5-beat
         concatenation form used by MEETI/PTB/eval: for each record, take
         consecutive beats and extract +/-125 pt (250 Hz) around each
         beat's r_peak from that beat's own 5 s window, concatenate to
         (1250,); stride 5, last-beat repeat pad (identical rule to
         meeti_5s/ptb_5s).  Removes the train-vs-eval form mismatch; for
         normal RR the segments join near-seamlessly (each beat's window
         contains its neighbors' r_peaks).

Teacher-prob note: teacher_oos_probs() indexes the beat-centered MIMIC
windows; the 'form' arm changes the window set, so task-B reweighting on
top of 'form' must aggregate per-beat teacher probs per concatenated window
(mean over the 5 beats) - implemented here as form_teacher_probs().
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import M3, W_POINTS, BEATS_PER_WIN  # noqa: E402

HALF_BEAT = 125          # +/-125 pt @250 Hz around each r_peak


def zscore_windows(x):
    """Per-window z-score (literature convention, all-domain consistent)."""
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def mimic_form_aligned(zscore=True, chunk=512):
    """MIMIC -> 5-beat concatenation windows (form == MEETI/PTB/eval).

    Window label / subject split identical to r8_data_5s.mimic_5s
    (record report_class; teacher_supervised_subspace subject split seed 42).
    """
    meta = pd.read_csv(M3 / 'beat_meta.csv')
    mask = np.load(M3 / 'student_beat_mask.npy').astype(bool)
    meta = meta[mask].reset_index(drop=True)
    y = meta['report_class'].map({'normal': 0, 'abnormal': 1})
    assert y.notna().all()
    y = y.to_numpy(np.int32)
    rids = meta['study_id'].to_numpy(np.int64)
    subjects = meta['subject_id'].to_numpy()
    rpeaks = meta['r_peak_250'].to_numpy(np.int64)      # index in 1250-pt win

    wins = np.load(M3 / 'beat_windows.npy', mmap_mode='r')[mask]
    # decimate once to 1250-pt lead-II windows (same as r8 mimic_5s)
    x1250 = np.zeros((len(meta), W_POINTS), dtype=np.float32)
    for s in range(0, len(meta), chunk):
        e = min(s + chunk, len(meta))
        x1250[s:e] = signal.decimate(
            np.asarray(wins[s:e, 1, :], np.float32), q=2, axis=1)

    # group beat rows per record (beat_meta rows are in beat order)
    order = np.argsort(rids, kind='stable')
    r_sorted, y_sorted, subj_sorted = rids[order], y[order], subjects[order]
    rp_sorted, x_sorted = rpeaks[order], x1250[order]
    uniq, first, counts = np.unique(r_sorted, return_index=True,
                                    return_counts=True)
    xs, win_rec, win_y, win_beats = [], [], [], []
    for i in range(len(uniq)):
        s = int(first[i])
        nb = int(counts[i])
        n_win = int(math.ceil(nb / BEATS_PER_WIN))
        beat_sel = np.minimum(np.arange(n_win * BEATS_PER_WIN), nb - 1)
        rows = beat_sel + s
        # (n_win*5, 250): +-125 pt around each beat's own r_peak; clamp the
        # center so the segment is always the full 250 pt (r_peaks within
        # 125 pt of a window edge shift slightly off-center, still valid
        # lead-II signal from the same beat-centered window)
        centers = np.clip(rp_sorted[rows], HALF_BEAT, W_POINTS - HALF_BEAT)
        seg = np.stack([x_sorted[r][c - HALF_BEAT:c + HALF_BEAT]
                        for r, c in zip(rows, centers)])
        xs.append(seg.reshape(n_win, W_POINTS))
        win_rec.append(np.full(n_win, i, np.int64))
        win_y.append(y_sorted[rows].reshape(n_win, BEATS_PER_WIN).max(axis=1))
        win_beats.append(beat_sel.reshape(n_win, BEATS_PER_WIN))
    x = np.concatenate(xs, axis=0)
    win_rec = np.concatenate(win_rec, axis=0)
    win_y = np.concatenate(win_y, axis=0).astype(np.int32)
    win_beats = np.concatenate(win_beats, axis=0)
    rec_y = np.array([y_sorted[int(first[i])] for i in range(len(uniq))],
                     np.int32)
    rec_subjects = np.array([subj_sorted[int(first[i])]
                             for i in range(len(uniq))])

    # subject split, verbatim teacher_supervised_subspace.py (seed 42)
    from sklearn.model_selection import StratifiedShuffleSplit
    uniq_s, subj_idx = np.unique(rec_subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq_s), dtype=int)
    for yi, si in zip(rec_y, subj_idx):
        subj_y[si] = max(subj_y[si], int(yi))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq_s)), subj_y))
    rec_tr = np.isin(subj_idx, tr_s)
    rec_va = np.isin(subj_idx, va_s)

    if zscore:
        x = zscore_windows(x)
    return {'x': x, 'win_rec': win_rec, 'win_y': win_y, 'rec_y': rec_y,
            'rec_tr': rec_tr, 'rec_va': rec_va, 'win_beats': win_beats,
            'n_records': int(len(uniq))}


def form_teacher_probs(mim_form, mim_beat_centered):
    """Per-beat-window teacher probs aggregated to form-aligned windows.

    mim_form['win_beats'] rows index the beat-centered windows passed in
    mim_beat_centered (order preserved by construction), whose teacher OOS
    probs come from r8_data_5s.teacher_oos_probs (cached, scaler-corrected).
    """
    from r8_data_5s import teacher_oos_probs
    p_beat, auc = teacher_oos_probs(mim_beat_centered)   # (6456_tr_subset,)
    # teacher probs exist only for train-subject beat windows; map via
    # beat-global indices: rebuild the tr mask of beat rows
    tr = mim_beat_centered['tr']
    p_global = np.full(len(tr), np.nan, np.float32)
    p_global[tr] = p_beat
    # win_beats indexes ALL beat rows (0..6455); keep windows fully in train
    beats = mim_form['win_beats']
    p_win = p_global[beats].mean(axis=1)
    keep = ~np.isnan(p_win)
    return p_win, keep, float(auc)


_BUNDLE_CACHE = {}


def load_all(norm=True, form=False):
    """Full training/eval bundle with the requested MIMIC arm.

    Cached per (norm, form) at module level: benchmark and multi-seed
    loops call this repeatedly and the window rebuild (decimate +
    per-record regroup) dominates runtime otherwise.
    """
    from r8_data_5s import (meeti_5s, ptb_5s, device_5s)  # noqa: E402
    from data.split_guard import compute_ptb_split  # noqa: E402

    key = (norm, form)
    if key in _BUNDLE_CACHE:
        return _BUNDLE_CACHE[key]

    if form:
        mim = mimic_form_aligned(zscore=norm)
    else:
        from r8_data_5s import mimic_5s
        mim = mimic_5s()
        if norm:
            mim = dict(mim)
            mim['x'] = zscore_windows(mim['x'])
    me = meeti_5s()
    split = compute_ptb_split()
    bundle = {'mimic': mim, 'meeti': me,
              'ptb_splits': {k: np.asarray(split[k], bool)
                             for k in ['train', 'val', 'test']},
              'ptb_train': ptb_5s(split['train']),
              'device': device_5s()}
    _BUNDLE_CACHE[key] = bundle
    return bundle
