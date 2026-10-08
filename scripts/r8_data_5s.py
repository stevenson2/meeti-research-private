#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8 data construction: single-lead 5 s windows for the record student.

FROZEN CALIBER (R8 launch prompt, 2026-09-05) - do not change without
re-freezing the caliber in a new prompt:

  MIMIC  : data/processed_small/m3_beat_centered/beat_windows.npy
           (6456,12,2500 @500Hz) -> channel II (index 1, r7 precedent) ->
           scipy.signal.decimate(q=2) -> (6456,1250) @250Hz.
           Window label = record report_class (normal=0/abnormal=1).
           Subject split = teacher_supervised_subspace.py protocol
           (subject StratifiedShuffleSplit test=0.2 seed=42, stratified by
           max record label per subject); ONLY train-subject windows are
           used for training, sample weight 0.2.
  MEETI  : data/processed_small/beats.npy (250 pt/beat, 250 Hz) -> per record
           consecutive 5-beat windows, stride 5 (non-overlapping); records
           with n%5 != 0 pad the LAST window by repeating the last beat
           (record count unchanged, window count = ceil(n/5), every record
           contributes >=1 window). Window label = record weak label,
           weight 0.2. Subject split = load_meeti_records (seed 42).
           Record-level score = mean of that record's window probabilities.
  PTB    : data.split_guard.load_arrays('ptb') + compute_ptb_split().
           Train patients: 5-beat windows per record (same stride-5 /
           last-beat-repeat rule, extended from MEETI), window label =
           max beat label within the window, weight 1.0.
           Val/test patients: windows built per record with the same rule,
           record score = window mean, AUC vs record label
           (= max beat label over the whole record).
  Device : pc_tools/ecg_dl/data/real/real_normal_beats_*.npy (already
           250 Hz) -> sliding 5-beat windows, stride 1.

Teacher probabilities (task B only): for the MIMIC train-subject windows,
teacher_local_768.npy (full-12) -> 5-fold GroupKFold BY SUBJECT -> per fold
PLS16(scale=False, max_iter=2000) + LogisticRegression(max_iter=5000) on the
16-d projection -> out-of-fold p_teacher in [0,1].  OOS protocol identical
to r4_train_label_fix.py; PLS16+LR head identical to
teacher_supervised_subspace.py / r7_lead2_teacher_probe.py.
"""
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')

ROOT = Path('/home/devcontainers/meeti')
M3 = ROOT / 'data/processed_small/m3_beat_centered'
MEETI_DIR = ROOT / 'data/processed_small'
REAL_DIR = PROJ / 'pc_tools/ecg_dl/data/real'
TEACHER_CACHE = ROOT / 'runs/r8b_w5_teacher_probs'

W_POINTS = 1250          # 5 s @ 250 Hz
BEATS_PER_WIN = 5
REAL_FILES = {'ecg_real_052': 'real_normal_beats_exp7c.npy',
              'rec_latest': 'real_normal_beats_rec_latest.npy'}


def mimic_5s(chunk=512):
    """MIMIC beat-centered 5 s windows, lead II, decimated to 250 Hz."""
    meta = pd.read_csv(M3 / 'beat_meta.csv')
    mask = np.load(M3 / 'student_beat_mask.npy').astype(bool)
    meta = meta[mask].reset_index(drop=True)
    y = meta['report_class'].map({'normal': 0, 'abnormal': 1})
    assert y.notna().all(), 'unexpected report_class values in beat_meta'
    y = y.to_numpy(np.int32)
    rids = meta['study_id'].to_numpy(np.int64)
    subjects = meta['subject_id'].to_numpy()

    wins = np.load(M3 / 'beat_windows.npy', mmap_mode='r')[mask]
    x = np.zeros((len(meta), W_POINTS), dtype=np.float32)
    for s in range(0, len(meta), chunk):
        e = min(s + chunk, len(meta))
        xi = np.asarray(wins[s:e, 1, :], dtype=np.float32)   # lead II
        x[s:e] = signal.decimate(xi, q=2, axis=1).astype(np.float32)

    # subject split, verbatim teacher_supervised_subspace.py
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), dtype=int)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], int(yi))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    tr = np.isin(subj_idx, tr_s)
    va = np.isin(subj_idx, va_s)
    return {'x': x, 'y': y, 'rids': rids, 'subjects': subjects,
            'tr': tr, 'va': va, 'n_subjects': int(len(uniq))}


def meeti_5s():
    """MEETI records -> 5-beat windows (stride 5, last-beat repeat pad)."""
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)

    def to_y(cls):
        return 1 if cls == 'abnormal' else (0 if cls == 'normal' else -1)

    manifest['_y'] = manifest['report_class'].apply(to_y)
    manifest = manifest[manifest['_y'] >= 0].reset_index(drop=True)
    n_beats = manifest['n_beats'].astype(int).to_numpy()
    assert int(n_beats.sum()) == len(beats)
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]]).astype(np.int64)
    y = manifest['_y'].to_numpy(np.int32)
    subjects = manifest['subject_id'].to_numpy()
    n_rec = len(manifest)

    xs, win_rec = [], []
    for i in range(n_rec):
        nb = int(n_beats[i])
        n_win = int(math.ceil(nb / BEATS_PER_WIN))
        flat = np.minimum(np.arange(n_win * BEATS_PER_WIN), nb - 1)
        xs.append(beats[flat + positions[i]].reshape(n_win, W_POINTS))
        win_rec.append(np.full(n_win, i, np.int64))
    x = np.concatenate(xs, axis=0)
    win_rec = np.concatenate(win_rec, axis=0)

    # subject split, verbatim load_meeti_records (seed 42)
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), np.int32)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], yi)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    rec_tr = np.isin(subj_idx, tr_s)
    rec_va = np.isin(subj_idx, va_s)
    return {'x': x, 'win_rec': win_rec, 'rec_y': y,
            'rec_tr': rec_tr, 'rec_va': rec_va,
            'n_records': n_rec, 'n_subjects': int(len(uniq)),
            'rec_subjects': subjects}


def ptb_5s(split_mask):
    """PTB windows for one patient-level split mask (beat-aligned)."""
    from data.split_guard import load_arrays  # noqa: E402

    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats[np.asarray(split_mask)], np.float32)
    labels = np.asarray(labels)[np.asarray(split_mask)]
    rids = np.asarray(rids)[np.asarray(split_mask)]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True,
                                    return_counts=True)
    n_rec = len(uniq)
    rec_y = np.zeros(n_rec, np.int32)
    xs, win_rec, win_y = [], [], []
    for i in range(n_rec):
        s = int(first[i])
        nb = int(counts[i])
        e = s + nb
        rec_y[i] = int(labels[s:e].max())
        n_win = int(math.ceil(nb / BEATS_PER_WIN))
        pos = np.minimum(np.arange(n_win * BEATS_PER_WIN), nb - 1)
        xs.append(beats[pos + s].reshape(n_win, W_POINTS))
        win_rec.append(np.full(n_win, i, np.int64))
        lab = np.asarray(labels[s:e])[pos].reshape(n_win, BEATS_PER_WIN)
        win_y.append(lab.max(axis=1))
    x = np.concatenate(xs, axis=0)
    return {'x': x, 'win_rec': np.concatenate(win_rec, axis=0),
            'win_y': np.concatenate(win_y, axis=0).astype(np.int32),
            'rec_y': rec_y, 'rids': uniq, 'n_records': n_rec}


def device_5s():
    """Real-device normal recordings -> sliding 5-beat windows, stride 1."""
    out = {}
    for name, fname in REAL_FILES.items():
        b = np.load(REAL_DIR / fname).astype(np.float32)     # (n,250)
        n_win = len(b) - BEATS_PER_WIN + 1
        idx = (np.arange(n_win)[:, None]
               + np.arange(BEATS_PER_WIN)[None, :])
        out[name] = {'x': b[idx].reshape(n_win, W_POINTS),
                     'file': fname, 'n_beats': int(len(b))}
    return out


def record_scores(p_win, win_rec, n_rec):
    """Record-level score = mean of the record's window probabilities."""
    cnt = np.bincount(win_rec, minlength=n_rec).astype(np.float64)
    tot = np.bincount(win_rec, weights=p_win.astype(np.float64),
                      minlength=n_rec)
    return tot / np.maximum(cnt, 1)


def teacher_oos_probs(mim):
    """Out-of-fold teacher probability for the MIMIC train-subject windows.

    5-fold GroupKFold by subject; per fold PLS16(scale=False, max_iter=2000)
    on full-12 teacher_local_768.npy -> LogisticRegression on the 16-d
    projection -> predict_proba.  Cached under runs/r8b_w5_teacher_probs/.
    """
    TEACHER_CACHE.mkdir(parents=True, exist_ok=True)
    cache = TEACHER_CACHE / 'p_teacher_mimic_tr.npy'
    if cache.exists():
        p = np.load(cache)
        with open(TEACHER_CACHE / 'meta.json') as f:
            meta = json.load(f)
        auc = float(meta.get('oos_auc_window', meta.get('oos_auc', 0.0)))
        return p, auc

    teacher = np.load(M3 / 'teacher_local_768.npy', mmap_mode='r')
    emb = np.asarray(teacher[mim['tr']], np.float64)
    y = mim['y'][mim['tr']]
    subj = mim['subjects'][mim['tr']]
    p = np.zeros(len(y), np.float32)
    for ftr, fte in GroupKFold(n_splits=5).split(emb, y, groups=subj):
        scaler = StandardScaler().fit(emb[ftr])
        e_tr, e_te = scaler.transform(emb[ftr]), scaler.transform(emb[fte])
        pls = PLSRegression(n_components=16, scale=False, max_iter=2000)
        pls.fit(e_tr, y[ftr])
        lr = LogisticRegression(max_iter=5000).fit(pls.transform(e_tr),
                                                   y[ftr])
        p[fte] = lr.predict_proba(pls.transform(e_te))[:, 1]
    auc = float(roc_auc_score(y, p))
    np.save(cache, p)
    with open(TEACHER_CACHE / 'meta.json', 'w') as f:
        json.dump({'oos_auc_window': auc,
                   'n_windows': int(len(y)),
                   'protocol': 'GroupKFold(5, by subject) StandardScaler -> '
                               'PLS16(scale=False, max_iter=2000) -> '
                               'LR(max_iter=5000), full-12 '
                               'teacher_local_768 (OOS = r4, head = '
                               'teacher_supervised_subspace/r7)'},
                  f, indent=2)
    return p, auc
