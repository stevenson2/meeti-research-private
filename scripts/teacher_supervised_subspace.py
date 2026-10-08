#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A: supervised subspace (PLS) of ECG-FM local features.

Use weak MIMIC labels only on train subjects to fit a PLS projection of the
768-d teacher features.  Report record-level AUC for k=16..128 on val subjects.
Also writes train-subject KD targets for use by protective distillation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/devcontainers/meeti')
M3 = ROOT / 'data/processed_small/m3_beat_centered'
OUT = ROOT / 'runs/local_kd_pls'
KS = [16, 32, 48, 64, 96, 128]


def record_auc(p, y_beat, rids):
    ids = np.unique(rids)
    ymap = {int(r): int(y) for r, y in zip(rids, y_beat)}
    rp = np.array([p[rids == i].mean() for i in ids])
    ry = np.array([ymap[int(i)] for i in ids])
    return float(roc_auc_score(ry, rp))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default=str(OUT))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(M3 / 'beat_meta.csv')
    mask = np.load(M3 / 'student_beat_mask.npy').astype(bool)
    meta = meta[mask].reset_index(drop=True)
    beats = np.load(M3 / 'student_beats_lead2.npy')[mask].astype(np.float32)
    teacher = np.load(M3 / 'teacher_local_768.npy')[mask].astype(np.float64)
    rids = meta['study_id'].to_numpy(np.int64)
    subjects = meta['subject_id'].to_numpy()
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    y = meta['report_class'].map({'normal': 0, 'abnormal': 1}).to_numpy()
    subj_y = np.zeros(len(uniq), dtype=int)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], int(yi))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    tr = np.isin(subj_idx, tr_s)
    va = np.isin(subj_idx, va_s)
    print(f'subjects={len(uniq)} train_beats={int(tr.sum())} '
          f'val_beats={int(va.sum())} train_records={len(np.unique(rids[tr]))} '
          f'val_records={len(np.unique(rids[va]))}', flush=True)

    scaler = StandardScaler().fit(teacher[tr])
    teacher_tr = scaler.transform(teacher[tr])
    teacher_va = scaler.transform(teacher[va])

    results = []
    selected = None
    for k in KS:
        pls = PLSRegression(n_components=k, scale=False, max_iter=2000)
        pls.fit(teacher_tr, y[tr])
        p_tr = pls.transform(teacher_tr)
        p_va = pls.transform(teacher_va)
        # Fit LR on PLS train features (not full teacher)
        lr = LogisticRegression(max_iter=5000).fit(p_tr, y[tr])
        p = lr.predict_proba(p_va)[:, 1]
        auc = record_auc(p, y[va], rids[va])
        results.append({'k': k, 'record_auc': round(auc, 4)})
        print('PLS', k, 'AUC', round(auc, 4), flush=True)
        if k <= 64 and (selected is None or auc > selected['record_auc']):
            selected = {'k': k, 'record_auc': auc}

    # full teacher reference for comparison
    lr_full = LogisticRegression(max_iter=5000).fit(teacher_tr, y[tr])
    p_full = lr_full.predict_proba(teacher_va)[:, 1]
    full_auc = record_auc(p_full, y[va], rids[va])

    if selected is None:
        selected = {'k': 64, 'record_auc': 0.0}
    print('selected <=64 PLS:', selected, 'full_teacher_auc', full_auc, flush=True)

    # write full PLS projections and train-subject KD files for selected k
    k = selected['k']
    pls = PLSRegression(n_components=k, scale=False, max_iter=2000)
    pls.fit(teacher_tr, y[tr])
    full_proj = pls.transform(scaler.transform(teacher)).astype(np.float32)
    np.save(out / f'teacher_pls{k}_full.npy', full_proj)
    np.save(out / f'teacher_pls{k}_train.npy', full_proj[tr])
    np.save(out / 'student_beats_lead2_train.npy', beats[tr])
    meta[['beat_idx', 'stem', 'study_id', 'subject_id', 'report_class']] \
        .loc[tr].to_csv(out / 'train_subject_meta.csv', index=False)
    np.save(out / 'train_mask.npy', tr)

    summary = {
        'k_auc': results,
        'full_teacher_auc': round(full_auc, 4),
        'selected_k': k,
        'selected_k_auc': round(float(selected['record_auc']), 4),
        'n_train_beats': int(tr.sum()),
        'n_val_beats': int(va.sum()),
        'files': {
            'teacher_pls_train': str(out / f'teacher_pls{k}_train.npy'),
            'student_beats_train': str(out / 'student_beats_lead2_train.npy'),
        },
    }
    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
