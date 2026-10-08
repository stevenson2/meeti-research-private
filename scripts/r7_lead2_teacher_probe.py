#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 7: Lead-II-only teacher probe (decisive test for the KD route).

Question: does the ECG-FM teacher's weak-label advantage survive when it can
only see lead II (the only lead the deployment device records)?

Variants (same window set, same protocol as teacher_supervised_subspace.py):
  full12  : existing teacher_local_768.npy (control anchor, PLS16 AUC 0.8935)
  lead2   : lead II replicated into all 12 input channels
  lead1   : lead I replicated (control - separates 'lead II carries the
            signal' from 'any single-lead fill degrades the FM equally')

Extraction preprocessing replicated verbatim from extract_ecg_fm_teacher.py
(global z-score over the (12,2500) tensor; NOTE: replicating one channel 12x
gives the same global z-score as z-scoring that channel alone, so the fill
variant is unambiguous). Sanity check: re-extract 500 full12 windows and
correlate with the stored embeddings (expect ~1.0).

Eval: subject split (StratifiedShuffleSplit seed 42), StandardScaler(train),
PLS16 + LogisticRegression, record-level AUC on val subjects; full-768 LR
AUC; participation ratio (effective rank); PCA-top64 AUC.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/devcontainers/meeti')
M3 = ROOT / 'data/processed_small/m3_beat_centered'
OUT = ROOT / 'runs/r7_lead2_teacher_probe'
CKPT = '/mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt'
BATCH = 16


def extract(model, windows, indices, desc):
    n = len(indices)
    emb = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(n, s + BATCH)
        xs = []
        for j in range(s, e):
            x = np.asarray(windows[indices[j]], dtype=np.float32)  # (12,2500)
            mu, sd = x.mean(), x.std() + 1e-6
            xs.append(((x - mu) / sd)[None])
        xb = torch.from_numpy(np.concatenate(xs)).float()
        with torch.no_grad():
            out = model(source=xb)
        emb[s:e] = out['features'].mean(dim=1).numpy().astype(np.float32)
        if (e % 1000) < BATCH or e == n:
            print(f'{desc} {e}/{n} elapsed={time.time()-t0:.0f}s', flush=True)
    return emb


def record_auc(p, y_beat, rids):
    ids = np.unique(rids)
    ymap = {int(r): int(y) for r, y in zip(rids, y_beat)}
    rp = np.array([p[rids == i].mean() for i in ids])
    ry = np.array([ymap[int(i)] for i in ids])
    return float(roc_auc_score(ry, rp))


def participation_ratio(emb, tr):
    cov = np.cov(emb[tr].T)
    lam = np.linalg.eigvalsh(cov)[::-1]
    lam = np.clip(lam, 0, None)
    return float((lam.sum() ** 2) / (np.square(lam).sum() + 1e-12))


def eval_variant(name, emb, y, rids, tr, va):
    scaler = StandardScaler().fit(emb[tr])
    e_tr, e_va = scaler.transform(emb[tr]), scaler.transform(emb[va])
    pls = PLSRegression(n_components=16, scale=False, max_iter=2000)
    pls.fit(e_tr, y[tr])
    lr = LogisticRegression(max_iter=5000).fit(pls.transform(e_tr), y[tr])
    auc16 = record_auc(lr.predict_proba(pls.transform(e_va))[:, 1],
                       y[va], rids[va])
    lr_full = LogisticRegression(max_iter=5000).fit(e_tr, y[tr])
    auc768 = record_auc(lr_full.predict_proba(e_va)[:, 1], y[va], rids[va])
    pca = PCA(n_components=64, random_state=0).fit(e_tr)
    lr_pca = LogisticRegression(max_iter=5000).fit(
        pca.transform(e_tr), y[tr])
    auc64 = record_auc(lr_pca.predict_proba(pca.transform(e_va))[:, 1],
                       y[va], rids[va])
    row = {'pls16_auc': round(auc16, 4), 'full768_auc': round(auc768, 4),
           'pca64_auc': round(auc64, 4),
           'participation_ratio': round(participation_ratio(emb, tr), 1)}
    print(name, row, flush=True)
    return row


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(M3 / 'beat_meta.csv')
    mask = np.load(M3 / 'student_beat_mask.npy').astype(bool)
    meta = meta[mask].reset_index(drop=True)
    windows = np.load(M3 / 'beat_windows.npy', mmap_mode='r')[mask]
    rids = meta['study_id'].to_numpy(np.int64)
    subjects = meta['subject_id'].to_numpy()
    y = meta['report_class'].map({'normal': 0, 'abnormal': 1}).to_numpy()
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), dtype=int)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], int(yi))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    tr = np.isin(subj_idx, tr_s)
    va = np.isin(subj_idx, va_s)
    print(f'n={len(meta)} train={int(tr.sum())} val={int(va.sum())}',
          flush=True)

    from fairseq_signals.models import build_model_from_checkpoint
    model = build_model_from_checkpoint(checkpoint_path=CKPT)
    model.eval()

    # sanity: 500-window full12 re-extraction vs stored embeddings
    teacher_full = np.load(M3 / 'teacher_local_768.npy')[mask]
    sanity_idx = np.arange(500)
    re = extract(model, windows, sanity_idx, 'sanity_full12')
    corr = float(np.corrcoef(re.ravel(), teacher_full[sanity_idx].ravel())[0, 1])
    print(f'sanity full12 corr={corr:.4f}', flush=True)

    # lead II (index 1) and lead I (index 0) fills
    lead2 = np.asarray(windows[:, 1, :], dtype=np.float32)   # (N,2500)
    lead1 = np.asarray(windows[:, 0, :], dtype=np.float32)
    idx_all = np.arange(len(meta))
    emb_l2 = extract(model, lead2[:, None, :].repeat(12, axis=1), idx_all,
                     'lead2_fill')
    np.save(OUT / 'emb_lead2_fill.npy', emb_l2)
    emb_l1 = extract(model, lead1[:, None, :].repeat(12, axis=1), idx_all,
                     'lead1_fill')
    np.save(OUT / 'emb_lead1_fill.npy', emb_l1)

    results = {'sanity_full12_corr': corr}
    results['full12'] = eval_variant('full12', teacher_full.astype(np.float64),
                                     y, rids, tr, va)
    results['lead2_fill'] = eval_variant('lead2_fill', emb_l2.astype(np.float64),
                                         y, rids, tr, va)
    results['lead1_fill'] = eval_variant('lead1_fill', emb_l1.astype(np.float64),
                                         y, rids, tr, va)

    # decision
    l2, f12 = results['lead2_fill'], results['full12']
    results['decision'] = {
        'lead2_keeps_advantage': bool(l2['pls16_auc'] >= 0.85),
        'lead2_rank_fits_fc1': bool(l2['participation_ratio'] < 64),
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2), flush=True)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
