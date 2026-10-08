#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10 lit row: ECG-FM linear probe on frozen embeddings (paper protocol).

Literature protocol (ECG-FM, local fulltext papers_fulltext/ECG-FM.txt):
"In the Linear experiment, we performed linear probing with learning rate
1e-5. For this frozen evaluation, pretrained model embeddings are extracted
and fed as inputs to a single linear layer to generate predictions."

PRIMARY head here = converged linear probe (sklearn LogisticRegression,
lbfgs max_iter=5000, deterministic) on the same frozen full-12 MIMIC
train-subject embeddings the PLS16 head used - the standard linear-probe
implementation.

DISCLOSED protocol_variant: the paper's LITERAL Adam(lr=1e-5) schedule
severely undertrains on our small head-fit set (5,165 windows; in-domain
OOS window AUC 0.56 after 200 epochs vs PLS16 0.8201 on identical
features) - its benchmark numbers are embedded below as constants and
reported in the row JSON for honesty.

Zero-inference row: eval-domain embeddings cached under
runs/benchmark_v1/teacher_emb_*.npz (lead-II-fill form, same as the
fm_teacher_lead2fill row).  Head fit ONCE on MIMIC train-subject windows.

Output: runs/benchmark_v1/lit_row_lit_ecgfm-linear_lead2fill_mimichead.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import mimic_5s  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/benchmark_v1'
M3 = ROOT / 'data/processed_small/m3_beat_centered'
COMPOSITE_W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}

# literal Adam(lr=1e-5, 200 epochs, batch 256, seed 42) variant - measured
# in the first run of this script (runs/r10_fm_linear/adam1e5_variant.json)
ADAM_VARIANT = {'indomain_oos_window_auc': 0.56,
                'meeti_val': 0.483, 'ptb_val': 0.4201, 'ptb_test': 0.466,
                'S': -9.16,
                'note': 'undertrained at the literal paper schedule on our '
                        '5.2k-window head-fit set; kept for disclosure'}

SOURCE = ('ECG-FM: McCradden et al. 2025 NEJM AI (bowang-lab/ECG-FM; local '
          'fulltext papers_fulltext/ECG-FM.txt); weights: '
          '/mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt '
          '(official PhysioNet release, same checkpoint as the fm_teacher '
          'row)')
ADAPTATION = ('lead-II fill replicated to 12 channels + per-window global '
              'z-score (ECG-FM convention; eval windows are per-beat '
              'z-scored by the deploy chain so the transform is ~identity '
              'there); 5-beat concat windows 1250 pt @250 Hz')
CAVEAT = ('head fitted on MIMIC train-subject windows (mimichead, no PTB/'
          'MEETI contact); encoder pretrained on MIMIC-IV-ECG + PTB-XL + '
          'SPH - pretraining COVERS our MIMIC-derived weak-label domain '
          'and has PTB-institution overlap with the PTB-DB gold domain '
          '(different database); lead-II fill is OOD for a 12-lead '
          'encoder (lower-bound deployable view, same as fm_teacher row)')


def top2mean_by_record(p_win, win_rec, n_rec):
    order = np.argsort(win_rec, kind='stable')
    p, r = p_win[order], win_rec[order]
    uniq, first, counts = np.unique(r, return_index=True, return_counts=True)
    out = np.zeros(n_rec)
    for i, (s, c) in enumerate(zip(first, counts)):
        v = np.sort(p[s:s + c])[::-1]
        out[i] = v[:min(2, len(v))].mean()
    return out


def record_mean(p_win, win_rec, n_rec):
    cnt = np.bincount(win_rec, minlength=n_rec).astype(np.float64)
    tot = np.bincount(win_rec, weights=p_win.astype(np.float64),
                      minlength=n_rec)
    return tot / np.maximum(cnt, 1)


def main():
    mim = mimic_5s()
    teacher = np.load(M3 / 'teacher_local_768.npy', mmap_mode='r')
    emb_tr = np.asarray(teacher[mim['tr']], np.float32)
    y_tr = mim['y'][mim['tr']]

    # in-domain OOS sanity (GroupKFold by subject, same protocol as the
    # PLS16 teacher probs cache: runs/r8b_w5_teacher_probs, OOS 0.8201)
    oos = np.zeros(len(y_tr), np.float32)
    for ftr, fte in GroupKFold(n_splits=5).split(
            emb_tr, y_tr, groups=mim['subjects'][mim['tr']]):
        lr_f = LogisticRegression(max_iter=5000).fit(emb_tr[ftr], y_tr[ftr])
        oos[fte] = lr_f.predict_proba(emb_tr[fte])[:, 1]
    oos_auc = float(roc_auc_score(y_tr, oos))
    print('linear probe (LR) in-domain OOS window AUC:', round(oos_auc, 4),
          '| PLS16 0.8201 | literal-Adam1e-5 0.56', flush=True)

    head = LogisticRegression(max_iter=5000).fit(emb_tr, y_tr)

    def win_p(emb):
        return head.predict_proba(np.asarray(emb, np.float64))[:, 1]

    row = {'family': 'lit_ecgfm-linear_lead2fill_mimichead',
           'source': SOURCE,
           'adaptation': ADAPTATION,
           'caveat': CAVEAT,
           'protocol': {'head': 'linear probe on frozen embeddings: '
                                'LogisticRegression(max_iter=5000, lbfgs), '
                                'deterministic (primary)',
                        'fit_data': 'MIMIC train-subject windows, full-12 '
                                    'teacher_local_768 (identical window '
                                    'set to the fm_teacher PLS16 head)',
                        'indomain_oos_window_auc': round(oos_auc, 4),
                        'record_score': 'top2mean (legacy mean also '
                                        'reported)'},
           'protocol_variant_adam1e5': ADAM_VARIANT}

    for name in ['meeti_val', 'ptb_val', 'ptb_test']:
        d = np.load(OUT / f'teacher_emb_{name}.npz')
        p = win_p(d['x_emb'])
        win_rec, rec_y = d['win_rec'], d['rec_y']
        n_rec = int(win_rec.max()) + 1
        row[name] = float(roc_auc_score(
            rec_y, top2mean_by_record(p, win_rec, n_rec)))
        row[f'{name}_legacy_mean'] = float(roc_auc_score(
            rec_y, record_mean(p, win_rec, n_rec)))
        print(name, round(row[name], 4), flush=True)

    d = np.load(OUT / 'teacher_emb_device.npz')
    p = win_p(d['x_emb'])
    row['device_5beat'] = {'n_windows': int(len(p)),
                           'p_mean': float(p.mean()),
                           'frac_gt_0.5': float((p > 0.5).mean())}

    w = COMPOSITE_W
    row['S'] = round(100 * (w['ptb_val'] * (row['ptb_val'] - 0.5) / 0.5
                            + w['ptb_test'] * (row['ptb_test'] - 0.5) / 0.5
                            + w['meeti_val'] * (row['meeti_val'] - 0.5)
                            / 0.5), 2)
    print('S =', row['S'], flush=True)

    with open(OUT / 'lit_row_lit_ecgfm-linear_lead2fill_mimichead.json',
              'w') as f:
        json.dump(row, f, indent=2)
    print('row saved.', flush=True)


if __name__ == '__main__':
    main()
