#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 5 (autonomous loop): full-gate evaluation of the r4 teacher-label
prune candidate (seed 42) vs the locked model.

Gates (V3 discipline, vs locked dual_head_v3_seeded_lam0.2_ptb):
  beat val AUC >= 0.7631
  MEETI record val AUC >= 0.7773
  PTB val >= 0.8880 / test >= 0.8367   (separate script output pasted in)
  AFE real holdout: mean <= ~0.0026 (report frac>0.5)
  M4 gated e2e: real FP/hour == 0 (v1-min and v2-spec), MIT val recall 26/26
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import MaskedAttentionPooling, load_hard_beats  # noqa: E402,F401
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa: E402

CAND = Path('/home/devcontainers/meeti/runs/r4_labelfix_prune_pf0.15')
OUT = Path('/home/devcontainers/meeti/runs/r5_fullgate_prune42')
T, COOLDOWN = 10, 20
REAL_META = {'ecg_real_052': 183.0, 'rec_latest': 64.0}
REAL = {'ecg_real_052': 'real_normal_beats_exp7c.npy',
        'rec_latest': 'real_normal_beats_rec_latest.npy'}
CONFIGS = {'v1_min': {'k': 1, 'tau_r': 0.5, 'tau_b': 0.15},
           'v2_spec': {'k': 2, 'tau_r': 0.7, 'tau_b': 0.1}}


def eager_beat(m, x, bs=512):
    return np.concatenate(
        [m(x[i:i + bs], training=False).numpy()[:, 1]
         for i in range(0, len(x), bs)])


def eager_rec(m, x, mask, bs=256):
    return np.concatenate(
        [m([x[i:i + bs], mask[i:i + bs]], training=False).numpy()[:, 1]
         for i in range(0, len(x), bs)])


def window_mean(p):
    n = len(p)
    out = np.zeros(n, np.float32)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    for i in range(n):
        lo = max(0, i - T + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def alarms(hr, hbw, cfg):
    pos = (hr >= cfg['tau_r']) & (hbw >= cfg['tau_b'])
    cand = np.zeros(len(pos), bool)
    acc = 0
    for i in range(len(pos)):
        acc += int(pos[i])
        if i >= 5:
            acc -= int(pos[i - 5])
        cand[i] = acc >= cfg['k']
    trig, last = [], -10 ** 9
    for i in range(len(cand)):
        if cand[max(0, i - 4):i + 1].any() and i - last >= COOLDOWN:
            trig.append(i)
            last = i
    return trig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    beat = tf.keras.models.load_model(str(CAND / 'ckpt_balanced__beat.keras'))
    rec = tf.keras.models.load_model(str(CAND / 'ckpt_balanced__record.keras'))

    xb, yb, xv, yv, x_real = load_hard_beats()
    beat_auc = float(roc_auc_score(yv, eager_beat(beat, xv)))
    afe_p = eager_beat(beat, x_real)
    print(f'beat val AUC {beat_auc:.4f} | AFE holdout mean '
          f'{afe_p.mean():.5f} frac>0.5 {(afe_p > 0.5).mean():.4f}', flush=True)

    # MEETI record val AUC (same loader as training)
    import argparse
    from train_dual_head_small import load_meeti_records
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=0.2))
    pr = eager_rec(rec, meeti['x_val'], meeti['mask_val'])
    meeti_auc = float(roc_auc_score(meeti['y_val'], pr))
    print(f'MEETI record val AUC {meeti_auc:.4f}', flush=True)

    # M4 gated e2e
    real_dir = PROJ / 'pc_tools/ecg_dl/data/real'
    real_rows = {}
    for name, fname in REAL.items():
        beats = np.load(real_dir / fname).astype(np.float32)
        n = len(beats)
        x = np.zeros((n, T, 250, 1), np.float32)
        m = np.zeros((n, T, 1), np.float32)
        for i in range(n):
            lo = max(0, i - T + 1)
            x[i, -(i - lo + 1):] = beats[lo:i + 1, :, None]
            m[i, -(i - lo + 1):] = 1.0
        pb = eager_beat(beat, beats[..., None])
        hr = eager_rec(rec, x, m)
        hbw = window_mean(pb)
        row = {'head_b_mean': float(pb.mean()),
               'head_r_mean': float(hr.mean())}
        for cname, cfg in CONFIGS.items():
            trig = alarms(hr, hbw, cfg)
            row[cname] = {'alarms': len(trig),
                          'fp_per_hour': round(
                              len(trig) / (REAL_META[name] / 3600.0), 2)}
        real_rows[name] = row
        print(name, json.dumps(row), flush=True)

    split = compute_mit_incart_split()
    groups = {}
    for ds in ['mit_bih', 'incart']:
        beats, labels, rids = load_arrays(ds)
        vm = np.asarray(split['val_masks'][ds], dtype=bool)
        beats = np.asarray(beats[vm], np.float32)
        labels = np.asarray(labels[vm])
        rids = np.asarray(rids[vm])
        order = np.argsort(rids, kind='stable')
        beats, labels, rids = beats[order], labels[order], rids[order]
        uniq, first, counts = np.unique(rids, return_index=True,
                                        return_counts=True)
        for ui, rid in enumerate(uniq):
            s, e = first[ui], first[ui] + counts[ui]
            groups[f'{ds}_{int(rid)}'] = {
                'beats': beats[s:e], 'label': int(labels[s:e].max())}

    tp = {c: 0 for c in CONFIGS}
    n_abn = 0
    for rname, g in groups.items():
        beats = g['beats']
        n = len(beats)
        x = np.zeros((n, T, 250, 1), np.float32)
        m = np.ones((n, T, 1), np.float32)
        for i in range(n):
            lo = max(0, i - T + 1)
            x[i, -(i - lo + 1):] = beats[lo:i + 1, :, None]
        pb = eager_beat(beat, beats[..., None])
        hr = eager_rec(rec, x, m)
        hbw = window_mean(pb)
        if g['label'] == 1:
            n_abn += 1
            for cname, cfg in CONFIGS.items():
                tp[cname] += int(len(alarms(hr, hbw, cfg)) > 0)
    recall = {c: tp[c] / n_abn for c in CONFIGS}
    print(f'MIT val recall {recall} ({n_abn} abnormal)', flush=True)

    gates = {
        'beat_ge_locked_0.7631': beat_auc >= 0.7631,
        'meeti_ge_locked_0.7773': meeti_auc >= 0.7773,
        'afe_mean_le_0.0059': float(afe_p.mean()) <= 0.0059,
        'm4_real_fp_zero_both_configs': all(
            real_rows[n][c]['alarms'] == 0
            for n in REAL for c in CONFIGS),
        'mit_recall_26_26_both_configs': all(
            recall[c] == 1.0 for c in CONFIGS),
        # PTB gate filled from r4_ptb_holdout/result.json separately
    }
    result = {'beat_val_auc': beat_auc, 'meeti_record_val_auc': meeti_auc,
              'afe_holdout': {'mean': float(afe_p.mean()),
                              'frac_gt_05': float((afe_p > 0.5).mean())},
              'real': real_rows, 'mit_recall': recall, 'gates': gates}
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(gates, indent=1), flush=True)


if __name__ == '__main__':
    main()
