#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""评估一对 beat_probe / record_probe 模型。

指标:
  - Head_B: 公共 val AUC, 真实 AFE holdout mean p / frac>0.5
  - Head_R: MEETI subject-holdout 记录 AUC, PTB train/val/test 记录级金标准 AUC
输出 JSON 到项目外 runs/ 目录。
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, load_hard_beats, load_meeti_records,
)

T = 10


def ptb_record_data(mask):
    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats, dtype=np.float32)
    labels = np.asarray(labels)
    rids = np.asarray(rids)
    mask = np.asarray(mask, dtype=bool)
    beats, labels, rids = beats[mask], labels[mask], rids[mask]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
    n = len(uniq)
    x = np.zeros((n, T, beats.shape[1]), dtype=np.float32)
    xm = np.zeros((n, T, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s, e = first[i], first[i] + counts[i]
        take = min(int(counts[i]), T)
        x[i, :take] = beats[s:s + take]
        xm[i, :take] = 1.0
        y[i] = int(labels[s:e].max())
    return x, xm, y


def predict_rec(probe, x, m, batch_size=256):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(probe.predict([x[i:i + batch_size], m[i:i + batch_size]],
                                 batch_size=batch_size, verbose=0))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beat-probe', required=True)
    ap.add_argument('--record-probe', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/eval_probe_pairs')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    beat_probe = tf.keras.models.load_model(args.beat_probe)
    rec_probe = tf.keras.models.load_model(args.record_probe)
    print('loaded', args.beat_probe, args.record_probe, flush=True)

    _, _, xv, yv, x_real = load_hard_beats()
    pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]

    meeti = load_meeti_records(
        argparse.Namespace(meeti_weight=0.2))
    pm = predict_rec(rec_probe, meeti['x_val'], meeti['mask_val'])[:, 1]

    split = compute_ptb_split()
    ptb = {}
    for name, mask in [('train', split['train']),
                       ('val', split['val']),
                       ('test', split['test'])]:
        x, xm, y = ptb_record_data(mask)
        p = predict_rec(rec_probe, x, xm)[:, 1]
        ptb[name] = {
            'n_records': int(len(x)),
            'label_dist': np.bincount(y, minlength=2).tolist(),
            'auc': float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None,
            'mean_p': float(p.mean()),
            'frac_gt_0.5': float((p > 0.5).mean()),
        }

    result = {
        'tag': args.tag,
        'beat_probe': args.beat_probe,
        'record_probe': args.record_probe,
        'head_b': {
            'public_val_auc': float(roc_auc_score(yv, pb)),
            'real_mean': float(pr_real.mean()),
            'real_frac_gt_0.5': float((pr_real > 0.5).mean()),
        },
        'head_r': {
            'meeti_holdout_auc': float(roc_auc_score(meeti['y_val'], pm)),
            'meeti_holdout_n': int(len(pm)),
            'ptb': ptb,
        },
    }
    with open(out / f'{args.tag}.json', 'w') as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        out / f'{args.tag}_preds.npz',
        val_beat_p=pb, val_beat_y=yv, real_p=pr_real,
        meeti_val_p=pm, meeti_val_y=meeti['y_val'],
        ptb_train_p=predict_rec(rec_probe, *ptb_record_data(split['train'])[:2]),
        ptb_val_p=predict_rec(rec_probe, *ptb_record_data(split['val'])[:2]),
        ptb_test_p=predict_rec(rec_probe, *ptb_record_data(split['test'])[:2]),
    )
    print(json.dumps(result, indent=2), flush=True)
    print('saved ->', out / f'{args.tag}.json', flush=True)


if __name__ == '__main__':
    main()
