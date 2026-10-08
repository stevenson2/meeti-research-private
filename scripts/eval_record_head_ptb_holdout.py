#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 PTB 记录级金标准验证已训练的 Head_R (record_probe.keras)。

只评估, 不训练。PTB 记录窗口与 M1 同口径: 每条记录前 T=10 拍, 不足右补零,
记录标签 = max(拍标签)。患者级划分由 data/split_guard.compute_ptb_split 提供,
train/val/test 均计算并输出 AUC。
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
from train_dual_head_small import MaskedAttentionPooling  # noqa: E402,F401

T = 10


def build_record_windows(beats, rids):
    order = np.argsort(rids, kind='stable')
    beats, rids = beats[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
    x = np.zeros((len(uniq), T, beats.shape[1]), dtype=np.float32)
    mask = np.zeros((len(uniq), T, 1), dtype=np.float32)
    for i in range(len(uniq)):
        s, e = first[i], first[i] + counts[i]
        take = min(int(counts[i]), T)
        x[i, :take] = beats[s:s + take]
        mask[i, :take] = 1.0
    return x, mask


def predict(probe, x, mask):
    out = []
    for i in range(0, len(x), 256):
        out.append(probe.predict([x[i:i + 256], mask[i:i + 256]],
                                 batch_size=256, verbose=0))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe',
                    default='/home/devcontainers/meeti/runs/dual_head_small/'
                            'record_probe.keras')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/'
                            'eval_record_head_ptb_holdout')
    ap.add_argument('--tag', default='m1_record_head')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    gpus = tf.config.list_physical_devices('GPU')
    print('gpus', gpus, flush=True)
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)

    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats, dtype=np.float32)
    labels = np.asarray(labels)
    rids = np.asarray(rids)
    split = compute_ptb_split()
    print('PTB beats', len(beats), 'split stats',
          {k: v for k, v in split['stats'].items()
           if isinstance(v, (int, float, str, bool))}, flush=True)

    probe = tf.keras.models.load_model(args.probe)
    print('loaded probe', args.probe, flush=True)

    results = {'probe': args.probe, 'tag': args.tag, 'splits': {}}
    all_pred = {}
    for name, mask in [('train', split['train']),
                       ('val', split['val']),
                       ('test', split['test'])]:
        mask = np.asarray(mask, dtype=bool)
        x, xm = build_record_windows(beats[mask], rids[mask])
        y = []
        # 重新按排序后的 record 顺序取标签: build_record_windows 内部按 rid 排序
        order = np.argsort(rids[mask], kind='stable')
        ls = labels[mask][order]
        r_sorted = rids[mask][order]
        uniq = np.unique(r_sorted)
        # 用 searchsorted 聚合 max
        first = np.searchsorted(r_sorted, uniq, side='left')
        n = len(uniq)
        for i in range(n):
            s = first[i]
            e = first[i + 1] if i + 1 < n else len(r_sorted)
            y.append(int(ls[s:e].max()))
        y = np.asarray(y, dtype=np.int32)
        p = predict(probe, x, xm)[:, 1]
        auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
        results['splits'][name] = {
            'n_records': int(len(x)), 'label_dist': np.bincount(y, minlength=2).tolist(),
            'auc': auc, 'mean_p': float(p.mean()),
            'frac_gt_0.5': float((p > 0.5).mean()),
        }
        all_pred[name] = {'p': p, 'y': y}
        print(name, results['splits'][name], flush=True)

    with open(out / f'{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out / f'{args.tag}_preds.npz',
                        train_p=all_pred['train']['p'],
                        train_y=all_pred['train']['y'],
                        val_p=all_pred['val']['p'], val_y=all_pred['val']['y'],
                        test_p=all_pred['test']['p'],
                        test_y=all_pred['test']['y'])
    print('saved ->', out, flush=True)


if __name__ == '__main__':
    main()
