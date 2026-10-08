#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8: PTB record-level holdout eval for the 5 s single-lead student.

Only evaluates, no training.  Windows use the FROZEN R8 5-beat construction
(see r8_data_5s.py docstring): per record consecutive 5-beat windows, stride
5, last-beat-repeat padding; record score = mean of window probabilities;
record label = max beat label over the record.  Patient-level split from
data/split_guard.compute_ptb_split (seed 42, untouched).

Output caliber mirrors scripts/eval_record_head_ptb_holdout.py
(splits -> n_records/label_dist/auc/mean_p/frac_gt_0.5 + preds npz).
NOTE the disclosed caliber change vs that script: windows are 1250-pt 5 s
single-lead II segments instead of first-10-beat 250-pt windows.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import ptb_5s, record_scores  # noqa: E402
from data.split_guard import compute_ptb_split  # noqa: E402


def eval_split(probe, split_mask, batch_size=256):
    d = ptb_5s(split_mask)
    p_win = probe.predict(d['x'], batch_size=batch_size, verbose=0)[:, 1]
    p_rec = record_scores(p_win, d['win_rec'], d['n_records'])
    auc = (float(roc_auc_score(d['rec_y'], p_rec))
           if len(np.unique(d['rec_y'])) > 1 else None)
    return {
        'n_records': int(d['n_records']),
        'label_dist': np.bincount(d['rec_y'], minlength=2).tolist(),
        'auc': auc,
        'mean_p': float(p_rec.mean()),
        'frac_gt_0.5': float((p_rec > 0.5).mean()),
    }, p_rec, d['rec_y']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True,
                    help='5s student ckpt (e.g. ckpt_best__record5s.keras)')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/'
                            'r8_eval_ptb_holdout_5s')
    ap.add_argument('--tag', default='r8_student_5s')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    split = compute_ptb_split()
    print('PTB split stats',
          {k: v for k, v in split['stats'].items()
           if isinstance(v, (int, float, str, bool))}, flush=True)

    probe = tf.keras.models.load_model(args.model)
    print('loaded model', args.model, flush=True)

    results = {'model': args.model, 'tag': args.tag, 'splits': {}}
    preds = {}
    for name, mask in [('train', split['train']),
                       ('val', split['val']),
                       ('test', split['test'])]:
        res, p_rec, y_rec = eval_split(probe, np.asarray(mask, dtype=bool))
        results['splits'][name] = res
        preds[f'{name}_p'] = p_rec
        preds[f'{name}_y'] = y_rec
        print(name, res, flush=True)

    with open(out / f'{args.tag}.json', 'w') as f:
        json.dump(results, f, indent=2)
    np.savez_compressed(out / f'{args.tag}_preds.npz', **preds)
    print('saved ->', out, flush=True)


if __name__ == '__main__':
    main()
