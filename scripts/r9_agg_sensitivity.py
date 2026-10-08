#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9 eval-side probe: record-score aggregation sensitivity (zero training).

Frozen gate caliber = record score = MEAN of window probabilities.  This
probe quantifies, on the r9_form-ptbbal checkpoints, how much the PTB val
gap to the 0.878 gate depends on that aggregation choice: median / max /
top-k-mean / majority-margin variants are computed from the SAME window
probabilities (no re-training, no gate verdict change - analysis only).
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import ptb_5s  # noqa: E402
from r9_data import load_all  # noqa: E402
from data.split_guard import compute_ptb_split  # noqa: E402

OUT = Path('/home/devcontainers/meeti/runs/r9_agg_sensitivity')
SEEDS = [42, 43, 44]


def aggregations(p_win, win_rec, n_rec):
    """Record scores under alternative aggregations of window probs."""
    order = np.argsort(win_rec, kind='stable')
    p, r = p_win[order], win_rec[order]
    uniq, first, counts = np.unique(r, return_index=True,
                                    return_counts=True)
    out = {'mean': np.zeros(n_rec), 'median': np.zeros(n_rec),
           'max': np.zeros(n_rec), 'top2mean': np.zeros(n_rec)}
    for i, (s, c) in enumerate(zip(first, counts)):
        v = np.sort(p[s:s + c])[::-1]
        out['mean'][i] = v.mean()
        out['median'][i] = np.median(v)
        out['max'][i] = v[0]
        out['top2mean'][i] = v[:min(2, len(v))].mean()
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    data = load_all(norm=True, form=True)
    split = compute_ptb_split()
    results = {}
    for seed in SEEDS:
        ck = (f'/home/devcontainers/meeti/runs/'
              f'r9_form-ptbbal_seed{seed}/ckpt_best__record5s.keras')
        probe = tf.keras.models.load_model(ck)
        results[seed] = {}
        for name in ['val', 'test']:
            d = ptb_5s(split[name])
            p_win = probe.predict(d['x'], batch_size=256, verbose=0)[:, 1]
            aggs = aggregations(p_win, d['win_rec'], d['n_records'])
            results[seed][name] = {
                agg: float(roc_auc_score(d['rec_y'], scores))
                for agg, scores in aggs.items()}
        print(seed, results[seed], flush=True)
    mean_by_agg = {
        f'{split_name}_{agg}': float(np.mean(
            [results[s][split_name][agg] for s in SEEDS]))
        for split_name in ['val', 'test']
        for agg in ['mean', 'median', 'max', 'top2mean']}
    results['mean_3seed'] = mean_by_agg
    results['frozen_gates'] = {'ptb_val': 0.878, 'ptb_test': 0.8267}
    results['note'] = ('analysis only - frozen gate caliber is mean; '
                       'other aggregations are headroom estimates. '
                       'top-k/max are label-consistent: record label = '
                       'max beat label, so the most-positive window '
                       'carries the aligned statistic.')
    with open(OUT / 'result.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(json.dumps(mean_by_agg, indent=1), flush=True)


if __name__ == '__main__':
    main()
