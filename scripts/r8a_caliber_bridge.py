#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8a caliber-bridge measurement (report-only, zero training).

The R8a gates compare the 5 s student against locked-model anchors measured
on a different window caliber.  The locked record head structurally cannot
consume 1250-pt windows (its input is (T=10, 250, 1)), so the bridge is done
in ITS native beat units:

  PTB val/test:
    caliber_10beat_locked : first-10-beat windows (the anchor caliber;
                            sanity-reproduces 0.8880/0.8367 with THIS eval
                            pipeline), and
    caliber_5beat_locked  : first-5-beat context, beats 6-10 zero-padded +
                            masked (the same 5-beat context length the 5 s
                            student sees, in locked units).
  Device:
    locked head on stride-1 sliding 10-beat windows (full context) and
    stride-1 sliding 5-beat windows (5-beat context, zero-padded) of the two
    real all-normal recordings -> window-level frac>0.5 comparable to the
    student's 5-beat device frac.

Locked ckpt is loaded READ-ONLY; nothing is saved back to its directory.
"""
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import device_5s, REAL_DIR, BEATS_PER_WIN  # noqa: E402
from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
from train_dual_head_small import MaskedAttentionPooling  # noqa: E402,F401

LOCKED = ('/home/devcontainers/meeti/runs/'
          'dual_head_v3_seeded_lam0.2_ptb/ckpt_balanced__record.keras')
OUT = Path('/home/devcontainers/meeti/runs/r8a_caliber_bridge')
T_BEATS = 10


def ptb_record_windows(split_mask, n_context):
    """First-n_context beats per record, zero-padded to T_BEATS + mask."""
    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats[np.asarray(split_mask)], np.float32)
    labels = np.asarray(labels)[np.asarray(split_mask)]
    rids = np.asarray(rids)[np.asarray(split_mask)]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True,
                                    return_counts=True)
    n = len(uniq)
    x = np.zeros((n, T_BEATS, 250), np.float32)
    m = np.zeros((n, T_BEATS, 1), np.float32)
    y = np.zeros(n, np.int32)
    for i in range(n):
        s, e = int(first[i]), int(first[i]) + int(counts[i])
        y[i] = int(labels[s:e].max())
        take = min(int(counts[i]), n_context)
        x[i, :take] = beats[s:s + take]
        m[i, :take] = 1.0
    return x, m, y


def eval_ptb(probe, split_mask, n_context):
    x, m, y = ptb_record_windows(split_mask, n_context)
    p = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None
    return {'n_records': int(len(y)),
            'label_dist': np.bincount(y, minlength=2).tolist(),
            'auc': auc, 'mean_p': float(p.mean()),
            'frac_gt_0.5': float((p > 0.5).mean())}


def sliding_beat_windows(beats, n_context):
    """(n, n_context, 250) stride-1 sliding windows from a beat array."""
    n_win = len(beats) - n_context + 1
    idx = np.arange(n_win)[:, None] + np.arange(n_context)[None, :]
    return beats[idx]


def eval_device(probe):
    out = {}
    for name, d in device_5s().items():
        b = np.load(REAL_DIR / d['file']).astype(np.float32)  # (n,250)
        entry = {'n_beats': int(len(b))}
        for n_context in (10, 5):
            wins = sliding_beat_windows(b, n_context)
            x = np.zeros((len(wins), T_BEATS, 250), np.float32)
            m = np.zeros((len(wins), T_BEATS, 1), np.float32)
            x[:, :n_context] = wins
            m[:, :n_context] = 1.0
            p = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
            entry[f'ctx{n_context}'] = {
                'n_windows': int(len(wins)), 'p_mean': float(p.mean()),
                'frac_gt_0.5': float((p > 0.5).mean())}
        out[name] = entry
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    probe = tf.keras.models.load_model(LOCKED)
    print('locked record head loaded read-only:', LOCKED, flush=True)
    split = compute_ptb_split()
    res = {'locked_ckpt': LOCKED, 'ptb': {}, 'device': eval_device(probe)}
    for name in ['val', 'test']:
        mask = np.asarray(split[name], bool)
        res['ptb'][name] = {
            'caliber_10beat_locked': eval_ptb(probe, mask, 10),
            'caliber_5beat_locked': eval_ptb(probe, mask, 5)}
        print(name, json.dumps(res['ptb'][name]), flush=True)
    print('device', json.dumps(res['device']), flush=True)
    with open(OUT / 'result.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
