#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R11 on-board bench: build the replay bundle (deploy-chain beat windows).

Sets (all in deploy-chain per-beat z-scored 250-pt form, identical to the
unified benchmark eval inputs):
  device    : 2 real-AFE normal recordings (full) - FP must be 0
  mi_abn    : MIT+INCART leak-free val abnormal records (cap 150 beats)
              - recall >= 26/26 expectation from M4 e2e
  mi_nrm    : MIT+INCART val normal records (cap 150 beats) - FP count
  ptb_nrm   : PTB val normal records (cap 200 beats) - FP count
  ptb_abn   : PTB val abnormal records (cap 200 beats) - recall
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
import r8_data_5s  # noqa: F401,E402
from data.split_guard import load_arrays, compute_mit_incart_split, \
    compute_ptb_split  # noqa: E402
from r8_data_5s import REAL_FILES, REAL_DIR  # noqa: E402

CAP_MI = 150
CAP_PTB = 200
OUT = Path('/mnt/c/Users/cai/AppData/Local/Temp/r10read/bench_bundle.npz')


def record_view(beats, labels, rids, mask, cap):
    beats = np.asarray(beats, np.float32)[mask]
    labels = np.asarray(labels)[mask]
    rids = np.asarray(rids)[mask]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True,
                                    return_counts=True)
    recs, rec_labels, rec_ids = [], [], []
    for i in range(len(uniq)):
        s, nb = int(first[i]), int(counts[i])
        lab = int(labels[s:s + nb].max())
        take = min(nb, cap)
        recs.append(beats[s:s + take])
        rec_labels.append(lab)
        rec_ids.append(str(uniq[i]))
    return recs, np.array(rec_labels, np.int32), rec_ids


out = {}

# device (normal, full)
dev = []
for name, fname in REAL_FILES.items():
    b = np.load(REAL_DIR / fname).astype(np.float32)
    dev.append(b)
    print('device', name, b.shape)
out['device'] = np.array(dev, dtype=object)
out['device_names'] = np.array(list(REAL_FILES.keys()))

# MIT+INCART val
split = compute_mit_incart_split()
recs, ys, ids = [], [], []
for tag in ['mit_bih', 'incart']:
    b, l, r = load_arrays(tag)
    m = np.asarray(split['val_masks'][tag], bool)
    rr, yy, ii = record_view(b, l, r, m, CAP_MI)
    recs += rr
    ys.append(yy)
    ids += [tag + '_' + x for x in ii]
ys = np.concatenate(ys)
recs_abn = [x for x, y in zip(recs, ys) if y == 1]
recs_nrm = [x for x, y in zip(recs, ys) if y == 0]
out['mi_abn'] = np.array(recs_abn, dtype=object)
out['mi_nrm'] = np.array(recs_nrm, dtype=object)
out['mi_abn_ids'] = np.array([i for i, y in zip(ids, ys) if y == 1])
out['mi_nrm_ids'] = np.array([i for i, y in zip(ids, ys) if y == 0])
print('mi val abnormal records:', len(recs_abn), 'normal:', len(recs_nrm))

# PTB val
ptb = compute_ptb_split()
b, l, r = load_arrays('ptb')
mask = np.asarray(ptb['val'], bool)
recs, ys, ids = record_view(b, l, r, mask, CAP_PTB)
out['ptb_abn'] = np.array([x for x, y in zip(recs, ys) if y == 1],
                          dtype=object)
out['ptb_nrm'] = np.array([x for x, y in zip(recs, ys) if y == 0],
                          dtype=object)
out['ptb_abn_ids'] = np.array([i for i, y in zip(ids, ys) if y == 1])
out['ptb_nrm_ids'] = np.array([i for i, y in zip(ids, ys) if y == 0])
print('ptb val abnormal:', int(ys.sum()), 'normal:', int(len(ys) - ys.sum()))

np.savez_compressed(OUT, **out)
print('saved ->', OUT)
