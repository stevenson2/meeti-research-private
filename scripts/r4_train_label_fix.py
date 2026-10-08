#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 4 (autonomous loop): teacher-corrected weak labels for Head_R.

Mechanism (different from failed KD losses B/C): the ECG-FM teacher signal is
used to identify likely-mislabeled MEETI train records (confident-learning
style), then those records are PRUNED or DOWNWEIGHTED in Head_R training.
Model architecture / protocol identical to locked v3 (fc1=64, lam 0.2, +PTB).

Teacher scores: PLS16 on the 768-d ECG-FM record embeddings, fitted with
5-fold GroupKFold BY SUBJECT so every MEETI record gets an OUT-OF-SAMPLE
score (no transductive leakage). Only TRAIN-side records are modified;
the MEETI val subjects and their labels stay frozen (evaluation unchanged).

Arms:
  prune      : drop bottom --prune-frac self-confidence train records (w=0)
  downweight : multiply those records' weight by 0.1
  none       : baseline reproduction (sanity, no flags applied)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.cross_decomposition import PLSRegression
from tensorflow.keras import Model, layers

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, check_disk_redline, load_hard_beats,
    record_windows, load_ptb_train_records)

T_BEATS = 10
MEETI_DIR = Path('/home/devcontainers/meeti/data/processed_small')


def load_meeti_with_teacher():
    """Replicate load_meeti_records split (seed 42) but keep record identity
    and teacher OOS scores aligned per record."""
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)
    emb = np.load(MEETI_DIR / 'teacher_emb_768.npy')

    def to_y(cls):
        return 1 if cls == 'abnormal' else (0 if cls == 'normal' else -1)

    manifest['_y'] = manifest['report_class'].apply(to_y)
    # teacher embeddings are 2 five-second segments per record, in manifest
    # order (process_mimic_raw.py stacks [seg0, seg1] per record)
    assert len(emb) == 2 * len(manifest), \
        f'teacher emb {len(emb)} != 2 x manifest {len(manifest)}'
    emb = emb.reshape(len(manifest), 2, 768).mean(axis=1)   # per-record
    lab = manifest[manifest['_y'] >= 0].reset_index(drop=True)
    n_beats = lab['n_beats'].astype(int).to_numpy()
    assert int(n_beats.sum()) == len(beats)
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]])
    y = lab['_y'].to_numpy().astype(np.int32)
    subj = lab['subject_id'].to_numpy()

    # out-of-sample PLS16 teacher scores (GroupKFold by subject)
    emb_lab = emb[lab.index.to_numpy()]
    oos = np.zeros(len(lab), dtype=np.float32)
    gkf = GroupKFold(n_splits=5)
    for tr, te in gkf.split(emb_lab, y, groups=subj):
        pls = PLSRegression(n_components=16)
        pls.fit(emb_lab[tr], y[tr])
        oos[te] = pls.predict(emb_lab[te]).ravel()
    val_auc = float(roc_auc_score(y, oos))
    print(f'teacher OOS PLS16 record AUC (all labeled): {val_auc:.4f}',
          flush=True)

    # identical subject split as load_meeti_records (random_state=42)
    uniq, subj_idx = np.unique(subj, return_inverse=True)
    subj_y = np.zeros(len(uniq), dtype=np.int32)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], yi)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_subj, va_subj = next(sss.split(np.arange(len(uniq)), subj_y))
    is_val_record = np.isin(subj_idx, va_subj)
    print(f'records train={int((~is_val_record).sum())} '
          f'val={int(is_val_record.sum())} '
          f'(load_meeti_records expects 1912-ish windows incl PTB)',
          flush=True)
    val_mask_auc = roc_auc_score(y[is_val_record], oos[is_val_record]) \
        if is_val_record.any() else float('nan')
    print(f'teacher OOS AUC on val subjects: {val_mask_auc:.4f}', flush=True)

    return lab, positions, n_beats, y, oos, is_val_record, val_auc, \
        float(val_mask_auc)


def build_full_model(lr, lam):
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
    encoder = Model(base.input, base.get_layer('fc1').output,
                    name='beat_encoder')
    beat_in = layers.Input((250, 1), name='beat_input')
    head_b = layers.Dense(2, activation='softmax', name='head_b')(
        encoder(beat_in))
    rec_in = layers.Input((T_BEATS, 250, 1), name='record_input')
    rec_mask_in = layers.Input((T_BEATS, 1), name='record_mask')
    rec_emb = layers.TimeDistributed(encoder, name='td_encoder')(rec_in)
    rec_proj = layers.TimeDistributed(
        layers.Dense(32, activation='relu', name='head_r_proj'),
        name='td_head_r_proj')(rec_emb)
    rec_pool = MaskedAttentionPooling(name='head_r_attention')(
        [rec_proj, rec_mask_in])
    head_r = layers.Dense(2, activation='softmax', name='head_r')(rec_pool)
    model = Model([beat_in, rec_in, rec_mask_in], [head_b, head_r],
                  name='dual_head_v3_r4')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
              'head_r': tf.keras.losses.SparseCategoricalCrossentropy()},
        loss_weights={'head_b': 1.0, 'head_r': lam})
    return (model,
            Model(model.inputs[0], model.outputs[0]),
            Model([model.inputs[1], model.inputs[2]], model.outputs[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['prune', 'downweight', 'none'],
                    default='prune')
    ap.add_argument('--prune-frac', type=float, default=0.15)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.2)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    suffix = '' if args.seed == 42 else f'_seed{args.seed}'
    run_dir = Path(f'/home/devcontainers/meeti/runs/'
                   f'r4_labelfix_{args.arm}_pf{args.prune_frac}{suffix}')
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    check_disk_redline()

    (lab, positions, n_beats, y, oos, is_val, teacher_auc_all,
     teacher_auc_val) = load_meeti_with_teacher()

    # self-confidence = OOS score assigned to the GIVEN weak label
    self_conf = np.where(y == 1, oos, -oos)          # higher = more confident
    is_train = ~is_val
    thr = np.quantile(self_conf[is_train], args.prune_frac)
    flagged = is_train & (self_conf <= thr)
    if args.arm == 'none':
        flagged[:] = False
    print(f'arm={args.arm} flagged={int(flagged.sum())} '
          f'of train={int(is_train.sum())} '
          f'(label1 flagged={int((flagged & (y == 1)).sum())}, '
          f'label0 flagged={int((flagged & (y == 0)).sum())})', flush=True)

    # assemble record windows + weights (replicating load_meeti_records)
    x_all, m_all, y_all, _ = record_windows(
        np.load(MEETI_DIR / 'beats.npy').astype(np.float32),
        positions, n_beats, y, np.zeros(len(lab), np.float32))
    w_all = np.full(len(lab), 0.2, dtype=np.float32)   # meeti_weight
    if args.arm == 'prune':
        w_all[flagged] = 0.0
    elif args.arm == 'downweight':
        w_all[flagged] *= 0.1
    keep = is_train  # train-side records only
    rec_x, rec_mask = x_all[keep], m_all[keep]
    rec_y, rec_w = y_all[keep], w_all[keep]
    rec_xv, rec_mv, rec_yv = x_all[is_val], m_all[is_val], y_all[is_val]

    ptb = load_ptb_train_records()
    rec_x = np.concatenate([rec_x, ptb['x']])
    rec_mask = np.concatenate([rec_mask, ptb['mask']])
    rec_y = np.concatenate([rec_y, ptb['y']])
    rec_w = np.concatenate([rec_w, ptb['w']])

    xb, yb, xv, yv, x_real = load_hard_beats()
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    n_hard, n_rec = len(xb), len(rec_x)
    rec_steps = math.ceil(n_hard / 128)
    rec_bs = math.ceil(n_rec / rec_steps)
    hard_total, rec_total = rec_steps * 128, rec_steps * rec_bs

    def pad_indices(perm, target):
        reps = int(math.ceil(target / len(perm)))
        return np.tile(perm, reps)[:target]

    model, beat_probe, rec_probe = build_full_model(args.lr, args.lam)
    hist = {'beat': [], 'rec': []}
    best = {'balanced': -1.0}
    for epoch in range(args.epochs):
        b_idx = pad_indices(rng.permutation(n_hard), hard_total)
        r_idx = pad_indices(rng.permutation(n_rec), rec_total)
        wb = np.where(yb[b_idx] == 0, 2.0, 1.0).astype(np.float32)
        wr = rec_w[r_idx].copy()
        wb[-(hard_total - n_hard):] = 0.0
        wr[-(rec_total - n_rec):] = 0.0
        for step in range(rec_steps):
            model.train_on_batch(
                [xb[b_idx[step * 128:(step + 1) * 128]],
                 rec_x[r_idx[step * rec_bs:(step + 1) * rec_bs]],
                 rec_mask[r_idx[step * rec_bs:(step + 1) * rec_bs]]],
                [yb[b_idx[step * 128:(step + 1) * 128]],
                 rec_y[r_idx[step * rec_bs:(step + 1) * rec_bs]]],
                sample_weight=[
                    wb[step * 128:(step + 1) * 128],
                    wr[step * rec_bs:(step + 1) * rec_bs]])
        pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
        pr = rec_probe.predict([rec_xv, rec_mv], batch_size=256,
                               verbose=0)[:, 1]
        b_auc = float(roc_auc_score(yv, pb))
        r_auc = float(roc_auc_score(rec_yv, pr))
        hist['beat'].append(b_auc)
        hist['rec'].append(r_auc)
        if min(b_auc, r_auc) > best['balanced']:
            best['balanced'] = min(b_auc, r_auc)
            best['epoch'] = epoch + 1
            best['beat'] = b_auc
            best['rec'] = r_auc
            beat_probe.save(str(run_dir / 'ckpt_balanced__beat.keras'))
            rec_probe.save(str(run_dir / 'ckpt_balanced__record.keras'))
        print(f'ep{epoch + 1} beat={b_auc:.4f} rec={r_auc:.4f}', flush=True)

    result = {
        'arm': args.arm, 'prune_frac': args.prune_frac,
        'flagged': int(flagged.sum()),
        'flagged_label_dist': [int((flagged & (y == 0)).sum()),
                               int((flagged & (y == 1)).sum())],
        'teacher_oos_auc_all': teacher_auc_all,
        'teacher_oos_auc_meeti_val': teacher_auc_val,
        'best': best,
        'history': hist,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result | {'history': '...'}, indent=1), flush=True)


if __name__ == '__main__':
    main()
