#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A2/A3/A5: AF-specialized record head on PTB-XL single-lead lead II.

Same declared family as R13 (raw mV lead II, (1000,1), folds1-9 train /
fold10 test).  Three tasks share one recipe (E=20, b=64, lr=3e-4, balanced
inverse-frequency sample weights, best-val-AUC checkpoint):

  af     : y_af (AFIB present) vs rest           -- R14-A2
  nonaf  : non-AF records only; non-NORM vs NORM -- R14-A3 control
  genbin : abnormal (non-NORM) vs NORM, all      -- R14-A5 paired comparator
           (R13 2cls anchor re-run under the paired protocol)

Selection discipline: folds1-9 are split 5% patient-level holdout for epoch
selection / theta_tr only.  fold-10 is never used for any selection.

Outputs: runs/r14_af_head/<task>_seed<seed>/{result.json,probs.npz,
best.weights.h5,train.log}
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from models_patch.ptbxl_builders import build_ptbxl_large  # noqa: E402
from ecg_bench import spec_at_recall, youden_threshold  # noqa: E402

DATA = Path('/home/devcontainers/meeti/runs/r14_af_prep')
OUT_ROOT = Path('/home/devcontainers/meeti/runs/r14_af_head')
TASKS = ('af', 'nonaf', 'genbin')


def load_task(task):
    tr = np.load(DATA / 'ptbxl_af_train.npz')
    te = np.load(DATA / 'ptbxl_af_test.npz')
    x_tr = tr['x'].astype(np.float32)[..., None]
    x_te = te['x'].astype(np.float32)[..., None]
    y_af_tr = tr['y_af'].astype(np.int8)
    y_af_te = te['y_af'].astype(np.int8)
    norm_tr = (tr['y5'][:, 3] > 0.5)
    norm_te = (te['y5'][:, 3] > 0.5)
    pat_tr = tr['patient_id'].astype(np.int64)
    pat_te = te['patient_id'].astype(np.int64)

    if task == 'af':
        m_tr = np.ones(len(x_tr), bool)
        m_te = np.ones(len(x_te), bool)
        y_tr = y_af_tr
        y_te = y_af_te
    elif task == 'nonaf':
        m_tr = y_af_tr == 0
        m_te = y_af_te == 0
        y_tr = (~norm_tr).astype(np.int8)
        y_te = (~norm_te).astype(np.int8)
    elif task == 'genbin':
        m_tr = np.ones(len(x_tr), bool)
        m_te = np.ones(len(x_te), bool)
        y_tr = (~norm_tr).astype(np.int8)
        y_te = (~norm_te).astype(np.int8)
    else:
        raise ValueError(task)

    return dict(
        task=task,
        x_tr=x_tr[m_tr], y_tr=y_tr[m_tr], pat_tr=pat_tr[m_tr],
        ecg_tr=tr['ecg_ids'][m_tr],
        x_te=x_te[m_te], y_te=y_te[m_te], pat_te=pat_te[m_te],
        ecg_te=te['ecg_ids'][m_te],
        n_all_tr=int(len(x_tr)), n_all_te=int(len(x_te)),
        all_zero5_tr=int((tr['y5'].sum(axis=1) == 0).sum()),
        all_zero5_te=int((te['y5'].sum(axis=1) == 0).sum()),
    )


def make_holdout(pat, seed, frac=0.05, y=None, min_each=5):
    """Patient-level holdout of ~frac records; force >=min_each per class."""
    upat = np.unique(pat)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(upat)
    pcount = {p: int((pat == p).sum()) for p in upat}
    target = frac * len(pat)
    val_pat, n_val = [], 0
    for p in perm:
        val_pat.append(p)
        n_val += pcount[p]
        if n_val >= target:
            break
    val_mask = np.isin(pat, val_pat)
    if y is not None:
        for p in perm[len(val_pat):]:
            pos = int(((y == 1) & val_mask).sum())
            neg = int(((y == 0) & val_mask).sum())
            if pos >= min_each and neg >= min_each:
                break
            val_pat.append(p)
            val_mask = np.isin(pat, val_pat)
    return val_mask


def class_weights(y):
    n = len(y)
    n_pos = max(int(y.sum()), 1)
    n_neg = max(n - n_pos, 1)
    return {0: n / (2.0 * n_neg), 1: n / (2.0 * n_pos)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=TASKS)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--holdout-frac', type=float, default=0.05)
    ap.add_argument('--out-suffix', default='')
    args = ap.parse_args()

    out = OUT_ROOT / f'{args.task}_seed{args.seed}{args.out_suffix}'
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / 'train.log', 'w', buffering=1)

    def logp(*a):
        print(*a, flush=True)
        print(*a, file=log, flush=True)

    t0 = time.time()
    d = load_task(args.task)
    y_tr, pat_tr = d['y_tr'], d['pat_tr']
    val_mask = make_holdout(pat_tr, args.seed, args.holdout_frac, y_tr)
    tr_mask = ~val_mask
    logp(f'[{args.task} seed{args.seed}] pool={len(y_tr)} '
         f'train={int(tr_mask.sum())} val={int(val_mask.sum())} '
         f'pos_rate train/val/test='
         f'{y_tr[tr_mask].mean():.4f}/{y_tr[val_mask].mean():.4f}/'
         f'{d["y_te"].mean():.4f}')

    x_trn, y_trn = d['x_tr'][tr_mask], y_tr[tr_mask].astype(np.float32)
    x_val, y_val = d['x_tr'][val_mask], y_tr[val_mask].astype(np.float32)
    cw = class_weights(y_trn)
    w_trn = np.where(y_trn > 0.5, cw[1], cw[0]).astype(np.float32)
    logp(f'class_weights {cw} mean_w={w_trn.mean():.4f}')

    tf.keras.utils.set_random_seed(args.seed)
    model = build_ptbxl_large(input_shape=(1000, 1), n_classes=1,
                              output_activation='sigmoid',
                              name=f'r14_{args.task}')
    opt = tf.keras.optimizers.Adam(args.lr)
    loss_fn = tf.keras.losses.BinaryCrossentropy(reduction='none')
    rng = np.random.default_rng(args.seed)

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb, wb):
        with tf.GradientTape() as tape:
            p = model(xb, training=True)
            loss = tf.reduce_mean(wb * loss_fn(yb, p))
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    n = len(x_trn)
    steps = math.ceil(n / args.batch_size)
    hist, val_aucs = [], []
    best_auc, best_ep = -1.0, -1
    for ep in range(args.epochs):
        perm = rng.permutation(n)
        losses = []
        for s in range(steps):
            idx = perm[s * args.batch_size:(s + 1) * args.batch_size]
            losses.append(float(train_step(x_trn[idx], y_trn[idx, None],
                                           w_trn[idx])))
        p_val = model.predict(x_val, batch_size=256, verbose=0)[:, 0]
        va = float(roc_auc_score(y_val, p_val))
        hist.append(float(np.mean(losses)))
        val_aucs.append(round(va, 4))
        if va > best_auc:
            best_auc, best_ep = va, ep
            model.save_weights(str(out / 'best.weights.h5'))
        logp(f'[{args.task} seed{args.seed}] ep{ep+1}/{args.epochs} '
             f'loss={hist[-1]:.5f} val_auc={va:.4f} '
             f'elapsed={time.time()-t0:.0f}s')

    model.load_weights(str(out / 'best.weights.h5'))
    p_val = model.predict(x_val, batch_size=256, verbose=0)[:, 0]
    p_te = model.predict(d['x_te'], batch_size=256, verbose=0)[:, 0]
    y_te = d['y_te']
    theta_tr = float(youden_threshold(y_val, p_val))
    spec90 = spec_at_recall(y_te, p_te)
    yp = p_te >= theta_tr
    tp = int((yp & (y_te == 1)).sum())
    fn = int(((~yp) & (y_te == 1)).sum())
    fp = int((yp & (y_te == 0)).sum())
    tn = int(((~yp) & (y_te == 0)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    yp5 = p_te >= 0.5
    tp5 = int((yp5 & (y_te == 1)).sum())
    fp5 = int((yp5 & (y_te == 0)).sum())
    tn5 = int(((~yp5) & (y_te == 0)).sum())
    rec5 = tp5 / max(tp5 + int(((~yp5) & (y_te == 1)).sum()), 1)
    spec5 = tn5 / max(tn5 + fp5, 1)

    metrics = {
        'test_auc': round(float(roc_auc_score(y_te, p_te)), 4),
        'test_ap': round(float(average_precision_score(y_te, p_te)), 4),
        'spec_at_R90_test': spec90,
        'theta_tr_val_youden': round(theta_tr, 4),
        'R_at_theta_tr': round(rec, 4),
        'spec_at_theta_tr': round(spec, 4),
        'R_at_0.5': round(rec5, 4),
        'spec_at_0.5': round(spec5, 4),
        'bacc_at_0.5': round((rec5 + spec5) / 2, 4),
        'prob_mean_test': round(float(p_te.mean()), 4),
        'val_auc_best': round(best_auc, 4),
    }
    result = {
        'arm': f'r14_af_head_{args.task}',
        'task': args.task,
        'seed': args.seed,
        'recipe': 'E20 b64 lr3e-4 balanced sample weights, best val-AUC ckpt',
        'family_declaration': 'ptbxl_lead2_raw_10s_100hz_v1 (R13 family)',
        'n_pool_train': int(len(y_tr)),
        'n_train': int(tr_mask.sum()),
        'n_val': int(val_mask.sum()),
        'n_test': int(len(y_te)),
        'pos_rate': {'train': round(float(y_trn.mean()), 4),
                     'val': round(float(y_val.mean()), 4),
                     'test': round(float(y_te.mean()), 4)},
        'class_weights': {str(k): round(float(v), 4) for k, v in cw.items()},
        'holdout': {
            'protocol': f'patient-level ~5% of task pool, seed={args.seed}',
            'val_ecg_ids': d['ecg_tr'][val_mask][:20].tolist(),
            'n_val_patients': int(len(np.unique(pat_tr[val_mask]))),
            'best_epoch': int(best_ep + 1),
        },
        'loss_history': [round(v, 5) for v in hist],
        'val_auc_history': val_aucs,
        'metrics': metrics,
        'n_all_train_pool': d['n_all_tr'],
        'n_all_test_pool': d['n_all_te'],
        'all_zero5_labels': {'train': d['all_zero5_tr'],
                             'test': d['all_zero5_te']},
        'elapsed_s': round(time.time() - t0, 1),
    }
    if args.task == 'af':
        result['verdict'] = {'smoke_go_line_auc_ge_0.90':
                             bool(metrics['test_auc'] >= 0.90)}
    (out / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    np.savez_compressed(
        out / 'probs.npz',
        p_val=p_val.astype(np.float32), y_val=y_val.astype(np.int8),
        p_test=p_te.astype(np.float32), y_test=y_te.astype(np.int8),
        ecg_test=d['ecg_te'], ecg_val=d['ecg_tr'][val_mask],
        patient_test=d['pat_te'])
    logp('RESULT ' + json.dumps(result, ensure_ascii=False))
    log.close()


if __name__ == '__main__':
    main()
