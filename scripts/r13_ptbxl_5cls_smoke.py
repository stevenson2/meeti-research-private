#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R13 phase-0 smoke: PTB-XL single-lead 5-class vs binary NORM-vs-rest.

New declared family: raw mV lead-II 10 s @100 Hz, input (1000, 1),
trained on PTB-XL folds 1-9, evaluated on fold-10.

Smoke clause allows one seed (42), E=20, batch 64.  The result is a
feasibility/decision input for Route A/B, not a benchmark candidate.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Smoke runs are CPU-first unless the caller explicitly overrides.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from models_patch.ptbxl_builders import build_ptbxl_large  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DATA_DIR = ROOT / 'runs/r13_ptbxl_5cls_smoke'
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


def load_data():
    tr = np.load(DATA_DIR / 'ptbxl_lead2_train.npz')
    te = np.load(DATA_DIR / 'ptbxl_lead2_test.npz')
    x_tr = tr['x'].astype(np.float32)[..., None]
    y5_tr = tr['y5'].astype(np.float32)
    x_te = te['x'].astype(np.float32)[..., None]
    y5_te = te['y5'].astype(np.float32)
    # Binary NORM-vs-rest: positive = abnormal (not NORM).
    y2_tr = (y5_tr[:, 3] == 0).astype(np.int32)
    y2_te = (y5_te[:, 3] == 0).astype(np.int32)
    return x_tr, y5_tr, y2_tr, x_te, y5_te, y2_te


def train_model(model, loss_fn, x_tr, y_tr, epochs, batch_size, seed, tag):
    opt = tf.keras.optimizers.Adam(3e-4)
    rng = np.random.default_rng(seed)

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb):
        with tf.GradientTape() as tape:
            logits = model(xb, training=True)
            loss = loss_fn(yb, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss

    n = len(x_tr)
    n_steps = math.ceil(n / batch_size)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for step in range(n_steps):
            idx = perm[step * batch_size:(step + 1) * batch_size]
            losses.append(float(train_step(x_tr[idx], y_tr[idx])))
        avg = float(np.mean(losses))
        history.append(avg)
        print(f'[{tag}] ep{epoch + 1}/{epochs} loss={avg:.5f} '
              f'elapsed={time.time()-t0:.0f}s', flush=True)
    return history


def eval_5cls(model, x_te, y5_te, y2_te):
    p = model.predict(x_te, batch_size=256, verbose=0)
    class_auc = {CLASSES[i]: float(roc_auc_score(y5_te[:, i], p[:, i]))
                 for i in range(5)}
    macro = float(np.mean(list(class_auc.values())))
    norm_vs_rest = float(roc_auc_score(y5_te[:, 3], p[:, 3]))
    abnormal_vs_norm_5cls = float(roc_auc_score(y2_te, 1.0 - p[:, 3]))
    return {
        'class_auc': class_auc,
        'macro_auc': round(macro, 4),
        'norm_vs_rest_auc': round(norm_vs_rest, 4),
        'abnormal_vs_norm_5cls_auc': round(abnormal_vs_norm_5cls, 4),
        'raw_positive_scores_max': float(p.max()),
    }


def eval_2cls(model, x_te, y2_te):
    p = model.predict(x_te, batch_size=256, verbose=0)[:, 1]
    return {
        'binary_auc': round(float(roc_auc_score(y2_te, p)), 4),
        'raw_positive_mean': float(p.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tasks', default='both',
                    choices=['5cls', '2cls', 'both'])
    args = ap.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    x_tr, y5_tr, y2_tr, x_te, y5_te, y2_te = load_data()
    print('train', x_tr.shape, y5_tr.shape, 'test', x_te.shape,
          flush=True)
    print('binary abnormal prevalence train/test',
          float(y2_tr.mean()), float(y2_te.mean()), flush=True)

    result = {
        'arm': 'r13_ptbxl_5cls_smoke',
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'family_declaration': (
            'ptbxl_lead2_raw_10s_100hz_v1: single lead II, raw mV, '
            '(1000,1), PTB-XL folds1-9 train / fold10 test'),
        'n_train': int(len(x_tr)),
        'n_test': int(len(x_te)),
        'reference_lit_fold10_macro_auc': {
            'lit_resnet': 0.9189,
            'lit_lenet': 0.9146,
            'lit_xresnet': 0.9197,
        },
        'go_line': 'macro_auc>=0.85 AND binary_auc>=0.85',
    }

    if args.tasks in ('5cls', 'both'):
        model5 = build_ptbxl_large(
            input_shape=(1000, 1), n_classes=5,
            output_activation='sigmoid', name='ptbxl_5cls_smoke')
        loss5 = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        hist5 = train_model(model5, loss5, x_tr, y5_tr, args.epochs,
                            args.batch_size, args.seed, '5cls')
        model5.save(str(DATA_DIR / 'ckpt_smoke_5cls_seed42.keras'))
        ev5 = eval_5cls(model5, x_te, y5_te, y2_te)
        result['5cls'] = {
            'loss_history': [round(v, 5) for v in hist5],
            'final_loss': round(hist5[-1], 5),
            'metrics': ev5,
        }
        print('5cls metrics', json.dumps(ev5, indent=2), flush=True)

    if args.tasks in ('2cls', 'both'):
        model2 = build_ptbxl_large(
            input_shape=(1000, 1), n_classes=2,
            output_activation='softmax', name='ptbxl_2cls_smoke')
        loss2 = tf.keras.losses.SparseCategoricalCrossentropy(
            from_logits=False)
        hist2 = train_model(model2, loss2, x_tr, y2_tr, args.epochs,
                            args.batch_size, args.seed, '2cls')
        model2.save(str(DATA_DIR / 'ckpt_smoke_2cls_seed42.keras'))
        ev2 = eval_2cls(model2, x_te, y2_te)
        result['2cls'] = {
            'loss_history': [round(v, 5) for v in hist2],
            'final_loss': round(hist2[-1], 5),
            'metrics': ev2,
        }
        print('2cls metrics', json.dumps(ev2, indent=2), flush=True)

    ok5 = result.get('5cls', {}).get('metrics', {}).get('macro_auc', 0)
    ok2 = result.get('2cls', {}).get('metrics', {}).get('binary_auc', 0)
    result['verdict'] = {
        'macro_ge_0.85': bool(ok5 >= 0.85),
        'binary_ge_0.85': bool(ok2 >= 0.85),
        'overall_go': bool(ok5 >= 0.85 and ok2 >= 0.85),
    }
    with open(DATA_DIR / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('RESULT', json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
