#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R12-T3: DANN arm for the record head (gradient-level domain-invariance).

Candidate recipe (r9 form + ptbbal, r9_train.py) + gradient reversal domain
adversary (Ganin 2016; Hasani 2021 CinC positive evidence).  Domain head:
MIMIC-train / MEETI-train / PTB-train (3-way).  The device domain is NEVER
trained.  R11 lesson respected: this is gradient-level invariance only -
no input augmentation (aug arm failed -5.0pp paired), no resampling.

  L = CE_label(sample-weighted, candidate recipe)  +  lam * CE_domain
  (GRL flips the domain-CE gradient sign at the encoder: encoder learns
   domain-invariant features while the domain head learns to discriminate.)

lambda grid {0.1, 0.3, 1.0}; discipline: seed42 smoke first, >=3 seeds for
any claim, paired vs candidate same seed; acceptance = paired PTB val gain
in >=2/3 seeds AND spec@R90 not worse (BENCHMARK_V1 v1.5 sec 15.2).

Saved probe = label head only (plain functional model, bench-loadable).
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model, layers

sys.path.insert(0, str(Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme '
                            '25261/ecg-programme-tju-flex.electron-master')
                       / 'pc_tools/ecg_dl'))
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
import os  # noqa: E402
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402
from r9_data import load_all  # noqa: E402
from r8a_train_5s_student import (eval_meeti_val, eval_device,  # noqa: E402
                                  eval_split)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs'

_spec = importlib.util.spec_from_file_location(
    'ecg_bench', '/home/devcontainers/meeti/scripts/ecg_bench.py')
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)   # module level only; main() is never called


@tf.keras.utils.register_keras_serializable(package='DannECG')
class GradientReversal(layers.Layer):
    """Identity forward, negated gradient (Ganin & Lempitsky 2015)."""

    def call(self, x):
        @tf.custom_gradient
        def _rev(x):
            def grad(dy):
                return -dy
            return tf.identity(x), grad
        return _rev(x)


def build_dann(lr):
    base = build_ecg_resnet_lite_large(input_shape=(1250, 1))
    enc = Model(base.input, base.get_layer('fc1').output, name='enc_5s')
    inp = layers.Input((1250, 1), name='record_5s_input')
    f = enc(inp)
    label = layers.Dense(2, activation='softmax', name='head_r5s')(f)
    g = GradientReversal(name='grl')(f)
    h = layers.Dense(32, activation='relu', name='dom_h')(g)
    dom = layers.Dense(3, activation='softmax', name='dom_out')(h)
    tmodel = Model(inp, [label, dom], name='dann_student_5s')
    probe = Model(inp, label, name='record_student_5s')
    tmodel.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return tmodel, probe




def finalize_tail(probe, data, args, run_dir, best, history, t0,
                  train_windows=None, finalized=False):
    me, split, dev = data['meeti'], data['ptb_splits'], data['device']
    meeti_auc, _, _ = eval_meeti_val(probe, me)
    ptb = {}
    for name in ['train', 'val', 'test']:
        met, p_rec, y_rec = eval_split(probe, split[name])
        ptb[name] = {'auc': met['auc'],
                     'spec_at_R90': eb.spec_at_recall(y_rec, p_rec, 0.90)}
    device = eval_device(probe, dev)
    result = {
        'arm': 'r12_t3_dann', 'lam': args.lam, 'seed': args.seed,
        'config': vars(args),
        'domains': {'0': 'MIMIC-train', '1': 'MEETI-train',
                    '2': 'PTB-train',
                    'note': 'device domain never trained'},
        'domain_head': 'GRL -> Dense(32 relu) -> Dense(3 softmax)',
        'train_windows': train_windows or {},
        'best': best,
        'metrics': {'meeti_val_record_auc': meeti_auc,
                    'ptb_val_record_auc': ptb['val']['auc'],
                    'ptb_test_record_auc': ptb['test']['auc'],
                    'device': device},
        'ptb_splits': ptb, 'history': history,
        'finalized_from_ckpt': finalized,
        'elapsed_s': round(time.time() - t0, 1)}
    base_seed = json.load(open(
        OUT / f'r9_form-ptbbal_seed{args.seed}/result.json'))
    d_val = (ptb['val']['auc']
             - base_seed['metrics']['ptb_val_record_auc'])
    d_test = (ptb['test']['auc']
              - base_seed['metrics']['ptb_test_record_auc'])
    result['paired_vs_candidate'] = {
        'ptb_val_delta': round(d_val, 4),
        'ptb_test_delta': round(d_test, 4)}
    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[dann lam{args.lam} s{args.seed}] DONE meeti {meeti_auc:.4f} '
          f'ptb_val {ptb["val"]["auc"]:.4f} (paired {d_val:+.4f}) '
          f'ptb_test {ptb["test"]["auc"]:.4f} (paired {d_test:+.4f}) '
          f'specR90val {ptb["val"]["spec_at_R90"]["spec"]:.3f} '
          f'[{time.time() - t0:.0f}s]', flush=True)
    return result

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lam', type=float, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mimic-weight', type=float, default=0.2)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--suffix', default='')
    ap.add_argument('--finalize', action='store_true',
                    help='skip training; evaluate saved ckpt, write '
                         'result.json (crash recovery)')
    args, _ = ap.parse_known_args()
    if args.finalize:
        for g in tf.config.list_physical_devices('GPU'):
            tf.config.experimental.set_memory_growth(g, True)
        t0 = time.time()
        data = load_all(norm=True, form=True)
        run_dir = OUT / f'r12_dann_lam{args.lam}_seed{args.seed}{args.suffix}'
        probe = tf.keras.models.load_model(
            str(run_dir / 'ckpt_best__record5s.keras'))
        finalize_tail(probe, data, args, run_dir,
                      {'meeti_val_auc': None, 'epoch': None,
                       'note': 'history lost (crash); best epoch in stdout'},
                      None, t0, finalized=True)
        return

    run_dir = OUT / f'r12_dann_lam{args.lam}_seed{args.seed}{args.suffix}'
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    data = load_all(norm=True, form=True)
    mim, me = data['mimic'], data['meeti']
    ptb_tr, dev = data['ptb_train'], data['device']
    split = data['ptb_splits']

    # candidate (form) assembly + ptbbal prevalence-inverse PTB weights
    tr_win = mim['rec_tr'][mim['win_rec']]
    x_mim = mim['x'][tr_win]
    y_mim = mim['win_y'][tr_win]
    d_mim = np.zeros(len(y_mim), np.int32)
    w_mim = np.full(len(y_mim), args.mimic_weight, np.float32)

    tr_win_me = me['rec_tr'][me['win_rec']]
    x_me = me['x'][tr_win_me]
    y_me = me['rec_y'][me['win_rec']][tr_win_me]
    d_me = np.ones(len(y_me), np.int32)
    w_me = np.full(len(y_me), args.meeti_weight, np.float32)

    y_ptb = ptb_tr['win_y']
    n0 = max(int((y_ptb == 0).sum()), 1)
    n1 = max(int((y_ptb == 1).sum()), 1)
    w_ptb = np.where(y_ptb == 0, len(y_ptb) / (2 * n0),
                     len(y_ptb) / (2 * n1)).astype(np.float32)
    d_ptb = np.full(len(y_ptb), 2, np.int32)

    x_train = np.concatenate([x_mim, x_me, ptb_tr['x']], axis=0)
    y_train = np.concatenate([y_mim, y_me, y_ptb], axis=0)
    w_train = np.concatenate([w_mim, w_me, w_ptb])
    d_train = np.concatenate([d_mim, d_me, d_ptb])
    n_dom = np.bincount(d_train, minlength=3)
    print(f'[dann lam{args.lam} s{args.seed}] windows: mimic {len(y_mim)} + '
          f'meeti {len(y_me)} + ptb {len(y_ptb)} = {len(y_train)}; '
          f'domain counts {n_dom.tolist()}', flush=True)

    tmodel, probe = build_dann(args.lr)
    sce = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam(args.lr)
    lam = args.lam

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb, wb, db):
        with tf.GradientTape() as tape:
            lp, dp = tmodel(xb, training=True)
            l_lab = sce(yb, lp, sample_weight=wb)
            l_dom = sce(db, dp)
            loss = l_lab + lam * l_dom
        grads = tape.gradient(loss, tmodel.trainable_variables)
        opt.apply_gradients(zip(grads, tmodel.trainable_variables))
        return loss, l_lab, l_dom
    n = len(y_train)
    n_steps = math.ceil(n / args.batch_size)
    best_auc, best_ep = -1.0, -1
    history = {'loss': [], 'label': [], 'domain': [], 'meeti_val_auc': []}
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        ls, ll, ld = [], [], []
        for step in range(n_steps):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            loss, l_lab, l_dom = train_step(
                x_train[idx], y_train[idx], w_train[idx], d_train[idx])
            ls.append(float(loss))
            ll.append(float(l_lab))
            ld.append(float(l_dom))
        auc, _, _ = eval_meeti_val(probe, me)
        history['loss'].append(float(np.mean(ls)))
        history['label'].append(float(np.mean(ll)))
        history['domain'].append(float(np.mean(ld)))
        history['meeti_val_auc'].append(auc)
        if auc > best_auc:
            best_auc, best_ep = auc, epoch + 1
            probe.save(str(run_dir / 'ckpt_best__record5s.keras'))
        print(f'[dann lam{args.lam} s{args.seed}] ep{epoch + 1} '
              f'loss={history["loss"][-1]:.4f} label={history["label"][-1]:.4f}'
              f' dom={history["domain"][-1]:.4f} meeti={auc:.4f} '
              f'(best ep{best_ep} {best_auc:.4f})', flush=True)

    finalize_tail(probe, data, args, run_dir,
                  {'meeti_val_auc': best_auc, 'epoch': best_ep},
                  history, t0,
                  train_windows={'mimic': int(len(y_mim)),
                                 'meeti': int(len(y_me)),
                                 'ptb': int(len(y_ptb))})


if __name__ == '__main__':
    main()
