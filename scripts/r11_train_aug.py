#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R11: PTB-val variance suppression arms on top of the r9_form-ptbbal
candidate recipe (form + prevalence-inverse PTB weights).

Arms (train-side only, eval caliber unchanged):
  aug      : strong ECG augmentation applied on-the-fly to ALL train
             windows with prob 0.5/window: amplitude scale U[0.7,1.3],
             baseline wander a*sin(2*pi*f*t+phi) with a~U[0,0.05],
             f~U[0.1,0.5] (cycles/window), gaussian noise N(0, U[0,0.03])
             (SEMISEG evidence: amplitude scaling + baseline shift).
  resample : PTB train windows balanced-sampled per epoch (50/50 pos/neg
             with replacement) INSTEAD of prevalence-inverse weights;
             MIMIC/MEETI unchanged.

Paired comparison vs runs/r9_form-ptbbal_seed{S}/result.json per seed.
Success (frozen): >=2/3 seeds paired ptb_val improvement AND gates hold
(mean caliber: meeti>=0.7773, ptb_val/test >= locked-0.01).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import record_scores  # noqa: E402
from r8a_train_5s_student import (build_student_5s, eval_meeti_val,  # noqa
                                  eval_device, eval_split, ANCHORS,
                                  GATE_SLACK)
from r9_data import load_all  # noqa: E402
from train_dual_head_small import check_disk_redline  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')


def augment_batch(x, rng):
    """Strong ECG augmentation on a (B,1250) z-scored window batch."""
    x = x.copy()
    n = len(x)
    mask = rng.random(n) < 0.5
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return x
    xb = x[idx].astype(np.float32)
    # amplitude scale
    xb *= rng.uniform(0.7, 1.3, size=(len(idx), 1)).astype(np.float32)
    # baseline wander
    a = rng.uniform(0.0, 0.05, size=(len(idx), 1)).astype(np.float32)
    f = rng.uniform(0.1, 0.5, size=(len(idx), 1)).astype(np.float32)
    phi = rng.uniform(0.0, 2 * np.pi, size=(len(idx), 1)).astype(np.float32)
    t = np.linspace(0.0, 1.0, xb.shape[1], dtype=np.float32)[None, :]
    xb += a * np.sin(2 * np.pi * f * t + phi)
    # gaussian noise
    xb += rng.normal(0.0, 1.0, size=xb.shape).astype(np.float32) * \
        rng.uniform(0.0, 0.03, size=(len(idx), 1)).astype(np.float32)
    x[idx] = xb
    return x


def train_one_seed(arm, seed, args, data):
    tag = f'r11_{arm}'
    run_dir = ROOT / f'runs/{tag}_seed{seed}{args.suffix}'
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mim, me = data['mimic'], data['meeti']
    ptb_tr, dev = data['ptb_train'], data['device']

    tr_win = mim['rec_tr'][mim['win_rec']]
    x_mim = mim['x'][tr_win]
    y_mim = mim['win_y'][tr_win]
    w_mim = np.full(len(y_mim), args.mimic_weight, np.float32)

    tr_win_me = me['rec_tr'][me['win_rec']]
    x_me = me['x'][tr_win_me]
    y_me = me['rec_y'][me['win_rec']][tr_win_me]
    w_me = np.full(len(y_me), args.meeti_weight, np.float32)

    y_ptb = ptb_tr['win_y']
    x_ptb = ptb_tr['x']
    i0_all = i1_all = None
    n_half = 0
    if arm == 'aug':
        n0 = max(int((y_ptb == 0).sum()), 1)
        n1 = max(int((y_ptb == 1).sum()), 1)
        w_ptb = np.where(y_ptb == 0, len(y_ptb) / (2 * n0),
                         len(y_ptb) / (2 * n1)).astype(np.float32)
        y_ptb_use, w_ptb_use = y_ptb, w_ptb
    else:  # resample: balanced PTB sampling replaces inverse-freq weights
        i0_all = np.where(y_ptb == 0)[0]
        i1_all = np.where(y_ptb == 1)[0]
        n_half = int(min(len(i0_all), len(i1_all)))
        y_ptb_use, w_ptb_use = None, np.ones(2 * n_half, np.float32)
    ptb_meta = {'n0': int((y_ptb == 0).sum()), 'n1': int((y_ptb == 1).sum()),
                'i0_all': i0_all, 'i1_all': i1_all, 'n_half': n_half}

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    model = build_student_5s(args.lr)
    best = {'meeti_val_auc': -1.0}
    history = {'loss': [], 'meeti_val_auc': []}
    for epoch in range(args.epochs):
        if arm == 'aug':
            x_train = np.concatenate([augment_batch(x_mim, rng),
                                      augment_batch(x_me, rng),
                                      augment_batch(x_ptb, rng)], axis=0)
            y_train = np.concatenate([y_mim, y_me, y_ptb], axis=0)
            w_train = np.concatenate([w_mim, w_me, w_ptb])
        else:
            i0 = rng.choice(ptb_meta['i0_all'],
                            size=ptb_meta['n_half'], replace=True)
            i1 = rng.choice(ptb_meta['i1_all'],
                            size=ptb_meta['n_half'], replace=True)
            iptb = np.concatenate([i0, i1])
            x_train = np.concatenate([x_mim, x_me, x_ptb[iptb]], axis=0)
            y_train = np.concatenate([y_mim, y_me, y_ptb[iptb]], axis=0)
            w_train = np.concatenate([w_mim, w_me, w_ptb_use])
        perm = rng.permutation(len(y_train))
        n_steps = math.ceil(len(y_train) / args.batch_size)
        losses = []
        for step in range(n_steps):
            idx = perm[step * args.batch_size:
                       (step + 1) * args.batch_size]
            losses.append(model.train_on_batch(
                x_train[idx], y_train[idx], sample_weight=w_train[idx]))
        auc, _, _ = eval_meeti_val(model, me)
        history['loss'].append(float(np.mean(losses)))
        history['meeti_val_auc'].append(auc)
        if auc > best['meeti_val_auc']:
            best = {'meeti_val_auc': auc, 'epoch': epoch + 1}
            model.save(str(run_dir / 'ckpt_best__record5s.keras'))
        print(f'[{arm} s{seed}] ep{epoch + 1} '
              f'loss={history["loss"][-1]:.4f} meeti={auc:.4f} '
              f'(best ep{best["epoch"]} {best["meeti_val_auc"]:.4f})',
              flush=True)

    probe = tf.keras.models.load_model(
        str(run_dir / 'ckpt_best__record5s.keras'))
    meeti_auc, _, _ = eval_meeti_val(probe, me)
    ptb = {name: eval_split(probe, data['ptb_splits'][name])[0]
           for name in ['train', 'val', 'test']}
    device = eval_device(probe, dev)

    result = {
        'arm': tag, 'seed': seed,
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size,
                   'lr': args.lr, 'base_recipe': 'r9 form+ptbbal',
                   'aug': arm == 'aug',
                   'aug_spec': 'p0.5/window: amp U[0.7,1.3], wander '
                               'a<=0.05 f<=0.5cyc, noise sd<=0.03',
                   'resample': arm == 'resample'},
        'train_windows': {'mimic': int(len(y_mim)), 'meeti': int(len(y_me)),
                          'ptb': int(len(y_ptb))},
        'best': best,
        'metrics': {'meeti_val_record_auc': meeti_auc,
                    'ptb_val_record_auc': ptb['val']['auc'],
                    'ptb_test_record_auc': ptb['test']['auc'],
                    'device': device},
        'ptb_splits': ptb,
        'history': history,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'[{arm} s{seed}] DONE meeti {meeti_auc:.4f} '
          f'ptb_val {ptb["val"]["auc"]:.4f} ptb_test '
          f'{ptb["test"]["auc"]:.4f} dev_frac '
          f'{device["pooled"]["frac_gt_0.5"]:.4f} '
          f'[{time.time() - t0:.0f}s]', flush=True)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['aug', 'resample'], required=True)
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mimic-weight', type=float, default=0.2)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    check_disk_redline()
    data = load_all(norm=True, form=True)
    for s in seeds:
        train_one_seed(args.arm, s, args, data)


if __name__ == '__main__':
    main()
