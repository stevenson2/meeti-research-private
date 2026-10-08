#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R9: normalization / form-alignment arms for the 5s single-lead student.

Training recipe identical to frozen R8a (r8a_train_5s_student.py):
resnet-lite-large fc1(64) + Dense(2), Adam 3e-4, batch 64, 20 epochs,
window-level BCE with weights MIMIC 0.2 / MEETI 0.2 / PTB 1.0, checkpoint =
best MEETI val record AUC epoch, seeds 42/43/44, 3-seed paired comparison
vs R8a per seed.  Only the MIMIC input construction changes:

  --arm norm : per-window z-score of MIMIC beat-centered windows (P1 probe:
               deploy-chain windows are exactly unit-scale, MIMIC is raw mV)
  --arm form : norm + MIMIC windows rebuilt as 5-beat concatenations
               (+/-125 pt around consecutive r_peaks) = same form as
               MEETI/PTB train and ALL eval windows

Gates unchanged from R8a (MEETI >= 0.7773; PTB val/test >= locked-0.01);
device-domain frac reported with the documented reference correction
(locked record head is also frac=1.0; the <=1% line belongs to the beat
head) - discriminative gates decide, device stats disclosed.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import record_scores  # noqa: E402
from r8a_train_5s_student import (build_student_5s, eval_meeti_val,  # noqa
                                  eval_device, eval_split, ANCHORS,
                                  GATE_SLACK)
from r9_data import load_all  # noqa: E402
from train_dual_head_small import check_disk_redline  # noqa: E402


def train_one_seed(arm, seed, args, data):
    tag = f'r9_{arm}'
    if args.w5:
        tag += '-w5'
    if args.ptb_bal:
        tag += '-ptbbal'
    if args.cosine:
        tag += '-cos'
    if args.hard_neg_ptb != 1.0 or args.hard_neg_meeti != 1.0:
        tag += '-hardneg'
    run_dir = Path(f'/home/devcontainers/meeti/runs/'
                   f'{tag}_seed{seed}{args.suffix}')
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mim, me = data['mimic'], data['meeti']
    ptb_tr, dev = data['ptb_train'], data['device']

    if arm == 'norm':
        x_mim = mim['x'][mim['tr']]
        y_mim = mim['y'][mim['tr']]
    else:  # form
        tr_win = mim['rec_tr'][mim['win_rec']]
        x_mim = mim['x'][tr_win]
        y_mim = mim['win_y'][tr_win]
    w_mim = np.full(len(y_mim), args.mimic_weight, np.float32)
    teacher_info = None
    if args.w5:
        if arm != 'norm':
            raise SystemExit('--w5 requires --arm norm (teacher probs are '
                             'indexed by beat-centered windows)')
        from r8_data_5s import teacher_oos_probs
        p_teacher, oos_auc = teacher_oos_probs(mim)
        p_given = np.where(y_mim == 1, p_teacher, 1.0 - p_teacher)
        w_mim = np.asarray(
            args.mimic_weight * (0.25 + 0.75 * p_given), np.float32)
        teacher_info = {'teacher_oos_auc': oos_auc,
                        'w_min': float(w_mim.min()),
                        'w_mean': float(w_mim.mean())}
        print(f'[w5 s{seed}] teacher OOS {oos_auc:.4f} w mean '
              f'{w_mim.mean():.4f}', flush=True)

    tr_win_me = me['rec_tr'][me['win_rec']]
    x_me = me['x'][tr_win_me]
    y_me = me['rec_y'][me['win_rec']][tr_win_me]
    w_me = np.full(len(y_me), args.meeti_weight, np.float32)

    y_ptb = ptb_tr['win_y']
    if args.ptb_bal:
        # prevalence-inverse window weighting (ECG-FM convention) against
        # the 87.7% positive window-max prior
        n0 = max(int((y_ptb == 0).sum()), 1)
        n1 = max(int((y_ptb == 1).sum()), 1)
        w_ptb = np.where(y_ptb == 0, len(y_ptb) / (2 * n0),
                         len(y_ptb) / (2 * n1)).astype(np.float32)
    else:
        w_ptb = np.ones(len(y_ptb), np.float32)
    if args.hard_neg_ptb != 1.0:
        # T5 hard-negative oversampling (as window weights, expectation-
        # equivalent to duplication): upweight PTB NORMAL windows only -
        # opposite direction to the failed R11 balanced-resampling arm
        w_ptb = np.where(y_ptb == 0, w_ptb * args.hard_neg_ptb,
                         w_ptb).astype(np.float32)
    if args.hard_neg_meeti != 1.0:
        w_me = np.where(y_me == 0, w_me * args.hard_neg_meeti,
                        w_me).astype(np.float32)

    x_train = np.concatenate([x_mim, x_me, ptb_tr['x']], axis=0)
    y_train = np.concatenate([y_mim, y_me, y_ptb], axis=0)
    w_train = np.concatenate([w_mim, w_me, w_ptb])
    print(f'[{arm} s{seed}] train windows: mimic {len(y_mim)} + '
          f'meeti {len(y_me)} + ptb {len(ptb_tr["win_y"])} = {len(y_train)}',
          flush=True)

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    model = build_student_5s(args.lr)
    n = len(y_train)
    n_steps = math.ceil(n / args.batch_size)
    best = {'meeti_val_auc': -1.0}
    history = {'loss': [], 'meeti_val_auc': [], 'lr': []}
    for epoch in range(args.epochs):
        if args.cosine:
            # cosine annealing lr -> lr*0.033 (SEMISEG-style schedule);
            # stabilizes late-epoch convergence (s42-type runs were still
            # climbing at ep20)
            lr_t = args.lr * (0.033 + 0.5 * (1 - 0.033)
                              * (1 + math.cos(math.pi * epoch / args.epochs)))
            lr_var = model.optimizer.learning_rate
            if hasattr(lr_var, 'assign'):
                lr_var.assign(lr_t)
            else:
                model.optimizer.learning_rate = lr_t
            history['lr'].append(float(lr_t))
        perm = rng.permutation(n)
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
                   'lr': args.lr, 'mimic_weight': args.mimic_weight,
                   'meeti_weight': args.meeti_weight, 'ptb_weight': 1.0,
                   'w5_teacher_reweight': bool(args.w5),
                   'ptb_prevalence_inverse': bool(args.ptb_bal),
                   'cosine_lr': bool(args.cosine),
                   'mimic_input': ('per-window z-scored beat-centered 5s'
                                   if arm == 'norm' else
                                   'per-window z-scored 5-beat concat'),
                   'data': 'r9_data.py (post-R8a normalization/form fix)'},
        'train_windows': {'mimic': int(len(y_mim)),
                          'meeti': int(len(y_me)),
                          'ptb': int(len(ptb_tr['win_y']))},
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


def summarize(tag, seeds, results):
    def ms(vals):
        return {'mean': float(np.mean(vals)),
                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'values': [float(v) for v in vals]}

    s = {'arm': tag, 'seeds': seeds,
         'meeti_val_record_auc': ms([r['metrics']['meeti_val_record_auc']
                                     for r in results]),
         'ptb_val_record_auc': ms([r['metrics']['ptb_val_record_auc']
                                   for r in results]),
         'ptb_test_record_auc': ms([r['metrics']['ptb_test_record_auc']
                                    for r in results]),
         'device_frac_gt_0.5': [r['metrics']['device']['pooled']
                                ['frac_gt_0.5'] for r in results]}
    paired = []
    for seed, r in zip(seeds, results):
        base = json.load(open(f'/home/devcontainers/meeti/runs/'
                              f'r8a_5s_student_seed{seed}/result.json'))
        paired.append({
            'seed': seed,
            'delta_meeti_val': r['metrics']['meeti_val_record_auc']
            - base['metrics']['meeti_val_record_auc'],
            'delta_ptb_val': r['metrics']['ptb_val_record_auc']
            - base['metrics']['ptb_val_record_auc'],
            'delta_ptb_test': r['metrics']['ptb_test_record_auc']
            - base['metrics']['ptb_test_record_auc']})
    s['paired_vs_r8a'] = paired
    s['gates'] = {
        'meeti_val_ge_locked': bool(
            s['meeti_val_record_auc']['mean'] >= ANCHORS['meeti_val']),
        'ptb_val_ge_locked_minus_1pp': bool(
            s['ptb_val_record_auc']['mean']
            >= ANCHORS['ptb_val'] - GATE_SLACK),
        'ptb_test_ge_locked_minus_1pp': bool(
            s['ptb_test_record_auc']['mean']
            >= ANCHORS['ptb_test'] - GATE_SLACK),
    }
    s['verdict'] = ('PASS_DISCRIMINATIVE_GATES'
                    if all(s['gates'].values()) else 'GATE_FAILED')
    out = f'/home/devcontainers/meeti/runs/{tag}_summary.json'
    with open(out, 'w') as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print(json.dumps(s, indent=2), flush=True)
    print('summary ->', out, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['norm', 'form'], required=True)
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mimic-weight', type=float, default=0.2)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--w5', action='store_true',
                    help='teacher OOS probability bounded reweighting of '
                         'MIMIC windows (task B; norm arm only)')
    ap.add_argument('--ptb-bal', action='store_true',
                    help='prevalence-inverse PTB window weights')
    ap.add_argument('--cosine', action='store_true',
                    help='cosine annealing lr schedule (-> lr*0.033)')
    ap.add_argument('--hard-neg-ptb', type=float, default=1.0,
                    help='T5: extra weight multiplier on PTB train NORMAL '
                         'windows (1.0 = off; recipe candidate otherwise)')
    ap.add_argument('--hard-neg-meeti', type=float, default=1.0,
                    help='T5: extra weight multiplier on MEETI train '
                         'NORMAL windows (1.0 = off)')
    ap.add_argument('--suffix', default='')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    check_disk_redline()
    data = load_all(norm=True, form=(args.arm == 'form'))
    results = [train_one_seed(args.arm, s, args, data) for s in seeds]
    if args.suffix:
        print('scratch run: skip summary', flush=True)
    else:
        tag = f'r9_{args.arm}'
        if args.w5:
            tag += '-w5'
        if args.ptb_bal:
            tag += '-ptbbal'
        if args.cosine:
            tag += '-cos'
        if args.hard_neg_ptb != 1.0 or args.hard_neg_meeti != 1.0:
            tag += '-hardneg'
        summarize(tag, seeds, results)


if __name__ == '__main__':
    main()
