#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R8a/R8b-w5: standalone 5 s single-lead record student (no KD).

Task A (arm r8a) = pure-capacity baseline for the R8 route: replace the
10-beat TimeDistributed record head with a single-lead 5 s (1250 pt @250 Hz)
independent encoder + Dense(2) head.  If it cannot match the locked record
head WITHOUT distillation, the whole 5 s route stops.

Task B (arm r8b-w5) = continuous bounded teacher-score reweighting of the
MIMIC windows (the R4/5 mechanism, not pruning, not binarized downweight):
    w_i = 0.2 * (0.25 + 0.75 * p_teacher_given_label_i)     in [0.05, 0.2]
    p_given = p_teacher if y==1 else (1 - p_teacher)
MEETI (0.2) and PTB (1.0) weights unchanged; every other hyperparameter
identical to r8a, same seeds -> paired comparison.

FROZEN CONFIG (R8 launch prompt, 2026-09-05):
  model     build_ecg_resnet_lite_large(input_shape=(1250,1)), encoder =
            fc1(64), head Dense(2, softmax); NO layer shared with the beat
            head (structurally removes the B/C shared-encoder failure mode).
  training  Adam 3e-4, batch 64, 20 epochs, tf.keras.utils.set_random_seed(s),
            s in {42,43,44}; each epoch sweeps ALL MIMIC+MEETI+PTB record
            windows (window-level BCE, sample weights MIMIC 0.2 / MEETI 0.2 /
            PTB 1.0); checkpoint = epoch with the highest MEETI val record
            AUC (locked best_record caliber) -> ckpt_best__record5s.keras.
  data      r8_data_5s.py (frozen calibers in its docstring).

Gates (R8a, frozen): MEETI val record AUC mean >= 0.7773 OR >=2/3 seeds
individually >= 0.7773; PTB val/test record AUC means >= 0.8880-0.01 /
0.8367-0.01; device domain frac>0.5 <= 1% on both real recordings.
Gates (R8b-w5, frozen): paired vs r8a same seed, MEETI val AUC improves in
>=2/3 seeds AND PTB val/test means >= locked-0.01 AND device frac <= 1%.
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
from r8_data_5s import (mimic_5s, meeti_5s, ptb_5s, device_5s,  # noqa: E402
                        record_scores, teacher_oos_probs)
from r8_eval_ptb_holdout import eval_split  # noqa: E402
from train_dual_head_small import check_disk_redline  # noqa: E402
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402
from tensorflow.keras import Model, layers  # noqa: E402

ANCHORS = {'meeti_val': 0.7773, 'ptb_val': 0.8880, 'ptb_test': 0.8367}
GATE_SLACK = 0.01


def build_student_5s(lr):
    """Independent 5 s record student: resnet-lite-large fc1(64) + softmax."""
    base = build_ecg_resnet_lite_large(input_shape=(1250, 1))
    encoder = Model(base.input, base.get_layer('fc1').output,
                    name='enc_5s')
    inp = layers.Input((1250, 1), name='record_5s_input')
    head = layers.Dense(2, activation='softmax', name='head_r5s')(
        encoder(inp))
    model = Model(inp, head, name='record_student_5s')
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return model


def eval_meeti_val(probe, me, batch_size=256):
    va_win = me['rec_va'][me['win_rec']]          # bool over windows
    p_win = probe.predict(me['x'][va_win], batch_size=batch_size,
                          verbose=0)[:, 1]
    val_records = np.where(me['rec_va'])[0]
    remap = np.zeros(me['n_records'], np.int64)
    remap[val_records] = np.arange(len(val_records))
    rec_ids = me['win_rec'][va_win]               # global record index
    p_rec = record_scores(p_win, remap[rec_ids], len(val_records))
    y_rec = me['rec_y'][val_records]
    return float(roc_auc_score(y_rec, p_rec)), p_rec, y_rec


def eval_device(probe, dev, batch_size=256):
    out = {}
    pooled = []
    for name, d in dev.items():
        p = probe.predict(d['x'], batch_size=batch_size, verbose=0)[:, 1]
        out[name] = {'n_windows': int(len(p)), 'n_beats': d['n_beats'],
                     'p_mean': float(p.mean()),
                     'frac_gt_0.5': float((p > 0.5).mean())}
        pooled.append(p)
    pooled = np.concatenate(pooled)
    out['pooled'] = {'n_windows': int(len(pooled)),
                     'p_mean': float(pooled.mean()),
                     'frac_gt_0.5': float((pooled > 0.5).mean())}
    return out


def train_one_seed(arm, seed, args, data):
    tag = 'r8a_5s_student' if arm == 'r8a' else 'r8b_w5'
    run_dir = Path(f'/home/devcontainers/meeti/runs/'
                   f'{tag}_seed{seed}{getattr(args, "suffix", "")}')
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    mim, me = data['mimic'], data['meeti']
    ptb_tr, dev = data['ptb_train'], data['device']
    x_mim, y_mim = mim['x'][mim['tr']], mim['y'][mim['tr']]
    w_mim = np.full(len(y_mim), args.mimic_weight, np.float32)
    teacher_info = None
    if arm == 'r8b-w5':
        p_teacher, oos_auc = teacher_oos_probs(mim)
        p_given = np.where(y_mim == 1, p_teacher, 1.0 - p_teacher)
        w_mim = np.asarray(
            args.mimic_weight * (0.25 + 0.75 * p_given), np.float32)
        teacher_info = {
            'teacher_oos_auc': oos_auc,
            'w_min': float(w_mim.min()), 'w_mean': float(w_mim.mean()),
            'w_max': float(w_mim.max()),
        }
        print(f'teacher OOS AUC {oos_auc:.4f} | w range '
              f'[{w_mim.min():.4f}, {w_mim.max():.4f}]', flush=True)

    tr_win_me = me['rec_tr'][me['win_rec']]
    x_me, y_me = me['x'][tr_win_me], me['rec_y'][me['win_rec']][tr_win_me]
    w_me = np.full(len(y_me), args.meeti_weight, np.float32)

    x_train = np.concatenate([x_mim, x_me, ptb_tr['x']], axis=0)
    y_train = np.concatenate([y_mim, y_me, ptb_tr['win_y']], axis=0)
    w_train = np.concatenate([w_mim, w_me,
                              np.ones(len(ptb_tr['win_y']), np.float32)])
    print(f'train windows: mimic {len(y_mim)} + meeti {len(y_me)} + '
          f'ptb {len(ptb_tr["win_y"])} = {len(y_train)}', flush=True)

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)

    model = build_student_5s(args.lr)
    n = len(y_train)
    n_steps = math.ceil(n / args.batch_size)
    best = {'meeti_val_auc': -1.0}
    history = {'loss': [], 'meeti_val_auc': []}
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        losses = []
        for step in range(n_steps):
            idx = perm[step * args.batch_size:
                       (step + 1) * args.batch_size]
            losses.append(model.train_on_batch(
                x_train[idx], y_train[idx],
                sample_weight=w_train[idx]))
        auc, _, _ = eval_meeti_val(model, me)
        history['loss'].append(float(np.mean(losses)))
        history['meeti_val_auc'].append(auc)
        if auc > best['meeti_val_auc']:
            best = {'meeti_val_auc': auc, 'epoch': epoch + 1}
            model.save(str(run_dir / 'ckpt_best__record5s.keras'))
        print(f'ep{epoch + 1} loss={history["loss"][-1]:.4f} '
              f'meeti_val_auc={auc:.4f} '
              f'(best ep{best["epoch"]} {best["meeti_val_auc"]:.4f})',
              flush=True)

    # ---- final three-domain eval on the reloaded best checkpoint ----
    probe = tf.keras.models.load_model(
        str(run_dir / 'ckpt_best__record5s.keras'))
    meeti_auc, _, _ = eval_meeti_val(probe, me)
    ptb = {}
    for name in ['train', 'val', 'test']:
        res, _, _ = eval_split(probe, data['ptb_splits'][name])
        ptb[name] = res
    device = eval_device(probe, dev)

    result = {
        'arm': arm, 'seed': seed,
        'config': {'epochs': args.epochs, 'batch_size': args.batch_size,
                   'lr': args.lr, 'mimic_weight': args.mimic_weight,
                   'meeti_weight': args.meeti_weight, 'ptb_weight': 1.0,
                   'input': '(1250,1) lead II 5s @250Hz',
                   'checkpoint': 'best MEETI val record AUC epoch',
                   'data_caliber': 'r8_data_5s.py frozen docstring'},
        'model_params': int(model.count_params()),
        'train_windows': {'mimic': int(len(y_mim)),
                          'meeti': int(len(y_me)),
                          'ptb': int(len(ptb_tr['win_y']))},
        'best': best,
        'metrics': {'meeti_val_record_auc': meeti_auc,
                    'ptb_val_record_auc': ptb['val']['auc'],
                    'ptb_test_record_auc': ptb['test']['auc'],
                    'ptb_train_record_auc': ptb['train']['auc'],
                    'device': device},
        'ptb_splits': ptb,
        'teacher': teacher_info,
        'history': history,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f'seed {seed} done: meeti {meeti_auc:.4f} ptb val '
          f'{ptb["val"]["auc"]:.4f} test {ptb["test"]["auc"]:.4f} '
          f'device frac {device["pooled"]["frac_gt_0.5"]:.4f} '
          f'[{time.time() - t0:.0f}s]', flush=True)
    return result


def summarize(arm, seeds, results):
    """3-seed aggregates + frozen gate verdicts (prompt section 1.3 / 2)."""
    def ms(vals):
        return {'mean': float(np.mean(vals)),
                'sd': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                'values': [float(v) for v in vals]}

    s = {'arm': arm, 'seeds': seeds,
         'meeti_val_record_auc': ms([r['metrics']['meeti_val_record_auc']
                                     for r in results]),
         'ptb_val_record_auc': ms([r['metrics']['ptb_val_record_auc']
                                   for r in results]),
         'ptb_test_record_auc': ms([r['metrics']['ptb_test_record_auc']
                                    for r in results])}
    pooled_frac = [r['metrics']['device']['pooled']['frac_gt_0.5']
                   for r in results]
    dev_frac_max = max(pooled_frac)
    s['device_frac_gt_0.5'] = {'values': pooled_frac, 'max': dev_frac_max}

    if arm == 'r8a':
        mean_meeti = s['meeti_val_record_auc']['mean']
        n_ge = sum(1 for v in s['meeti_val_record_auc']['values']
                   if v >= ANCHORS['meeti_val'])
        s['gates'] = {
            'meeti_val_ge_locked': bool(mean_meeti >= ANCHORS['meeti_val']
                                        or n_ge >= 2),
            'meeti_val_mean': mean_meeti,
            'meeti_val_seeds_ge_locked': n_ge,
            'ptb_val_ge_locked_minus_1pp': bool(
                s['ptb_val_record_auc']['mean']
                >= ANCHORS['ptb_val'] - GATE_SLACK),
            'ptb_test_ge_locked_minus_1pp': bool(
                s['ptb_test_record_auc']['mean']
                >= ANCHORS['ptb_test'] - GATE_SLACK),
            'device_frac_le_1pct': bool(dev_frac_max <= 0.01),
        }
        s['verdict'] = ('PASS_ALL_GATES' if all(s['gates'].values())
                        else 'GATE_FAILED')
    else:
        paired = []
        for seed, r in zip(seeds, results):
            base = json.load(open(
                f'/home/devcontainers/meeti/runs/'
                f'r8a_5s_student_seed{seed}/result.json'))
            paired.append({
                'seed': seed,
                'delta_meeti_val': r['metrics']['meeti_val_record_auc']
                - base['metrics']['meeti_val_record_auc'],
                'delta_ptb_val': r['metrics']['ptb_val_record_auc']
                - base['metrics']['ptb_val_record_auc'],
                'delta_ptb_test': r['metrics']['ptb_test_record_auc']
                - base['metrics']['ptb_test_record_auc']})
        n_impr = sum(1 for p in paired if p['delta_meeti_val'] > 0)
        s['paired_vs_r8a'] = paired
        s['gates'] = {
            'meeti_val_improves_ge_2of3': bool(n_impr >= 2),
            'ptb_val_ge_locked_minus_1pp': bool(
                s['ptb_val_record_auc']['mean']
                >= ANCHORS['ptb_val'] - GATE_SLACK),
            'ptb_test_ge_locked_minus_1pp': bool(
                s['ptb_test_record_auc']['mean']
                >= ANCHORS['ptb_test'] - GATE_SLACK),
            'device_frac_le_1pct': bool(dev_frac_max <= 0.01),
        }
        s['verdict'] = ('W5_EFFECTIVE' if all(s['gates'].values())
                        else 'W5_NEGATIVE')

    out = (f'/home/devcontainers/meeti/runs/'
           f'{"r8a_5s_student" if arm == "r8a" else "r8b_w5"}_summary.json')
    with open(out, 'w') as f:
        json.dump(s, f, indent=2, ensure_ascii=False)
    print(json.dumps(s, indent=2), flush=True)
    print('summary ->', out, flush=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['r8a', 'r8b-w5'], default='r8a')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--mimic-weight', type=float, default=0.2)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--suffix', default='',
                    help='optional run-dir suffix (scratch tests only)')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]
    check_disk_redline()
    print(f'R8 arm={args.arm} seeds={seeds} epochs={args.epochs} '
          f'lr={args.lr} batch={args.batch_size}', flush=True)

    from data.split_guard import compute_ptb_split  # noqa: E402
    print('building data (frozen r8_data_5s calibers)...', flush=True)
    mim = mimic_5s()
    me = meeti_5s()
    split = compute_ptb_split()
    ptb_tr = ptb_5s(split['train'])
    dev = device_5s()
    data = {'mimic': mim, 'meeti': me, 'ptb_train': ptb_tr,
            'ptb_splits': {k: np.asarray(split[k], bool)
                           for k in ['train', 'val', 'test']},
            'device': dev}

    results = []
    for seed in seeds:
        results.append(train_one_seed(args.arm, seed, args, data))
    if args.suffix:
        print(f'suffix={args.suffix}: skip gate summary (scratch run)',
              flush=True)
    else:
        summarize(args.arm, seeds, results)


if __name__ == '__main__':
    main()
