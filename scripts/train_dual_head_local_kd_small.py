#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 local feature distillation (only after probe gate passes).

Same v3 recipe (combined Adam, balanced hard/record padding, Keras seed 42,
same probe/eval protocol).  The ECG-FM 5 s window is NEVER attached per beat;
only the pre-extracted center-time-step local feature (teacher_local_768) is
used as an MSE target for a projection of the shared beat fc1 embedding.

Loss: L_beat_CE + lam * L_record_CE + local_lam * MSE(fc1_local_proj, local_target)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model, layers

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, check_disk_redline, load_hard_beats,
    load_meeti_records, load_ptb_train_records,
)

T_BEATS = 10
KD_BEATS = Path('/home/devcontainers/meeti/data/processed_small/m3_beat_centered/'
                'student_beats_lead2.npy')
KD_TARGET = Path('/home/devcontainers/meeti/data/processed_small/m3_beat_centered/'
                 'teacher_local_768.npy')


def build_full_model(lr, lam, local_lam, target_dim=768):
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
    encoder = Model(base.input, base.get_layer('fc1').output,
                    name='beat_encoder')

    beat_in = layers.Input((250, 1), name='beat_input')
    beat_emb = encoder(beat_in)
    head_b = layers.Dense(2, activation='softmax', name='head_b')(beat_emb)

    kd_in = layers.Input((250, 1), name='kd_beat_input')
    kd_emb = encoder(kd_in)
    local_proj = layers.Dense(target_dim, name='fc1_local_proj')(kd_emb)

    rec_in = layers.Input((T_BEATS, 250, 1), name='record_input')
    rec_mask_in = layers.Input((T_BEATS, 1), name='record_mask')
    rec_emb = layers.TimeDistributed(encoder, name='td_encoder')(rec_in)
    rec_proj = layers.TimeDistributed(
        layers.Dense(32, activation='relu', name='head_r_proj'),
        name='td_head_r_proj')(rec_emb)
    rec_pool = MaskedAttentionPooling(name='head_r_attention')(
        [rec_proj, rec_mask_in])
    head_r = layers.Dense(2, activation='softmax', name='head_r')(rec_pool)

    model = Model([beat_in, rec_in, rec_mask_in, kd_in],
                  [head_b, head_r, local_proj],
                  name='dual_head_v3_local_kd')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
              'head_r': tf.keras.losses.SparseCategoricalCrossentropy(),
              'fc1_local_proj': tf.keras.losses.MeanSquaredError()},
        loss_weights={'head_b': 1.0, 'head_r': lam,
                      'fc1_local_proj': local_lam},
    )
    beat_probe = Model(model.inputs[0], model.outputs[0], name='beat_probe')
    rec_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[1],
                      name='record_probe')
    return model, beat_probe, rec_probe


def pad_indices(perm, target_len):
    if target_len <= len(perm):
        return perm[:target_len]
    reps = int(math.ceil(target_len / len(perm)))
    return np.tile(perm, reps)[:target_len]


def predict_rec(probe, x, mask, batch_size=256):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(probe.predict([x[i:i + batch_size], mask[i:i + batch_size]],
                                 batch_size=batch_size, verbose=0))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hard-batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.2)
    ap.add_argument('--local-lam', type=float, default=0.01)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--use-ptb', action='store_true')
    ap.add_argument('--normalize-target', default='l2', choices=['l2', 'none'])
    ap.add_argument('--kd-beats', default=str(KD_BEATS))
    ap.add_argument('--kd-target', default=str(KD_TARGET))
    ap.add_argument('--warmup-epochs', type=int, default=0)
    ap.add_argument('--tag', default='dual_head_local_kd_v3')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_local_kd_v3')
    args = ap.parse_args()

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    disk_before = check_disk_redline()

    gpus = tf.config.list_physical_devices('GPU')
    print('gpus', gpus, flush=True)
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(42)
    rng = np.random.default_rng(42)

    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti_args = argparse.Namespace(meeti_weight=args.meeti_weight)
    meeti = load_meeti_records(meeti_args)
    rec_x, rec_mask, rec_y, rec_w = (meeti['x'], meeti['mask'],
                                     meeti['y'], meeti['w'])
    rec_xv, rec_mv, rec_yv = meeti['x_val'], meeti['mask_val'], meeti['y_val']
    source_info = {'meeti': {'n_train': meeti['n_train'],
                             'n_val': meeti['n_val'],
                             'n_subjects': meeti['n_subjects'],
                             'train_label_dist': meeti['train_label_dist'],
                             'val_label_dist': meeti['val_label_dist']}}
    if args.use_ptb:
        ptb = load_ptb_train_records()
        rec_x = np.concatenate([rec_x, ptb['x']], axis=0)
        rec_mask = np.concatenate([rec_mask, ptb['mask']], axis=0)
        rec_y = np.concatenate([rec_y, ptb['y']], axis=0)
        rec_w = np.concatenate([rec_w, ptb['w']], axis=0)
        source_info['ptb'] = {k: ptb[k] for k in
                              ('n_records', 'label_dist', 'split_stats')}

    # local KD pairs: student firmware beats -> teacher center-step features.
    kd_x = np.load(args.kd_beats, mmap_mode='r').astype(np.float32)[..., None]
    kd_target = np.asarray(np.load(args.kd_target, mmap_mode='r'),
                           dtype=np.float32)
    if len(kd_x) != len(kd_target):
        raise SystemExit(f'kd pair mismatch: {len(kd_x)} beats vs '
                         f'{len(kd_target)} targets')
    if args.normalize_target == 'l2':
        norms = np.linalg.norm(kd_target, axis=1, keepdims=True) + 1e-6
        kd_target = kd_target / norms
    n_kd = len(kd_x)
    n_hard, n_rec = len(xb), len(rec_x)
    rec_steps = math.ceil(n_hard / args.hard_batch)
    rec_bs = math.ceil(n_rec / rec_steps)
    kd_bs = math.ceil(n_kd / rec_steps)
    hard_total = rec_steps * args.hard_batch
    rec_total = rec_steps * rec_bs
    kd_total = rec_steps * kd_bs
    hard_pad = hard_total - n_hard
    rec_pad = rec_total - n_rec
    kd_pad = kd_total - n_kd
    print(f'hard train={n_hard} hard_bs={args.hard_batch} '
          f'rec_steps={rec_steps} pad={hard_pad} val={len(xv)} '
          f'real={len(x_real)}', flush=True)
    print(f'record train={n_rec} rec_bs={rec_bs} total={rec_total} '
          f'pad={rec_pad} val={len(rec_xv)}', flush=True)
    print(f'kd train={n_kd} kd_bs={kd_bs} total={kd_total} pad={kd_pad} '
          f'target_norm={args.normalize_target} warmup={args.warmup_epochs} '
          f'kd_beats={args.kd_beats} kd_target={args.kd_target}', flush=True)
    print(f'lam={args.lam} local_lam={args.local_lam} '
          f'meeti_weight={args.meeti_weight} use_ptb={args.use_ptb} '
          f'epochs={args.epochs}', flush=True)

    model, beat_probe, rec_probe = build_full_model(
        args.lr, args.lam, args.local_lam, target_dim=int(kd_target.shape[1]))
    print(f'full params: {model.count_params():,}', flush=True)

    history = {
        'train_total_loss': [], 'train_head_b_loss': [],
        'train_head_r_loss': [], 'train_local_mse': [],
        'val_beat_auc': [], 'val_record_auc': [],
        'real_mean': [], 'real_frac_gt_0.5': [],
    }
    best_beat_auc, best_record_auc = -1.0, -1.0
    best_beat_epoch, best_record_epoch = -1, -1
    best_balanced_score, best_balanced_epoch = -1.0, -1

    def save_ckpt(prefix):
        beat_probe.save(str(RUN / f'{prefix}__beat.keras'))
        rec_probe.save(str(RUN / f'{prefix}__record.keras'))

    for epoch in range(args.epochs):
        b_perm = rng.permutation(n_hard)
        r_perm = rng.permutation(n_rec)
        k_perm = rng.permutation(n_kd)
        b_idx = pad_indices(b_perm, hard_total)
        r_idx = pad_indices(r_perm, rec_total)
        k_idx = pad_indices(k_perm, kd_total)
        wb_pad = np.where(yb[b_idx] == 0, 2.0, 1.0).astype(np.float32)
        wr_pad = rec_w[r_idx].copy()
        wk_pad = np.ones(kd_total, dtype=np.float32)
        if hard_pad > 0:
            wb_pad[-hard_pad:] = 0.0
        if rec_pad > 0:
            wr_pad[-rec_pad:] = 0.0
        if kd_pad > 0:
            wk_pad[-kd_pad:] = 0.0
        if epoch < args.warmup_epochs:
            wk_pad[:] = 0.0

        losses = []
        for step in range(rec_steps):
            s_b = step * args.hard_batch
            e_b = s_b + args.hard_batch
            s_r = step * rec_bs
            e_r = s_r + rec_bs
            s_k = step * kd_bs
            e_k = s_k + kd_bs
            loss = model.train_on_batch(
                [xb[b_idx[s_b:e_b]], rec_x[r_idx[s_r:e_r]],
                 rec_mask[r_idx[s_r:e_r]], kd_x[k_idx[s_k:e_k]]],
                [yb[b_idx[s_b:e_b]], rec_y[r_idx[s_r:e_r]],
                 kd_target[k_idx[s_k:e_k]]],
                sample_weight=[wb_pad[s_b:e_b], wr_pad[s_r:e_r],
                               wk_pad[s_k:e_k]],
            )
            losses.append(np.asarray(loss, dtype=np.float64))

        losses = np.stack(losses).mean(axis=0)
        history['train_total_loss'].append(float(losses[0]))
        history['train_head_b_loss'].append(float(losses[1]))
        history['train_head_r_loss'].append(float(losses[2]))
        history['train_local_mse'].append(float(losses[3]))

        pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
        beat_auc = float(roc_auc_score(yv, pb))
        pr = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
        rec_auc = float(roc_auc_score(rec_yv, pr))
        pr_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
        real_mean = float(pr_real.mean())
        real_frac = float((pr_real > 0.5).mean())

        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['real_mean'].append(real_mean)
        history['real_frac_gt_0.5'].append(real_frac)

        if beat_auc > best_beat_auc:
            best_beat_auc, best_beat_epoch = beat_auc, epoch + 1
            save_ckpt('ckpt_best_beat')
        if rec_auc > best_record_auc:
            best_record_auc, best_record_epoch = rec_auc, epoch + 1
            save_ckpt('ckpt_best_record')
        balanced = min(beat_auc, rec_auc)
        if balanced > best_balanced_score:
            best_balanced_score, best_balanced_epoch = balanced, epoch + 1
            save_ckpt('ckpt_balanced')

        print(f'epoch {epoch + 1}/{args.epochs} '
              f'loss={losses[0]:.4f} b={losses[1]:.4f} r={losses[2]:.4f} '
              f'kd={losses[3]:.4f} '
              f'beatAUC={beat_auc:.4f} recAUC={rec_auc:.4f} '
              f'realMean={real_mean:.5f} realFrac={real_frac:.4f}', flush=True)

    elapsed = time.time() - t0
    disk_after = check_disk_redline()

    pb_final = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_final = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
    pr_real_final = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
    final_beat_auc = float(roc_auc_score(yv, pb_final))
    final_record_auc = float(roc_auc_score(rec_yv, pr_final))

    save_error = None
    try:
        model.save_weights(str(RUN / 'full.weights.h5'))
        beat_probe.save(str(RUN / 'final_beat.keras'))
        rec_probe.save(str(RUN / 'final_record.keras'))
    except Exception as exc:
        save_error = f'{type(exc).__name__}: {exc}'

    np.savez_compressed(
        RUN / 'predictions.npz',
        val_beat_p=pb_final, val_beat_y=yv,
        val_record_p=pr_final, val_record_y=rec_yv,
        real_p=pr_real_final,
    )

    result = {
        'tag': args.tag,
        'model_params': int(model.count_params()),
        'config': {
            'epochs': args.epochs, 'hard_batch': args.hard_batch,
            'rec_steps': rec_steps, 'rec_batch': rec_bs,
            'kd_batch': kd_bs, 'kd_total': kd_total, 'kd_pad': kd_pad,
            'hard_pad': hard_pad, 'record_pad': rec_pad,
            'lr': args.lr, 'lam': args.lam,
            'local_lam': args.local_lam,
            'local_target_norm': args.normalize_target,
            'kd_beats': str(Path(args.kd_beats)),
            'kd_target': str(Path(args.kd_target)),
            'warmup_epochs': args.warmup_epochs,
            'meeti_weight': args.meeti_weight, 'use_ptb': args.use_ptb,
            'T_beats': T_BEATS,
            'loop': 'v3 combined Adam + loss_weights; hard/record/local each '
                    'scanned once per epoch; real-sample repeat padding with '
                    'zero sample weight; Keras seed 42',
            'distillation_rule': 'MSE(fc1_local_proj(fc1), teacher_local) '
                                 'per beat; no global 5s embedding per beat',
        },
        'data': {
            'hard_train_beats': n_hard, 'hard_val_beats': int(len(xv)),
            'real_holdout_beats': int(len(x_real)),
            'record_train_total': int(n_rec),
            'kd_train_beats': n_kd,
            'sources': source_info,
        },
        'metrics': {
            'beat_val_auc_final': final_beat_auc,
            'record_val_auc_final': final_record_auc,
            'best_beat_val_auc': best_beat_auc,
            'best_beat_epoch': best_beat_epoch,
            'best_record_val_auc': best_record_auc,
            'best_record_epoch': best_record_epoch,
            'best_balanced_min_auc': best_balanced_score,
            'best_balanced_epoch': best_balanced_epoch,
            'real_mean': float(pr_real_final.mean()),
            'real_frac_gt_0.5': float((pr_real_final > 0.5).mean()),
        },
        'history': history,
        'elapsed_s': round(elapsed, 1),
        'disk_free_gb': disk_after,
        'disk_before_gb': disk_before,
        'save_error': save_error,
    }
    with open(RUN / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result['metrics'], indent=2), flush=True)
    print(f'elapsed={elapsed:.1f}s outputs -> {RUN}', flush=True)


if __name__ == '__main__':
    main()
