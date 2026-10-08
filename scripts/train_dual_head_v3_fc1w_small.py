#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Route-2 capacity ablation: dual-head v3 with wider fc1 (64 -> 128/256).

Identical protocol to train_dual_head_v3_small.py (single combined Adam,
loss_weights head_b=1.0/head_r=lam, per-epoch full scan of both datasets,
zero-weight repeat padding, seed 42, +PTB records). ONLY change: fc1 width,
via a local copy of the encoder builder (PROJ's resnet_lite_1d hardcodes 64).
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
from models.resnet_lite_1d import res_block  # noqa: E402

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, check_disk_redline, load_hard_beats,
    load_meeti_records, load_ptb_train_records,
)

T_BEATS = 10


def build_encoder_large(input_shape, fc1_units):
    """Local copy of build_ecg_resnet_lite_large with configurable fc1 width."""
    inputs = layers.Input(shape=input_shape, name='ecg_input')
    x = layers.Conv1D(16, 7, strides=2, padding='same', use_bias=False,
                      name='stem')(inputs)
    x = layers.BatchNormalization(name='stem_bn')(x)
    x = layers.ReLU(name='stem_rl')(x)
    filters = (16, 32, 64, 128)
    blocks_per_stage = (2, 3, 3, 1)
    kernel_sizes = (7, 5, 3, 3)
    strides = (1, 2, 2, 1)
    bid = 0
    for si in range(len(filters)):
        f, k = filters[si], kernel_sizes[si]
        for blk in range(blocks_per_stage[si]):
            s = strides[si] if blk == 0 else 1
            x = res_block(x, filters=f, kernel_size=k, stride=s, block_id=bid)
            bid += 1
    x = layers.GlobalAveragePooling1D(name='gap')(x)
    x = layers.Dense(fc1_units, activation='relu', name='fc1')(x)
    x = layers.Dropout(0.4, name='do')(x)
    out = layers.Dense(2, activation='softmax', name='out')(x)
    return Model(inputs, out, name=f'ecg_resnet_lite_fc1{fc1_units}')


def build_full_model(lr, lam, fc1_units):
    base = build_encoder_large((250, 1), fc1_units)
    encoder = Model(base.input, base.get_layer('fc1').output,
                    name='beat_encoder')

    beat_in = layers.Input((250, 1), name='beat_input')
    beat_emb = encoder(beat_in)
    head_b = layers.Dense(2, activation='softmax', name='head_b')(beat_emb)

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
                  name=f'dual_head_v3_fc1{fc1_units}')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
              'head_r': tf.keras.losses.SparseCategoricalCrossentropy()},
        loss_weights={'head_b': 1.0, 'head_r': lam},
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
    ap.add_argument('--fc1-units', type=int, default=128)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--hard-batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.2)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--use-ptb', action='store_true', default=True)
    ap.add_argument('--run-dir', default=None)
    args = ap.parse_args()
    if args.run_dir is None:
        args.run_dir = (f'/home/devcontainers/meeti/runs/'
                        f'dual_head_v3_fc1{args.fc1_units}_lam0.2_ptb')

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    check_disk_redline()

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=args.meeti_weight))
    rec_x, rec_mask, rec_y, rec_w = meeti['x'], meeti['mask'], meeti['y'], meeti['w']
    rec_xv, rec_mv, rec_yv = meeti['x_val'], meeti['mask_val'], meeti['y_val']
    ptb = load_ptb_train_records()
    rec_x = np.concatenate([rec_x, ptb['x']], axis=0)
    rec_mask = np.concatenate([rec_mask, ptb['mask']], axis=0)
    rec_y = np.concatenate([rec_y, ptb['y']], axis=0)
    rec_w = np.concatenate([rec_w, ptb['w']], axis=0)

    n_hard, n_rec = len(xb), len(rec_x)
    rec_steps = math.ceil(n_hard / args.hard_batch)
    rec_bs = math.ceil(n_rec / rec_steps)
    hard_total = rec_steps * args.hard_batch
    rec_total = rec_steps * rec_bs
    print(f'fc1={args.fc1_units} hard={n_hard} rec={n_rec} '
          f'steps={rec_steps}', flush=True)

    model, beat_probe, rec_probe = build_full_model(args.lr, args.lam,
                                                    args.fc1_units)
    print(f'full params: {model.count_params():,}', flush=True)

    history = {'val_beat_auc': [], 'val_record_auc': [],
               'real_mean': [], 'real_frac_gt_0.5': []}
    best = {'beat': -1.0, 'record': -1.0, 'balanced': -1.0}

    def save_ckpt(prefix):
        beat_probe.save(str(RUN / f'{prefix}__beat.keras'))
        rec_probe.save(str(RUN / f'{prefix}__record.keras'))

    for epoch in range(args.epochs):
        b_idx = pad_indices(rng.permutation(n_hard), hard_total)
        r_idx = pad_indices(rng.permutation(n_rec), rec_total)
        wb = np.where(yb[b_idx] == 0, 2.0, 1.0).astype(np.float32)
        wr = rec_w[r_idx].copy()
        wb[-(hard_total - n_hard):] = 0.0
        wr[-(rec_total - n_rec):] = 0.0
        for step in range(rec_steps):
            s_b, e_b = step * args.hard_batch, (step + 1) * args.hard_batch
            s_r, e_r = step * rec_bs, (step + 1) * rec_bs
            model.train_on_batch(
                [xb[b_idx[s_b:e_b]], rec_x[r_idx[s_r:e_r]],
                 rec_mask[r_idx[s_r:e_r]]],
                [yb[b_idx[s_b:e_b]], rec_y[r_idx[s_r:e_r]]],
                sample_weight=[wb[s_b:e_b], wr[s_r:e_r]])

        pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
        beat_auc = float(roc_auc_score(yv, pb))
        pr = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
        rec_auc = float(roc_auc_score(rec_yv, pr))
        pr_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['real_mean'].append(float(pr_real.mean()))
        history['real_frac_gt_0.5'].append(float((pr_real > 0.5).mean()))
        if beat_auc > best['beat']:
            best['beat'] = beat_auc
            save_ckpt('ckpt_best_beat')
        if rec_auc > best['record']:
            best['record'] = rec_auc
            save_ckpt('ckpt_best_record')
        balanced = min(beat_auc, rec_auc)
        if balanced > best['balanced']:
            best['balanced'] = balanced
            best['balanced_epoch'] = epoch + 1
            save_ckpt('ckpt_balanced')
        print(f'epoch {epoch + 1}/{args.epochs} beatAUC={beat_auc:.4f} '
              f'recAUC={rec_auc:.4f} realMean={pr_real.mean():.5f}',
              flush=True)

    model.save_weights(str(RUN / 'full.weights.h5'))
    result = {
        'tag': f'dual_head_v3_fc1{args.fc1_units}_lam0.2_ptb',
        'config': vars(args) | {'t_beats': T_BEATS},
        'model_params': int(model.count_params()),
        'metrics': {
            'best_beat_val_auc': best['beat'],
            'best_record_val_auc': best['record'],
            'best_balanced_min_auc': best['balanced'],
            'best_balanced_epoch': best.get('balanced_epoch'),
            'final_real_mean': history['real_mean'][-1],
        },
        'history': history,
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(RUN / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result['metrics'], indent=2), flush=True)


if __name__ == '__main__':
    main()
