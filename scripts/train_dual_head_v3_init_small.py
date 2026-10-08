#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""STEP 1: v3 dual-head training with clean-baseline initialization.

Same recipe as train_dual_head_v3_small.py:
  - single combined Adam + loss_weights
  - hard_bs=128 -> rec_steps=ceil(n_hard/hard_bs) -> rec_bs=ceil(n_rec/rec_steps)
  - real-sample repeat padding with zero sample weight on padding positions
  - tf.keras.utils.set_random_seed(42) and np.random.default_rng(42)
  - same probe save/eval protocol

New:
  --init-weights PATH    source .h5 checkpoint (clean_baseline_v3/v4/v5 only)
  --init-scope {encoder,fc1,none}
  --freeze-backbone {0,1}

exp7c lineage is rejected by default for clean conclusions.  If an experiment
is intentionally run as an engineering upper bound, pass --allow-leaky; such
results are tagged possible-leakage and must never enter clean conclusions.
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
CLEAN_BASELINE_STEMS = (
    'best_resnet_large_clean_baseline_v3',
    'best_resnet_large_clean_baseline_v4',
    'best_resnet_large_clean_baseline_v5',
)


def build_full_model_from_encoder(lr, lam, encoder):
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
                  name='dual_head_v3_init')
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


def weight_layers_by_name(model):
    return {layer.name: layer for layer in model.layers if layer.weights}


def copy_encoder_init(encoder, source_model, scope):
    """Copy layer weights from a single-head source model into encoder."""
    enc_layers = weight_layers_by_name(encoder)
    src_layers = weight_layers_by_name(source_model)

    copied, diffs = [], {}
    if scope == 'encoder':
        enc_names = list(enc_layers.keys())
        missing = [n for n in enc_names if n not in src_layers]
        extra = [n for n in src_layers if n not in enc_names]
        if missing:
            raise RuntimeError(
                f'init encoder layers missing from source: {missing}')
        # Source is the same single-head architecture; only 'out' is expected
        # outside the fc1-terminated encoder.  Anything else means layer
        # names/architectures do not match and the copy must be rejected.
        if extra != ['out']:
            raise RuntimeError(
                f'source/encoder architecture mismatch: source has extra '
                f'weight layers {extra} outside encoder (expected only ["out"])')
        src_for_encoder = [n for n in src_layers if n in enc_layers]
        if src_for_encoder != enc_names:
            raise RuntimeError(
                f'layer-name/order mismatch: encoder={enc_names}, '
                f'source matching subset={src_for_encoder}')
        targets = enc_names
    elif scope == 'fc1':
        if 'fc1' not in enc_layers:
            raise RuntimeError('encoder has no fc1 layer')
        if 'fc1' not in src_layers:
            raise RuntimeError('source model has no fc1 layer')
        targets = ['fc1']
    else:
        targets = []

    for name in targets:
        dst = enc_layers[name]
        src = src_layers[name]
        sw = src.get_weights()
        dw = dst.get_weights()
        if len(sw) != len(dw):
            raise RuntimeError(
                f'weight count mismatch for {name}: source {len(sw)} vs '
                f'encoder {len(dw)}')
        if [tuple(w.shape) for w in sw] != [tuple(w.shape) for w in dw]:
            raise RuntimeError(
                f'weight shape mismatch for {name}: source '
                f'{[tuple(w.shape) for w in sw]} vs encoder '
                f'{[tuple(w.shape) for w in dw]}')
        dst.set_weights([np.asarray(w, dtype=w.dtype) for w in sw])
        copied.append(name)
        diffs[name] = 0.0
        for a, b in zip(dst.get_weights(), sw):
            if a.size:
                diffs[name] = max(diffs[name],
                                  float(np.max(np.abs(a - b))))
        if diffs[name] != 0.0:
            raise RuntimeError(
                f'weight copy verification failed for {name}: '
                f'max_abs_diff={diffs[name]}')
        print(f'  init-copy verified {name}: max_abs_diff=0', flush=True)

    return copied, diffs


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
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--use-ptb', action='store_true')
    ap.add_argument('--init-weights', default=None)
    ap.add_argument('--init-scope', default='encoder',
                    choices=['encoder', 'fc1', 'none'])
    ap.add_argument('--freeze-backbone', type=int, default=0, choices=[0, 1])
    ap.add_argument('--allow-leaky', action='store_true')
    ap.add_argument('--tag', default='dual_head_v3_init')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_v3_init')
    args = ap.parse_args()

    if args.init_weights is None and args.init_scope != 'none':
        raise SystemExit('--init-scope requires --init-weights '
                         '(or explicitly pass --init-scope none)')

    init_path = None if args.init_weights is None else Path(args.init_weights)
    init_stem = init_path.stem if init_path is not None else ''
    init_is_clean = init_path is not None and any(
        s in init_stem for s in CLEAN_BASELINE_STEMS)
    if init_path is not None and 'exp7c' in init_stem.lower():
        if not args.allow_leaky:
            raise SystemExit(
                'REJECTED: exp7c lineage is possible-leakage and cannot be '
                'used as a clean init.  Only pass --allow-leaky for an '
                'explicit engineering-upper-bound run tagged possible-leakage.')
        init_is_clean = False

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    disk_before = check_disk_redline()

    gpus = tf.config.list_physical_devices('GPU')
    print('gpus', gpus, flush=True)
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(42)  # fixed Keras init + tf global random
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

    # ---- build base/encoder and initialize from source ----------------
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
    encoder = Model(base.input, base.get_layer('fc1').output,
                    name='beat_encoder')

    copied_layers, layer_diffs = [], {}
    if init_path is not None and args.init_scope != 'none':
        print(f'loading init weights from {init_path}', flush=True)
        source_model = tf.keras.models.load_model(init_path, compile=False)
        copied_layers, layer_diffs = copy_encoder_init(
            encoder, source_model, args.init_scope)
        del source_model

    if args.freeze_backbone:
        encoder.trainable = False
        for layer in encoder.layers:
            layer.trainable = False
        print('freeze-backbone: encoder.trainable=False (all encoder layers)',
              flush=True)

    model, beat_probe, rec_probe = build_full_model_from_encoder(
        args.lr, args.lam, encoder)
    trainable_weight_count = len(model.trainable_weights)
    print(f'full params: {model.count_params():,} '
          f'trainable_weight_tensors: {trainable_weight_count}', flush=True)

    n_hard, n_rec = len(xb), len(rec_x)
    rec_steps = math.ceil(n_hard / args.hard_batch)
    rec_bs = math.ceil(n_rec / rec_steps)
    hard_total = rec_steps * args.hard_batch
    rec_total = rec_steps * rec_bs
    hard_pad = hard_total - n_hard
    rec_pad = rec_total - n_rec
    print(f'hard train={n_hard} hard_bs={args.hard_batch} '
          f'rec_steps={rec_steps} pad={hard_pad} val={len(xv)} '
          f'real={len(x_real)}', flush=True)
    print(f'record train={n_rec} rec_bs={rec_bs} total={rec_total} '
          f'pad={rec_pad} val={len(rec_xv)}', flush=True)
    print(f'lam={args.lam} meeti_weight={args.meeti_weight} '
          f'use_ptb={args.use_ptb} epochs={args.epochs} '
          f'init={init_path} scope={args.init_scope} '
          f'freeze_backbone={args.freeze_backbone}', flush=True)

    history = {
        'train_total_loss': [], 'train_head_b_loss': [],
        'train_head_r_loss': [],
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
        b_idx = pad_indices(b_perm, hard_total)
        r_idx = pad_indices(r_perm, rec_total)
        wb_pad = np.where(yb[b_idx] == 0, 2.0, 1.0).astype(np.float32)
        wr_pad = rec_w[r_idx].copy()
        if hard_pad > 0:
            wb_pad[-hard_pad:] = 0.0
        if rec_pad > 0:
            wr_pad[-rec_pad:] = 0.0

        losses = []
        for step in range(rec_steps):
            s_b = step * args.hard_batch
            e_b = s_b + args.hard_batch
            s_r = step * rec_bs
            e_r = s_r + rec_bs
            loss = model.train_on_batch(
                [xb[b_idx[s_b:e_b]], rec_x[r_idx[s_r:e_r]],
                 rec_mask[r_idx[s_r:e_r]]],
                [yb[b_idx[s_b:e_b]], rec_y[r_idx[s_r:e_r]]],
                sample_weight=[wb_pad[s_b:e_b], wr_pad[s_r:e_r]],
            )
            losses.append(np.asarray(loss, dtype=np.float64))

        losses = np.stack(losses).mean(axis=0)
        history['train_total_loss'].append(float(losses[0]))
        history['train_head_b_loss'].append(float(losses[1]))
        history['train_head_r_loss'].append(float(losses[2]))

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
        'trainable_weight_tensors': trainable_weight_count,
        'config': {
            'epochs': args.epochs, 'hard_batch': args.hard_batch,
            'rec_steps': rec_steps, 'rec_batch': rec_bs,
            'hard_pad': hard_pad, 'record_pad': rec_pad,
            'lr': args.lr, 'lam': args.lam,
            'meeti_weight': args.meeti_weight, 'use_ptb': args.use_ptb,
            'init_weights': str(init_path) if init_path is not None else None,
            'init_scope': args.init_scope,
            'init_is_clean': init_is_clean,
            'init_possible_leakage': (init_path is not None
                                      and 'exp7c' in init_stem.lower()),
            'freeze_backbone': bool(args.freeze_backbone),
            'copied_layers': copied_layers,
            'layer_max_abs_diffs': layer_diffs,
            'T_beats': T_BEATS,
            'loop': 'single combined Adam + loss_weights; each dataset '
                    'scanned once per epoch; real-sample repeat padding '
                    'with zero sample weight; Keras seed 42',
        },
        'data': {
            'hard_train_beats': n_hard, 'hard_val_beats': int(len(xv)),
            'real_holdout_beats': int(len(x_real)),
            'record_train_total': int(n_rec),
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
