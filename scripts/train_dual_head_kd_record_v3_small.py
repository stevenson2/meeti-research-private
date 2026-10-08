#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""受控 KD A/B 的 KD 分支: 与 train_dual_head_v3_small.py 同配方 + Head_R 教师投影。

与 no-KD v3 的差异只有:
  1. Head_R 注意力池化后增加训练期 Dense(256,relu)->Dense(768) 教师投影;
  2. total loss 增加 kd_lam * MSE(projection, ECG-FM 2段全局嵌入均值);
  3. teacher 梯度在图上只连接 Head_R 分支与共享骨干, 不连接 Head_B Dense。
其余完全一致: 同一 combined Adam, 同一 balanced padding loop, 同一 Keras seed,
同一 hard_bs=128 / rec_bs 推导, 同一 λ=0.2, MEETI only, 20 epochs。
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
)
from train_dual_head_kd_record_small import load_meeti_data  # noqa: E402

T_BEATS = 10


def build_model(lr, lam, kd_lam):
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
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

    kd_h = layers.Dense(256, activation='relu', name='kd_proj1')(rec_pool)
    teacher_proj = layers.Dense(768, name='teacher_proj')(kd_h)

    model = Model([beat_in, rec_in, rec_mask_in],
                  [head_b, head_r, teacher_proj], name='dual_head_kd_v3')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
              'head_r': tf.keras.losses.SparseCategoricalCrossentropy(),
              'teacher_proj': tf.keras.losses.MeanSquaredError()},
        loss_weights={'head_b': 1.0, 'head_r': lam,
                      'teacher_proj': kd_lam},
    )
    beat_probe = Model(model.inputs[0], model.outputs[0], name='beat_probe')
    rec_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[1],
                      name='record_probe')
    teacher_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[2],
                          name='teacher_probe')
    return model, beat_probe, rec_probe, teacher_probe


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
    ap.add_argument('--kd-lam', type=float, default=0.001)
    ap.add_argument('--tag', default='dual_head_kd_v3')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_kd_v3')
    args = ap.parse_args()

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    disk_before = check_disk_redline()

    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(42)  # 与 no-KD v3 完全同种子
    rng = np.random.default_rng(42)

    xb, yb, xv, yv, x_real = load_hard_beats()
    data = load_meeti_data()
    rec_x, rec_mask, rec_y = data['x'], data['mask'], data['y']
    rec_teacher = data['teacher']
    rec_w = np.full(len(rec_x), 0.2, dtype=np.float32)
    rec_xv, rec_mv, rec_yv, rec_tv = (data['x_val'], data['mask_val'],
                                      data['y_val'], data['teacher_val'])

    n_hard, n_rec = len(xb), len(rec_x)
    rec_steps = math.ceil(n_hard / args.hard_batch)
    rec_bs = math.ceil(n_rec / rec_steps)
    hard_total = rec_steps * args.hard_batch
    rec_total = rec_steps * rec_bs
    hard_pad, rec_pad = hard_total - n_hard, rec_total - n_rec
    print(f'hard={n_hard} hard_bs={args.hard_batch} steps={rec_steps} '
          f'pad={hard_pad} val={len(xv)} real={len(x_real)}', flush=True)
    print(f'record={n_rec} rec_bs={rec_bs} total={rec_total} pad={rec_pad} '
          f'val={len(rec_xv)}', flush=True)
    print(f'lam={args.lam} kd_lam={args.kd_lam} epochs={args.epochs}',
          flush=True)

    model, beat_probe, rec_probe, teacher_probe = build_model(
        args.lr, args.lam, args.kd_lam)
    print(f'full params: {model.count_params():,}', flush=True)

    history = {
        'train_total_loss': [], 'train_head_b_loss': [],
        'train_head_r_loss': [], 'train_kd_loss': [],
        'val_beat_auc': [], 'val_record_auc': [], 'val_kd_mse': [],
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
        wt_pad = (wr_pad > 0).astype(np.float32)
        if hard_pad > 0:
            wb_pad[-hard_pad:] = 0.0
        if rec_pad > 0:
            wr_pad[-rec_pad:] = 0.0
            wt_pad[-rec_pad:] = 0.0

        losses = []
        for step in range(rec_steps):
            s_b = step * args.hard_batch
            e_b = s_b + args.hard_batch
            s_r = step * rec_bs
            e_r = s_r + rec_bs
            loss = model.train_on_batch(
                [xb[b_idx[s_b:e_b]], rec_x[r_idx[s_r:e_r]],
                 rec_mask[r_idx[s_r:e_r]]],
                [yb[b_idx[s_b:e_b]], rec_y[r_idx[s_r:e_r]],
                 rec_teacher[r_idx[s_r:e_r]]],
                sample_weight=[wb_pad[s_b:e_b], wr_pad[s_r:e_r],
                               wt_pad[s_r:e_r]],
            )
            losses.append(np.asarray(loss, dtype=np.float64))

        losses = np.stack(losses).mean(axis=0)
        history['train_total_loss'].append(float(losses[0]))
        history['train_head_b_loss'].append(float(losses[1]))
        history['train_head_r_loss'].append(float(losses[2]))
        history['train_kd_loss'].append(float(losses[3]))

        pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
        beat_auc = float(roc_auc_score(yv, pb))
        pr = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
        rec_auc = float(roc_auc_score(rec_yv, pr))
        pt = predict_rec(teacher_probe, rec_xv, rec_mv)
        kd_mse = float(np.mean((pt - rec_tv) ** 2))
        pr_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
        real_mean = float(pr_real.mean())
        real_frac = float((pr_real > 0.5).mean())

        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['val_kd_mse'].append(kd_mse)
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
              f'kd={losses[3]:.4f} beatAUC={beat_auc:.4f} '
              f'recAUC={rec_auc:.4f} kdMSE={kd_mse:.2f} '
              f'realMean={real_mean:.5f}', flush=True)

    elapsed = time.time() - t0
    disk_after = check_disk_redline()

    pb_final = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_final = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
    pt_final = predict_rec(teacher_probe, rec_xv, rec_mv)
    pr_real_final = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]

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
        val_teacher_pred=pt_final, val_teacher_y=rec_tv,
        real_p=pr_real_final,
    )

    result = {
        'tag': args.tag,
        'model_params': int(model.count_params()),
        'config': {
            'epochs': args.epochs, 'hard_batch': args.hard_batch,
            'rec_steps': rec_steps, 'rec_batch': rec_bs,
            'hard_pad': hard_pad, 'record_pad': rec_pad,
            'lr': args.lr, 'lam': args.lam, 'kd_lam': args.kd_lam,
            'teacher': 'ECG-FM 2x5s global emb mean per record',
            'seed': 'tf.keras.utils.set_random_seed(42)',
        },
        'data': {
            'hard_train_beats': n_hard, 'hard_val_beats': int(len(xv)),
            'real_holdout_beats': int(len(x_real)),
            'meeti_train_records': data['n_train'],
            'meeti_val_records': data['n_val'],
            'meeti_subjects': data['n_subjects'],
            'teacher_stats': data['teacher_stats'],
        },
        'metrics': {
            'beat_val_auc_final': float(roc_auc_score(yv, pb_final)),
            'record_val_auc_final': float(roc_auc_score(rec_yv, pr_final)),
            'val_kd_mse': float(np.mean((pt_final - rec_tv) ** 2)),
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
