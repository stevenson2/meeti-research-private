#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1 消融版: 双头共享骨干, 每个 epoch 拍级/记录级数据集各恰好扫一遍。

与 train_dual_head_small.py 的差异:
  1. 不再把记录 batch 循环塞进硬拍 batch (原版 MEETI 每 epoch 约扫 3.75 遍);
  2. 每个 epoch 随机交错 93 个硬拍 batch + N 个记录 batch;
  3. 记录头损失通过 scaled loss 施加 lambda (样本权重只表示任务内样本权重);
  4. 每个 epoch 保存 best-beat-AUC / best-record-AUC 的 full model 权重。

数据口径与 M1 完全一致: Head_B 只吃 public hard beats, Head_R 只吃 MEETI
subject-train 记录 (可选 --use-ptb 追加 PTB train 患者记录级金标准)。
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


def build_models(lr, lam):
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

    full = Model([beat_in, rec_in, rec_mask_in], [head_b, head_r],
                 name='dual_head_v2')
    beat_model = Model(full.inputs[0], full.outputs[0], name='dual_head_v2_beat')
    rec_model = Model([full.inputs[1], full.inputs[2]], full.outputs[1],
                      name='dual_head_v2_record')

    beat_model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy())

    def scaled_ce(y_true, y_pred):
        return lam * tf.keras.losses.sparse_categorical_crossentropy(
            y_true, y_pred)

    rec_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=scaled_ce)
    return full, beat_model, rec_model


def predict_rec(probe, x, mask, batch_size=256):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(probe.predict([x[i:i + batch_size], mask[i:i + batch_size]],
                                 batch_size=batch_size, verbose=0))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--meeti-weight', type=float, default=0.2)
    ap.add_argument('--use-ptb', action='store_true')
    ap.add_argument('--tag', default='dual_head_v2')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_v2')
    args = ap.parse_args()

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    disk_before = check_disk_redline()

    gpus = tf.config.list_physical_devices('GPU')
    print('gpus', gpus, flush=True)
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.keras.utils.set_random_seed(42)  # 固定 Keras 初始化 + tf 全局随机
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

    wb = np.where(yb == 0, 2.0, 1.0).astype(np.float32)
    n_hard, n_rec = len(xb), len(rec_x)
    hard_steps = math.ceil(n_hard / args.batch_size)
    rec_steps = math.ceil(n_rec / args.batch_size)
    print(f'hard train={n_hard} ({hard_steps} steps/epoch) val={len(xv)} '
          f'real={len(x_real)}', flush=True)
    print(f'record train={n_rec} ({rec_steps} steps/epoch) '
          f'val={len(rec_xv)}', flush=True)
    print(f'lam={args.lam} meeti_weight={args.meeti_weight} '
          f'use_ptb={args.use_ptb} epochs={args.epochs} '
          f'batch={args.batch_size}', flush=True)

    full, beat_model, rec_model = build_models(args.lr, args.lam)
    print(f'full params: {full.count_params():,}', flush=True)

    history = {
        'train_beat_loss': [], 'train_record_loss': [],
        'val_beat_auc': [], 'val_record_auc': [],
        'real_mean': [], 'real_frac_gt_0.5': [],
    }
    best_beat_auc, best_record_auc = -1.0, -1.0
    best_beat_epoch, best_record_epoch = -1, -1

    for epoch in range(args.epochs):
        b_perm = rng.permutation(n_hard)
        r_perm = rng.permutation(n_rec)
        tasks = ['b'] * hard_steps + ['r'] * rec_steps
        rng.shuffle(tasks)
        beat_losses, rec_losses = [], []
        bi = ri = 0
        for task in tasks:
            if task == 'b':
                s = bi * args.batch_size
                e = min(s + args.batch_size, n_hard)
                idx = b_perm[s:e]
                loss = beat_model.train_on_batch(
                    xb[idx], yb[idx], sample_weight=wb[idx])
                beat_losses.append(float(loss))
                bi += 1
            else:
                s = ri * args.batch_size
                e = min(s + args.batch_size, n_rec)
                idx = r_perm[s:e]
                loss = rec_model.train_on_batch(
                    [rec_x[idx], rec_mask[idx]], rec_y[idx],
                    sample_weight=rec_w[idx])
                rec_losses.append(float(loss))
                ri += 1

        history['train_beat_loss'].append(float(np.mean(beat_losses)))
        history['train_record_loss'].append(float(np.mean(rec_losses)))

        pb = beat_model.predict(xv, batch_size=256, verbose=0)[:, 1]
        beat_auc = float(roc_auc_score(yv, pb))
        pr = predict_rec(rec_model, rec_xv, rec_mv)[:, 1]
        rec_auc = float(roc_auc_score(rec_yv, pr))
        pr_real = beat_model.predict(x_real, batch_size=64, verbose=0)[:, 1]
        real_mean = float(pr_real.mean())
        real_frac = float((pr_real > 0.5).mean())

        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['real_mean'].append(real_mean)
        history['real_frac_gt_0.5'].append(real_frac)

        if beat_auc > best_beat_auc:
            best_beat_auc = beat_auc
            best_beat_epoch = epoch + 1
            full.save(str(RUN / 'best_beat.keras'))
        if rec_auc > best_record_auc:
            best_record_auc = rec_auc
            best_record_epoch = epoch + 1
            full.save(str(RUN / 'best_record.keras'))

        print(f'epoch {epoch + 1}/{args.epochs} '
              f'bLoss={history["train_beat_loss"][-1]:.4f} '
              f'rLoss={history["train_record_loss"][-1]:.4f} '
              f'beatAUC={beat_auc:.4f} recAUC={rec_auc:.4f} '
              f'realMean={real_mean:.5f} realFrac={real_frac:.4f}', flush=True)

    elapsed = time.time() - t0
    disk_after = check_disk_redline()

    pb_final = beat_model.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_final = predict_rec(rec_model, rec_xv, rec_mv)[:, 1]
    pr_real_final = beat_model.predict(x_real, batch_size=64, verbose=0)[:, 1]
    final_beat_auc = float(roc_auc_score(yv, pb_final))
    final_record_auc = float(roc_auc_score(rec_yv, pr_final))

    save_error = None
    try:
        full.save(str(RUN / 'final.keras'))
        beat_model.save(str(RUN / 'final_beat.keras'))
        rec_model.save(str(RUN / 'final_record.keras'))
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
        'model_params': int(full.count_params()),
        'config': {
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'lam': args.lam,
            'meeti_weight': args.meeti_weight, 'use_ptb': args.use_ptb,
            'T_beats': T_BEATS,
            'loop': 'random-interleaved; beat dataset and record dataset '
                    'each scanned exactly once per epoch',
            'hard_steps_per_epoch': hard_steps,
            'record_steps_per_epoch': rec_steps,
            'record_window': 'first T beats, right zero-pad',
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
