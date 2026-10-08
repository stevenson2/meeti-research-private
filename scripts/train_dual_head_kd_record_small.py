#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M2: 在 M1 双头模型上, 仅给 Head_R 增加 ECG-FM 全局嵌入蒸馏 (MSE)。

teacher_emb_768.npy: 每条 MEETI 记录 2 个 5s 段, 按 processed_manifest 顺序排列。
本脚本取每条记录 2 个全局嵌入的均值作为记录级教师目标。
教师投影从 Head_R 注意力池化后的 32 维记录嵌入 -> 768 维。
教师 MSE 梯度只经过 Head_R 分支与共享骨干; 数学上与 Head_B 分类 Dense 断开。

CE 部分与 M1 完全同口径: Head_B 硬标签 + Head_R MEETI subject-train 弱标签,
10 epochs, batch 64, lam=0.5, meeti_weight=0.2。
输出: /home/devcontainers/meeti/runs/dual_head_kd_record_small/
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from tensorflow.keras import Model, layers

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402

NPZ = PROJ / 'pc_tools/ecg_dl/models/deploy_match/distill_pilot_teacher_targets.npz'
MEETI_DIR = Path('/home/devcontainers/meeti/data/processed_small')
T_BEATS = 10


@tf.keras.utils.register_keras_serializable(package='DualHeadECG')
class MaskedAttentionPooling(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, name='attn_score')

    def call(self, inputs):
        emb, mask = inputs
        score = self.score_dense(emb)
        mask = tf.cast(mask, score.dtype)
        score = tf.where(mask > 0.5, score, tf.fill(tf.shape(score), -1e9))
        att = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(att * emb, axis=1)


def load_hard_beats():
    d = np.load(NPZ)
    x = np.asarray(d['x_train_perm'], dtype=np.float32)[..., None]
    y = np.asarray(d['y_train'], dtype=np.int32)
    xv = np.asarray(d['x_val'], dtype=np.float32)[..., None]
    yv = np.asarray(d['y_val'], dtype=np.int32)
    xr = np.asarray(d['x_real_holdout'], dtype=np.float32)[..., None]
    return x, y, xv, yv, xr


def record_windows(beats, positions, n_beats_list, y_list):
    n = len(n_beats_list)
    x = np.zeros((n, T_BEATS, beats.shape[1]), dtype=np.float32)
    mask = np.zeros((n, T_BEATS, 1), dtype=np.float32)
    for i, nb in enumerate(n_beats_list):
        nb = int(nb)
        take = min(nb, T_BEATS)
        x[i, :take] = beats[positions[i]:positions[i] + take]
        mask[i, :take] = 1.0
    return x, mask, np.asarray(y_list, dtype=np.int32)


def load_meeti_data():
    """与 M1 相同的 MEETI subject 级 80/20 划分; 同时返回教师记录级目标。"""
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)

    def to_y(cls):
        return 1 if cls == 'abnormal' else (0 if cls == 'normal' else -1)

    y_raw = manifest['report_class'].apply(to_y).to_numpy()
    keep = y_raw >= 0
    orig_idx = np.where(keep)[0]
    manifest = manifest[keep].reset_index(drop=True)
    y_raw = y_raw[keep]
    n_beats = manifest['n_beats'].astype(int).to_numpy()
    assert int(n_beats.sum()) == len(beats)
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]]).astype(np.int64)
    x_all, mask_all, y_all = record_windows(beats, positions, n_beats, y_raw)

    emb = np.load(MEETI_DIR / 'teacher_emb_768.npy').astype(np.float32)
    assert len(emb) == 2 * len(manifest), (len(emb), len(manifest))
    teacher_rec = (emb[2 * orig_idx] + emb[2 * orig_idx + 1]) * 0.5

    subjects = manifest['subject_id'].to_numpy()
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), dtype=np.int32)
    for yi, si in zip(y_all, subj_idx):
        subj_y[si] = max(subj_y[si], yi)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(sss.split(np.arange(len(uniq)), subj_y))
    tr_mask = np.isin(subj_idx, tr_idx)
    va_mask = np.isin(subj_idx, va_idx)

    return {
        'x': x_all[tr_mask], 'mask': mask_all[tr_mask], 'y': y_all[tr_mask],
        'teacher': teacher_rec[tr_mask],
        'w': np.full(int(tr_mask.sum()), 0.2, dtype=np.float32),
        'x_val': x_all[va_mask], 'mask_val': mask_all[va_mask],
        'y_val': y_all[va_mask], 'teacher_val': teacher_rec[va_mask],
        'n_train': int(tr_mask.sum()), 'n_val': int(va_mask.sum()),
        'n_subjects': int(len(uniq)),
        'train_label_dist': np.bincount(y_all[tr_mask], minlength=2).tolist(),
        'val_label_dist': np.bincount(y_all[va_mask], minlength=2).tolist(),
        'teacher_stats': {
            'n': int(len(teacher_rec)), 'dim': int(teacher_rec.shape[1]),
            'mean': float(teacher_rec.mean()), 'std': float(teacher_rec.std()),
            'row_l2_mean': float(np.linalg.norm(teacher_rec, axis=1).mean()),
        },
    }


def check_disk_redline(min_gb=10.0):
    info = {}
    for p in ['/mnt/c', '/mnt/d']:
        free = shutil.disk_usage(p).free / (1024 ** 3)
        info[p] = round(free, 1)
        if free < min_gb:
            raise SystemExit(f'磁盘红线: {p} 剩余 {free:.1f}GB < {min_gb}GB, 中止')
    return info


def build_model(lr, lam, kd_lam):
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
    encoder = Model(base.input, base.get_layer('fc1').output, name='beat_encoder')

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

    # 训练期教师投影 (不部署到 ESP32)
    kd_h = layers.Dense(256, activation='relu', name='kd_proj1')(rec_pool)
    teacher_proj = layers.Dense(768, name='teacher_proj')(kd_h)

    model = Model([beat_in, rec_in, rec_mask_in],
                  [head_b, head_r, teacher_proj], name='dual_head_kd_record')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss={
            'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
            'head_r': tf.keras.losses.SparseCategoricalCrossentropy(),
            'teacher_proj': tf.keras.losses.MeanSquaredError(),
        },
        loss_weights={'head_b': 1.0, 'head_r': lam, 'teacher_proj': kd_lam},
    )
    beat_probe = Model(model.inputs[0], model.outputs[0], name='beat_probe')
    rec_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[1],
                      name='record_probe')
    teacher_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[2],
                          name='teacher_probe')
    return model, beat_probe, rec_probe, teacher_probe


def predict_rec(probe, x, mask, batch_size=256):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(probe.predict([x[i:i + batch_size], mask[i:i + batch_size]],
                                 batch_size=batch_size, verbose=0))
    return np.concatenate(out, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.5)
    ap.add_argument('--kd-lam', type=float, default=0.001)
    ap.add_argument('--tag', default='dual_head_kd_record_small')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_kd_record_small')
    args = ap.parse_args()

    RUN = Path(args.run_dir)
    RUN.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    disk_before = check_disk_redline()

    gpus = tf.config.list_physical_devices('GPU')
    print('gpus', gpus, flush=True)
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    tf.random.set_seed(42)
    rng = np.random.default_rng(42)

    xb, yb, xv, yv, x_real = load_hard_beats()
    data = load_meeti_data()
    rec_x, rec_mask, rec_y, rec_w = data['x'], data['mask'], data['y'], data['w']
    rec_teacher = data['teacher']
    rec_xv, rec_mv, rec_yv, rec_tv = (data['x_val'], data['mask_val'],
                                      data['y_val'], data['teacher_val'])
    wb = np.where(yb == 0, 2.0, 1.0).astype(np.float32)
    n_hard, n_rec = len(xb), len(rec_x)
    n_steps = math.ceil(n_hard / args.batch_size)
    print(f'hard beats={n_hard} val={len(xv)} real={len(x_real)}', flush=True)
    print(f'meeti records train={n_rec} val={len(rec_xv)} '
          f'subjects={data["n_subjects"]}', flush=True)
    print('teacher', data['teacher_stats'], flush=True)
    print(f'lam={args.lam} kd_lam={args.kd_lam} epochs={args.epochs} '
          f'batch={args.batch_size}', flush=True)

    model, beat_probe, rec_probe, teacher_probe = build_model(
        args.lr, args.lam, args.kd_lam)
    print(f'model params: {model.count_params():,}', flush=True)

    history = {
        'train_total_loss': [], 'train_head_b_loss': [],
        'train_head_r_loss': [], 'train_kd_loss': [],
        'val_beat_auc': [], 'val_record_auc': [], 'val_kd_mse': [],
        'real_mean': [], 'real_frac_gt_0.5': [],
    }

    for epoch in range(args.epochs):
        b_perm = rng.permutation(n_hard)
        r_perm = rng.permutation(n_rec)
        losses = []
        for step in range(n_steps):
            s = step * args.batch_size
            e = min(s + args.batch_size, n_hard)
            b_idx = b_perm[s:e]
            bs_actual = e - s
            r_s = (step * args.batch_size) % n_rec
            if r_s + bs_actual <= n_rec:
                r_idx = r_perm[r_s:r_s + bs_actual]
            else:
                r_idx = np.concatenate(
                    [r_perm[r_s:], r_perm[:r_s + bs_actual - n_rec]])

            step_loss = model.train_on_batch(
                [xb[b_idx], rec_x[r_idx], rec_mask[r_idx]],
                [yb[b_idx], rec_y[r_idx], rec_teacher[r_idx]],
                sample_weight=[wb[b_idx], rec_w[r_idx],
                               np.ones(bs_actual, dtype=np.float32)],
            )
            losses.append(np.asarray(step_loss, dtype=np.float64))

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

        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['val_kd_mse'].append(kd_mse)
        history['real_mean'].append(float(pr_real.mean()))
        history['real_frac_gt_0.5'].append(float((pr_real > 0.5).mean()))
        print(f'epoch {epoch + 1}/{args.epochs} '
              f'loss={losses[0]:.4f} b={losses[1]:.4f} r={losses[2]:.4f} '
              f'kd={losses[3]:.4f} beatAUC={beat_auc:.4f} '
              f'recAUC={rec_auc:.4f} kdMSE={kd_mse:.2f} '
              f'realMean={history["real_mean"][-1]:.5f}', flush=True)

    elapsed = time.time() - t0
    disk_after = check_disk_redline()

    pb_final = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_final = predict_rec(rec_probe, rec_xv, rec_mv)[:, 1]
    pt_final = predict_rec(teacher_probe, rec_xv, rec_mv)
    pr_real_final = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]

    save_error = None
    try:
        model.save(str(RUN / 'model.keras'))
        beat_probe.save(str(RUN / 'beat_probe.keras'))
        rec_probe.save(str(RUN / 'record_probe.keras'))
    except Exception as exc:
        save_error = f'{type(exc).__name__}: {exc}'
        try:
            model.save_weights(str(RUN / 'model.weights.h5'))
        except Exception as exc2:
            save_error += f' | weights: {type(exc2).__name__}: {exc2}'

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
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'lam': args.lam, 'kd_lam': args.kd_lam,
            'T_beats': T_BEATS,
            'teacher': 'ECG-FM 2 x 5s global emb mean per record',
            'teacher_projection': 'Dense(256,relu)->Dense(768)',
            'head_b_data': 'distill_pilot_teacher_targets.npz x_train_perm/y_train',
            'head_r_train': 'MEETI subject-train records',
            'head_r_val': 'MEETI subject-holdout records',
        },
        'data': {
            'hard_train_beats': n_hard, 'hard_val_beats': int(len(xv)),
            'real_holdout_beats': int(len(x_real)),
            'meeti_train_records': data['n_train'],
            'meeti_val_records': data['n_val'],
            'meeti_subjects': data['n_subjects'],
            'meeti_train_label_dist': data['train_label_dist'],
            'meeti_val_label_dist': data['val_label_dist'],
            'teacher_stats': data['teacher_stats'],
        },
        'metrics': {
            'beat_val_auc': float(roc_auc_score(yv, pb_final)),
            'record_val_auc': float(roc_auc_score(rec_yv, pr_final)),
            'val_kd_mse': float(np.mean((pt_final - rec_tv) ** 2)),
            'real_mean': float(pr_real_final.mean()),
            'real_frac_gt_0.5': float((pr_real_final > 0.5).mean()),
            'm1_reference_record_val_auc': 0.8184650240725007,
            'record_auc_delta_vs_m1':
                float(roc_auc_score(rec_yv, pr_final)) - 0.8184650240725007,
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
