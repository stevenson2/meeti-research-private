#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1: 共享骨干双头 MIL 学生 (Head_B 拍级 + Head_R 记录级) 小规模训练。

模型:
  input(250,1) -> ResNet-Lite-Large 骨干 -> fc1 Dense(64, relu)
    -> Head_B: Dense(2, softmax)                      # 拍级硬标签
    -> Head_R: TimeDistributed(Dense(32, relu))        # 每条记录 T=10 拍
               -> masked attention pool -> Dense(2, softmax)

数据:
  Head_B 仅吃 distill_pilot_teacher_targets.npz 的 x_train_perm/y_train。
  Head_R 吃 MEETI 记录级弱标签 (按 subject 划分 train/val, 绝不进入 Head_B);
          可选追加 PTB train 患者记录级金标准 (经 data/split_guard.py 防泄漏)。

输出全部写到项目外: /home/devcontainers/meeti/runs/dual_head_small/
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
PTB_DIR = Path('/home/devcontainers/ecg_data')
T_BEATS = 10


@tf.keras.utils.register_keras_serializable(package='DualHeadECG')
class MaskedAttentionPooling(layers.Layer):
    """Masked attention pooling: score = Dense(1), softmax over T (mask=-inf), sum."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.score_dense = layers.Dense(1, name='attn_score')

    def call(self, inputs):
        emb, mask = inputs
        score = self.score_dense(emb)                    # (B, T, 1)
        mask = tf.cast(mask, score.dtype)
        score = tf.where(mask > 0.5, score,
                         tf.fill(tf.shape(score), -1e9))
        att = tf.nn.softmax(score, axis=1)
        return tf.reduce_sum(att * emb, axis=1)          # (B, D)


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def load_hard_beats():
    """Head_B 硬标签: MIT/INCART/PTB 部署链, 仅 x_train_perm/y_train 进入训练."""
    d = np.load(NPZ)
    x = np.asarray(d['x_train_perm'], dtype=np.float32)[..., None]
    y = np.asarray(d['y_train'], dtype=np.int32)
    xv = np.asarray(d['x_val'], dtype=np.float32)[..., None]
    yv = np.asarray(d['y_val'], dtype=np.int32)
    xr = np.asarray(d['x_real_holdout'], dtype=np.float32)[..., None]
    return x, y, xv, yv, xr


def record_windows(beats, positions, n_beats_list, y_list, w_list):
    """按 record 组织前 T_BEATS 个 beat, 不足右侧 zero-padding + mask。

    beats:      (N, 250) 连续排列
    positions:  len(R)+1 起始下标
    n_beats_list: len(R) 每记录拍数
    """
    n = len(n_beats_list)
    x = np.zeros((n, T_BEATS, beats.shape[1]), dtype=np.float32)
    mask = np.zeros((n, T_BEATS, 1), dtype=np.float32)
    for i, nb in enumerate(n_beats_list):
        nb = int(nb)
        take = min(nb, T_BEATS)
        if take <= 0:
            raise RuntimeError(f'record {i} has no beats')
        x[i, :take] = beats[positions[i]:positions[i] + take]
        mask[i, :take] = 1.0
    y = np.asarray(y_list, dtype=np.int32)
    w = np.asarray(w_list, dtype=np.float32)
    return x, mask, y, w


def load_meeti_records(args):
    """MEETI 报告弱标签 -> Head_R。按 subject 做 80/20 记录级无泄漏划分。"""
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)

    def to_y(cls):
        if cls == 'abnormal':
            return 1
        if cls == 'normal':
            return 0
        return -1

    manifest['_y'] = manifest['report_class'].apply(to_y)
    manifest = manifest[manifest['_y'] >= 0].reset_index(drop=True)
    n_beats = manifest['n_beats'].astype(int).to_numpy()
    if int(n_beats.sum()) != len(beats):
        raise RuntimeError(f'manifest beat sum {int(n_beats.sum())} != '
                           f'beats.npy rows {len(beats)}')
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]]).astype(np.int64)
    x_all, mask_all, y_all, _ = record_windows(
        beats, positions, n_beats,
        manifest['_y'].to_numpy(),
        np.zeros(len(manifest), dtype=np.float32))

    # 患者级划分: 每个 subject 全部记录归入同一侧; subject 级分层标签 = 该
    # subject 任一记录 abnormal 则 abnormal。
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
        'w': np.full(int(tr_mask.sum()), args.meeti_weight, dtype=np.float32),
        'x_val': x_all[va_mask], 'mask_val': mask_all[va_mask],
        'y_val': y_all[va_mask],
        'n_records': len(manifest),
        'n_train': int(tr_mask.sum()), 'n_val': int(va_mask.sum()),
        'n_subjects': len(uniq),
        'train_label_dist': np.bincount(y_all[tr_mask], minlength=2).tolist(),
        'val_label_dist': np.bincount(y_all[va_mask], minlength=2).tolist(),
    }


def load_ptb_train_records():
    """PTB 记录级金标准: 仅 split_guard 的 train 患者, 记录标签 = max(拍标签)。"""
    from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402

    beats, labels, rids = load_arrays('ptb')
    rids = np.asarray(rids)
    labels = np.asarray(labels)
    split = compute_ptb_split()
    tr_mask = np.asarray(split['train'])
    beats = np.asarray(beats[tr_mask], dtype=np.float32)
    labels = labels[tr_mask]
    rids = rids[tr_mask]

    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first_idx, counts = np.unique(rids, return_index=True, return_counts=True)
    n = len(uniq)
    x = np.zeros((n, T_BEATS, beats.shape[1]), dtype=np.float32)
    mask = np.zeros((n, T_BEATS, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s, e = first_idx[i], first_idx[i] + counts[i]
        y[i] = int(labels[s:e].max())          # 记录级 0/1: 任一异常拍 -> abnormal
        take = min(int(counts[i]), T_BEATS)
        x[i, :take] = beats[s:s + take]
        mask[i, :take] = 1.0
    return {
        'x': x, 'mask': mask, 'y': y,
        'w': np.ones(n, dtype=np.float32),     # PTB 为强金标准, 权重 1.0
        'n_records': int(n),
        'label_dist': np.bincount(y, minlength=2).tolist(),
        'split_stats': {k: v for k, v in split['stats'].items()
                        if isinstance(v, (int, float, str, bool))},
    }


def check_disk_redline(min_gb=10.0):
    info = {}
    for p in ['/mnt/c', '/mnt/d']:
        free = shutil.disk_usage(p).free / (1024 ** 3)
        info[p] = round(free, 1)
        if free < min_gb:
            raise SystemExit(f'磁盘红线: {p} 剩余 {free:.1f}GB < {min_gb}GB, 中止')
    return info


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------

def build_dual_head_model():
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

    model = Model([beat_in, rec_in, rec_mask_in], [head_b, head_r],
                  name='dual_head_ecg')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(args.lr),
        loss={
            'head_b': tf.keras.losses.SparseCategoricalCrossentropy(),
            'head_r': tf.keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={'head_b': 1.0, 'head_r': args.lam},
    )
    # 轻量评估 probe: 与主模型共享同一批 layer 权重。
    beat_probe = Model(model.inputs[0], model.outputs[0], name='beat_probe')
    rec_probe = Model([model.inputs[1], model.inputs[2]], model.outputs[1],
                      name='record_probe')
    return model, beat_probe, rec_probe


def predict_record(probe, x, mask, batch_size=256):
    out = []
    for i in range(0, len(x), batch_size):
        out.append(probe.predict([x[i:i + batch_size],
                                  mask[i:i + batch_size]],
                                 batch_size=batch_size, verbose=0))
    return np.concatenate(out, axis=0)


# ---------------------------------------------------------------------------
# 训练 / 验证
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--lam', type=float, default=0.5,
                    help='记录头 loss 权重 λ')
    ap.add_argument('--meeti-weight', type=float, default=0.2,
                    help='MEETI 弱标签样本权重')
    ap.add_argument('--use-ptb', action='store_true',
                    help='Head_R 追加 PTB train 患者记录级金标准')
    ap.add_argument('--tag', default='dual_head_small')
    ap.add_argument('--run-dir',
                    default='/home/devcontainers/meeti/runs/dual_head_small')
    global args
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

    # ---- 数据 ----
    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(args)
    rec_x, rec_mask, rec_y, rec_w = (meeti['x'], meeti['mask'],
                                     meeti['y'], meeti['w'])
    rec_xv, rec_mv, rec_yv = meeti['x_val'], meeti['mask_val'], meeti['y_val']
    rec_source_info = {'meeti': {
        'n_train': meeti['n_train'], 'n_val': meeti['n_val'],
        'n_subjects': meeti['n_subjects'],
        'train_label_dist': meeti['train_label_dist'],
        'val_label_dist': meeti['val_label_dist'],
    }}
    if args.use_ptb:
        ptb = load_ptb_train_records()
        rec_x = np.concatenate([rec_x, ptb['x']], axis=0)
        rec_mask = np.concatenate([rec_mask, ptb['mask']], axis=0)
        rec_y = np.concatenate([rec_y, ptb['y']], axis=0)
        rec_w = np.concatenate([rec_w, ptb['w']], axis=0)
        rec_source_info['ptb'] = {k: ptb[k] for k in
                                  ('n_records', 'label_dist', 'split_stats')}

    # Head_B 的样本权重与基线 class_weight={0:2.0, 1:1.0} 一致。
    wb = np.where(yb == 0, 2.0, 1.0).astype(np.float32)
    n_hard, n_rec = len(xb), len(rec_x)
    n_steps = math.ceil(n_hard / args.batch_size)
    print(f'hard beats train={n_hard} val={len(xv)} real={len(x_real)}', flush=True)
    print(f'record train={n_rec} (MEETI {meeti["n_train"]}) '
          f'val={len(rec_xv)} (MEETI holdout, subject-split)', flush=True)
    print(f'meeti train labels {meeti["train_label_dist"]} '
          f'val labels {meeti["val_label_dist"]}', flush=True)
    print(f'loss lambda={args.lam} meeti_weight={args.meeti_weight} '
          f'batch={args.batch_size} epochs={args.epochs}', flush=True)

    # ---- 模型 ----
    model, beat_probe, rec_probe = build_dual_head_model()
    print(f'model params: {model.count_params():,}', flush=True)

    # ---- 训练循环 ----
    history = {
        'train_total_loss': [], 'train_head_b_loss': [], 'train_head_r_loss': [],
        'val_beat_auc': [], 'val_record_auc': [],
        'real_mean': [], 'real_frac_gt_0.5': [],
    }
    best_beat_auc, best_record_auc = -1.0, -1.0
    best_state = None

    for epoch in range(args.epochs):
        b_perm = rng.permutation(n_hard)
        r_perm = rng.permutation(n_rec)
        losses = []
        for step in range(n_steps):
            s = step * args.batch_size
            e = min(s + args.batch_size, n_hard)
            b_idx = b_perm[s:e]
            bs_actual = e - s

            # 记录头数据与硬拍批次对齐: 每个硬拍 batch 配一个记录 batch,
            # 记录数据不足时在该 epoch 内循环补齐 (small-scale 可接受)。
            r_s = (step * args.batch_size) % n_rec
            if r_s + bs_actual <= n_rec:
                r_idx = r_perm[r_s:r_s + bs_actual]
            else:
                r_idx = np.concatenate([r_perm[r_s:], r_perm[:r_s + bs_actual - n_rec]])

            step_loss = model.train_on_batch(
                [xb[b_idx], rec_x[r_idx], rec_mask[r_idx]],
                [yb[b_idx], rec_y[r_idx]],
                sample_weight=[wb[b_idx], rec_w[r_idx]],
            )
            losses.append(np.asarray(step_loss, dtype=np.float64))

        losses = np.stack(losses).mean(axis=0)
        history['train_total_loss'].append(float(losses[0]))
        history['train_head_b_loss'].append(float(losses[1]))
        history['train_head_r_loss'].append(float(losses[2]))

        # 验证: Head_B 公共 val + Head_R MEETI 留出记录 + 真实 AFE holdout
        pb = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
        beat_auc = float(roc_auc_score(yv, pb))
        pr = predict_record(rec_probe, rec_xv, rec_mv)[:, 1]
        rec_auc = float(roc_auc_score(rec_yv, pr))
        pr_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
        real_mean = float(pr_real.mean())
        real_frac = float((pr_real > 0.5).mean())

        history['val_beat_auc'].append(beat_auc)
        history['val_record_auc'].append(rec_auc)
        history['real_mean'].append(real_mean)
        history['real_frac_gt_0.5'].append(real_frac)

        # 保存最终 epoch 的预测结果, 供后续分析。
        preds = {
            'val_beat_p': pb, 'val_beat_y': yv,
            'val_record_p': pr, 'val_record_y': rec_yv,
            'real_p': pr_real,
        }
        if beat_auc >= best_beat_auc:
            best_beat_auc = beat_auc
            best_state = preds
        if rec_auc >= best_record_auc:
            best_record_auc = rec_auc
            best_preds = preds

        print(f'epoch {epoch + 1}/{args.epochs} '
              f'loss={history["train_total_loss"][-1]:.4f} '
              f'b={history["train_head_b_loss"][-1]:.4f} '
              f'r={history["train_head_r_loss"][-1]:.4f} '
              f'beatAUC={beat_auc:.4f} recAUC={rec_auc:.4f} '
              f'realMean={real_mean:.4f} realFrac={real_frac:.4f}', flush=True)

    elapsed = time.time() - t0
    disk_after = check_disk_redline()

    # ---- 最终评估 + 输出 ----
    pb_final = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    pr_final = predict_record(rec_probe, rec_xv, rec_mv)[:, 1]
    pr_real_final = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]
    final_beat_auc = float(roc_auc_score(yv, pb_final))
    final_record_auc = float(roc_auc_score(rec_yv, pr_final))

    save_error = None
    try:
        model.save(str(RUN / 'model.keras'))
        beat_probe.save(str(RUN / 'beat_probe.keras'))
        rec_probe.save(str(RUN / 'record_probe.keras'))
    except Exception as exc:  # pragma: no cover
        save_error = f'{type(exc).__name__}: {exc}'
        try:
            model.save_weights(str(RUN / 'model.weights.h5'))
            beat_probe.save_weights(str(RUN / 'beat_probe.weights.h5'))
            rec_probe.save_weights(str(RUN / 'record_probe.weights.h5'))
        except Exception as exc2:
            save_error += f' | weights: {type(exc2).__name__}: {exc2}'

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
            'epochs': args.epochs, 'batch_size': args.batch_size,
            'lr': args.lr, 'lam': args.lam,
            'meeti_weight': args.meeti_weight, 'use_ptb': args.use_ptb,
            'T_beats': T_BEATS, 'record_window': 'first T beats, right zero-pad',
            'head_b_data': 'distill_pilot_teacher_targets.npz x_train_perm/y_train',
            'head_r_train': ('MEETI subject-train records' +
                             (' + PTB train-patient records' if args.use_ptb else '')),
            'head_r_val': 'MEETI subject-holdout records',
        },
        'data': {
            'hard_train_beats': n_hard, 'hard_val_beats': int(len(xv)),
            'real_holdout_beats': int(len(x_real)),
            'meeti_records': meeti['n_records'],
            'meeti_train_records': meeti['n_train'],
            'meeti_val_records': meeti['n_val'],
            'meeti_subjects': meeti['n_subjects'],
            'meeti_train_label_dist': meeti['train_label_dist'],
            'meeti_val_label_dist': meeti['val_label_dist'],
            'record_train_total': int(n_rec),
            'record_sources': rec_source_info,
        },
        'metrics': {
            'beat_val_auc': final_beat_auc,
            'record_val_auc': final_record_auc,
            'real_mean': float(pr_real_final.mean()),
            'real_frac_gt_0.5': float((pr_real_final > 0.5).mean()),
            'best_beat_val_auc_epoch': best_beat_auc,
            'best_record_val_auc_epoch': best_record_auc,
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
