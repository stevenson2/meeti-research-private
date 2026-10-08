#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 1 (autonomous loop): int8 for the locked dual probes.

Design:
  - record probe: rebuilt as a NO-MASK single-input model (10-beat window,
    plain Dense+softmax attention; firmware arms detection after 10 collected
    beats). Same locked weights, exact-parity verified vs the masked probe on
    full-mask windows. This removes the -1e9 masked-softmax path that poisoned
    the earlier post-training int8 calibration (record max dp was 0.185).
  - int8 via post-training quantization with a 2000-sample representative
    dataset (train beats / full-mask train windows). QAT deferred: tfmq 0.8.1
    is incompatible with Keras 3 (TF 2.21); revisit via TF_USE_LEGACY_KERAS
    only if these gates fail.

Gates (vs Keras locked):
  beat val AUC drop <= 0.005; record full-mask window max |dp| <= 0.01;
  e2e gated replay (real FP + MIT val recall) unchanged.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model
# QAT deferred: tfmq 0.8.1 incompatible with Keras 3 (TF 2.21)

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_v3_small import build_full_model  # noqa: E402
from train_dual_head_small import (  # noqa: E402
    load_hard_beats, load_meeti_records, load_ptb_train_records)

RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/r1_int8')
T = 10


@tf.keras.utils.register_keras_serializable(package='DualHeadECG')
class SumAxis1(layers.Layer):
    def call(self, z):
        return tf.reduce_sum(z, axis=1)


@tf.keras.utils.register_keras_serializable(package='DualHeadECG')
class FlattenTime(layers.Layer):
    """(B, T, 250, 1) -> (B*T, 250, 1) without TimeDistributed/TRANSPOSE."""

    def call(self, x):
        return tf.reshape(x, (-1, x.shape[2], x.shape[3]))


@tf.keras.utils.register_keras_serializable(package='DualHeadECG')
class UnflattenTime(layers.Layer):
    """(B*T, D) -> (B, T, D)."""

    def __init__(self, t, **kwargs):
        super().__init__(**kwargs)
        self.t = t

    def call(self, x):
        return tf.reshape(x, (-1, self.t, x.shape[-1]))

    def get_config(self):
        cfg = super().get_config()
        cfg['t'] = self.t
        return cfg


def build_record_nomask(rec_probe):
    """Rebuild the record head without mask input, loading locked weights.

    Avoids TimeDistributed (its TRANSPOSE lowering breaks TFLite int8
    calibration): flatten time into batch, run the shared encoder, unflatten,
    then plain attention pooling.
    """
    enc_model = rec_probe.get_layer('td_encoder').layer
    proj_dense = rec_probe.get_layer('td_head_r_proj').layer
    attn_l = rec_probe.get_layer('head_r_attention')
    head_l = rec_probe.get_layer('head_r')
    att_w, att_b = attn_l.score_dense.get_weights()

    rec_in = layers.Input((T, 250, 1), name='rec_in_nomask')
    flat = FlattenTime(name='flat_time')(rec_in)                # (B*T,250,1)
    emb = enc_model(flat)                                       # (B*T,64)
    proj_flat = proj_dense(emb)                                 # (B*T,32)
    proj = UnflattenTime(T, name='unflat_time')(proj_flat)      # (B,10,32)
    score = layers.Dense(1, name='attn_score_nomask')(proj)     # (B,10,1)
    att = layers.Softmax(axis=1, name='attn_softmax_nomask')(score)
    pooled = SumAxis1(name='attn_pool_nomask')(layers.Multiply()([att, proj]))
    out = layers.Dense(2, activation='softmax', name='head_r_nomask')(pooled)
    m = Model(rec_in, out, name='record_nomask')
    m.get_layer('attn_score_nomask').set_weights([att_w, att_b])
    m.get_layer('head_r_nomask').set_weights(head_l.get_weights())
    return m


def convert_int8(keras_model, out_path, rep_gen):
    c = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    c.optimizations = [tf.lite.Optimize.DEFAULT]
    c.representative_dataset = rep_gen
    c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    c.inference_input_type = tf.int8
    c.inference_output_type = tf.int8
    out_path.write_bytes(c.convert())


def tflite_pred_single(path, x):
    """x: (n, ...) float samples matching the model input shape minus batch."""
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    ind, outd = it.get_input_details()[0], it.get_output_details()[0]
    iscale, izp = ind['quantization']
    oscale, ozp = outd['quantization']
    out = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        if iscale == 0.0:
            it.set_tensor(ind['index'], x[i][None].astype(np.float32))
        else:
            q = np.clip(np.round(x[i] / iscale + izp),
                        -128, 127).astype(np.int8)
            it.set_tensor(ind['index'], q[None])
        it.invoke()
        r = it.get_tensor(outd['index']).astype(np.float32)
        out[i] = (r[0, 1] - ozp) * oscale if oscale else r[0, 1]
    return np.clip(out, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--lr', type=float, default=1e-5)
    ap.add_argument('--batch', type=int, default=64)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    # ---- locked model ----
    model, beat_probe, rec_probe = build_full_model(args.lr, 0.2)
    model.load_weights(str(RUN / 'full.weights.h5'))

    rec_nomask = build_record_nomask(rec_probe)

    # ---- data ----
    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=0.2))
    ptb = load_ptb_train_records()
    rec_x = np.concatenate([meeti['x'], ptb['x']], axis=0)
    rec_y = np.concatenate([meeti['y'], ptb['y']], axis=0)
    rec_w = np.concatenate([meeti['w'], ptb['w']], axis=0)
    rec_m = np.concatenate([meeti['mask'], ptb['mask']], axis=0)
    full = rec_m[:, :, 0].sum(axis=1) == T
    rec_xf, rec_yf = rec_x[full], rec_y[full]
    print(f'train beats={len(xb)} rec windows={len(rec_x)} '
          f'full-mask={full.sum()}', flush=True)

    # ---- parity: nomask (float) vs masked probe on full-mask windows ----
    idx = np.random.default_rng(0).choice(len(rec_xf), min(500, len(rec_xf)),
                                          replace=False)
    ones = np.ones((len(idx), T, 1), np.float32)
    p_masked = np.concatenate(
        [rec_probe([rec_xf[idx][i:i + 256], ones[i:i + 256]],
                   training=False).numpy()[:, 1]
         for i in range(0, len(idx), 256)])
    p_nomask = np.concatenate(
        [rec_nomask(rec_xf[idx][i:i + 256], training=False).numpy()[:, 1]
         for i in range(0, len(idx), 256)])
    nomask_parity = float(np.abs(p_masked - p_nomask).max())
    print(f'nomask-vs-masked parity max|dp|={nomask_parity:.2e}', flush=True)

    # ---- Keras reference predictions FIRST, via EAGER calls ----
    # (Keras 3 .predict on probes sharing encoder layers across rebuilt
    #  graphs intermittently fails with 'unknown TensorShape'; eager direct
    #  calls avoid the traced-function cache entirely)
    def eager_beat(m, x, bs=512):
        outs = [m(x[i:i + bs], training=False).numpy()[:, 1]
                for i in range(0, len(x), bs)]
        return np.concatenate(outs)

    def eager_rec(m, x, mask, bs=256):
        outs = []
        for i in range(0, len(x), bs):
            outs.append(m([x[i:i + bs], mask[i:i + bs]],
                          training=False).numpy()[:, 1])
        return np.concatenate(outs)

    k_beat_auc = float(roc_auc_score(yv, eager_beat(beat_probe, xv)))

    real_dir = PROJ / 'pc_tools/ecg_dl/data/real'
    real_files = {'ecg_real_052': 'real_normal_beats_exp7c.npy',
                  'rec_latest': 'real_normal_beats_rec_latest.npy'}
    keras_real = {}
    for name, fname in real_files.items():
        beats = np.load(real_dir / fname).astype(np.float32)
        win = np.array([beats[j:j + T, :, None]
                        for j in range(len(beats) - T + 1)], np.float32)
        ones = np.ones((len(win), T, 1), np.float32)
        keras_real[name] = {
            'win': win,
            'p_rec': eager_rec(rec_probe, win, ones),
            'p_beat': eager_beat(beat_probe, beats[..., None]),
        }
        print(f'keras ref {name}: rec_mean='
              f'{keras_real[name]["p_rec"].mean():.4f}', flush=True)
    sub = np.random.default_rng(1).choice(len(xv), 2000, replace=False)
    pv_k = eager_beat(beat_probe, xv[sub])

    # ---- post-training int8 (2000-sample representative) ----
    rng_c = np.random.default_rng(0)
    cal_b = rng_c.choice(len(xb), 2000, replace=False)
    beat_calib = xb[cal_b]                                  # (2000,250,1)

    def beat_rep():
        for i in range(len(beat_calib)):
            yield [beat_calib[i:i + 1]]

    full_idx = np.where(full)[0]
    cal_r = rng_c.choice(len(full_idx), min(2000, len(full_idx)),
                         replace=False)
    rec_calib = rec_x[full_idx[cal_r]]                      # (n,10,250,1)

    def rec_rep():
        for i in range(len(rec_calib)):
            yield [rec_calib[i:i + 1]]

    convert_int8(beat_probe, OUT / 'ptq_beat_int8.tflite', beat_rep)
    convert_int8(rec_nomask, OUT / 'ptq_record_nomask_int8.tflite', rec_rep)
    sizes = {p.name: p.stat().st_size for p in OUT.glob('*.tflite')}
    print('tflite sizes', sizes, flush=True)

    # ---- int8 parity (Keras refs precomputed above) ----
    parity_real = {}
    for name, ref in keras_real.items():
        p_int8 = tflite_pred_single(
            OUT / 'ptq_record_nomask_int8.tflite', ref['win'])
        beats = np.load(real_dir / real_files[name]).astype(np.float32)
        pb_i = tflite_pred_single(
            OUT / 'ptq_beat_int8.tflite',
            beats[..., None].astype(np.float32))
        parity_real[name] = {
            'record_max_dp': float(np.abs(ref['p_rec'] - p_int8).max()),
            'beat_max_dp': float(np.abs(ref['p_beat'] - pb_i).max()),
            'n_windows': int(len(ref['win'])),
        }
        print(name, parity_real[name], flush=True)

    # MIT val subset beat AUC via int8 (pv_k precomputed pre-conversion)
    pv_i = tflite_pred_single(OUT / 'ptq_beat_int8.tflite',
                              xv[sub].astype(np.float32))
    result = {
        'config': vars(args) | {'note': 'record model rebuilt without mask; '
                               'firmware arms after 10 beats'},
        'nomask_float_parity_max_dp': nomask_parity,
        'beat_val_auc': {'keras': k_beat_auc,
                         'int8_sub': float(roc_auc_score(yv[sub], pv_i)),
                         'keras_sub': float(roc_auc_score(yv[sub], pv_k))},
        'tflite_sizes': sizes,
        'parity_real': parity_real,
        'gates': {
            'beat_auc_sub_drop_le_0.005': bool(
                roc_auc_score(yv[sub], pv_k) - roc_auc_score(yv[sub], pv_i)
                <= 0.005),
            'record_int8_max_dp_le_0.01': bool(
                max(v['record_max_dp'] for v in parity_real.values()) <= 0.01),
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result['gates']), flush=True)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
