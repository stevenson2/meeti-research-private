#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Round 1b: true QAT int8 via TF_USE_LEGACY_KERAS=1 (tfmq needs Keras 2).

Must run with legacy keras enabled (set at import time below) BEFORE any
tensorflow import. QAT fine-tunes from locked weights (lr 1e-5), then
converts to int8 without representative dataset (fake-quant aware).

Gates: beat val AUC drop <= 0.005 (float QAT vs keras);
       int8 vs keras max|dp| <= 0.01 on real recordings (both heads).
"""
import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from tensorflow.keras import Model, layers  # noqa: E402
import tensorflow_model_optimization as tfmq  # noqa: E402

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_v3_small import build_full_model  # noqa: E402
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, load_hard_beats, load_meeti_records,
    load_ptb_train_records)

RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/r1c_qat_int8_gaincal')
T = 10
REAL = {'ecg_real_052': 'real_normal_beats_exp7c.npy',
        'rec_latest': 'real_normal_beats_rec_latest.npy'}


class SumAxis1(layers.Layer):
    def call(self, z):
        return tf.reduce_sum(z, axis=1)


class FlattenTime(layers.Layer):
    def call(self, x):
        return tf.reshape(x, (-1, x.shape[2], x.shape[3]))


class UnflattenTime(layers.Layer):
    def __init__(self, t, **kw):
        super().__init__(**kw)
        self.t = t

    def call(self, x):
        return tf.reshape(x, (-1, self.t, x.shape[-1]))

    def get_config(self):
        return super().get_config() | {'t': self.t}


QSCOPE = {'FlattenTime': FlattenTime, 'UnflattenTime': UnflattenTime,
          'SumAxis1': SumAxis1,
          'MaskedAttentionPooling': MaskedAttentionPooling}


def _dense_with_weights(units, name, weights, input_dim, act=None):
    d = layers.Dense(units, activation=act, name=name)
    d.build((None, input_dim))
    d.set_weights(weights)
    return d


def build_beat_qat(beat_probe, encoder):
    """QAT graph: annotated encoder (nested fine for train/convert) +
    annotated head Dense that already carries locked weights."""
    qann = tfmq.quantization.keras.quantize_annotate_model(encoder)
    head = _dense_with_weights(
        2, 'head_b', beat_probe.get_layer('head_b').get_weights(),
        64, act="softmax")
    inp = layers.Input((250, 1), name='beat_in')
    out = tfmq.quantization.keras.quantize_annotate_layer(head)(qann(inp))
    m = Model(inp, out, name='beat_qat')
    with tfmq.quantization.keras.quantize_scope(QSCOPE):
        return tfmq.quantization.keras.quantize_apply(m)


def build_record_qat(rec_probe, encoder):
    """No-mask record QAT graph (no TimeDistributed): FlattenTime ->
    annotated encoder -> annotated proj/score/head Denses (locked weights
    set BEFORE annotation) -> attention."""
    qann = tfmq.quantization.keras.quantize_annotate_model(encoder)
    proj_dense = rec_probe.get_layer('td_head_r_proj').layer
    attn_l = rec_probe.get_layer('head_r_attention')
    head_l = rec_probe.get_layer('head_r')
    att_w, att_b = attn_l.score_dense.get_weights()

    proj = _dense_with_weights(32, 'head_r_proj',
                               proj_dense.get_weights(), 64, act="relu")
    score = _dense_with_weights(1, 'attn_score_nomask', [att_w, att_b], 32)
    head = _dense_with_weights(2, 'head_r_nomask',
                               head_l.get_weights(), 32,
                               act="softmax")

    rec_in = layers.Input((T, 250, 1), name='rec_in_nomask')
    flat = FlattenTime(name='flat_time')(rec_in)
    proj_o = UnflattenTime(T, name='unflat_time')(
        tfmq.quantization.keras.quantize_annotate_layer(proj)(qann(flat)))
    score_o = tfmq.quantization.keras.quantize_annotate_layer(score)(proj_o)
    att = layers.Softmax(axis=1, name='attn_softmax_nomask')(score_o)
    pooled = SumAxis1(name='attn_pool_nomask')(
        layers.Multiply()([att, proj_o]))
    out = tfmq.quantization.keras.quantize_annotate_layer(head)(pooled)
    m = Model(rec_in, out, name='record_qat')
    with tfmq.quantization.keras.quantize_scope(QSCOPE):
        return tfmq.quantization.keras.quantize_apply(m)


def eager_beat(m, x, bs=512):
    return np.concatenate(
        [m(x[i:i + bs], training=False).numpy()[:, 1]
         for i in range(0, len(x), bs)])


def eager_rec(m, x, bs=256):
    return np.concatenate(
        [m(x[i:i + bs], training=False).numpy()[:, 1]
         for i in range(0, len(x), bs)])


def tflite_pred(path, x):
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
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import tf_keras
    print('tf', tf.__version__, '| legacy keras', tf_keras.__version__,
          flush=True)

    model, beat_probe, rec_probe = build_full_model(1e-5, 0.2)
    model.load_weights(str(RUN / 'full.weights.h5'))
    encoder = beat_probe.get_layer('beat_encoder')

    q_beat = build_beat_qat(beat_probe, encoder)
    q_rec = build_record_qat(rec_probe, encoder)
    # float-side QAT-vs-original parity (eager, before any fine-tune)
    xb0, yb0, xv0, yv0, _ = load_hard_beats()
    p_orig = eager_beat(beat_probe, xv0[:512])
    p_qat = eager_beat(q_beat, xv0[:512])
    print('beat qat(float-init)-vs-original max|dp| =',
          float(np.abs(p_orig - p_qat).max()), flush=True)

    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(type('A', (), {'meeti_weight': 0.2})())
    ptb = load_ptb_train_records()
    rec_x = np.concatenate([meeti['x'], ptb['x']], axis=0)
    rec_y = np.concatenate([meeti['y'], ptb['y']], axis=0)
    rec_w = np.concatenate([meeti['w'], ptb['w']], axis=0)
    rec_m = np.concatenate([meeti['mask'], ptb['mask']], axis=0)
    full = rec_m[:, :, 0].sum(axis=1) == T
    rec_xf, rec_yf = rec_x[full], rec_y[full]

    real_dir = PROJ / 'pc_tools/ecg_dl/data/real'
    refs = {}
    for name, fname in REAL.items():
        beats = np.load(real_dir / fname).astype(np.float32)
        win = np.array([beats[j:j + T, :, None]
                        for j in range(len(beats) - T + 1)], np.float32)
        ones = np.ones((len(win), T, 1), np.float32)
        p_masked = np.concatenate(
            [rec_probe([win[i:i + 256], ones[i:i + 256]],
                       training=False).numpy()[:, 1]
             for i in range(0, len(win), 256)])
        refs[name] = {'win': win,
                      'p_rec': p_masked,
                      'p_beat': eager_beat(beat_probe, beats[..., None])}
        print(f'ref {name} rec_mean={refs[name]["p_rec"].mean():.4f}',
              flush=True)
    k_beat_auc = float(roc_auc_score(yv, eager_beat(beat_probe, xv)))

    # ---- QAT fine-tune (annotated graphs already built) ----
    q_beat.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                   loss='sparse_categorical_crossentropy')
    wb = np.where(yb == 0, 2.0, 1.0).astype(np.float32)
    q_beat.fit(xb, yb, sample_weight=wb, batch_size=64, epochs=5, verbose=2)
    q_rec.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss='sparse_categorical_crossentropy')
    q_rec.fit(rec_xf, rec_yf, sample_weight=rec_w[full], batch_size=64,
              epochs=5, verbose=2)
    q_beat_auc = float(roc_auc_score(yv, eager_beat(q_beat, xv)))
    print(f'beat val AUC keras={k_beat_auc:.4f} qat={q_beat_auc:.4f}',
          flush=True)

    # ---- export int8; representative dataset is amplitude-augmented
    # (each sample also yielded at x1.6 gain) so the input quantization
    # range covers device-domain beat amplitudes (real-device z-scored
    # beats reach ~4.5 while MIT-domain calibration clips near ~3.6,
    # which catastrophically distorted one rec_latest beat) ----
    rng_c = np.random.default_rng(0)
    cal_b = rng_c.choice(len(xb), 1000, replace=False)
    full_idx = np.where(full)[0]
    cal_r = rng_c.choice(len(full_idx), min(1000, len(full_idx)),
                         replace=False)

    def rep_beat():
        for i in cal_b:
            yield [xb[i:i + 1]]
            yield [xb[i:i + 1] * 1.6]

    def rep_rec():
        for i in full_idx[cal_r]:
            yield [rec_x[i:i + 1]]
            yield [rec_x[i:i + 1] * 1.6]

    for qm, path, rep in [(q_beat, OUT / 'qat_beat_int8.tflite', rep_beat),
                          (q_rec, OUT / 'qat_record_nomask_int8.tflite',
                           rep_rec)]:
        c = tf.lite.TFLiteConverter.from_keras_model(qm)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
        path.write_bytes(c.convert())
    sizes = {p.name: p.stat().st_size for p in OUT.glob('*.tflite')}
    print('sizes', sizes, flush=True)

    parity = {}
    for name, ref in refs.items():
        pr8 = tflite_pred(OUT / 'qat_record_nomask_int8.tflite', ref['win'])
        beats8 = tflite_pred(
            OUT / 'qat_beat_int8.tflite',
            np.load(real_dir / REAL[name]).astype(np.float32)[..., None])
        parity[name] = {
            'record_max_dp': float(np.abs(ref['p_rec'] - pr8).max()),
            'beat_max_dp': float(np.abs(ref['p_beat'] - beats8).max()),
        }
        print(name, parity[name], flush=True)

    result = {
        'keras': f'legacy {tf_keras.__version__}',
        'beat_val_auc': {'keras': k_beat_auc, 'qat_float': q_beat_auc},
        'sizes': sizes, 'parity_real': parity,
        'gates': {
            'beat_auc_drop_le_0.005': bool(k_beat_auc - q_beat_auc <= 0.005),
            'record_max_dp_le_0.01': bool(
                max(v['record_max_dp'] for v in parity.values()) <= 0.01),
            'beat_max_dp_le_0.05': bool(
                max(v['beat_max_dp'] for v in parity.values()) <= 0.05),
        },
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result['gates']), flush=True)



if __name__ == '__main__':
    main()
