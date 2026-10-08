#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direction 1: export locked dual-head probes to TFLite + parity check.

Exports (into runs/m4_tflite_export/):
  dual_beat_fp16.tflite / dual_record_fp16.tflite
  dual_beat_int8.tflite  / dual_record_int8.tflite   (full integer, rep data
      from MIT/INCART train beats; record windows built from same beats)

Parity vs Keras locked probes:
  - real AFE beats (both recordings): max |dp| per head
  - MIT/INCART val subset: max |dp| + AUC agreement
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import MaskedAttentionPooling, load_hard_beats  # noqa: E402,F401
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa: E402

RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/m4_tflite_export')
T = 10
REAL = {
    'ecg_real_052': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_exp7c.npy',
    'rec_latest': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_rec_latest.npy',
}
N_CALIB, N_VAL_SUB = 300, 2000


def make_tflite_pred(path):
    it = tf.lite.Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def predict_beat(it, beats):
    ind = it.get_input_details()[0]
    outd = it.get_output_details()[0]
    iscale, izp = ind['quantization']
    oscale, ozp = outd['quantization']
    out = np.empty(len(beats), dtype=np.float32)
    for i, b in enumerate(beats):
        if iscale == 0.0:  # float model
            it.set_tensor(ind['index'], b[None, :, None].astype(np.float32))
        else:
            xq = np.clip(np.round(b[:, None] / iscale + izp),
                         -128, 127).astype(np.int8)
            it.set_tensor(ind['index'], xq[None])
        it.invoke()
        q = it.get_tensor(outd['index']).astype(np.float32)
        out[i] = (q[0, 1] - ozp) * oscale if oscale else q[0, 1]
    return np.clip(out, 0.0, 1.0)


def predict_record(it, x, m):
    inds = {d['name']: d for d in it.get_input_details()}
    outd = it.get_output_details()[0]
    xi, mi = inds.get('record_input'), inds.get('record_mask')
    if xi is None:
        xi, mi = it.get_input_details()[0], it.get_input_details()[1]
    oscale, ozp = outd['quantization']
    out = np.empty(len(x), dtype=np.float32)
    for i in range(len(x)):
        if xi['quantization'][0] == 0.0:
            it.set_tensor(xi['index'], x[i:i + 1].astype(np.float32))
            it.set_tensor(mi['index'], m[i:i + 1].astype(np.float32))
        else:
            isx, izx = xi['quantization']
            ism, izm = mi['quantization']
            xq = np.clip(np.round(x[i] / isx + izx),
                         -128, 127).astype(np.int8)      # (T, 250, 1)
            mq = np.clip(np.round(m[i] / ism + izm),
                         -128, 127).astype(np.int8)      # (T, 1)
            it.set_tensor(xi['index'], xq[None])          # (1,T,250,1)
            it.set_tensor(mi['index'], mq[None])          # (1,T,1)
        it.invoke()
        q = it.get_tensor(outd['index']).astype(np.float32)
        out[i] = (q[0, 1] - ozp) * oscale if oscale else q[0, 1]
    return np.clip(out, 0.0, 1.0)


def causal_windows(beats):
    n = len(beats)
    x = np.zeros((n, T, beats.shape[1], 1), dtype=np.float32)
    m = np.zeros((n, T, 1), dtype=np.float32)
    for i in range(n):
        lo = max(0, i - T + 1)
        x[i, -(i - lo + 1):] = beats[lo:i + 1, ..., None]
        m[i, -(i - lo + 1):] = 1.0
    return x, m


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    beat_k = tf.keras.models.load_model(
        str(RUN / 'ckpt_balanced__beat.keras'))
    rec_k = tf.keras.models.load_model(
        str(RUN / 'ckpt_balanced__record.keras'))
    print('keras probes loaded', flush=True)

    # calibration + parity data
    xb, yb, xv, yv, _ = load_hard_beats()
    rng = np.random.default_rng(0)
    calib_idx = rng.choice(len(xb), N_CALIB, replace=False)
    calib_beats = xb[calib_idx]                       # (300, 250, 1)
    val_idx = rng.choice(len(xv), N_VAL_SUB, replace=False)
    val_beats, val_y = xv[val_idx], yv[val_idx]
    calib_x, calib_m = causal_windows(calib_beats[:, :, 0])

    def beat_rep():
        for i in range(N_CALIB):
            yield [calib_beats[i:i + 1]]

    def rec_rep():
        for i in range(N_CALIB):
            yield [calib_x[i:i + 1], calib_m[i:i + 1]]

    exports = {}
    for tag, keras_model, rep in [
            ('beat', beat_k, beat_rep), ('record', rec_k, rec_rep)]:
        # fp16
        c = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.target_spec.supported_types = [tf.float16]
        p16 = OUT / f'dual_{tag}_fp16.tflite'
        p16.write_bytes(c.convert())
        # full int8
        c = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
        p8 = OUT / f'dual_{tag}_int8.tflite'
        try:
            p8.write_bytes(c.convert())
            exports[tag] = {'fp16': str(p16), 'int8': str(p8)}
        except Exception as exc:
            exports[tag] = {'fp16': str(p16),
                            'int8_error': f'{type(exc).__name__}: {exc}'}
        print(tag, 'exported', exports[tag], flush=True)

    # ---------- parity ----------
    parity = {}
    real_beats = {n: np.load(p).astype(np.float32)
                  for n, p in REAL.items()}

    k_beat_real = {n: beat_k.predict(b[..., None], batch_size=512,
                                     verbose=0)[:, 1]
                   for n, b in real_beats.items()}
    k_rec_real = {}
    for n, b in real_beats.items():
        x, m = causal_windows(b)
        k_rec_real[n] = rec_k.predict([x, m], batch_size=512, verbose=0)[:, 1]

    for tag in ('fp16', 'int8'):
        it_b = make_tflite_pred(OUT / f'dual_beat_{tag}.tflite')
        it_r = make_tflite_pred(OUT / f'dual_record_{tag}.tflite')
        rows = {}
        for n, b in real_beats.items():
            tb = predict_beat(it_b, b)
            x, m = causal_windows(b)
            tr = predict_record(it_r, x, m)
            rows[n] = {
                'beat_max_dp': float(np.abs(tb - k_beat_real[n]).max()),
                'record_max_dp': float(np.abs(tr - k_rec_real[n]).max()),
            }
        # MIT val subset
        tv = predict_beat(it_b, val_beats[:, :, 0])
        kv = beat_k.predict(val_beats, batch_size=512, verbose=0)[:, 1]
        vx, vm = causal_windows(val_beats[:, :, 0])
        tvr = predict_record(it_r, vx, vm)
        kvr = rec_k.predict([vx, vm], batch_size=512, verbose=0)[:, 1]
        rows['mitval_sub'] = {
            'beat_max_dp': float(np.abs(tv - kv).max()),
            'record_max_dp': float(np.abs(tvr - kvr).max()),
            'beat_auc_keras': float(roc_auc_score(val_y, kv)),
            'beat_auc_tflite': float(roc_auc_score(val_y, tv)),
            'n': int(len(val_y)),
        }
        parity[tag] = rows
        print(tag, json.dumps(rows, indent=1), flush=True)

    result = {'exports': exports, 'parity': parity,
              'note': 'int8 record windows: T-major layout; masked padding '
                      'uses -1e9 scores -> verify softmax behavior in parity'}
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
