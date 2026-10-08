#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R6b: decisive test - PTQ with GAIN-AUGMENTED calibration (input scale in
PTQ comes from the representative dataset, unlike QAT's frozen FakeQuant).
If the rec_latest beat#96 flip disappears, the firmware recipe is
'PTQ + wide-range calibration'; record head re-exported the same way."""
import os

os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import tensorflow as tf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_v3_small import build_full_model  # noqa: E402
from train_dual_head_small import (  # noqa: E402
    load_hard_beats, load_meeti_records, load_ptb_train_records)

# reuse layer helpers from r1b (module import only; main() is guarded)
import r1b_qat_legacy as R1B  # noqa: E402

RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/r6b_ptq_gaincal')
T = 10
REAL = {'ecg_real_052': 'real_normal_beats_exp7c.npy',
        'rec_latest': 'real_normal_beats_rec_latest.npy'}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, beat_probe, rec_probe = build_full_model(1e-5, 0.2)
    model.load_weights(str(RUN / 'full.weights.h5'))
    rec_nomask = R1B.build_record_nomask(rec_probe, beat_probe.get_layer('beat_encoder')) \
        if hasattr(R1B, 'build_record_nomask') else None
    if rec_nomask is None:
        # r1b's no-mask builder is build_record_qat(rec_probe, encoder) which
        # applies QAT; for PTQ we need the plain graph: build via annotation
        # pieces but WITHOUT quantize_apply — reuse internals
        from tensorflow.keras import Model, layers
        import tensorflow_model_optimization as tfmq
        enc = beat_probe.get_layer('beat_encoder')
        qann_dummy = enc  # plain encoder call, no annotation
        proj_dense = rec_probe.get_layer('td_head_r_proj').layer
        attn_l = rec_probe.get_layer('head_r_attention')
        head_l = rec_probe.get_layer('head_r')
        att_w, att_b = attn_l.score_dense.get_weights()
        rec_in = layers.Input((T, 250, 1), name='rec_in_nomask')
        flat = R1B.FlattenTime(name='flat_time')(rec_in)
        proj = R1B.UnflattenTime(T, name='unflat_time')(
            proj_dense(qann_dummy(flat)))
        score = layers.Dense(1, name='attn_score_nomask')(proj)
        att = layers.Softmax(axis=1, name='attn_softmax_nomask')(score)
        pooled = R1B.SumAxis1(name='attn_pool_nomask')(
            layers.Multiply()([att, proj]))
        out = layers.Dense(2, activation='softmax',
                           name='head_r_nomask')(pooled)
        rec_nomask = Model(rec_in, out, name='record_nomask_ptq')
        rec_nomask.get_layer('attn_score_nomask').set_weights([att_w, att_b])
        rec_nomask.get_layer('head_r_nomask').set_weights(head_l.get_weights())

    xb, yb, xv, yv, _ = load_hard_beats()
    meeti = load_meeti_records(type('A', (), {'meeti_weight': 0.2})())
    ptb = load_ptb_train_records()
    rec_x = np.concatenate([meeti['x'], ptb['x']], axis=0)
    rec_m = np.concatenate([meeti['mask'], ptb['mask']], axis=0)
    full = rec_m[:, :, 0].sum(axis=1) == T
    rec_xf = rec_x[full]

    rng = np.random.default_rng(0)
    cal_b = rng.choice(len(xb), 1000, replace=False)
    cal_r = rng.choice(len(rec_xf), min(1000, len(rec_xf)), replace=False)

    def rep_beat():
        for i in cal_b:
            yield [xb[i:i + 1]]
            yield [xb[i:i + 1] * 1.7]

    def rep_rec():
        for i in cal_r:
            yield [rec_xf[i:i + 1]]
            yield [rec_xf[i:i + 1] * 1.7]

    def convert(km, rep, path):
        c = tf.lite.TFLiteConverter.from_keras_model(km)
        c.optimizations = [tf.lite.Optimize.DEFAULT]
        c.representative_dataset = rep
        c.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        c.inference_input_type = tf.int8
        c.inference_output_type = tf.int8
        path.write_bytes(c.convert())
        it = tf.lite.Interpreter(model_path=str(path))
        it.allocate_tensors()
        ind = it.get_input_details()[0]
        s, z = ind['quantization']
        print(f'{path.name}: input scale={s:.6f} zp={z} '
              f'(range [{(-128 - z) * s:.2f}, {(127 - z) * s:.2f}])',
              flush=True)

    convert(beat_probe, rep_beat, OUT / 'ptq_beat_int8.tflite')
    convert(rec_nomask, rep_rec, OUT / 'ptq_record_nomask_int8.tflite')

    real_dir = PROJ / 'pc_tools/ecg_dl/data/real'
    res = {}
    for name, fname in REAL.items():
        beats = np.load(real_dir / fname).astype(np.float32)
        win = np.array([beats[j:j + T, :, None]
                        for j in range(len(beats) - T + 1)], np.float32)
        ones = np.ones((len(win), T, 1), np.float32)
        p_rec_k = np.concatenate(
            [rec_probe([win[i:i + 256], ones[i:i + 256]],
                       training=False).numpy()[:, 1]
             for i in range(0, len(win), 256)])
        p_b_k = np.concatenate(
            [beat_probe(beats[i:i + 512, :, None],
                        training=False).numpy()[:, 1]
             for i in range(0, len(beats), 512)])
        p_rec_8 = R1B.tflite_pred(OUT / 'ptq_record_nomask_int8.tflite', win)
        p_b_8 = R1B.tflite_pred(OUT / 'ptq_beat_int8.tflite',
                                beats[..., None].astype(np.float32))
        res[name] = {
            'beat_max_dp': float(np.abs(p_b_k - p_b_8).max()),
            'beat_p99_dp': float(np.quantile(np.abs(p_b_k - p_b_8), 0.99)),
            'record_max_dp': float(np.abs(p_rec_k - p_rec_8).max()),
            'record_p99_dp': float(
                np.quantile(np.abs(p_rec_k - p_rec_8), 0.99)),
        }
        print(name, res[name], flush=True)

    with open(OUT / 'result.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
