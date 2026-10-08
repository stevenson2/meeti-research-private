#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R12-T6 feasibility probe: per-beat readout from the frozen 5s encoder.

Question (BENCHMARK dual-capability frame): does the record-head encoder
already contain per-beat discriminative features, i.e. can ONE encoder
serve two readouts (record top2mean + beat head)?

Probe (frozen encoder, tiny head):
  trunk   = candidate record5s ckpt up to the layer feeding 'gap'
            (stride-8 map, ~(157,128) for a 1250-pt window)
  head    = pad map to 160 -> reshape (5,32,128) -> segment mean
            -> TimeDistributed(Dense(32 relu)) -> TimeDistributed(Dense(2))
  train   = MIT+INCART train-patient beats chunked into 5-beat windows
            (per-beat labels), encoder frozen, head only, few epochs
  eval    = per-beat AUC on MI val/test (S_beat-style) + device beat
            disclosure (p_mean / frac>0.5 on the two real recordings)

GO line: S_beat-style >= 55.9 (locked beat head) with device stats
disclosed.  This is a feasibility probe, not a deployable candidate.
"""
import argparse
import json
import math
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
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
import os  # noqa: E402
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa
from r8_data_5s import REAL_FILES, REAL_DIR  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r12_t6_beat_readout'


def load_mi_windows(max_train_windows):
    """5-beat windows from MI train patients (labels per position), plus
    per-beat val/test arrays for evaluation."""
    split = compute_mit_incart_split()

    def beats_of(tag, mask_key):
        b, l, r = load_arrays(tag)
        if mask_key is None:
            m = ~(np.asarray(split['val_masks'][tag], bool)
                  | np.asarray(split['test_masks'][tag], bool))
        else:
            m = np.asarray(split[mask_key][tag], bool)
        b = np.asarray(b, np.float32)[m]
        l = np.asarray(l)[m].astype(np.int32)
        r = np.asarray(r)[m]
        order = np.argsort(r, kind='stable')
        return b[order], l[order], r[order]

    # train windows: chunk each record's beats into consecutive 5s
    xs, ys = [], []
    rng = np.random.default_rng(0)
    for tag in ['mit_bih', 'incart']:
        b, l, r = beats_of(tag, None)
        uniq, first, counts = np.unique(r, return_index=True,
                                        return_counts=True)
        for i in range(len(uniq)):
            s, nb = int(first[i]), int(counts[i])
            nb5 = (nb // 5) * 5
            if nb5 < 5:
                continue
            w = b[s:s + nb5].reshape(-1, 5, 250)
            y = l[s:s + nb5].reshape(-1, 5)
            xs.append(w)
            ys.append(y)
    x = np.concatenate(xs)
    y = np.concatenate(ys).astype(np.int32)
    if max_train_windows and len(x) > max_train_windows:
        idx = rng.choice(len(x), size=max_train_windows, replace=False)
        x, y = x[idx], y[idx]
    # eval per-beat arrays
    ev = {}
    for name, key in [('val', 'val_masks'), ('test', 'test_masks')]:
        bs, ls = [], []
        for tag in ['mit_bih', 'incart']:
            b, l, _ = beats_of(tag, key)
            bs.append(b)
            ls.append(l)
        ev[name] = (np.concatenate(bs), np.concatenate(ls))
    return (x.reshape(len(x), 1250), y), ev


def build_probe(ckpt):
    full = tf.keras.models.load_model(str(ckpt))
    enc = full.get_layer('enc_5s')
    gap = enc.get_layer('gap')
    trunk = Model(enc.input, gap.input, name='trunk_stride8')
    trunk.trainable = False
    inp = layers.Input((1250, 1), name='record_5s_input')
    fmap = trunk(inp)                                # (B, ~157, 128)
    L = tf.constant(160, tf.int32)
    padded = tf.keras.layers.Lambda(
        lambda t: tf.pad(t, [[0, 0], [0, 160 - tf.shape(t)[1]], [0, 0]]))(
        fmap)
    seg = layers.Reshape((5, 32, padded.shape[2]),
                         name='seg_reshape')(padded)
    seg = layers.Lambda(lambda t: tf.reduce_mean(t, axis=2),
                        name='seg_mean')(seg)        # (B, 5, 128)
    h = layers.TimeDistributed(layers.Dense(32, activation='relu'),
                               name='beat_read_h')(seg)
    out = layers.TimeDistributed(layers.Dense(2, activation='softmax'),
                                 name='beat_read_out')(h)
    probe = Model(inp, out, name='beat_readout_probe')
    probe.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy())
    return probe, trunk


def beat_scores(probe, trunk, x_flat, batch=256):
    """Per-beat positive scores for flat (N,250) beat arrays: chunk into
    consecutive 5-beat windows (drop remainder), read each position."""
    n5 = (len(x_flat) // 5) * 5
    w = x_flat[:n5].reshape(-1, 5, 250).reshape(n5 // 5, 1250)[..., None]
    out = []
    for i in range(0, len(w), batch):
        p = probe.predict(w[i:i + batch], batch_size=batch, verbose=0)
        out.append(p[:, :, 1].reshape(-1))
    return np.concatenate(out), n5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default=str(
        ROOT / 'runs/r9_form-ptbbal_seed42/ckpt_best__record5s.keras'))
    ap.add_argument('--max-train-windows', type=int, default=30000)
    ap.add_argument('--epochs', type=int, default=5)
    ap.add_argument('--batch-size', type=int, default=256)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    t0 = time.time()

    (x_tr, y_tr), ev = load_mi_windows(args.max_train_windows)
    print(f'train windows {len(x_tr)} (beats {len(y_tr)}); val beats '
          f'{len(ev["val"][1])}, test beats {len(ev["test"][1])}', flush=True)
    probe, trunk = build_probe(args.ckpt)
    print(f'probe params (trainable): '
          f'{int(np.sum([np.prod(v.shape) for v in probe.trainable_variables]))}',
          flush=True)

    n = len(x_tr)
    n_steps = math.ceil(n / args.batch_size)
    rng = np.random.default_rng(42)
    hist = []
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        losses = []
        for step in range(n_steps):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            xb = x_tr[idx].reshape(-1, 1250, 1)
            yb = y_tr[idx]                      # (B, 5) per-position labels
            losses.append(probe.train_on_batch(xb, yb))
        hist.append(float(np.mean(losses)))
        print(f'[t6] ep{epoch + 1} loss={hist[-1]:.4f} '
              f'[{time.time() - t0:.0f}s]', flush=True)

    res = {'arm': 'r12_t6_beat_readout_probe',
           'encoder': args.ckpt, 'encoder_frozen': True,
           'train_windows': int(len(x_tr)), 'epochs': args.epochs,
           'loss_history': hist}
    for name in ['val', 'test']:
        xb, yb = ev[name]
        p, n5 = beat_scores(probe, trunk, xb)
        auc = float(roc_auc_score(yb[:n5], p))
        res[f'mi_{name}_beat_auc'] = round(auc, 4)
        res[f'mi_{name}_n_beats'] = int(n5)
    sbeat = round(100 * ((res['mi_val_beat_auc'] - 0.5) / 0.5
                         + (res['mi_test_beat_auc'] - 0.5) / 0.5) / 2, 1)
    res['S_beat_style'] = sbeat
    dev = {}
    for name, fname in REAL_FILES.items():
        b = np.load(REAL_DIR / fname).astype(np.float32)
        if b.ndim == 3:
            b = b.reshape(b.shape[0], -1)
        p, n5 = beat_scores(probe, trunk, b)
        dev[name] = {'n_beats': int(n5),
                     'p_mean': round(float(p.mean()), 4),
                     'frac_gt_0.5': round(float((p > 0.5).mean()), 4),
                     'p_max': round(float(p.max()), 4)}
    res['device_beat'] = dev
    res['go_line'] = {
        'S_beat_style_ge_55.9': bool(sbeat >= 55.9),
        'note': 'feasibility probe only (frozen encoder, 5-segment '
                'approximate pooling); device stats disclosed, not gated'}
    with open(OUT / 'result.json', 'w') as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print('T6 RESULT:', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
