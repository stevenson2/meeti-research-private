#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B3: AF-head domain adaptation (frozen encoder + readout head).

R12-T6 route: freeze the R14 AF-head trunk, cache its 128-d GAP features,
initialise a small readout head from the R14 fc1/out weights and train ONLY
the head on an expanded multi-domain AF pool:

  PTB-XL folds1-8 (raw mV) + afdb fit-record windows (<=5k AF / <=5k N,
  100 Hz, per-window p99 aligned to the PTB train median) + CinC2017
  A(738)/N(738)/O(738) (300->100 Hz, p99 aligned).

Validation: PTB fold9 + a record-held-out afdb val subset; checkpoint by
mean of the two AUCs (no fold-10 / afdb-eval use in selection).

go-line: afdb held-out AUC >= 0.90 AND PTB fold-10 AUC >= 0.95
(vs R14: afdb 0.763/0.782/0.900, fold-10 0.965/0.969/0.968).

Outputs: runs/r15_af_adapt/{result.json, head_seed*.weights.h5}
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import spec_at_recall, train_theta_at_recall, youden_threshold  # noqa: E402
from models_patch.ptbxl_builders import build_ptbxl_large  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r15_af_adapt'
PTB = ROOT / 'runs/r14_af_prep'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
POOL = ROOT / 'runs/r15_af_pool'
BASE_WEIGHTS = ROOT / 'runs/r14_af_head/af_seed42/best.weights.h5'
WIN = 1000


def ptb_p99():
    tr = np.load(PTB / 'ptbxl_af_train.npz')
    return float(np.median(np.percentile(np.abs(tr['x']), 99, axis=1)))


def scale_p99(x, target):
    p = np.percentile(np.abs(x), 99, axis=1, keepdims=True)
    return (x * (target / np.maximum(p, 1e-9))).astype(np.float32)


def afdb_windows(records, n_per_class, target, seed):
    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for L in (1, 0):
        cand_all = [i for i in np.where(lab == L)[0] if rec[i] in records]
        rng.shuffle(cand_all)
        take = cand_all[:n_per_class]
        # group by record to read each signal once
        by_rec = {}
        for i in take:
            by_rec.setdefault(rec[i], []).append(i)
        for r, idxs in by_rec.items():
            obj = wfdb.rdrecord(str(AFDB / r))
            sig = np.asarray(obj.p_signal, np.float64)[:, 1]
            for i in idxs:
                w = sig[start[i]:start[i] + 2500]
                if len(w) < 2500:
                    continue
                w100 = resample_poly(w, 2, 5)
                xs.append(w100.astype(np.float32))
                ys.append(L)
    X = np.stack(xs)
    return scale_p99(X, target), np.array(ys, np.int8)


def load_base():
    m = build_ptbxl_large(input_shape=(1000, 1), n_classes=1,
                          output_activation='sigmoid', name='r14_af')
    m.load_weights(str(BASE_WEIGHTS))
    return m


def build_head(base):
    inp = tf.keras.Input(shape=(128,), name='gap_feat')
    h = tf.keras.layers.Dense(64, activation='relu', name='adapt_fc1')(inp)
    h = tf.keras.layers.Dropout(0.3, name='adapt_do')(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='adapt_out')(h)
    head = tf.keras.Model(inp, out, name='af_adapt_head')
    head.get_layer('adapt_fc1').set_weights(base.get_layer('fc1').get_weights())
    head.get_layer('adapt_out').set_weights(
        base.get_layer('out_multi').get_weights())
    return head


def train_head(Xtr, ytr, Xv1, yv1, Xv2, yv2, seed, epochs=60):
    tf.keras.utils.set_random_seed(seed)
    inp = tf.keras.Input(shape=(128,), name='gap_feat')
    h = tf.keras.layers.Dense(64, activation='relu', name='adapt_fc1')(inp)
    h = tf.keras.layers.Dropout(0.3, name='adapt_do')(h)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='adapt_out')(h)
    head = tf.keras.Model(inp, out)
    base = load_base()
    head.get_layer('adapt_fc1').set_weights(base.get_layer('fc1').get_weights())
    head.get_layer('adapt_out').set_weights(
        base.get_layer('out_multi').get_weights())
    n_pos = max(int((ytr == 1).sum()), 1)
    n_neg = max(int((ytr == 0).sum()), 1)
    w = np.where(ytr == 1, len(ytr) / (2.0 * n_pos),
                 len(ytr) / (2.0 * n_neg)).astype(np.float32)
    opt = tf.keras.optimizers.Adam(3e-4)
    loss_fn = tf.keras.losses.BinaryCrossentropy(reduction='none')
    rng = np.random.default_rng(seed)
    best = {'score': -1, 'epoch': -1}
    hist = []
    for ep in range(epochs):
        perm = rng.permutation(len(Xtr))
        losses = []
        for s in range(0, len(Xtr), 256):
            idx = perm[s:s + 256]
            with tf.GradientTape() as tape:
                p = head(Xtr[idx], training=True)
                loss = tf.reduce_mean(w[idx] * loss_fn(ytr[idx, None], p))
            grads = tape.gradient(loss, head.trainable_variables)
            opt.apply_gradients(zip(grads, head.trainable_variables))
            losses.append(float(loss))
        p1 = head.predict(Xv1, batch_size=1024, verbose=0)[:, 0]
        p2 = head.predict(Xv2, batch_size=1024, verbose=0)[:, 0]
        a1 = float(roc_auc_score(yv1, p1))
        a2 = float(roc_auc_score(yv2, p2))
        score = 0.5 * (a1 + a2)
        hist.append({'ep': ep + 1, 'loss': round(float(np.mean(losses)), 5),
                     'ptb9_auc': round(a1, 4), 'afdb_val_auc': round(a2, 4),
                     'score': round(score, 4)})
        if score > best['score']:
            best = {'score': score, 'epoch': ep + 1,
                    'ptb9_auc': round(a1, 4), 'afdb_val_auc': round(a2, 4)}
            head.save_weights(str(OUT / f'head_seed{seed}.weights.h5'))
    head.load_weights(str(OUT / f'head_seed{seed}.weights.h5'))
    return head, best, hist


def op_metrics(y, s, theta=None):
    out = {'auc': round(float(roc_auc_score(y, s)), 4),
           'ap': round(float(average_precision_score(y, s)), 4),
           'spec_at_R90': spec_at_recall(y, s)}
    if theta is not None:
        yp = s >= theta
        out['R_at_theta'] = round(float((yp & (y == 1)).sum())
                                  / max(int((y == 1).sum()), 1), 4)
        out['spec_at_theta'] = round(float(((~yp) & (y == 0)).sum())
                                     / max(int((y == 0).sum()), 1), 4)
    return out


def device_probs(model, feat, target, stride_s=1.0):
    import struct
    from fractions import Fraction
    electron = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                    'ecg-programme-tju-flex.electron-master')
    sys.path.insert(0, str(electron / 'pc_tools/ecg_dl'))
    from eval_deploy_match import causal_hp_05_fs250, deployment_chain

    def read_ecgr(p):
        raw = p.read_bytes()
        n = struct.unpack_from('<I', raw, 18)[0]
        dur = struct.unpack_from('<I', raw, 14)[0]
        x = np.frombuffer(raw, dtype='<i2', count=n, offset=32).astype(
            np.float64) / 8000.0
        return x, n / dur, dur

    out = {}
    for name, p in [('ecg_real_052_chain',
                     electron / 'pc_tools/ecg_dl/data/real/ecg_real_052.ecgr'),
                    ('rec_latest_raw', electron / 'rec_latest.ecgr')]:
        x, fs, dur = read_ecgr(p)
        ch = causal_hp_05_fs250(deployment_chain(x, fs))
        s100 = resample_poly(ch, 2, 5)
        step = max(int(stride_s * 100), 1)
        W = np.stack([s100[i:i + WIN] for i in range(0, len(s100) - WIN + 1,
                                                     step)])
        res = {}
        for pol, wp in [('+', W), ('-', -W)]:
            Ws = scale_p99(wp, target)
            Fs = feat.predict(Ws[..., None], batch_size=256, verbose=0)
            pv = model.predict(Fs, batch_size=1024, verbose=0)[:, 0]
            res[pol] = {'n': int(len(pv)), 'max': round(float(pv.max()), 4),
                        'mean': round(float(pv.mean()), 4)}
        out[name] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--afdb-train-per-class', type=int, default=5000)
    ap.add_argument('--afdb-val-per-class', type=int, default=1000)
    ap.add_argument('--afdb-eval-per-class', type=int, default=1000)
    ap.add_argument('--cinc-val-frac', type=float, default=0.2)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    target = ptb_p99()
    print('ptb p99 target', target, flush=True)

    # ---- PTB folds1-8 train / fold9 val / fold10 test ----
    tr = np.load(PTB / 'ptbxl_af_train.npz')
    x, y, folds = tr['x'], tr['y_af'].astype(int), tr['folds']
    m_tr, m_v, m_te = folds <= 8, folds == 9, None
    te = np.load(PTB / 'ptbxl_af_test.npz')
    x_tr, y_tr = x[m_tr], y[m_tr]
    x_v9, y_v9 = x[m_v], y[m_v]
    x_te, y_te = te['x'], te['y_af'].astype(int)
    print('PTB train', x_tr.shape, 'fold9', x_v9.shape, 'fold10', x_te.shape,
          flush=True)

    # ---- afdb fit-record split (record-level) ----
    af = np.load(ROOT / 'runs/r15_laneA_rr_gate/features.npz')
    fit_recs = list(af['fit_recs'])
    eval_recs = list(af['eval_recs'])
    fit_sorted = sorted(fit_recs)
    afdb_tr_recs = fit_sorted[:-3]
    afdb_val_recs = fit_sorted[-3:]
    xa_tr, ya_tr = afdb_windows(afdb_tr_recs, args.afdb_train_per_class,
                                target, 42)
    xa_v, ya_v = afdb_windows(afdb_val_recs, args.afdb_val_per_class,
                              target, 43)
    xa_te, ya_te = afdb_windows(eval_recs, args.afdb_eval_per_class,
                                target, 44)
    print('afdb train', xa_tr.shape, ya_tr.mean(), 'val', xa_v.shape,
          ya_v.mean(), 'eval', xa_te.shape, ya_te.mean(), flush=True)

    # ---- CinC2017 ----
    cin = np.load(POOL / 'cinc2017.npz')
    xc, yc = cin['x'], cin['y'].astype(int)
    rng = np.random.default_rng(42)
    recs = np.array(sorted(set(cin['rec'])))
    val_recs = set(rng.choice(recs, int(len(recs) * args.cinc_val_frac),
                              replace=False))
    mc_v = np.isin(cin['rec'], list(val_recs))
    xc_tr, yc_tr = scale_p99(xc[~mc_v], target), yc[~mc_v]
    xc_v, yc_v = scale_p99(xc[mc_v], target), yc[mc_v]
    print('cinc train', xc_tr.shape, yc_tr.mean(), 'val', xc_v.shape,
          yc_v.mean(), flush=True)

    # ---- frozen encoder feature cache ----
    base = load_base()
    feat = tf.keras.Model(inputs=base.input,
                          outputs=base.get_layer('gap').output,
                          name='r14_trunk_gap')
    feat.trainable = False

    def gap(X):
        X = np.asarray(X)
        if X.ndim == 4:
            X = X[..., 0]
        if X.ndim == 2:
            X = X[..., None]
        return feat.predict(X, batch_size=512, verbose=0)

    F_tr = np.concatenate([gap(x_tr), gap(xa_tr), gap(xc_tr)])
    Y_tr = np.concatenate([y_tr, ya_tr, yc_tr])
    F_v9, F_av, F_cv = gap(x_v9), gap(xa_v), gap(xc_v)
    F_te, F_ae, F_ce = gap(x_te), gap(xa_te), gap(xc_v)
    print('features', F_tr.shape, 'elapsed', round(time.time() - t0, 1),
          flush=True)

    result = {'arm': 'r15_af_adapt', 'base_weights': str(BASE_WEIGHTS),
              'ptb_p99_target': target,
              'recipe': 'frozen R14 trunk + cached 128-d GAP features + head '
                        'init from fc1/out; PTB folds1-8 + afdb fit windows + '
                        'CinC2017 A/N/O; p99 amplitude alignment',
              'n_train': int(len(Y_tr)),
              'train_domains': {'ptb': int(len(y_tr)),
                                'afdb': int(len(ya_tr)),
                                'cinc2017': int(len(yc_tr))},
              'val': {'ptb_fold9': int(len(y_v9)),
                      'afdb_val_records': afdb_val_recs,
                      'cinc_val': int(len(yc_v))},
              'go_line': 'afdb eval AUC >= 0.90 AND PTB fold-10 AUC >= 0.95',
              'seeds': {}}
    for seed in args.seeds:
        head, best, hist = train_head(F_tr, Y_tr, F_v9, y_v9, F_av, ya_v,
                                      seed, args.epochs)
        p_te = head.predict(F_te, batch_size=1024, verbose=0)[:, 0]
        p_ae = head.predict(F_ae, batch_size=1024, verbose=0)[:, 0]
        p_cv = head.predict(F_cv, batch_size=1024, verbose=0)[:, 0]
        p_v9 = head.predict(F_v9, batch_size=1024, verbose=0)[:, 0]
        theta_y = float(youden_threshold(y_v9, p_v9))
        theta_r90 = float(train_theta_at_recall(y_v9, p_v9, 0.90))
        m_te = op_metrics(y_te, p_te, theta_y)
        m_ae = op_metrics(ya_te, p_ae, theta_y)
        m_ce = op_metrics(yc_v, p_cv)
        result['seeds'][str(seed)] = {
            'best': best, 'history': hist[-10:],
            'ptb_fold10': m_te, 'afdb_eval': m_ae, 'cinc_val': m_ce,
            'theta_youden_fold9': round(theta_y, 4),
            'theta_R90_fold9': round(theta_r90, 4),
            'go_afdb': bool(m_ae['auc'] >= 0.90),
            'go_fold10': bool(m_te['auc'] >= 0.95),
            'go_pass': bool(m_ae['auc'] >= 0.90 and m_te['auc'] >= 0.95),
        }
        print('seed', seed, json.dumps(result['seeds'][str(seed)],
                                       ensure_ascii=False), flush=True)
    # device disclosure with the first seed head
    head.load_weights(str(OUT / f"head_seed{args.seeds[0]}.weights.h5"))
    result['device_disclosure_seed%s' % args.seeds[0]] = device_probs(
        head, feat, target)
    result['elapsed_s'] = round(time.time() - t0, 1)
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('DONE', json.dumps({s: result['seeds'][s]['go_pass']
                              for s in result['seeds']}), flush=True)


if __name__ == '__main__':
    main()
