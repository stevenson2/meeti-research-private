#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A4: AF-head repair - full-network fine-tune / DANN / domain aug.

Base: R14 AF head (build_ptbxl_large, weights runs/r14_af_head/af_seed42/
best.weights.h5).  R15-B3 showed frozen-trunk + readout-only adaptation fails
(afdb held-out AUC 0.74); this arm trains the WHOLE network with a low LR.

Arms:
  finetune : trunk 1e-5 / head 1e-4, 10-15 epochs, early stop on
             mean(PTB fold9 AUC, afdb held-out-val AUC)
  dann     : same + gradient-reversal domain classifier (domains PTB / afdb /
             CinC2017), lambda in {0.1, 0.3}, ramped over training
  aug      : fine-tune + domain augmentation (amplitude 0.5-2x, polarity
             flip, baseline drift, Gaussian noise)

Train pool: PTB folds1-8 + afdb fit 9 records (5k AF / 5k N) + CinC2017
A/N/O + aftdb 400.  Val: PTB fold9 + 3 afdb fit records.
Eval: PTB fold-10 / afdb held-out 11 records / ltafdb sample / CinC val /
device 052 + rec_latest (disclosure only).

go-line: afdb held-out AUC >= 0.90 AND PTB fold-10 AUC >= 0.95.
Outputs: runs/r16_af_head_adapt/{result.json, ckpt_*.weights.h5}
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
OUT = ROOT / 'runs/r16_af_head_adapt'
PTB = ROOT / 'runs/r14_af_prep'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')
SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
LTA_SEG = ROOT / 'runs/r15_ltafdb_prep/segments.npz'
POOL = ROOT / 'runs/r15_af_pool'
BASE_WEIGHTS = ROOT / 'runs/r14_af_head/af_seed42/best.weights.h5'
P99_TARGET = 0.5934749841690063
WIN = 1000


# --------------------------------------------------------------- data
def scale_p99(x, target):
    p = np.percentile(np.abs(x), 99, axis=1, keepdims=True)
    return (x * (target / np.maximum(p, 1e-9))).astype(np.float32)


def afdb_windows(records, n_per_class, target, seed, cache=None):
    if cache and Path(cache).exists():
        d = np.load(cache)
        return d['x'], d['y'].astype(int)
    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for L in (1, 0):
        cand_all = [i for i in np.where(lab == L)[0] if rec[i] in records]
        rng.shuffle(cand_all)
        take = cand_all[:n_per_class]
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
                xs.append(resample_poly(w, 2, 5).astype(np.float32))
                ys.append(L)
    X = np.stack(xs)
    X = scale_p99(X, target)
    y = np.array(ys, np.int8)
    if cache:
        np.savez_compressed(cache, x=X, y=y)
    return X, y


def ltafdb_windows(n_per_class, target, seed=42,
                   cache=OUT / 'ltafdb_eval.npz'):
    if cache.exists():
        d = np.load(cache)
        return d['x'], d['y'].astype(int)
    seg = np.load(LTA_SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for r in sorted(set(rec)):
        idx = np.where(rec == r)[0]
        obj = wfdb.rdrecord(str(LTAFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        for L in (1, 0):
            cand = idx[lab[idx] == L]
            if len(cand) == 0:
                continue
            take = rng.choice(cand, min(n_per_class, len(cand)),
                              replace=False)
            for i in take:
                w = sig[start[i]:start[i] + 1280]
                if len(w) < 1280:
                    continue
                xs.append(resample_poly(w, 25, 32).astype(np.float32))
                ys.append(L)
        print('ltafdb eval', r, len(ys), flush=True)
    X = scale_p99(np.stack(xs), target)
    y = np.array(ys, np.int8)
    np.savez_compressed(cache, x=X, y=y)
    return X, y


def device_windows(target):
    """100 Hz deploy-chain windows for 052 / rec_latest (both polarities)."""
    sys.path.insert(0, str(ROOT / 'scripts'))
    from r15_laneA_rr_gate import deploy_chain250, read_ecgr  # noqa
    electron = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                    'ecg-programme-tju-flex.electron-master')
    files = {'ecg_real_052': electron / 'pc_tools/ecg_dl/data/real/'
             'ecg_real_052.ecgr',
             'rec_latest': electron / 'rec_latest.ecgr'}
    out = {}
    for name, path in files.items():
        ch250, fs, dur = deploy_chain250(path)
        ch100 = resample_poly(ch250, 2, 5)
        n = (len(ch100) - WIN) // 100 + 1
        X = np.stack([ch100[i * 100:i * 100 + WIN]
                      for i in range(n)]).astype(np.float32)
        X = scale_p99(X, target)[..., None]
        out[name] = {'x': X, 'x_neg': -X, 'dur_s': dur}
    return out


# --------------------------------------------------------------- model
def load_base():
    m = build_ptbxl_large(input_shape=(1000, 1), n_classes=1,
                          output_activation='sigmoid', name='r14_af')
    m.load_weights(str(BASE_WEIGHTS))
    return m


def is_head_var(v):
    path = getattr(v, 'path', v.name)
    return path.startswith('fc1') or path.startswith('out_multi')


def make_optimizers(lr_trunk=1e-5, lr_head=1e-4):
    return (tf.keras.optimizers.Adam(lr_trunk),
            tf.keras.optimizers.Adam(lr_head))


def augment_batch(x, rng, target):
    """Domain augmentation: amplitude, polarity, baseline drift, noise."""
    amp = rng.uniform(0.5, 2.0, size=(len(x), 1, 1)).astype(np.float32)
    pol = rng.choice([-1.0, 1.0], size=(len(x), 1, 1)).astype(np.float32)
    x = x * amp * pol
    t = np.arange(x.shape[1], dtype=np.float32) / 100.0
    drift = (rng.uniform(0.0, 0.3, size=(len(x), 1))
             * np.sin(2 * np.pi * rng.uniform(0.05, 0.5, size=(len(x), 1)) * t
                      + rng.uniform(0, 6.28, size=(len(x), 1)))).astype(
                          np.float32)[..., None]
    x = x + drift * target
    noise = rng.normal(0, 0.02 * target, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def grl_layer(x, lambd):
    @tf.custom_gradient
    def _rev(v):
        y = tf.identity(v)

        def grad(dy):
            return -lambd * dy
        return y, grad
    return _rev(x)


def build_adapt_model(n_domains=3):
    base = load_base()
    inp = base.input
    af_out = base.output
    gap = base.get_layer('gap').output
    h = tf.keras.layers.Dense(32, activation='relu')(gap)
    h = tf.keras.layers.Lambda(
        lambda z: grl_layer(z, 1.0), name='grl')(h)
    dom_out = tf.keras.layers.Dense(n_domains, activation='softmax',
                                    name='dom_out')(h)
    return tf.keras.Model(inp, [af_out, dom_out]), base


def train_loop(model, Xtr, ytr, dtr, Xva, yva, Xv2, yv2, arm, seed,
               epochs, batch, lam, use_aug, smoke=False):
    tf.keras.utils.set_random_seed(seed)
    rng = np.random.default_rng(seed)
    trunk_opt, head_opt = make_optimizers()
    bce = tf.keras.losses.BinaryCrossentropy()
    cce = tf.keras.losses.SparseCategoricalCrossentropy()
    train_vars = model.trainable_variables
    trunk_vars = [v for v in train_vars if not is_head_var(v)]
    head_vars = [v for v in train_vars if is_head_var(v)]
    has_dom = arm == 'dann'

    @tf.function(reduce_retracing=True)
    def step(xb, yb, db, lam_t):
        with tf.GradientTape() as tape:
            if has_dom:
                p_af, p_dom = model(xb, training=True)
                loss_af = bce(yb, p_af)
                loss_dom = cce(db, p_dom)
                loss = loss_af + lam_t * loss_dom
            else:
                p_af = model(xb, training=True)
                loss = bce(yb, p_af)
        grads = tape.gradient(loss, train_vars)
        gt = [(g, v) for g, v in zip(grads, train_vars)
              if not is_head_var(v) and g is not None]
        gh = [(g, v) for g, v in zip(grads, train_vars)
              if is_head_var(v) and g is not None]
        trunk_opt.apply_gradients(gt)
        head_opt.apply_gradients(gh)
        return loss

    n = len(ytr)
    steps = max(n // batch, 1)
    best = {'score': -1.0, 'weights': None, 'epoch': -1,
            'ptb9_auc': None, 'afdb_val_auc': None}
    hist = []
    for ep in range(epochs):
        lam_t = tf.constant(
            lam * min(1.0, (ep + 1) / max(epochs * 0.5, 1)), tf.float32)
        idx = rng.permutation(n)
        t0 = time.time()
        losses = []
        for s in range(steps):
            b = idx[s * batch:(s + 1) * batch]
            xb = Xtr[b]
            if use_aug:
                xb = augment_batch(xb, rng, P99_TARGET)
            losses.append(float(step(xb, ytr[b].astype(np.float32),
                                     dtr[b].astype(np.int32), lam_t)))
        p1 = model.predict(Xva, batch_size=512, verbose=0)
        p2 = model.predict(Xv2, batch_size=512, verbose=0)
        p1 = p1[0][:, 0] if isinstance(p1, list) else p1[:, 0]
        p2 = p2[0][:, 0] if isinstance(p2, list) else p2[:, 0]
        a1 = roc_auc_score(yva, p1) if len(np.unique(yva)) > 1 else 0.0
        a2 = roc_auc_score(yv2, p2) if len(np.unique(yv2)) > 1 else 0.0
        score = 0.5 * (a1 + a2)
        hist.append({'ep': ep, 'loss': round(float(np.mean(losses)), 5),
                     'ptb9_auc': round(float(a1), 4),
                     'afdb_val_auc': round(float(a2), 4),
                     'score': round(float(score), 4),
                     'elapsed_s': round(time.time() - t0, 1)})
        print(f'[{arm} seed{seed}] ep{ep} loss {np.mean(losses):.4f} '
              f'ptb9 {a1:.4f} afdbval {a2:.4f} score {score:.4f} '
              f'({time.time()-t0:.0f}s)', flush=True)
        if score > best['score']:
            best = {'score': float(score), 'epoch': ep,
                    'weights': model.get_weights(),
                    'ptb9_auc': float(a1), 'afdb_val_auc': float(a2)}
        if smoke and ep >= 1:
            break
    model.set_weights(best['weights'])
    return model, best, hist


def eval_model(model, X, y, name):
    p = model.predict(X, batch_size=512, verbose=0)
    p = p[0][:, 0] if isinstance(p, list) else p[:, 0]
    out = {'n': int(len(y)), 'n_pos': int(np.sum(y))}
    if len(np.unique(y)) > 1:
        out['auc'] = round(float(roc_auc_score(y, p)), 4)
        out['ap'] = round(float(average_precision_score(y, p)), 4)
        sp = spec_at_recall(y, p, 0.90)
        out['spec_at_R90'] = sp
    return out, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['finetune', 'dann', 'aug'],
                    default='finetune')
    ap.add_argument('--lam', type=float, default=0.3)
    ap.add_argument('--seeds', default='42')
    ap.add_argument('--epochs', type=int, default=15)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--afdb-train-per-class', type=int, default=5000)
    ap.add_argument('--afdb-eval-per-class', type=int, default=600)
    ap.add_argument('--lta-per-class', type=int, default=30)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        args.epochs = 2
        args.afdb_train_per_class = 500
        args.afdb_eval_per_class = 100
        args.lta_per_class = 5

    tr = np.load(PTB / 'ptbxl_af_train.npz')
    te = np.load(PTB / 'ptbxl_af_test.npz')
    folds = tr['folds'].astype(int)
    x_tr = tr['x'][..., None]
    y_tr = tr['y_af'].astype(int)
    m_tr = folds <= 8
    m_va = folds == 9
    Xp, yp = x_tr[m_tr], y_tr[m_tr]
    Xp9, yp9 = x_tr[m_va], y_tr[m_va]
    Xp10, yp10 = te['x'][..., None], te['y_af'].astype(int)

    afnpz = np.load(ROOT / 'runs/r16_rr_shape/afdb_features.npz')
    fit_recs = sorted(set(afnpz['afdb_rec'][afnpz['afdb_split'] == 0].tolist()))
    eval_recs = sorted(set(afnpz['afdb_rec'][afnpz['afdb_split'] == 1].tolist()))
    val_recs = [fit_recs[0], fit_recs[4], fit_recs[8]]
    train_recs = [r for r in fit_recs if r not in val_recs]
    Xa, ya = afdb_windows(train_recs, args.afdb_train_per_class, P99_TARGET,
                          seed=42, cache=OUT / 'afdb_train.npz')
    Xa_v, ya_v = afdb_windows(val_recs, 300, P99_TARGET, seed=43,
                              cache=OUT / 'afdb_val.npz')
    Xa_e, ya_e = afdb_windows(eval_recs, args.afdb_eval_per_class,
                              P99_TARGET, seed=44,
                              cache=OUT / 'afdb_eval.npz')
    Xa, Xa_v, Xa_e = Xa[..., None], Xa_v[..., None], Xa_e[..., None]
    d = np.load(POOL / 'cinc2017.npz')
    Xc, yc = d['x'][..., None], d['y'].astype(int)
    d = np.load(POOL / 'aftdb.npz')
    Xt, yt = d['x'][..., None], d['y'].astype(int)

    Xtr = np.concatenate([Xp, Xa, Xc, Xt])
    ytr = np.concatenate([yp, ya, yc, yt])
    dtr = np.concatenate([np.zeros(len(yp), int), np.ones(len(ya), int),
                          np.full(len(yc), 2, int),
                          np.ones(len(yt), int)])
    Xl, yl = ltafdb_windows(args.lta_per_class, P99_TARGET,
                            cache=OUT / f'ltafdb_eval_{args.lta_per_class}.npz')
    Xl = Xl[..., None]
    dev = device_windows(P99_TARGET)

    result = {
        'arm': f'r16_af_head_adapt_{args.arm}',
        'config': {'epochs': args.epochs, 'batch': args.batch,
                   'lam': args.lam, 'lr_trunk': 1e-5, 'lr_head': 1e-4,
                   'seeds': args.seeds, 'smoke': args.smoke,
                   'afdb_train_per_class': args.afdb_train_per_class,
                   'n_domains': 3},
        'data': {'train_n': int(len(ytr)), 'train_pos': int(ytr.sum()),
                 'ptb_fold9_n': int(len(yp9)),
                 'afdb_val_n': int(len(ya_v)),
                 'afdb_eval_n': int(len(ya_e)),
                 'ltafdb_n': int(len(yl))},
        'go_line': 'afdb held-out AUC >= 0.90 AND PTB fold-10 AUC >= 0.95',
        'seeds': {},
        'disclosures': [
            'base = R14 AF head seed42; R15-B3 readout-only adaptation failed',
            'afdb validation uses 3 held-out fit records (not eval records)',
            'device 052/rec_latest are disclosure only (no device AF '
            'positives); polarity reported both ways',
            'ltafdb eval samples ECG2, 30/class/record',
        ],
    }
    for seed in [int(s) for s in args.seeds.split(',')]:
        model, base = build_adapt_model() if args.arm == 'dann' else (load_base(), None)
        model, best, hist = train_loop(
            model, Xtr, ytr, dtr, Xp9, yp9, Xa_v, ya_v, args.arm, seed,
            args.epochs, args.batch, args.lam, args.arm == 'aug',
            smoke=args.smoke)
        best_meta = {k: v for k, v in best.items() if k != 'weights'}
        row = {'best': best_meta, 'history': hist}
        row['ptb_fold10'], _ = eval_model(model, Xp10, yp10, 'ptb10')
        row['afdb_eval'], pe = eval_model(model, Xa_e, ya_e, 'afdb')
        row['ltafdb'], pl = eval_model(model, Xl, yl, 'ltafdb')
        row['cinc_val'], _ = eval_model(model, Xc, yc, 'cinc')
        # device disclosure (threshold from PTB fold9 R90)
        p9 = model.predict(Xp9, batch_size=512, verbose=0)
        p9 = p9[0][:, 0] if isinstance(p9, list) else p9[:, 0]
        th_r90 = train_theta_at_recall(yp9, p9, 0.90)
        th_youden = youden_threshold(yp9, p9)
        dev_out = {}
        for name, dd in dev.items():
            pd = model.predict(dd['x'], batch_size=512, verbose=0)
            pd = pd[0][:, 0] if isinstance(pd, list) else pd[:, 0]
            pd_neg = model.predict(dd['x_neg'], batch_size=512, verbose=0)
            pd_neg = (pd_neg[0][:, 0] if isinstance(pd_neg, list)
                      else pd_neg[:, 0])
            dev_out[name] = {
                'pos_polarity': {'max': round(float(pd.max()), 4),
                                 'median': round(float(np.median(pd)), 4),
                                 'frac_ge_theta_R90': round(float(
                                     (pd >= th_r90).mean()), 4)},
                'neg_polarity': {'max': round(float(pd_neg.max()), 4),
                                 'median': round(float(np.median(pd_neg)), 4),
                                 'frac_ge_theta_R90': round(float(
                                     (pd_neg >= th_r90).mean()), 4)},
            }
        row['device'] = {'theta_R90_fold9': round(float(th_r90), 4),
                         'theta_youden_fold9': round(float(th_youden), 4),
                         'windows': dev_out}
        row['go_afdb'] = bool(row['afdb_eval'].get('auc', 0) >= 0.90)
        row['go_fold10'] = bool(row['ptb_fold10'].get('auc', 0) >= 0.95)
        row['go_pass'] = bool(row['go_afdb'] and row['go_fold10'])
        result['seeds'][str(seed)] = row
        model.save_weights(str(OUT / f'ckpt_{args.arm}_seed{seed}.weights.h5'))
        print(json.dumps({'seed': seed, 'best': best_meta,
                          'fold10': row['ptb_fold10'].get('auc'),
                          'afdb': row['afdb_eval'].get('auc'),
                          'ltafdb': row['ltafdb'].get('auc'),
                          'go': row['go_pass']}, ensure_ascii=False),
              flush=True)
    result['verdict'] = {
        'n_seeds': len(result['seeds']),
        'n_go': int(sum(1 for r in result['seeds'].values()
                        if r['go_pass'])),
        'pass': bool(any(r['go_pass'] for r in result['seeds'].values())),
    }
    tag = args.arm if args.arm != 'dann' else f'dann_lam{args.lam}'
    (OUT / f'result_{tag}.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps(result['verdict'], ensure_ascii=False))
    print('written', OUT / f'result_{tag}.json')


if __name__ == '__main__':
    main()
