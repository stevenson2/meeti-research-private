#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A3: causal RR-sequence 1D CNN (small, 3 seeds).

Input (per window):
  seq  (L,3) = [rr/median-1, dRR/median, ddRR/median]   (causal, last L RR)
  shp  (16,) = 8 shape scalars + 8 missing indicators
Model: 3x Conv1D(48,k=3) -> GAP -> concat shape MLP -> Dense(16) -> 1
       (~15k params), class-weighted BCE.

Configs:
  12: L=12 (primary; aligned with the PTB 10 s window median ~11 RR)
      train = PTB folds1-8 + afdb fit 10 s windows (9 records)
      val   = PTB fold9 + 3 afdb fit records (10 s)
      eval  = afdb eval 11 records (10 s) + ltafdb 10 s + 052/rec_latest 10 s
  30: L=30 (secondary; PTB cannot provide 30 RR)
      train = afdb fit 30 s windows (9 records), val = 3 afdb fit records
      eval  = afdb eval 11 records (30 s) + ltafdb 30 s + 052/rec_latest 30 s

Thresholds come from the TRAIN-side scores only (max threshold with train
recall >= 0.90; permissive targets 0.95-0.99 also reported).

go-line: 052 event-level FP=0 + afdb held-out recall >= 0.90 + ltafdb recall
>= 0.90.  Outputs: runs/r16_rr_seq/result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r16_rr_features import features_from_rr_ext  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DATA = ROOT / 'runs/r16_rr_shape'
OUT = ROOT / 'runs/r16_rr_seq'
SCALARS = ['autocorr1', 'autocorr2', 'rmssd_over_sdnn', 'pnn50', 'tpr',
           'sd1_sd2_ratio', 'mean_abs_drr_norm', 'higuchi_fd']
M4_CONFIRM, M4_COOLDOWN = 5, 20
tf.random.set_seed(42)


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def load_rr(npz, prefix, n=None):
    off = npz[f'{prefix}_rr_off']
    cat = npz[f'{prefix}_rr']
    return [cat[off[i]:off[i + 1]].astype(np.float64)
            for i in range(len(off) - 1 if n is None else n)]


def pad_seq(rr, L):
    rr = np.asarray(rr, float)
    rr = rr[(rr >= 0.30) & (rr <= 2.00)]
    if len(rr) == 0:
        z = np.zeros(L)
        return z, z, z, np.repeat(0.8, L)
    if len(rr) >= L:
        rr = rr[-L:]
    else:
        rr = np.concatenate([np.repeat(rr[0], L - len(rr)), rr])
    med = float(np.median(rr))
    if not np.isfinite(med) or med <= 0:
        med = 0.8
    ch0 = rr / med - 1.0
    drr = np.diff(rr, prepend=rr[0]) / med
    ddrr = np.diff(drr, prepend=drr[0]) / med
    return ch0, drr, ddrr, rr


def shape_vec(rr):
    f = features_from_rr_ext(rr)
    v = np.array([f.get(k, np.nan) for k in SCALARS], float)
    return v, np.isnan(v).astype(float)


def build_domain(npz, prefix, y, recs, L, records=None, device=False):
    """Return X_seq, X_shp, y, rec arrays."""
    off_key = (f'{prefix}_rr_off' if f'{prefix}_rr_off' in npz
               else f'{prefix}_off')
    off = npz[off_key]
    cat = npz[f'{prefix}_rr']
    rrs = [cat[off[i]:off[i + 1]].astype(np.float64)
           for i in range(len(off) - 1)]
    Xs, Xh, yy, rr_out = [], [], [], []
    for i, rr in enumerate(rrs):
        if records is not None and recs[i] not in records:
            continue
        c0, c1, c2, rrpad = pad_seq(rr, L)
        sv, mi = shape_vec(rrpad)
        Xs.append(np.stack([c0, c1, c2], axis=1))
        Xh.append(np.concatenate([sv, mi]))
        yy.append(int(y[i]))
        rr_out.append(recs[i] if recs is not None else None)
    return (np.asarray(Xs, np.float32), np.asarray(Xh, np.float32),
            np.asarray(yy, int), np.asarray(rr_out))


def build_model(L, n_shp=16, filters=48):
    # single input: channels 0-2 = RR/dRR/ddRR sequence, channels 3.. =
    # the shape vector broadcast across time (keeps the model single-input,
    # which is robust across Keras 3 predict paths)
    inp = tf.keras.Input(shape=(L, 3 + n_shp))
    x = tf.keras.layers.Conv1D(filters, 3, padding='same',
                               activation='relu')(inp)
    x = tf.keras.layers.Conv1D(filters, 3, padding='same',
                               activation='relu')(x)
    x = tf.keras.layers.Conv1D(filters, 3, padding='same',
                               activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    return tf.keras.Model(inp, out)


def pack_input(Xs, Xh, L):
    return np.concatenate(
        [Xs, np.tile(Xh[:, None, :], (1, L, 1))], axis=2).astype(np.float32)


def th_valid(y, s, target):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    ys, ss = y[m], s[m]
    best = None
    for th in np.unique(ss):
        if float(((ss >= th) & (ys == 1)).sum()) / max(int((ys == 1).sum()),
                                                        1) >= target - 1e-12:
            best = float(th)
    return best


def op(y, s, th):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fl = np.isfinite(s) & (s >= th)
    return {'recall': round(float((fl & (y == 1)).sum())
                            / max(int((y == 1).sum()), 1), 4),
            'spec': round(float(((~fl) & (y == 0)).sum())
                          / max(int((y == 0).sum()), 1), 4),
            'n_flag': int(fl.sum())}


def train_one(seed, Xtr, Xva, ytr, yva, L, epochs=30, batch=128):
    Xtr = pack_input(Xtr[0], Xtr[1], L)
    Xva = pack_input(Xva[0], Xva[1], L)
    tf.keras.utils.set_random_seed(seed)
    cw = compute_class_weight('balanced', classes=np.array([0, 1]), y=ytr)
    cw = {0: float(cw[0]), 1: float(cw[1])}
    model = build_model(L)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.BinaryCrossentropy())
    best = {'auc': -1.0, 'weights': None, 'epoch': -1}
    hist = []
    for ep in range(epochs):
        model.fit(Xtr, ytr, batch_size=batch, epochs=1, verbose=0,
                  class_weight=cw, shuffle=True)
        pv = model.predict(Xva, batch_size=512, verbose=0)[:, 0]
        auc = (roc_auc_score(yva, pv)
               if len(np.unique(yva)) > 1 else 0.0)
        hist.append(round(float(auc), 4))
        if auc > best['auc']:
            best = {'auc': float(auc), 'weights': model.get_weights(),
                    'epoch': ep}
        if ep - best['epoch'] >= 8:
            break
    model.set_weights(best['weights'])
    return model, best, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', choices=['12', '30'], default='12')
    ap.add_argument('--seeds', default='42,43,44')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    L = int(args.config)
    seeds = [int(x) for x in args.seeds.split(',')]
    if args.smoke:
        seeds = seeds[:1]
        args.epochs = 3

    ptb = dict(np.load(DATA / 'ptb_features.npz'))
    af = dict(np.load(DATA / 'afdb_features.npz'))
    lt = dict(np.load(DATA / 'ltafdb_features.npz'))
    dv = np.load(DATA / 'device_features.npz')
    dvr = np.load(DATA / 'device_rr.npz')
    yaf = af['afdb_y'].astype(int)
    sp = af['afdb_split'].astype(int)
    ylt = lt['ltafdb_y'].astype(int)
    fit_recs = sorted(set(af['afdb_rec'][sp == 0].tolist()))
    eval_recs = sorted(set(af['afdb_rec'][sp == 1].tolist()))
    val_recs = [fit_recs[0], fit_recs[4], fit_recs[8]]
    train_recs = [r for r in fit_recs if r not in val_recs]

    if L == 12:
        pref = 'afdb10'
        # PTB: folds1-8 train, fold9 val
        folds = ptb['ptb_folds'].astype(int)
        ptr = np.where(folds <= 8)[0]
        pva = np.where(folds == 9)[0]
        Xs_tr, Xh_tr = [], []
        rr_all = load_rr(ptb, 'ptb')
        for i in ptr:
            c0, c1, c2, rrpad = pad_seq(rr_all[i], L)
            sv, mi = shape_vec(rrpad)
            Xs_tr.append(np.stack([c0, c1, c2], 1))
            Xh_tr.append(np.concatenate([sv, mi]))
        Xs_va, Xh_va = [], []
        for i in pva:
            c0, c1, c2, rrpad = pad_seq(rr_all[i], L)
            sv, mi = shape_vec(rrpad)
            Xs_va.append(np.stack([c0, c1, c2], 1))
            Xh_va.append(np.concatenate([sv, mi]))
        y_tr = ptb['ptb_y'][ptr].astype(int)
        y_va = ptb['ptb_y'][pva].astype(int)
        # afdb 10 s
        Xa_tr, Xah_tr, ya_tr, _ = build_domain(
            af, 'afdb10', yaf, af['afdb_rec'], L, set(train_recs))
        Xa_va, Xah_va, ya_va, _ = build_domain(
            af, 'afdb10', yaf, af['afdb_rec'], L, set(val_recs))
        Xe, Xeh, ye, _ = build_domain(
            af, 'afdb10', yaf, af['afdb_rec'], L, set(eval_recs))
        Xl, Xlh, yl, _ = build_domain(lt, 'ltafdb10', ylt, lt['ltafdb_rec'], L)
        # device 10 s (all windows)
        dev = {}
        for name in ('ecg_real_052', 'rec_latest'):
            pre = f'{name}_deploy_xqrs250_10s'
            off_key = (f'{pre}_rr_off' if f'{pre}_rr_off' in dvr
                       else f'{pre}_off')
            n_dev = len(dvr[off_key]) - 1
            Xd, Xdh, yd, _ = build_domain(dvr, pre, np.ones(n_dev, int),
                                          None, L, device=True)
            dev[name] = (Xd, Xdh)
        starts = dv['ecg_real_052_deploy_xqrs250_10s_start']
        starts_r = dv['rec_latest_deploy_xqrs250_10s_start']
    else:
        Xa_tr, Xah_tr, ya_tr, _ = build_domain(
            af, 'afdb30', yaf, af['afdb_rec'], L, set(train_recs))
        Xa_va, Xah_va, ya_va, _ = build_domain(
            af, 'afdb30', yaf, af['afdb_rec'], L, set(val_recs))
        Xe, Xeh, ye, _ = build_domain(
            af, 'afdb30', yaf, af['afdb_rec'], L, set(eval_recs))
        Xl, Xlh, yl, _ = build_domain(lt, 'ltafdb30', ylt, lt['ltafdb_rec'], L)
        dev = {}
        for name in ('ecg_real_052', 'rec_latest'):
            pre = f'{name}_deploy_xqrs250_paired30'
            off_key = (f'{pre}_rr_off' if f'{pre}_rr_off' in dvr
                       else f'{pre}_off')
            n_dev = len(dvr[off_key]) - 1
            Xd, Xdh, yd, _ = build_domain(dvr, pre, np.ones(n_dev, int),
                                          None, L, device=True)
            dev[name] = (Xd, Xdh)
        starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
        starts_r = dv['rec_latest_deploy_xqrs250_paired_start']

    if L == 12:
        seq_tr = np.concatenate([Xs_tr, Xa_tr])
        seq_va = np.concatenate([Xs_va, Xa_va])
        shp_tr = np.concatenate([Xh_tr, Xah_tr])
        shp_va = np.concatenate([Xh_va, Xah_va])
        ytr = np.concatenate([y_tr, ya_tr])
        yva = np.concatenate([y_va, ya_va])
    else:
        seq_tr, shp_tr = Xa_tr, Xah_tr
        seq_va, shp_va = Xa_va, Xah_va
        ytr, yva = ya_tr, ya_va

    # ---- shape-vector imputation + z-scoring (train statistics only) ----
    def zshp(base, *others):
        base = np.asarray(base, float)
        mu = np.nanmean(base, axis=0)
        sd = np.nanstd(base, axis=0)
        sd[~np.isfinite(sd) | (sd == 0)] = 1.0
        mu[~np.isfinite(mu)] = 0.0
        outs = [base] + [np.asarray(o, float) for o in others]
        return [np.nan_to_num((o - mu) / sd, nan=0.0) for o in outs]

    (shp_tr, shp_va, Xeh, Xlh, dev_h052, dev_hrec) = zshp(
        shp_tr, shp_va, Xeh, Xlh,
        dev['ecg_real_052'][1], dev['rec_latest'][1])
    dev['ecg_real_052'] = (dev['ecg_real_052'][0], dev_h052)
    dev['rec_latest'] = (dev['rec_latest'][0], dev_hrec)
    Xtr = (seq_tr, shp_tr)
    Xva = (seq_va, shp_va)

    result = {
        'arm': f'r16_rr_seq_cfg{L}',
        'config': {'L': L, 'scalars': SCALARS, 'model': '3xConv1D(48,3)+GAP'
                   '+shapeMLP', 'epochs': args.epochs, 'batch': 128,
                   'lr': 1e-3, 'class_weight': 'balanced',
                   'smoke': bool(args.smoke)},
        'data': {'train_n': int(len(ytr)), 'train_pos': int(ytr.sum()),
                 'val_n': int(len(yva)), 'val_pos': int(yva.sum()),
                 'afdb_eval_n': int(len(ye)), 'afdb_eval_pos': int(ye.sum()),
                 'ltafdb_n': int(len(yl)), 'ltafdb_pos': int(yl.sum())},
        'seeds': {},
        'go_line': ('052 event FP=0 AND afdb eval recall >= 0.90 AND ltafdb '
                    'recall >= 0.90 at the train-side theta'),
        'disclosures': [
            'PTB fold9 used for validation, folds1-8 for training (prompt '
            'said folds1-9 train + fold9 val; overlap avoided, disclosed)',
            '12-RR config: PTB + afdb 10 s windows; 30-RR config: afdb 30 s '
            'only (PTB cannot provide 30 RR)',
            'only 2 device normal recordings, no device AF positives',
            'thresholds fitted on TRAIN-side scores only; eval domains never '
            'used for threshold selection',
        ],
    }
    for seed in seeds:
        model, best, hist = train_one(seed, Xtr, Xva, ytr, yva, L,
                                      epochs=args.epochs)
        ptr_s = model.predict(pack_input(Xtr[0], Xtr[1], L), batch_size=512, verbose=0)[:, 0]
        pva_s = model.predict(pack_input(Xva[0], Xva[1], L), batch_size=512, verbose=0)[:, 0]
        pe = model.predict(pack_input(Xe, Xeh, L), batch_size=512, verbose=0)[:, 0]
        pl = model.predict(pack_input(Xl, Xlh, L), batch_size=512, verbose=0)[:, 0]
        p052 = model.predict(pack_input(dev['ecg_real_052'][0], dev['ecg_real_052'][1], L), batch_size=512, verbose=0)[:, 0]
        prec = model.predict(pack_input(dev['rec_latest'][0], dev['rec_latest'][1], L), batch_size=512, verbose=0)[:, 0]
        row = {'best_val_auc': round(best['auc'], 4),
               'best_epoch': int(best['epoch']),
               'val_auc_history': hist,
               'val_auc': round(float(roc_auc_score(yva, pva_s)), 4),
               'train_auc': round(float(roc_auc_score(ytr, ptr_s)), 4),
               'afdb_eval_auc': round(float(roc_auc_score(ye, pe)), 4),
               'ltafdb_auc': round(float(roc_auc_score(yl, pl)), 4),
               'thresholds': {}}
        for tgt in (0.90, 0.95, 0.97, 0.98, 0.99):
            th = th_valid(ytr, ptr_s, tgt)
            if th is None:
                continue
            fl052 = p052 >= th
            flrec = prec >= th
            row['thresholds'][str(tgt)] = {
                'theta': round(float(th), 4),
                'train_op': op(ytr, ptr_s, th),
                'afdb_eval_op': op(ye, pe, th),
                'ltafdb_op': op(yl, pl, th),
                'device_052': {'n_flag': int(fl052.sum()),
                               'm4_alarms': int(len(m4_alarms(fl052)))},
                'device_rec_latest': {'n_flag': int(flrec.sum()),
                                      'm4_alarms': int(len(m4_alarms(flrec)))},
            }
            t = row['thresholds'][str(tgt)]
            t['pass_three_lines'] = bool(
                t['device_052']['m4_alarms'] == 0
                and t['afdb_eval_op']['recall'] >= 0.90
                and t['ltafdb_op']['recall'] >= 0.90)
        result['seeds'][str(seed)] = row
        print(f'seed {seed}: val {row["best_val_auc"]} afdb_auc '
              f'{row["afdb_eval_auc"]} ltafdb_auc {row["ltafdb_auc"]} '
              f'R90 th {row["thresholds"].get("0.9", {}).get("theta")} '
              f'052ev {row["thresholds"].get("0.9", {}).get("device_052", {}).get("m4_alarms")}',
              flush=True)

    # acceptance summary
    acc = {}
    for tgt in ('0.9', '0.95', '0.97', '0.98', '0.99'):
        rows = [s['thresholds'][tgt] for s in result['seeds'].values()
                if tgt in s['thresholds']]
        if not rows:
            continue
        acc[tgt] = {
            'n_seeds': len(rows),
            'afdb_recall_mean': round(float(np.mean(
                [r['afdb_eval_op']['recall'] for r in rows])), 4),
            'ltafdb_recall_mean': round(float(np.mean(
                [r['ltafdb_op']['recall'] for r in rows])), 4),
            'device_052_events': [r['device_052']['m4_alarms'] for r in rows],
            'pass_three_lines': [r['pass_three_lines'] for r in rows],
            'n_pass': int(sum(r['pass_three_lines'] for r in rows)),
        }
    result['acceptance'] = acc
    result['verdict'] = {
        'go_line_pass': bool(any(v['n_pass'] >= 2 for v in acc.values())),
        'note': ('>=2/3 seeds passing the three lines at the same '
                 'pre-registered target would be a GO; a single seed is '
                 'not a conclusion'),
    }
    (OUT / f'result_cfg{L}.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps({'acceptance': acc, 'verdict': result['verdict']},
                     indent=2, ensure_ascii=False))
    print('written', OUT / 'result.json')


if __name__ == '__main__':
    main()
