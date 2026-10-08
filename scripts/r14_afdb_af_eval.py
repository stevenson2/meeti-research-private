#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14 optional: external AF validation of the AF head on MIT-BIH afdb.

Uses the pre-built runs/r14_afdb_prep/segments.npz (83,704 ten-second pure
AFIB/N windows, 23 records with signal; manual .atr rhythm labels).  A
balanced subsample is evaluated because the full sweep is 234 h of signal.

Task semantics: AFIB (positive) vs N = "all other rhythms" (negative),
matching the PTB-XL AF-vs-rest semantics; AFL/J/unknown segments excluded
by the prep.  Thresholds theta_tr / theta_R90tr are PTB-side (leak-free);
afdb is never used for selection.

Preprocessing: afdb ECG2 -> 100 Hz -> per-record amplitude scaled so the
record p99(|x|) matches the PTB-XL train lead-II median p99 (the AF head
expects raw-mV PTB amplitude).  Disclosed as a PC reconstruction.

Outputs: runs/r14_afdb_af_eval/result.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import spec_at_recall, train_theta_at_recall, youden_threshold  # noqa: E402
from r14_rr_baseline import detect_rr, features_from_rr  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
AFDB = Path('/home/devcontainers/ecg_data/afdb')
SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
OUT = ROOT / 'runs/r14_afdb_af_eval'
AF_ROOT = ROOT / 'runs/r14_af_head'
WIN250 = 2500


def load_model(seed):
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    import tensorflow as tf
    from models_patch.ptbxl_builders import build_ptbxl_large
    m = build_ptbxl_large(input_shape=(1000, 1), n_classes=1,
                          output_activation='sigmoid', name='r14_af')
    m.load_weights(str(AF_ROOT / f'af_seed{seed}' / 'best.weights.h5'))
    return m


def ptb_p99():
    tr = np.load(ROOT / 'runs/r14_af_prep/ptbxl_af_train.npz')
    return float(np.median(np.percentile(np.abs(tr['x']), 99, axis=1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-per-class', type=int, default=2000)
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--sample-seed', type=int, default=42)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    seg = np.load(SEG)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    rng = np.random.default_rng(args.sample_seed)
    sel_af = rng.choice(np.where(lab == 1)[0], args.n_per_class, replace=False)
    sel_n = rng.choice(np.where(lab == 0)[0], args.n_per_class, replace=False)
    sel = np.concatenate([sel_af, sel_n])
    rng.shuffle(sel)

    target = ptb_p99()
    xs, ys = [], []
    for r in np.unique(rec[sel]):
        idxs = sel[rec[sel] == r]
        obj = wfdb.rdrecord(str(AFDB / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        p99 = float(np.percentile(np.abs(sig), 99))
        scale = target / max(p99, 1e-9)
        for i in idxs:
            w = sig[start[i]:start[i] + WIN250]
            if len(w) < WIN250:
                continue
            w100 = resample_poly(w, 2, 5)  # 250 Hz -> 100 Hz
            xs.append((w100 * scale).astype(np.float32))
            ys.append(lab[i])
        print(f'read {r}: selected {len(idxs)} windows '
              f'(total {len(xs)})', flush=True)
    X = np.stack(xs)[..., None]
    y = np.array(ys, int)

    # RR features on the same windows (amplitude scale irrelevant)
    ent = np.array([features_from_rr(detect_rr(w[:, 0])[0])['rr_entropy']
                    for w in X], float)
    rr_auc = float(roc_auc_score(y[np.isfinite(ent)], ent[np.isfinite(ent)]))

    out = {
        'arm': 'r14_afdb_af_eval',
        'task': 'AFIB vs N(all other rhythms); AFL/J excluded by prep',
        'n_windows': int(len(y)), 'n_pos': int(y.sum()),
        'n_records': int(len(np.unique(rec[sel]))),
        'sample_seed': args.sample_seed,
        'ptb_p99_target': target,
        'preprocessing': 'afdb ECG2 250Hz -> 100Hz -> per-record p99 match '
                         'to PTB-XL train lead-II median p99',
        'rr_entropy_baseline_auc': round(rr_auc, 4),
        'seeds': {},
    }
    for seed in args.seeds:
        rj = json.loads((AF_ROOT / f'af_seed{seed}' / 'result.json').read_text())
        theta_tr = rj['metrics']['theta_tr_val_youden']
        theta_r90 = rj['metrics'].get('theta_R90tr_val')
        model = load_model(seed)
        p = model.predict(X, batch_size=256, verbose=0)[:, 0]
        a = float(roc_auc_score(y, p))
        ap = float(average_precision_score(y, p))
        s90 = spec_at_recall(y, p)
        entry = {
            'auc': round(a, 4), 'ap': round(ap, 4),
            'spec_at_R90_afdb': s90,
            'theta_tr_ptb': theta_tr,
            'R_at_theta_tr': round(float((p[y == 1] >= theta_tr).mean()), 4),
            'spec_at_theta_tr': round(float((p[y == 0] < theta_tr).mean()), 4),
            'prob_mean_pos': round(float(p[y == 1].mean()), 4),
            'prob_mean_neg': round(float(p[y == 0].mean()), 4),
        }
        if theta_r90 is not None:
            entry.update({
                'theta_R90tr_ptb': theta_r90,
                'R_at_theta_R90tr': round(
                    float((p[y == 1] >= theta_r90).mean()), 4),
                'spec_at_theta_R90tr': round(
                    float((p[y == 0] < theta_r90).mean()), 4),
            })
        out['seeds'][str(seed)] = entry
        print(seed, json.dumps(entry), flush=True)
    (OUT / 'result.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
