#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Report-only: PTB val event-level check of the V1 AND-gate (taub=0.2, taur=0.5).

PTB val patients (gold labels) were NOT used for gate selection; this is extra
robustness evidence for the M4 fix. Protocol identical to m4_afe_event_eval.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')

import numpy as np

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')

from train_dual_head_v3_small import build_full_model  # noqa: E402
from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402

T, CONFIRM, COOLDOWN = 10, 5, 20
RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/m4_ptbval_gate_check')
TAU_B, TAU_R = 0.2, 0.5


def window_mean(p):
    n = len(p)
    out = np.zeros(n, dtype=np.float32)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    for i in range(n):
        lo = max(0, i - T + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def alarm_triggers(flags):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - CONFIRM + 1)
        if flags[lo:i + 1].any() and i - last >= COOLDOWN:
            trig.append(i)
            last = i
    return trig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model, beat_probe, _ = build_full_model(lr=1e-4, lam=0.2)
    model.load_weights(str(RUN / 'full.weights.h5'))
    rec_probe = build_record_probe(model)
    print('loaded', flush=True)

    beats, labels, rids = load_arrays('ptb')
    pv = np.asarray(compute_ptb_split()['val'], dtype=bool)
    beats = np.asarray(beats[pv], dtype=np.float32)
    labels = np.asarray(labels[pv])
    rids = np.asarray(rids[pv])
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)

    rows = []
    tp0 = tp1 = 0
    fp0 = fp1 = 0
    n_abn = n_norm = 0
    all_pb, all_pr, all_rid, all_lab = [], [], [], []
    for ui, rid in enumerate(uniq):
        s, e = first[ui], first[ui] + counts[ui]
        rb = beats[s:e]
        n = len(rb)
        pb = beat_probe.predict(rb[..., None], batch_size=512, verbose=0)[:, 1]
        x = np.zeros((n, T, rb.shape[1], 1), dtype=np.float32)
        m = np.zeros((n, T, 1), dtype=np.float32)
        for i in range(n):
            lo = max(0, i - T + 1)
            x[i, -(i - lo + 1):] = rb[lo:i + 1, ..., None]
            m[i, -(i - lo + 1):] = 1.0
        pr = rec_probe.predict([x, m], batch_size=512, verbose=0)[:, 1]
        hbw = window_mean(pb)
        all_pb.append(pb)
        all_pr.append(pr)
        all_rid.append(np.full(n, rid))
        all_lab.append(np.full(n, int(labels[s:e].max())))
        t0 = alarm_triggers(pr >= TAU_R)
        t1 = alarm_triggers((pr >= TAU_R) & (hbw >= TAU_B))
        lab = int(labels[s:e].max())
        n_abn += lab == 1
        n_norm += lab == 0
        tp0 += (lab == 1) and len(t0) > 0
        tp1 += (lab == 1) and len(t1) > 0
        fp0 += (lab == 0) and len(t0) > 0
        fp1 += (lab == 0) and len(t1) > 0
        rows.append({'record': str(rid), 'label': lab, 'n_beats': int(n),
                     'trig_baseline': len(t0), 'trig_gate': len(t1)})
        if ui % 20 == 0:
            print(f'{ui}/{len(uniq)}', flush=True)

    np.savez_compressed(
        OUT / 'arrays.npz',
        head_b=np.concatenate(all_pb).astype(np.float32),
        head_r=np.concatenate(all_pr).astype(np.float32),
        rid=np.concatenate(all_rid),
        rec_label=np.concatenate(all_lab))
    result = {
        'purpose': 'report-only robustness: V1 AND-gate on PTB val (gold), '
                   'not used for selection',
        'gate': f'head_r>={TAU_R} AND window_mean(head_b)>={TAU_B}',
        'n_patients': {'abnormal': int(n_abn), 'normal': int(n_norm)},
        'event_recall': {'baseline': tp0 / max(n_abn, 1),
                         'gate': tp1 / max(n_abn, 1)},
        'normal_fp_records': {'baseline': int(fp0), 'gate': int(fp1)},
        'rows': rows,
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'},
                     indent=2), flush=True)


def build_record_probe(model):
    import tensorflow as tf
    return tf.keras.Model([model.inputs[1], model.inputs[2]], model.outputs[1])


if __name__ == '__main__':
    main()
