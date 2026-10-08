#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direction 1: end-to-end gated event replay with exported TFLite models.

Replays the deployment event detectors through the TFLite probes (not Keras):
  v1-min  : 1-of-5, head_r>=0.5,  mean10(head_b)>=0.15, cooldown 20
  v2-spec : 2-of-5, head_r>=0.7,  mean10(head_b)>=0.1,  cooldown 20

Scope: fp16 models on all real recordings + all MIT/INCART val records
(decision parity vs Keras reference from m4_head_r_diagnosis arrays);
int8 models on real recordings + a 3-record MIT subset (spot check).
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

from m4_export_tflite import (  # noqa: E402
    make_tflite_pred, predict_beat, predict_record, causal_windows)
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa: E402

EXP = Path('/home/devcontainers/meeti/runs/m4_tflite_export')
OUT = Path('/home/devcontainers/meeti/runs/m4_tflite_e2e')
T = 10
REAL_META = {'ecg_real_052': 183.0, 'rec_latest': 64.0}
REAL = {
    'ecg_real_052': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_exp7c.npy',
    'rec_latest': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_rec_latest.npy',
}
CONFIGS = {
    'v1_min': {'k': 1, 'tau_r': 0.5, 'tau_b': 0.15},
    'v2_spec': {'k': 2, 'tau_r': 0.7, 'tau_b': 0.1},
}
COOLDOWN = 20


def window_mean(p):
    n = len(p)
    out = np.zeros(n, dtype=np.float32)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    for i in range(n):
        lo = max(0, i - T + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def alarm_triggers(hr, hbw, cfg):
    pos = (hr >= cfg['tau_r']) & (hbw >= cfg['tau_b'])
    cand = np.zeros(len(pos), bool)
    acc = 0
    for i in range(len(pos)):
        acc += int(pos[i])
        if i >= 5:
            acc -= int(pos[i - 5])
        cand[i] = acc >= cfg['k']
    trig, last = [], -10 ** 9
    for i in range(len(cand)):
        lo = max(0, i - 4)
        if cand[lo:i + 1].any() and i - last >= COOLDOWN:
            trig.append(i)
            last = i
    return trig


def replay(it_b, it_r, beats):
    pb = predict_beat(it_b, beats)
    x, m = causal_windows(beats)
    pr = predict_record(it_r, x, m)
    return pb, pr, window_mean(pb)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dia = np.load('/home/devcontainers/meeti/runs/m4_head_r_diagnosis/'
                  'arrays.npz')
    mit_ref = {'hr': dia['mitval__head_r'], 'rid': dia['mitval__rid'],
               'lab': dia['mitval__rec_label'],
               'hb': dia['mitval__head_b']}

    results = {'configs': CONFIGS, 'real': {}, 'mitval': {}, 'parity': {}}

    for variant in ['fp16', 'int8']:
        it_b = make_tflite_pred(EXP / f'dual_beat_{variant}.tflite')
        it_r = make_tflite_pred(EXP / f'dual_record_{variant}.tflite')

        # ----- real recordings -----
        for name, path in REAL.items():
            beats = np.load(path).astype(np.float32)
            pb, pr, hbw = replay(it_b, it_r, beats)
            row = {'head_b_max': float(pb.max()),
                   'head_r_mean': float(pr.mean())}
            for cname, cfg in CONFIGS.items():
                trig = alarm_triggers(pr, hbw, cfg)
                row[cname] = {
                    'alarms': len(trig),
                    'fp_per_hour': round(len(trig) / (REAL_META[name] / 3600.0), 2)}
            results['real'][f'{variant}__{name}'] = row
            print(variant, name, json.dumps(row), flush=True)

        # ----- MIT val -----
        split = compute_mit_incart_split()
        groups = {}
        for ds in ['mit_bih', 'incart']:
            beats, labels, rids = load_arrays(ds)
            vm = np.asarray(split['val_masks'][ds], dtype=bool)
            beats = np.asarray(beats[vm], dtype=np.float32)
            labels = np.asarray(labels[vm])
            rids = np.asarray(rids[vm])
            order = np.argsort(rids, kind='stable')
            beats, labels, rids = beats[order], labels[order], rids[order]
            uniq, first, counts = np.unique(rids, return_index=True,
                                            return_counts=True)
            for ui, rid in enumerate(uniq):
                s, e = first[ui], first[ui] + counts[ui]
                groups[f'{ds}_{int(rid)}'] = {
                    'beats': beats[s:e], 'label': int(labels[s:e].max())}

        subset = (list(groups) if variant == 'fp16'
                  else list(groups)[:3])
        tp = {c: 0 for c in CONFIGS}
        n_abn = 0
        agree = {c: 0 for c in CONFIGS}
        n_cmp = 0
        # keras reference aligned per record (boundary-safe window means)
        uniq_rids, first_r, counts_r = np.unique(
            mit_ref['rid'], return_index=True, return_counts=True)
        ref_by_rid = {str(u): (mit_ref['hr'][f:f + c],
                               window_mean(mit_ref['hb'][f:f + c]))
                      for u, f, c in zip(uniq_rids, first_r, counts_r)}
        for rname in subset:
            rec = groups[rname]
            pb, pr, hbw = replay(it_b, it_r, rec['beats'])
            if rec['label'] == 1:
                n_abn += 1
            hr_k, hbw_k = ref_by_rid[rname]
            score_dp = float(np.abs(pr - hr_k).max())
            for cname, cfg in CONFIGS.items():
                trig = alarm_triggers(pr, hbw, cfg)
                trig_k = alarm_triggers(hr_k, hbw_k, cfg)
                if rec['label'] == 1 and len(trig) > 0:
                    tp[cname] += 1
                agree[cname] += int(len(trig) == len(trig_k))
            n_cmp += 1
            results['parity'][f'{variant}__{rname}'] = {
                'max_dp_head_r': score_dp, 'n_beats': int(len(rec['beats']))}
        results['mitval'][variant] = {
            'n_records': n_cmp, 'n_abnormal': n_abn,
            'recall': {c: tp[c] / n_abn if n_abn else None for c in CONFIGS},
            'decision_agree_frac': {c: agree[c] / n_cmp for c in CONFIGS},
            'scope': 'all' if variant == 'fp16' else 'first3',
        }
        print(variant, json.dumps(results['mitval'][variant]), flush=True)

    with open(OUT / 'result.json', 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
