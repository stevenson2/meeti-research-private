#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R11 PC-side parity + gate validation of the EXACT deployed artifacts.

Validates (TFLite interpreter, CPU) that the two firmware-embedded models
reproduce the benchmark numbers, and simulates the firmware AND-gate
(bit-faithful to ecg_ai_dual.cpp: hbw mean of last-5 p_b, 1-of-5 + cooldown 20):
  record5s_fp16.tflite (deploy seed43) : PTB val/test + MEETI val top2mean AUC
  dual_beat_int8.tflite (locked beat)  : beat_val + beat_mi val/test AUC
  gate v1-min on device beats          : FP must be 0
  gate v1-min on MI val abnormal       : recall
Outputs runs/deploy_candidate_r11/pc_parity_result.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

import r8_data_5s  # noqa: F401,E402
from r8_data_5s import ptb_5s, device_5s, REAL_FILES, REAL_DIR  # noqa: E402
from data.split_guard import load_arrays, compute_ptb_split, \
    compute_mit_incart_split  # noqa: E402
from r9_data import load_all  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DEP = ROOT / 'runs/deploy_candidate_r11'
W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}


def make_interpreter(path):
    it = tf.lite.Interpreter(model_path=str(path),
                             num_threads=4)
    it.allocate_tensors()
    return it


def tflite_prob(it, x, batch=256):
    inp = it.get_input_details()[0]
    out = it.get_output_details()[0]
    scalars = []
    is_int8 = inp['dtype'] == np.int8
    ps = []
    fixed1 = (inp['shape'][0] == 1)
    shape = inp['shape'].copy()
    for s in range(0, len(x), 1 if fixed1 else batch):
        xb = x[s:s + (1 if fixed1 else batch)]
        if fixed1:
            for j in range(len(xb)):
                v = xb[j]
                if is_int8:
                    scale, zp = inp['quantization']
                    q = np.clip(np.round(v / scale) + zp,
                                -128, 127).astype(np.int8)
                    it.set_tensor(inp['index'], q.reshape(shape))
                else:
                    it.set_tensor(inp['index'],
                                  v.astype(np.float32).reshape(shape))
                it.invoke()
                o = it.get_tensor(out['index'])
                if o.dtype == np.int8:
                    os_, oz = out['quantization']
                    o = (o.astype(np.float32) - oz) * os_
                scalars.append(float(np.ravel(o)[1]))
            continue
        if is_int8:
            scale, zp = inp['quantization']
            q = np.clip(np.round(xb / scale) + zp, -128, 127).astype(np.int8)
            it.set_tensor(inp['index'], q.reshape(shape))
        else:
            it.set_tensor(inp['index'], xb.astype(np.float32).reshape(shape))
        it.invoke()
        o = it.get_tensor(out['index'])
        if o.dtype == np.int8:
            os_, oz = out['quantization']
            o = (o.astype(np.float32) - oz) * os_
        ps.append(o[:, 1])
    if scalars:
        return np.array(scalars, np.float64)
    return np.concatenate(ps)


def top2mean_by_record(p_win, win_rec, n_rec):
    order = np.argsort(win_rec, kind='stable')
    p, r = p_win[order], win_rec[order]
    uniq, first, counts = np.unique(r, return_index=True,
                                    return_counts=True)
    out = np.zeros(n_rec)
    for i, (s, c) in enumerate(zip(first, counts)):
        v = np.sort(p[s:s + c])[::-1]
        out[i] = v[:min(2, len(v))].mean()
    return out


def main():
    res = {'note': 'PC TFLite parity of firmware-embedded artifacts '
                   '(interpreter semantics identical to TFLM Invoke)'}
    rit = make_interpreter(DEP / 'record5s_fp16.tflite')
    bit = make_interpreter(DEP / 'companion_beat_fp16.tflite')
    bint = make_interpreter(ROOT / 'runs/m4_tflite_export/'
                            'dual_beat_int8.tflite')

    data = load_all(norm=True, form=True)
    split = compute_ptb_split()

    # ---- record head (deployed candidate, fp16) ----
    me, sp = data['meeti'], data['ptb_splits']
    for name, d in [('ptb_val', ptb_5s(sp['val'])),
                    ('ptb_test', ptb_5s(sp['test']))]:
        p = tflite_prob(rit, d['x'])
        auc = float(roc_auc_score(
            d['rec_y'], top2mean_by_record(p, d['win_rec'], d['n_records'])))
        res[f'record_{name}'] = round(auc, 4)
        print('record', name, round(auc, 4), flush=True)
    va_win = me['rec_va'][me['win_rec']]
    val_records = np.where(me['rec_va'])[0]
    remap = np.zeros(me['n_records'], np.int64)
    remap[val_records] = np.arange(len(val_records))
    p = tflite_prob(rit, me['x'][va_win])
    res['record_meeti_val'] = round(float(roc_auc_score(
        me['rec_y'][val_records],
        top2mean_by_record(p, remap[me['win_rec'][va_win]],
                           len(val_records)))), 4)
    print('record meeti_val', res['record_meeti_val'], flush=True)

    # ---- beat head (deployed int8, locked) ----
    _, _, xv, yv, _ = __import__('train_dual_head_small').load_hard_beats()
    pv = tflite_prob(bint, xv)
    res['beat_val_int8'] = round(float(roc_auc_score(yv, pv)), 4)
    print('beat_val(int8)', res['beat_val_int8'], flush=True)
    from ecg_bench import eval_beat_mit_incart  # heavy; use direct here
    # MIT+INCART via split_guard (direct, no ecg_bench import side effects)
    from data.split_guard import load_arrays as la
    mspl = compute_mit_incart_split()
    for tag, suffix in [('val_masks', 'val'), ('test_masks', 'test')]:
        xs, ys = [], []
        for t in ['mit_bih', 'incart']:
            b, l, _ = la(t)
            m = np.asarray(mspl[suffix][t] if suffix in mspl
                           else mspl['val_masks' if suffix == 'val'
                                     else 'test_masks'][t], bool)
            xs.append(np.asarray(b, np.float32)[m])
            ys.append(np.asarray(l)[m].astype(np.int32))
        x = np.concatenate(xs)
        y = np.concatenate(ys)
        pb = tflite_prob(bint, x[..., None] if x.ndim == 2 else x)
        res[f'beat_mi_{suffix}_int8'] = round(float(roc_auc_score(y, pb)), 4)
        print(f'beat_mi_{suffix}(int8)', res[f'beat_mi_{suffix}_int8'],
              flush=True)

    # ---- gate simulation (bit-faithful to firmware v1-min) ----
    def gate_run(beats_seqs, tau_r=0.5, tau_b=0.15, n=5, k=1, cooldown=20):
        alarms = 0
        pos_hist, cd = [], 0
        for seq in beats_seqs:
            pos_hist, cd = [], 0
            pb_hist = []
            for w in seq:
                pb = float(tflite_prob(bint, w[None], batch=1)[0])
                pb_hist.append(pb)
                hbw = float(np.mean(pb_hist[-5:]))
                pr = 0.0
                have = False
                # rolling 5-beat windows per firmware (stride 1 beat)
                return_early = True
                pb_hist_ring = pb_hist[-5:]
                if len(pb_hist_ring) >= 5:
                    xr = np.concatenate(pb_hist_ring, axis=0)[None]
                    pr = float(tflite_prob(rit, np.concatenate(
                        [w for w in seq[max(0, len(pb_hist) - 5):len(pb_hist)]],
                        axis=0)[None], batch=1)[0])
                    have = True
                pos = (have and pr >= tau_r and hbw >= tau_b)
                pos_hist.append(pos)
                cand = any(pos_hist[-n:])
                if cd > 0:
                    cd -= 1
                elif cand:
                    alarms += 1
                    cd = cooldown
            # one alarm sequence per record
        return alarms

    # device FP (record-level: 1-of-5 with cooldown on the beat stream)
    dev_alarms = 0
    for name, fname in REAL_FILES.items():
        b = np.load(REAL_DIR / fname).astype(np.float32)
        pb = tflite_prob(bint, b)
        pb_hist, pos_hist, cd = [], [], 0
        a = 0
        for i, p in enumerate(pb):
            pb_hist.append(p)
            hbw = float(np.mean(pb_hist[-5:]))
            have = i >= 4
            pr = 0.0
            if have:
                xr = np.concatenate(b[i - 4:i + 1], axis=0)[None]
                pr = float(tflite_prob(rit, xr, batch=1)[0])
            pos = (have and pr >= 0.5 and hbw >= 0.15)
            pos_hist.append(pos)
            cand = any(pos_hist[-5:])
            if cd > 0:
                cd -= 1
            elif cand:
                a += 1
                cd = 20
        res[f'device_fp_{name}'] = a
        dev_alarms += a
        print('device FP', name, a, flush=True)

    # MI val abnormal recall (record events: 26 expected)
    mspl = compute_mit_incart_split()
    recalled, total = 0, 0
    for t in ['mit_bih', 'incart']:
        b, l, r = la(t)
        m = np.asarray(mspl['val_masks'][t], bool)
        b = np.asarray(b, np.float32)[m]
        l = np.asarray(l)[m]
        r = np.asarray(r)[m]
        order = np.argsort(r, kind='stable')
        b, l, r = b[order], l[order], r[order]
        uniq, first, counts = np.unique(r, return_index=True,
                                        return_counts=True)
        for i in range(len(uniq)):
            s, nb = int(first[i]), int(counts[i])
            if l[s:s + nb].max() != 1:
                continue
            total += 1
            seq = b[s:s + nb]
            pb = tflite_prob(bint, seq)
            pb_hist, pos_hist, cd = [], [], 0
            hit = 0
            for j, p in enumerate(pb):
                pb_hist.append(p)
                hbw = float(np.mean(pb_hist[-5:]))
                have = j >= 4
                pr = 0.0
                if have:
                    xr = np.concatenate(seq[j - 4:j + 1], axis=0)[None]
                    pr = float(tflite_prob(rit, xr, batch=1)[0])
                pos = (have and pr >= 0.5 and hbw >= 0.15)
                pos_hist.append(pos)
                cand = any(pos_hist[-5:])
                if cd > 0:
                    cd -= 1
                elif cand:
                    hit = 1
                    cd = 20
            recalled += hit
    res['mi_val_abn_records'] = total
    res['mi_val_recall_records'] = recalled
    print(f'MI val abnormal recall: {recalled}/{total}', flush=True)

    with open(DEP / 'pc_parity_result.json', 'w') as f:
        json.dump(res, f, indent=2)
    print('saved ->', DEP / 'pc_parity_result.json', flush=True)


if __name__ == '__main__':
    main()
