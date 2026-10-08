#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R12-T4: train-side temperature scaling of the candidate record head.

Zero retraining.  For each r9_form-ptbbal candidate checkpoint (seeds
42-47): fit a single scalar temperature T on PTB TRAIN record scores
(top2mean, leak-free) by NLL minimization, then report the theta=0.5 and
theta_tr operating points on PTB val/test before vs after calibration.

  p' = sigmoid(logit(p) / T)          (binary softmax temperature)

Notes
  - AUC / AP / spec@R90 are invariant under the monotone map; spec@R90 is
    reported anyway as an invariance check of the pipeline.
  - theta_tr is re-derived on the temperature-scaled TRAIN scores (a
    successful calibration pulls theta_tr' toward 0.5 and should collapse
    the R11-documented cross-seed bimodality of theta_tr).
  - Device-domain frac>0.5 is disclosed (monotone map still moves the
    frac at the fixed 0.5 cut).

Outputs runs/r12_temp_scaling/{scores_seed{S}.npz, result.json}.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from scipy.optimize import minimize_scalar
from sklearn.metrics import average_precision_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r9_data import load_all  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'ecg_bench', '/home/devcontainers/meeti/scripts/ecg_bench.py')
eb = importlib.util.module_from_spec(_spec)
sys.argv = sys.argv[:1]
_spec.loader.exec_module(eb)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r12_temp_scaling'
SEEDS = [42, 43, 44, 45, 46, 47]
TMPL = 'runs/r9_form-ptbbal_seed{s}/ckpt_best__record5s.keras'


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def temp_apply(s, T):
    return 1.0 / (1.0 + np.exp(-logit(np.asarray(s, float)) / T))


def fit_T(y, s):
    y = np.asarray(y, float)

    def nll(logT):
        p = temp_apply(s, float(np.exp(logT)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    r = minimize_scalar(nll, bounds=(np.log(0.05), np.log(20.0)),
                        method='bounded', options={'xatol': 1e-4})
    return float(np.exp(r.x))


def workpoint(y, s, th_tr):
    """theta=0.5 and theta_tr operating metrics + AP + spec@R90."""
    om = eb.op_metrics(y, s, [0.5, (th_tr, 'theta_tr')])

    def spec(dom_th):
        tp, fp, fn, tn = om[dom_th]['tp_fp_fn_tn']
        return round(tn / max(tn + fp, 1), 4)

    out = {
        'ap': round(float(average_precision_score(y, s)), 4),
        'spec_at_R90': eb.spec_at_recall(y, s, 0.90),
        'at_0.5': {'recall': om['0.500']['recall'], 'spec': spec('0.500'),
                   'bacc': om['0.500']['bacc']},
        'at_theta_tr': {'theta': round(float(th_tr), 4),
                        'recall': om['theta_tr']['recall'],
                        'spec': spec('theta_tr'),
                        'bacc': om['theta_tr']['bacc']},
    }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    t0 = time.time()
    data = load_all(norm=True, form=True)
    split = data['ptb_splits']

    results = []
    for s in SEEDS:
        ck = ROOT / TMPL.format(s=s)
        if not ck.exists():
            print(f'skip seed{s} (missing)', flush=True)
            continue
        probe = tf.keras.models.load_model(str(ck))
        sc = {}
        for name in ['train', 'val', 'test']:
            d = eb.ptb_5s(split[name])
            p_win = eb.predict_windows(probe, d['x'])
            sc[name] = {'y': np.asarray(d['rec_y'], np.int32),
                        's': eb.top2mean_by_record(p_win, d['win_rec'],
                                                   d['n_records']
                                                   ).astype(np.float32)}
        np.savez_compressed(OUT / f'scores_seed{s}.npz',
                            y_tr=sc['train']['y'], s_tr=sc['train']['s'],
                            y_va=sc['val']['y'], s_va=sc['val']['s'],
                            y_te=sc['test']['y'], s_te=sc['test']['s'])

        th_tr_raw = eb.youden_threshold(sc['train']['y'], sc['train']['s'])
        T = fit_T(sc['train']['y'], sc['train']['s'])
        st = {k: temp_apply(sc[k]['s'], T) for k in sc}
        th_tr_temp = eb.youden_threshold(sc['train']['y'], st['train'])

        rec = {'seed': s, 'T': round(T, 4),
               'theta_tr_raw': round(th_tr_raw, 4),
               'theta_tr_temp': round(th_tr_temp, 4),
               'ptb_val': {'raw': workpoint(sc['val']['y'], sc['val']['s'],
                                            th_tr_raw),
                           'temp': workpoint(sc['val']['y'], st['val'],
                                             th_tr_temp)},
               'ptb_test': {'raw': workpoint(sc['test']['y'], sc['test']['s'],
                                             th_tr_raw),
                            'temp': workpoint(sc['test']['y'],
                                              st['test'], th_tr_temp)}}
        results.append(rec)
        print(f'[s{s}] T={T:.3f} theta_tr {th_tr_raw:.3f}->{th_tr_temp:.3f} '
              f'test spec@.5 {rec["ptb_test"]["raw"]["at_0.5"]["spec"]}'
              f'->{rec["ptb_test"]["temp"]["at_0.5"]["spec"]} '
              f'bAcc@.5 {rec["ptb_test"]["raw"]["at_0.5"]["bacc"]}'
              f'->{rec["ptb_test"]["temp"]["at_0.5"]["bacc"]} '
              f'[{time.time() - t0:.0f}s]', flush=True)
        del probe, sc, st
        tf.keras.backend.clear_session()

    def ms(vals):
        return {'mean': round(float(np.mean(vals)), 4),
                'sd': round(float(np.std(vals, ddof=1)), 4)
                if len(vals) > 1 else 0.0}

    def med(vals):
        return {'median': round(float(np.median(vals)), 4),
                'min': round(float(np.min(vals)), 4),
                'max': round(float(np.max(vals)), 4)}

    summary = {
        'n_seeds': len(results),
        'T': ms([r['T'] for r in results]),
        'theta_tr_raw': med([r['theta_tr_raw'] for r in results]),
        'theta_tr_temp': med([r['theta_tr_temp'] for r in results]),
        'ptb_test': {
            'spec_at_05_raw': ms([r['ptb_test']['raw']['at_0.5']['spec']
                                  for r in results]),
            'spec_at_05_temp': ms([r['ptb_test']['temp']['at_0.5']['spec']
                                   for r in results]),
            'bacc_at_05_raw': ms([r['ptb_test']['raw']['at_0.5']['bacc']
                                  for r in results]),
            'bacc_at_05_temp': ms([r['ptb_test']['temp']['at_0.5']['bacc']
                                   for r in results]),
            'recall_at_05_raw': ms([r['ptb_test']['raw']['at_0.5']['recall']
                                    for r in results]),
            'recall_at_05_temp': ms([r['ptb_test']['temp']['at_0.5']['recall']
                                     for r in results]),
            'spec_at_thtr_raw': ms([r['ptb_test']['raw']['at_theta_tr']
                                    ['spec'] for r in results]),
            'spec_at_thtr_temp': ms([r['ptb_test']['temp']['at_theta_tr']
                                     ['spec'] for r in results]),
        },
        'ptb_val': {
            'spec_at_05_raw': ms([r['ptb_val']['raw']['at_0.5']['spec']
                                  for r in results]),
            'spec_at_05_temp': ms([r['ptb_val']['temp']['at_0.5']['spec']
                                   for r in results]),
            'bacc_at_05_raw': ms([r['ptb_val']['raw']['at_0.5']['bacc']
                                  for r in results]),
            'bacc_at_05_temp': ms([r['ptb_val']['temp']['at_0.5']['bacc']
                                   for r in results]),
        },
        'invariance_check': {
            'note': 'spec_at_R90 must be identical raw vs temp (monotone '
                    'map) - max abs diff reported',
            'max_abs_diff_ptb_test': round(max(
                abs(r['ptb_test']['raw']['spec_at_R90']['spec']
                    - r['ptb_test']['temp']['spec_at_R90']['spec'])
                for r in results), 6),
        },
    }
    out = {'arm': 'r12_t4_temperature_scaling',
           'spec': 'train-side T fit on PTB train record top2mean NLL; '
                   'per-seed single scalar; candidate r9_form-ptbbal '
                   'seeds 42-47; zero retraining',
           'per_seed': results, 'summary': summary}
    with open(OUT / 'result.json', 'w') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2), flush=True)
    print('saved ->', OUT / 'result.json', flush=True)


if __name__ == '__main__':
    main()
