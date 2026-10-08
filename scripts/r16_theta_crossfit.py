#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A5: threshold cross-fitting + device-domain calibration disclosure.

Part 1 (pre-registered primary): use the B2 20% patient-level holdout
probabilities (3 seeds, 3927 records, ~269 AF) and cross-fit the AF-head
threshold in 5 patient-level folds.  Per fold: theta_tr (Youden) and
theta_R90tr (max threshold with recall >= 0.90) are computed on the other
4 folds and evaluated on the held-out fold.  Criterion (R15-B2): theta_tr
range <= 0.30.

Part 2 (secondary, same calibration question for the RR gate): record-level
5-fold cross-fit of the A1/A2 features on the 12 afdb fit records, evaluated
on the 11 afdb held-out records, ltafdb and ecg_real_052.

Device-domain calibration set: no user-provided device normal recordings
beyond ecg_real_052 (the negative control) and rec_latest exist, so no device
calibration is possible; disclosed.

Outputs: runs/r16_theta_crossfit/result.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import train_theta_at_recall, youden_threshold  # noqa: E402
from r16_rr_features import SHAPE_FEATURES  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DATA = ROOT / 'runs/r16_rr_shape'
OUT = ROOT / 'runs/r16_theta_crossfit'
PTB_TRAIN = ROOT / 'runs/r14_af_prep/ptbxl_af_train.npz'
M4_CONFIRM, M4_COOLDOWN = 5, 20


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def op(y, s, th):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fl = np.isfinite(s) & (s >= th)
    return {
        'recall': round(float((fl & (y == 1)).sum())
                        / max(int((y == 1).sum()), 1), 4),
        'spec': round(float(((~fl) & (y == 0)).sum())
                      / max(int((y == 0).sum()), 1), 4),
        'n_flag': int(fl.sum()),
    }


def ci95(vals):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(42)
    bs = [float(np.mean(rng.choice(vals, len(vals), replace=True)))
          for _ in range(2000)]
    return [round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def summarize(thetas):
    t = np.asarray(thetas, float)
    return {
        'n': int(len(t)), 'mean': round(float(t.mean()), 4),
        'sd': round(float(t.std(ddof=1)), 4) if len(t) > 1 else None,
        'min': round(float(t.min()), 4), 'max': round(float(t.max()), 4),
        'range': round(float(t.max() - t.min()), 4),
        'ci95_mean': ci95(t),
        'values': [round(float(x), 4) for x in t],
    }


def af_head_crossfit():
    tr = np.load(PTB_TRAIN)
    m = {int(e): int(p) for e, p in zip(tr['ecg_ids'], tr['patient_id'])}
    out = {'arm': 'r16_theta_crossfit_af_head', 'seeds': {}, 'pooled': {}}
    all_youden, all_r90 = [], []
    for seed in (42, 43, 44):
        d = np.load(ROOT / f'runs/r14_af_head/af_seed{seed}_calib20/probs.npz')
        p, y, ev = d['p_val'].astype(float), d['y_val'].astype(int), d['ecg_val']
        pids = np.array([m[int(e)] for e in ev])
        upat = np.unique(pids)
        ypat = np.array([int(y[pids == q].max()) for q in upat])
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        folds = []
        for k, (cal_pat_idx, held_idx) in enumerate(skf.split(upat, ypat)):
            cal_pat = set(upat[cal_pat_idx].tolist())
            held_pat = set(upat[held_idx].tolist())
            cal = np.array([q in cal_pat for q in pids])
            held = np.array([q in held_pat for q in pids])
            th_y = float(youden_threshold(y[cal], p[cal]))
            th_r = float(train_theta_at_recall(y[cal], p[cal], 0.90))
            folds.append({
                'fold': k, 'n_cal': int(cal.sum()), 'n_held': int(held.sum()),
                'pos_cal': int(y[cal].sum()), 'pos_held': int(y[held].sum()),
                'theta_youden': round(th_y, 4),
                'theta_R90tr': round(th_r, 4),
                'held_op_at_youden': op(y[held], p[held], th_y),
                'held_op_at_R90tr': op(y[held], p[held], th_r),
            })
            all_youden.append(th_y)
            all_r90.append(th_r)
        out['seeds'][str(seed)] = {
            'theta_youden': summarize([f['theta_youden'] for f in folds]),
            'theta_R90tr': summarize([f['theta_R90tr'] for f in folds]),
            'folds': folds,
        }
    out['pooled']['theta_youden'] = summarize(all_youden)
    out['pooled']['theta_R90tr'] = summarize(all_r90)
    out['criterion_theta_tr_range_le_0.30'] = bool(
        out['pooled']['theta_youden']['range'] <= 0.30)
    out['criterion_theta_R90tr_range_le_0.30'] = bool(
        out['pooled']['theta_R90tr']['range'] <= 0.30)
    out['device_calibration'] = {
        'status': 'not possible',
        'reason': ('no user-provided device normal recordings beyond '
                   'ecg_real_052 (negative control) and rec_latest; device AF '
                   'positives absent'),
    }
    return out


def rr_gate_crossfit():
    af = dict(np.load(DATA / 'afdb_features.npz'))
    lt = dict(np.load(DATA / 'ltafdb_features.npz'))
    dv = np.load(DATA / 'device_features.npz')
    yaf = af['afdb_y'].astype(int)
    sp = af['afdb_split'].astype(int)
    ylt = lt['ltafdb_y'].astype(int)
    fit_recs = sorted(set(af['afdb_rec'][sp == 0].tolist()))
    eval_recs = sorted(set(af['afdb_rec'][sp == 1].tolist()))
    rng = np.random.default_rng(42)
    order = rng.permutation(len(fit_recs))
    fold_of = {fit_recs[order[i]]: i % 5 for i in range(len(fit_recs))}
    feats = ['rr_entropy', 'drr_entropy', 'cv', 'mean_abs_drr_norm',
             'rmssd', 'sdnn', 'sd1', 'katz_fd', 'range_norm']
    out = {'arm': 'r16_theta_crossfit_rr_gate',
           'fit_records': fit_recs, 'eval_records': eval_recs,
           'fold_assignment': {r: int(fold_of[r]) for r in fit_recs},
           'features': {}}
    for k in feats:
        sfit = af[f'afdb30_{k}'][sp == 0]
        yfit = yaf[sp == 0]
        recfit = af['afdb_rec'][sp == 0]
        se = af[f'afdb30_{k}'][sp == 1]
        ye = yaf[sp == 1]
        sl = lt[f'ltafdb30_{k}']
        sd = dv[f'ecg_real_052_deploy_xqrs250_paired30_{k}']
        starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
        # orientation on the full fit pool (AF-high)
        from sklearn.metrics import roc_auc_score
        mm = np.isfinite(sfit)
        auc = roc_auc_score(yfit[mm], sfit[mm])
        sign = 1.0 if auc >= 0.5 else -1.0
        sfit, se, sl, sd = [sign * np.asarray(x, float)
                            for x in (sfit, se, sl, sd)]
        folds = []
        thetas = []
        for f in range(5):
            cal = np.array([fold_of[r] != f for r in recfit])
            th = train_theta_at_recall(yfit[cal & np.isfinite(sfit)],
                                       sfit[cal & np.isfinite(sfit)], 0.90)
            if th is None:
                continue
            thetas.append(float(th))
            fl052 = np.isfinite(sd) & (sd >= th)
            folds.append({
                'fold': f, 'n_cal': int(cal.sum()),
                'theta_R90tr': round(float(th), 4),
                'fit_cal_op': op(yfit[cal], sfit[cal], th),
                'afdb_eval_op': op(ye, se, th),
                'ltafdb_op': op(ylt, sl, th),
                'device_052': {'n_flag': int(fl052.sum()),
                               'm4_alarms': int(len(m4_alarms(fl052)))},
            })
        summ = summarize(thetas) if thetas else None
        # pooled cross-fit operating point: mean theta of the 5 folds
        pooled = None
        if thetas:
            thm = float(np.mean(thetas))
            fl052 = np.isfinite(sd) & (sd >= thm)
            pooled = {
                'theta_mean': round(thm, 4),
                'afdb_eval_op': op(ye, se, thm),
                'ltafdb_op': op(ylt, sl, thm),
                'device_052': {'n_flag': int(fl052.sum()),
                               'm4_alarms': int(len(m4_alarms(fl052)))},
                'pass_three_lines': bool(
                    len(m4_alarms(fl052)) == 0
                    and op(ye, se, thm)['recall'] >= 0.90
                    and op(ylt, sl, thm)['recall'] >= 0.90),
            }
        out['features'][k] = {
            'sign': sign, 'fit_auc': round(float(auc), 4),
            'theta_summary': summ, 'folds': folds, 'pooled_operating_point': pooled,
            'criterion_theta_range_le_0.30': bool(
                summ and summ['range'] <= 0.30),
        }
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {
        'arm': 'r16_theta_crossfit (A5)',
        'prereg': 'runs/R16_PREREG.md',
        'criterion': 'theta_tr range <= 0.30 (R15-B2)',
        'af_head': af_head_crossfit(),
        'rr_gate': rr_gate_crossfit(),
        'disclosures': [
            'AF-head cross-fit uses the B2 20% holdout probabilities '
            '(3927 records / 269 AF); folds are patient-level and can carry '
            'only ~54 AF records each -> wide sampling error is expected',
            'RR-gate cross-fit uses 12 afdb fit records / 5 folds -> ~2-3 '
            'records per fold, extreme sampling error; record-level split '
            'cannot fix the fit/eval record-distribution shift',
            'no device-domain calibration set exists (only 2 device normal '
            'recordings, no device AF positives)',
            'spec@R90 is the primary caliber; point thresholds are disclosure',
        ],
    }
    (OUT / 'result.json').write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps({
        'af_head_pooled_youden': res['af_head']['pooled']['theta_youden'],
        'af_head_pooled_R90tr': res['af_head']['pooled']['theta_R90tr'],
        'af_head_criterion_youden': res['af_head'][
            'criterion_theta_tr_range_le_0.30'],
        'af_head_criterion_R90tr': res['af_head'][
            'criterion_theta_R90tr_range_le_0.30'],
        'rr_gate': {k: {'range': v['theta_summary']['range'] if v['theta_summary'] else None,
                        'criterion': v['criterion_theta_range_le_0.30'],
                        'pooled_pass': (v['pooled_operating_point'] or {}).get(
                            'pass_three_lines')}
                    for k, v in res['rr_gate']['features'].items()},
    }, indent=2, ensure_ascii=False))
    print('written', OUT / 'result.json')


if __name__ == '__main__':
    main()
