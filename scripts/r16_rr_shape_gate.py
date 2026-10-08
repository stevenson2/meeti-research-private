#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A2: pre-registered two-stage shape gate (decisive).

Structure (runs/R16_PREREG.md):
  Stage 1 irregularity : rr_entropy >= theta1
  Stage 2 shape reject : shape feature or LR >= theta2 on Stage-1 positives
Both thresholds are derived from the TRAIN side only.

Variants:
  A (literal prompt reading): theta1 = max train threshold with train recall
    >= 0.90; theta2 = Youden on the Stage-1-positive training subset.
  B (primary, pre-registered clarification): theta1 = max train threshold
    with train recall >= 0.97 (recall-preserving stage); theta2 = max
    threshold with COMPOSITE train recall >= 0.90 (composite spec@R90).
    Rationale: Stage 2 can only remove windows, so a theta1 anchored at
    recall 0.90 makes the composite >=0.90 acceptance structurally
    infeasible.

Acceptance (both variants, same lines):
  1. ecg_real_052 event-level FP=0 (M4 1-of-5, cooldown 20)
  2. afdb held-out 11 records AF recall >= 0.90
  3. ltafdb sampled AF recall >= 0.90
rec_latest is disclosed separately.

Stage-2 scorer selection: on the Stage-1-positive TRAIN subset, each shape
feature and two LR combos is oriented AF-high and scored by the training-side
composite spec@R90; the best training-side scorer is the pre-registered
single winner.  An ORACLE Stage-2 threshold (post-hoc, uses afdb-eval labels)
is reported as a necessary-condition diagnostic.

Outputs: runs/r16_rr_shape_gate/result.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r16_rr_features import SHAPE_FEATURES, ALL_FEATURES  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
DATA = ROOT / 'runs/r16_rr_shape'
OUT = ROOT / 'runs/r16_rr_shape_gate'
M4_CONFIRM = 5
M4_COOLDOWN = 20
THRESHOLD_TARGETS = {'A': 0.90, 'B': 0.97}
COMPOSITE_TARGET = 0.90
STAGE1 = 'rr_entropy'
STAGE2_CANDIDATES = SHAPE_FEATURES + ['lr_shape', 'lr_shape_plus_entropy']


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def recall_all(y, s, th):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fl = np.isfinite(s) & (s >= th)
    return float((fl & (y == 1)).sum()) / max(int((y == 1).sum()), 1)


def spec_all(y, s, th):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    fl = np.isfinite(s) & (s >= th)
    return float(((~fl) & (y == 0)).sum()) / max(int((y == 0).sum()), 1)


def th_valid(y, s, target):
    """Largest threshold with train recall >= target on valid (finite) rows."""
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    ys, ss = y[m], s[m]
    if len(ss) == 0:
        return None
    best = None
    for th in np.unique(ss):
        if float(((ss >= th) & (ys == 1)).sum()) / max(int((ys == 1).sum()),
                                                        1) >= target - 1e-12:
            best = float(th)
    return best


def th_youden(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    ys, ss = y[m], s[m]
    o = np.argsort(ss, kind='stable')
    ys, ss = ys[o], ss[o]
    P, N = int(ys.sum()), int(len(ys) - ys.sum())
    tp = P - np.cumsum(ys)
    fp = N - np.cumsum(1 - ys)
    j = tp / max(P, 1) - fp / max(N, 1)
    i = int(np.argmax(j[:-1])) + 1 if len(ss) > 1 else 0
    return float(ss[min(i, len(ss) - 1)])


def design(Xd, keys, med=None):
    Xm = np.column_stack([np.asarray(Xd[k], float) for k in keys])
    ind = np.column_stack([np.isnan(Xm[:, j]).astype(float)
                           for j in range(Xm.shape[1])])
    if med is None:
        med = np.nanmedian(Xm, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.hstack([np.where(np.isnan(Xm), med, Xm), ind]), med


def orient(y, s):
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(np.asarray(y)[m])) < 2:
        return 1.0, None
    a = float(roc_auc_score(np.asarray(y)[m], np.asarray(s)[m]))
    return (1.0 if a >= 0.5 else -1.0), a


def device_eval(flags, starts):
    alarms = m4_alarms(flags)
    return {'n_windows': int(len(flags)), 'n_flag': int(flags.sum()),
            'm4_alarms': int(len(alarms)),
            'm4_alarm_starts_s': [round(float(starts[i]), 1)
                                  for i in alarms[:10]]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    af = dict(np.load(DATA / 'afdb_features.npz'))
    lt = dict(np.load(DATA / 'ltafdb_features.npz'))
    ptb = dict(np.load(DATA / 'ptb_features.npz'))
    dv = np.load(DATA / 'device_features.npz')
    yaf = af['afdb_y'].astype(int)
    sp = af['afdb_split'].astype(int)
    ylt = lt['ltafdb_y'].astype(int)

    result = {
        'arm': 'r16_rr_shape_gate (A2)',
        'prereg': 'runs/R16_PREREG.md',
        'structure': 'Stage1 rr_entropy >= theta1 -> Stage2 shape/LR >= theta2',
        'threshold_targets': THRESHOLD_TARGETS,
        'composite_target': COMPOSITE_TARGET,
        'stage2_candidates': STAGE2_CANDIDATES,
        'window_lengths': {},
        'posthoc_single_stage': {},
        'disclosures': [
            'only 2 device normal recordings, no device AF positives',
            '30 s thresholds fit on afdb fit records only (PTB is 10 s)',
            'NaN/invalid windows are counted as misses in recall (R15 '
            'convention); detector-failure rates disclosed',
            'Stage-2 scorer selection uses training-side composite spec@R90 '
            'only; oracle thresholds are post-hoc diagnostics',
            'afdb/ltafdb domains never used for threshold selection',
        ],
    }

    for wl in ('10s', '30s'):
        if wl == '10s':
            def fit_arrays():
                return (np.concatenate([ptb['ptb_' + STAGE1],
                                        af[f'afdb10_{STAGE1}'][sp == 0]]),
                        np.concatenate([ptb['ptb_y'].astype(int),
                                        yaf[sp == 0]]),
                        {k: np.concatenate([ptb['ptb_' + k],
                                            af[f'afdb10_{k}'][sp == 0]])
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def eval_arrays():
                return (af[f'afdb10_{STAGE1}'][sp == 1], yaf[sp == 1],
                        {k: af[f'afdb10_{k}'][sp == 1]
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def lt_arrays():
                return (lt[f'ltafdb10_{STAGE1}'], ylt,
                        {k: lt[f'ltafdb10_{k}']
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def dev_arrays(name):
                return (dv[f'{name}_deploy_xqrs250_paired10_{STAGE1}'],
                        {k: dv[f'{name}_deploy_xqrs250_paired10_{k}']
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            dev_starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
            dev_starts_r = dv['rec_latest_deploy_xqrs250_paired_start']
        else:
            def fit_arrays():
                return (af[f'afdb30_{STAGE1}'][sp == 0], yaf[sp == 0],
                        {k: af[f'afdb30_{k}'][sp == 0]
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def eval_arrays():
                return (af[f'afdb30_{STAGE1}'][sp == 1], yaf[sp == 1],
                        {k: af[f'afdb30_{k}'][sp == 1]
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def lt_arrays():
                return (lt[f'ltafdb30_{STAGE1}'], ylt,
                        {k: lt[f'ltafdb30_{k}']
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            def dev_arrays(name):
                return (dv[f'{name}_deploy_xqrs250_paired30_{STAGE1}'],
                        {k: dv[f'{name}_deploy_xqrs250_paired30_{k}']
                         for k in STAGE2_CANDIDATES + ['rr_entropy', 'drr_entropy'] if k in ALL_FEATURES})
            dev_starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
            dev_starts_r = dv['rec_latest_deploy_xqrs250_paired_start']

        s1_fit, y_fit, X2_fit = fit_arrays()
        s1_eval, y_eval, X2_eval = eval_arrays()
        s1_lt, y_lt, X2_lt = lt_arrays()
        s1_052, X2_052 = dev_arrays('ecg_real_052')
        s1_rec, X2_rec = dev_arrays('rec_latest')

        wl_res = {'fit_n': int(len(y_fit)), 'fit_pos': int(y_fit.sum()),
                  'variants': {}}
        for vname, target in THRESHOLD_TARGETS.items():
            th1 = th_valid(y_fit, s1_fit, target)
            # Stage-1 masks (finite & >= th1)
            m1_fit = np.isfinite(s1_fit) & (s1_fit >= th1)
            m1_eval = np.isfinite(s1_eval) & (s1_eval >= th1)
            m1_lt = np.isfinite(s1_lt) & (s1_lt >= th1)
            m1_052 = np.isfinite(s1_052) & (s1_052 >= th1)
            m1_rec = np.isfinite(s1_rec) & (s1_rec >= th1)
            # Stage-1-positive training subset
            sub = m1_fit
            y_sub = y_fit[sub]
            # build Stage-2 scores on the training subset
            cand = {}
            # LR combos
            lr_keys = {
                'lr_shape': SHAPE_FEATURES,
                'lr_shape_plus_entropy': ['rr_entropy', 'drr_entropy']
                                         + SHAPE_FEATURES,
            }
            lr_models = {}
            for lname, keys in lr_keys.items():
                Xd, med = design({k: X2_fit[k][sub] for k in keys
                                  if k in X2_fit}, keys)
                mu, sd = Xd.mean(0), Xd.std(0)
                sd[sd == 0] = 1.0
                lr = LogisticRegression(C=1.0, class_weight='balanced',
                                        max_iter=3000)
                lr.fit((Xd - mu) / sd, y_sub)
                lr_models[lname] = (lr, med, mu, sd, keys)
                sf = lr.predict_proba((Xd - mu) / sd)[:, 1]

                def lr_scores(X2, keys, med, mu, sd, lr):
                    Xd_d, _ = design({k: X2[k] for k in keys if k in X2},
                                     keys, med)
                    return lr.predict_proba((Xd_d - mu) / sd)[:, 1]

                se = lr_scores(X2_eval, keys, med, mu, sd, lr)
                sl = lr_scores(X2_lt, keys, med, mu, sd, lr)
                s052 = lr_scores(X2_052, keys, med, mu, sd, lr)
                srec = lr_scores(X2_rec, keys, med, mu, sd, lr)
                Xd_full, _ = design(X2_fit, keys, med)
                sfit_full = lr.predict_proba((Xd_full - mu) / sd)[:, 1]
                cand[lname] = (sf, sfit_full, se, sl, s052, srec)
            for k in STAGE2_CANDIDATES:
                if k not in X2_fit:
                    continue
                cand[k] = (X2_fit[k][sub], X2_fit[k], X2_eval[k],
                           X2_lt[k], X2_052[k], X2_rec[k])
            # orient + threshold each candidate on the Stage-1-positive
            # TRAIN subset only
            scored = {}
            for k, tup in cand.items():
                sf_sub, sfit_full, se, sl, s052, srec = tup
                sign, auc = orient(y_sub, sf_sub)
                sf_sub = sign * np.asarray(sf_sub, float)
                sfit_full = sign * np.asarray(sfit_full, float)
                se = sign * np.asarray(se, float)
                sl = sign * np.asarray(sl, float)
                s052 = sign * np.asarray(s052, float)
                srec = sign * np.asarray(srec, float)
                if vname == 'A':
                    th2 = th_youden(y_sub, sf_sub)
                else:
                    # R15 convention: fit the threshold on VALID rows
                    # (finite Stage-1 and Stage-2 features); detector-failure
                    # windows cannot be scored and are counted as misses at
                    # evaluation time only.
                    val = m1_fit & np.isfinite(sfit_full)
                    P_val = max(int((val & (y_fit == 1)).sum()), 1)
                    best = None
                    for th in np.unique(sfit_full[np.isfinite(sfit_full)]):
                        fl = val & (sfit_full >= th)
                        if float((fl & (y_fit == 1)).sum()) / P_val \
                                >= COMPOSITE_TARGET - 1e-12:
                            best = float(th)
                    th2 = best
                if th2 is None:
                    continue
                def comp_recall(y, m1, s2, th):
                    fl = m1 & np.isfinite(s2) & (s2 >= th)
                    return float((fl & (y == 1)).sum()) / max(
                        int((y == 1).sum()), 1)
                def comp_spec(y, m1, s2, th):
                    fl = m1 & np.isfinite(s2) & (s2 >= th)
                    return float(((~fl) & (y == 0)).sum()) / max(
                        int((y == 0).sum()), 1)
                row = {
                    'stage2_auc_on_stage1pos_train': (round(auc, 4)
                                                      if auc else None),
                    'sign': sign, 'theta2': round(float(th2), 4),
                    'train_composite_recall': round(comp_recall(
                        y_fit, m1_fit, sfit_full, th2), 4),
                    'train_composite_spec': round(comp_spec(
                        y_fit, m1_fit, sfit_full, th2), 4),
                    'afdb_eval_recall': round(comp_recall(
                        y_eval, m1_eval, se, th2), 4),
                    'afdb_eval_spec': round(comp_spec(
                        y_eval, m1_eval, se, th2), 4),
                    'ltafdb_recall': round(comp_recall(
                        y_lt, m1_lt, sl, th2), 4),
                    'ltafdb_spec': round(comp_spec(
                        y_lt, m1_lt, sl, th2), 4),
                    'device_052': device_eval(
                        m1_052 & np.isfinite(s052) & (s052 >= th2),
                        dev_starts),
                    'device_rec_latest': device_eval(
                        m1_rec & np.isfinite(srec) & (srec >= th2),
                        dev_starts_r),
                }
                row['pass'] = bool(
                    row['device_052']['m4_alarms'] == 0
                    and row['afdb_eval_recall'] >= 0.90
                    and row['ltafdb_recall'] >= 0.90)
                # ORACLE theta2 (post-hoc): largest theta2 keeping BOTH
                # afdb-eval and ltafdb recall >= 0.90; then check 052.
                orc = None
                for th in np.unique(se[np.isfinite(se)]):
                    ra = comp_recall(y_eval, m1_eval, se, th)
                    rl = comp_recall(y_lt, m1_lt, sl, th)
                    if ra >= 0.90 - 1e-12 and rl >= 0.90 - 1e-12:
                        orc = float(th)
                if orc is not None:
                    fl052 = m1_052 & np.isfinite(s052) & (s052 >= orc)
                    row['theta2_oracle_eval'] = round(float(orc), 4)
                    row['device_052_oracle'] = device_eval(fl052, dev_starts)
                    row['pass_oracle_threshold'] = bool(
                        row['device_052_oracle']['m4_alarms'] == 0)
                else:
                    row['theta2_oracle_eval'] = None
                    row['pass_oracle_threshold'] = False
                scored[k] = row
            # select Stage-2 scorer by training-side composite spec@R90
            sel = None
            for k, row in scored.items():
                key = (row['train_composite_recall'] >= COMPOSITE_TARGET,
                       row['train_composite_spec'])
                if sel is None or key > sel[0]:
                    sel = (key, k)
            winner = sel[1] if sel else None
            wl_res['variants'][vname] = {
                'theta1': round(float(th1), 4),
                'stage1_train_recall_all': round(recall_all(
                    y_fit, s1_fit, th1), 4),
                'stage1_train_recall_valid': round(float(
                    (m1_fit & (y_fit == 1)).sum()) / max(
                    int(((y_fit == 1) & np.isfinite(s1_fit)).sum()), 1), 4),
                'stage1_afdb_eval_recall': round(recall_all(
                    y_eval, s1_eval, th1), 4),
                'stage1_ltafdb_recall': round(recall_all(
                    y_lt, s1_lt, th1), 4),
                'stage1_052_flags': int(m1_052.sum()),
                'stage1_052_events': int(len(m4_alarms(m1_052))),
                'stage1_rec_latest_events': int(len(m4_alarms(m1_rec))),
                'stage1_pos_train': int(m1_fit.sum()),
                'stage1_pos_train_af': int((m1_fit & (y_fit == 1)).sum()),
                'selected_stage2': winner,
                'candidates': scored,
            }
        result['window_lengths'][wl] = wl_res

    # ---------------- post-hoc single-stage sensitivity ----------------
    # NOT pre-registered: scans the same features with permissive train-side
    # recall targets.  Reported separately to show where the blocker sits.
    for wl in ('10s', '30s'):
        if wl == '10s':
            def getfit():
                return (np.concatenate([ptb['ptb_' + STAGE1],
                                        af[f'afdb10_{STAGE1}'][sp == 0]]),
                        np.concatenate([ptb['ptb_y'].astype(int),
                                        yaf[sp == 0]]),
                        {k: np.concatenate([ptb['ptb_' + k],
                                            af[f'afdb10_{k}'][sp == 0]])
                         for k in ALL_FEATURES})
            def geteval():
                return (af[f'afdb10_{STAGE1}'][sp == 1], yaf[sp == 1],
                        {k: af[f'afdb10_{k}'][sp == 1]
                         for k in ALL_FEATURES})
            def getlt():
                return (lt[f'ltafdb10_{STAGE1}'], ylt,
                        {k: lt[f'ltafdb10_{k}'] for k in ALL_FEATURES})
            def getdev(name):
                return ({k: dv[f'{name}_deploy_xqrs250_paired10_{k}']
                         for k in ALL_FEATURES},)
            starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
            starts_r = dv['rec_latest_deploy_xqrs250_paired_start']
        else:
            def getfit():
                return (af[f'afdb30_{STAGE1}'][sp == 0], yaf[sp == 0],
                        {k: af[f'afdb30_{k}'][sp == 0]
                         for k in ALL_FEATURES})
            def geteval():
                return (af[f'afdb30_{STAGE1}'][sp == 1], yaf[sp == 1],
                        {k: af[f'afdb30_{k}'][sp == 1]
                         for k in ALL_FEATURES})
            def getlt():
                return (lt[f'ltafdb30_{STAGE1}'], ylt,
                        {k: lt[f'ltafdb30_{k}'] for k in ALL_FEATURES})
            def getdev(name):
                return ({k: dv[f'{name}_deploy_xqrs250_paired30_{k}']
                         for k in ALL_FEATURES},)
            starts = dv['ecg_real_052_deploy_xqrs250_paired_start']
            starts_r = dv['rec_latest_deploy_xqrs250_paired_start']
        s1_fit, y_fit, Xf = getfit()
        s1_eval, y_eval, Xe = geteval()
        s1_lt, y_lt, Xl = getlt()
        X052 = getdev('ecg_real_052')[0]
        Xrec = getdev('rec_latest')[0]
        post = {}
        for k in ALL_FEATURES:
            sf = Xf[k]
            sign, auc = orient(y_fit, sf)
            if auc is None:
                continue
            sf, se, sl = sign * sf, sign * Xe[k], sign * Xl[k]
            s052, srec = sign * X052[k], sign * Xrec[k]
            for tgt in (0.95, 0.97, 0.98, 0.99):
                th = th_valid(y_fit, sf, tgt)
                if th is None:
                    continue
                fl052 = np.isfinite(s052) & (s052 >= th)
                flrec = np.isfinite(srec) & (srec >= th)
                row = {
                    'theta': round(float(th), 4),
                    'afdb_eval_recall': round(recall_all(y_eval, se, th), 4),
                    'ltafdb_recall': round(recall_all(y_lt, sl, th), 4),
                    'device_052_flags': int(fl052.sum()),
                    'device_052_events': int(len(m4_alarms(fl052))),
                    'device_rec_latest_events': int(len(m4_alarms(flrec))),
                }
                row['pass'] = bool(row['device_052_events'] == 0
                                   and row['afdb_eval_recall'] >= 0.90
                                   and row['ltafdb_recall'] >= 0.90)
                post.setdefault(k, {})[str(tgt)] = row
        result['posthoc_single_stage'][wl] = post

    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    summary = {}
    for wl in ('10s', '30s'):
        for v in ('A', 'B'):
            vv = result['window_lengths'][wl]['variants'][v]
            summary[f'{wl}_{v}'] = {
                'theta1': vv['theta1'],
                'stage1_afdb_recall': vv['stage1_afdb_eval_recall'],
                'stage1_ltafdb_recall': vv['stage1_ltafdb_recall'],
                'stage1_052_events': vv['stage1_052_events'],
                'selected_stage2': vv['selected_stage2'],
                'selected_pass': (vv['candidates'][vv['selected_stage2']]
                                  ['pass'] if vv['selected_stage2'] else None),
                'n_candidates_pass': int(sum(
                    1 for r in vv['candidates'].values() if r['pass'])),
            }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print('written', OUT / 'result.json')


if __name__ == '__main__':
    main()
