#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16-A1: shape-aware RR feature feasibility (zero training, pre-registered).

Target (pre-registered in runs/R16_PREREG.md):
  exists a single feature or LR combo such that ecg_real_052 has event-level
  FP=0 under the M4 protocol (1-of-5, cooldown 20) AND the afdb held-out
  records keep AF recall >= 0.90.

Threshold rule: theta = largest TRAIN-side threshold with train recall >= 0.90
(PTB folds1-9 + afdb fit for 10 s; afdb fit only for 30 s).  Device and
afdb-eval are never used for threshold selection.  For each scorer we also
report the ORACLE eval-side threshold (the most favourable threshold that
still keeps afdb recall >= 0.90); if 052 still alarms there, no threshold of
that scorer can satisfy both criteria (monotone score), so this is a strong
necessary-condition diagnostic (clearly labelled post-hoc).

Outputs: runs/r16_rr_shape/{result.json, result_single.json}
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from ecg_bench import train_theta_at_recall  # noqa: E402
from r16_rr_features import ALL_FEATURES, BASE_FEATURES, SHAPE_FEATURES  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r16_rr_shape'
M4_CONFIRM = 5
M4_COOLDOWN = 20
R_TARGET = 0.90
IRREG_ENT10 = 1.5
IRREG_ENT30 = 2.0
FEATS = ALL_FEATURES


def m4_alarms(flags, confirm=M4_CONFIRM, cooldown=M4_COOLDOWN):
    trig, last = [], -10 ** 9
    for i in range(len(flags)):
        lo = max(0, i - confirm + 1)
        if flags[lo:i + 1].any() and i - last >= cooldown:
            trig.append(i)
            last = i
    return trig


def auc_safe(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    if m.sum() < 20 or len(np.unique(y[m])) < 2:
        return None
    return round(float(roc_auc_score(y[m], s[m])), 4)


def orient_fit(sf, yf):
    """Return (sign, auc) so that sign*score is AF-oriented on fit."""
    a = auc_safe(yf, sf)
    if a is None:
        return None, None
    return (1.0 if a >= 0.5 else -1.0), a


def op_point(y, s, theta):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    flags = np.isfinite(s) & (s >= theta)
    P = max(int((y == 1).sum()), 1)
    N = max(int((y == 0).sum()), 1)
    rec = float((flags & (y == 1)).sum()) / P
    spec = float(((~flags) & (y == 0)).sum()) / N
    return {'n': int(len(y)), 'n_valid': int(np.isfinite(s).sum()),
            'recall': round(rec, 4), 'spec': round(spec, 4),
            'n_flag': int(flags.sum())}


def device_op(scores, starts, theta):
    scores = np.asarray(scores, float)
    flags = np.isfinite(scores) & (scores >= theta)
    alarms = m4_alarms(flags)
    return {
        'n_windows': int(len(scores)),
        'n_valid': int(np.isfinite(scores).sum()),
        'n_flag': int(flags.sum()),
        'm4_alarms': int(len(alarms)),
        'm4_alarm_starts_s': [round(float(starts[i]), 1) for i in alarms[:10]],
        'max_score': round(float(np.nanmax(scores)), 4) if np.isfinite(scores).any() else None,
        'median_score': round(float(np.nanmedian(scores)), 4) if np.isfinite(scores).any() else None,
    }


def design(Xd, keys, med=None):
    Xm = np.column_stack([np.asarray(Xd[k], float) for k in keys])
    ind = np.column_stack([np.isnan(Xm[:, j]).astype(float)
                           for j in range(Xm.shape[1])])
    if med is None:
        med = np.nanmedian(Xm, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    return np.hstack([np.where(np.isnan(Xm), med, Xm), ind]), med


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    ptb = dict(np.load(OUT / 'ptb_features.npz'))
    af = dict(np.load(OUT / 'afdb_features.npz'))
    lt = dict(np.load(OUT / 'ltafdb_features.npz'))
    dv = dict(np.load(OUT / 'device_features.npz'))
    if args.quick:
        for d in (ptb, af, lt):
            pass
        ptb = {k: (v[:2000] if k.startswith('ptb_') and v.ndim == 1
                   and len(v) > 2000 else v) for k, v in ptb.items()}

    y_ptb = ptb['ptb_y'].astype(int)
    y_af = af['afdb_y'].astype(int)
    split_af = af['afdb_split'].astype(int)
    y_lt = lt['ltafdb_y'].astype(int)
    dev_starts = {
        'ecg_real_052': dv['ecg_real_052_deploy_xqrs250_10s_start'],
        'rec_latest': dv['rec_latest_deploy_xqrs250_10s_start'],
    }
    dev_paired_starts = {
        'ecg_real_052': dv['ecg_real_052_deploy_xqrs250_paired_start'],
        'rec_latest': dv['rec_latest_deploy_xqrs250_paired_start'],
    }
    det = 'deploy_xqrs250'

    result = {
        'arm': 'r16_rr_shape_prep (A1)',
        'protocol': ('zero-training shape-aware RR feasibility; theta = max '
                     'train-side threshold with train recall >= 0.90; device '
                     'and afdb-eval never used for threshold selection; '
                     'M4 1-of-5 cooldown 20'),
        'prereg': 'runs/R16_PREREG.md',
        'features': FEATS,
        'n_features': len(FEATS),
        'domains': {},
        'single_feature': {},
        'lr_combo': {},
        'irregular_negative_subset': {},
        'acceptance': {},
        'disclosures': [],
    }

    # ---------------- fit/eval pools ----------------
    def pool_10(which):
        if which == 'fit':
            keys = [f'ptb_{k}' for k in FEATS] + [f'afdb10_{k}' for k in FEATS]
            # PTB block then afdb-fit block
            Xp = {k: ptb[f'ptb_{k}'] for k in FEATS}
            Xa = {k: af[f'afdb10_{k}'][split_af == 0] for k in FEATS}
            X = {k: np.concatenate([Xp[k], Xa[k]]) for k in FEATS}
            y = np.concatenate([y_ptb, y_af[split_af == 0]])
            dom = np.array(['ptb'] * len(y_ptb) +
                           ['afdb_fit'] * int((split_af == 0).sum()))
            return X, y, dom
        X = {k: af[f'afdb10_{k}'][split_af == 1] for k in FEATS}
        return X, y_af[split_af == 1], np.array(['afdb_eval'] *
                                                int((split_af == 1).sum()))

    def pool_30(which):
        if which == 'fit':
            X = {k: af[f'afdb30_{k}'][split_af == 0] for k in FEATS}
            return X, y_af[split_af == 0], np.array(['afdb_fit'] *
                                                    int((split_af == 0).sum()))
        X = {k: af[f'afdb30_{k}'][split_af == 1] for k in FEATS}
        return X, y_af[split_af == 1], np.array(['afdb_eval'] *
                                                int((split_af == 1).sum()))

    def lt_pool(wl):
        pre = 'ltafdb10' if wl == 10 else 'ltafdb30'
        return {k: lt[f'{pre}_{k}'] for k in FEATS}, y_lt, \
            np.array(['ltafdb'] * len(y_lt))

    pools = {
        '10s': {'fit': pool_10('fit'), 'eval': pool_10('eval'),
                'ltafdb': lt_pool(10)},
        '30s': {'fit': pool_30('fit'), 'eval': pool_30('eval'),
                'ltafdb': lt_pool(30)},
    }
    for wl in pools:
        Xf, yf, _ = pools[wl]['fit']
        Xe, ye, _ = pools[wl]['eval']
        Xl, yl, _ = pools[wl]['ltafdb']
        result['domains'][wl] = {
            'fit_n': int(len(yf)), 'fit_pos': int(yf.sum()),
            'afdb_eval_n': int(len(ye)), 'afdb_eval_pos': int(ye.sum()),
            'ltafdb_n': int(len(yl)), 'ltafdb_pos': int(yl.sum()),
        }

    # ---------------- single-feature sweep ----------------
    for wl in ('10s', '30s'):
        Xf, yf, _ = pools[wl]['fit']
        Xe, ye, _ = pools[wl]['eval']
        Xl, yl, _ = pools[wl]['ltafdb']
        table = {}
        for k in FEATS:
            sign, auc = orient_fit(Xf[k], yf)
            if sign is None:
                table[k] = {'fit_auc': None, 'skip': 'no fit AUC'}
                continue
            sf = sign * np.asarray(Xf[k], float)
            se = sign * np.asarray(Xe[k], float)
            sl = sign * np.asarray(Xl[k], float)
            mf = np.isfinite(sf)
            th = train_theta_at_recall(yf[mf], sf[mf], R_TARGET)
            if th is None:
                table[k] = {'fit_auc': auc, 'skip': 'no theta'}
                continue
            row = {'fit_auc': round(auc, 4), 'sign': sign,
                   'theta_train_R90': round(float(th), 4),
                   'fit_op': op_point(yf, sf, th),
                   'afdb_eval_op': op_point(ye, se, th),
                   'ltafdb_op': op_point(yl, sl, th)}
            # oracle eval-side threshold (post-hoc necessary-condition test)
            me = np.isfinite(se)
            th_or = train_theta_at_recall(ye[me], se[me], R_TARGET)
            if th_or is not None:
                row['theta_oracle_eval_R90'] = round(float(th_or), 4)
                row['afdb_eval_oracle_op'] = op_point(ye, se, th_or)
                row['ltafdb_oracle_op'] = op_point(yl, sl, th_or)
            # device
            dev = {}
            for name in dev_starts:
                if wl == '10s':
                    s_all = sign * dv[f'{name}_{det}_10s_{k}']
                    dev[name + '_10s_all'] = device_op(
                        s_all, dev_starts[name], th)
                    s_p = sign * dv[f'{name}_{det}_paired10_{k}']
                    dev[name + '_paired10'] = device_op(
                        s_p, dev_paired_starts[name], th)
                    if th_or is not None:
                        dev[name + '_paired10_oracle'] = device_op(
                            s_p, dev_paired_starts[name], th_or)
                else:
                    s_p = sign * dv[f'{name}_{det}_paired30_{k}']
                    dev[name + '_paired30'] = device_op(
                        s_p, dev_paired_starts[name], th)
                    if th_or is not None:
                        dev[name + '_paired30_oracle'] = device_op(
                            s_p, dev_paired_starts[name], th_or)
            row['device'] = dev
            # pass flags (deployable: theta_train_R90; oracle: post-hoc)
            e052 = dev.get('ecg_real_052_10s_all' if wl == '10s'
                           else 'ecg_real_052_paired30', {})
            row['pass_deployable'] = bool(
                e052.get('m4_alarms', 1) == 0
                and row['afdb_eval_op']['recall'] >= R_TARGET)
            e052o = dev.get('ecg_real_052_paired10_oracle' if wl == '10s'
                            else 'ecg_real_052_paired30_oracle', {})
            row['pass_oracle_threshold'] = bool(
                e052o.get('m4_alarms', 1) == 0)
            table[k] = row
        result['single_feature'][wl] = table

    # ---------------- irregular-negative subset AUC ----------------
    def irregular_subset(X, y, pre, ent_th, feat_keys):
        ent = np.asarray(X['rr_entropy'], float)
        neg = (y == 0) & np.isfinite(ent) & (ent >= ent_th)
        pos = y == 1
        out = {'n_pos': int(pos.sum()), 'n_irregular_neg': int(neg.sum()),
               'entropy_threshold': ent_th}
        for k in feat_keys:
            s = np.asarray(X[k], float)
            mm = pos | neg
            auc = auc_safe(y[mm], s[mm])
            out[k] = auc
        return out

    shape_sel = [k for k in SHAPE_FEATURES]
    for wl in ('10s', '30s'):
        Xe, ye, _ = pools[wl]['eval']
        Xl, yl, _ = pools[wl]['ltafdb']
        Xf, yf, _ = pools[wl]['fit']
        eth = IRREG_ENT10 if wl == '10s' else IRREG_ENT30
        result['irregular_negative_subset'][wl] = {
            'afdb_eval': irregular_subset(Xe, ye, wl, eth, shape_sel),
            'ltafdb': irregular_subset(Xl, yl, wl, eth, shape_sel),
            'fit_pool': irregular_subset(Xf, yf, wl, eth, shape_sel),
        }

    # ---------------- LR combos ----------------
    for wl in ('10s', '30s'):
        Xf, yf, _ = pools[wl]['fit']
        Xe, ye, _ = pools[wl]['eval']
        Xl, yl, _ = pools[wl]['ltafdb']
        combos = {
            'lr_all': FEATS,
            'lr_shape': SHAPE_FEATURES,
            'lr_shape_plus_entropy': ['rr_entropy', 'drr_entropy'] + SHAPE_FEATURES,
        }
        result['lr_combo'][wl] = {}
        for cname, keys in combos.items():
            Xd_f, med = design(Xf, keys)
            mu, sd = Xd_f.mean(0), Xd_f.std(0)
            sd[sd == 0] = 1.0
            lr = LogisticRegression(C=1.0, class_weight='balanced',
                                    max_iter=3000)
            lr.fit((Xd_f - mu) / sd, yf)
            sf = lr.predict_proba((Xd_f - mu) / sd)[:, 1]
            Xd_e, _ = design(Xe, keys, med)
            se = lr.predict_proba((Xd_e - mu) / sd)[:, 1]
            Xd_l, _ = design(Xl, keys, med)
            sl = lr.predict_proba((Xd_l - mu) / sd)[:, 1]
            sign, auc = orient_fit(sf, yf)
            if sign is None:
                continue
            sf, se, sl = sign * sf, sign * se, sign * sl
            mf = np.isfinite(sf)
            th = train_theta_at_recall(yf[mf], sf[mf], R_TARGET)
            me = np.isfinite(se)
            th_or = train_theta_at_recall(ye[me], se[me], R_TARGET)
            row = {'n_feat': len(keys), 'fit_auc': round(auc, 4),
                   'theta_train_R90': round(float(th), 4),
                   'fit_op': op_point(yf, sf, th),
                   'afdb_eval_op': op_point(ye, se, th),
                   'ltafdb_op': op_point(yl, sl, th),
                   'theta_oracle_eval_R90': (round(float(th_or), 4)
                                             if th_or is not None else None),
                   'afdb_eval_oracle_op': (op_point(ye, se, th_or)
                                           if th_or is not None else None),
                   'device': {}}
            devs = {}
            for name in dev_starts:
                if wl == '10s':
                    Xdev_all = {k: dv[f'{name}_{det}_10s_{k}'] for k in keys}
                    Xdev_p = {k: dv[f'{name}_{det}_paired10_{k}']
                              for k in keys}
                else:
                    Xdev_all = {k: dv[f'{name}_{det}_paired30_{k}']
                                for k in keys}
                    Xdev_p = Xdev_all
                Xd_d, _ = design(Xdev_all, keys, med)
                s_all = sign * lr.predict_proba(
                    (Xd_d - mu) / sd)[:, 1]
                Xd_dp, _ = design(Xdev_p, keys, med)
                s_p = sign * lr.predict_proba((Xd_dp - mu) / sd)[:, 1]
                if wl == '10s':
                    row['device'][name + '_10s_all'] = device_op(
                        s_all, dev_starts[name], th)
                    row['device'][name + '_paired10'] = device_op(
                        s_p, dev_paired_starts[name], th)
                    if th_or is not None:
                        row['device'][name + '_paired10_oracle'] = device_op(
                            s_p, dev_paired_starts[name], th_or)
                else:
                    row['device'][name + '_paired30'] = device_op(
                        s_p, dev_paired_starts[name], th)
                    if th_or is not None:
                        row['device'][name + '_paired30_oracle'] = device_op(
                            s_p, dev_paired_starts[name], th_or)
            key052 = ('ecg_real_052_10s_all' if wl == '10s'
                      else 'ecg_real_052_paired30')
            row['pass_deployable'] = bool(
                row['device'].get(key052, {}).get('m4_alarms', 1) == 0
                and row['afdb_eval_op']['recall'] >= R_TARGET)
            row['pass_oracle_threshold'] = bool(
                row['device'].get(key052 + '_oracle', {}).get(
                    'm4_alarms', 1) == 0)
            result['lr_combo'][wl][cname] = row

    # ---------------- device whole-record diagnostics ----------------
    result['device_whole_record'] = {}
    for name in dev_starts:
        entry = json.loads(str(dv[f'{name}_meta']))
        result['device_whole_record'][name] = {
            'dur_s': entry['dur_s'], 'fs_raw': entry['fs_raw'],
            'n_peaks_xqrs': entry['n_peaks_xqrs'],
            'n_peaks_pt100': entry['n_peaks_pt100'],
            'whole_deploy_xqrs250': {
                k: (round(v, 4) if isinstance(v, float)
                    and np.isfinite(v) else v)
                for k, v in entry['whole_deploy_xqrs250'].items()},
            'whole_pt100': {
                k: (round(v, 4) if isinstance(v, float)
                    and np.isfinite(v) else v)
                for k, v in entry['whole_pt100'].items()},
        }

    # ---------------- acceptance summary ----------------
    best = {}
    for wl in ('10s', '30s'):
        cand = []
        for k, row in result['single_feature'][wl].items():
            if 'theta_train_R90' not in row:
                continue
            cand.append((k, row['pass_deployable'],
                         row['afdb_eval_op']['recall'],
                         row['device'].get(
                             'ecg_real_052_10s_all' if wl == '10s'
                             else 'ecg_real_052_paired30', {}
                         ).get('m4_alarms')))
        cand.sort(key=lambda x: (-int(x[1]), -(x[2] or 0)))
        best[wl] = cand[:10]
    result['acceptance'] = {
        'criterion': '052 event-level FP=0 (M4 1-of-5/cooldown20) AND afdb '
                     'held-out recall >= 0.90 at the train-side theta_R90',
        'n_features_scored': len(FEATS),
        'n_pass_deployable_10s': int(sum(
            1 for r in result['single_feature']['10s'].values()
            if r.get('pass_deployable'))),
        'n_pass_deployable_30s': int(sum(
            1 for r in result['single_feature']['30s'].values()
            if r.get('pass_deployable'))),
        'n_pass_oracle_10s': int(sum(
            1 for r in result['single_feature']['10s'].values()
            if r.get('pass_oracle_threshold'))),
        'n_pass_oracle_30s': int(sum(
            1 for r in result['single_feature']['30s'].values()
            if r.get('pass_oracle_threshold'))),
        'best_by_window': {k: [list(x) for x in v] for k, v in best.items()},
    }
    result['disclosures'] = [
        'only 2 device normal recordings, no device AF positives',
        'PTB has 10 s records only; 30 s thresholds are fitted on afdb fit '
        'records alone (no PTB 30 s)',
        '10 s windows give ~10-13 RR; shape/entropy variance disclosed, '
        '30 s paired windows use beat-position history and are causal',
        'irregular-negative subsets are defined by entropy thresholds '
        f'(10 s >= {IRREG_ENT10}, 30 s >= {IRREG_ENT30}); they are mixed '
        'PAC/PVC/sinus-arrhythmia negatives, not a clean clinical label',
        'oracle thresholds are post-hoc diagnostics, not deployable',
        'multiple shape features are screened in A1 (feasibility search); '
        'A2 may only use the single pre-registered winner',
    ]

    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps({
        'domains': result['domains'],
        'acceptance': result['acceptance'],
    }, indent=2, ensure_ascii=False))
    print('written', OUT / 'result.json')


if __name__ == '__main__':
    main()
