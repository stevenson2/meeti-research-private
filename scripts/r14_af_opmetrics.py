#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14 post-hoc operating-point layer for the AF head (zero training).

Adds to each runs/r14_af_head/af_seed*/result.json:
  - theta_R90tr (BENCHMARK v1.5 leak-free anchor: largest TRAIN-side val
    threshold with val recall >= 0.90) and its fold-10 R/spec;
  - combined Lane-A gate (AF prob >= theta AND RR entropy >= theta_rr from
    A4) fold-10 recall/spec at both theta_tr (Youden) and theta_R90tr.

Uses only saved val/test probabilities (probs.npz) and A4 features.npz;
no model inference, no retraining.
"""
import json
from pathlib import Path

import numpy as np

sys_path = '/home/devcontainers/meeti/scripts'
import sys  # noqa: E402
sys.path.insert(0, sys_path)
from ecg_bench import train_theta_at_recall, youden_threshold  # noqa: E402

ROOT = Path('/home/devcontainers/meeti/runs/r14_af_head')
A4 = Path('/home/devcontainers/meeti/runs/r14_rr_baseline')


def op(y, s, th):
    yp = s >= th
    tp = int((yp & (y == 1)).sum())
    fp = int((yp & (y == 0)).sum())
    fn = int(((~yp) & (y == 1)).sum())
    tn = int(((~yp) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return {'recall': round(rec, 4), 'spec': round(spec, 4),
            'bacc': round((rec + spec) / 2, 4),
            'tp_fp_fn_tn': [tp, fp, fn, tn]}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', default='',
                    help='only af_seed*<suffix> dirs (e.g. _calib20)')
    args = ap.parse_args()
    rr = json.loads((A4 / 'result.json').read_text())
    theta_rr = rr['rr_entropy_confirmator']['theta_tr_youden']
    feats = np.load(A4 / 'features.npz')
    ent_te = feats['te_rr_entropy'].astype(float)
    # sign orientation identical to A4: higher = more AF (train AUC > 0.5)
    summary = {'theta_rr_A4': theta_rr, 'seeds': {}}
    for d in sorted(ROOT.glob(f'af_seed*{args.suffix}')):
        if not (d / 'probs.npz').exists() or not (d / 'result.json').exists():
            print('skip incomplete', d.name, flush=True)
            continue
        seed = int(d.name.split('seed')[1].replace(args.suffix, ''))
        rj = json.loads((d / 'result.json').read_text())
        pr = np.load(d / 'probs.npz')
        p_val, y_val = pr['p_val'].astype(float), pr['y_val'].astype(int)
        p_te, y_te = pr['p_test'].astype(float), pr['y_test'].astype(int)
        theta_tr = float(youden_threshold(y_val, p_val))
        theta_r90 = float(train_theta_at_recall(y_val, p_val, 0.90))
        assert len(p_te) == len(ent_te), (len(p_te), len(ent_te))
        ok = np.isfinite(ent_te)
        comb_tr = (p_te >= theta_tr) & (ent_te >= theta_rr) & ok
        comb_r90 = (p_te >= theta_r90) & (ent_te >= theta_rr) & ok
        rj['metrics']['theta_R90tr_val'] = round(theta_r90, 4)
        rj['metrics']['R_at_theta_R90tr'] = op(y_te, p_te, theta_r90)['recall']
        rj['metrics']['spec_at_theta_R90tr'] = op(y_te, p_te, theta_r90)['spec']
        rj['metrics']['theta_tr_val_youden'] = round(theta_tr, 4)
        rj['metrics']['R_at_theta_tr'] = op(y_te, p_te, theta_tr)['recall']
        rj['metrics']['spec_at_theta_tr'] = op(y_te, p_te, theta_tr)['spec']
        rj['laneA_combined_gate'] = {
            'definition': 'AF prob >= theta AND RR entropy >= theta_rr_A4',
            'theta_rr_A4': theta_rr,
            'at_theta_tr': op(y_te[ok], p_te[ok], theta_tr),
            'at_theta_R90tr': op(y_te[ok], p_te[ok], theta_r90),
            'combined_recall_at_theta_tr': round(
                float((comb_tr & (y_te == 1)).sum())
                / max(int((y_te == 1).sum()), 1), 4),
            'combined_spec_at_theta_tr': round(
                float(((~comb_tr) & (y_te == 0)).sum())
                / max(int((y_te == 0).sum()), 1), 4),
            'combined_recall_at_theta_R90tr': round(
                float((comb_r90 & (y_te == 1)).sum())
                / max(int((y_te == 1).sum()), 1), 4),
            'combined_spec_at_theta_R90tr': round(
                float(((~comb_r90) & (y_te == 0)).sum())
                / max(int((y_te == 0).sum()), 1), 4),
            'n_test_valid_rr': int(ok.sum()),
        }
        (d / 'result.json').write_text(
            json.dumps(rj, indent=2, ensure_ascii=False), encoding='utf-8')
        summary['seeds'][str(seed)] = {
            'theta_tr': round(theta_tr, 4),
            'theta_R90tr': round(theta_r90, 4),
            'R_at_theta_tr': rj['metrics']['R_at_theta_tr'],
            'spec_at_theta_tr': rj['metrics']['spec_at_theta_tr'],
            'R_at_theta_R90tr': rj['metrics']['R_at_theta_R90tr'],
            'spec_at_theta_R90tr': rj['metrics']['spec_at_theta_R90tr'],
            'combined_R_at_theta_R90tr':
                rj['laneA_combined_gate']['combined_recall_at_theta_R90tr'],
            'combined_spec_at_theta_R90tr':
                rj['laneA_combined_gate']['combined_spec_at_theta_R90tr'],
        }
    (ROOT / f'af_opmetrics_summary{args.suffix}.json').write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
