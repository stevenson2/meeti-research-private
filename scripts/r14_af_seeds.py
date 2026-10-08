#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A5: paired 3-seed verdict for the AF head vs the general binary head.

Reads runs/r14_af_head/{af,genbin}_seed{42,43,44}/result.json and applies
the pre-registered R14-A5 discipline:

  PASS iff  for all 3 seeds:  AF_AUC >= genbin_AUC  AND  AF_AUC >= 0.90
       and  mean(AF spec@R90) >= mean(genbin spec@R90) - sd(genbin spec@R90)

The R13 seed42 genbin anchor (0.8750, different protocol: no holdout,
softmax, no class weights) is reported as a reference only.
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path('/home/devcontainers/meeti/runs/r14_af_head')
SEEDS = [42, 43, 44]
OUT = ROOT / 'paired_summary.json'


def load(task, seed):
    p = ROOT / f'{task}_seed{seed}' / 'result.json'
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    rows = []
    for s in SEEDS:
        a, g = load('af', s), load('genbin', s)
        if a is None or g is None:
            rows.append({'seed': s, 'af': None, 'genbin': None})
            continue
        rows.append({
            'seed': s,
            'af_auc': a['metrics']['test_auc'],
            'genbin_auc': g['metrics']['test_auc'],
            'auc_diff': round(a['metrics']['test_auc']
                              - g['metrics']['test_auc'], 4),
            'af_ap': a['metrics']['test_ap'],
            'genbin_ap': g['metrics']['test_ap'],
            'af_specR90': a['metrics']['spec_at_R90_test']['spec'],
            'genbin_specR90': g['metrics']['spec_at_R90_test']['spec'],
            'af_theta_tr': a['metrics']['theta_tr_val_youden'],
            'af_R_at_theta_tr': a['metrics']['R_at_theta_tr'],
            'af_spec_at_theta_tr': a['metrics']['spec_at_theta_tr'],
            'genbin_theta_tr': g['metrics']['theta_tr_val_youden'],
        })
    complete = [r for r in rows if r.get('af_auc') is not None]
    out = {
        'arm': 'r14_af_seeds',
        'seeds': SEEDS,
        'paired_rows': rows,
        'n_complete_seeds': len(complete),
    }
    if len(complete) == 3:
        af_auc = np.array([r['af_auc'] for r in complete])
        gb_auc = np.array([r['genbin_auc'] for r in complete])
        af_s90 = np.array([r['af_specR90'] for r in complete])
        gb_s90 = np.array([r['genbin_specR90'] for r in complete])
        per_seed_pass = [(r['af_auc'] >= r['genbin_auc']) and
                         (r['af_auc'] >= 0.90) for r in complete]
        spec_ok = bool(af_s90.mean() >= gb_s90.mean() - gb_s90.std())
        out.update({
            'af_auc_mean_sd': [round(float(af_auc.mean()), 4),
                               round(float(af_auc.std(ddof=1)), 4)],
            'genbin_auc_mean_sd': [round(float(gb_auc.mean()), 4),
                                   round(float(gb_auc.std(ddof=1)), 4)],
            'af_specR90_mean_sd': [round(float(af_s90.mean()), 4),
                                   round(float(af_s90.std(ddof=1)), 4)],
            'genbin_specR90_mean_sd': [round(float(gb_s90.mean()), 4),
                                       round(float(gb_s90.std(ddof=1)), 4)],
            'per_seed_auc_ge_genbin_and_ge_090': per_seed_pass,
            'n_seeds_passing': int(sum(per_seed_pass)),
            'spec_within_1sd_rule': spec_ok,
            'r13_genbin_seed42_anchor': 0.8750,
            'verdict': ('PASS research-level candidate' if all(per_seed_pass)
                        and spec_ok else
                        'FAIL / downgraded (see paired rows)'),
            'discipline': ('>=3 seeds; paired same-seed comparison; '
                           'single-seed <2pp differences carry no conclusion'),
        })
    OUT.write_text(json.dumps(out, indent=2, ensure_ascii=False),
                   encoding='utf-8')
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
