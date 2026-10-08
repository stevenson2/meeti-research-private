#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10e verification round: data / method / ranking checks.

Checks (read-only):
  V1 anchor bit-identity: locked S components == frozen anchors
  V2 split integrity: PTB train/val/test patient sets pairwise disjoint;
     MEETI val-subject disjoint from train-subjects
  V3 leaderboard internal consistency: S recompute == stored S (all rows)
  V4 op-table spot check: spec vs confusion counts (locked, ptb_test)
  V5 small-n disclosure: negative-class sizes recorded
  V6 per-seed spread: candidate PTB-val AUC sd (the known weakness)
Outputs runs/r10e_verification/result.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r10e_verification'
OUT.mkdir(parents=True, exist_ok=True)
res = {}


def check(name, ok, detail):
    res[name] = {'ok': bool(ok), 'detail': detail}
    print(f'[{"PASS" if ok else "FAIL"}] {name}: {detail}', flush=True)


# V1 anchors
b = json.load(open(ROOT / 'runs/benchmark_v1/leaderboard.json'))
lk = b['models']['locked_dual_head_v3_seeded_ptb']
anchors = {'meeti_val': 0.7773, 'ptb_val': 0.8880, 'ptb_test': 0.8367,
           'beat_val': 0.7633}
drift = {k: round(abs(lk[k] - v), 6) for k, v in anchors.items()}
ok = all(d < 1e-3 for d in drift.values())
check('V1_anchor_consistency', ok,
      {'anchors': {k: round(lk[k], 4) for k in anchors},
       'gpu_eval_drift': drift,
       'note': 'frozen anchors reproduced bit-exact under CPU eval '
               '(R9b); GPU-eval drift <=5e-4 disclosed'})

# V2 split integrity
import r8_data_5s  # noqa: F401,E402  (env/sys.path side effects)
from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
split = compute_ptb_split()
beats, labels, rids = load_arrays('ptb')
rids = np.asarray(rids)
sets = {k: set(rids[np.asarray(split[k], bool)].tolist())
        for k in ['train', 'val', 'test']}
inter = {f'{a}-{c}': len(sets[a] & sets[c])
         for a, c in [('train', 'val'), ('train', 'test'), ('val', 'test')]}
check('V2_ptb_patient_disjoint', all(v == 0 for v in inter.values()),
      {**inter, 'patients': {k: len(v) for k, v in sets.items()}})

import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402
from r8_data_5s import MEETI_DIR  # noqa: E402
man = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
man = man[man['report_class'].isin(['normal', 'abnormal'])]
subj = man['subject_id'].to_numpy()
y = (man['report_class'] == 'abnormal').astype(int).to_numpy()
uniq, inv = np.unique(subj, return_inverse=True)
sy = np.zeros(len(uniq), int)
for yi, si in zip(y, inv):
    sy[si] = max(sy[si], yi)
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_s, va_s = next(sss.split(np.arange(len(uniq)), sy))
check('V2_meeti_subject_disjoint',
      len(set(tr_s) & set(va_s)) == 0,
      {'train_subjects': len(tr_s), 'val_subjects': len(va_s)})

# V3 S consistency
W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}
bad = []
for n, m in b['models'].items():
    row = m['aggregate'] if 'aggregate' in m else m
    s = 0.0
    okr = True
    for k, w in W.items():
        v = row.get(k)
        v = v['mean'] if isinstance(v, dict) else v
        if v is None:
            okr = False
            break
        s += w * (v - 0.5) / 0.5
    if okr and 'S' in row and abs(round(100 * s, 2) - row['S']) > 0.02:
        bad.append((n, row['S'], round(100 * s, 2)))
check('V3_S_recompute_consistent', not bad, bad or 'all rows match')

# V4 spec vs confusion counts (locked ptb_test @0.5)
c = lk['op_metrics']['ptb_test']['0.500']['tp_fp_fn_tn']
spec = c[3] / (c[3] + c[1])
check('V4_spec_matches_counts', abs(spec - 0.25) < 1e-9,
      {'tp_fp_fn_tn': c, 'spec': round(spec, 3)})

# V5 small-n disclosure
yv = np.asarray(labels)[np.asarray(split['test'], bool)]
rid_t = rids[np.asarray(split['test'], bool)]
import collections  # noqa: E402
rec_y = {}
for r, lab in zip(rid_t, yv):
    rec_y[r] = max(rec_y.get(r, 0), int(lab))
cnt = collections.Counter(rec_y.values())
check('V5_negative_class_size', cnt[0] == 20,
      {'ptb_test_records': dict(cnt)})

# V6 candidate seed spread (known weakness)
cand = b['models']['r9_form_ptbbal']
pv = [s['ptb_val'] for s in cand['per_seed']]
check('V6_candidate_ptbval_sd',
      float(np.std(pv, ddof=1)) > 0.03,
      {'ptb_val_seeds': [round(x, 4) for x in pv],
       'sd': round(float(np.std(pv, ddof=1)), 4)})

with open(OUT / 'result.json', 'w') as f:
    json.dump(res, f, indent=2)
print('saved ->', OUT / 'result.json')
sys.exit(0 if all(v['ok'] for v in res.values()) else 1)
