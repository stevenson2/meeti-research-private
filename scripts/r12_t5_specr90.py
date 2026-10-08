#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R12: spec@R90 post-hoc eval for the T5 hardneg arm seeds 42-44 (PTB val
+ test, mean and top2mean calibers) - complements r9_train's summary."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r9_data import load_all  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    'ecg_bench', '/home/devcontainers/meeti/scripts/ecg_bench.py')
eb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eb)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r12_t5_hardneg_specr90.json'

for g in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(g, True)

data = load_all(norm=True, form=True)
split = data['ptb_splits']
rows = []
for s in (42, 43, 44):
    ck = ROOT / f'runs/r9_form-ptbbal-hardneg_seed{s}/ckpt_best__record5s.keras'
    if not ck.exists():
        print(f'skip seed{s}', flush=True)
        continue
    probe = tf.keras.models.load_model(str(ck))
    rec = {'seed': s}
    for name in ['val', 'test']:
        d = eb.ptb_5s(split[name])
        p = eb.predict_windows(probe, d['x'])
        t2 = eb.top2mean_by_record(p, d['win_rec'], d['n_records'])
        mn = eb.record_scores(p, d['win_rec'], d['n_records'])
        rec[name] = {
            'auc_top2mean': round(float(
                __import__('sklearn.metrics', fromlist=['roc_auc_score'])
                .roc_auc_score(d['rec_y'], t2)), 4),
            'auc_mean': round(float(
                __import__('sklearn.metrics', fromlist=['roc_auc_score'])
                .roc_auc_score(d['rec_y'], mn)), 4),
            'spec_at_R90_top2mean': eb.spec_at_recall(d['rec_y'], t2, 0.90),
        }
    rows.append(rec)
    print(rec, flush=True)
    del probe
    tf.keras.backend.clear_session()

with open(OUT, 'w') as f:
    json.dump({'arm': 'r9_form-ptbbal-hardneg', 'seeds': rows}, f, indent=2)
print('saved ->', OUT, flush=True)
