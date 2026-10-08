#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10: build PTB-XL fold-10 (strat_fold==10) validation data.

Mirrors hhi-aml/xai4ecg preprocessing.py: wfdb.rdsamp on filename_lr
(100 Hz, 1000 samples, 12 leads, raw mV, no normalization), scp_codes
aggregated to the 5 diagnostic superclasses via scp_statements.csv
(diagnostic_class, no likelihood threshold), multihot column order =
utils.label_mappings['superdiagnostic'] = ['CD','HYP','MI','NORM','STTC'].

Outputs /home/devcontainers/ptbxl_data/fold10_{x,y5}.npz + meta json.
Purpose: validate the official xai4ecg checkpoints load correctly by
reproducing their reported test macro-AUROC (bootstrap mean ~0.928 resnet
/ ~0.924 xresnet on 'Quantitative Evaluation (all)').
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

DATA = Path('/home/devcontainers/ptbxl_data')
from project_paths import P as _PP  # unified library
LOCAL_ROOT = Path(_PP['ptbxl_copy'])
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


def main():
    # dataset root: user-provided local copy (PTB-XL 1.0.3, records100/500)
    root = LOCAL_ROOT
    assert (root / 'ptbxl_database.csv').exists(), f'missing db csv in {root}'
    print('dataset root:', root)

    df = pd.read_csv(root / 'ptbxl_database.csv', index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: eval(x))
    agg_df = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    diag = agg_df[agg_df.diagnostic == 1.0]

    def aggregate(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in diag.index:
                c = diag.loc[key].diagnostic_class
                if str(c) != 'nan' and c not in tmp:
                    tmp.append(c)
        return tmp

    test = df[df.strat_fold == 10]
    print('fold-10 records:', len(test))

    xs, ys = [], []
    for ecg_id, row in test.iterrows():
        sig, _ = wfdb.rdsamp(str(root / row.filename_lr))
        xs.append(sig.astype(np.float32))
        cls = aggregate(row.scp_codes)
        ys.append([1.0 if c in cls else 0.0 for c in CLASSES])
    x = np.stack(xs)
    y = np.asarray(ys, np.float32)
    print('x:', x.shape, 'y:', y.shape,
          'prevalence:', y.mean(axis=0).round(3).tolist())

    np.savez_compressed(DATA / 'fold10.npz', x=x, y5=y)
    with open(DATA / 'fold10_meta.json', 'w') as f:
        json.dump({'classes': CLASSES, 'n': int(len(y)),
                   'fold': 10, 'source': str(root)}, f, indent=2)
    print('saved ->', DATA / 'fold10.npz')


if __name__ == '__main__':
    main()
