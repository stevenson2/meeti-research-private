#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R13 phase-0 prep: PTB-XL single-lead lead-II 100 Hz dataset.

Builds:
  runs/r13_ptbxl_5cls_smoke/ptbxl_lead2_train.npz
  runs/r13_ptbxl_5cls_smoke/ptbxl_lead2_test.npz

Train = all PTB-XL strat_fold != 10 records (~19.6k).
Test  = strat_fold == 10 records (2203), reconciled with the existing
fold10.npz cache (lead II channel: fold10.npz x[:, 1]).

No normalization is applied here: this is a new declared PTB-XL raw-mV
single-lead family, separate from the frozen student_5s benchmark family.
"""
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r13_ptbxl_5cls_smoke'
from project_paths import P as _PP  # 统一路径
PTBXL = Path(_PP['ptbxl_copy'])
FOLD10 = Path('/home/devcontainers/ptbxl_data/fold10.npz')
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


def load_diag(root):
    df = pd.read_csv(root / 'ptbxl_database.csv', index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(lambda s: eval(s))
    agg = pd.read_csv(root / 'scp_statements.csv', index_col=0)
    diag = agg[agg['diagnostic'] == 1.0]

    def aggregate(y_dic):
        out = []
        for key in y_dic:
            if key in diag.index:
                c = diag.loc[key].get('diagnostic_class')
                if str(c) != 'nan' and c not in out:
                    out.append(c)
        return out

    return df, diag, aggregate


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    df, _, aggregate = load_diag(PTBXL)
    n = len(df)
    print('PTB-XL rows:', n, flush=True)

    x = np.zeros((n, 1000), dtype=np.float32)
    y5 = np.zeros((n, 5), dtype=np.float32)
    folds = df['strat_fold'].to_numpy(np.int32)
    ecg_ids = df.index.to_numpy(np.int64)
    for i, (ecg_id, row) in enumerate(df.iterrows()):
        sig, _ = wfdb.rdsamp(str(PTBXL / row['filename_lr']))
        if sig.shape[0] < 1000:
            sig = np.pad(sig, ((0, 1000 - sig.shape[0]), (0, 0)))
        x[i] = sig[:1000, 1].astype(np.float32)
        cls = aggregate(row['scp_codes'])
        y5[i] = [1.0 if c in cls else 0.0 for c in CLASSES]
        if (i + 1) % 2000 == 0:
            print('extracted', i + 1, f'elapsed {time.time()-t0:.0f}s',
                  flush=True)

    train_mask = folds != 10
    test_mask = folds == 10
    print('train n', int(train_mask.sum()), 'test n', int(test_mask.sum()),
          flush=True)

    np.savez_compressed(
        OUT / 'ptbxl_lead2_train.npz',
        x=x[train_mask].astype(np.float32), y5=y5[train_mask].astype(np.float32),
        folds=folds[train_mask], ecg_ids=ecg_ids[train_mask])
    np.savez_compressed(
        OUT / 'ptbxl_lead2_test.npz',
        x=x[test_mask].astype(np.float32), y5=y5[test_mask].astype(np.float32),
        folds=folds[test_mask], ecg_ids=ecg_ids[test_mask])

    # Reconcile with the frozen fold10 cache.
    fold = np.load(FOLD10)
    xref = fold['x'][:, :, 1].astype(np.float32)
    yref = fold['y5'].astype(np.float32)
    x_new = x[test_mask].astype(np.float32)
    y_new = y5[test_mask].astype(np.float32)
    max_abs = float(np.abs(xref - x_new).max())
    label_mismatch = int((np.abs(yref - y_new).sum(axis=1) > 1e-6).sum())

    train_prev = y5[train_mask].mean(axis=0).round(4).tolist()
    test_prev = y5[test_mask].mean(axis=0).round(4).tolist()
    y_bin = (y5[:, 3] == 0).astype(np.float32)
    result = {
        'arm': 'r13_ptbxl_prep',
        'output_dir': str(OUT),
        'n_total': int(n),
        'n_train': int(train_mask.sum()),
        'n_test': int(test_mask.sum()),
        'classes': CLASSES,
        'train_prevalence_5cls': train_prev,
        'test_prevalence_5cls': test_prev,
        'train_binary_abnormal_prevalence': float(y_bin[train_mask].mean()),
        'test_binary_abnormal_prevalence': float(y_bin[test_mask].mean()),
        'reconcile_vs_fold10': {
            'lead2_channel': 1,
            'max_abs_diff': max_abs,
            'label_mismatch': label_mismatch,
            'ok': bool(max_abs < 1e-4 and label_mismatch == 0),
        },
        'input': '(1000,1) lead II @100 Hz raw mV, 10 s per record',
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(OUT / 'prep_result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
