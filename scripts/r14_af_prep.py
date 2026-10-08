#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A1: PTB-XL AF label prep on the R13 single-lead lead-II family.

Reuses the R13 extraction result (runs/r13_ptbxl_5cls_smoke/
ptbxl_lead2_{train,test}.npz: lead II @100 Hz, 1000 pts, raw mV) and adds
record-level AF labels from ptbxl_database.csv scp_codes:

  y_af      = 'AFIB' in scp_codes   (AFIB present => positive, regardless
                                     of co-occurring statements; declared
                                     AF-vs-rest semantics)
  y_af_pure = y_af AND NOT diagnostic-class NORM (sensitivity variant)

Fold split is untouched: folds 1-9 train (19,634) / fold-10 test (2,203).
The test slice is reconciled against the frozen fold10.npz cache
(lead II = channel 1), same protocol as R13 (max_abs_diff must be 0).

Outputs (runs/r14_af_prep/):
  ptbxl_af_train.npz / ptbxl_af_test.npz  (x, y_af, y_af_pure, y5, folds,
                                           ecg_ids, patient_id)
  result.json                             (counts + semantics + reconcile)

Optional MEETI report-field scan lives in scripts/r14_meeti_af_scan.py
(separate timeboxed job; this script stays zero-network, CPU-only).
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path('/home/devcontainers/meeti')
R13 = ROOT / 'runs/r13_ptbxl_5cls_smoke'
OUT = ROOT / 'runs/r14_af_prep'
from project_paths import P as _PP  # 统一路径
PTBXL = Path(_PP['ptbxl_copy'])
FOLD10 = Path('/home/devcontainers/ptbxl_data/fold10.npz')
CLASSES = ['CD', 'HYP', 'MI', 'NORM', 'STTC']


def load_scp(root):
    df = pd.read_csv(root / 'ptbxl_database.csv', index_col='ecg_id')
    df['scp_codes'] = df['scp_codes'].apply(eval)
    return df


def split_counts(y_af, y5, mask):
    """Return AF/non-AF breakdown for a split mask."""
    af = y_af[mask].astype(bool)
    norm = y5[mask][:, 3].astype(bool)
    nonaf_abn = (~af) & (~norm)
    return {
        'n': int(mask.sum()),
        'af_n': int(af.sum()),
        'af_pct': round(100 * float(af.mean()), 3),
        'af_pure_n': int((af & (~norm)).sum()),
        'af_with_norm_n': int((af & norm).sum()),
        'nonaf_n': int((~af).sum()),
        'nonaf_norm_n': int(((~af) & norm).sum()),
        'nonaf_abn_n': int(nonaf_abn.sum()),
        'nonaf_abn_pct_of_nonaf': round(
            100 * float(nonaf_abn.sum()) / max(int((~af).sum()), 1), 3),
    }


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    tr = np.load(R13 / 'ptbxl_lead2_train.npz')
    te = np.load(R13 / 'ptbxl_lead2_test.npz')
    x_tr, y5_tr, ecg_tr = tr['x'], tr['y5'], tr['ecg_ids']
    x_te, y5_te, ecg_te = te['x'], te['y5'], te['ecg_ids']

    df = load_scp(PTBXL)
    lab = {}
    for ecg_id, row in df.iterrows():
        codes = row['scp_codes']
        keys = set(codes.keys())
        lab[int(ecg_id)] = (1 if 'AFIB' in keys else 0,
                            set(keys),
                            int(row['patient_id']),
                            str(row.get('filename_lr', '')))

    def labels_for(ecg_ids):
        y_af = np.zeros(len(ecg_ids), dtype=np.int8)
        pat = np.zeros(len(ecg_ids), dtype=np.int64)
        missing = 0
        for i, e in enumerate(ecg_ids):
            if int(e) not in lab:
                missing += 1
                continue
            y_af[i] = lab[int(e)][0]
            pat[i] = lab[int(e)][2]
        return y_af, pat, missing

    y_af_tr, pat_tr, miss_tr = labels_for(ecg_tr)
    y_af_te, pat_te, miss_te = labels_for(ecg_te)
    y_af_pure_tr = ((y_af_tr == 1) & (y5_tr[:, 3] == 0)).astype(np.int8)
    y_af_pure_te = ((y_af_te == 1) & (y5_te[:, 3] == 0)).astype(np.int8)

    # Reconcile x against frozen fold10 cache (same protocol as R13).
    fold = np.load(FOLD10)
    xref = fold['x'][:, :, 1].astype(np.float32)
    max_abs = float(np.abs(xref - x_te.astype(np.float32)).max())
    yref = fold['y5'].astype(np.float32)
    label_mismatch = int((np.abs(yref - y5_te.astype(np.float32)).sum(axis=1)
                          > 1e-6).sum())

    np.savez_compressed(
        OUT / 'ptbxl_af_train.npz',
        x=x_tr.astype(np.float32), y_af=y_af_tr, y_af_pure=y_af_pure_tr,
        y5=y5_tr.astype(np.float32), folds=tr['folds'],
        ecg_ids=ecg_tr, patient_id=pat_tr)
    np.savez_compressed(
        OUT / 'ptbxl_af_test.npz',
        x=x_te.astype(np.float32), y_af=y_af_te, y_af_pure=y_af_pure_te,
        y5=y5_te.astype(np.float32), folds=te['folds'],
        ecg_ids=ecg_te, patient_id=pat_te)

    # Co-occurring statements among AF records (disclosure).
    af_codes = Counter()
    n_af_all = 0
    for e in list(ecg_tr) + list(ecg_te):
        v = lab.get(int(e))
        if v and v[0] == 1:
            n_af_all += 1
            for k in v[1]:
                af_codes[k] += 1
    top_af_codes = af_codes.most_common(15)

    all_keys = Counter()
    for e, v in lab.items():
        for k in v[1]:
            all_keys[k] += 1
    afl_total = int(all_keys.get('AFL', 0))

    result = {
        'arm': 'r14_af_prep',
        'family_declaration': (
            'ptbxl_lead2_raw_10s_100hz_v1 (same as R13): single lead II, '
            'raw mV, (1000,1), PTB-XL folds1-9 train / fold10 test'),
        'af_vs_rest_semantics': (
            "positive = 'AFIB' present in scp_codes (multi-label; any "
            "co-occurring statement does not remove the positive). "
            "Negative = AFIB absent (includes NORM and all other "
            "abnormalities). Sensitivity variant y_af_pure = AFIB present "
            "AND diagnostic superclass NORM absent."),
        'label_source': 'ptbxl_database.csv scp_codes (AFIB key); '
                        'AFL independent key count in PTB-XL: %d' % afl_total,
        'n_total': int(len(df)),
        'n_train': int(len(x_tr)),
        'n_test': int(len(x_te)),
        'train': split_counts(y_af_tr, y5_tr, np.ones(len(x_tr), bool)),
        'test_fold10': split_counts(y_af_te, y5_te,
                                    np.ones(len(x_te), bool)),
        'reconcile_vs_fold10': {
            'lead2_channel': 1,
            'max_abs_diff': max_abs,
            'label_mismatch': label_mismatch,
            'ok': bool(max_abs == 0.0 and label_mismatch == 0),
        },
        'missing_ecg_ids': {'train': miss_tr, 'test': miss_te},
        'top_af_cooccurring_codes': top_af_codes,
        'n_af_records_all': n_af_all,
        'input': '(1000,1) lead II @100 Hz raw mV, 10 s per record',
        'elapsed_s': round(time.time() - t0, 1),
    }
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
