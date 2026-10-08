#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16 detector QC: PT100 (A4 family) vs database QRS annotations.

Quantifies the detector-failure rate that drives the NaN/miss penalty in the
R16 gates.  For afdb (250 Hz) and ltafdb (128 Hz) we compare, on 10 s / 30 s
windows, the PT100 beat count against the database .qrs annotation count.
Records: >=20 per domain (all 23 afdb records, 30 ltafdb records).
Outputs: runs/r16_rr_shape/detector_qc.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r14_rr_baseline import detect_peaks  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r16_rr_shape/detector_qc.json'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')


def qc_domain(dst, fs, up, down, records, max_windows=20, seed=42):
    rng = np.random.default_rng(seed)
    rows = []
    for r in records:
        obj = wfdb.rdrecord(str(dst / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 1]
        qrs = wfdb.rdann(str(dst / r), 'qrs').sample
        n_samp = len(sig)
        for L in (10, 30):
            win = L * fs
            starts = np.arange(0, max(n_samp - win, 1), win)
            if len(starts) > max_windows:
                starts = rng.choice(starts, max_windows, replace=False)
            for s in starts:
                w = sig[s:s + win]
                if len(w) < win:
                    continue
                w100 = resample_poly(w, up, down)
                p = detect_peaks(w100)
                n_ann = int(((qrs >= s) & (qrs < s + win)).sum())
                rows.append({'rec': r, 'L': L, 'start': int(s),
                             'pt100_beats': int(len(p)),
                             'qrs_beats': int(n_ann),
                             'diff': int(len(p) - n_ann)})
    d = np.array([x['diff'] for x in rows])
    ae = np.abs(d)
    out = {
        'n_windows': len(rows),
        'n_records': len(records),
        'abs_diff_median': float(np.median(ae)),
        'abs_diff_p90': float(np.percentile(ae, 90)),
        'abs_diff_max': int(ae.max()),
        'within_1_frac': float((ae <= 1).mean()),
        'within_2_frac': float((ae <= 2).mean()),
        'pt100_lt3_beats_frac': float(
            (np.array([x['pt100_beats'] for x in rows]) < 3).mean()),
        'qrs_normal_but_pt100_lt3': int(sum(
            1 for x in rows if x['qrs_beats'] >= 5 and x['pt100_beats'] < 3)),
        'examples_fail': [x for x in rows if x['pt100_beats'] < 3][:10],
        'examples_worst': sorted(rows, key=lambda x: -abs(x['diff']))[:10],
    }
    return out


def main():
    af = [p.stem for p in sorted(AFDB.glob('*.hea'))]
    af = [r for r in af if r not in ('00735', '03665')]
    lt = [p.stem for p in sorted(LTAFDB.glob('*.hea'))][:30]
    res = {
        'arm': 'r16_detector_qc',
        'note': ('PT100 = R14-A4 Pan-Tompkins-style detector at 100 Hz; '
                 'reference = database .qrs annotations (afdb unreviewed, '
                 'ltafdb qrs).  10 s and 30 s windows.'),
        'afdb': qc_domain(AFDB, 250, 2, 5, af, max_windows=10),
        'ltafdb': qc_domain(LTAFDB, 128, 25, 32, lt, max_windows=10),
    }
    OUT.write_text(json.dumps(res, indent=2, ensure_ascii=False, default=str),
                   encoding='utf-8')
    print(json.dumps({k: {kk: v[kk] for kk in
                          ('n_windows', 'n_records', 'abs_diff_median',
                           'abs_diff_p90', 'within_1_frac', 'within_2_frac',
                           'pt100_lt3_beats_frac', 'qrs_normal_but_pt100_lt3')}
                      for k, v in res.items() if isinstance(v, dict)},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
