#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B5: external AF training/eval pool -> R13 family (1000 pts @100 Hz mV).

Sources and corrected label semantics (verified against PhysioNet docs):
  CinC2017  : record-level N/A/O/~.  A = AF positive (738); N and O are
              sampled as negatives (738 each); ~ (noisy) excluded.
  afpdb     : PAF-Prediction DB.  n* and p* 30-min records are NON-AF
              (p* = pre-PAF hard negatives); nc*/pc* 5-min continuations are
              mixed (contain the PAF episode) -> y=-1 excluded; t* test
              records are unlabelled -> y=-1 excluded.
  aftdb     : AF-Termination DB.  ALL 80 records are 1-min AF excerpts
              (n/s/t = termination class; a/b = challenge test sets) -> y=1;
              termination class stored separately.

All windows: 10 s, 100 Hz, raw mV, resampled from the native rate; amplitude
NOT re-scaled here (B3 applies the PTB p99 alignment).  Channel 0 is used for
afpdb/aftdb (channels unlabeled); disclosed.

Outputs (runs/r15_af_pool/):
  cinc2017.npz / afpdb.npz / aftdb.npz   x, y, rec, start_s, cls
  result.json                            per-source counts + semantics
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import wfdb
from scipy.io import loadmat
from scipy.signal import resample_poly

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r15_af_pool'
DATA = Path('/home/devcontainers/ecg_data')
WIN = 1000


def pad_or_take(y):
    if len(y) < WIN:
        return np.pad(y, (0, WIN - len(y))).astype(np.float32)
    return y[:WIN].astype(np.float32)


def cinc2017():
    root = DATA / 'cinc2017/training2017'
    ref = pd.read_csv(root / 'REFERENCE.csv', header=None,
                      names=['rec', 'label'])
    by = {L: ref[ref.label == L]['rec'].tolist()
          for L in ['A', 'N', 'O', '~']}
    rng = np.random.default_rng(42)
    pick = {'A': by['A'],
            'N': list(rng.choice(by['N'], min(len(by['A']), len(by['N'])),
                                 replace=False)),
            'O': list(rng.choice(by['O'], min(len(by['A']), len(by['O'])),
                                 replace=False))}
    xs, ys, recs, cls = [], [], [], []
    for L in ['A', 'N', 'O']:
        for r in pick[L]:
            m = loadmat(root / f'{r}.mat')
            v = np.asarray(m['val'], np.float64).ravel() / 1000.0  # mV
            y100 = resample_poly(v, 1, 3)                          # 300->100
            xs.append(pad_or_take(y100))
            ys.append(1 if L == 'A' else 0)
            recs.append(r)
            cls.append(L)
    return (np.stack(xs), np.array(ys, np.int8), np.array(recs),
            np.zeros(len(xs), np.float32), np.array(cls),
            {'source': 'CinC2017 training2017 (300 Hz, mV)',
                            'labels': 'A=AF positive; N/O=sampled negatives; '
                                      '~ excluded',
                            'counts': dict(Counter(cls)),
                            'windows_per_record': 1,
                            'window': 'first 10 s after 300->100 Hz resample'})


def afpdb():
    root = DATA / 'afpdb'
    recs = [l.strip() for l in (root / 'RECORDS').read_text().split() if l.strip()]
    xs, ys, out_rec, starts, classes = [], [], [], [], []
    for r in recs:
        base = r.split('/')[-1]
        if base.startswith('t'):
            y_lab, n_win = -1, 20      # unlabelled test set
        elif base.endswith('c'):
            y_lab, n_win = -1, 5       # mixed continuation (contains PAF)
        else:
            y_lab, n_win = 0, 20       # n* or p* 30-min non-AF hard negative
        obj = wfdb.rdrecord(str(root / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 0]
        s100 = resample_poly(sig, 25, 32)  # 128 -> 100 Hz
        step = max((len(s100) - WIN) // n_win, 1)
        for i in range(n_win):
            a = min(i * step, max(len(s100) - WIN, 0))
            w = s100[a:a + WIN]
            if len(w) < WIN:
                continue
            xs.append(w.astype(np.float32))
            ys.append(y_lab)
            out_rec.append(base)
            starts.append(round(a / 100.0, 1))
            classes.append('t' if base.startswith('t')
                           else ('cont' if base.endswith('c') else base[0]))
    return (np.stack(xs), np.array(ys, np.int8), np.array(out_rec),
            np.array(starts), np.array(classes),
            {'source': 'afpdb (128 Hz, 2 ch, ch0 used)',
             'labels': 'n*/p* 30-min = y0 hard negatives (p*=pre-PAF, no AF '
                       'inside the record); nc*/pc* continuations = y-1 mixed '
                       '(contain the PAF episode); t* = y-1 unlabelled test',
             'counts': dict(Counter(classes)),
             'windows_per_record': '30-min: 20; 5-min continuation: 5'})


def aftdb():
    root = DATA / 'aftdb'
    recs = [l.strip() for l in (root / 'RECORDS').read_text().split() if l.strip()]
    xs, ys, out_rec, starts, classes = [], [], [], [], []
    for r in recs:
        obj = wfdb.rdrecord(str(root / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, 0]
        s100 = resample_poly(sig, 25, 32)
        cls = r.split('/')[0].split('-')[-1][0] + r.split('/')[-1][0]
        n_win = max((len(s100) - WIN) // WIN, 1)
        for i in range(n_win):
            a = i * WIN
            w = s100[a:a + WIN]
            if len(w) < WIN:
                continue
            xs.append(w.astype(np.float32))
            ys.append(1)               # all aftdb records are AF excerpts
            out_rec.append(r)
            starts.append(round(a / 100.0, 1))
            classes.append(cls)
    return (np.stack(xs), np.array(ys, np.int8), np.array(out_rec),
            np.array(starts), np.array(classes),
            {'source': 'aftdb (128 Hz, 2 ch, ch0 used)',
             'labels': 'ALL records are 1-min AF excerpts -> y=1; n/s/t = '
                       'termination class; a/b = challenge test sets',
             'counts': dict(Counter(classes)),
             'windows_per_record': '6 x 10 s non-overlap'})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'arm': 'r15_af_pool', 'family': '1000 pts @100 Hz raw mV (R13)'}
    for name, fn in [('cinc2017', cinc2017), ('afpdb', afpdb),
                     ('aftdb', aftdb)]:
        print('building', name, flush=True)
        x, y, rec, start_s, cls, meta = fn()
        np.savez_compressed(OUT / f'{name}.npz', x=x, y=y, rec=rec,
                            start_s=start_s, cls=cls)
        meta.update({'n_windows': int(len(x)), 'n_pos': int((y == 1).sum()),
                     'n_neg': int((y == 0).sum()),
                     'n_unlabelled': int((y == -1).sum())})
        res[name] = meta
        print(name, json.dumps(meta, ensure_ascii=False), flush=True)
    (OUT / 'result.json').write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print('saved', OUT, flush=True)


if __name__ == '__main__':
    main()
