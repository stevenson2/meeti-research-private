#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R15-B4: ltafdb window-level prep (third external AF domain).

Extends the r14 afdb prep to the MIT-BIH Long-Term AF Database
(ecg_data/ltafdb, 84 records x 21-25 h, 128 Hz, 2 channels, mV).

Task semantics: AFIB (positive) vs N (negative); VT / AFL / J / other
rhythm segments are EXCLUDED (disclosed).  10 s non-overlap windows are
grid-aligned in the native 128 Hz domain and, at eval time, resampled to
100 Hz to match the R13 family (signals stay on disk, metadata only).
Flat-zero / unread blocks are filtered with the same >=0.4 s rule as afdb.

Outputs (runs/r15_ltafdb_prep/):
  segments.npz  rec, start (128 Hz), label, n_beats_qrs
  result.json   label inventory, per-record stats, caveats, reconcile
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

DST = Path('/home/devcontainers/ecg_data/ltafdb')
OUT = Path('/home/devcontainers/meeti/runs/r15_ltafdb_prep')
FS = 128
WIN = 10 * FS
FLAT_S = 0.4
FLAT_RUN = int(FLAT_S * FS)
POS, NEG = 'AFIB', 'N'


def parse_segments(ann):
    segs, cur_lab, cur_start = [], None, None
    for s, aux in zip(ann.sample, ann.aux_note):
        a = (aux or '').strip()
        if a.startswith('('):
            if cur_lab is not None:
                segs.append((cur_start, int(s), cur_lab))
            cur_lab, cur_start = a[1:].upper(), int(s)
        elif a == ')' and cur_lab is not None:
            segs.append((cur_start, int(s), cur_lab))
            cur_lab, cur_start = None, None
    if cur_lab is not None:
        segs.append((cur_start, None, cur_lab))
    return segs


def flat_regions(sig):
    flat = (np.abs(sig) < 1e-8).any(axis=1)
    idx = np.where(flat)[0]
    out = []
    if len(idx):
        br = np.where(np.diff(idx) > 1)[0]
        starts = np.concatenate([[idx[0]], idx[br + 1]])
        ends = np.concatenate([idx[br], [idx[-1]]])
        for a, b in zip(starts, ends):
            if b - a + 1 >= FLAT_RUN:
                out.append([int(a), int(b)])
    return out


def overlaps(a, b, regs):
    return any(a < fb and fa < b for fa, fb in regs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--reconcile', type=int, default=3)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    records = sorted({p.stem for p in DST.glob('*.hea')})
    print('records', len(records), flush=True)

    win_rec, win_start, win_lab, win_beats = [], [], [], []
    per_rec, label_inv, seg_inv = {}, Counter(), Counter()
    for rec in records:
        ann = wfdb.rdann(str(DST / rec), 'atr')
        segs = parse_segments(ann)
        for _, _, l in segs:
            label_inv[l] += 1
        obj = wfdb.rdrecord(str(DST / rec))
        sig = np.asarray(obj.p_signal, np.float32)
        n_samp = sig.shape[0]
        regs = flat_regions(sig)
        qrs = wfdb.rdann(str(DST / rec), 'qrs').sample
        n_af = n_n = 0
        for (a, b, lab) in segs:
            seg_inv[lab if lab in (POS, NEG) else 'other'] += 1
            if lab not in (POS, NEG):
                continue
            b_eff = n_samp if b is None else min(b, n_samp)
            w0 = a - a % WIN
            while w0 + WIN <= b_eff:
                wa, wb = w0, w0 + WIN
                if not overlaps(wa, wb, regs):
                    nb = int(((qrs >= wa) & (qrs < wb)).sum())
                    win_rec.append(rec)
                    win_start.append(wa)
                    win_lab.append(1 if lab == POS else 0)
                    win_beats.append(nb)
                    if lab == POS:
                        n_af += 1
                    else:
                        n_n += 1
                w0 += WIN
        per_rec[rec] = {
            'duration_h': round(n_samp / FS / 3600, 2),
            'n_samples': int(n_samp),
            'rhythm_labels': sorted({l for _, _, l in segs}),
            'n_segments': len(segs),
            'n_win_afib': n_af, 'n_win_n': n_n,
            'flat_regions': len(regs),
            'af_burden_pct': round(100 * n_af / max(n_af + n_n, 1), 1),
            'p_min': [round(float(sig[:, c].min()), 3) for c in range(sig.shape[1])],
            'p_max': [round(float(sig[:, c].max()), 3) for c in range(sig.shape[1])],
        }
        print(f'{rec}: {per_rec[rec]["duration_h"]}h AF {n_af} / N {n_n} '
              f'flat {len(regs)} labels {per_rec[rec]["rhythm_labels"]}',
              flush=True)
        del sig

    np.savez_compressed(
        OUT / 'segments.npz', rec=np.array(win_rec),
        start=np.array(win_start, np.int64),
        label=np.array(win_lab, np.int8),
        n_beats_qrs=np.array(win_beats, np.int16))
    lab = np.array(win_lab)
    result = {
        'arm': 'r15_ltafdb_prep',
        'source': 'MIT-BIH Long-Term AF Database (ltafdb), '
                  '/home/devcontainers/ecg_data/ltafdb (not in git)',
        'task_semantics': 'AFIB (positive) vs N (negative); VT/AFL/J/other '
                          'excluded; 10 s non-overlap windows on the native '
                          '128 Hz grid; eval-time resample 128->100 Hz',
        'fs_native': FS,
        'n_records': len(records),
        'n_windows_afib': int(lab.sum()),
        'n_windows_n': int((lab == 0).sum()),
        'rhythm_label_inventory': dict(label_inv),
        'segment_inventory': dict(seg_inv),
        'flat_rule_s': FLAT_S,
        'per_record': per_rec,
        'caveats': [
            'ltafdb is AF-rich: N windows are scarce -> spec estimates are '
            'weak; the domain is primarily an AF-recall external check',
            'VT segments are frequent and excluded',
            'signals are read with wfdb at eval time; not cached',
            'channel default = ECG2 (index 1), matching the afdb prep; '
            'ECG1 sensitivity possible later',
        ],
    }
    if args.reconcile:
        recs = records[:args.reconcile]
        mism = 0
        for rec in recs:
            ann = wfdb.rdann(str(DST / rec), 'atr')
            segs = parse_segments(ann)
            obj = wfdb.rdrecord(str(DST / rec))
            sig = np.asarray(obj.p_signal, np.float32)
            regs = flat_regions(sig)
            exp = []
            for (a, b, lab2) in segs:
                if lab2 not in (POS, NEG):
                    continue
                b_eff = sig.shape[0] if b is None else min(b, sig.shape[0])
                w0 = a - a % WIN
                while w0 + WIN <= b_eff:
                    if not overlaps(w0, w0 + WIN, regs):
                        exp.append((w0, 1 if lab2 == POS else 0))
                    w0 += WIN
            got = [(int(s), int(l)) for s, l, r in
                   zip(win_start, win_lab, win_rec) if r == rec]
            if sorted(exp) != sorted(got):
                mism += 1
            print(f'reconcile {rec}: npz {len(got)} vs reread {len(exp)} '
                  f'{"OK" if sorted(exp) == sorted(got) else "MISMATCH"}',
                  flush=True)
            del sig
        result['reconcile'] = {'records': recs, 'n_mismatch': mism,
                               'ok': mism == 0}
    (OUT / 'result.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print('SUMMARY', json.dumps({k: result[k] for k in
          ['n_records', 'n_windows_afib', 'n_windows_n',
           'rhythm_label_inventory', 'segment_inventory']},
          ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
