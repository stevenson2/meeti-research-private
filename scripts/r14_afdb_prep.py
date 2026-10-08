#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A7: MIT-BIH Atrial Fibrillation Database (afdb) -> Lane-A eval domain.

User-approved download from https://archive.physionet.org/physiobank/database/afdb/
(raw files in /home/devcontainers/ecg_data/afdb/, NOT in git).

Known caveats (from the database documentation / user briefing, all handled):
  - 25 records, each ~10 h, 2 leads (ECG1/ECG2), 250 Hz, format 212, 12-bit,
    bandwidth 0.1-40 Hz.
  - 00735 / 03665 have NO signal files (rhythm/beat annotations only).
  - Manual .atr rhythm labels: AFIB, AFL, J, N (N = "all other rhythms").
    Binary eval task = AFIB (positive) vs N (negative); AFL / J / unknown
    segments are EXCLUDED (borderline rhythms, disclosed).
  - .qrs beat annotations are unreviewed (used only as per-window beat-count
    sanity); .qrsc (reviewed) exists for 05091 and 07859.
  - Unreadable tape blocks were replaced by flat zero segments
    (04043 blk39; 08405 blk1067; 08434 blk648/857/894). Windows overlapping a
    flat-zero run (>=100 exact-zero samples on either channel) are dropped.

Output (runs/r14_afdb_prep/):
  segments.npz - per-10s-window metadata (no signal cache; eval scripts read
                 .dat via wfdb at eval time): rec, win_start_sample (250 Hz
                 domain), label (1=AFIB, 0=N), n_beats_qrs
  result.json  - per-record stats + rhythm inventory + caveats
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np
import wfdb

DST = Path('/home/devcontainers/ecg_data/afdb')
OUT = Path('/home/devcontainers/meeti/runs/r14_afdb_prep')
WIN_S = 10
FS = 250
WIN = WIN_S * FS                     # 2500 samples
FLAT_RUN = 100                       # >=0.4 s of exact zeros = replaced block
POS, NEG = 'AFIB', 'N'

records = [r.strip() for r in (DST / 'RECORDS').read_text().split() if r.strip()]

win_rec, win_start, win_lab, win_beats = [], [], [], []
per_rec = {}
label_inventory = Counter()
seg_counts = Counter()

for rec in records:
    ann = wfdb.rdann(str(DST / rec), 'atr')
    # ---- rhythm segmentation from .atr ----
    segs = []                            # (start, end, label)
    cur_lab, cur_start = None, None
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
    for lab in {l for _, _, l in segs}:
        label_inventory[lab] += 1

    has_sig = (DST / f'{rec}.dat').exists()
    info = {'has_signal': has_sig, 'rhythm_labels': sorted({l for _, _, l in segs}),
            'n_segments': len(segs)}

    # ---- .qrs beat sanity (unreviewed) ----
    qrs = wfdb.rdann(str(DST / rec), 'qrs')
    qrs_samples = qrs.sample
    info['n_qrs'] = int(len(qrs_samples))

    if not has_sig:
        info['note'] = 'no signal file (00735/03665); annotations counted only'
        seg_counts['no_signal'] += 1
        per_rec[rec] = info
        print(f'{rec}: no signal; labels {info["rhythm_labels"]}', flush=True)
        continue

    rec_obj = wfdb.rdrecord(str(DST / rec))
    sig = np.asarray(rec_obj.p_signal, np.float32)   # (N, 2) mV
    n_samp = sig.shape[0]
    info['sig_name'] = list(rec_obj.sig_name)
    info['n_samples'] = int(n_samp)
    info['duration_h'] = round(n_samp / FS / 3600, 2)
    info['p_min'] = [round(float(sig[:, c].min()), 3) for c in range(sig.shape[1])]
    info['p_max'] = [round(float(sig[:, c].max()), 3) for c in range(sig.shape[1])]

    # ---- flat-zero regions (replaced unreadable tape blocks) ----
    flat = (np.abs(sig) < 1e-8).any(axis=1)
    flat_idx = np.where(flat)[0]
    flat_regions = []
    if len(flat_idx):
        br = np.where(np.diff(flat_idx) > 1)[0]
        starts = np.concatenate([[flat_idx[0]], flat_idx[br + 1]])
        ends = np.concatenate([flat_idx[br], [flat_idx[-1]]])
        for a, b in zip(starts, ends):
            if b - a + 1 >= FLAT_RUN:
                flat_regions.append([int(a), int(b)])
    info['flat_regions'] = flat_regions
    info['flat_regions_n'] = len(flat_regions)

    def is_flat(a, b):
        for fa, fb in flat_regions:
            if a < fb and fa < b:
                return True
        return False

    # ---- windows inside pure AFIB / N segments ----
    n_afib = n_norm = 0
    for (a, b, lab) in segs:
        seg_counts[lab if lab in (POS, NEG) else 'other'] += 1
        if lab not in (POS, NEG):
            continue
        b_eff = n_samp if b is None else min(b, n_samp)
        w0 = a - a % WIN                       # align to non-overlap grid
        while w0 + WIN <= b_eff:
            wa, wb = w0, w0 + WIN
            if not is_flat(wa, wb):
                nb = int(((qrs_samples >= wa) & (qrs_samples < wb)).sum())
                win_rec.append(rec)
                win_start.append(wa)
                win_lab.append(1 if lab == POS else 0)
                win_beats.append(nb)
                if lab == POS:
                    n_afib += 1
                else:
                    n_norm += 1
            w0 += WIN
    info['n_win_afib'] = n_afib
    info['n_win_n'] = n_norm
    af_time = n_afib * WIN_S
    tot_time = (n_afib + n_norm) * WIN_S
    info['af_burden_pct'] = round(100 * af_time / max(tot_time, 1), 1)
    per_rec[rec] = info
    print(f'{rec}: {info["duration_h"]}h AF {n_afib} win / N {n_norm} win | '
          f'flat {len(flat_regions)} | labels {info["rhythm_labels"]}',
          flush=True)
    del sig

OUT.mkdir(parents=True, exist_ok=True)
np.savez_compressed(
    OUT / 'segments.npz',
    rec=np.array(win_rec), start=np.array(win_start, np.int64),
    label=np.array(win_lab, np.int8), n_beats_qrs=np.array(win_beats, np.int16))

lab = np.array(win_lab)
result = {
    'arm': 'r14_afdb_prep',
    'source': 'https://archive.physionet.org/physiobank/database/afdb/ '
              '(user-approved 2026-09-08); raw in /home/devcontainers/ecg_data/afdb',
    'task_semantics': 'AFIB (positive) vs N="all other rhythms" (negative); '
                      'AFL/J/unknown excluded (borderline, disclosed); '
                      'manual .atr rhythm labels; 10 s non-overlap windows '
                      '@250 Hz; flat-zero replaced blocks dropped',
    'channel_note': 'signals kept on disk (wfdb-readable at eval time); '
                    'channels ECG1/ECG2, 212 format; declared eval default '
                    'lead = ECG2 (index 1) - assumption: Holter ch2 closer to '
                    'lead II; sensitivity over ECG1 possible later',
    'n_records_total': len(records),
    'n_records_with_signal': sum(1 for v in per_rec.values() if v['has_signal']),
    'n_windows_afib': int(lab.sum()),
    'n_windows_n': int((lab == 0).sum()),
    'rhythm_label_inventory': dict(label_inventory),
    'segment_inventory': dict(seg_counts),
    'per_record': per_rec,
}
(OUT / 'result.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
print('SUMMARY:', json.dumps({k: result[k] for k in
      ['n_records_with_signal', 'n_windows_afib', 'n_windows_n',
       'rhythm_label_inventory']}, ensure_ascii=False), flush=True)
print('saved ->', OUT, flush=True)
