#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A1 optional: MEETI report-field AF scan (weak-label auxiliary domain).

The MEETI metadata tree is a metadata-only copy of MIMIC-IV-ECG derived
reports (~897 .mat files per p-group, ~1000 groups => ~0.9M files).  A full
scipy.io.loadmat sweep is infeasible inside the 1 h timebox (measured
throughput of the R9-era scan: 11,093 files / 266 s ~= 42 files/s => ~6 h).
This script therefore:

  1. summarises the existing local scan data/selected8k/scanned_manifest.csv
     (p1000..p1014 convenience sample, 11,093 records, scanned in R9);
  2. optionally extends the scan over more p-groups within a time budget
     (--extend-groups / --timebox-s), writing scan_extra.csv incrementally;
  3. applies the pre-registered strict regex 'atrial fib|afib' (case
     insensitive) plus a broad variant and an explicit-negation check.

Decision rule from the round prompt: if the AF count is < 50, drop the
MEETI AF arm for sample-size reasons (disclosed).  The count here is a
convenience-sample estimate, not a full-domain count.
"""
import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = Path('/home/devcontainers/meeti')
SCANNED = ROOT / 'data/selected8k/scanned_manifest.csv'
OUT = ROOT / 'runs/r14_af_prep'
ZENODO = Path('/mnt/d/MEETI_10k/extracted/20260209_MEETI')
STRICT = re.compile(r'atrial fib|afib', re.I)
BROAD = re.compile(r'atrial fibrillation|atrial flutter|afib|a[ -]?fib', re.I)
NEG = re.compile(r'(no|without|not in|denies|absence of|free of)\s+'
                 r'(atrial fib|afib)', re.I)


def counts(df, tag):
    r = df['report'].fillna('').astype(str)
    m_strict = r.str.contains(STRICT)
    m_broad = r.str.contains(BROAD)
    return {
        'tag': tag,
        'n_records': int(len(df)),
        'n_subjects': int(df['subject_id'].nunique()) if 'subject_id' in df else None,
        'af_strict_n': int(m_strict.sum()),
        'af_strict_pct': round(100 * float(m_strict.mean()), 3),
        'af_broad_n': int(m_broad.sum()),
        'af_broad_pct': round(100 * float(m_broad.mean()), 3),
        'explicit_af_negation_n': int((m_strict & r.str.contains(NEG)).sum()),
        'n_groups': int(df['subject_id'].str[:5].nunique())
        if 'subject_id' in df else None,
    }


def scan_extra(groups, timebox_s):
    t0 = time.time()
    rows, n, err = [], 0, 0
    out_csv = OUT / 'scan_extra.csv'
    for g in groups:
        gdir = ZENODO / g
        if not gdir.exists():
            print('skip missing', g, flush=True)
            continue
        for mat in sorted(gdir.rglob('*.mat')):
            if time.time() - t0 > timebox_s:
                print('timebox reached', flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
                return rows, time.time() - t0
            n += 1
            try:
                d = loadmat(str(mat), squeeze_me=False)
                report = str(np.asarray(d.get('report', [''])).ravel()[0])[:300]
                rid = (str(np.asarray(d['id']).ravel()[0])
                       if 'id' in d else mat.stem)
                rel = mat.relative_to(ZENODO).as_posix().replace('.mat', '')
                rows.append({'record_path': 'files/' + rel, 'stem': 'files/' + rel,
                             'study_id': rid, 'subject_id': rel.split('/')[1],
                             'report': report, 'report_class': '',
                             'signal_shape': '', 'n_featuredb': 0})
            except Exception:
                err += 1
            if n % 2000 == 0:
                print(f'extra scanned={n} err={err} '
                      f'elapsed={time.time()-t0:.0f}s', flush=True)
                pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f'group {g} done n={n} elapsed={time.time()-t0:.0f}s', flush=True)
        pd.DataFrame(rows).to_csv(out_csv, index=False)
    return rows, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--extend-groups', default='')
    ap.add_argument('--timebox-s', type=int, default=1800)
    args = ap.parse_args()

    base = pd.read_csv(SCANNED)
    parts = [counts(base, 'existing_scanned_manifest(p1000-p1014)')]
    extra_df = None
    if args.extend_groups:
        a, b = args.extend_groups.split(':')
        groups = [f'p{i}' for i in range(int(a[1:]), int(b[1:]) + 1)]
        rows, el = scan_extra(groups, args.timebox_s)
        if rows:
            extra_df = pd.DataFrame(rows)
            parts.append(counts(extra_df, f'extra_scan({args.extend_groups}, '
                                          f'{el}s)'))
            union = pd.concat([base, extra_df]).drop_duplicates('stem')
            parts.append(counts(union, 'union(existing+extra)'))
            extra_df.to_csv(OUT / 'scan_extra.csv', index=False)
    result = {
        'arm': 'r14_meeti_af_scan',
        'source': 'MEETI metadata .mat report fields (MIMIC-IV-ECG derived)',
        'strict_regex': 'atrial fib|afib (case insensitive)',
        'full_scan_feasibility': {
            'measured_mat_per_pgroup': 897,
            'n_pgroups': 1000,
            'estimated_total_mat': 897000,
            'measured_throughput_files_per_s': 41.7,
            'estimated_full_scan_h': round(897000 / 41.7 / 3600, 1),
            'timebox_h': 1.0,
            'verdict': 'full sweep infeasible inside 1 h box; convenience '
                       'sample used with disclosure',
        },
        'counts': parts,
        'meeti_af_arm_decision': (
            'AF count >= 50 => MEETI AF arm not dropped on sample-size '
            'grounds (still a convenience-sample count; no MEETI training '
            'was run in R14)'),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'meeti_af_scan.json').write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
