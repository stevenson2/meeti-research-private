#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16 data layer: RR sequences + extended RR features for all domains.

Domains
-------
ptb      PTB-XL folds1-9, lead II 100 Hz 10 s (R14-A4 detector)
afdb     afdb fit/eval records, ECG2 250 Hz -> 100 Hz, 10 s + 30 s windows
ltafdb   ltafdb sampled windows, ECG2 128 Hz -> 100 Hz, 10 s + 30 s
device   ecg_real_052 / rec_latest deploy-chain XQRS @250 Hz, 10 s + 30 s

Window convention: every sampled window is indexed by the start s of its
last 10 s sub-window.  The 10 s window is [s, s+10); the paired 30 s
window is [s-20, s+10), i.e. it ENDS at the same instant (causal on
device: beat-position history only).  Candidate 30 s windows require the
three consecutive 10 s grid windows to share the same rhythm label, so
the paired set is a label-pure subset (disclosed: short N segments drop
out, mostly in ltafdb).

All signals stay on disk; only RR sequences/features are cached under
runs/r16_rr_shape/.
"""
import json
import struct
import sys
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import resample_poly

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r14_rr_baseline import detect_peaks, detect_rr  # noqa: E402
from r16_rr_features import (ALL_FEATURES, features_from_rr_ext)  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r16_rr_shape'
AFDB = Path('/home/devcontainers/ecg_data/afdb')
LTAFDB = Path('/home/devcontainers/ecg_data/ltafdb')
AFDB_SEG = ROOT / 'runs/r14_afdb_prep/segments.npz'
LTA_SEG = ROOT / 'runs/r15_ltafdb_prep/segments.npz'
PTB_TRAIN = ROOT / 'runs/r14_af_prep/ptbxl_af_train.npz'
ELECTRON = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                'ecg-programme-tju-flex.electron-master')
DEVICE = {
    'ecg_real_052': ELECTRON / 'pc_tools/ecg_dl/data/real/ecg_real_052.ecgr',
    'rec_latest': ELECTRON / 'rec_latest.ecgr',
}
META_KEYS = ('n_rr', 'mean_rr', 'median_rr', 'hr')
PACK_KEYS = ALL_FEATURES + list(META_KEYS)
FS100 = 100
WIN10_S = 10.0
WIN30_S = 30.0


# --------------------------------------------------------------- packing
def pack_features(dicts, prefix):
    return {f'{prefix}_{k}': np.array([d.get(k, np.nan) for d in dicts], float)
            for k in PACK_KEYS}


def pack_rr(rrs, prefix):
    lens = np.array([len(r) for r in rrs], np.int64)
    off = np.concatenate([[0], np.cumsum(lens)])
    cat = (np.concatenate(rrs).astype(np.float32) if len(rrs)
           else np.zeros(0, np.float32))
    return {f'{prefix}_rr': cat, f'{prefix}_rr_off': off}


def unpack_rr(npz, prefix, i):
    off = npz[f'{prefix}_rr_off']
    return npz[f'{prefix}_rr'][off[i]:off[i + 1]].astype(np.float64)


# ---------------------------------------------------------------- device
def read_ecgr(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 18)[0]
    dur = struct.unpack_from('<I', raw, 14)[0]
    x = np.frombuffer(raw, dtype='<i2', count=n, offset=32).astype(np.float64)
    return x / 8000.0, n / dur, dur


def deploy_chain250(path):
    sys.path.insert(0, str(ELECTRON / 'pc_tools/ecg_dl'))
    from eval_deploy_match import causal_hp_05_fs250, deployment_chain
    x, fs, dur = read_ecgr(path)
    ch = causal_hp_05_fs250(deployment_chain(x, fs))
    return ch, fs, dur


def device_peak_times(name):
    from wfdb.processing import xqrs_detect
    ch250, fs, dur = deploy_chain250(DEVICE[name])
    r250 = xqrs_detect(ch250.astype(np.float64), fs=250, verbose=False)
    t_primary = r250 / 250.0
    ch100 = resample_poly(ch250, 2, 5)
    p100 = detect_peaks(ch100)
    t_sens = p100 / float(FS100)
    return {'deploy_xqrs250': t_primary, 'pt100': t_sens,
            'dur_s': float(dur), 'fs_raw': float(fs),
            'n_peaks_xqrs': int(len(r250)), 'n_peaks_pt100': int(len(p100))}


def rr_in_window(peak_times, a, b):
    m = (peak_times >= a) & (peak_times < b)
    pk = peak_times[m]
    rr = np.diff(pk)
    return rr[(rr >= 0.30) & (rr <= 2.00)]


def build_device(rebuild=False):
    cache = OUT / 'device_features.npz'
    if cache.exists() and not rebuild:
        return dict(np.load(cache))
    out, rr_pack = {}, {}
    for name in DEVICE:
        pk = device_peak_times(name)
        entry = {'dur_s': pk['dur_s'], 'fs_raw': pk['fs_raw'],
                 'n_peaks_xqrs': pk['n_peaks_xqrs'],
                 'n_peaks_pt100': pk['n_peaks_pt100']}
        for det in ('deploy_xqrs250', 'pt100'):
            t = pk[det]
            rr_whole = np.diff(t)
            rr_whole = rr_whole[(rr_whole >= 0.30) & (rr_whole <= 2.00)]
            entry[f'whole_{det}'] = features_from_rr_ext(rr_whole)
            rr_pack[f'{name}_{det}_whole'] = rr_whole
            # all 10 s windows (start s = window start)
            s10 = np.arange(0.0, max(pk['dur_s'] - WIN10_S, 0) + 1e-9, 1.0)
            d10 = [features_from_rr_ext(rr_in_window(t, s, s + WIN10_S))
                   for s in s10]
            # paired windows: 10 s [s,s+10) and 30 s [s-20,s+10)
            sp = np.arange(20.0, max(pk['dur_s'] - WIN10_S, 20.0) + 1e-9, 1.0)
            d10p = [features_from_rr_ext(rr_in_window(t, s, s + WIN10_S))
                    for s in sp]
            d30p = [features_from_rr_ext(rr_in_window(t, s - 20.0,
                                                      s + WIN10_S))
                    for s in sp]
            entry[f'{det}_n_10s_all'] = int(len(d10))
            entry[f'{det}_n_10s_paired'] = int(len(d10p))
            entry[f'{det}_n_30s_paired'] = int(len(d30p))
            out[f'{name}_{det}_10s_start'] = s10
            out[f'{name}_{det}_paired_start'] = sp
            out.update(pack_features(d10, f'{name}_{det}_10s'))
            out.update(pack_features(d10p, f'{name}_{det}_paired10'))
            out.update(pack_features(d30p, f'{name}_{det}_paired30'))
            rr_pack[f'{name}_{det}_10s'] = [
                rr_in_window(t, s, s + WIN10_S) for s in s10]
            rr_pack[f'{name}_{det}_paired10'] = [
                rr_in_window(t, s, s + WIN10_S) for s in sp]
            rr_pack[f'{name}_{det}_paired30'] = [
                rr_in_window(t, s - 20.0, s + WIN10_S) for s in sp]
        out[f'{name}_meta'] = np.array(json.dumps(entry))
        print(f'device {name}: peaks {pk["n_peaks_xqrs"]} dur {pk["dur_s"]}',
              flush=True)
    out.update(pack_rr(rr_pack[f'{list(DEVICE)[0]}_deploy_xqrs250_10s'],
                       'tmp'))
    # store rr sequences separately
    np.savez_compressed(OUT / 'device_rr.npz',
                        **{f'{k}_rr': np.concatenate(v).astype(np.float32)
                           for k, v in rr_pack.items()
                           if isinstance(v, list) and len(v)},
                        **{f'{k}_off': np.concatenate(
                            [[0], np.cumsum([len(x) for x in v])]).astype(np.int64)
                           for k, v in rr_pack.items()
                           if isinstance(v, list) and len(v)},
                        **{f'{k}': v for k, v in rr_pack.items()
                           if not isinstance(v, list)})
    # remove tmp keys from feature npz
    out = {k: v for k, v in out.items() if not k.startswith('tmp_')}
    np.savez_compressed(cache, **out)
    return out


# ------------------------------------------------------------------ PTB
def build_ptb(rebuild=False, limit=0):
    cache = OUT / 'ptb_features.npz'
    if cache.exists() and not rebuild:
        return dict(np.load(cache))
    d = np.load(PTB_TRAIN)
    x = d['x']
    y = d['y_af'].astype(int)
    folds = d['folds'].astype(int)
    ecg = d['ecg_ids']
    n = len(x) if not limit else min(limit, len(x))
    rows, rrs = [], []
    for i in range(n):
        rr, _ = detect_rr(x[i])
        rows.append(features_from_rr_ext(rr))
        rrs.append(rr)
        if (i + 1) % 2000 == 0:
            print(f'ptb {i+1}/{n}', flush=True)
    out = pack_features(rows, 'ptb')
    out.update(pack_rr(rrs, 'ptb'))
    out.update(ptb_y=y[:n], ptb_folds=folds[:n], ptb_ecg=ecg[:n])
    np.savez_compressed(cache, **out)
    return out


# -------------------------------------------------------------- external
def _seg_index(seg_path):
    seg = np.load(seg_path)
    rec, start, lab = seg['rec'], seg['start'], seg['label'].astype(int)
    by_rec = {}
    for r, s, l in zip(rec, start, lab):
        by_rec.setdefault(str(r), {})[int(s)] = int(l)
    return by_rec


def build_external(domain, per_class=600, seed=42, rebuild=False,
                   limit_records=0):
    """Return (features, labels, records, starts, split, rr pack keys)."""
    assert domain in ('afdb', 'ltafdb')
    cache = OUT / f'{domain}_features.npz'
    if cache.exists() and not rebuild:
        return dict(np.load(cache))
    if domain == 'afdb':
        dst, seg_path, fs, ch = AFDB, AFDB_SEG, 250, 1
        up, down = 2, 5
    else:
        dst, seg_path, fs, ch = LTAFDB, LTA_SEG, 128, 1
        up, down = 25, 32
    win = 10 * fs
    by_rec = _seg_index(seg_path)
    records = sorted(by_rec)
    if limit_records:
        records = records[:limit_records]
    rng = np.random.default_rng(seed)
    rows10, rows30, rr10, rr30 = [], [], [], []
    rec_out, y_out, start_out, split_out = [], [], [], []
    for ri, r in enumerate(records):
        obj = wfdb.rdrecord(str(dst / r))
        sig = np.asarray(obj.p_signal, np.float64)[:, ch]
        n_samp = len(sig)
        starts = np.array(sorted(by_rec[r]))
        labmap = by_rec[r]
        for L in (1, 0):
            cand = [s for s in starts
                    if labmap[s] == L
                    and s - 2 * win >= 0
                    and labmap.get(s - win) == L
                    and s + win <= n_samp]
            if not cand:
                continue
            take = rng.choice(np.array(cand),
                              min(per_class, len(cand)), replace=False)
            for s in np.sort(take):
                w10 = sig[s:s + win]
                w30 = sig[s - 2 * win:s + win]
                rr10_ = detect_rr(resample_poly(w10, up, down))[0]
                rr30_ = detect_rr(resample_poly(w30, up, down))[0]
                rows10.append(features_from_rr_ext(rr10_))
                rows30.append(features_from_rr_ext(rr30_))
                rr10.append(rr10_)
                rr30.append(rr30_)
                rec_out.append(r)
                y_out.append(L)
                start_out.append(int(s))
                split_out.append(0 if ri % 2 == 0 else 1)
        print(f'{domain} {r} ({ri+1}/{len(records)}) windows={len(y_out)}',
              flush=True)
    out = {}
    out.update(pack_features(rows10, f'{domain}10'))
    out.update(pack_features(rows30, f'{domain}30'))
    out.update(pack_rr(rr10, f'{domain}10'))
    out.update(pack_rr(rr30, f'{domain}30'))
    out[f'{domain}_rec'] = np.array(rec_out)
    out[f'{domain}_y'] = np.array(y_out, np.int8)
    out[f'{domain}_start'] = np.array(start_out, np.int64)
    out[f'{domain}_split'] = np.array(split_out, np.int8)
    np.savez_compressed(cache, **out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--domain', default='all')
    ap.add_argument('--per-class', type=int, default=600)
    ap.add_argument('--limit-ptb', type=int, default=0)
    ap.add_argument('--limit-records', type=int, default=0)
    ap.add_argument('--rebuild', action='store_true')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.domain in ('all', 'ptb'):
        d = build_ptb(a.rebuild, a.limit_ptb)
        print('ptb', {k: d[k].shape for k in ('ptb_y', 'ptb_rr')})
    if a.domain in ('all', 'afdb'):
        d = build_external('afdb', a.per_class, rebuild=a.rebuild,
                           limit_records=a.limit_records)
        print('afdb', len(d['afdb_y']), int(d['afdb_y'].sum()))
    if a.domain in ('all', 'ltafdb'):
        d = build_external('ltafdb', max(a.per_class // 10, 20),
                           rebuild=a.rebuild,
                           limit_records=a.limit_records)
        print('ltafdb', len(d['ltafdb_y']), int(d['ltafdb_y'].sum()))
    if a.domain in ('all', 'device'):
        d = build_device(a.rebuild)
        print('device keys', len(d))
