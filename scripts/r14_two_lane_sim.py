#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R14-A6: two-lane gate PC simulation (zero training).

Lane A = AF head (theta_tr) AND RR-entropy confirmation (A4 train-side
Youden).  Lane B = frozen M4 gate (documented, unchanged).

Pre-registered device acceptance:
  (1) both real normal recordings are silent at theta_tr (AF head FP=0);
  (2) fold-10 AF recall @theta_tr >= 0.90.

Device reconstructions (all disclosed; the AF head expects raw-mV lead II
10 s @100 Hz):
  ecg_real_052_chain : raw .ecgr AFE stream -> firmware deployment chain
                       (deployment_chain + causal HP0.5@250) -> 100 Hz.
                       PRIMARY for ecg_real_052 (most faithful).
  ecg_real_052_raw   : raw stream -> 100 Hz (no firmware chain). Sensitivity.
  rec_latest_ola     : 101 beat windows (1 s @250 Hz, per-beat z-scored)
                       overlap-added on the true average-RR timeline
                       (64 s / 101 beats). PRIMARY for rec_latest.
  rec_latest_resamp  : each beat window resampled to average RR and stitched.
                       Sensitivity.

All signals are amplitude-scaled to the PTB-XL train lead-II p99 (plus
std / z-score / polarity variants as sensitivity).  MIT+INCART has no AF
sub-label (beat-level binary), so Lane A is not evaluated there.
"""
import argparse
import json
import struct
import sys
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import resample, resample_poly

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r14_rr_baseline import detect_rr, features_from_rr  # noqa: E402

ROOT = Path('/home/devcontainers/meeti')
AF_ROOT = ROOT / 'runs/r14_af_head'
OUT = ROOT / 'runs/r14_two_lane_sim'
WIN = 1000
FS = 100
ELECTRON = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
                'ecg-programme-tju-flex.electron-master')
REAL_DIR = ELECTRON / 'pc_tools/ecg_dl/data/real'
PRIMARY = {'ecg_real_052': 'ecg_real_052_chain',
           'rec_latest': 'rec_latest_ola'}


def load_af_model(seed):
    import os
    os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
    import tensorflow as tf
    from models_patch.ptbxl_builders import build_ptbxl_large
    m = build_ptbxl_large(input_shape=(1000, 1), n_classes=1,
                          output_activation='sigmoid', name='r14_af')
    m.load_weights(str(AF_ROOT / f'af_seed{seed}' / 'best.weights.h5'))
    return m


def read_ecgr(path):
    raw = path.read_bytes()
    n = struct.unpack_from('<I', raw, 18)[0]
    dur = struct.unpack_from('<I', raw, 14)[0]
    x = np.frombuffer(raw, dtype='<i2', count=n, offset=32).astype(np.float64)
    return x / 8000.0, n / dur, dur


def to_100hz(y, fs):
    if abs(fs - FS) < 1e-6:
        return y
    frac = Fraction(FS / fs).limit_denominator(100000)
    return resample_poly(y, frac.numerator, frac.denominator)


def build_ecg_real_052():
    y, fs, dur = read_ecgr(REAL_DIR / 'ecg_real_052.ecgr')
    out = {}
    out['ecg_real_052_raw'] = (
        to_100hz(y, fs),
        {'source': 'ecg_real_052.ecgr', 'fs_raw': fs, 'dur_s': dur,
         'reconstruction': 'raw AFE stream -> 100 Hz (no firmware chain)',
         'role': 'sensitivity'})
    try:
        sys.path.insert(0, str(ELECTRON / 'pc_tools/ecg_dl'))
        from eval_deploy_match import (causal_hp_05_fs250,
                                       deployment_chain)
        ch250 = causal_hp_05_fs250(deployment_chain(y, fs))
        out['ecg_real_052_chain'] = (
            resample_poly(ch250, 2, 5),
            {'source': 'ecg_real_052.ecgr', 'fs_raw': fs, 'dur_s': dur,
             'reconstruction': 'raw AFE -> firmware deployment chain '
                               '(comb+HP/LP+causal HP0.5@250) -> 100 Hz',
             'role': 'primary'})
    except Exception as e:  # pragma: no cover
        out['ecg_real_052_chain'] = (
            out['ecg_real_052_raw'][0],
            {'source': 'ecg_real_052.ecgr',
             'reconstruction': f'deploy chain unavailable ({e}); raw used',
             'role': 'primary-fallback'})
    return out


def build_rec_latest():
    beats = np.load(REAL_DIR / 'real_normal_beats_rec_latest.npy').astype(
        np.float64)
    dur = 64.0
    rr_s = dur / len(beats)
    n_rs = max(int(round(rr_s * 250)), 2)
    out = {}
    stitched = np.concatenate([resample(b, n_rs) for b in beats])
    out['rec_latest_resamp'] = (
        to_100hz(stitched, 250.0),
        {'source': 'real_normal_beats_rec_latest.npy',
         'n_beats': int(len(beats)), 'dur_s': dur,
         'reconstruction': 'per-beat 1 s @250 Hz resampled to avg RR and '
                           'stitched -> 100 Hz (per-beat z-scored)',
         'role': 'sensitivity'})
    length = int(len(beats) * rr_s * 250) + 250
    acc = np.zeros(length)
    wsum = np.zeros(length)
    hann = np.hanning(250)
    for i, b in enumerate(beats):
        s = int(round(i * rr_s * 250))
        acc[s:s + 250] += b * hann
        wsum[s:s + 250] += hann
    ola = np.where(wsum > 1e-9, acc / np.maximum(wsum, 1e-9), 0.0)
    out['rec_latest_ola'] = (
        to_100hz(ola, 250.0),
        {'source': 'real_normal_beats_rec_latest.npy',
         'n_beats': int(len(beats)), 'dur_s': dur,
         'reconstruction': 'per-beat 1 s @250 Hz overlap-added on the true '
                           'average-RR timeline -> 100 Hz (Hann weights)',
         'role': 'primary'})
    return out


def ptb_refs():
    tr = np.load(ROOT / 'runs/r14_af_prep/ptbxl_af_train.npz')
    x = tr['x']
    return (float(np.median(np.std(x, axis=1))),
            float(np.median(np.percentile(np.abs(x), 99, axis=1))))


def variants(y, target_std, target_p99):
    p99 = float(np.percentile(np.abs(y), 99))
    std = float(np.std(y))
    return {
        'match_p99': y * (target_p99 / max(p99, 1e-9)),
        'match_std': y * (target_std / max(std, 1e-9)),
        'zscore_to_ptb_std': (y - y.mean()) / max(std, 1e-9) * target_std,
    }


def windows(y, stride_s):
    step = max(int(stride_s * FS), 1)
    if len(y) < WIN:
        y = np.pad(y, (0, WIN - len(y)))
    return np.stack([y[i:i + WIN] for i in range(0, len(y) - WIN + 1, step)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, nargs='+', default=[42])
    ap.add_argument('--stride-s', type=float, default=1.0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    target_std, target_p99 = ptb_refs()
    signals = {}
    signals.update(build_ecg_real_052())
    signals.update(build_rec_latest())

    rr_res = json.loads((ROOT / 'runs/r14_rr_baseline/result.json').read_text())
    theta_rr = rr_res['rr_entropy_confirmator']['theta_tr_youden']
    results = {'arm': 'r14_two_lane_sim',
               'ptb_train_ref': {'std_median': target_std,
                                 'p99_median': target_p99},
               'theta_rr_from_A4': theta_rr,
               'primary_signals': PRIMARY,
               'seeds': args.seeds, 'stride_s': args.stride_s,
               'signals': {}, 'per_seed': {}}
    for name, (y, meta) in signals.items():
        results['signals'][name] = dict(
            meta, n_samples=int(len(y)),
            dur_100hz_s=round(len(y) / FS, 2),
            raw_std=float(np.std(y)),
            raw_p99abs=float(np.percentile(np.abs(y), 99)))

    for seed in args.seeds:
        rj = json.loads((AF_ROOT / f'af_seed{seed}' / 'result.json').read_text())
        theta = rj['metrics']['theta_tr_val_youden']
        rec = rj['metrics']['R_at_theta_tr']
        model = load_af_model(seed)
        per_rec = {}
        for name, (y, meta) in signals.items():
            vs = variants(y, target_std, target_p99)
            Wp = windows(vs['match_p99'], args.stride_s)
            ent = np.array([features_from_rr(detect_rr(w)[0])['rr_entropy']
                            for w in Wp], np.float64)
            var_out = {}
            for vname, vy in vs.items():
                for pol, sy in [('+', vy), ('-', -vy)]:
                    W = windows(sy, args.stride_s)
                    p = model.predict(W[..., None], batch_size=256,
                                      verbose=0)[:, 0]
                    key = f'{vname}_{pol}'
                    var_out[key] = {
                        'n_windows': int(len(W)),
                        'max_prob': round(float(p.max()), 4),
                        'mean_prob': round(float(p.mean()), 4),
                        'n_ge_theta_tr': int((p >= theta).sum()),
                    }
                    if vname == 'match_p99' and pol == '+':
                        alarm = (p >= theta) & np.isfinite(ent) & (ent >= theta_rr)
                        var_out[key]['n_combined_laneA_alarms'] = int(alarm.sum())
            var_out['rr_entropy_match_p99'] = {
                'n_valid': int(np.isfinite(ent).sum()),
                'median': round(float(np.nanmedian(ent)), 4)
                if np.isfinite(ent).any() else None,
                'max': round(float(np.nanmax(ent)), 4)
                if np.isfinite(ent).any() else None,
                'n_ge_theta_rr': int((np.nan_to_num(ent, nan=-1)
                                      >= theta_rr).sum()),
            }
            var_out['theta_tr'] = theta
            per_rec[name] = var_out
        primary_names = [PRIMARY['ecg_real_052'], PRIMARY['rec_latest']]
        primary_silent = all(
            per_rec[n]['match_p99_+']['n_ge_theta_tr'] == 0
            for n in primary_names)
        primary_combined = all(
            per_rec[n]['match_p99_+'].get('n_combined_laneA_alarms', 0) == 0
            for n in primary_names)
        worst_ge = max(
            per_rec[n][k]['n_ge_theta_tr']
            for n in per_rec for k in per_rec[n]
            if isinstance(per_rec[n][k], dict) and 'n_ge_theta_tr' in per_rec[n][k])
        worst_combined = max(
            per_rec[n][k].get('n_combined_laneA_alarms', 0)
            for n in per_rec for k in per_rec[n]
            if isinstance(per_rec[n][k], dict))
        results['per_seed'][str(seed)] = {
            'theta_tr': theta,
            'fold10_R_at_theta_tr': rec,
            'fold10_recall_ge_090': bool(rec >= 0.90),
            'device_primary_afhead_silent_FP0': bool(primary_silent),
            'device_primary_combined_gate_FP0': bool(primary_combined),
            'device_worst_case_windows_ge_theta_all_variants': int(worst_ge),
            'device_worst_case_combined_alarms_all_variants': int(worst_combined),
            'recordings': per_rec,
        }
    verdict_afhead = all(
        results['per_seed'][str(s)]['device_primary_afhead_silent_FP0']
        and results['per_seed'][str(s)]['fold10_recall_ge_090']
        for s in args.seeds)
    verdict_combined = all(
        results['per_seed'][str(s)]['device_primary_combined_gate_FP0']
        and results['per_seed'][str(s)]['fold10_recall_ge_090']
        for s in args.seeds)
    results['acceptance'] = {
        'criterion_1_device_afhead_silent': 'AF head FP=0 on both primary '
                                            'device reconstructions',
        'criterion_2_fold10_recall': 'fold-10 AF recall @theta_tr >= 0.90',
        'afhead_silent_pass': bool(verdict_afhead),
        'combined_gate_fp0_observation': bool(verdict_combined),
        'note': 'the pre-registered criterion is the AF-head-silent one; the '
                'combined-gate FP=0 is reported as a post-hoc observation',
        'mit_incart_disclosure': 'MIT+INCART has no AF sub-label '
                                 '(beat-level binary); Lane A not evaluated there',
    }
    (OUT / 'result.json').write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({k: results[k] for k in
                      ['ptb_train_ref', 'theta_rr_from_A4', 'per_seed',
                       'acceptance']}, indent=2, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
