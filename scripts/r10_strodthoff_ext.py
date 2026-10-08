#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10 extension (user challenge Q1/Q2):
(a) lenet row (xai4ecg builder reconstructible from ckpt skeleton);
(b) amplitude sensitivity for the Strodthoff rows - decompose the collapse.

Rationale (Q2): the Strodthoff models were trained on PTB-XL RAW mV, while
our unified sec-11.2 adaptation feeds z-scored unit-scale windows (~8x
amplitude gap, the same mismatch R9 diagnosed for our own models).  To
attribute the collapse honestly we:
  1. derive scale s = median per-record global std of lead II on PTB-XL
     TRAIN folds 1-8 (their training data only - no eval-domain selection);
  2. sweep {s/4, s/2, s, 2s, 1.0} on PTB val ONLY (diagnostic);
  3. re-evaluate all benchmark domains at the pre-registered s and store
     as 'amplitude_matched_variant' inside each row JSON (disclosed, NOT
     replacing the unified-caliber primary numbers).
"""
import json
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wfdb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r10_lit_strodthoff import (fake_lightning, prep_windows,  # noqa: E402
                                windows_from_beats, meeti_windows3,
                                device_windows3, top2mean_by_record,
                                record_mean, BATCH, NORM_IDX, CKPTS)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/benchmark_v1'
RUNDIR = ROOT / 'runs/r10_strodthoff'
from project_paths import P as _PP  # unified library
PTBXL_ROOT = Path(_PP['ptbxl_copy'])
LENET_SOURCE = ('Strodthoff-lab XAI4ECG audit suite LeNet baseline '
                '(same release as resnet/xresnet rows); see lit rows for '
                'citation')


class LeNetInner(nn.Module):
    """ckpt skeleton: sequential 0Conv 1BN 2ReLU 3Pool 4Drop | 5-9 | 10Conv
    11BN 12ReLU 13AvgPool 14Flatten (params only at 0,1,5,6,10,11)."""

    def __init__(self):
        super().__init__()
        self.sequential = nn.Sequential(
            nn.Conv1d(12, 32, 5, 2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(32, 64, 5, 2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Dropout(0.1),
            nn.Conv1d(64, 128, 5, 2), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten())

    def forward(self, x):
        return self.sequential(x)


def build_lenet():
    inner = LeNetInner()
    head = nn.Sequential(nn.Dropout(0.5),
                         nn.Sequential(nn.Flatten(), nn.Linear(128, 5)))
    model = nn.Sequential(inner, head)
    ck = torch.load(CKPTS / 'lenet_superdiagnostic/checkpoints/'
                    'best_model.ckpt', map_location='cpu', weights_only=False)
    sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


def train_fold_scale(n_sample=400, seed=0):
    """s = median per-record global std of lead II on PTB-XL folds 1-8."""
    df = pd.read_csv(PTBXL_ROOT / 'ptbxl_database.csv', index_col='ecg_id')
    tr = df[df.strat_fold.isin(range(1, 9))]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(tr), size=min(n_sample, len(tr)), replace=False)
    stds = []
    for i in idx:
        row = tr.iloc[i]
        sig, _ = wfdb.rdsamp(str(PTBXL_ROOT / row.filename_lr))
        stds.append(float(np.std(sig[:, 1])))        # lead II
    s = float(np.median(stds))
    print(f'PTB-XL train folds lead-II std: median={s:.4f} '
          f'(n={len(stds)}, p25={np.percentile(stds,25):.4f}, '
          f'p75={np.percentile(stds,75):.4f})', flush=True)
    return s


@torch.no_grad()
def scores(model, x750, scale=1.0):
    ps = []
    for b in range(0, len(x750), BATCH):
        xb = prep_windows(x750[b:b + BATCH]) * scale
        lg = model(torch.from_numpy(xb.astype(np.float32))).numpy()
        ps.append(1 / (1 + np.exp(-lg)))
    sig = np.concatenate(ps)
    return 1 - sig[:, NORM_IDX]


def auc_rec(p, win_rec, rec_y, n_rec):
    return float(roc_auc_score(rec_y, top2mean_by_record(p, win_rec, n_rec)))


def get_ptb(name, split, arrays_cache={}):
    if name not in arrays_cache:
        from data.split_guard import load_arrays  # noqa: E402
        beats, labels, rids = load_arrays('ptb')
        arrays_cache['raw'] = (np.asarray(beats, np.float32),
                               np.asarray(labels), np.asarray(rids))
    beats, labels, rids = arrays_cache['raw']
    mask = np.asarray(split[name], bool)
    return windows_from_beats(beats[mask], labels[mask], rids[mask])


def main():
    fake_lightning()
    import r8_data_5s  # noqa: F401,E402  (sys.path/env side effects)
    from data.split_guard import compute_ptb_split  # noqa: E402
    split = compute_ptb_split()
    RUNDIR.mkdir(parents=True, exist_ok=True)

    s_train = train_fold_scale()
    sens = {'scale_s_from_ptbxl_train_leadII_std': s_train,
            'note': 's derived from their training-data statistics only; '
                    'sweep below is diagnostic on PTB val; the amplitude-'
                    'matched variant uses the pre-registered s'}

    models = {'resnet': None, 'xresnet': None}
    from r10_lit_strodthoff import build_model
    for arch in models:
        models[arch] = build_model(arch)
    models['lenet'] = build_lenet()
    print('lenet loaded strict=True OK', flush=True)

    # fold-10 validation for lenet (same protocol as resnet/xresnet rows)
    d10 = np.load('/home/devcontainers/ptbxl_data/fold10.npz')
    x10 = np.ascontiguousarray(
        d10['x'][:, 375:625, :].transpose(0, 2, 1), np.float32)
    with torch.no_grad():
        lg = np.concatenate([models['lenet'](torch.from_numpy(
            x10[b:b + BATCH])).numpy() for b in range(0, len(x10), BATCH)])
    p10 = 1 / (1 + np.exp(-lg))
    aucs10 = [float(roc_auc_score(d10['y5'][:, i], p10[:, i]))
              for i in range(5)]
    lenet_val = {'per_class': aucs10, 'macro': float(np.mean(aucs10)),
                 'n': int(len(d10['y5'])),
                 'classes': ['CD', 'HYP', 'MI', 'NORM', 'STTC']}
    print('lenet fold-10 macro AUROC:', round(lenet_val['macro'], 4),
          flush=True)
    with open(RUNDIR / 'fold10_validation_lenet.json', 'w') as f:
        json.dump(lenet_val, f, indent=2)

    # ---- amplitude sweep (diagnostic, PTB val only) ----
    pv = get_ptb('val', split)
    for arch, model in models.items():
        row_sens = {}
        for tag, sc in [('x0.0625', s_train / 4), ('x0.125', s_train / 2),
                        ('x_s', s_train), ('x2s', 2 * s_train),
                        ('unified_zscore', 1.0)]:
            p = scores(model, pv['x'], scale=sc)
            row_sens[tag] = {'scale': round(sc, 4),
                             'ptb_val_auc': auc_rec(p, pv['win_rec'],
                                                    pv['rec_y'],
                                                    pv['n_records'])}
            print(f'{arch} {tag} (x{sc:.4f}) ptb_val '
                  f'{row_sens[tag]["ptb_val_auc"]:.4f}', flush=True)
        sens[arch] = row_sens

    # ---- amplitude-matched variant at pre-registered s (all domains) ----
    pt = get_ptb('test', split)
    me = meeti_windows3()
    va = me['rec_va']
    remap = np.full(me['n_records'], -1, np.int64)
    remap[np.where(va)[0]] = np.arange(int(va.sum()))
    sel = remap[me['win_rec']] >= 0
    for arch, model in models.items():
        variant = {'scale_s': s_train}
        p = scores(model, pv['x'], scale=s_train)
        variant['ptb_val'] = auc_rec(p, pv['win_rec'], pv['rec_y'],
                                     pv['n_records'])
        p = scores(model, pt['x'], scale=s_train)
        variant['ptb_test'] = auc_rec(p, pt['win_rec'], pt['rec_y'],
                                      pt['n_records'])
        p = scores(model, me['x'][sel], scale=s_train)
        variant['meeti_val'] = auc_rec(p, remap[me['win_rec']][sel],
                                       me['rec_y'][va], int(va.sum()))
        w = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}
        variant['S'] = round(100 * sum(
            w[k] * (variant[k] - 0.5) / 0.5 for k in w), 2)
        print(f'{arch} amplitude-matched: MEETI {variant["meeti_val"]:.4f} '
              f'/ PTBv {variant["ptb_val"]:.4f} / PTBt '
              f'{variant["ptb_test"]:.4f} -> S {variant["S"]}', flush=True)
        sens[arch]['amplitude_matched_variant'] = variant

        # fold-10 control: does s also change THEIR domain? (should be ~nil
        # there since fold10 is already raw mV - sanity only, skip for time)
    with open(RUNDIR / 'amplitude_sensitivity.json', 'w') as f:
        json.dump(sens, f, indent=2)
    print('saved sensitivity ->', RUNDIR / 'amplitude_sensitivity.json',
          flush=True)

    # patch existing row JSONs with the sensitivity pointer + variant
    for arch in ['resnet', 'xresnet']:
        rf = OUT / f'lit_row_lit_{arch}_lead2fill_3beat.json'
        row = json.load(open(rf))
        row['amplitude_sensitivity'] = sens[arch]
        json.dump(row, open(rf, 'w'), indent=2)

    # ---- lenet full row (unified caliber primary) ----
    SOURCE = ('Strodthoff-lab XAI4ECG LeNet baseline; weights: same '
              'Fraunhofer release (lenet_superdiagnostic/best_model.ckpt); '
              'paper: Wagner et al. 2024 CBM arXiv:2305.17043, suite from '
              'Strodthoff et al. 2020 JBHI arXiv:2004.13701')
    row = {'family': 'lit_lenet_lead2fill_3beat', 'source': SOURCE,
           'caveat': 'same as lit_resnet/xresnet rows (PTB-XL trained, '
                     'different-source DB/split, lead2 fill OOD, 2.5s crop '
                     'vs 3s beat-concat, raw-mV train vs unit-scale eval)',
           'adaptation': 'identical to lit_resnet/xresnet rows',
           'protocol': {'binary_map': '1 - P(NORM)',
                        'record_score': 'top2mean (legacy mean also '
                                        'reported)',
                        'seeds': 'single official ckpt (sec 11.6)'},
           'validation': lenet_val,
           'amplitude_sensitivity': sens['lenet']}
    p = scores(models['lenet'], pv['x'])
    row['ptb_val'] = auc_rec(p, pv['win_rec'], pv['rec_y'], pv['n_records'])
    row['ptb_val_legacy_mean'] = float(roc_auc_score(
        pv['rec_y'], record_mean(p, pv['win_rec'], pv['n_records'])))
    p = scores(models['lenet'], pt['x'])
    row['ptb_test'] = auc_rec(p, pt['win_rec'], pt['rec_y'], pt['n_records'])
    row['ptb_test_legacy_mean'] = float(roc_auc_score(
        pt['rec_y'], record_mean(p, pt['win_rec'], pt['n_records'])))
    p = scores(models['lenet'], me['x'][sel])
    row['meeti_val'] = auc_rec(p, remap[me['win_rec']][sel], me['rec_y'][va],
                               int(va.sum()))
    row['meeti_val_legacy_mean'] = float(roc_auc_score(
        me['rec_y'][va], record_mean(p, remap[me['win_rec']][sel],
                                     int(va.sum()))))
    dev = {}
    for name, dd in device_windows3().items():
        p = scores(models['lenet'], dd['x'])
        dev[name] = {'n_windows': int(len(p)), 'p_mean': float(p.mean()),
                     'frac_gt_0.5': float((p > 0.5).mean())}
    row['device_3beat'] = dev
    w = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}
    row['S'] = round(100 * sum(w[k] * (row[k] - 0.5) / 0.5 for k in w), 2)
    print('lenet S =', row['S'], flush=True)
    with open(OUT / 'lit_row_lit_lenet_lead2fill_3beat.json', 'w') as f:
        json.dump(row, f, indent=2)
    print('lenet row saved', flush=True)


if __name__ == '__main__':
    main()
