#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R10 lit rows: official Strodthoff-lab xai4ecg checkpoints (resnet50-1D /
xresnet1d50, PTB-XL superdiagnostic=5 superclass heads).

Two stages:
  --validate  reproduce their reported PTB-XL fold-10 macro-AUROC
              (CenterCrop 250 of the raw 1000-pt 12-lead record; reference
              bootstrap means ~0.928 resnet / ~0.924 xresnet) to prove the
              checkpoints are loaded with correct semantics.
  --bench     unified benchmark rows: 3-beat concat windows (750 pt @250Hz,
              stride 3, last-beat repeat pad - same rule family as
              ptb_5s/meeti_5s) -> resample_poly 2/5 to 100 Hz (300 pt) ->
              CenterCrop 250 (the model's native input/eval transform) ->
              per-window z-score (Sec 11.2 rule; ~identity on our unit-scale
              eval windows) -> lead-II fill replicated to 12 channels ->
              binary score = 1 - P(NORM) (variant: max non-NORM sigmoid).
              Record score = top2mean (legacy mean also reported).

No training anywhere; official frozen checkpoints => single-checkpoint rows
(no seeds, BENCHMARK_V1 sec 11.6).  Zero contact with PTB val/test patients.
"""
import argparse
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import signal
from sklearn.metrics import roc_auc_score

sys.path.insert(0, '/home/devcontainers/meeti/scripts')

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/benchmark_v1'
XAI = Path('/home/devcontainers/xai4ecg-master')
CKPTS = Path('/home/devcontainers/strodthoff_out/output')
PTBXL = Path('/home/devcontainers/ptbxl_data')
NORM_IDX = 3                       # class order ['CD','HYP','MI','NORM','STTC']
BATCH = 256

SOURCE = ('Strodthoff et al. 2020 JBHI (arXiv:2004.13701) family; official '
          'trained checkpoints from Wagner, Mehari, Haverkamp, Strodthoff '
          '2024 CBM (arXiv:2305.17043) XAI4ECG audit repo '
          'github.com/hhi-aml/xai4ecg, weights: '
          'datacloud.hhi.fraunhofer.de/s/gLkjQL94d7FXBbS '
          '(output/<model>_superdiagnostic/checkpoints/best_model.ckpt)')
CAVEAT = ('trained on PTB-XL (21.8k records, strat_fold 9/10 split) - a '
          'DIFFERENT-SOURCE database and split from our PTB-DB split_guard '
          'gold domain (same PTB institution; patient overlap not '
          'verifiable); input = 2.5 s center crop @100 Hz vs our 3 s '
          'beat-concat windows; trained on raw mV while our eval windows '
          'are per-beat z-scored unit scale (amplitude OOD, disclosed); '
          'lead-II fill is OOD for 12-lead models; 5-class sigmoid head '
          'mapped to binary via 1-P(NORM)')


def fake_lightning():
    pl = types.ModuleType('pytorch_lightning')
    pl.LightningModule = nn.Module
    sys.modules['pytorch_lightning'] = pl
    sys.path.insert(0, str(XAI / 'code'))


class XResInner(nn.Module):
    """ECGResNet backbone minus its l1 head: features -> flatten -> bn
    (matches ckpt keys model.0.features.* / model.0.bn.*)."""

    def __init__(self, feats, nf):
        super().__init__()
        self.features = feats
        self.bn = nn.BatchNorm1d(nf)

    def forward(self, x):
        h = self.features(x).flatten(1)
        return self.bn(h)


def build_model(arch):
    """Wrap xai4ecg backbones so state-dict keys match the Lightning ckpts:
    model.0.<backbone>, model.0.bn (xresnet), model.1.1.1.<Linear head>."""
    import ecg_model as em
    if arch == 'resnet':
        rn = em.resnet50(num_classes=5)
        rn.fc = nn.Identity()                     # ckpt has no fc keys
        inner = rn                                # keys: 0.conv1..., avgpool paramless
        nf = 512
    else:
        xr = em.xresnet1d50(num_classes=5, widen=1.0, concat_pooling=True)
        ch = list(xr.children())
        xr[0][0] = nn.Conv1d(12, 32, kernel_size=5, stride=2, padding=2)
        feats = nn.Sequential(*ch[:-1], ch[-1][0])   # ECGResNet.features
        nf = int(ch[-1][-1].in_features)             # 512 (concat pool 2x256)
        inner = XResInner(feats, nf)
    head = nn.Sequential(nn.Dropout(0.5),
                         nn.Sequential(nn.Flatten(), nn.Linear(nf, 5)))
    model = nn.Sequential(inner, head)
    ck = torch.load(CKPTS / f'{arch}_superdiagnostic/checkpoints/'
                    'best_model.ckpt', map_location='cpu', weights_only=False)
    sd = {k[len('model.'):]: v for k, v in ck['state_dict'].items()}
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model


@torch.no_grad()
def logits(model, x):
    out = []
    for s in range(0, len(x), BATCH):
        out.append(model(torch.from_numpy(x[s:s + BATCH])).numpy())
    return np.concatenate(out)


def validate(model, arch):
    d = np.load(PTBXL / 'fold10.npz')
    x, y5 = d['x'], d['y5']
    x = np.ascontiguousarray(x[:, 375:625, :].transpose(0, 2, 1),
                             dtype=np.float32)   # (N,12,250) CenterCrop 250
    p = 1 / (1 + np.exp(-logits(model, x)))
    aucs = [float(roc_auc_score(y5[:, i], p[:, i])) for i in range(5)]
    macro = float(np.mean(aucs))
    print(f'{arch} fold-10 per-class AUROC:',
          [round(a, 4) for a in aucs], flush=True)
    print(f'{arch} macro AUROC: {macro:.4f}', flush=True)
    return {'per_class': aucs, 'macro': macro, 'n': int(len(y5)),
            'classes': ['CD', 'HYP', 'MI', 'NORM', 'STTC'],
            'input': 'CenterCrop 250 of raw 1000-pt 12-lead 100 Hz '
                     '(their exact test transform)'}


# ---------------- benchmark windows (3-beat concat, rule of ptb_5s/meeti_5s)

def windows_from_beats(beats, labels, rids, per_win=3):
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True,
                                    return_counts=True)
    import math
    n_rec = len(uniq)
    rec_y = np.zeros(n_rec, np.int32)
    xs, win_rec = [], []
    for i in range(n_rec):
        s, nb = int(first[i]), int(counts[i])
        rec_y[i] = int(labels[s:s + nb].max())
        n_win = int(math.ceil(nb / per_win))
        pos = np.minimum(np.arange(n_win * per_win), nb - 1)
        xs.append(beats[pos + s].reshape(n_win, per_win * 250))
        win_rec.append(np.full(n_win, i, np.int64))
    return {'x': np.concatenate(xs, axis=0),
            'win_rec': np.concatenate(win_rec, axis=0),
            'rec_y': rec_y, 'rids': uniq, 'n_records': n_rec}


def meeti_windows3():
    import pandas as pd
    import math
    MEETI_DIR = ROOT / 'data/processed_small'
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)

    def to_y(c):
        return 1 if c == 'abnormal' else (0 if c == 'normal' else -1)

    manifest['_y'] = manifest['report_class'].apply(to_y)
    manifest = manifest[manifest['_y'] >= 0].reset_index(drop=True)
    n_beats = manifest['n_beats'].astype(int).to_numpy()
    assert int(n_beats.sum()) == len(beats)
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]]).astype(np.int64)
    y = manifest['_y'].to_numpy(np.int32)
    subjects = manifest['subject_id'].to_numpy()
    n_rec = len(manifest)
    xs, win_rec = [], []
    for i in range(n_rec):
        nb = int(n_beats[i])
        n_win = int(math.ceil(nb / 3))
        flat = np.minimum(np.arange(n_win * 3), nb - 1)
        xs.append(beats[flat + positions[i]].reshape(n_win, 750))
        win_rec.append(np.full(n_win, i, np.int64))
    x = np.concatenate(xs, axis=0)
    win_rec = np.concatenate(win_rec, axis=0)
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), np.int32)
    for yi, si in zip(y, subj_idx):
        subj_y[si] = max(subj_y[si], yi)
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    _, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    rec_va = np.isin(subj_idx, va_s)
    return {'x': x, 'win_rec': win_rec, 'rec_y': y, 'rec_va': rec_va,
            'n_records': n_rec}


def device_windows3():
    from r8_data_5s import REAL_DIR, REAL_FILES  # noqa: E402
    out = {}
    for name, fname in REAL_FILES.items():
        b = np.load(REAL_DIR / fname).astype(np.float32)
        n_win = len(b) - 3 + 1
        idx = np.arange(n_win)[:, None] + np.arange(3)[None, :]
        out[name] = {'x': b[idx].reshape(n_win, 750)}
    return out


def prep_windows(x750):
    """(N,750) -> (N,250,12): resample 250->100 Hz, CenterCrop 250,
    per-window z-score, lead-II fill to 12 channels."""
    r = signal.resample_poly(x750.astype(np.float64), 2, 5, axis=1)
    x = r[:, 25:275]                                   # center 250 of 300
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True) + 1e-6
    x = (x - mu) / sd
    return np.repeat(x[:, None, :], 12, axis=1).astype(np.float32)


def top2mean_by_record(p_win, win_rec, n_rec):
    order = np.argsort(win_rec, kind='stable')
    p, r = p_win[order], win_rec[order]
    uniq, first, counts = np.unique(r, return_index=True, return_counts=True)
    out = np.zeros(n_rec)
    for i, (s, c) in enumerate(zip(first, counts)):
        v = np.sort(p[s:s + c])[::-1]
        out[i] = v[:min(2, len(v))].mean()
    return out


def record_mean(p_win, win_rec, n_rec):
    cnt = np.bincount(win_rec, minlength=n_rec).astype(np.float64)
    tot = np.bincount(win_rec, weights=p_win.astype(np.float64),
                      minlength=n_rec)
    return tot / np.maximum(cnt, 1)


def bench(model, arch):
    import r8_data_5s  # noqa: F401  (sys.path/env side effects for data.*)
    from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
    split = compute_ptb_split()
    row = {'family': f'lit_{arch}_lead2fill_3beat',
           'source': SOURCE, 'caveat': CAVEAT,
           'adaptation': '3-beat concat windows (750 pt @250 Hz, stride 3, '
                         'last-beat repeat pad) -> resample_poly 2/5 to '
                         '100 Hz (300 pt) -> CenterCrop 250 (native input '
                         'transform) -> per-window z-score -> lead-II fill '
                         'x12 channels; official frozen ckpt, no training',
           'protocol': {'binary_map': '1 - P(NORM) (primary); max non-NORM '
                                      'sigmoid (variant)',
                        'norm_class_idx': NORM_IDX,
                        'record_score': 'top2mean (legacy mean also '
                                        'reported)',
                        'seeds': 'single official checkpoint (zero-'
                                 'training row, sec 11.6)'},
           'validation': 'see runs/r10_strodthoff/fold10_validation.json'}

    def scores(x750):
        ps = []
        for s in range(0, len(x750), BATCH):
            xb = torch.from_numpy(prep_windows(x750[s:s + BATCH]))
            with torch.no_grad():
                lg = model(xb).numpy()
            sig = 1 / (1 + np.exp(-lg))
            ps.append(sig)
        sig = np.concatenate(ps)
        primary = 1 - sig[:, NORM_IDX]
        variant = np.max(np.delete(sig, NORM_IDX, axis=1), axis=1)
        return primary, variant

    for name in ['val', 'test']:
        beats, labels, rids = load_arrays('ptb')
        mask = np.asarray(split[name], bool)
        d = windows_from_beats(np.asarray(beats, np.float32)[mask],
                               np.asarray(labels)[mask],
                               np.asarray(rids)[mask])
        primary, variant = scores(d['x'])
        n_rec = d['n_records']
        row[f'ptb_{name}'] = float(roc_auc_score(
            d['rec_y'], top2mean_by_record(primary, d['win_rec'], n_rec)))
        row[f'ptb_{name}_legacy_mean'] = float(roc_auc_score(
            d['rec_y'], record_mean(primary, d['win_rec'], n_rec)))
        row[f'ptb_{name}_max_variant'] = float(roc_auc_score(
            d['rec_y'], top2mean_by_record(variant, d['win_rec'], n_rec)))
        print(f'ptb_{name}', round(row[f'ptb_{name}'], 4), flush=True)

    me = meeti_windows3()
    va = me['rec_va']
    remap = np.full(me['n_records'], -1, np.int64)
    remap[np.where(va)[0]] = np.arange(int(va.sum()))
    sel = remap[me['win_rec']] >= 0
    primary, variant = scores(me['x'][sel])
    n_val = int(va.sum())
    row['meeti_val'] = float(roc_auc_score(
        me['rec_y'][va], top2mean_by_record(primary, remap[me['win_rec']][sel],
                                            n_val)))
    row['meeti_val_legacy_mean'] = float(roc_auc_score(
        me['rec_y'][va], record_mean(primary, remap[me['win_rec']][sel],
                                     n_val)))
    print('meeti_val', round(row['meeti_val'], 4), flush=True)

    dev = {}
    for name, d in device_windows3().items():
        primary, _ = scores(d['x'])
        dev[name] = {'n_windows': int(len(primary)),
                     'p_mean': float(primary.mean()),
                     'frac_gt_0.5': float((primary > 0.5).mean())}
    row['device_3beat'] = dev

    w = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}
    row['S'] = round(100 * (w['ptb_val'] * (row['ptb_val'] - 0.5) / 0.5
                            + w['ptb_test'] * (row['ptb_test'] - 0.5) / 0.5
                            + w['meeti_val'] * (row['meeti_val'] - 0.5) / 0.5),
                     2)
    print('S =', row['S'], flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--validate', action='store_true')
    ap.add_argument('--bench', action='store_true')
    args, _ = ap.parse_known_args()
    outdir = ROOT / 'runs/r10_strodthoff'
    outdir.mkdir(parents=True, exist_ok=True)
    fake_lightning()

    for arch in ['resnet', 'xresnet']:
        model = build_model(arch)
        print(f'{arch} loaded strict=True OK', flush=True)
        if args.validate:
            res = validate(model, arch)
            with open(outdir / f'fold10_validation_{arch}.json', 'w') as f:
                json.dump(res, f, indent=2)
        if args.bench:
            row = bench(model, arch)
            with open(OUT / f'lit_row_lit_{arch}_lead2fill_3beat.json',
                      'w') as f:
                json.dump(row, f, indent=2)
            print('row saved:', arch, flush=True)


if __name__ == '__main__':
    main()
