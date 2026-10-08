#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Extract ECG-FM center-time-step local features from beat-centered windows.

Input : beat_windows.npy (B,12,2500) at 500 Hz (raw MIMIC mV scale).
Model : /mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt
Output: teacher_local_768.npy (B,768), selected center step = round(2500/5000*156)=78
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from fairseq_signals.models import build_model_from_checkpoint

CKPT = '/mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt'
CENTER_STEP = int(round(2500 / 5000 * 156))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--windows',
                    default='/home/devcontainers/meeti/data/processed_small/m3_beat_centered/beat_windows.npy')
    ap.add_argument('--out',
                    default='/home/devcontainers/meeti/data/processed_small/m3_beat_centered/teacher_local_768.npy')
    ap.add_argument('--meta',
                    default='/home/devcontainers/meeti/data/processed_small/m3_beat_centered/teacher_local_meta.json')
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--limit', type=int, default=0,
                    help='debug: only first N windows')
    ap.add_argument('--device', default=None)
    args = ap.parse_args()

    win = np.load(args.windows, mmap_mode='r')
    n_total = int(win.shape[0])
    if args.limit:
        n_total = min(n_total, args.limit)
    if win.shape[1:] != (12, 2500):
        raise SystemExit(f'expected (B,12,2500), got {win.shape}')

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'windows={win.shape} n={n_total} device={device} '
          f'center_step={CENTER_STEP}', flush=True)

    model = build_model_from_checkpoint(checkpoint_path=CKPT)
    model.to(device)
    model.eval()

    out = args.out
    emb = np.lib.format.open_memmap(
        out, mode='w+', dtype=np.float32, shape=(n_total, 768))
    t0 = time.time()
    start = 0
    while start < n_total:
        end = min(n_total, start + args.batch_size)
        xs = []
        for i in range(start, end):
            x = np.asarray(win[i], dtype=np.float32)  # (12,2500)
            mu = float(x.mean())
            sd = float(x.std()) + 1e-6
            xs.append(((x - mu) / sd)[None])
        xb = torch.from_numpy(np.concatenate(xs)).float().to(device)
        with torch.no_grad():
            outd = model(source=xb)
        feats = outd['features']  # (B,156,768)
        if feats.shape[1] < CENTER_STEP + 1:
            raise RuntimeError(f'ECG-FM feature time dim {feats.shape[1]} '
                               f'smaller than center step {CENTER_STEP}')
        local = feats[:, CENTER_STEP, :].cpu().numpy().astype(np.float32)
        emb[start:end] = local
        start = end
        if (end % 500) < args.batch_size or end == n_total:
            print(f'{end}/{n_total} elapsed={time.time() - t0:.0f}s', flush=True)
    emb.flush()
    del emb

    meta = {
        'checkpoint': CKPT,
        'input': str(Path(args.windows)),
        'out': str(Path(out)),
        'n_windows': n_total,
        'shape': [n_total, 768],
        'center_step_0based': CENTER_STEP,
        'center_step_formula': 'round(2500/5000*156)=78',
        'feature_dim': 768,
        'normalization': 'per-window (x-mean)/(std+1e-6) over 12x2500',
        'device': str(device),
        'elapsed_s': round(time.time() - t0, 1),
    }
    with open(args.meta, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(json.dumps(meta, indent=2), flush=True)


if __name__ == '__main__':
    main()
