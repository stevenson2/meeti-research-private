#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BENCHMARK_V1: ECG-FM teacher embeddings for the benchmark domains.

Teacher = ECG-FM (frozen) + StandardScaler->PLS16->LR head fitted ONCE on
the MIMIC train-subject windows' teacher_local_768.npy (full-12, the same
features the students' KD targets come from).  Benchmark domains only have
single-lead 250 Hz beats, so the teacher is evaluated in its DEPLOYABLE
lead-II-fill form: 5-beat concat window (1250 pt) replicated to 12 channels,
global z-score per window (FM convention, same as r7 probe).

Outputs runs/benchmark_v1/teacher_emb_{meeti_val,ptb_val,ptb_test,device}.npz
   x_emb (N,768), win_rec, rec_y
Latency smoke: --smoke 8.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import meeti_5s, ptb_5s, device_5s  # noqa: E402
from r9_data import load_all  # noqa: E402
from data.split_guard import compute_ptb_split  # noqa: E402

CKPT = '/mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt'
OUT = Path('/home/devcontainers/meeti/runs/benchmark_v1')
BATCH = 16


def load_fm():
    from fairseq_signals.models import build_model_from_checkpoint
    model = build_model_from_checkpoint(checkpoint_path=CKPT)
    model.eval()
    return model


def extract(model, x1250, desc):
    """x1250: (N,1250) lead-II windows -> (N,768) embeddings."""
    n = len(x1250)
    emb = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    for s in range(0, n, BATCH):
        e = min(n, s + BATCH)
        xs = []
        for j in range(s, e):
            w = x1250[j].astype(np.float32)
            mu, sd = w.mean(), w.std() + 1e-6
            xs.append(((w - mu) / sd)[None, None, :].repeat(12, axis=1))
        xb = torch.from_numpy(np.concatenate(xs)).float()
        with torch.no_grad():
            out = model(source=xb)
        emb[s:e] = out['features'].mean(dim=1).numpy().astype(np.float32)
        if (e % 160) < BATCH or e == n:
            r = (n - e) * (time.time() - t0) / max(e, 1)
            print(f'{desc} {e}/{n} elapsed={time.time()-t0:.0f}s '
                  f'eta={r:.0f}s', flush=True)
    return emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--smoke', type=int, default=0)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    data = load_all(norm=True, form=True)
    split = compute_ptb_split()

    sets = {}
    me = data['meeti']
    va_win = me['rec_va'][me['win_rec']]
    val_records = np.where(me['rec_va'])[0]
    remap = np.zeros(me['n_records'], np.int64)
    remap[val_records] = np.arange(len(val_records))
    sets['meeti_val'] = (me['x'][va_win],
                         remap[me['win_rec'][va_win]],
                         me['rec_y'][val_records])
    for name in ['val', 'test']:
        d = ptb_5s(split[name])
        sets[f'ptb_{name}'] = (d['x'], d['win_rec'], d['rec_y'])
    dev = device_5s()
    dx = np.concatenate([v['x'] for v in dev.values()], axis=0)
    dn = len(dev['ecg_real_052']['x'])
    sets['device'] = (dx,
                      np.concatenate([np.zeros(dn, np.int64),
                                      np.ones(len(dx) - dn, np.int64)]),
                      np.zeros(len(dx), np.int32))

    total = sum(len(v[0]) for v in sets.values())
    print('windows per set:',
          {k: len(v[0]) for k, v in sets.items()}, f'total={total}',
          flush=True)

    model = load_fm()
    if args.smoke:
        t0 = time.time()
        extract(model, sets['meeti_val'][0][:args.smoke], 'smoke')
        dt = (time.time() - t0) / args.smoke
        print(f'latency {dt:.3f}s/window -> full run ~{dt*total/60:.0f} min',
              flush=True)
        return

    for name, (x, win_rec, rec_y) in sets.items():
        out = OUT / f'teacher_emb_{name}.npz'
        if out.exists():
            print('skip', name, '(exists)', flush=True)
            continue
        emb = extract(model, x, name)
        np.savez_compressed(out, x_emb=emb, win_rec=win_rec, rec_y=rec_y)
        print('saved', out, flush=True)


if __name__ == '__main__':
    main()
