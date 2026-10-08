#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 ECG-FM pretrained encoder 对已保存的 MEETI 5s 段抽 768 维全局嵌入(CPU)。"""
import json, time
from pathlib import Path
import numpy as np
import torch
from fairseq_signals.models import build_model_from_checkpoint

CKPT = '/mnt/d/models/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt'
SEG = Path('/home/devcontainers/meeti/data/processed_small/teacher_segments.npy')
OUT = Path('/home/devcontainers/meeti/data/processed_small/teacher_emb_768.npy')
META = Path('/home/devcontainers/meeti/data/processed_small/teacher_emb_meta.json')
BATCH = 16

def main():
    segs = np.load(SEG, mmap_mode='r')  # (N,2,12,2500)
    print('segments', segs.shape, flush=True)
    model = build_model_from_checkpoint(checkpoint_path=CKPT)
    model.eval()
    n = int(segs.shape[0])
    emb = np.zeros((n, 768), dtype=np.float32)
    t0 = time.time()
    for start in range(0, n, BATCH):
        end = min(n, start + BATCH)
        xs = []
        for i in range(start, end):
            x = np.asarray(segs[i], dtype=np.float32).T  # (2500,12)->(12,2500)
            mu = x.mean(); sd = x.std() + 1e-6
            xs.append(((x - mu) / sd)[None])  # (1,12,2500)
        xb = torch.from_numpy(np.concatenate(xs)).float()
        with torch.no_grad():
            out = model(source=xb)
        feats = out['features']  # (B,156,768)
        g = feats.mean(dim=1).numpy().astype(np.float32)
        emb[start:end] = g
        if (end % 1000) < BATCH or end == n:
            print(f'{end}/{n} elapsed={time.time()-t0:.0f}s', flush=True)
    np.save(OUT, emb)
    META.write_text(json.dumps({
        'checkpoint': CKPT, 'n_records': int(segs.shape[0]),
        'n_segments': n, 'dim': 768,
        'pooling': 'mean over time dim of encoder features',
        'elapsed_s': round(time.time()-t0, 1),
        'out': str(OUT),
    }, indent=2), encoding='utf-8')
    print('saved', OUT, emb.shape, 'elapsed', time.time()-t0, flush=True)

if __name__ == '__main__':
    main()
