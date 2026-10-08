#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4 diagnosis: why does locked Head_R saturate (0.75-0.95) on real AFE normal beats?

Compares, under the SAME causal sliding-window protocol as m4_afe_event_eval.py:
  - real AFE device recordings (2, all normal)
  - MIT/INCART split_guard val records (26 abnormal / 1 normal)
  - PTB val patient records (gold labels, report-only)
  - MEETI val record windows (training-style first-10-beat windows)

For each group we collect Head_R window scores, Head_B per-beat probabilities,
fc1 embeddings (64-d) and attention entropy/argmax over the 10-beat window,
then quantify the domain shift of real beats in fc1 space.

Outputs -> /home/devcontainers/meeti/runs/m4_head_r_diagnosis/
  analysis.json  (group stats)
  arrays.npz     (scores/embeddings per group for follow-up post-processing)
"""
import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')  # CPU: avoid PTX JIT stall

import numpy as np
import tensorflow as tf

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')

from train_dual_head_v3_small import build_full_model  # noqa: E402
from train_dual_head_small import load_meeti_records  # noqa: E402
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa: E402

T = 10
RUN = Path('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb')
OUT = Path('/home/devcontainers/meeti/runs/m4_head_r_diagnosis')
REAL = {
    'ecg_real_052': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_exp7c.npy',
    'rec_latest': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_rec_latest.npy',
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', default=str(RUN / 'full.weights.h5'))
    ap.add_argument('--beat-probe', default=str(RUN / 'ckpt_balanced__beat.keras'))
    ap.add_argument('--record-probe', default=str(RUN / 'ckpt_balanced__record.keras'))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- rebuild full model, load locked balanced weights ----
    model, beat_probe, rec_probe = build_full_model(lr=1e-4, lam=0.2)
    model.load_weights(args.weights)
    print('full weights loaded', flush=True)

    # sub-models: fc1 encoder + (attention, head_r) from the full model
    fc1_model = tf.keras.Model(model.inputs[0], model.get_layer('head_b').input)
    rec_fc1 = model.get_layer('td_encoder')
    rec_proj = model.get_layer('td_head_r_proj')
    rec_attn = model.get_layer('head_r_attention')
    attn_model = tf.keras.Model(
        [model.inputs[1], model.inputs[2]],
        [model.outputs[1], rec_proj(rec_fc1(model.inputs[1]))])
    att_w, att_b = rec_attn.score_dense.get_weights()

    def head_b(beats):
        return beat_probe.predict(beats[..., None], batch_size=512,
                                  verbose=0)[:, 1]

    def head_r_windows(beats):
        """Causal last-10-beat windows, identical to m4_afe_event_eval."""
        n = len(beats)
        x = np.zeros((n, T, beats.shape[1], 1), dtype=np.float32)
        m = np.zeros((n, T, 1), dtype=np.float32)
        for i in range(n):
            lo = max(0, i - T + 1)
            x[i, -(i - lo + 1):] = beats[lo:i + 1, ..., None]
            m[i, -(i - lo + 1):] = 1.0
        p, proj = attn_model.predict([x, m], batch_size=512, verbose=0)
        score = proj @ att_w[:, 0] + att_b[0]          # (n, T)
        score = np.where(m[:, :, 0] > 0.5, score, -1e9)
        score = score - score.max(axis=1, keepdims=True)
        e = np.exp(score) * (m[:, :, 0] > 0.5)
        att = e / (e.sum(axis=1, keepdims=True) + 1e-12)
        return p[:, 1], att, x, m

    def attn_stats(att, m):
        ent = []
        argmax_last3 = []
        for a, mk in zip(att, m[:, :, 0]):
            a = a[mk > 0.5]
            a = a / (a.sum() + 1e-12)
            ent.append(float(-(a * np.log(a + 1e-12)).sum()))
            argmax_last3.append(bool(np.argmax(a) >= len(a) - 3))
        return float(np.mean(ent)), float(np.mean(argmax_last3))

    store = {}
    groups = {}

    # ---------- real AFE recordings ----------
    for name, path in REAL.items():
        beats = np.load(path).astype(np.float32)
        pb = head_b(beats)
        pr, att, x, m = head_r_windows(beats)
        fc1 = fc1_model.predict(beats[..., None], batch_size=512, verbose=0)
        ent, am3 = attn_stats(att, m)
        store[f'real__{name}__head_b'] = pb
        store[f'real__{name}__head_r'] = pr
        store[f'real__{name}__fc1'] = fc1
        groups[f'real_{name}'] = {
            'n_beats': int(len(beats)),
            'head_b_mean': float(pb.mean()),
            'head_b_q95': float(np.quantile(pb, 0.95)),
            'head_r_mean': float(pr.mean()),
            'head_r_q05_q50_q95': [float(np.quantile(pr, q)) for q in (.05, .5, .95)],
            'attn_entropy_mean': ent,
            'attn_argmax_in_last3_frac': am3,
        }
        print(name, json.dumps(groups[f'real_{name}']), flush=True)

    # ---------- MIT/INCART val (same protocol as m4) ----------
    split = compute_mit_incart_split()
    val_groups = {}
    for ds in ['mit_bih', 'incart']:
        beats, labels, rids = load_arrays(ds)
        vm = np.asarray(split['val_masks'][ds], dtype=bool)
        beats = np.asarray(beats[vm], dtype=np.float32)
        labels = np.asarray(labels[vm])
        rids = np.asarray(rids[vm])
        order = np.argsort(rids, kind='stable')
        beats, labels, rids = beats[order], labels[order], rids[order]
        uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
        for ui, rid in enumerate(uniq):
            s, e = first[ui], first[ui] + counts[ui]
            val_groups[f'{ds}_{int(rid)}'] = {
                'beats': beats[s:e], 'label': int(labels[s:e].max())}

    mit_rows, mit_all_r, mit_all_b, mit_fc1_all = [], [], [], []
    mit_rid_per_beat, mit_lab_per_beat = [], []
    mit_ent_by_label = {0: [], 1: []}
    for rid, rec in val_groups.items():
        pb = head_b(rec['beats'])
        pr, att, x, m = head_r_windows(rec['beats'])
        fc1 = fc1_model.predict(rec['beats'][..., None], batch_size=512, verbose=0)
        ent, am3 = attn_stats(att, m)
        mit_ent_by_label[rec['label']].append(ent)
        mit_all_r.append(pr)
        mit_all_b.append(pb)
        mit_fc1_all.append(fc1)
        mit_rid_per_beat.append(np.full(len(rec['beats']), rid))
        mit_lab_per_beat.append(np.full(len(rec['beats']), rec['label']))
        mit_rows.append({
            'record': rid, 'label': rec['label'], 'n_beats': int(len(rec['beats'])),
            'head_r_mean': float(pr.mean()),
            'head_r_q95': float(np.quantile(pr, 0.95)),
            'head_b_mean': float(pb.mean()),
            'attn_entropy_mean': ent,
        })
        print('mitval', rid, mit_rows[-1], flush=True)
    mit_r = np.concatenate(mit_all_r)
    mit_b = np.concatenate(mit_all_b)
    mit_fc1 = np.concatenate(mit_fc1_all, axis=0)
    store['mitval__head_r'] = mit_r
    store['mitval__head_b'] = mit_b
    store['mitval__fc1'] = mit_fc1
    store['mitval__rid'] = np.concatenate(mit_rid_per_beat)
    store['mitval__rec_label'] = np.concatenate(mit_lab_per_beat)

    for lab in (0, 1):
        rows = [r for r in mit_rows if r['label'] == lab]
        if rows:
            groups[f'mitval_label{lab}'] = {
                'n_records': len(rows),
                'head_r_mean_of_means': float(np.mean([r['head_r_mean'] for r in rows])),
                'head_r_q05_q50_q95': [
                    float(np.quantile(np.concatenate(
                        [mit_all_r[i] for i, r in enumerate(mit_rows)
                         if r['label'] == lab]), q)) for q in (.05, .5, .95)],
                'attn_entropy_mean': (float(np.mean(mit_ent_by_label[lab]))
                                      if mit_ent_by_label[lab] else None),
            }

    # ---------- PTB val patients (report-only) ----------
    try:
        from data.split_guard import compute_ptb_split
        beats, labels, rids = load_arrays('ptb')
        pv = np.asarray(compute_ptb_split()['val'], dtype=bool)
        beats = np.asarray(beats[pv], dtype=np.float32)
        labels = np.asarray(labels[pv])
        rids = np.asarray(rids[pv])
        order = np.argsort(rids, kind='stable')
        beats, labels, rids = beats[order], labels[order], rids[order]
        uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
        ptb_r_by_lab = {0: [], 1: []}
        for ui, rid in enumerate(uniq):
            s, e = first[ui], first[ui] + counts[ui]
            pr, att, x, m = head_r_windows(beats[s:e])
            lab = int(labels[s:e].max())
            ptb_r_by_lab[lab].append(pr)
        for lab in (0, 1):
            if ptb_r_by_lab[lab]:
                cat = np.concatenate(ptb_r_by_lab[lab])
                store[f'ptbval_label{lab}__head_r'] = cat
                groups[f'ptbval_label{lab}'] = {
                    'n_records': len(ptb_r_by_lab[lab]),
                    'head_r_q05_q50_q95': [float(np.quantile(cat, q))
                                           for q in (.05, .5, .95)],
                }
    except Exception as exc:  # PTB optional for diagnosis
        groups['ptbval_error'] = f'{type(exc).__name__}: {exc}'

    # ---------- MEETI val (training-style first-10 windows) ----------
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=0.2))
    pr_meeti = rec_probe.predict(
        [meeti['x_val'], meeti['mask_val']], batch_size=512, verbose=0)[:, 1]
    store['meetival__head_r'] = pr_meeti
    store['meetival__y'] = meeti['y_val']
    for lab in (0, 1):
        sel = meeti['y_val'] == lab
        if sel.any():
            groups[f'meetival_label{lab}'] = {
                'n_windows': int(sel.sum()),
                'head_r_q05_q50_q95': [float(np.quantile(pr_meeti[sel], q))
                                       for q in (.05, .5, .95)],
            }

    # ---------- fc1 domain shift: real vs MIT val ----------
    mu_mit, sd_mit = mit_fc1.mean(0), mit_fc1.std(0) + 1e-6
    for name in REAL:
        fc1 = store[f'real__{name}__fc1']
        z = (fc1.mean(0) - mu_mit) / sd_mit
        cov = np.cov(mit_fc1.T) + 1e-3 * np.eye(mit_fc1.shape[1])
        covinv = np.linalg.inv(cov)
        d = (fc1 - mit_fc1.mean(0)) @ covinv * (fc1 - mit_fc1.mean(0))
        mit_d = (mit_fc1 - mit_fc1.mean(0)) @ covinv * (mit_fc1 - mit_fc1.mean(0))
        groups[f'real_{name}']['fc1_z_shift'] = {
            'n_dims_gt_2': int((np.abs(z) > 2).sum()),
            'top5_dims': np.argsort(-np.abs(z))[:5].astype(int).tolist(),
            'top5_z': [round(float(v), 2) for v in z[np.argsort(-np.abs(z))[:5]]],
            'mahal_mean_real': float(d.sum(1).mean()),
            'mahal_mean_mit': float(mit_d.sum(1).mean()),
        }
        print(name, 'fc1 shift', groups[f'real_{name}']['fc1_z_shift'], flush=True)

    np.savez_compressed(OUT / 'arrays.npz', **store)
    with open(OUT / 'analysis.json', 'w') as f:
        json.dump(groups, f, indent=2, ensure_ascii=False, default=str)
    print('saved ->', OUT, flush=True)


if __name__ == '__main__':
    main()
