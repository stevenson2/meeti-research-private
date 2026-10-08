#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""按 curated manifest 评估仓库 pc_tools/ecg_dl 的前代模型 (同一测试标准)。

Manifest 规则: 同一微调谱系只选最后一个版本 (最高 version/latest mtime,
multi-seed 选最高 seed)。只有 250 拍输入且含 2 类输出的模型进入主测试;
3-beat(750) / PTBXL(1000) / SSL encoder(512) / 3-class gate 记录为 not_applicable。
输出: /home/devcontainers/meeti/runs/eval_repo_manifest/
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
from train_dual_head_small import (  # noqa: E402
    load_hard_beats, load_meeti_records,
)

T = 10


def ptb_record_data(mask):
    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats, dtype=np.float32)
    labels = np.asarray(labels)
    rids = np.asarray(rids)
    mask = np.asarray(mask, dtype=bool)
    beats, labels, rids = beats[mask], labels[mask], rids[mask]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
    n = len(uniq)
    x = np.zeros((n, T, beats.shape[1]), dtype=np.float32)
    xm = np.zeros((n, T, 1), dtype=np.float32)
    y = np.zeros(n, dtype=np.int32)
    for i in range(n):
        s, e = first[i], first[i] + counts[i]
        take = min(int(counts[i]), T)
        x[i, :take] = beats[s:s + take]
        xm[i, :take] = 1.0
        y[i] = int(labels[s:e].max())
    return x, xm, y


def model_predict_beat(model, x, batch=256):
    """返回每拍 p(abnormal) 一维数组; 多输出模型取第一个 2 类头。"""
    out = []
    for i in range(0, len(x), batch):
        y = model.predict(x[i:i + batch], batch_size=batch, verbose=0)
        if isinstance(y, (list, tuple)):
            y = y[0]
        out.append(np.asarray(y)[:, 1])
    return np.concatenate(out)


def record_scores(model, x, mask, batch=256):
    n = len(x)
    flat = x.reshape(-1, x.shape[-1], 1)
    p = []
    for i in range(0, len(flat), batch):
        y = model.predict(flat[i:i + batch], batch_size=batch, verbose=0)
        if isinstance(y, (list, tuple)):
            y = y[0]
        p.append(np.asarray(y)[:, 1])
    p = np.concatenate(p).reshape(n, T)
    m = mask[..., 0]
    denom = np.maximum(m.sum(axis=1), 1.0)
    mean_p = (p * m).sum(axis=1) / denom
    max_p = np.where(m > 0, p, -1.0).max(axis=1)
    return mean_p, max_p


def auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest',
                    default='/home/devcontainers/meeti/runs/all_models_manifest.json')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/eval_repo_manifest')
    ap.add_argument('--start', type=int, default=0)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    manifest = json.load(open(args.manifest))
    if args.limit:
        manifest = manifest[args.start:args.start + args.limit]
    elif args.start:
        manifest = manifest[args.start:]

    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=0.2))
    split = compute_ptb_split()
    ptb_data = {name: ptb_record_data(split[name])
                for name in ['train', 'val', 'test']}

    rows = []
    for entry in manifest:
        tag, model_path = entry['tag'], Path(entry['path'])
        print(f'=== {tag}: {model_path.name}', flush=True)
        if not model_path.exists():
            row = {'tag': tag, 'model': str(model_path), 'error': 'missing'}
            rows.append(row)
            continue
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
        except Exception as exc:
            row = {'tag': tag, 'model': str(model_path),
                   'error': f'{type(exc).__name__}: {exc}'}
            rows.append(row)
            print('  load FAIL', row['error'], flush=True)
            continue

        in_shape = getattr(model, 'input_shape', None)
        out_shapes = [tuple(o.shape) for o in model.outputs] \
            if hasattr(model, 'outputs') else [getattr(model, 'output_shape', None)]
        applicable = (in_shape is not None and len(in_shape) == 3
                      and in_shape[1] == 250
                      and any(len(s) == 2 and s[-1] == 2 for s in out_shapes))
        if not applicable:
            row = {'tag': tag, 'model': str(model_path),
                   'model_name': model.name,
                   'input_shape': list(in_shape) if in_shape else None,
                   'output_shapes': out_shapes,
                   'params': int(model.count_params()),
                   'not_applicable_reason':
                       'input is not 250-sample beat or no 2-class output'}
            rows.append(row)
            print('  NOT APPLICABLE', in_shape, out_shapes, flush=True)
            continue

        pv = model_predict_beat(model, xv)
        pr_real = model_predict_beat(model, x_real, batch=64)
        mm, mmax = record_scores(model, meeti['x_val'], meeti['mask_val'])
        ptb = {}
        for name, (x, xm, y) in ptb_data.items():
            mean_p, max_p = record_scores(model, x, xm)
            ptb[name] = {
                'n_records': int(len(x)),
                'label_dist': np.bincount(y, minlength=2).tolist(),
                'mean_p_auc': auc(y, mean_p),
                'max_p_auc': auc(y, max_p),
                'mean_p_mean': float(mean_p.mean()),
            }

        row = {
            'tag': tag,
            'model': str(model_path),
            'model_name': model.name,
            'input_shape': list(in_shape),
            'output_shapes': out_shapes,
            'params': int(model.count_params()),
            'head_b': {
                'public_val_auc': auc(yv, pv),
                'real_mean': float(pr_real.mean()),
                'real_frac_gt_0.5': float((pr_real > 0.5).mean()),
            },
            'meeti_record_mean_p_auc': auc(meeti['y_val'], mm),
            'meeti_record_max_p_auc': auc(meeti['y_val'], mmax),
            'ptb_record': ptb,
        }
        rows.append(row)
        print(json.dumps({
            'tag': tag, 'beat_auc': row['head_b']['public_val_auc'],
            'meeti_mean': row['meeti_record_mean_p_auc'],
            'ptb_val_mean': ptb['val']['mean_p_auc'],
            'ptb_test_mean': ptb['test']['mean_p_auc'],
            'real_mean': row['head_b']['real_mean'],
            'real_frac': row['head_b']['real_frac_gt_0.5'],
        }, indent=2), flush=True)
        with open(out / f'{tag}.json', 'w') as f:
            json.dump(row, f, indent=2)

    with open(out / 'summary.json', 'w') as f:
        json.dump(rows, f, indent=2)
    print('saved ->', out / 'summary.json', 'n=', len(rows), flush=True)


if __name__ == '__main__':
    main()
