#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把仓库 pc_tools/ecg_dl 里的前代 .h5 模型放到与双头/KD 完全相同的测试上。

对每个单头拍级模型计算:
  - 公共拍级 val AUC (distill_pilot_teacher_targets.npz x_val/y_val)
  - 真实 AFE holdout mean p / frac>0.5
  - MEETI subject-holdout 记录级 AUC:
      每条记录前 10 拍 -> beat p, 有效拍 mean-p 作为记录分数
  - PTB train/val/test 记录级金标准 AUC (记录标签=max拍标签):
      同上 mean-p 聚合, 同时输出 max-p 聚合作为参考
所有输出写项目外: /home/devcontainers/meeti/runs/eval_legacy_models/
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
DEFAULT_MODELS = {
    'repo_baseline_v3': PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v3.h5',
    'repo_baseline_v4': PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v4.h5',
    'repo_baseline_v5': PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v5.h5',
    'repo_distill_pilot_seed42': PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
        'best_resnet_large_distill_pilot_seed42.h5',
    'repo_distill_pilot_v2_seed42': PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
        'best_resnet_large_distill_pilot_v2_a020_t07_seed42.h5',
    'repo_capacity_w2_seed42': PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
        'best_resnet_capacity_cap_w2_seed42.h5',
    'repo_exp7c_v4': PROJ / 'pc_tools/ecg_dl/models/'
                      'best_resnet_large_exp7c_v4.h5',
    'repo_v6a5': PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
                  'best_v6a5_clean10k_hardnorm_anchor.h5',
}


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


def record_scores(model, x, mask, batch=256):
    """单头拍级模型: 对窗口内每个 beat 预测, 返回 mean-p 和 max-p。"""
    n = len(x)
    flat = x.reshape(-1, x.shape[-1], 1)
    p = []
    for i in range(0, len(flat), batch):
        p.append(model.predict(flat[i:i + batch], batch_size=batch, verbose=0)[:, 1])
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
    ap.add_argument('--models', default='default')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/eval_legacy_models')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    xb, yb, xv, yv, x_real = load_hard_beats()
    meeti = load_meeti_records(argparse.Namespace(meeti_weight=0.2))
    split = compute_ptb_split()
    ptb_data = {name: ptb_record_data(split[name])
                for name in ['train', 'val', 'test']}

    summary = []
    for tag, model_path in DEFAULT_MODELS.items():
        model_path = Path(model_path)
        print(f'=== {tag}: {model_path.name}', flush=True)
        model = tf.keras.models.load_model(str(model_path), compile=False)

        pv = model.predict(xv, batch_size=256, verbose=0)[:, 1]
        pr_real = model.predict(x_real, batch_size=64, verbose=0)[:, 1]
        beat_auc = auc(yv, pv)

        m_mean, m_max = record_scores(model, meeti['x_val'], meeti['mask_val'])
        meeti_mean_auc = auc(meeti['y_val'], m_mean)
        meeti_max_auc = auc(meeti['y_val'], m_max)

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
            'params': int(model.count_params()),
            'head_b': {
                'public_val_auc': beat_auc,
                'real_mean': float(pr_real.mean()),
                'real_frac_gt_0.5': float((pr_real > 0.5).mean()),
            },
            'meeti_record_mean_p_auc': meeti_mean_auc,
            'meeti_record_max_p_auc': meeti_max_auc,
            'ptb_record': ptb,
        }
        with open(out / f'{tag}.json', 'w') as f:
            json.dump(row, f, indent=2)
        summary.append(row)
        print(json.dumps({
            'tag': tag, 'beat_auc': beat_auc,
            'meeti_mean_p_auc': meeti_mean_auc,
            'ptb_val_mean_p_auc': ptb['val']['mean_p_auc'],
            'ptb_test_mean_p_auc': ptb['test']['mean_p_auc'],
            'real_mean': row['head_b']['real_mean'],
            'real_frac_gt_0.5': row['head_b']['real_frac_gt_0.5'],
        }, indent=2), flush=True)

    with open(out / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print('saved summary ->', out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
