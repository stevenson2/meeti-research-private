#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""BENCHMARK_V1 unified evaluator (runs/BENCHMARK_V1.md).  Eval-only.

One protocol for every model in the project:
  splits/labels : frozen (MEETI subject split seed42, split_guard PTB,
                  real-device holdout) - identical across families
  families      : dual_head_10beat (native 10-beat windows + mask)
                  student_5s      (5-beat concat windows, stride 5)
  record score  : dual_head -> native head probability
                  student_5s -> top2mean of window probs (user-approved
                  2026-09-06); legacy mean reported alongside
  aggregation   : training models = 3 seeds (42/43/44) mean+/-sd
Outputs runs/benchmark_v1/{leaderboard.json, LEADERBOARD.md}.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import (ptb_5s, device_5s, record_scores,  # noqa: E402
                        MEETI_DIR)
from train_dual_head_small import (MaskedAttentionPooling,  # noqa: E402,F401
                                   record_windows, load_hard_beats)

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/benchmark_v1'
LOCKED = ROOT / 'runs/dual_head_v3_seeded_lam0.2_ptb'
T_BEATS = 10
SEEDS = [42, 43, 44]

# (leaderboard name, family, run dir template, ckpt name) + per-arm seeds
STUDENT_ARMS = [
    ('r8a_raw', 'runs/r8a_5s_student_seed{s}/ckpt_best__record5s.keras',
     SEEDS),
    ('r9_norm', 'runs/r9_norm_seed{s}/ckpt_best__record5s.keras', SEEDS),
    ('r9_form', 'runs/r9_form_seed{s}/ckpt_best__record5s.keras', SEEDS),
    ('r9_form_ptbbal', 'runs/r9_form-ptbbal_seed{s}/ckpt_best__record5s.keras',
     [42, 43, 44, 45, 46, 47]),   # candidate: 6-seed expansion
    ('r9_norm_w5', 'runs/r9_norm-w5_seed{s}/ckpt_best__record5s.keras',
     SEEDS),
]

ANCHORS = {'locked': {'meeti_val': 0.7773, 'ptb_val': 0.8880,
                      'ptb_test': 0.8367, 'beat_val': 0.7631}}

# Composite score S (BENCHMARK_V1 v1.1 add-on, original metrics untouched):
#   S = 100 * sum_i w_i * (AUC_i - 0.5) / 0.5   (50 = chance, 100 = perfect)
#   weights: ptb_val 0.35 (gold), ptb_test 0.40 (untouched holdout),
#            meeti_val 0.25 (weak-label reference domain)
#   device domain is NOT scored: record-head saturation on real-normal
#   windows is a known mode handled by the deployment AND-gate; it is
#   reported as a deployment-readiness disclosure instead.
COMPOSITE_W = {'ptb_val': 0.35, 'ptb_test': 0.40, 'meeti_val': 0.25}

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')

# legacy single-head beat models (family beat_250), formal lineages
LEGACY_MODELS = {
    'repo_baseline_v3': (PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v3.h5', ''),
    'repo_baseline_v4': (PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v4.h5', ''),
    'repo_baseline_v5': (PROJ / 'pc_tools/ecg_dl/models/'
                         'best_resnet_large_clean_baseline_v5.h5', ''),
    'repo_distill_pilot_seed42': (PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
                                  'best_resnet_large_distill_pilot_seed42.h5',
                                  ''),
    'repo_distill_pilot_v2_seed42': (
        PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
        'best_resnet_large_distill_pilot_v2_a020_t07_seed42.h5', ''),
    'repo_capacity_w2_seed42': (
        PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
        'best_resnet_capacity_cap_w2_seed42.h5', ''),
    'repo_exp7c_v4': (PROJ / 'pc_tools/ecg_dl/models/'
                      'best_resnet_large_exp7c_v4.h5',
                      'FLAG: exp7c lineage possible label leakage'),
    'repo_v6a5': (PROJ / 'pc_tools/ecg_dl/models/deploy_match/'
                  'best_v6a5_clean10k_hardnorm_anchor.h5', ''),
    # --- expanded fleet (user request: cnn_v2 / resnet L-M-S / ensemble) ---
    'repo_original_cnn_v2': (PROJ / 'pc_tools/ecg_dl/models/'
                             'final_cnn_v2.h5', ''),
    'repo_archived_cnn_v2': (PROJ / 'pc_tools/ecg_dl/models/archived/'
                             'best_model.h5', ''),
    'repo_resnet_medium': (PROJ / 'pc_tools/ecg_dl/models/archived/'
                           'best_resnet_medium.h5', ''),
    'repo_resnet_small': (PROJ / 'pc_tools/ecg_dl/models/archived/'
                          'best_small.h5', ''),
    'repo_cnn_m': (PROJ / 'pc_tools/ecg_dl/models/final_cnn_m.h5', ''),
    'new_m1_beat': (ROOT / 'runs/dual_head_small/beat_probe.keras', ''),
    'new_v3_noptb_beat': (ROOT / 'runs/dual_head_v3_seeded_lam0.2/'
                          'ckpt_balanced__beat.keras', ''),
    'new_v3_ptb_beat': (ROOT / 'runs/dual_head_v3_seeded_lam0.2_ptb/'
                        'ckpt_balanced__beat.keras',
                        'note: beat head of the locked dual-head (standalone '
                        'row; not an independent model)'),
}

# soft-voting ensembles: record/beat prob = mean over member outputs
ENSEMBLES = {
    'legacy_ensemble': [PROJ / 'pc_tools/ecg_dl/models/'
                        f'ensemble_seed{s}.h5' for s in (42, 123, 456)],
}


class SoftVotingEnsemble:
    """Duck-typed Keras model: predict() = mean of member probabilities."""

    def __init__(self, paths):
        self.members = [tf.keras.models.load_model(str(p), compile=False)
                        for p in paths]

    def predict(self, x, batch_size=512, verbose=0):
        ps = [m.predict(x, batch_size=batch_size, verbose=verbose)
              for m in self.members]
        return np.mean(ps, axis=0)

    def count_params(self):
        return sum(m.count_params() for m in self.members)

    @property
    def input_shape(self):
        return self.members[0].input_shape


def composite_score(row):
    """S from ptb_val/ptb_test/meeti_val AUCs (top2mean caliber)."""
    s = 0.0
    for k, w in COMPOSITE_W.items():
        v = row.get(k)
        v = v['mean'] if isinstance(v, dict) else v
        s += w * (v - 0.5) / 0.5
    return round(100 * s, 2)


def composite_beat(row):
    """S_beat: MIT+INCART leak-free beat-level sub-score (beat-capable
    families only).  50 = chance, 100 = perfect (val/test equal-weight
    average of the normalized AUCs)."""
    if 'beat_mi_val' in row and 'beat_mi_test' in row:
        v, t = row['beat_mi_val'], row['beat_mi_test']
        v = v['mean'] if isinstance(v, dict) else v
        t = t['mean'] if isinstance(t, dict) else t
        return round(100 * ((v - 0.5) / 0.5 + (t - 0.5) / 0.5) / 2, 2)
    return None


def mit_incart_beats():
    """Merged MIT-BIH+INCART beats + leak-free patient-level split masks.

    compute_mit_incart_split returns per-dataset masks aligned to each
    dataset's own arrays; we evaluate per dataset and merge predictions.
    """
    from data.split_guard import (compute_mit_incart_split,  # noqa: E402
                                  load_arrays)
    split = compute_mit_incart_split()
    out = {'stats': {k: v for k, v in split['stats'].items()
                     if isinstance(v, (int, float, str, bool))}}
    for key_mask, suffix in [('val_masks', 'val'), ('test_masks', 'test')]:
        xs, ys = [], []
        for tag in ['mit_bih', 'incart']:
            b, l, _ = load_arrays(tag)
            m = np.asarray(split[key_mask][tag], bool)
            xs.append(np.asarray(b, np.float32)[m])
            ys.append(np.asarray(l)[m].astype(np.int32))
        out[suffix] = (np.concatenate(xs), np.concatenate(ys))
    return out


def eval_beat_mit_incart(probe, seg=1):
    """Beat-level AUC on MIT+INCART val/test patients.

    seg=1: native unit = one 250-pt beat.  seg=3: 750-pt models - native
    unit = 3 consecutive beats, non-overlapping windows, window label =
    max of the 3 beats (native-form adaptation, documented in
    BENCHMARK_V1.md).  Leak-free patient-level split, seed 42.
    """
    split = mit_incart_beats()
    res = {'beat_mi_split': split['stats']}
    for name in ['val', 'test']:
        x, y = split[name]
        if seg > 1:
            n_win = len(x) // seg
            x = x[:n_win * seg].reshape(n_win, seg * 250)
            y = y[:n_win * seg].reshape(n_win, seg).max(axis=1)
        p = beat_probs(probe, x)
        res[f'beat_mi_{name}'] = float(roc_auc_score(y, p))
        res[f'beat_mi_{name}_n'] = int(len(y))
        res[f'beat_mi_{name}_dist'] = np.bincount(y, minlength=2).tolist()
        res[f'beat_mi_{name}_op'] = op_metrics(y, p, [0.5, 0.35])
    return res


def top2mean_by_record(p_win, win_rec, n_rec):
    """User-approved record score: mean of the top-2 window probs."""
    order = np.argsort(win_rec, kind='stable')
    p, r = p_win[order], win_rec[order]
    uniq, first, counts = np.unique(r, return_index=True,
                                    return_counts=True)
    out = np.zeros(n_rec)
    for i, (s, c) in enumerate(zip(first, counts)):
        v = np.sort(p[s:s + c])[::-1]
        out[i] = v[:min(2, len(v))].mean()
    return out


def youden_threshold(y, s):
    """Youden-J maximizing threshold on TRAIN-side scores (leak-free)."""
    y = np.asarray(y)
    s = np.asarray(s)
    order = np.argsort(s, kind='stable')
    ys, ss = y[order], s[order]
    P, N = int(ys.sum()), int(len(ys) - ys.sum())
    tp = P - np.cumsum(ys)          # scores >= cut
    fp = N - np.cumsum(1 - ys)
    j = tp / max(P, 1) - fp / max(N, 1)
    i = int(np.argmax(j[:-1])) + 1 if len(s) > 1 else 1
    return float(ss[min(i, len(ss) - 1)])


def spec_at_recall(y, s, r_target=0.90):
    """v1.5 spec@R90: max specificity s.t. recall >= r_target, sweeping
    thresholds over THIS domain's scores (partial-AUC-family trade-off
    visibility metric, not a deployment threshold)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    P = max(int(y.sum()), 1)
    best = None
    for th in np.unique(s):
        yp = s >= th
        tp = int((yp & (y == 1)).sum())
        rec = tp / P
        if rec >= r_target - 1e-9:
            fp = int((yp & (y == 0)).sum())
            tn = int((y == 0).sum()) - fp
            spec = tn / max(tn + fp, 1)
            if best is None or spec > best['spec']:
                best = {'spec': round(spec, 4),
                        'theta': round(float(th), 4),
                        'recall': round(rec, 4)}
    return best


def train_theta_at_recall(y, s, r_target=0.90):
    """v1.5 leak-free deployment anchor: largest TRAIN-side threshold with
    train recall >= r_target (= max train spec under the constraint)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    P = max(int(y.sum()), 1)
    best = None
    for th in np.unique(s):
        rec = int(((s >= th) & (y == 1)).sum()) / P
        if rec >= r_target - 1e-9:
            best = float(th)
    return best


def op_metrics(y, s, thetas):
    """v1.4 operating-point layer: recall/precision/F1/acc/bacc + AP.
    A non-float entry 'theta_tr' in thetas marks the train-tuned theta
    (fixed key name so multi-seed row merging stays consistent)."""
    y = np.asarray(y).astype(int)
    s = np.asarray(s, float)
    out = {}
    if len(np.unique(y)) > 1:
        out['avg_precision'] = float(average_precision_score(y, s))
    for th in thetas:
        if isinstance(th, tuple):
            th, key = th[0], th[1]
        else:
            key = th if isinstance(th, str) else f'{th:.3f}'
        yp = (s >= th).astype(int)
        tp = int(((yp == 1) & (y == 1)).sum())
        fp = int(((yp == 1) & (y == 0)).sum())
        fn = int(((yp == 0) & (y == 1)).sum())
        tn = int(((yp == 0) & (y == 0)).sum())
        rec = tp / max(tp + fn, 1)
        prec = tp / max(tp + fp, 1)
        out[key] = {
            'recall': round(rec, 4), 'precision': round(prec, 4),
            'f1': round(2 * prec * rec / max(prec + rec, 1e-9), 4),
            'acc': round((tp + tn) / max(len(y), 1), 4),
            'bacc': round((rec + tn / max(tn + fp, 1)) / 2, 4),
            'tp_fp_fn_tn': [tp, fp, fn, tn]}
    return out


def predict_windows(probe, x, batch_size=256):
    return probe.predict(x, batch_size=batch_size, verbose=0)[:, 1]


# ---------------- dual_head_10beat family ----------------

def ptb_records_10beat(split_mask):
    from data.split_guard import load_arrays  # noqa: E402
    beats, labels, rids = load_arrays('ptb')
    beats = np.asarray(beats[np.asarray(split_mask)], np.float32)
    labels = np.asarray(labels)[np.asarray(split_mask)]
    rids = np.asarray(rids)[np.asarray(split_mask)]
    order = np.argsort(rids, kind='stable')
    beats, labels, rids = beats[order], labels[order], rids[order]
    uniq, first, counts = np.unique(rids, return_index=True,
                                    return_counts=True)
    x = np.zeros((len(uniq), T_BEATS, 250), np.float32)
    m = np.zeros((len(uniq), T_BEATS, 1), np.float32)
    y = np.zeros(len(uniq), np.int32)
    for i in range(len(uniq)):
        s, e = int(first[i]), int(first[i]) + int(counts[i])
        y[i] = int(labels[s:e].max())
        take = min(int(counts[i]), T_BEATS)
        x[i, :take] = beats[s:s + take]
        m[i, :take] = 1.0
    return x, m, y


def meeti_val_records_10beat():
    """MEETI val-subject records, locked caliber (load_meeti_records)."""
    manifest = pd.read_csv(MEETI_DIR / 'processed_manifest.csv')
    beats = np.load(MEETI_DIR / 'beats.npy').astype(np.float32)

    def to_y(c):
        return 1 if c == 'abnormal' else (0 if c == 'normal' else -1)

    manifest['_y'] = manifest['report_class'].apply(to_y)
    manifest = manifest[manifest['_y'] >= 0].reset_index(drop=True)
    n_beats = manifest['n_beats'].astype(int).to_numpy()
    positions = np.concatenate([[0], np.cumsum(n_beats)[:-1]]).astype(np.int64)
    y = manifest['_y'].to_numpy(np.int32)
    subjects = manifest['subject_id'].to_numpy()
    x, mask, y_all, _ = record_windows(beats, positions, n_beats, y,
                                       np.zeros(len(manifest), np.float32))
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), np.int32)
    for yi, si in zip(y_all, subj_idx):
        subj_y[si] = max(subj_y[si], yi)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_s, va_s = next(sss.split(np.arange(len(uniq)), subj_y))
    va = np.isin(subj_idx, va_s)
    return x[va], mask[va], y_all[va]


def eval_locked():
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    probe = tf.keras.models.load_model(
        str(LOCKED / 'ckpt_balanced__record.keras'))
    beat_probe = tf.keras.models.load_model(
        str(LOCKED / 'ckpt_balanced__beat.keras'))
    from data.split_guard import compute_ptb_split  # noqa: E402
    split = compute_ptb_split()
    res = {'family': 'dual_head_10beat', 'seeds': 'single (best-of-4 ckpt)'}

    def auc(p, y):
        return float(roc_auc_score(y, p))

    x, m, y = ptb_records_10beat(np.asarray(split['train'], bool))
    ptr = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
    th_tr = youden_threshold(y, ptr)
    th_r90 = train_theta_at_recall(y, ptr, 0.90)
    res['op_metrics'] = {'theta_train_youden_ptb': round(th_tr, 4),
                         'theta_train_r90_ptb': round(th_r90, 4)}
    x, m, y = ptb_records_10beat(np.asarray(split['val'], bool))
    pv = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
    res['ptb_val'] = auc(pv, y)
    res['op_metrics']['ptb_val'] = op_metrics(
        y, pv, [0.5, (th_tr, 'theta_tr'), (th_r90, 'theta_r90tr')])
    res['op_metrics']['ptb_val']['spec_at_R90'] = spec_at_recall(y, pv, 0.90)
    x, m, y = ptb_records_10beat(np.asarray(split['test'], bool))
    pte = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
    res['ptb_test'] = auc(pte, y)
    res['op_metrics']['ptb_test'] = op_metrics(
        y, pte, [0.5, (th_tr, 'theta_tr'), (th_r90, 'theta_r90tr')])
    res['op_metrics']['ptb_test']['spec_at_R90'] = spec_at_recall(
        y, pte, 0.90)
    x, m, y = meeti_val_records_10beat()
    pm = probe.predict([x, m], batch_size=256, verbose=0)[:, 1]
    res['meeti_val'] = auc(pm, y)
    res['op_metrics']['meeti_val'] = op_metrics(y, pm, [0.5, (th_tr, 'theta_tr')])
    _, _, xv, yv, _ = load_hard_beats()
    bprob = beat_probe.predict(xv, batch_size=256, verbose=0)[:, 1]
    res['beat_val'] = auc(bprob, yv)
    res['op_metrics']['beat_val'] = op_metrics(yv, bprob, [0.5, 0.35])
    res.update(eval_beat_mit_incart(beat_probe))
    # device: stride-1 10-beat windows
    dev_stats = {}
    for name, d in device_5s().items():
        b = np.load(
            '/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/'
            f'data/real/{d["file"]}').astype(np.float32)
        n_win = len(b) - T_BEATS + 1
        idx = np.arange(n_win)[:, None] + np.arange(T_BEATS)[None, :]
        wx = np.zeros((n_win, T_BEATS, 250), np.float32)
        wm = np.zeros((n_win, T_BEATS, 1), np.float32)
        wx[:, :T_BEATS] = b[idx]
        wm[:, :T_BEATS] = 1.0
        p = probe.predict([wx, wm], batch_size=256, verbose=0)[:, 1]
        dev_stats[name] = {'n_windows': int(n_win),
                           'p_mean': float(p.mean()),
                           'frac_gt_0.5': float((p > 0.5).mean())}
    res['device_10beat'] = dev_stats
    res['ptb_val_legacy_anchor'] = ANCHORS['locked']['ptb_val']
    res['ptb_test_legacy_anchor'] = ANCHORS['locked']['ptb_test']
    return res


# ---------------- beat_250 family (legacy single-head beat models) -----

def beat_probs(model, beats, batch=512):
    if beats.ndim == 2:                      # (N,250) -> (N,250,1)
        beats = beats[..., None]
    p = []
    for i in range(0, len(beats), batch):
        p.append(model.predict(beats[i:i + batch],
                               batch_size=batch, verbose=0)[:, 1])
    return np.concatenate(p)


def topk2_by_record(p, rec_ids, n_rec):
    """top-2 mean of per-beat/window scores per record (same as top2mean)."""
    return top2mean_by_record(p, rec_ids, n_rec)


def eval_legacy(ckpt_path):
    """Legacy beat-only model on the unified benchmark.

    Native unit = one 250-pt beat.  Record contexts: first 10 beats
    (dual-head caliber).  Record score = top2mean of the 10 beat probs
    (unified caliber; legacy mean_p also reported).  Device: real beats
    scored individually (that is the model's native deployment unit).
    Accepts a checkpoint path or a preloaded model object (ensembles).
    """
    from data.split_guard import compute_ptb_split  # noqa: E402
    if isinstance(ckpt_path, (str, Path)):
        probe = tf.keras.models.load_model(str(ckpt_path), compile=False)
    else:
        probe = ckpt_path
    seg = 3 if int(probe.input_shape[1]) == 750 else 1
    split = compute_ptb_split()
    res = {'native_input_pts': int(probe.input_shape[1]),
           'op_metrics': {'theta_note': 'theta_fix=0.5 only (v1.4 '
                          'disclosure; train-side Youden deferred)'}}

    _, _, xv, yv, x_real = load_hard_beats()
    if seg == 1:
        pv = beat_probs(probe, xv)
        res['beat_val'] = float(roc_auc_score(yv, pv))
        pr = beat_probs(probe, x_real)
        res['device_beat'] = {'n_beats': int(len(pr)),
                              'p_mean': float(pr.mean()),
                              'frac_gt_0.5': float((pr > 0.5).mean())}
    else:
        n3 = (len(xv) // 3) * 3
        pv = beat_probs(probe, xv[:n3].reshape(-1, 750)).repeat(3)
        res['beat_val'] = float(roc_auc_score(yv[:n3], pv))
        nr3 = (len(x_real) // 3) * 3
        pr3 = beat_probs(probe, x_real[:nr3].reshape(-1, 750))
        pr = pr3.repeat(3)
        res['device_beat'] = {'n_beats': int(len(pr)),
                              'p_mean': float(pr.mean()),
                              'frac_gt_0.5': float((pr > 0.5).mean())}
    res.update(eval_beat_mit_incart(probe, seg=seg))

    from data.split_guard import load_arrays  # noqa: E402
    for name in ['val', 'test']:
        beats, labels, rids = load_arrays('ptb')
        mask = np.asarray(split[name], bool)
        beats = np.asarray(beats[mask], np.float32)
        labels = np.asarray(labels)[mask]
        rids = np.asarray(rids)[mask]
        order = np.argsort(rids, kind='stable')
        beats, labels, rids = beats[order], labels[order], rids[order]
        uniq, first, counts = np.unique(rids, return_index=True,
                                        return_counts=True)
        y = np.array([int(labels[int(first[i]):int(first[i])
                                  + int(counts[i])].max())
                      for i in range(len(uniq))], np.int32)
        # first 10 beats per record (seg=1) / first 9 beats = 3 x 3-beat
        # windows (seg=3, native form)
        ctx = 10 if seg == 1 else 9
        xs, rec_of_win = [], []
        for i in range(len(uniq)):
            take = min(int(counts[i]), ctx)
            take -= take % seg
            for j in range(0, take, seg):
                xs.append(beats[int(first[i]) + j:
                                 int(first[i]) + j + seg].reshape(-1))
                rec_of_win.append(i)
        X = np.stack(xs)
        p = beat_probs(probe, X)
        rec_of_win = np.asarray(rec_of_win)
        t2 = top2mean_by_record(p, rec_of_win, len(uniq))
        res[f'ptb_{name}'] = float(roc_auc_score(y, t2))
        res['op_metrics'][f'ptb_{name}'] = op_metrics(y, t2, [0.5])
        res['op_metrics'][f'ptb_{name}']['spec_at_R90'] = \
            spec_at_recall(y, t2, 0.90)
        cnt = np.bincount(rec_of_win, minlength=len(uniq)).astype(np.float64)
        mean_p = np.bincount(rec_of_win, weights=p,
                             minlength=len(uniq)) / np.maximum(cnt, 1)
        res[f'ptb_{name}_legacy_mean'] = float(roc_auc_score(y, mean_p))

    # MEETI val records: first 10 beats (seg=1) / first 9 = 3 windows
    x, m, y = meeti_val_records_10beat()
    if seg == 1:
        flat = x.reshape(-1, 250)
        p = beat_probs(probe, flat).reshape(x.shape[0], 10)
        valid = m[..., 0] > 0.5
        pm = np.where(valid, p, -1e9)            # NaN-safe top-2 selection
        top2 = np.sort(pm, axis=1)[:, ::-1][:, :2]
        top2m = np.where(top2[:, 1] <= -1e8, top2[:, 0],
                         top2.mean(axis=1))       # records with a single beat
        res['meeti_val'] = float(roc_auc_score(y, top2m))
        res['op_metrics']['meeti_val'] = op_metrics(y, top2m, [0.5])
        cnt = valid.sum(axis=1)
        mean_m = np.where(cnt > 0,
                          (pm * valid).sum(axis=1) / np.maximum(cnt, 1), 0.0)
        res['meeti_val_legacy_mean'] = float(roc_auc_score(y, mean_m))
    else:
        x9 = x[:, :9].reshape(x.shape[0], 3, 750)
        p = beat_probs(probe, x9.reshape(-1, 750)).reshape(x.shape[0], 3)
        t2 = top2mean_by_record(p.ravel(),
                                np.repeat(np.arange(len(y)), 3), len(y))
        res['meeti_val'] = float(roc_auc_score(y, t2))
        res['op_metrics']['meeti_val'] = op_metrics(y, t2, [0.5])
        res['meeti_val_legacy_mean'] = float(roc_auc_score(y, p.mean(axis=1)))
    return res


# ---------------- fm_teacher family ----------------

def eval_teacher():
    """ECG-FM (frozen) + PLS16-LR head fit on MIMIC train windows.

    Benchmark domains are single-lead, so the teacher runs in its
    deployable lead-II-fill form (5-beat concat windows, global z-score).
    Record score = top2mean of per-window probabilities (unified caliber);
    legacy mean also reported.
    """
    from r8_data_5s import mimic_5s  # noqa: E402
    M3 = ROOT / 'data/processed_small/m3_beat_centered'
    # head fit uses the BEAT-CENTERED window set (rows aligned with
    # teacher_local_768); the benchmark eval sets stay form-aligned
    mim = mimic_5s()
    teacher = np.load(M3 / 'teacher_local_768.npy', mmap_mode='r')
    emb_tr = np.asarray(teacher[mim['tr']], np.float64)
    y_tr = mim['y'][mim['tr']]
    from sklearn.cross_decomposition import PLSRegression  # noqa: E402
    from sklearn.linear_model import LogisticRegression  # noqa: E402
    from sklearn.preprocessing import StandardScaler  # noqa: E402
    scaler = StandardScaler().fit(emb_tr)
    pls = PLSRegression(n_components=16, scale=False, max_iter=2000)
    pls.fit(scaler.transform(emb_tr), y_tr)
    lr = LogisticRegression(max_iter=5000).fit(
        pls.transform(scaler.transform(emb_tr)), y_tr)
    head = {'scaler_mean': scaler.mean_.tolist(),
            'scaler_scale': scaler.scale_.tolist(),
            'pls_x': pls.x_rotations_.tolist(),
            'lr_coef': lr.coef_.tolist()[0], 'lr_intercept':
                float(lr.intercept_[0])}
    with open(OUT / 'teacher_head.json', 'w') as f:
        json.dump({'fit': 'MIMIC train-subject windows, full-12 '
                          'teacher_local_768, StandardScaler->PLS16->LR',
                   'n_fit': int(len(y_tr))}, f, indent=2)

    def win_p(emb):
        z = (emb - scaler.mean_) / scaler.scale_
        t = z @ pls.x_rotations_
        return lr.predict_proba(t)[:, 1]

    res = {'family': 'fm_teacher_lead2fill',
           'note': 'deployable lead-II-fill teacher view; full12 ceiling '
                   '0.8935 is on MIMIC val subjects (not comparable)',
           'op_metrics': {'theta_note': 'theta_fix=0.5 only '
                          '(train-side embeddings not cached; v1.4 '
                          'disclosure)'}}
    for name in ['meeti_val', 'ptb_val', 'ptb_test']:
        d = np.load(OUT / f'teacher_emb_{name}.npz')
        p = win_p(d['x_emb'].astype(np.float64))
        win_rec, rec_y = d['win_rec'], d['rec_y']
        n_rec = int(win_rec.max()) + 1
        t2 = top2mean_by_record(p, win_rec, n_rec)
        res[name] = float(roc_auc_score(rec_y, t2))
        res['op_metrics'][name] = op_metrics(rec_y, t2, [0.5])
        if name in ('ptb_val', 'ptb_test'):
            res['op_metrics'][name]['spec_at_R90'] = \
                spec_at_recall(rec_y, t2, 0.90)
        res[f'{name}_legacy_mean'] = float(roc_auc_score(
            rec_y, record_scores(p, win_rec, n_rec)))

    d = np.load(OUT / 'teacher_emb_device.npz')
    p = win_p(d['x_emb'].astype(np.float64))
    res['device_5beat'] = {'n_windows': int(len(p)),
                           'p_mean': float(p.mean()),
                           'frac_gt_0.5': float((p > 0.5).mean())}
    return res

def eval_student(ckpt_path):
    from r9_data import load_all  # noqa: E402
    probe = tf.keras.models.load_model(ckpt_path)
    data = load_all(norm=True, form=True)
    me, split = data['meeti'], data['ptb_splits']
    res = {}

    # v1.4: leak-free per-model operating threshold (PTB train side)
    dtr = ptb_5s(split['train'])
    ptr = top2mean_by_record(predict_windows(probe, dtr['x']),
                             dtr['win_rec'], dtr['n_records'])
    th_tr = youden_threshold(dtr['rec_y'], ptr)
    th_r90 = train_theta_at_recall(dtr['rec_y'], ptr, 0.90)
    res['op_metrics'] = {'theta_train_youden_ptb': round(th_tr, 4),
                         'theta_train_r90_ptb': round(th_r90, 4)}

    # MEETI val records: top2mean + legacy mean
    va_win = me['rec_va'][me['win_rec']]
    p_win = predict_windows(probe, me['x'][va_win])
    val_records = np.where(me['rec_va'])[0]
    remap = np.zeros(me['n_records'], np.int64)
    remap[val_records] = np.arange(len(val_records))
    rec_ids = remap[me['win_rec'][va_win]]
    me_s = top2mean_by_record(p_win, rec_ids, len(val_records))
    res['meeti_val'] = float(roc_auc_score(me['rec_y'][val_records], me_s))
    res['op_metrics']['meeti_val'] = op_metrics(
        me['rec_y'][val_records], me_s, [0.5, (th_tr, 'theta_tr')])
    res['meeti_val_legacy_mean'] = float(roc_auc_score(
        me['rec_y'][val_records],
        record_scores(p_win, rec_ids, len(val_records))))

    # PTB val/test: top2mean full-record, legacy mean, and @10beats
    for name in ['val', 'test']:
        d = ptb_5s(split[name])
        p = predict_windows(probe, d['x'])
        rec_s = top2mean_by_record(p, d['win_rec'], d['n_records'])
        res[f'ptb_{name}'] = float(roc_auc_score(d['rec_y'], rec_s))
        res['op_metrics'][f'ptb_{name}'] = op_metrics(
            d['rec_y'], rec_s,
            [0.5, (th_tr, 'theta_tr'), (th_r90, 'theta_r90tr')])
        res['op_metrics'][f'ptb_{name}']['spec_at_R90'] = \
            spec_at_recall(d['rec_y'], rec_s, 0.90)
        res[f'ptb_{name}_legacy_mean'] = float(roc_auc_score(
            d['rec_y'], record_scores(p, d['win_rec'], d['n_records'])))
        # context-matched: first 10 beats = first 2 windows of each record
        # (top2mean over 1-2 windows == their plain mean); win_rec is
        # already compact 0..n_rec-1 and every record has >=1 window
        counts = np.bincount(d['win_rec'], minlength=d['n_records'])
        first_idx = np.concatenate([[0], np.cumsum(counts)[:-1]])
        pos = np.arange(len(d['win_rec'])) - np.repeat(first_idx, counts)
        sel = pos < 2
        p2 = top2mean_by_record(p[sel], d['win_rec'][sel], d['n_records'])
        res[f'ptb_{name}_at_10beats'] = float(roc_auc_score(
            d['rec_y'], p2))

    # device: 5-beat stride-1 windows (disclosure)
    dev = {}
    for name, dd in data['device'].items():
        p = predict_windows(probe, dd['x'])
        dev[name] = {'n_windows': int(len(p)), 'p_mean': float(p.mean()),
                     'frac_gt_0.5': float((p > 0.5).mean())}
    res['device_5beat'] = dev
    return res


def aggregate(rows):
    """rows: list of per-seed dicts -> mean/sd per numeric field."""
    agg = {}
    for k in rows[0]:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        if len(vals) == len(rows):
            agg[k] = {'mean': float(np.mean(vals)),
                      'sd': float(np.std(vals, ddof=1)) if len(vals) > 1
                      else 0.0,
                      'values': [float(v) for v in vals]}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--legacy', default=None,
                    help='evaluate ONE legacy model in this fresh process '
                         'and write runs/benchmark_v1/legacy_row_<name>.json '
                         '(avoids in-process Keras/JIT state conflicts); '
                         'the full run merges these files')
    args, _ = ap.parse_known_args()
    if args.legacy:
        if args.legacy in ENSEMBLES:
            model = SoftVotingEnsemble(ENSEMBLES[args.legacy])
            flag = (f'soft-voting mean of {len(model.members)} members; '
                    'inference cost = {0}x single model'.format(
                        len(model.members)))
        else:
            path, flag = LEGACY_MODELS[args.legacy]
            model = path
        row = eval_legacy(model)
        row['family'] = ('beat_250_ensemble' if args.legacy in ENSEMBLES
                         else 'beat_250')
        if not hasattr(model, 'count_params'):
            model = tf.keras.models.load_model(str(model), compile=False)
        row['params'] = int(model.count_params())
        if flag:
            row['flag'] = flag
        with open(OUT / f'legacy_row_{args.legacy}.json', 'w') as f:
            json.dump(row, f, indent=2)
        print('legacy row saved:', args.legacy, flush=True)
        return

    OUT.mkdir(parents=True, exist_ok=True)
    board = {'spec': 'BENCHMARK_V1 v1.5 (runs/BENCHMARK_V1.md)',
             'record_score': {'student_5s': 'top2mean (legacy mean also '
                             'reported)', 'dual_head_10beat': 'native',
                             'beat_250': 'top2mean of first-10-beat probs '
                             '(legacy mean_p also reported)',
                             'fm_teacher_lead2fill': 'top2mean of per-window '
                             'PLS16-LR probs (legacy mean also reported)'},
             'composite': {'formula': 'S = 100*sum w_i*(AUC_i-0.5)/0.5; '
                           'S_beat = 100*[(AUC_mi_val-0.5)/0.5 + '
                           '(AUC_mi_test-0.5)/0.5]/2',
                           'weights': COMPOSITE_W,
                           'note': 'add-on only; original metrics unchanged; '
                                   'device domain excluded (deployment '
                                   'readiness reported separately); '
                                   'S_beat = MIT+INCART leak-free beat-level '
                                   'sub-score, beat-capable families only'},
             'models': {}}

    print('evaluating locked dual_head (read-only)...', flush=True)
    board['models']['locked_dual_head_v3_seeded_ptb'] = eval_locked()

    for name, tmpl, arm_seeds in STUDENT_ARMS:
        rows = []
        for s in arm_seeds:
            ck = ROOT / tmpl.format(s=s)
            if not ck.exists():
                print(f'skip {name} seed{s} (missing)', flush=True)
                continue
            print(f'evaluating {name} seed{s}...', flush=True)
            rows.append(eval_student(str(ck)))
        if rows:
            board['models'][name] = {
                'family': 'student_5s', 'seeds': arm_seeds,
                'aggregate': aggregate(rows), 'per_seed': rows}

    teacher_emb_ok = all(
        (OUT / f'teacher_emb_{n}.npz').exists()
        for n in ['meeti_val', 'ptb_val', 'ptb_test', 'device'])
    if teacher_emb_ok:
        print('evaluating fm_teacher (lead2-fill, PLS16-LR head)...',
              flush=True)
        board['models']['fm_teacher_lead2fill'] = eval_teacher()
    else:
        print('teacher embeddings missing -> teacher row skipped',
              flush=True)

    for name in list(LEGACY_MODELS) + list(ENSEMBLES):
        row_file = OUT / f'legacy_row_{name}.json'
        if row_file.exists():
            row = json.load(open(row_file))
            board['models'][f'legacy_{name}' if name in LEGACY_MODELS
                            else name] = row
            print(f'merged legacy {name}', flush=True)
        else:
            print(f'skip legacy {name} (no row file; run --legacy {name})',
                  flush=True)

    # literature rows (BENCHMARK_V1 sec 11, lit_<model>_<form> family):
    # self-contained JSONs written by scripts/r10_lit_*.py, merged here
    for row_file in sorted(OUT.glob('lit_row_*.json')):
        row = json.load(open(row_file))
        name = row_file.stem[len('lit_row_'):]
        board['models'][name] = row
        board['record_score'][name] = row.get('record_score', 'top2mean')
        print(f'merged lit {name}', flush=True)

    with open(OUT / 'leaderboard.json', 'w') as f:
        json.dump(board, f, indent=2, ensure_ascii=False)
    write_md(board)
    print('saved ->', OUT, flush=True)


def write_md(board):
    cols = ['meeti_val', 'ptb_val', 'ptb_test', 'ptb_val_at_10beats',
            'ptb_test_at_10beats', 'beat_mi_val', 'beat_mi_test',
            'meeti_val_legacy_mean', 'ptb_val_legacy_mean',
            'ptb_test_legacy_mean', 'beat_val']

    def fmt(cell):
        if isinstance(cell, dict) and 'mean' in cell:
            return f'{cell["mean"]:.4f}±{cell["sd"]:.4f}'
        if isinstance(cell, float):
            return f'{cell:.4f}'
        return '—'

    def comp(name, m, fn):
        row = m['aggregate'] if 'aggregate' in m else m
        v = fn(row)
        return f'**{v:.1f}**' if v is not None else '—'

    def ranks(fn):
        out = []
        for name, m in board['models'].items():
            row = m['aggregate'] if 'aggregate' in m else m
            v = fn(row)
            if v is not None:
                out.append((v, name))
        return sorted(out, reverse=True)

    ranked = ranks(composite_score)
    ranked_beat = ranks(composite_beat)

    lines = ['# LEADERBOARD — BENCHMARK_V1 v1.5',
             '',
             '生成：scripts/ecg_bench.py（只评测，零训练）。口径见',
             'runs/BENCHMARK_V1.md；窗类记录分数=top2mean（legacy mean 并列）；',
             '训练类模型=3 seeds mean±sd；S/S_beat 为附加一体化评分',
             '（原始指标原样保留）。',
             '',
             '## 一体化评分排名 S（记录级 = 100·Σw·(AUC−0.5)/0.5，w: '
             'ptb_val .35 / ptb_test .40 / meeti_val .25）',
             '',
             '| 排名 | 模型 | S |',
             '|---|---|---|']
    for i, (s, name) in enumerate(ranked, 1):
        lines.append(f'| {i} | {name} | **{s:.1f}** |')
    lines += ['',
              '## 拍级子评分排名 S_beat（MIT+INCART 无泄漏拍级，'
              '= 100·[(AUC_val−0.5)/0.5+(AUC_test−0.5)/0.5]/2；'
              '仅拍级 capable 家族）',
              '',
              '| 排名 | 模型 | S_beat |',
              '|---|---|---|']
    for i, (s, name) in enumerate(ranked_beat, 1):
        lines.append(f'| {i} | {name} | **{s:.1f}** |')
    lines += ['',
              '## 原始基准明细',
              '',
              '| 模型 | ' + ' | '.join(cols) + ' | S | S_beat |',
              '|---|' + '---|' * len(cols) + '---|---|']
    for name, m in board['models'].items():
        row = m['aggregate'] if 'aggregate' in m else m
        lines.append(f'| {name} | '
                     + ' | '.join(fmt(row.get(c)) for c in cols)
                     + f' | {comp(name, m, composite_score)} '
                     f'| {comp(name, m, composite_beat)} |')

    # v1.4 operating-point layer (record level on PTB test; beat level on
    # MIT+INCART test).  Multi-seed rows are seed-mean of each statistic;
    # sd appended for bacc/AP (small-negative-class caveat, v1.4.1 note);
    # theta_tr shown as median[min,max] across seeds (bimodal otherwise).
    def om_of(m):
        if 'aggregate' not in m:
            return m.get('op_metrics')
        oms = [s.get('op_metrics') for s in m.get('per_seed', [])
               if s.get('op_metrics')]
        if not oms:
            return None
        out = {}
        for k in oms[0]:
            v0 = oms[0][k]
            if not isinstance(v0, dict):
                vals = [o[k] for o in oms if k in o]
                out[k] = float(np.mean(vals)) if k.startswith('theta') else v0
                continue
            out[k] = {}
            for th in v0:
                st = v0[th]
                if not isinstance(st, dict):
                    out[k][th] = st
                    continue
                out[k][th] = {}
                for kk, vv in st.items():
                    if kk == 'tp_fp_fn_tn':
                        continue
                    try:
                        out[k][th][kk] = float(np.mean(
                            [o[k][th][kk] for o in oms
                             if th in o.get(k, {})]))
                    except (KeyError, TypeError):
                        out[k][th][kk] = vv
        return out

    def seed_op_sds(m, dom, th, key):
        if 'aggregate' not in m:
            return None
        vals = []
        for s in m.get('per_seed', []):
            try:
                vals.append(s['op_metrics'][dom][th][key])
            except (KeyError, TypeError):
                pass
        if len(vals) < 2:
            return None
        return float(np.std(vals, ddof=1))

    def g(om, dom, th, key):
        try:
            return om[dom][th][key]
        except (KeyError, TypeError):
            return None

    def theta_range(m, key='theta_train_youden_ptb'):
        if 'aggregate' not in m:
            th = m.get('op_metrics', {}).get(key)
            return (f'{th:.3f}' if isinstance(th, float) else None)
        vals = [s['op_metrics'][key] for s in m.get('per_seed', [])
                if s.get('op_metrics') and key in s['op_metrics']]
        if not vals:
            return None
        if len(vals) == 1:
            return f'{vals[0]:.3f}'
        return (f'{np.median(vals):.3f}'
                f'[{min(vals):.2f},{max(vals):.2f}]')

    lines += ['',
              '## 操作点指标层 v1.5（PTB test 记录级 + MIT+INCART test 拍级；'
              'θ=0.5 固定 + θ_tr=训练侧 Youden（多 seed 中位数[最小,最大]）+ '
              'spec@R90=该域分数扫描 R≥0.90 下最大 spec（partial-AUC 族 '
              'trade-off 指标）+ θ_R90tr=训练侧 R≥0.90 最大 θ（leak-free '
              '部署锚）；AP=平均精度；多 seed 行为逐 seed 均值）',
              '',
              '注意：PTB test 记录级阴性类仅 20 条（75+/20−），specificity/bAcc '
              '每条记录权重 5%；±1 条记录 = Δspec 0.05。bAcc/AP 差异小于 '
              '1 个跨 seed sd 不作结论。固定 θ 的 R 列受校准主导，配 spec 列'
              '阅读。legacy/fm_teacher 行仅报 spec@R90（θ_R90tr 与 v1.4 的 '
              'θ_tr 同因缓办，披露）；lit 行不强制。',
              '',
              '| 模型 | R@.5 | spec@.5 | F1@.5 | bAcc@.5 | AP | θ_tr | '
              'R@θtr | spec@θtr | spec@R90 val | spec@R90 test | θ_R90tr | '
              'R@θR90tr | spec@θR90tr | 拍R@.5 | 拍F1@.5 | 拍R@.35 |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|'
              '---|---|---|']
    for name, m in board['models'].items():
        om = om_of(m)
        if not om:
            continue
        r5 = g(om, 'ptb_test', '0.500', 'recall')
        f5 = g(om, 'ptb_test', '0.500', 'f1')
        b5 = g(om, 'ptb_test', '0.500', 'bacc')
        b5sd = seed_op_sds(m, 'ptb_test', '0.500', 'bacc')
        ap = (om.get('ptb_test') or {}).get('avg_precision')
        apsd = seed_op_sds(m, 'ptb_test', '0.500', 'avg_precision')
        th = theta_range(m)
        rtr = g(om, 'ptb_test', 'theta_tr', 'recall')
        ftr = g(om, 'ptb_test', 'theta_tr', 'f1')
        # spec: from confusion counts of the mean row is unavailable
        # (counts not merged); use mean of per-seed spec instead
        brow = m if 'aggregate' not in m else {}

        def spec_of(dom, thk):
            if 'aggregate' not in m:
                c = (m.get('op_metrics', {}).get(dom, {})
                     .get(thk, {}).get('tp_fp_fn_tn'))
                if c:
                    tn, fp = c[3], c[1]
                    return tn / max(tn + fp, 1)
                return None
            vals = [s['op_metrics'][dom][thk] for s in m.get('per_seed', [])
                    if s.get('op_metrics') and thk in s['op_metrics'].get(dom, {})]
            sp = []
            for o in vals:
                c = o.get('tp_fp_fn_tn')
                if c:
                    sp.append(c[3] / max(c[3] + c[1], 1))
            return float(np.mean(sp)) if sp else None
        sp5 = spec_of('ptb_test', '0.500')
        sptr = spec_of('ptb_test', 'theta_tr')
        sr90v = g(om, 'ptb_val', 'spec_at_R90', 'spec')
        sr90vsd = seed_op_sds(m, 'ptb_val', 'spec_at_R90', 'spec')
        sr90t = g(om, 'ptb_test', 'spec_at_R90', 'spec')
        sr90tsd = seed_op_sds(m, 'ptb_test', 'spec_at_R90', 'spec')
        th_r90 = theta_range(m, 'theta_train_r90_ptb')
        r90tr = g(om, 'ptb_test', 'theta_r90tr', 'recall')
        sp90tr = spec_of('ptb_test', 'theta_r90tr')
        bop = brow.get('beat_mi_test_op') or {}

        def bg(k, kk):
            try:
                return bop[k][kk]
            except (KeyError, TypeError):
                return None
        bm = bg('0.500', 'recall')
        bf = bg('0.500', 'f1')
        b35 = bg('0.350', 'recall')

        def s4(v, sd=None):
            if not isinstance(v, float):
                return '—'
            return f'{v:.3f}±{sd:.3f}' if isinstance(sd, float) else f'{v:.3f}'
        lines.append(f'| {name} | {s4(r5)} | {s4(sp5)} | {s4(f5)} | '
                     f'{s4(b5, b5sd)} | {s4(ap, apsd)} | {th or "—"} | '
                     f'{s4(rtr)} | {s4(sptr)} | '
                     f'{s4(sr90v, sr90vsd)} | {s4(sr90t, sr90tsd)} | '
                     f'{th_r90 or "—"} | {s4(r90tr)} | {s4(sp90tr)} | '
                     f'{s4(bm)} | {s4(bf)} | {s4(b35)} |')

    locked_dev = board['models']['locked_dual_head_v3_seeded_ptb'][
        'device_10beat']
    lines += ['',
              '设备域（披露，非 gate，不入 S）：locked 10 拍窗 p_mean '
              f'{locked_dev["ecg_real_052"]["p_mean"]:.3f}/'
              f'{locked_dev["rec_latest"]["p_mean"]:.3f}，frac=1.0；'
              '其余模型见 leaderboard.json。参照系更正：locked 记录头本就 '
              'frac=1.0，"≈0"属拍头 AFE holdout；记录头级部署结论须由'
              '伴生拍头+M4 门控给出（deployment readiness 未达成前，'
              '所有行仅作研究排名）。',
              '',
              '参考线（天花板，非可比，不入 S）：教师 full12 PLS16 0.8935 / '
              'lead2 填充 0.8504（MIMIC val subjects，另一评测集）。']
    (OUT / 'LEADERBOARD.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:24]), flush=True)


if __name__ == '__main__':
    main()
