#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M3 pre-registered probe gate.

Same record-level weak-label protocol on MIMIC subject split:
  - split subjects 80/20 (StratifiedShuffleSplit, random_state=42)
  - logistic regression on beat features, mean-p per record, record ROC
Gate (pre-registered):
  AUC_teacher_local >= AUC_student_fc1 + 0.02

Auxiliary:
  - locked dual-head model beat probabilities on real AFE normal holdout
  - student fc1 logistic-regression probe transferred to PTB record-level gold
    standard (teacher-local PTB windows are not available in this workspace)
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from train_dual_head_small import (  # noqa: E402
    MaskedAttentionPooling, load_hard_beats,
)

T = 10


def record_auc_from_beat_probs(probs, record_ids, y_beat):
    """Mean beat p per record -> record-level ROC."""
    ids = np.unique(record_ids)
    ymap = {int(r): int(y) for r, y in zip(record_ids, y_beat)}
    rp = np.array([probs[record_ids == i].mean() for i in ids])
    ry = np.array([ymap[int(i)] for i in ids])
    if len(np.unique(ry)) < 2:
        return None, rp, ry
    return float(roc_auc_score(ry, rp)), rp, ry


def record_dist(rid, y_beat):
    ymap = {int(r): int(y) for r, y in zip(rid, y_beat)}
    vals = np.array([ymap[int(r)] for r in np.unique(rid)])
    return np.bincount(vals, minlength=2).tolist()


def fit_predict_record_auc(feat_train, y_beat_train, rid_train,
                           feat_val, y_beat_val, rid_val):
    scaler = StandardScaler().fit(feat_train)
    lr = LogisticRegression(max_iter=1000)
    lr.fit(scaler.transform(feat_train), y_beat_train)
    p_val = lr.predict_proba(scaler.transform(feat_val))[:, 1]
    auc, rp, ry = record_auc_from_beat_probs(p_val, rid_val, y_beat_val)
    return auc, p_val, rp, ry


def ptb_record_data(mask):
    from data.split_guard import compute_ptb_split, load_arrays  # noqa: E402
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
    beat_feats = []
    beat_rids = []
    for i in range(n):
        s, e = first[i], first[i] + counts[i]
        y[i] = int(labels[s:e].max())
        take = min(int(counts[i]), T)
        b = beats[s:s + take]
        beat_feats.append(b)
        beat_rids.append(np.full(take, uniq[i], dtype=np.int64))
    return np.concatenate(beat_feats), np.concatenate(beat_rids), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--m3-dir',
                    default='/home/devcontainers/meeti/data/processed_small/m3_beat_centered')
    ap.add_argument('--beat-probe',
                    default='/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb/ckpt_balanced__beat.keras')
    ap.add_argument('--out-dir', default='/home/devcontainers/meeti/runs/m3_probe')
    args = ap.parse_args()

    m3 = Path(args.m3_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)

    meta = pd.read_csv(m3 / 'beat_meta.csv')
    if not len(meta):
        raise SystemExit('empty beat_meta')
    mask = np.load(m3 / 'student_beat_mask.npy').astype(bool)
    if not mask.all():
        print(f'filtering student-invalid beats: {int((~mask).sum())} removed',
              flush=True)
    meta = meta[mask].reset_index(drop=True)
    student_beats = np.load(m3 / 'student_beats_lead2.npy')[mask]
    teacher_local = np.load(m3 / 'teacher_local_768.npy')[mask]

    y_beat = meta['report_class'].map({'normal': 0, 'abnormal': 1}).to_numpy()
    if np.isnan(y_beat).any():
        bad = int(np.isnan(y_beat).sum())
        print(f'dropping unknown labels: {bad}', flush=True)
        keep = ~np.isnan(y_beat)
        meta = meta[keep].reset_index(drop=True)
        student_beats = student_beats[keep]
        teacher_local = teacher_local[keep]
        y_beat = y_beat[keep].astype(np.int64)
    else:
        y_beat = y_beat.astype(np.int64)
    rid = meta['study_id'].to_numpy(np.int64)

    # subject-level split (identical recipe to load_meeti_records)
    subjects = meta['subject_id'].to_numpy()
    uniq, subj_idx = np.unique(subjects, return_inverse=True)
    subj_y = np.zeros(len(uniq), dtype=np.int64)
    for yi, si in zip(y_beat, subj_idx):
        subj_y[si] = max(subj_y[si], int(yi))
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_subj, va_subj = next(sss.split(np.arange(len(uniq)), subj_y))
    tr = np.isin(subj_idx, tr_subj)
    va = np.isin(subj_idx, va_subj)
    print(f'subjects={len(uniq)} beats={len(meta)} train_beats={int(tr.sum())} '
          f'val_beats={int(va.sum())} '
          f'train_dist={np.bincount(y_beat[tr], minlength=2).tolist()} '
          f'val_dist={np.bincount(y_beat[va], minlength=2).tolist()}', flush=True)

    # student fc1 features from locked model
    beat_probe = tf.keras.models.load_model(args.beat_probe)
    encoder_model = beat_probe.get_layer('beat_encoder')
    stu = encoder_model.predict(student_beats, batch_size=256, verbose=0)
    print(f'student fc1 features {stu.shape}', flush=True)

    auc_student, p_stu, rp_stu, ry_stu = fit_predict_record_auc(
        stu[tr], y_beat[tr], rid[tr], stu[va], y_beat[va], rid[va])
    auc_teacher, p_tea, rp_tea, ry_tea = fit_predict_record_auc(
        teacher_local[tr], y_beat[tr], rid[tr],
        teacher_local[va], y_beat[va], rid[va])
    delta = (None if auc_teacher is None or auc_student is None
             else auc_teacher - auc_student)
    gate_pass = bool(auc_teacher is not None and auc_student is not None
                     and delta >= 0.02)

    # auxiliary: locked beat model AFE normal confidence
    _, _, _, _, x_real = load_hard_beats()
    p_real = beat_probe.predict(x_real, batch_size=64, verbose=0)[:, 1]

    # auxiliary: student fc1 LR probe transferred to PTB record gold standard
    scaler = StandardScaler().fit(stu[tr])
    lr = LogisticRegression(max_iter=1000).fit(scaler.transform(stu[tr]),
                                               y_beat[tr])
    ptb_aux = {}
    try:
        from data.split_guard import compute_ptb_split  # noqa: E402
        split = compute_ptb_split()
        for name, smask in [('train', split['train']),
                            ('val', split['val']),
                            ('test', split['test'])]:
            b, rids, yrec = ptb_record_data(smask)
            if len(b) == 0:
                continue
            f = encoder_model.predict(b, batch_size=256, verbose=0)
            p = lr.predict_proba(scaler.transform(f))[:, 1]
            ymap = {int(r): int(y) for r, y in zip(np.unique(rids), yrec)}
            ybeat = np.array([ymap[int(r)] for r in rids])
            auc, _, _ = record_auc_from_beat_probs(p, rids, ybeat)
            ptb_aux[name] = {
                'n_records': int(len(np.unique(rids))),
                'n_beats': int(len(b)),
                'record_auc': auc,
            }
    except Exception as exc:
        ptb_aux = {'error': f'{type(exc).__name__}: {exc}'}

    result = {
        'protocol': 'MIMIC weak-label subject-split record-level; LR on beat '
                    'features, mean-p per record',
        'split': {'n_subjects': int(len(uniq)),
                  'n_train_beats': int(tr.sum()),
                  'n_val_beats': int(va.sum()),
                  'n_train_records': int(len(np.unique(rid[tr]))),
                  'n_val_records': int(len(np.unique(rid[va]))),
                  'train_record_dist': record_dist(rid[tr], y_beat[tr]),
                  'val_record_dist': record_dist(rid[va], y_beat[va])},
        'student_fc1': {'dim': int(stu.shape[1]),
                        'record_auc': auc_student},
        'teacher_local': {'dim': int(teacher_local.shape[1]),
                          'record_auc': auc_teacher,
                          'center_step': 78},
        'gate': {
            'rule': 'AUC_teacher_local >= AUC_student_fc1 + 0.02',
            'delta': None if delta is None else round(float(delta), 6),
            'pass': gate_pass,
        },
        'aux_afe': {
            'n_real_normal_beats': int(len(x_real)),
            'mean_p': float(p_real.mean()),
            'frac_gt_0.5': float((p_real > 0.5).mean()),
        },
        'aux_ptb_student_fc1_probe': ptb_aux,
    }
    with open(out / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.savez_compressed(
        out / 'predictions.npz',
        stu=stu, teacher_local=teacher_local, y_beat=y_beat, rid=rid,
        tr=tr, va=va, p_stu=p_stu, rp_stu=rp_stu, ry_stu=ry_stu,
        p_tea=p_tea, rp_tea=rp_tea, ry_tea=ry_tea, p_real=p_real,
    )
    print(json.dumps(result, indent=2), flush=True)
    print('saved ->', out / 'result.json', flush=True)


if __name__ == '__main__':
    main()
