#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4: real AFE long-recording event-level FP/hour and MIT/INCART val recall.

Locked model: dual-head v3 from-scratch +PTB.
Controls      : repo exp7c TFLite models (possible-leakage label only).

Alarm protocol (frozen): causal 10-beat sliding window (step 1 beat),
1-of-5 positive window triggers an alarm candidate, cooldown 20 beats.
Head_R uses the locked record head. TFLite controls have no record head, so
their window score is the mean of the last 10 per-beat TFLite probabilities.
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
from train_dual_head_small import MaskedAttentionPooling  # noqa: E402
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa: E402

T = 10
MODEL_ROOT = PROJ / 'pc_tools/ecg_dl/models'
REAL_RECORDS = [
    {'name': 'ecg_real_052',
     'beats': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_exp7c.npy',
     'dur_s': 183.0},
    {'name': 'rec_latest',
     'beats': PROJ / 'pc_tools/ecg_dl/data/real/real_normal_beats_rec_latest.npy',
     'dur_s': 64.0},
]
TFLITE_CONTROLS = [
    {'tag': 'exp7c_clean_v6_qat_int8',
     'path': MODEL_ROOT / 'ecg_model_exp7c_clean_v6_qat_int8.tflite'},
    {'tag': 'exp7c_ecgfounder_v5_qat_int8',
     'path': MODEL_ROOT / 'ecg_model_exp7c_ecgfounder_v5_qat_int8.tflite'},
]
THRESH = 0.5
CONFIRM_WINDOWS = 5
COOLDOWN_BEATS = 20


def load_dual_probes(beat_path, rec_path):
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    beat_probe = tf.keras.models.load_model(beat_path)
    rec_probe = tf.keras.models.load_model(rec_path)
    return beat_probe, rec_probe


def dual_beat_p(beat_probe, beats):
    return beat_probe.predict(beats[..., None], batch_size=256, verbose=0)[:, 1]


def dual_window_scores(rec_probe, beats, batch_size=256):
    n = len(beats)
    scores = np.zeros(n, dtype=np.float32)
    xb = np.zeros((batch_size, T, beats.shape[1], 1), dtype=np.float32)
    mb = np.zeros((batch_size, T, 1), dtype=np.float32)
    for s in range(0, n, batch_size):
        e = min(n, s + batch_size)
        cur = e - s
        xb[:cur] = 0.0
        mb[:cur] = 0.0
        for j in range(cur):
            i = s + j
            lo = max(0, i - T + 1)
            take = i - lo + 1
            xb[j, -take:] = beats[lo:i + 1, :, None]
            mb[j, -take:] = 1.0
        p = rec_probe.predict([xb[:cur], mb[:cur]], batch_size=cur,
                              verbose=0)[:, 1]
        scores[s:e] = p
    return scores


def make_tflite_pred(model_path):
    it = tf.lite.Interpreter(model_path=str(model_path))
    it.allocate_tensors()
    ind = it.get_input_details()[0]
    outd = it.get_output_details()[0]
    iscale, izp = ind['quantization']
    oscale, ozp = outd['quantization']

    def pred(beats):
        beats = np.asarray(beats, dtype=np.float32)
        out = np.empty(len(beats), dtype=np.float32)
        for i in range(len(beats)):
            xq = np.clip(np.round(beats[i] / iscale + izp), -128, 127).astype(np.int8)
            it.set_tensor(ind['index'], np.reshape(xq, (1, -1, 1)))
            it.invoke()
            q = it.get_tensor(outd['index']).astype(np.float32)
            out[i] = float(np.clip((q[0, 1] - ozp) * oscale, 0.0, 1.0))
        return out
    return pred


def beat_to_window_scores(beat_p, n=None):
    """Causal mean of last 10 per-beat probabilities (TFLite surrogate)."""
    if n is None:
        n = len(beat_p)
    scores = np.zeros(n, dtype=np.float32)
    for i in range(n):
        lo = max(0, i - T + 1)
        scores[i] = beat_p[lo:i + 1].mean()
    return scores


def alarm_triggers(scores, threshold=THRESH, cooldown=COOLDOWN_BEATS):
    """1-of-5 causal group: any positive window in [i-4, i] is a candidate."""
    n = len(scores)
    triggers = []
    last = -10 ** 9
    for i in range(n):
        lo = max(0, i - CONFIRM_WINDOWS + 1)
        if np.any(scores[lo:i + 1] >= threshold):
            if i - last >= cooldown:
                triggers.append(int(i))
                last = i
    return triggers


def record_event_recall(record_groups, score_fn, tag):
    rows = []
    for rid, rec in record_groups.items():
        scores = score_fn(rec['beats'])
        trig = alarm_triggers(scores)
        rows.append({
            'record_id': rid,
            'tag': tag,
            'n_beats': int(len(rec['beats'])),
            'label': int(rec['label']),
            'triggers': int(len(trig)),
            'score_mean': float(np.mean(scores)),
            'score_max': float(np.max(scores)),
            'score_frac_gt_0.5': float(np.mean(scores >= THRESH)),
        })
    abn = [r for r in rows if r['label'] == 1]
    norm = [r for r in rows if r['label'] == 0]
    tp = sum(1 for r in abn if r['triggers'] > 0)
    fp = sum(1 for r in norm if r['triggers'] > 0)
    return {
        'tag': tag,
        'n_records': len(rows),
        'n_abnormal': len(abn),
        'n_normal': len(norm),
        'tp_records': int(tp),
        'fn_records': int(len(abn) - tp),
        'fp_records': int(fp),
        'event_recall': float(tp / len(abn)) if abn else None,
        'rows': rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--beat-probe',
                    default='/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb/ckpt_balanced__beat.keras')
    ap.add_argument('--record-probe',
                    default='/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb/ckpt_balanced__record.keras')
    ap.add_argument('--out-dir',
                    default='/home/devcontainers/meeti/runs/m4_afe_event_eval')
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    beat_probe, rec_probe = load_dual_probes(args.beat_probe, args.record_probe)
    print('loaded dual probes', flush=True)

    # ---------- real AFE long recordings ----------
    real_results = []
    for rec in REAL_RECORDS:
        beats = np.load(rec['beats']).astype(np.float32)
        dur_h = rec['dur_s'] / 3600.0
        bp = dual_beat_p(beat_probe, beats)
        rs = dual_window_scores(rec_probe, beats)
        trig = alarm_triggers(rs)
        real_results.append({
            'source': rec['name'],
            'model': 'locked_dual_head_v3_ptb',
            'possible_leakage': False,
            'n_beats': int(len(beats)),
            'dur_s': rec['dur_s'],
            'head_b_mean_p': float(bp.mean()),
            'head_b_frac_gt_0.5': float(np.mean(bp > THRESH)),
            'head_r_mean_p': float(rs.mean()),
            'head_r_frac_gt_0.5': float(np.mean(rs >= THRESH)),
            'alarms': int(len(trig)),
            'fp_per_hour': float(len(trig) / dur_h),
        })

    for ctl in TFLITE_CONTROLS:
        pred = make_tflite_pred(ctl['path'])
        for rec in REAL_RECORDS:
            beats = np.load(rec['beats']).astype(np.float32)
            bp = pred(beats)
            rs = beat_to_window_scores(bp)
            trig = alarm_triggers(rs)
            real_results.append({
                'source': rec['name'],
                'model': ctl['tag'],
                'possible_leakage': True,
                'n_beats': int(len(beats)),
                'dur_s': rec['dur_s'],
                'head_b_mean_p': float(bp.mean()),
                'head_b_frac_gt_0.5': float(np.mean(bp > THRESH)),
                'window_surrogate_mean_p': float(rs.mean()),
                'window_surrogate_frac_gt_0.5': float(np.mean(rs >= THRESH)),
                'alarms': int(len(trig)),
                'fp_per_hour': float(len(trig) / (rec['dur_s'] / 3600.0)),
            })
        print('real AFE done', ctl['tag'], flush=True)

    # ---------- MIT/INCART val event recall ----------
    split = compute_mit_incart_split()
    val_groups = {}
    for dataset in ['mit_bih', 'incart']:
        beats, labels, rids = load_arrays(dataset)
        vm = np.asarray(split['val_masks'][dataset], dtype=bool)
        beats = np.asarray(beats[vm], dtype=np.float32)
        labels = np.asarray(labels[vm])
        rids = np.asarray(rids[vm])
        order = np.argsort(rids, kind='stable')
        beats, labels, rids = beats[order], labels[order], rids[order]
        uniq, first, counts = np.unique(rids, return_index=True, return_counts=True)
        for ui, rid in enumerate(uniq):
            s, e = first[ui], first[ui] + counts[ui]
            val_groups[int(rid)] = {
                'beats': beats[s:e],
                'label': int(labels[s:e].max()),
            }
    print('val groups', len(val_groups), flush=True)

    recall_results = []
    # dual record head
    def dual_score_fn(beats):
        return dual_window_scores(rec_probe, beats)
    recall_results.append(record_event_recall(
        val_groups, dual_score_fn, 'locked_dual_head_v3_ptb'))
    print('dual val recall done', flush=True)

    # dual beat head auxiliary
    def dual_beat_score_fn(beats):
        bp = dual_beat_p(beat_probe, beats)
        return beat_to_window_scores(bp)
    recall_results.append(record_event_recall(
        val_groups, dual_beat_score_fn, 'locked_dual_head_v3_ptb_beat_surrogate'))
    print('dual beat val recall done', flush=True)

    # TFLite controls (possible-leakage)
    for ctl in TFLITE_CONTROLS:
        pred = make_tflite_pred(ctl['path'])
        def tflite_score_fn(beats, _pred=pred):
            return beat_to_window_scores(_pred(beats))
        recall_results.append(record_event_recall(
            val_groups, tflite_score_fn, ctl['tag']))
        print('tflite val recall done', ctl['tag'], flush=True)

    result = {
        'protocol': {
            'window': 'causal last-10-beat sliding window, step 1 beat',
            'trigger': '1 positive window in 5-window causal group',
            'cooldown_beats': COOLDOWN_BEATS,
            'threshold': THRESH,
            'head_r_source': 'locked record probe for dual model; '
                             'mean last-10 per-beat p for TFLite controls',
        },
        'real_afe_fp_per_hour': real_results,
        'mit_incart_val_event_recall': [
            {k: v for k, v in r.items() if k != 'rows'} for r in recall_results
        ],
        'detail_rows': {r['tag']: r['rows'] for r in recall_results},
        'controls_note': 'TFLite controls use exp7c lineage and are '
                         'possible-leakage; dual-head locked model is clean.',
    }
    with open(out / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False)[:4000], flush=True)
    print('saved ->', out / 'result.json', flush=True)


if __name__ == '__main__':
    main()
