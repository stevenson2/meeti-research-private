#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Direction 3: in-domain event-level specificity sweep.

Base detector (STEP6 V1 gate): window positive iff head_r >= tau_r AND
mean_last10(head_b) >= tau_b. Event confirmation on top (frozen protocol
skeleton: causal windows, cooldown 20):
  - k-of-5  : >= k positive windows within the 5-window causal group
  - sustained-S: last S consecutive windows all positive (then 1-of-5 group)

Data: PTB val (arrays from m4_ptbval_gate_check), MIT/INCART val + real AFE
(arrays from m4_head_r_diagnosis). Pure numpy.

Selection rule (frozen before looking at fold B / real):
  hard: real FP/hour == 0 on both recordings; MIT val recall >= 25/26;
        PTB val abnormal recall >= 0.90 (gate reference: 0.946)
  objective: minimize PTB-normal fold-A alarmed patients; ties -> smaller
        strictness (smaller k / S), then smaller tau_r
Verification (report-only): fold-B normals, full normal FP, all recall/FP.
"""
import json
from pathlib import Path

import numpy as np

T, COOLDOWN = 10, 20
OUT = Path('/home/devcontainers/meeti/runs/m4_event_specificity')
REAL_META = {'ecg_real_052': 183.0, 'rec_latest': 64.0}


def window_mean(p):
    n = len(p)
    out = np.zeros(n, dtype=np.float32)
    c = np.concatenate([[0.0], np.cumsum(p, dtype=np.float64)])
    for i in range(n):
        lo = max(0, i - T + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def alarm_triggers(cand):
    trig, last = [], -10 ** 9
    for i in range(len(cand)):
        lo = max(0, i - 4)
        if cand[lo:i + 1].any() and i - last >= COOLDOWN:
            trig.append(i)
            last = i
    return trig


def positives(hr, hbw, tau_r, tau_b):
    return (hr >= tau_r) & (hbw >= tau_b)


def confirm_flags(pos, mode, x):
    n = len(pos)
    if mode == 'k5':
        cnt = np.zeros(n, np.int32)
        acc = 0
        for i in range(n):
            acc += int(pos[i])
            if i >= 5:
                acc -= int(pos[i - 5])
            cnt[i] = acc
        return cnt >= x
    if mode == 'sus':
        run = 0
        f = np.zeros(n, bool)
        for i in range(n):
            run = run + 1 if pos[i] else 0
            if run >= x:
                f[i] = True
        return f
    raise ValueError(mode)


def seg_apply(arr_rid):
    rid = np.asarray(arr_rid)
    uniq, first, counts = np.unique(rid, return_index=True, return_counts=True)
    return [(str(u), int(f), int(c)) for u, f, c in zip(uniq, first, counts)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ptb = np.load('/home/devcontainers/meeti/runs/m4_ptbval_gate_check/'
                  'arrays.npz')
    dia = np.load('/home/devcontainers/meeti/runs/m4_head_r_diagnosis/'
                  'arrays.npz')

    data = {}
    ptb_segs_pre = seg_apply(ptb['rid'])
    data['ptbval'] = {
        'hr': ptb['head_r'],
        'hbw': np.concatenate(
            [window_mean(ptb['head_b'][s:s + c]) for _, s, c in ptb_segs_pre]),
        'rid': ptb['rid'], 'lab': ptb['rec_label']}
    mit_segs_pre = seg_apply(dia['mitval__rid'])
    data['mitval'] = {
        'hr': dia['mitval__head_r'],
        'hbw': np.concatenate(
            [window_mean(dia['mitval__head_b'][s:s + c])
             for _, s, c in mit_segs_pre]),
        'rid': dia['mitval__rid'], 'lab': dia['mitval__rec_label']}
    real = {n: {'hr': dia[f'real__{n}__head_r'],
                'hbw': window_mean(dia[f'real__{n}__head_b'])}
            for n in REAL_META}

    # PTB normal patients -> seeded fold A/B by sorted rid
    ptb_segs = seg_apply(data['ptbval']['rid'])
    ptb_lab = data['ptbval']['lab']
    norm_rids = sorted(r for r, s, c in ptb_segs if ptb_lab[s] == 0)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(norm_rids))
    fold_a = {norm_rids[i] for i in perm[:len(perm) // 2]}
    fold_b = set(norm_rids) - fold_a
    print(f'PTB normal patients: foldA={len(fold_a)} foldB={len(fold_b)}',
          flush=True)

    def run_config(tau_r, tau_b, mode, x):
        def flags(hr, hbw):
            return confirm_flags(positives(hr, hbw, tau_r, tau_b), mode, x)
        # MIT val recall
        mit_tp = mit_n = 0
        for r, s, c in seg_apply(data['mitval']['rid']):
            y = int(data['mitval']['lab'][s])
            if y == 1:
                mit_n += 1
                mit_tp += int(len(alarm_triggers(
                    flags(data['mitval']['hr'][s:s + c],
                          data['mitval']['hbw'][s:s + c]))) > 0)
        # PTB val
        ptb_tp = ptb_n = ptb_fp = 0
        fp_a = fp_b = 0
        for r, s, c in ptb_segs:
            y = int(ptb_lab[s])
            trig = len(alarm_triggers(flags(data['ptbval']['hr'][s:s + c],
                                            data['ptbval']['hbw'][s:s + c])))
            if y == 1:
                ptb_n += 1
                ptb_tp += int(trig > 0)
            else:
                ptb_fp += int(trig > 0)
                fp_a += int(r in fold_a and trig > 0)
                fp_b += int(r in fold_b and trig > 0)
        real_fp = max(len(alarm_triggers(flags(real[n]['hr'],
                                               real[n]['hbw'])))
                      / (REAL_META[n] / 3600.0) for n in REAL_META)
        return {
            'config': f'{mode}{x}_taur{tau_r}_taub{tau_b}',
            'mit_recall': mit_tp / mit_n,
            'ptb_recall': ptb_tp / ptb_n,
            'ptb_normal_fp': f'{ptb_fp}/20',
            'ptb_normal_fp_a': fp_a, 'ptb_normal_fp_b': fp_b,
            'real_max_fp_per_hour': float(real_fp),
        }

    rows = []
    for tau_r in [0.5, 0.6, 0.7, 0.8]:
        for tau_b in [0.1, 0.2, 0.3]:
            for mode, xs in [('k5', [1, 2, 3, 5]), ('sus', [1, 3, 5, 10, 20])]:
                for x in xs:
                    rows.append(run_config(tau_r, tau_b, mode, x))
                    print(rows[-1]['config'], rows[-1]['mit_recall'],
                          rows[-1]['ptb_recall'], rows[-1]['ptb_normal_fp'],
                          rows[-1]['real_max_fp_per_hour'], flush=True)

    # reference: STEP6 gate (k5 x=1, tau 0.5/0.2)
    ref = next(r for r in rows if r['config'] == 'k51_taur0.5_taub0.2')

    ok = [r for r in rows
          if r['real_max_fp_per_hour'] == 0.0
          and r['mit_recall'] >= 25 / 26
          and r['ptb_recall'] >= 0.90]
    ok.sort(key=lambda r: (r['ptb_normal_fp_a'], r['ptb_normal_fp'],
                           r['config']))
    selected = ok[0] if ok else None

    def fp_int(r):
        return int(r['ptb_normal_fp'].split('/')[0])

    passed = bool(
        selected
        and fp_int(selected) < fp_int(ref)
        and selected['ptb_normal_fp_b'] <= fp_int(ref) - fp_int(ref) // 2)
    result = {
        'base_gate_step6': ref,
        'selection_rule': 'hard: real FP==0, MIT recall>=25/26, PTB recall>='
                          '0.90; objective min foldA normal FP, tie -> '
                          'lexicographic config (less strict first)',
        'pass_rule': 'selected total normal FP < step6 gate AND foldB FP '
                     'improves by at least half the gate total',
        'folds': {'fold_a': sorted(fold_a), 'fold_b': sorted(fold_b)},
        'selected': selected,
        'pass': passed,
        'n_configs': len(rows),
        'rows': rows,
    }
    with open(OUT / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('REFERENCE(step6 gate):', json.dumps(ref), flush=True)
    print('SELECTED:', json.dumps(selected), flush=True)
    print('PASS:', passed, flush=True)


if __name__ == '__main__':
    main()
