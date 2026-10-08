#!/usr/bin/env python3
"""One-shot PLAN_STATE.json updater for the R12 wrap-up."""
import json
from pathlib import Path

P = Path('/home/devcontainers/meeti/PLAN_STATE.json')
s = json.loads(P.read_text(encoding='utf-8'))

s['step'] = 'R12-COMPLETE-DUAL-CAPABILITY-ROUND'
s['status'] = ('R12 done: bench v1.5 (spec@R90 + theta_R90tr columns, '
               'sec15 dual-capability gate) landed, anchors bit-exact CPU '
               '+ GPU drift<=5e-4; six arms resolved - T4 negative '
               '(monotone-map invariance theorem), T2 negative (S_beat '
               '70-75 but teacher device-noise transfer, gate FP 12-15 vs '
               'line 0), T3 DANN negative (seed42 +8pp trap, 1/3 seeds), '
               'T5 not promoted (val 2/3 but mean -1.9pp, fails sec5 gate), '
               'T6 GO (frozen-encoder beat readout S_beat 58.0 > locked '
               '55.9). Deploy candidate UNCHANGED (r9_form_ptbbal seed43 + '
               'locked companion beat + M4 gate). Flashing still paused '
               'pending physical unlock.')
s['next_action'] = ('NEXT ROUND QUEUE: 1) T6 formal arm (beat-aligned '
                    'pooling readout on the record encoder, MIT+INCART '
                    'train, deployment gate integration) - R12 probe GO '
                    '(S_beat 58.0 frozen-encoder); 2) device-aware KD if '
                    'beat distillation is retried (contrastive anchoring / '
                    'clean teacher - soft targets transfer teacher device '
                    'noise); 3) theta_R90tr operating point into deploy '
                    'selection reports; 4) flashing resume when user '
                    'physically unblocks (BOOT-held reset or re-plug), '
                    'then pio upload + pc_bench_driver.py COM19. Tag r12 '
                    'done; do not re-run frozen experiments.')

s['r12_round'] = {
    'report': '/home/devcontainers/meeti/docs/reports/SESSION_REPORT_R12.md',
    'infra': {
        'bench_version': 'v1.5',
        'new_columns': ['spec_at_R90 (eval-side sweep, ptb val/test)',
                        'theta_R90tr + R/spec@theta_R90tr (train-side '
                        'leak-free deployment anchor)'],
        'gate_clause': 'BENCHMARK_V1 sec15.3: candidates must report S + '
                       'S_beat + gate triple (device FP=0, MI val recall '
                       '26/26, PTB val normal FP); record-only/beat-only '
                       'no longer promotable alone',
        'verify': 'r10e_verify 6/6 PASS (CPU anchors bit-exact; GPU drift '
                  '<=5e-4 measured 7.7e-05/4.3e-05/3.3e-05/6.4e-05)'},
    'arms': {
        'T4_temperature': {
            'verdict': 'negative (mathematical invariance)',
            'detail': 'monotone map leaves theta=0.5 and theta_tr '
                      'workpoints bit-identical; per-seed T 0.115-0.824 '
                      'absorbs theta_tr bimodality; artifact '
                      'runs/r12_temp_scaling/'},
        'T2_beat_kd': {
            'verdict': 'negative (dual-capability gate)',
            'grid': {'a0.2_t2': {'S_beat': 74.9, 'beat_val': 0.8054,
                                 'device_frac': [0.757, 0.248]},
                     'a0.1_t2': {'S_beat': 70.1, 'beat_val': 0.7830,
                                 'gate_fp': [15, 15]},
                     'a0.2_t4': {'S_beat': 75.2, 'beat_val': 0.8057,
                                 'gate_fp': [12, 13],
                                 'device_frac': [0.052, 0.030]}},
            'lesson': 'teacher legacy_ensemble itself not device-clean '
                      '(p_mean 0.275, frac 0.075); soft-target KD '
                      'transfers device noise at any alpha/T; companion '
                      'beat head stays locked',
            'artifact': 'runs/r12_t2_kd_beat/'},
        'T3_dann': {
            'verdict': 'negative',
            'paired_ptb_val': {'lam0.3': [0.0804, -0.0484, -0.0984],
                               'lam0.1_s42': -0.0484, 'lam1.0_s42': -0.0418},
            'lesson': 'domain-invariant features conflict with PTB '
                      'gold-standard discrimination; single-seed +8pp '
                      'was a selection trap caught by the 3-seed rule',
            'artifact': 'runs/r12_dann_lam*_seed*/'},
        'T5_hardneg': {
            'verdict': 'not promoted',
            'paired_val': [0.0321, -0.1206, 0.0321],
            'mean_delta': {'val': -0.019, 'test': -0.014},
            'lesson': 'window prior (87.7% positive) is signal; prior-'
                      'perturbing sampling ops are all negative (R11 '
                      'resample + T5 oversample)',
            'artifact': 'runs/r9_form-ptbbal-hardneg_seed*/ + '
                        'runs/r12_t5_hardneg_specr90.json'},
        'T6_beat_readout': {
            'verdict': 'GO (feasibility probe)',
            'S_beat_style': 58.0,
            'detail': 'frozen record encoder + 5-segment pooling readout '
                      'beats locked dedicated beat head (55.9) with zero '
                      'encoder adaptation; formal arm registered',
            'artifact': 'runs/r12_t6_beat_readout/result.json'}},
    'deploy': 'unchanged: runs/deploy_candidate_r11/ (r9_form_ptbbal '
              'seed43 record + locked companion beat + M4 gate); flashing '
              'paused by user (physical BOOT needed)'}
P.write_text(json.dumps(s, indent=2, ensure_ascii=False), encoding='utf-8')
print('PLAN_STATE updated')
