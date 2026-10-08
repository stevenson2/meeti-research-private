#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate eval_init_vs_scratch JSONs and apply the STEP 2.4 gates.

Gates (new vs from-scratch control with same PTB config):
  G1 beatAUC(new) >= beatAUC(C) - 0.010
  G2 MEETI(new)   >= MEETI(C)   - 0.005
  G3 PTB (PTB runs only): val/test AUC >= C - 0.005 on both splits
  G4 AFE normal: mean_p <= 0.010 and frac>0.5 == 0
  G5 min(beatAUC, MEETI) drop <= 0.010
"""
import json
from pathlib import Path

ROOT = Path('/home/devcontainers/meeti')
EVAL_DIR = ROOT / 'runs/eval_init_vs_scratch'
CONTROL_DIR = ROOT / 'runs/eval_probe_pairs'
OUT_JSON = EVAL_DIR / 'summary_init_vs_scratch.json'
OUT_MD = EVAL_DIR / 'summary_init_vs_scratch.md'

RUNS = [
    'v3_init_clean_v3_noptb',
    'v3_init_clean_v3_ptb',
    'v3_init_clean_v4_ptb',
    'v3_init_clean_v5_ptb',
    'v3_init_fc1only_v3_ptb',
]
CHECKPOINTS = ['balanced', 'final']
CONTROL_NO_PTB = 'v3_seeded_lam0.2_balanced'
CONTROL_PTB = 'v3_seeded_lam0.2_ptb_balanced'


def load_json(path):
    with open(path) as f:
        return json.load(f)


def main():
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    control_no = load_json(CONTROL_DIR / f'{CONTROL_NO_PTB}.json')
    control_ptb = load_json(CONTROL_DIR / f'{CONTROL_PTB}.json')
    controls = {False: control_no, True: control_ptb}

    run_results = {}
    for run in RUNS:
        config = load_json(ROOT / f'runs/{run}/result.json')['config']
        use_ptb = bool(config['use_ptb'])
        c = controls[use_ptb]
        c_beat = c['head_b']['public_val_auc']
        c_meeti = c['head_r']['meeti_holdout_auc']
        c_min = min(c_beat, c_meeti)
        c_ptb_val = c['head_r']['ptb']['val']['auc']
        c_ptb_test = c['head_r']['ptb']['test']['auc']

        entry = {
            'run': run,
            'init_weights': config['init_weights'],
            'init_scope': config['init_scope'],
            'freeze_backbone': config['freeze_backbone'],
            'use_ptb': use_ptb,
            'control_tag': CONTROL_PTB if use_ptb else CONTROL_NO_PTB,
            'control_metrics': {
                'beat_auc': c_beat,
                'meeti_holdout_auc': c_meeti,
                'ptb_val_auc': c_ptb_val,
                'ptb_test_auc': c_ptb_test,
                'afe_mean': c['head_b']['real_mean'],
                'afe_frac_gt_0.5': c['head_b']['real_frac_gt_0.5'],
            },
            'checkpoints': {},
        }
        for ck in CHECKPOINTS:
            tag = f'{run}_ckpt_{ck}' if ck == 'balanced' else f'{run}_{ck}'
            path = EVAL_DIR / f'{tag}.json'
            if not path.exists():
                entry['checkpoints'][ck] = {'missing': True}
                continue
            e = load_json(path)
            hb, hr = e['head_b'], e['head_r']
            beat = hb['public_val_auc']
            meeti = hr['meeti_holdout_auc']
            afe_mean = hb['real_mean']
            afe_frac = hb['real_frac_gt_0.5']
            ptb_val = hr['ptb']['val']['auc']
            ptb_test = hr['ptb']['test']['auc']

            g1 = beat >= c_beat - 0.010
            g2 = meeti >= c_meeti - 0.005
            if use_ptb:
                g3 = (ptb_val is not None and ptb_test is not None
                      and ptb_val >= c_ptb_val - 0.005
                      and ptb_test >= c_ptb_test - 0.005)
            else:
                g3 = None  # not applicable
            g4 = afe_mean <= 0.010 and afe_frac == 0.0
            g5 = min(beat, meeti) >= c_min - 0.010

            gates = {
                'G1_beat': {'pass': bool(g1), 'delta': round(beat - c_beat, 6)},
                'G2_meeti': {'pass': bool(g2),
                             'delta': round(meeti - c_meeti, 6)},
                'G3_ptb': None if g3 is None else {
                    'pass': bool(g3),
                    'val_delta': round(ptb_val - c_ptb_val, 6),
                    'test_delta': round(ptb_test - c_ptb_test, 6),
                },
                'G4_afe': {'pass': bool(g4), 'mean': round(afe_mean, 6),
                           'frac_gt_0.5': afe_frac},
                'G5_min': {'pass': bool(g5),
                           'delta': round(min(beat, meeti) - c_min, 6)},
            }
            pass_all = g1 and g2 and (g3 is None or g3) and g4 and g5
            entry['checkpoints'][ck] = {
                'tag': tag,
                'metrics': {
                    'beat_auc': beat,
                    'meeti_holdout_auc': meeti,
                    'ptb_val_auc': ptb_val,
                    'ptb_test_auc': ptb_test,
                    'afe_mean': afe_mean,
                    'afe_frac_gt_0.5': afe_frac,
                    'min_beat_meeti': min(beat, meeti),
                },
                'gates': gates,
                'pass_all': bool(pass_all),
            }
        run_results[run] = entry

    any_balanced_pass = any(
        e['checkpoints'].get('balanced', {}).get('pass_all', False)
        for e in run_results.values())
    any_final_pass = any(
        e['checkpoints'].get('final', {}).get('pass_all', False)
        for e in run_results.values())

    summary = {
        'plan_gates': {
            'G1_beat_drop': 0.010, 'G2_meeti_drop': 0.005,
            'G3_ptb_drop': 0.005, 'G4_afe_mean_max': 0.010,
            'G4_afe_frac_gt_0.5_max': 0.0, 'G5_min_drop': 0.010,
        },
        'controls': {
            CONTROL_NO_PTB: control_no['head_b'] | control_no['head_r'],
            CONTROL_PTB: control_ptb['head_b'] | control_ptb['head_r'],
        },
        'runs': run_results,
        'any_balanced_pass_all': any_balanced_pass,
        'any_final_pass_all': any_final_pass,
    }
    with open(OUT_JSON, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # compact markdown table
    lines = [
        '# STEP 1 init-vs-scratch gate summary',
        '',
        '| run | ckpt | beat | MEETI | PTB va | PTB te | AFE mean | AFE frac | G1 | G2 | G3 | G4 | G5 | all |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|',
    ]
    for run, e in run_results.items():
        for ck in CHECKPOINTS:
            d = e['checkpoints'].get(ck, {})
            if not d or d.get('missing'):
                lines.append(f"| {run} | {ck} | missing |")
                continue
            m, g = d['metrics'], d['gates']
            g3 = 'NA' if g['G3_ptb'] is None else (
                'P' if g['G3_ptb']['pass'] else 'F')
            lines.append(
                f"| {run} | {ck} | {m['beat_auc']:.4f} | "
                f"{m['meeti_holdout_auc']:.4f} | {m['ptb_val_auc']:.4f} | "
                f"{m['ptb_test_auc']:.4f} | {m['afe_mean']:.4f} | "
                f"{m['afe_frac_gt_0.5']:.3f} | "
                f"{'P' if g['G1_beat']['pass'] else 'F'} | "
                f"{'P' if g['G2_meeti']['pass'] else 'F'} | {g3} | "
                f"{'P' if g['G4_afe']['pass'] else 'F'} | "
                f"{'P' if g['G5_min']['pass'] else 'F'} | "
                f"{'P' if d['pass_all'] else 'F'} |")
    lines += ['',
              f"- any balanced pass_all: {any_balanced_pass}",
              f"- any final pass_all: {any_final_pass}",
              '']
    OUT_MD.write_text('\n'.join(lines))

    print(json.dumps({
        'any_balanced_pass_all': any_balanced_pass,
        'any_final_pass_all': any_final_pass,
        'outputs': [str(OUT_JSON), str(OUT_MD)],
    }, indent=2))


if __name__ == '__main__':
    main()
