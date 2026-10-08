"""6-seed aggregate for the r9_form-ptbbal candidate (42-47)."""
import json

import numpy as np

SEEDS = [42, 43, 44, 45, 46, 47]
rows = [json.load(open(
    f'/home/devcontainers/meeti/runs/r9_form-ptbbal_seed{s}/result.json'))
    for s in SEEDS]


def ms(key):
    vals = [r['metrics'][key] for r in rows]
    return {'mean': float(np.mean(vals)),
            'sd': float(np.std(vals, ddof=1)), 'n': len(vals),
            'values': [float(v) for v in vals]}


paired = []
for s in SEEDS:
    try:
        base = json.load(open(
            f'/home/devcontainers/meeti/runs/'
            f'r8a_5s_student_seed{s}/result.json'))
        paired.append({
            'seed': s,
            'delta_meeti_val': rows[SEEDS.index(s)]['metrics']
            ['meeti_val_record_auc']
            - base['metrics']['meeti_val_record_auc'],
            'delta_ptb_val': rows[SEEDS.index(s)]['metrics']
            ['ptb_val_record_auc'] - base['metrics']['ptb_val_record_auc'],
            'delta_ptb_test': rows[SEEDS.index(s)]['metrics']
            ['ptb_test_record_auc'] - base['metrics']['ptb_test_record_auc']})
    except FileNotFoundError:
        paired.append({'seed': s, 'note': 'no r8a baseline seed'})

out = {
    'arm': 'r9_form-ptbbal', 'seeds': SEEDS,
    'meeti_val_record_auc': ms('meeti_val_record_auc'),
    'ptb_val_record_auc': ms('ptb_val_record_auc'),
    'ptb_test_record_auc': ms('ptb_test_record_auc'),
    'paired_vs_r8a': paired,
    'gates': {
        'meeti_val_ge_0.7773': ms('meeti_val_record_auc')['mean'] >= 0.7773,
        'ptb_val_ge_0.878': ms('ptb_val_record_auc')['mean'] >= 0.878,
        'ptb_test_ge_0.8267': ms('ptb_test_record_auc')['mean'] >= 0.8267},
    'note': '6-seed expansion of the R9 candidate; benchmarks: see '
            'runs/benchmark_v1 (42-44 aggregate reported there)'}
with open('/home/devcontainers/meeti/runs/'
          'r9_form-ptbbal_6seed_summary.json', 'w') as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: out[k] for k in
                  ['meeti_val_record_auc', 'ptb_val_record_auc',
                   'ptb_test_record_auc', 'gates']}, indent=1))
