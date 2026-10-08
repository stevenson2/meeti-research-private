"""P1 probe: quantify per-window scale/offset stats across all domains.

Quantifies the normalization mismatch hypothesized in
runs/R9_LITERATURE_GAP_ANALYSIS.md gap #1: MIMIC 5s windows are raw mV,
MEETI/PTB/device beats come from the deploy chain with its own scaling.
"""
import json
import sys

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from r8_data_5s import mimic_5s, meeti_5s, ptb_5s, device_5s  # noqa: E402
from data.split_guard import compute_ptb_split  # noqa: E402

out = {}


def stats(x, name):
    sd = x.std(axis=1)
    mu = x.mean(axis=1)
    out[name] = {
        'n': int(len(x)),
        'per_win_std_p50': float(np.median(sd)),
        'per_win_std_p10': float(np.percentile(sd, 10)),
        'per_win_std_p90': float(np.percentile(sd, 90)),
        'per_win_mean_p50': float(np.median(mu)),
        'per_win_mean_p10': float(np.percentile(mu, 10)),
        'per_win_mean_p90': float(np.percentile(mu, 90)),
        'global_std': float(x.std()),
    }
    print(name, json.dumps(out[name]), flush=True)


mim = mimic_5s()
stats(mim['x'], 'mimic_5s_raw')

me = meeti_5s()
stats(me['x'], 'meeti_5beat')

sp = compute_ptb_split()
ptb_tr = ptb_5s(sp['train'])
stats(ptb_tr['x'], 'ptb_5beat_train')

dev = device_5s()
for k, d in dev.items():
    stats(d['x'], f'device_5beat_{k}')

with open('/home/devcontainers/meeti/runs/r9_probe_scale_stats/result.json',
          'w') as f:
    json.dump(out, f, indent=2)
print('saved', flush=True)
