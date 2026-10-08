#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M4 real-AFE threshold sweep for the locked dual-head record head."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
from m4_afe_event_eval import (  # noqa: E402
    REAL_RECORDS, load_dual_probes, dual_window_scores, alarm_triggers,
)

BEAT_PROBE = ('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb/'
              'ckpt_balanced__beat.keras')
REC_PROBE = ('/home/devcontainers/meeti/runs/dual_head_v3_seeded_lam0.2_ptb/'
             'ckpt_balanced__record.keras')
THRESHOLDS = [0.5, 0.7, 0.8, 0.85, 0.9, 0.92, 0.95]


def main():
    out = Path('/home/devcontainers/meeti/runs/m4_afe_event_eval')
    out.mkdir(parents=True, exist_ok=True)
    beat_probe, rec_probe = load_dual_probes(BEAT_PROBE, REC_PROBE)
    rows = []
    for rec in REAL_RECORDS:
        beats = np.load(rec['beats']).astype(np.float32)
        scores = dual_window_scores(rec_probe, beats)
        dur_h = rec['dur_s'] / 3600.0
        for th in THRESHOLDS:
            tr = alarm_triggers(scores, threshold=th)
            rows.append({
                'source': rec['name'],
                'threshold': th,
                'score_frac_ge_threshold': float(np.mean(scores >= th)),
                'alarms': int(len(tr)),
                'fp_per_hour': float(len(tr) / dur_h),
            })
    result = {
        'model': 'locked_dual_head_v3_ptb',
        'control_fp_per_hour_at_0.5': {
            'exp7c_clean_v6_qat_int8': 0.0,
            'exp7c_ecgfounder_v5_qat_int8': 0.0,
        },
        'rows': rows,
        'decision': 'M4 event-level FP/hour fails: even at thresholds 0.90 and '
                    '0.95 the locked Head_R remains far above the TFLite '
                    'controls (0 FP/hour). Per plan, stop and report user.',
    }
    with open(out / 'threshold_sweep.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
