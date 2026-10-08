#!/bin/bash
# R10d: regenerate all legacy rows with v1.4 op-metrics (process isolation)
# GPU: inherits caller env (LD_LIBRARY_PATH set, CUDA unmasked)
cd /home/devcontainers/meeti
for name in baseline_v3 baseline_v4 baseline_v5 distill_pilot_seed42 \
            distill_pilot_v2_seed42 capacity_w2_seed42 exp7c_v4 v6a5 \
            original_cnn_v2 archived_cnn_v2 resnet_medium resnet_small \
            cnn_m new_m1_beat new_v3_noptb_beat new_v3_ptb_beat ensemble; do
  echo "=== $name ==="
  python3 scripts/ecg_bench.py --legacy "repo_${name}" 2>/dev/null || \
  python3 scripts/ecg_bench.py --legacy "new_${name}" 2>/dev/null || \
  python3 scripts/ecg_bench.py --legacy "$name" 2>/dev/null || echo "FAIL $name"
done
echo ALL-DONE
