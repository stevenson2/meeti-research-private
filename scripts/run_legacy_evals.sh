#!/bin/bash
cd /home/devcontainers/meeti
export LD_LIBRARY_PATH=$(find /usr/local/lib/python3.12/dist-packages/nvidia -maxdepth 2 -type d -name lib -printf '%p:')
for n in repo_baseline_v3 repo_baseline_v4 repo_baseline_v5 repo_distill_pilot_seed42 repo_distill_pilot_v2_seed42 repo_capacity_w2_seed42 repo_exp7c_v4 repo_v6a5 repo_original_cnn_v2 repo_archived_cnn_v2 repo_resnet_medium repo_resnet_small repo_cnn_m new_m1_beat new_v3_noptb_beat new_v3_ptb_beat legacy_ensemble; do
  f="runs/benchmark_v1/legacy_row_${n}.json"
  need=1
  if [ -f "$f" ]; then
    if python3 - "$f" << 'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
sys.exit(0 if 'beat_mi_val' in d else 1)
PYEOF
    then need=0; fi
  fi
  if [ "$need" = "0" ]; then
    echo "$n SKIP(has beat_mi)"
  elif python3 scripts/ecg_bench.py --legacy "$n" > "/tmp/legacy_${n}.log" 2>&1; then
    echo "$n OK"
  else
    echo "$n FAIL"
  fi
done
echo ALL_DONE
