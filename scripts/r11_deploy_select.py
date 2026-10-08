#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R11 deployment candidate selection + complete model save.

Selection convention (same as locked best-of-N, selection bias documented):
choose the seed with the highest PTB val record AUC among the 6-seed
r9_form_ptbbal fleet (val is the model-selection domain; PTB test untouched
by selection).  Copy ckpt + result into runs/deploy_candidate_r11/ and
export fp16 TFLite for the firmware.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/devcontainers/meeti/scripts')
ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/deploy_candidate_r11'
SEEDS = [42, 43, 44, 45, 46, 47]

rows = []
for s in SEEDS:
    r = json.load(open(ROOT / f'runs/r9_form-ptbbal_seed{s}/result.json'))
    m = r['metrics']
    rows.append({'seed': s,
                 'meeti': m['meeti_val_record_auc'],
                 'ptb_val': m['ptb_val_record_auc'],
                 'ptb_test': m['ptb_test_record_auc'],
                 'ckpt': str(ROOT / f'runs/r9_form-ptbbal_seed{s}/'
                              'ckpt_best__record5s.keras')})
    print(f"seed{s}: meeti {rows[-1]['meeti']:.4f} "
          f"val {rows[-1]['ptb_val']:.4f} test {rows[-1]['ptb_test']:.4f}")

best = max(rows, key=lambda r: r['ptb_val'])
print('selected seed', best['seed'], 'by PTB val (selection domain)')

OUT.mkdir(parents=True, exist_ok=True)
shutil.copy2(best['ckpt'], OUT / 'record5s_fp32.keras')
shutil.copy2(ROOT / 'runs/dual_head_v3_seeded_lam0.2_ptb/'
             'ckpt_balanced__beat.keras', OUT / 'companion_beat_fp32.keras')
meta = {
    'candidate': 'r9_form_ptbbal (6-seed fleet, benchmark S=70.4, '
                 'top2mean all-gates-pass)',
    'selection': 'best PTB val seed (best-of-6, same convention/bias as '
                 'locked best-of-4; PTB test untouched)',
    'record': best,
    'fleet': rows,
    'companion_beat': 'locked dual_head beat head (int8-exported in R6/R8e)',
    'r11_note': 'aug/resample arms failed smoketest seed42 (paired '
                'ptb_val -5.0pp / -1.6pp) -> candidate stands',
    'export': {}}
with open(OUT / 'selection.json', 'w') as f:
    json.dump(meta, f, indent=2)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf  # noqa: E402

model = tf.keras.models.load_model(best['ckpt'])
conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.target_spec.supported_types = [tf.float16]
tfl = conv.convert()
(OUT / 'record5s_fp16.tflite').write_bytes(tfl)
meta['export']['record5s_fp16_bytes'] = len(tfl)
json.dump(meta, open(OUT / 'selection.json', 'w'), indent=2)
print('exported record5s_fp16.tflite:', len(tfl), 'bytes')

b = tf.keras.models.load_model(str(OUT / 'companion_beat_fp32.keras'))
convb = tf.lite.TFLiteConverter.from_keras_model(b)
convb.optimizations = [tf.lite.Optimize.DEFAULT]
tb = convb.convert()
(OUT / 'companion_beat_fp16.tflite').write_bytes(tb)
meta['export']['companion_beat_fp16_bytes'] = len(tb)
json.dump(meta, open(OUT / 'selection.json', 'w'), indent=2)
print('exported companion_beat_fp16.tflite:', len(tb), 'bytes')
