#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R16 device replay reference: PC-side replication of the ESP-IDF AI chain.

Replicates components/ecg_core/filter.cpp + components/ecg_ai/ecg_ai.cpp on
the firmware's compiled-in MIT-BIH replay segments (record 100 normal /
record 106 VEB-dense, 500 Hz, 45 s loops) and compares the per-window
abnormal probability with the on-device AI_RESULT captures.

Chain (bit-faithful to the firmware):
  1. x - 1.65 V DC offset
  2. two-stage 10-tap causal moving average (zero-init)
  3. AI filter: causal HP 0.05 Hz + LP 40 Hz (DF2T, zero-init)
  4. 2:1 decimation (keep even indices)
  5. causal HP 0.5 Hz @250 Hz (DF2T, zero-init)
  6. window = last 250 decimated samples when sample_count % 250 == 6
  7. population z-score, int8 quantize (trunc(x/scale+0.5)+zp), TFLite
  8. p1 = abnormal probability; 1-of-5 confirm + cooldown 5

Outputs: runs/r16_device_test/replay_pc_reference.json
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

EL = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
          'ecg-programme-tju-flex.electron-master')
HEADER = EL / 'experiments/esp_idf_ecg_migration/components/ecg_core/include/' \
              'signal_generator/ecg_replay_data.h'
MODEL = EL / 'pc_tools/ecg_dl/models/deploy_match/ecg_model_v3a_int8.tflite'
OUT = Path('/home/devcontainers/meeti/runs/r16_device_test')
CAPTURES = {
    'replay_normal': Path('/mnt/c/esp/capture_replay_normal.log'),
    'replay_abnormal': Path('/mnt/c/esp/capture_replay_abnormal.log'),
    'simulator': Path('/mnt/c/esp/capture_simulator.log'),
}
# firmware coefficients
AI_CHAIN_HP = (0.9995558103876084, -1.9991116207752169, 0.9995558103876084,
               -1.9991114234707954, 0.9991118180796384)
LP = (0.046131802093312926, 0.09226360418662585, 0.046131802093312926,
      -1.3072850288493234, 0.4918122372225752)
AI_HP_250 = (0.99115359510166301, -1.982307190203326, 0.99115359510166301,
             -1.9822289297925284, 0.98238545061412508)
WIN = 250
STRIDE = 250
OFFSET = 6
THRESHOLD = 0.50
CONFIRM_N = 5
COOLDOWN = 5


def df2t(x, b0, b1, b2, a1, a2):
    y = np.zeros(len(x), np.float64)
    w1 = w2 = 0.0
    for i, v in enumerate(np.asarray(x, np.float64)):
        w = v - a1 * w1 - a2 * w2
        y[i] = b0 * w + b1 * w1 + b2 * w2
        w2, w1 = w1, w
    return y


def comb(x, taps=10):
    k = np.ones(taps) / taps
    return np.convolve(np.convolve(x, k)[:len(x)], k)[:len(x)]


def parse_c_array(path, name):
    txt = path.read_text(encoding='utf-8', errors='replace')
    m = re.search(rf'const float {name}\[\] = \{{(.*?)\}};', txt, re.S)
    if not m:
        raise ValueError(f'{name} not found')
    vals = re.findall(r'(-?\d+\.?\d*(?:[eE][-+]?\d+)?)f', m.group(1))
    return np.array([float(v) for v in vals], np.float64)


def firmware_chain(replay):
    s = replay - 1.65
    s = comb(s)
    s = df2t(s, *AI_CHAIN_HP)
    s = df2t(s, *LP)
    s250 = s[0::2]
    s250 = df2t(s250, *AI_HP_250)
    return s250


def windows(s250):
    out = []
    for sc in range(1, len(s250) + 1):
        if sc >= WIN and sc % STRIDE == OFFSET:
            out.append(s250[sc - WIN:sc].copy())
    return np.array(out)


def confirm(raw, n=CONFIRM_N, cooldown=COOLDOWN):
    hist = []
    idx = 0
    cd = 0
    conf = []
    for r in raw:
        if len(hist) < n:
            hist.append(bool(r))
        else:
            hist[idx] = bool(r)
        idx = (idx + 1) % n
        cand = any(hist)
        if cd > 0:
            cd -= 1
            conf.append(0)
        elif cand:
            conf.append(1)
            cd = cooldown
        else:
            conf.append(0)
    return np.array(conf)


def parse_capture(path):
    conf, raw, confirmed = [], [], []
    if not path.exists():
        return None
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        m = re.match(r'AI_RESULT,([0-9.]+),(\d+),(\d+)', line)
        if m:
            conf.append(float(m.group(1)))
            raw.append(int(m.group(2)))
            confirmed.append(int(m.group(3)))
    return {'conf': np.array(conf), 'raw': np.array(raw),
            'confirmed': np.array(confirmed)}


def best_align(pc, dev, max_shift=60):
    best = None
    for sh in range(0, max(len(dev) - len(pc) + 1, 1)):
        d = dev[sh:sh + len(pc)]
        if len(d) < len(pc):
            break
        a = pc - pc.mean()
        b = d - d.mean()
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        c = float(np.dot(a, b) / denom) if denom > 0 else 0.0
        mad = float(np.mean(np.abs(pc - d)))
        if best is None or c > best['corr']:
            best = {'shift': sh, 'corr': round(c, 4), 'mean_abs_delta': round(mad, 4),
                    'pc_mean': round(float(pc.mean()), 4),
                    'dev_mean': round(float(d.mean()), 4),
                    'pc_raw_rate': round(float((pc > THRESHOLD).mean()), 4),
                    'dev_raw_rate': round(float(
                        (dev[sh:sh + len(pc)] > THRESHOLD).mean()), 4)}
    return best


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=str(MODEL))
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    outd = interp.get_output_details()[0]
    in_scale = float(inp['quantization'][0])
    in_zp = int(inp['quantization'][1])
    out_scale = float(outd['quantization'][0])
    out_zp = int(outd['quantization'][1])

    res = {'arm': 'r16_device_replay_ref', 'model': str(MODEL),
           'model_bytes': MODEL.stat().st_size,
           'input_scale': in_scale, 'input_zero_point': in_zp,
           'output_scale': out_scale, 'output_zero_point': out_zp,
           'threshold': THRESHOLD, 'confirm': '1-of-5 cooldown5',
           'modes': {}}
    for seg, name in [('normal', 'ecg_replay_normal'),
                      ('abnormal', 'ecg_replay_abnormal')]:
        replay = parse_c_array(HEADER, name)
        s250 = firmware_chain(replay)
        wins = windows(s250)
        p1 = []
        for w in wins:
            z = w - w.mean()
            sd = z.std()
            if sd < 1e-6:
                sd = 1.0
            z = z / sd
            q = np.trunc(z / in_scale + 0.5) + in_zp
            q = np.clip(q, -128, 127).astype(np.int8)
            interp.set_tensor(inp['index'], q.reshape(inp['shape']))
            interp.invoke()
            o = interp.get_tensor(outd['index']).astype(np.int64)
            o = o.reshape(-1)
            p1.append(float((o[1] - out_zp) * out_scale))
        p1 = np.array(p1, np.float64)
        pc_conf = confirm(p1 > THRESHOLD)
        cap = parse_capture(CAPTURES[f'replay_{seg}'])
        entry = {
            'n_windows_pc': int(len(p1)),
            'pc_mean': round(float(p1.mean()), 4),
            'pc_median': round(float(np.median(p1)), 4),
            'pc_p90': round(float(np.percentile(p1, 90)), 4),
            'pc_raw_abnormal_rate': round(float((p1 > THRESHOLD).mean()), 4),
            'pc_confirmed_count': int(pc_conf.sum()),
        }
        if cap is not None and len(cap['conf']) > 0:
            entry['n_windows_device'] = int(len(cap['conf']))
            entry['device_mean'] = round(float(cap['conf'].mean()), 4)
            entry['device_median'] = round(float(np.median(cap['conf'])), 4)
            entry['device_raw_abnormal_rate'] = round(
                float(cap['raw'].mean()), 4)
            entry['device_confirmed_count'] = int(cap['confirmed'].sum())
            entry['alignment'] = best_align(p1, cap['conf'])
            entry['first10_pc'] = [round(float(x), 4) for x in p1[:10]]
            entry['first10_device'] = [round(float(x), 4)
                                       for x in cap['conf'][:10]]
        res['modes'][seg] = entry
    (OUT / 'replay_pc_reference.json').write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=str),
        encoding='utf-8')
    print(json.dumps(res, indent=2, ensure_ascii=False)[:5000])
    print('written', OUT / 'replay_pc_reference.json')


if __name__ == '__main__':
    main()
