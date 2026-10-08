#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""R12-T2: ensemble -> single beat-head distillation (soft-label KD).

Teacher = legacy_ensemble (soft-vote mean of ensemble_seed{42,123,456}.h5,
beat_mi val 0.9624 / test 0.9206 - MIT+INCART domain, benchmark-verified).
Student = the incumbent companion beat-head architecture (resnet-lite-large
encoder fc1(64) + Dense(2), the int8-friendly locked beat branch), trained
standalone on the MIT+INCART TRAIN patients only (split_guard, leak-free).

KD loss (teacher ships softmax probs, not logits - disclosed):
    L = (1-a) * CE(y, softmax(z))  +  a * T^2 * CE(q_T, softmax(z/T))
    q_T = sigmoid(logit(p_teacher) / T)      (tempered teacher target)

Domain discipline: PTB val/test and the device recordings are EVAL-ONLY
(never trained).  Checkpoint = best MIT+INCART val beat AUC epoch.

Acceptance gates (BENCHMARK_V1 v1.5 sec 15.3):
  S_beat >= 55.9 (locked beat head)  AND  gate device FP = 0
  AND MI val abnormal recall 26/26 (with the deployed record head).

Usage:
  python3 r12_t2_kd_beat.py --alpha 0.2 --temp 2 --seed 42 --epochs 20
  python3 r12_t2_kd_beat.py --gate-only --run <run_dir>  (gate check)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import Model, layers

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/'
            'ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ / 'pc_tools/ecg_dl'))
sys.path.insert(0, '/home/devcontainers/meeti/scripts')
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large  # noqa: E402
from data.split_guard import compute_mit_incart_split, load_arrays  # noqa

ROOT = Path('/home/devcontainers/meeti')
OUT = ROOT / 'runs/r12_t2_kd_beat'
MEMBERS = [PROJ / 'pc_tools/ecg_dl/models' / f'ensemble_seed{s}.h5'
           for s in (42, 123, 456)]
from r8_data_5s import REAL_FILES, REAL_DIR  # noqa: E402 (device holdout)
REAL = [(name, REAL_DIR / fname) for name, fname in REAL_FILES.items()]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def load_mi_domain():
    """MIT+INCART train/val beats (patient-level split, seed 42)."""
    split = compute_mit_incart_split()
    dom = {}
    for name, key in [('train', None), ('val', 'val_masks'),
                      ('test', 'test_masks')]:
        xs, ys = [], []
        for tag in ['mit_bih', 'incart']:
            b, l, _ = load_arrays(tag)
            if key is None:
                m = ~(np.asarray(split['val_masks'][tag], bool)
                      | np.asarray(split['test_masks'][tag], bool))
            else:
                m = np.asarray(split[key][tag], bool)
            xs.append(np.asarray(b, np.float32)[m])
            ys.append(np.asarray(l)[m].astype(np.int32))
        dom[name] = (np.concatenate(xs), np.concatenate(ys))
    return dom


def teacher_probs(x, batch=1024):
    """Soft-vote mean positive-class prob of the 3 ensemble members."""
    models = [tf.keras.models.load_model(str(m), compile=False)
              for m in MEMBERS]
    p = np.zeros(len(x), np.float64)
    for i in range(0, len(x), batch):
        xb = x[i:i + batch][..., None]
        pm = [m.predict(xb, batch_size=batch, verbose=0)[:, 1]
              for m in models]
        p[i:i + batch] = np.mean(pm, axis=0)
    del models
    tf.keras.backend.clear_session()
    return p


def build_student(lr):
    """Incumbent beat-head architecture, LOGITS output (KD-friendly)."""
    base = build_ecg_resnet_lite_large(input_shape=(250, 1))
    enc = Model(base.input, base.get_layer('fc1').output, name='beat_enc')
    inp = layers.Input((250, 1), name='beat_input')
    z = layers.Dense(2, name='head_b_logits')(enc(inp))
    m = Model(inp, z, name='beat_student_kd')
    m.compile(optimizer=tf.keras.optimizers.Adam(lr))
    return m


def soft_ce(q_target, q_student):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(
        q_target, q_student))


def train(args, dom, te_p):
    (x_tr, y_tr), (x_va, y_va) = dom['train'], dom['val']
    assert len(te_p) == len(y_tr), 'teacher probs must match the train set'
    q_t = 1.0 / (1.0 + np.exp(-logit(te_p) / args.temp))
    q_t = q_t.astype(np.float32)
    run_dir = OUT / f'a{args.alpha}_t{args.temp}_seed{args.seed}'
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tf.keras.utils.set_random_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    model = build_student(args.lr)
    opt = tf.keras.optimizers.Adam(args.lr)
    sce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    alpha, temp = args.alpha, args.temp

    @tf.function(reduce_retracing=True)
    def train_step(xb, yb, qb):
        with tf.GradientTape() as tape:
            z = model(xb, training=True)
            qs = tf.nn.softmax(z / temp)[:, 1]
            l_hard = sce(yb, z)
            l_soft = soft_ce(qb, qs)
            loss = ((1 - alpha) * l_hard
                    + alpha * temp ** 2 * l_soft)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))
        return loss, l_hard, l_soft
    n = len(y_tr)
    n_steps = math.ceil(n / args.batch_size)
    best_auc, best_ep = -1.0, -1
    history = {'loss': [], 'hard': [], 'soft': [], 'mi_val_auc': []}
    for epoch in range(args.epochs):
        perm = rng.permutation(n)
        losses, hards, softs = [], [], []
        for step in range(n_steps):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            xb = x_tr[idx][..., None]
            yb = y_tr[idx]
            qb = q_t[idx]
            loss, l_hard, l_soft = train_step(xb, yb, qb)
            losses.append(float(loss))
            hards.append(float(l_hard))
            softs.append(float(l_soft))
        pv = predict_pos(model, x_va)
        auc = float(roc_auc_score(y_va, pv))
        history['loss'].append(float(np.mean(losses)))
        history['hard'].append(float(np.mean(hards)))
        history['soft'].append(float(np.mean(softs)))
        history['mi_val_auc'].append(auc)
        if auc > best_auc:
            best_auc, best_ep = auc, epoch + 1
            model.save_weights(str(run_dir / 'ckpt_best.weights.h5'))
        print(f'[a{args.alpha} t{args.temp} s{args.seed}] ep{epoch + 1} '
              f'loss={history["loss"][-1]:.4f} mi_val={auc:.4f} '
              f'(best ep{best_ep} {best_auc:.4f})', flush=True)
    model.load_weights(str(run_dir / 'ckpt_best.weights.h5'))
    return model, run_dir, {'best_mi_val_auc': best_auc, 'best_epoch': best_ep,
                            'history': history,
                            'elapsed_s': round(time.time() - t0, 1)}


def predict_pos(model, x, batch=1024):
    if x.ndim == 2:
        x = x[..., None]
    out = []
    for i in range(0, len(x), batch):
        z = model.predict(x[i:i + batch], batch_size=batch, verbose=0)
        out.append(tf.nn.softmax(z).numpy()[:, 1])
    return np.concatenate(out)


def sbeat(auc_val, auc_test):
    return round(100 * ((auc_val - 0.5) / 0.5
                        + (auc_test - 0.5) / 0.5) / 2, 1)


def full_eval(model, dom):
    from train_dual_head_small import load_hard_beats  # noqa: E402
    res = {}
    for name in ['val', 'test']:
        x, y = dom[name]
        p = predict_pos(model, x)
        res[f'beat_mi_{name}'] = round(float(roc_auc_score(y, p)), 4)
        tp = int(((p >= 0.5) & (y == 1)).sum())
        fp = int(((p >= 0.5) & (y == 0)).sum())
        fn = int(((p < 0.5) & (y == 1)).sum())
        tn = int(((p < 0.5) & (y == 0)).sum())
        rec = tp / max(tp + fn, 1)
        spec = tn / max(tn + fp, 1)
        res[f'beat_mi_{name}_op05'] = {
            'recall': round(rec, 4), 'spec': round(spec, 4),
            'bacc': round((rec + spec) / 2, 4)}
    # theta=0.35 op on test (historical deployment point)
    x, y = dom['test']
    p = predict_pos(model, x)
    tp = int(((p >= 0.35) & (y == 1)).sum())
    fp = int(((p >= 0.35) & (y == 0)).sum())
    fn = int(((p < 0.35) & (y == 1)).sum())
    tn = int(((p < 0.35) & (y == 0)).sum())
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    res['beat_mi_test_op035'] = {'recall': round(rec, 4),
                                 'spec': round(spec, 4),
                                 'bacc': round((rec + spec) / 2, 4)}
    res['S_beat'] = sbeat(res['beat_mi_val'], res['beat_mi_test'])
    # locked beat_val domain (common val, anchor 0.7631)
    _, _, xv, yv, _ = load_hard_beats()
    pv = predict_pos(model, xv)
    res['beat_val'] = round(float(roc_auc_score(yv, pv)), 4)
    # device beats (eval-only disclosure)
    dev = {}
    for name, f in REAL:
        b = np.load(f).astype(np.float32)
        if b.ndim == 3:
            b = b.reshape(b.shape[0], -1)
        pb = predict_pos(model, b)
        dev[name] = {'n_beats': int(len(pb)), 'p_mean': round(float(pb.mean()), 4),
                     'frac_gt_0.5': round(float((pb > 0.5).mean()), 4),
                     'p_max': round(float(pb.max()), 4)}
    res['device_beat'] = dev
    return res


def gate_check(model):
    """M4 AND-gate simulation with the DEPLOYED record head (candidate
    seed43, r9_form-ptbbal) + this beat head.  Bit-faithful to the
    firmware/r11_pc_parity loop: pos = (record prob on the last-5-beat
    window >= tau_r) AND (mean of last-5 beat probs >= tau_b); alarm =
    k-of-n with cooldown 20.  Device FP line = 0; MI val abnormal record
    recall line = 26/26.
    """
    rec = tf.keras.models.load_model(
        str(ROOT / 'runs/r9_form-ptbbal_seed43/ckpt_best__record5s.keras'))

    class _Rec:
        def __init__(self):
            self.n = 0

        def probs(self, seq):
            # sliding 5-beat windows -> (n-4, 1250, 1), one batch
            w = np.stack([seq[i - 4:i + 1].reshape(-1)
                          for i in range(4, len(seq))])
            self.n += len(w)
            return rec.predict(w, batch_size=256, verbose=0)[:, 1]
    r = _Rec()

    def gate_seq(seq, tau_r, tau_b, k=1, n=5, cooldown=20):
        pb = predict_pos(model, seq)
        pr = r.probs(seq)                     # pr[j] aligns with i=j+4
        alarms, pos_hist, cd = 0, [], 0
        for i in range(len(seq)):
            hbw = float(np.mean(pb[max(0, i - 4):i + 1]))
            have = i >= 4
            pos = bool(have and pr[i - 4] >= tau_r and hbw >= tau_b)
            pos_hist.append(pos)
            cand = any(pos_hist[-n:])
            if cd > 0:
                cd -= 1
            elif cand:
                alarms += 1
                cd = cooldown
        return alarms

    out = {}
    for cfg, (k, tau_r, tau_b) in {'v1_min': (1, 0.5, 0.15),
                                   'v2_spec': (2, 0.7, 0.10)}.items():
        fp = 0
        for name, f in REAL:
            b = np.load(f).astype(np.float32)
            if b.ndim == 3:
                b = b.reshape(b.shape[0], -1)
            fp += gate_seq(b, tau_r, tau_b, k=k)
        out[f'{cfg}_device_fp'] = fp
    # MI val abnormal recall (M4 e2e acceptance line, both gate configs)
    split = compute_mit_incart_split()
    recall = {'v1_min': 0, 'v2_spec': 0}
    total = 0
    for tag in ['mit_bih', 'incart']:
        b, l, rr = load_arrays(tag)
        m = np.asarray(split['val_masks'][tag], bool)
        b = np.asarray(b, np.float32)[m]
        l = np.asarray(l)[m]
        rr = np.asarray(rr)[m]
        order = np.argsort(rr, kind='stable')
        b, l, rr = b[order], l[order], rr[order]
        uniq, first, counts = np.unique(rr, return_index=True,
                                        return_counts=True)
        for i in range(len(uniq)):
            s, nb = int(first[i]), int(counts[i])
            if l[s:s + nb].max() != 1:
                continue
            total += 1
            for cfg, (k, tau_r, tau_b) in {'v1_min': (1, 0.5, 0.15),
                                           'v2_spec': (2, 0.7, 0.10)}.items():
                if gate_seq(b[s:s + nb], tau_r, tau_b, k=k) > 0:
                    recall[cfg] += 1
    out['mi_val_abn_records'] = total
    out['mi_val_recall'] = recall
    out['record_windows_scored'] = r.n
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.2)
    ap.add_argument('--temp', type=float, default=2.0)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch-size', type=int, default=512)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--subset', type=int, default=0,
                    help='optional train-beat subset size (budget smoke)')
    ap.add_argument('--eval-only', default=None,
                    help='run_dir: reload weights and evaluate only')
    ap.add_argument('--gate-only', default=None,
                    help='run_dir: M4 AND-gate check only (device FP + '
                         'MI val recall with the deployed record head)')
    args = ap.parse_args()
    for g in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(g, True)
    OUT.mkdir(parents=True, exist_ok=True)

    if args.gate_only:
        run_dir = Path(args.gate_only)
        model = build_student(args.lr)
        model.build((None, 250, 1))
        model.load_weights(str(run_dir / 'ckpt_best.weights.h5'))
        g = gate_check(model)
        prev = json.load(open(run_dir / 'result.json'))
        prev['gate'] = g
        with open(run_dir / 'result.json', 'w') as f:
            json.dump(prev, f, indent=2, ensure_ascii=False)
        print('GATE:', json.dumps(g), flush=True)
        return

    dom = load_mi_domain()
    print(f"beats train/val/test: {len(dom['train'][1])} / "
          f"{len(dom['val'][1])} / {len(dom['test'][1])}", flush=True)
    tp_cache = OUT / 'teacher_probs_train.npz'
    if args.subset:
        key = f'p_sub{args.subset}'
    else:
        key = 'p_full'
    cache = np.load(tp_cache) if tp_cache.exists() else {}
    if key in cache:
        te_p = cache[key]
        print(f'teacher probs cached ({key})', flush=True)
    else:
        if args.subset:
            rng0 = np.random.default_rng(0)
            idx = rng0.choice(len(dom['train'][1]), size=args.subset,
                              replace=False)
            te_p = teacher_probs(dom['train'][0][idx])
        else:
            te_p = teacher_probs(dom['train'][0])
        merged = {k: np.asarray(cache[k])
                  for k in getattr(cache, "files", [])}
        merged[key] = te_p
        np.savez_compressed(tp_cache, **merged)
        print(f'teacher probs computed ({key})', flush=True)

    if args.subset:
        rng0 = np.random.default_rng(0)
        idx = rng0.choice(len(dom['train'][1]), size=args.subset,
                          replace=False)
        dom['train'] = (dom['train'][0][idx], dom['train'][1][idx])

    run_dir = OUT / f'a{args.alpha}_t{args.temp}_seed{args.seed}'
    if args.eval_only:
        run_dir = Path(args.eval_only)
        model = build_student(args.lr)
        model.build((None, 250, 1))
        model.load_weights(str(run_dir / 'ckpt_best.weights.h5'))
        prev = json.load(open(run_dir / 'result.json'))
        res = full_eval(model, dom)
        prev['full_eval'] = res
        with open(run_dir / 'result.json', 'w') as f:
            json.dump(prev, f, indent=2, ensure_ascii=False)
        print(json.dumps(res, indent=2), flush=True)
        return

    model, run_dir, tr_info = train(args, dom, te_p)
    res = full_eval(model, dom)
    result = {'arm': 'r12_t2_kd_beat', 'alpha': args.alpha, 'temp': args.temp,
              'seed': args.seed, 'config': vars(args),
              'teacher': 'legacy_ensemble soft-vote (3 x ensemble_seed*.h5)',
              'train_domain': 'MIT+INCART train patients only',
              'train': tr_info, 'eval': res}
    with open(run_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print('EVAL:', json.dumps(res), flush=True)
    print('saved ->', run_dir / 'result.json', flush=True)


if __name__ == '__main__':
    main()
