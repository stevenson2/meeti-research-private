#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小规模 ECG-FM 特征蒸馏实验: MEETI beat -> student fc1 投影 -> teacher 768 嵌入(MSE),
然后迁移权重并用公共库硬标签微调分类头。输出项目外。"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ/'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from tensorflow.keras import layers, Model

RUN = Path('/home/devcontainers/meeti/runs/ecgfm_feature_kd_small')
NPZ = PROJ/'pc_tools/ecg_dl/models/deploy_match/distill_pilot_teacher_targets.npz'
MEETI = Path('/home/devcontainers/meeti/data/processed_small')

gpus = tf.config.list_physical_devices('GPU')
print('gpus', gpus, flush=True)
for g in gpus: tf.config.experimental.set_memory_growth(g, True)

# ---------- teacher embedding per beat ----------
emb = np.load(MEETI/'teacher_emb_768.npy').astype(np.float32)
import pandas as pd
pm = pd.read_csv(MEETI/'processed_manifest.csv')
beats = np.load(MEETI/'beats.npy').astype(np.float32)
labels_weak = np.load(MEETI/'labels_weak.npy')
n_beats = pm['n_beats'].astype(int).tolist()
assert len(emb) == 2*len(pm), (len(emb), len(pm))
teacher_per_beat = np.zeros((len(beats), 768), dtype=np.float32)
pos=0
for k,nb in enumerate(n_beats):
    half=(nb+1)//2
    teacher_per_beat[pos:pos+half]=emb[2*k]
    teacher_per_beat[pos+half:pos+nb]=emb[2*k+1]
    pos += nb
assert pos == len(beats)
print('teacher per beat', teacher_per_beat.shape, flush=True)

# 小规模: MEETI 特征数据取 6000 拍
rng=np.random.default_rng(42)
midx=rng.choice(len(beats), 6000, replace=False)
x_ft=beats[midx][...,None]; y_ft=teacher_per_beat[midx]

# ---------- Phase 1: feature distillation ----------
tf.random.set_seed(42)
cls_model = build_ecg_resnet_lite_large(input_shape=(250,1))
x = cls_model.get_layer('fc1').output
proj = layers.Dense(256, activation='relu', name='kd_proj1')(x)
outp = layers.Dense(768, name='kd_proj2')(proj)
feat_model = Model(cls_model.input, outp)
feat_model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), loss='mse')
t0=time.time()
feat_model.fit(x_ft, y_ft, batch_size=128, epochs=8, verbose=2)
print('phase1 done', time.time()-t0, flush=True)

# ---------- Phase 2: transfer shared weights + hard-label finetune ----------
for layer in cls_model.layers:
    if layer.name in feat_model.layers:
        try:
            w = feat_model.get_layer(layer.name).get_weights()
            if w: layer.set_weights(w)
        except Exception:
            pass

d = np.load(NPZ)
xh = np.asarray(d['x_train_perm'], dtype=np.float32)[..., None]
yh = np.asarray(d['y_train'], dtype=np.int32)
xv = np.asarray(d['x_val'], dtype=np.float32)[..., None]
yv = np.asarray(d['y_val'], dtype=np.int32)
xr = np.asarray(d['x_real_holdout'], dtype=np.float32)[..., None]

cls_model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
hist=cls_model.fit(xh, yh, validation_data=(xv, yv), batch_size=64, epochs=20,
                   class_weight={0:2.0,1:1.0}, verbose=2,
                   callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                               patience=5, restore_best_weights=True)])
p_val=cls_model.predict(xv, batch_size=256, verbose=0)[:,1]
p_real=cls_model.predict(xr, batch_size=64, verbose=0)[:,1]
auc=float(roc_auc_score(yv,p_val))
result={'val_auc':auc,'real_mean':float(p_real.mean()),
        'real_frac_gt_0.5':float((p_real>0.5).mean()),
        'elapsed_s':round(time.time()-t0,1),
        'history':{k:[float(v) for v in vals] for k,vals in hist.history.items()}}
feat_model.save(str(RUN/'feature_model.keras'))
cls_model.save(str(RUN/'model.keras'))
with open(RUN/'result.json','w') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2), flush=True)
