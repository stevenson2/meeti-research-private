#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小规模实验: 公共库硬标签 + MEETI 记录级弱标签辅助训练(输出项目外)。"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ/'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large

RUN = Path('/home/devcontainers/meeti/runs/meeti_weak_small')
NPZ = PROJ/'pc_tools/ecg_dl/models/deploy_match/distill_pilot_teacher_targets.npz'
MEETI = Path('/home/devcontainers/meeti/data/processed_small')

gpus = tf.config.list_physical_devices('GPU')
print('gpus', gpus, flush=True)
for g in gpus: tf.config.experimental.set_memory_growth(g, True)

d = np.load(NPZ)
x_hard = np.asarray(d['x_train_perm'], dtype=np.float32)[..., None]
y_hard = np.asarray(d['y_train'], dtype=np.int32)
x_val = np.asarray(d['x_val'], dtype=np.float32)[..., None]
y_val = np.asarray(d['y_val'], dtype=np.int32)
x_real = np.asarray(d['x_real_holdout'], dtype=np.float32)[..., None]

x_meeti = np.load(MEETI/'beats.npy')[..., None]
y_meeti = np.load(MEETI/'labels_weak.npy').astype(np.int32)
keep = np.isin(y_meeti, [0,1])
x_meeti = x_meeti[keep]; y_meeti = y_meeti[keep]
# 控制小规模: 弱标签最多各 5000 拍
rng = np.random.default_rng(42)
idx=[]
for c in [0,1]:
    cidx=np.where(y_meeti==c)[0]
    idx += rng.choice(cidx, min(5000,len(cidx)), replace=False).tolist()
idx=np.array(idx)
x_meeti=x_meeti[idx]; y_meeti=y_meeti[idx]

x = np.concatenate([x_hard, x_meeti])
y = np.concatenate([y_hard, y_meeti])
w = np.concatenate([np.ones(len(x_hard), np.float32),
                    np.full(len(x_meeti), 0.2, np.float32)])
perm = rng.permutation(len(x)); x,y,w=x[perm],y[perm],w[perm]
print('train', x.shape, 'hard', len(x_hard), 'meeti', len(x_meeti), flush=True)

tf.random.set_seed(42)
model = build_ecg_resnet_lite_large(input_shape=(250,1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
t0=time.time()
hist=model.fit(x, y, sample_weight=w, validation_data=(x_val, y_val),
               batch_size=64, epochs=20, verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                            patience=5, restore_best_weights=True)])
p_val=model.predict(x_val, batch_size=256, verbose=0)[:,1]
p_real=model.predict(x_real, batch_size=64, verbose=0)[:,1]
auc=float(roc_auc_score(y_val,p_val))
model.save(str(RUN/'model.keras'))
result={'auc':auc,'real_mean':float(p_real.mean()),
        'real_frac_gt_0.5':float((p_real>0.5).mean()),
        'elapsed_s':round(time.time()-t0,1),
        'history':{k:[float(v) for v in vals] for k,vals in hist.history.items()}}
with open(RUN/'result.json','w') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2), flush=True)
