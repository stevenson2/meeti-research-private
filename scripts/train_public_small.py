#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""小规模公共库基线训练(输出全部写到项目外 /home/devcontainers/meeti/runs)。"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ/'pc_tools/ecg_dl'))
os.environ.setdefault('ECG_PROCESSED_DIR', '/home/devcontainers/ecg_data')
from models.resnet_lite_1d import build_ecg_resnet_lite_large
from sklearn.metrics import roc_auc_score

RUN = Path('/home/devcontainers/meeti/runs/public_small')
NPZ = PROJ/'pc_tools/ecg_dl/models/deploy_match/distill_pilot_teacher_targets.npz'

gpus = tf.config.list_physical_devices('GPU')
print('gpus', gpus, flush=True)
if gpus:
    for g in gpus: tf.config.experimental.set_memory_growth(g, True)

d = np.load(NPZ)
x = np.asarray(d['x_train_perm'], dtype=np.float32)[..., None]
y = np.asarray(d['y_train'], dtype=np.int32)
xv = np.asarray(d['x_val'], dtype=np.float32)[..., None]
yv = np.asarray(d['y_val'], dtype=np.int32)

tf.random.set_seed(42)
model = build_ecg_resnet_lite_large(input_shape=(250,1))
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=3e-4),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])
t0=time.time()
hist = model.fit(x, y, validation_data=(xv, yv), batch_size=32, epochs=30,
                 class_weight={0:2.0,1:1.0}, verbose=2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)])
p = model.predict(xv, batch_size=256, verbose=0)[:,1]
auc = float(roc_auc_score(yv, p))
model.save(str(RUN/'model.keras'))
with open(RUN/'history.json','w') as f: json.dump({'auc':auc,'elapsed_s':time.time()-t0,'history':{k:[float(v) for v in vals] for k,vals in hist.history.items()}}, f, indent=2)
print(json.dumps({'auc':auc,'elapsed_s':time.time()-t0}, indent=2), flush=True)
