#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 ECG-FM 特征蒸馏后的 fc1 特征上只训练分类头(冻结骨干), 输出项目外。"""
import json, os, sys, time
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.keras import layers, Model

PROJ = Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0, str(PROJ/'pc_tools/ecg_dl'))
RUN = Path('/home/devcontainers/meeti/runs/ecgfm_feature_probe_small')
FEAT = Path('/home/devcontainers/meeti/runs/ecgfm_feature_kd_small/feature_model.keras')
NPZ = PROJ/'pc_tools/ecg_dl/models/deploy_match/distill_pilot_teacher_targets.npz'

for g in tf.config.list_physical_devices('GPU'): tf.config.experimental.set_memory_growth(g, True)
feat = tf.keras.models.load_model(FEAT, compile=False)
feat.trainable = False
fc1 = feat.get_layer('fc1').output
x = layers.Dropout(0.4, name='probe_do')(fc1)
out = layers.Dense(2, activation='softmax', name='probe_out')(x)
model = Model(feat.input, out)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

d=np.load(NPZ)
xh=np.asarray(d['x_train_perm'],dtype=np.float32)[...,None]; yh=np.asarray(d['y_train'],dtype=np.int32)
xv=np.asarray(d['x_val'],dtype=np.float32)[...,None]; yv=np.asarray(d['y_val'],dtype=np.int32)
xr=np.asarray(d['x_real_holdout'],dtype=np.float32)[...,None]
t0=time.time()
hist=model.fit(xh,yh,validation_data=(xv,yv),batch_size=64,epochs=30,class_weight={0:2.0,1:1.0},verbose=2,
               callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=6,restore_best_weights=True)])
p=model.predict(xv,batch_size=256,verbose=0)[:,1]; pr=model.predict(xr,batch_size=64,verbose=0)[:,1]
result={'val_auc':float(roc_auc_score(yv,p)),'real_mean':float(pr.mean()),'real_frac_gt_0.5':float((pr>0.5).mean()),'elapsed_s':round(time.time()-t0,1),
        'history':{k:[float(v) for v in vals] for k,vals in hist.history.items()}}
model.save(str(RUN/'model.keras'))
with open(RUN/'result.json','w') as f: json.dump(result,f,indent=2)
print(json.dumps(result,indent=2),flush=True)
