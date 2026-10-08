#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TFLite 前代导出模型按同一测试评估 (每个谱系选最后一个导出)。"""
import argparse, json, os, sys
from pathlib import Path
import numpy as np, tensorflow as tf
from sklearn.metrics import roc_auc_score
PROJ=Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master')
sys.path.insert(0,str(PROJ/'pc_tools/ecg_dl')); os.environ.setdefault('ECG_PROCESSED_DIR','/home/devcontainers/ecg_data')
sys.path.insert(0,'/home/devcontainers/meeti/scripts')
from data.split_guard import compute_ptb_split, load_arrays
from train_dual_head_small import load_hard_beats, load_meeti_records
from eval_repo_manifest import ptb_record_data
T=10
MODELS={
 'repo_tflite_original':'ecg_model.tflite',
 'repo_tflite_exp5_latest':'deploy_match/exp5_clean_int8.tflite',
 'repo_tflite_p2a_latest':'deploy_match/p2a_int8.tflite',
 'repo_tflite_exp6_sgd':'ecg_model_exp6_sgd_int8.tflite',
 'repo_tflite_exp7b':'ecg_model_exp7b_int8.tflite',
 'repo_tflite_exp7c_clean_latest':'ecg_model_exp7c_clean_v6_qat_int8.tflite',
 'repo_tflite_exp7c_ecgfounder_latest':'ecg_model_exp7c_ecgfounder_v5_qat_int8.tflite',
 'repo_tflite_kd_latest':'ecg_model_kd_a070_t1_int8.tflite',
 'repo_tflite_v3a_deploy':'deploy_match/ecg_model_v3a_int8.tflite',
}
def make_pred(root,rel):
    p=root/rel; it=tf.lite.Interpreter(model_path=str(p)); it.allocate_tensors()
    ind=it.get_input_details()[0]; outd=it.get_output_details()[0]
    iscale,izp=ind['quantization']; oscale,ozp=outd['quantization']
    def pred(x):
        out=np.empty(len(x),np.float32)
        for i in range(len(x)):
            xq=np.clip(np.round(x[i]/iscale+izp),-128,127).astype(np.int8)
            it.set_tensor(ind['index'], np.reshape(xq, (1, -1, 1)))
            it.invoke()
            q=it.get_tensor(outd['index']).astype(np.float32)
            out[i]=float((q[0,1]-ozp)*oscale)
        return out
    return pred,it
def record_scores(pred,x,mask):
    n=len(x); flat=x.reshape(-1,x.shape[-1]); p=pred(flat).reshape(n,T); m=mask[...,0]
    mean_p=(p*m).sum(1)/np.maximum(m.sum(1),1.0); max_p=np.where(m>0,p,-1).max(1); return mean_p,max_p
def auc(y,p): return float(roc_auc_score(y,p)) if len(np.unique(y))>1 else None
root=Path('/mnt/c/Users/cai/OneDrive/Desktop/Fe programme 25261/ecg-programme-tju-flex.electron-master/pc_tools/ecg_dl/models')
out=Path('/home/devcontainers/meeti/runs/eval_tflite_selected'); out.mkdir(parents=True,exist_ok=True)
xb,yb,xv,yv,xr=load_hard_beats(); me=load_meeti_records(argparse.Namespace(meeti_weight=0.2)); sp=compute_ptb_split(); ptb={n:ptb_record_data(sp[n]) for n in ['train','val','test']}
rows=[]
for tag,rel in MODELS.items():
    print('===',tag,flush=True)
    try:
        pred,it=make_pred(root,rel)
        pv=pred(xv[...,0]); pr=pred(xr[...,0]); mm,mmax=record_scores(pred,me['x_val'],me['mask_val'])
        row={'tag':tag,'model':str(root/rel),'head_b':{'public_val_auc':auc(yv,pv),'real_mean':float(pr.mean()),'real_frac_gt_0.5':float((pr>.5).mean())},'meeti_record_mean_p_auc':auc(me['y_val'],mm),'meeti_record_max_p_auc':auc(me['y_val'],mmax),'ptb_record':{}}
        for n,(x,m,y) in ptb.items():
            mp,maxp=record_scores(pred,x,m); row['ptb_record'][n]={'n_records':int(len(x)),'mean_p_auc':auc(y,mp),'max_p_auc':auc(y,maxp),'mean_p_mean':float(mp.mean())}
        rows.append(row); json.dump(row,open(out/f'{tag}.json','w'),indent=2)
        print(json.dumps({'beat_auc':row['head_b']['public_val_auc'],'meeti':row['meeti_record_mean_p_auc'],'ptbva':row['ptb_record']['val']['mean_p_auc'],'ptbte':row['ptb_record']['test']['mean_p_auc'],'real_mean':row['head_b']['real_mean'],'frac':row['head_b']['real_frac_gt_0.5']},indent=2),flush=True)
    except Exception as e:
        print('FAIL',type(e).__name__,e,flush=True)
json.dump(rows,open(out/'summary.json','w'),indent=2); print('saved',len(rows),flush=True)
