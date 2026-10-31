import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
import csv,hashlib,json,sys
from pathlib import Path
import numpy as np
R=Path(os.environ['ECG_INT8_SOURCE'])
D=Path(os.environ['ECG_CROSSLEVEL_SOURCE'])
W=Path(__file__).resolve().parent;Q=W.parent/'int8_diagnosis';O=W.parents[1]/'outputs/PTQ_REPAIR_20261029'
O.mkdir(parents=True,exist_ok=True)
KEYS=[f'seed{s}_{a}' for s in (42,43,44) for a in 'AB']
PATHS=['FP32','ORIGINAL_INT8_margin','REPAIRED_INT8_margin']
with (D/'beat_manifest.csv').open(newline='',encoding='utf-8-sig') as f:M=list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def writecsv(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def load(key,part):
 fp=np.load(R/'predictions'/f'{key}_keras_{part}.npz');old=np.load(Q/f'{key}_margin_{part}.npz')
 new=np.load(W/f'{key}_repaired_{part}.npz');ids=fp['ids']
 assert np.array_equal(ids,old['ids']) and np.array_equal(ids,new['ids'])
 s={'FP32':fp['prob'][:,1].astype(float),'ORIGINAL_INT8_margin':old['zq'][:,1].astype(int)-old['zq'][:,0].astype(int),
    'REPAIRED_INT8_margin':new['zq'][:,1].astype(int)-new['zq'][:,0].astype(int)}
 start=1000 if part=='pre' else 0;ids=ids[start:];s={k:v[start:] for k,v in s.items()}
 y=np.array([M[int(i)]['label']=='V' for i in ids]);p=np.array([M[int(i)]['patient'] for i in ids])
 assert all(M[int(i)]['split']==('calibration' if part=='pre' else 'dev') for i in ids)
 return ids,y,p,s
def weights(y,p):
 wn=np.zeros(len(y));wv=np.zeros(len(y))
 ns=np.unique(p[~y]);vs=np.unique(p[y])
 for who in ns:
  a=(p==who)&~y;wn[a]=1/(len(ns)*a.sum())
 for who in vs:
  a=(p==who)&y;wv[a]=1/(len(vs)*a.sum())
 return wn,wv
def select(s,y,p,kind):
 wn,wv=weights(y,p)
 if kind!='FP32':
  nh=np.bincount(s+255,weights=wn,minlength=511);vh=np.bincount(s+255,weights=wv,minlength=511)
  nf=np.r_[np.cumsum(nh[::-1])[::-1],0.];vf=np.r_[np.cumsum(vh[::-1])[::-1],0.]
  j=max(np.flatnonzero(nf<=.02+1e-12),key=lambda j:(vf[j],-nf[j],j))
  return {'cutoff':int(j-255),'decision':'>='}
 idx=np.argsort(-s,kind='stable');sc=s[idx];ends=np.r_[np.flatnonzero(sc[1:]!=sc[:-1])+1,len(sc)]
 n=np.cumsum(wn[idx]);v=np.cumsum(wv[idx]);cand=[(0.,0.,float(sc[0]))]
 cand += [(float(v[j-1]),-float(n[j-1]),float(np.nextafter(np.float32(sc[j-1]),np.float32(-np.inf)))) for j in ends if n[j-1]<=.02+1e-12]
 return {'cutoff':max(cand)[2],'decision':'>'}
def metrics(s,y,p,lock):
 pred=s>lock['cutoff'] if lock['decision']=='>' else s>=lock['cutoff'];wn,wv=weights(y,p)
 return {'V_recall':float(wv[pred&y].sum()),'N_FPR':float(wn[pred&~y].sum()),
 'TP':int(np.sum(pred&y)),'FN':int(np.sum(~pred&y)),'FP':int(np.sum(pred&~y)),'TN':int(np.sum(~pred&~y))},pred
def freeze():
 assert not any((W/f'{k}_repaired_dev.npz').exists() for k in KEYS),'dev predictions exist before freeze'
 oldfreeze=json.loads((R/'conversion_freeze.json').read_text());locks={};infos={};calrows=[]
 rep=np.load(R/'representative_ids.npy');assert len(rep)==len(set(rep))==1000
 assert all(M[int(i)]['split']=='fit' for i in rep)
 for key in KEYS:
  info=json.loads((W/f'{key}_repair.json').read_text());assert info['recipe_config_sha256']==sha(W/'PREREG.json')
  assert info['source_weights_sha256']==oldfreeze['outputs'][key]['weights_sha256']
  assert sha(O/'models'/f'{key}_mse_int8_logits.tflite')==info['model_sha256']
  ids,y,p,scores=load(key,'pre');assert len(ids)==9097 and (~y).sum()==8495 and y.sum()==602 and len(set(p))==24
  assert not set(ids)&set(rep)
  locks[key]={}
  for path,s in scores.items():
   lock=select(s,y,p,path);m,_=metrics(s,y,p,lock);locks[key][path]=lock
   calrows.append({'key':key,'path':path,**lock,**m})
  infos[key]=info
 (O/'PREREG.json').write_bytes((W/'PREREG.json').read_bytes())
 (O/'calibration_thresholds.json').write_text(json.dumps(locks,indent=2),encoding='utf-8')
 writecsv('calibration_metrics.csv',calrows)
 (O/'repair_manifest.json').write_text(json.dumps(infos,indent=2),encoding='utf-8')
 frozen={'status':'ALL_SIX_MODELS_AND_CALIBRATION_THRESHOLDS_FROZEN_BEFORE_NEW_DEV_INFERENCE',
  'recipe_sha256':sha(W/'PREREG.json'),'representative_ids_sha256':sha(R/'representative_ids.npy'),
  'calibration_thresholds_sha256':sha(O/'calibration_thresholds.json'),
  'model_sha256':{k:v['model_sha256'] for k,v in infos.items()},
  'input_data_sha256':json.loads((R/'config_frozen.json').read_text())['input_sha256'],
  'dev_status':'already exposed historically; no repaired dev inference before this freeze'}
 (O/'FREEZE.json').write_text(json.dumps(frozen,indent=2),encoding='utf-8');print(json.dumps(frozen,indent=2))
def evaluate():
 frozen=json.loads((O/'FREEZE.json').read_text());assert sha(O/'calibration_thresholds.json')==frozen['calibration_thresholds_sha256']
 locks=json.loads((O/'calibration_thresholds.json').read_text());rows=[];patients=[];beatrows=[]
 for key in KEYS:
  assert sha(O/'models'/f'{key}_mse_int8_logits.tflite')==frozen['model_sha256'][key]
  ids,y,p,scores=load(key,'dev');assert len(ids)==15222 and (~y).sum()==14395 and y.sum()==827 and len(set(p))==40
  for path,s in scores.items():
   for mode in ('dev_budget_diagnostic','calibration_locked'):
    lock=select(s,y,p,path) if mode=='dev_budget_diagnostic' else locks[key][path]
    m,pred=metrics(s,y,p,lock);rows.append({'key':key,'seed':int(key[4:6]),'arm':key[-1],'path':path,'mode':mode,
      'budget':.02,**lock,**m,'budget_slack':.02-m['N_FPR']})
    for who in np.unique(p):
     n=(p==who)&~y;v=(p==who)&y
     patients.append({'key':key,'seed':int(key[4:6]),'arm':key[-1],'path':path,'mode':mode,'patient':who,
       'N_count':int(n.sum()),'V_count':int(v.sum()),'N_FPR':float(pred[n].mean()) if n.any() else '',
       'V_recall':float(pred[v].mean()) if v.any() else ''})
  for j,beat in enumerate(ids):beatrows.append({'key':key,'beat_id':int(beat),'patient':p[j],'label':int(y[j]),
   'fp32_pV':scores['FP32'][j],'original_int8_margin':int(scores['ORIGINAL_INT8_margin'][j]),'repaired_int8_margin':int(scores['REPAIRED_INT8_margin'][j])})
 writecsv('dev_operating_points.csv',rows);writecsv('per_patient.csv',patients);writecsv('dev_scores.csv',beatrows)
 summary=[]
 for mode in ('dev_budget_diagnostic','calibration_locked'):
  for seed in (42,43,44):
   row={'mode':mode,'seed':seed}
   def get(a,path):return next(x for x in rows if x['seed']==seed and x['arm']==a and x['path']==path and x['mode']==mode)
   for path in PATHS:
    a=get('A',path);b=get('B',path)
    row.update({f'A_{path}_recall':a['V_recall'],f'B_{path}_recall':b['V_recall'],
      f'A_{path}_FPR':a['N_FPR'],f'B_{path}_FPR':b['N_FPR'],f'BA_{path}_pp':100*(b['V_recall']-a['V_recall'])})
   for arm in 'AB':
    row[f'{arm}_repair_minus_original_pp']=100*(get(arm,'REPAIRED_INT8_margin')['V_recall']-get(arm,'ORIGINAL_INT8_margin')['V_recall'])
    row[f'{arm}_repair_minus_FP32_pp']=100*(get(arm,'REPAIRED_INT8_margin')['V_recall']-get(arm,'FP32')['V_recall'])
   summary.append(row)
 writecsv('seed_summary.csv',summary)
 boot=[]
 for mode in ('dev_budget_diagnostic','calibration_locked'):
  for path in PATHS:
   people=sorted({x['patient'] for x in patients if x['mode']==mode and x['path']==path and x['V_recall']!=''})
   d=[]
   for who in people:
    a=[x['V_recall'] for x in patients if x['mode']==mode and x['path']==path and x['patient']==who and x['arm']=='A' and x['V_recall']!='']
    b=[x['V_recall'] for x in patients if x['mode']==mode and x['path']==path and x['patient']==who and x['arm']=='B' and x['V_recall']!='']
    assert len(a)==len(b)==3;d.append(np.mean(b)-np.mean(a))
   d=np.array(d);rng=np.random.default_rng(20261029);samples=d[rng.integers(0,len(d),(1000,len(d)))].mean(axis=1)
   boot.append({'mode':mode,'path':path,'BA_mean_pp':100*d.mean(),'CI_low_pp':100*np.quantile(samples,.025),
       'CI_high_pp':100*np.quantile(samples,.975),'patients':len(d),'positive_patients':int((d>0).sum()),'negative_patients':int((d<0).sum()),'equal_patients':int((d==0).sum())})
 writecsv('bootstrap.csv',boot)
 print(json.dumps({'seed_summary':summary,'bootstrap':boot},indent=2))
if __name__=='__main__':freeze() if sys.argv[1]=='freeze' else evaluate()
