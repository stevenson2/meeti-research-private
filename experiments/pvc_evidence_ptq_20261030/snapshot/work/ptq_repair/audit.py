import os
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import csv,hashlib,json
from pathlib import Path
import numpy as np
R=Path(os.environ['ECG_INT8_SOURCE'])
D=Path(os.environ['ECG_CROSSLEVEL_SOURCE'])
W=Path(__file__).resolve().parent;Q=W.parent/'int8_diagnosis';O=W.parents[1]/'outputs/PTQ_REPAIR_20261029'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
M=read(D/'beat_manifest.csv');calrows=read(O/'calibration_metrics.csv');devrows=read(O/'dev_operating_points.csv')
frozen=json.loads((O/'FREEZE.json').read_text());checks=0;enumerated=0
assert hashlib.sha256((O/'calibration_thresholds.json').read_bytes()).hexdigest()==frozen['calibration_thresholds_sha256']
for key,digest in frozen['model_sha256'].items():
 assert hashlib.sha256((O/'models'/f'{key}_mse_int8_logits.tflite').read_bytes()).hexdigest()==digest
 for part in ('pre','dev'):
  fp=np.load(R/'predictions'/f'{key}_keras_{part}.npz');old=np.load(Q/f'{key}_margin_{part}.npz');new=np.load(W/f'{key}_repaired_{part}.npz')
  ids=fp['ids'];assert np.array_equal(ids,old['ids']) and np.array_equal(ids,new['ids'])
  scores={'FP32':fp['prob'][:,1].astype(float),'ORIGINAL_INT8_margin':old['zq'][:,1].astype(int)-old['zq'][:,0].astype(int),
    'REPAIRED_INT8_margin':new['zq'][:,1].astype(int)-new['zq'][:,0].astype(int)}
  start=1000 if part=='pre' else 0;ids=ids[start:];scores={k:v[start:] for k,v in scores.items()}
  y=np.array([M[int(i)]['label']=='V' for i in ids]);p=np.array([M[int(i)]['patient'] for i in ids])
  assert len(set(ids))==len(ids)==(9097 if part=='pre' else 15222)
  peopleN=np.unique(p[~y]);peopleV=np.unique(p[y]);nw=np.zeros(len(y));vw=np.zeros(len(y))
  for who in peopleN:
   a=(p==who)&~y;nw[a]=1/(len(peopleN)*a.sum())
  for who in peopleV:
   a=(p==who)&y;vw[a]=1/(len(peopleV)*a.sum())
  for row in calrows if part=='pre' else devrows:
   if row['key']!=key:continue
   s=scores[row['path']];cut=float(row['cutoff']);pred=s>cut if row['decision']=='>' else s>=cut
   n=float(np.mean([pred[(p==who)&~y].mean() for who in peopleN]));v=float(np.mean([pred[(p==who)&y].mean() for who in peopleV]))
   cnt=[int(np.sum(pred&y)),int(np.sum(~pred&y)),int(np.sum(pred&~y)),int(np.sum(~pred&~y))]
   assert cnt==[int(row[k]) for k in ['TP','FN','FP','TN']]
   assert abs(n-float(row['N_FPR']))<1e-12 and abs(v-float(row['V_recall']))<1e-12
   checks+=1
  for path in ('ORIGINAL_INT8_margin','REPAIRED_INT8_margin'):
   cand=[]
   for cut in range(-255,257):
    pred=scores[path]>=cut;cand.append((float(vw[pred&y].sum()),-float(nw[pred&~y].sum()),cut));enumerated+=1
   best=max(x for x in cand if -x[1]<=.02+1e-12)
   ref=next(x for x in (calrows if part=='pre' else devrows) if x['key']==key and x['path']==path and (part=='pre' or x['mode']=='dev_budget_diagnostic'))
   assert best[2]==int(ref['cutoff']),(key,part,path,best,ref)
result={'status':'PASS','count_patient_metric_checks':checks,'integer_cutoffs_exhaustively_checked':enumerated,
 'source':'saved identical beat IDs, labels and three score paths; independent selection enumeration, no extra model inference',
 'model_hashes_and_calibration_freeze_match':True,'board':'not_measured','training':'none'}
(O/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
