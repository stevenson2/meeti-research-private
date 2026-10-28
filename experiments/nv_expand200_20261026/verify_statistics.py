"""Synthetic saved-score plumbing check only; no model inference or fitting."""
from pathlib import Path
import json,csv
import numpy as np
import evaluate as ev
from common import writej,readj
ROOT=Path(__file__).resolve().parent;out=ROOT/'verification/synthetic_statistics';out.mkdir(parents=True,exist_ok=True);pred=out/'eval_synthetic';pred.mkdir(exist_ok=True)
assignment=readj(ROOT/'patient_split_frozen.json');new=[r['patient'] for r in assignment['patients'] if r['group']=='dev']
names=['p02884',*new[:3]]
y=np.r_[np.ones(15,np.int8),np.zeros(10,np.int8)];patient=np.r_[np.zeros(10,np.int16),np.ones(5,np.int16),np.zeros(5,np.int16),np.ones(3,np.int16),np.full(2,2,np.int16)];avail=np.ones(25,bool);avail[-1]=False
meta=dict(y=y,patient=patient,avail7=avail,segment=np.zeros(25,np.int32),annotation_index=np.arange(25))
ev.metadata=lambda split:meta
ev.patient_names=lambda split:names
keys=[f"{r['segment']} a{r['annotation_index']}" for r in csv.DictReader((ROOT.parent/'NV_BASELINE_R1/review_missed_V_dev.csv').open())]
assert len(keys)==15
ev.identity=lambda m,rows:[keys[i] if i<15 else f'synthetic a{i}' for i in rows]
base=np.where(y==1,.8,.2).astype(np.float32)
for d in ev.SHIFTS:
    p=base.copy()
    if d==4:p[0]=.2
    if d==-4:p[20]=.8
    np.savez(pred/f'dev_{d:+d}.npz',p=p,margin=p-.5)
    np.savez(pred/f'train_p02866_{d:+d}.npz',y=y,p=p)
writej(pred/'repeat.json',dict(exact=True,flips=0));writej(pred/'manifest.json',dict(resources={}))
writej(out/'evaluation.json',dict(directory=pred.name));writej(out/'training_summary.json',dict(run=dict(id='SYNTHETIC_STATISTICS_ONLY',seed=0,model='synthetic'),best_checkpoint_sha256='synthetic',best_epoch=0,epoch=0,best=.5,config_sha256='synthetic-not-a-formal-run'))
ev.summarize_run(out);m=readj(out/'metrics.json')
assert m['zero']['N_false_positive']==0 and m['zero']['V_detected']==15
assert m['robust']['N_any_wrong']==1 and m['robust']['V_all_detected']==14
assert m['per_patient'][-1]['zero']['N_false_positive_rate'] is None
assert m['per_patient'][-1]['any_flip_rate'] is None
assert m['cohorts']['legacy_dev']['V_patients']==1 and m['cohorts']['new_dev']['V_patients']==1
writej(ROOT/'synthetic_statistics_check.json',dict(passed=True,no_model_inference=True,no_training_fit=True,checked=['zero counts','common set exclusion','class-specific shifts','N-any/V-all counts','old/new cohorts','single-class and empty patients','fixed15 identity table','no NaN in JSON']))
print('SYNTHETIC SAVED-SCORE STATISTICS CHECK PASSED')
