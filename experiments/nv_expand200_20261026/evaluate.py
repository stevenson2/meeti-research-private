"""GPU predictions saved incrementally; pure statistics can run without TF."""
from common import *
import csv

def auc_ap(y,p):
    y=np.asarray(y); nV=int(y.sum()); nN=len(y)-nV
    if not nV: return None,None
    order=np.argsort(-p,kind='stable'); s=p[order]; t=y[order]
    ends=np.r_[np.flatnonzero(s[:-1]!=s[1:]),len(s)-1]
    tp=np.cumsum(t,dtype=np.float64)[ends]; fp=ends+1-tp
    ap=float(np.sum(np.diff(np.r_[0,tp])/nV*tp/(ends+1)))
    auc=float(np.trapezoid(np.r_[0,tp]/nV,np.r_[0,fp]/nN)) if nN else None
    return auc,ap

def class_metrics(y,p):
    n=int((y==0).sum()); v=int((y==1).sum()); fp=int(np.sum((y==0)&(p>.5))); tp=int(np.sum((y==1)&(p>.5)))
    return dict(N=n,V=v,N_false_positive=fp,V_detected=tp,N_false_positive_rate=fp/n if n else None,V_detection_rate=tp/v if v else None)

def evaluate_run(model,run,out,training):
    start=time.monotonic(); ck=sha(out/'best.weights.h5'); ev=out/('eval_'+ck[:16]); ev.mkdir(exist_ok=True)
    X,dv=data('dev'); rows=np.arange(len(dv['y']))
    legacy=ROOT.parent/'NV_LONGRUN_20261024'
    T=np.load(legacy/'train_X.npy',mmap_mode='r')
    with np.load(legacy/'train_meta.npz') as f: tr={k:f[k] for k in f.files}
    old_names=[p['patient'] for p in readj(ROOT.parent/'NV_BASELINE_R1/prep_manifest.json')['patients']['train']]
    ix26=np.flatnonzero(tr['patient']==old_names.index('p02866'))
    assert len(ix26)==8990
    forward=make_forward(model); bn0=bn_state(model)
    manifest=dict(checkpoint_sha256=ck,run=run,config_sha256=training['config_sha256'],predictions={},started=now())
    if (ev/'manifest.json').exists():
        manifest=readj(ev/'manifest.json'); assert manifest['checkpoint_sha256']==ck
    def save_one(split,d,repeat=False):
        name=f'{split}_{d:+d}'+('_repeat' if repeat else '')
        path=ev/(name+'.npz')
        if path.exists() and name in manifest['predictions']:
            assert sha(path)==manifest['predictions'][name]['sha256']; return
        matrix,meta,ix,col=(X,dv,rows,SHIFTS.index(d)) if split=='dev' else (T,tr,ix26,d+4)
        t=time.monotonic(); p,mg,nll,timing=infer(forward,matrix,meta,ix,col)
        assert np.isfinite(p).all() and np.isfinite(mg).all()
        np.savez(path,p=p,margin=mg,rows=ix,y=meta['y'][ix],patient=(meta['patient'][ix] if split=='dev' else np.full(len(ix),patient_names('train').index('p02866'),np.int16)),segment=meta['segment'][ix],annotation_index=meta['annotation_index'][ix],avail7=meta['avail7'][ix])
        manifest['predictions'][name]=dict(rows_namespace=('combined_dev' if split=='dev' else 'legacy_train_cache'),patient_namespace=('combined_dev' if split=='dev' else 'combined_train'),sha256=sha(path),seconds=time.monotonic()-t,**timing)
        writej(ev/'manifest.json',manifest)
        print('PREDICTION SAVED',run['id'],name,flush=True)
    save_one('dev',0); save_one('dev',0,True)
    with np.load(ev/'dev_+0.npz') as a,np.load(ev/'dev_+0_repeat.npz') as b:
        repeat=compare(a['p'],a['margin'],b['p'],b['margin'])
    writej(ev/'repeat.json',repeat)
    if repeat['flips']: raise RuntimeError('GPU_REPEAT_CLASSIFICATION_FLIPS: investigate before interpreting shifts')
    for d in SHIFTS:
        save_one('dev',d)
    for d in SHIFTS: save_one('train_p02866',d)
    manifest['bn_unchanged']=all(np.array_equal(a,b) for a,b in zip(bn0,bn_state(model)))
    assert manifest['bn_unchanged']
    manifest.update(seconds_this_process=time.monotonic()-start,resources=resources(),finished=now())
    writej(ev/'manifest.json',manifest)
    writej(out/'evaluation.json',{'directory':ev.name,'checkpoint_sha256':ck})
    summarize_run(out)

def summarize_run(out):
    start=time.monotonic(); ev=out/readj(out/'evaluation.json')['directory']
    tr=readj(out/'training_summary.json'); run=tr['run']; meta=metadata('dev'); y=meta['y']; patient=meta['patient']; avail=meta['avail7'].astype(bool)
    predictions={}
    for d in SHIFTS:
        with np.load(ev/f'dev_{d:+d}.npz') as f: predictions[d]=(f['p'],f['margin'])
    p0,m0=predictions[0]; names=patient_names('dev'); tables=[]; overall=[]; per=[]
    def row_for(mask,pid,d):
        p,m=predictions[d]; a=mask&avail
        row=dict(patient=pid,delta=d,**class_metrics(y[a],p[a]))
        row.update(compare(p[a],m[a],p0[a],m0[a]))
        row.update(N_to_V=int(np.sum(a&(p0<=.5)&(p>.5))),V_to_N=int(np.sum(a&(p0>.5)&(p<=.5))),flip_rate=float(np.mean((p[a]>.5)!=(p0[a]>.5))) if a.any() else None)
        for c,label in [(0,'N'),(1,'V')]:
            cm=a&(y==c)
            row[label+'_prediction_N_to_V']=int(np.sum(cm&(p0<=.5)&(p>.5)))
            row[label+'_prediction_V_to_N']=int(np.sum(cm&(p0>.5)&(p<=.5)))
        return row
    decisions=np.stack([predictions[d][0]>.5 for d in SHIFTS],axis=1)
    def robust(mask):
        a=mask&avail; n=a&(y==0); v=a&(y==1)
        nany=int(np.sum(np.any(decisions[n],axis=1))); vall=int(np.sum(np.all(decisions[v],axis=1)))
        return dict(N_common=int(n.sum()),V_common=int(v.sum()),N_any_wrong=nany,N_any_wrong_rate=nany/int(n.sum()) if n.any() else None,V_all_detected=vall,V_all_detected_rate=vall/int(v.sum()) if v.any() else None,any_flip=int(np.sum(np.any(decisions[a]!=decisions[a,3,None],axis=1))),any_flip_rate=float(np.mean(np.any(decisions[a]!=decisions[a,3,None],axis=1))) if a.any() else None)
    for pi,name in enumerate(names):
        mask=patient==pi; auc,ap=auc_ap(y[mask],p0[mask]); zero=class_metrics(y[mask],p0[mask])
        shifted=[row_for(mask,name,d) for d in SHIFTS]; tables.extend(shifted)
        worstN=max(shifted,key=lambda r:r['N_false_positive']); worstV=min(shifted,key=lambda r:r['V_detected'])
        per.append(dict(patient=name,zero=zero,auc=auc,ap=ap,**robust(mask),worst_N_delta=worstN['delta'],worst_N_false_positive=worstN['N_false_positive'],worst_V_delta=worstV['delta'],worst_V_detected=worstV['V_detected']))
    for d in SHIFTS:
        r=row_for(np.ones(len(y),bool),'pooled',d)
        items=[q for q in tables if q['delta']==d]
        r['macro_N_false_positive_rate']=float(np.mean([q['N_false_positive_rate'] for q in items if q['N']]))
        r['macro_V_detection_rate']=float(np.mean([q['V_detection_rate'] for q in items if q['V']]))
        overall.append(r)
    worstN=max(overall,key=lambda r:r['N_false_positive']); worstV=min(overall,key=lambda r:r['V_detected'])
    zero=class_metrics(y,p0); zero.update(macro_N_false_positive_rate=float(np.mean([r['zero']['N_false_positive_rate'] for r in per if r['zero']['N']])),macro_V_detection_rate=float(np.mean([r['zero']['V_detection_rate'] for r in per if r['zero']['V']])))
    zero['auc'],zero['ap']=auc_ap(y,p0)
    cohorts={}
    legacy_names=set(readj(ROOT/'patient_split_frozen.json')['legacy_split']['dev'])
    for cohort in ['legacy_dev','new_dev','combined_dev']:
        pis=[pi for pi,name in enumerate(names) if cohort=='combined_dev' or ((name in legacy_names)==(cohort=='legacy_dev'))]
        mask=np.isin(patient,pis); pp=[r for r in per if names.index(r['patient']) in pis]
        z=class_metrics(y[mask],p0[mask]); z['auc'],z['ap']=auc_ap(y[mask],p0[mask])
        z['macro_N_false_positive_rate']=float(np.mean([r['zero']['N_false_positive_rate'] for r in pp if r['zero']['N']]))
        z['macro_V_detection_rate']=float(np.mean([r['zero']['V_detection_rate'] for r in pp if r['zero']['V']])) if any(r['zero']['V'] for r in pp) else None
        cohorts[cohort]=dict(zero=z,robust=robust(mask),identity_patients=len(pis),N_patients=sum(r['zero']['N']>0 for r in pp),V_patients=sum(r['zero']['V']>0 for r in pp),macro_N_any_wrong_rate=float(np.mean([r['N_any_wrong_rate'] for r in pp if r['N_common']])),macro_V_all_detected_rate=float(np.mean([r['V_all_detected_rate'] for r in pp if r['V_common']])) if any(r['V_common'] for r in pp) else None,per_patient=pp)
    missed={}
    for r in csv.DictReader(open(ROOT.parent/'NV_BASELINE_R1/review_missed_V_dev.csv')): missed[f"{r['segment']} a{r['annotation_index']}"]=r['v3a_detected']=='True'
    ids=identity(meta,np.arange(len(y))); cases=[]
    for i,key in enumerate(ids):
        if key in missed: cases.append(dict(identity=key,new13=missed[key],p=[float(predictions[d][0][i]) for d in SHIFTS],margin=[float(predictions[d][1][i]) for d in SHIFTS],pattern=''.join('1' if q else '0' for q in decisions[i]),stable=bool(decisions[i].all())))
    assert len(cases)==15 and sum(r['new13'] for r in cases)==13
    p26=[]
    for d in SHIFTS:
        with np.load(ev/f'train_p02866_{d:+d}.npz') as f: p26.append(dict(delta=d,**class_metrics(f['y'],f['p'])))
    summary=dict(cohorts=cohorts,run=run,checkpoint_sha256=tr['best_checkpoint_sha256'],best_epoch=tr['best_epoch'],epochs=tr['epoch'],best_dev_balanced_ce=tr['best'],zero=zero,robust=robust(np.ones(len(y),bool)),worst_N_delta=worstN['delta'],worst_N_false_positive=worstN['N_false_positive'],worst_N_false_positive_rate=worstN['N_false_positive_rate'],worst_V_delta=worstV['delta'],worst_V_detected=worstV['V_detected'],worst_V_detection_rate=worstV['V_detection_rate'],per_patient=per,shifts=overall,missed15=cases,p02866_train=p26,repeat=readj(ev/'repeat.json'),evaluation_resources=readj(ev/'manifest.json')['resources'],statistics_seconds=time.monotonic()-start)
    writej(out/'metrics.json',summary)
    # JSON preserves nested error distributions; CSV offers flat count tables.
    for filename,rows in [('per_patient_shift.csv',tables),('overall_shift.csv',overall),('missed15.csv',cases)]:
        with open(out/filename,'w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    writej(out/'complete.json',dict(id=run['id'],finished=now(),config_sha256=tr['config_sha256'],checkpoint_sha256=tr['best_checkpoint_sha256'],metrics_sha256=sha(out/'metrics.json')))
    print('STATISTICS COMPLETE',run['id'],zero,flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(); p.add_argument('runid'); a=p.parse_args(); summarize_run(ROOT/'runs'/a.runid)
