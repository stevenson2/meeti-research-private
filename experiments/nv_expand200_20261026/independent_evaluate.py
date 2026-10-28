"""Fixed-checkpoint independent inference. No training/selection API."""
from common import *
from evaluate import auc_ap,class_metrics
import csv,argparse
def mean_defined(values):
    v=[x for x in values if x is not None];return float(np.mean(v)) if v else None
def summarize_predictions(meta,names,predictions):
    y=meta['y'];patient=meta['patient'];avail=meta['avail7'].astype(bool);p0,m0=predictions[0]
    decisions=np.stack([predictions[d][0]>.5 for d in SHIFTS],1)
    def robust(mask):
        a=mask&avail;n=a&(y==0);v=a&(y==1);nn=int(n.sum());nv=int(v.sum());nw=int(decisions[n].any(1).sum());vd=int(decisions[v].all(1).sum());flip=int(np.any(decisions[a]!=decisions[a,3,None],1).sum())
        return dict(N_common=nn,V_common=nv,N_any_wrong=nw,V_all_detected=vd,N_any_wrong_rate=nw/nn if nn else None,V_all_detected_rate=vd/nv if nv else None,any_flip=flip,any_flip_rate=flip/int(a.sum()) if a.any() else None)
    def shifted(mask,pid,d):
        a=mask&avail;p,m=predictions[d];row=dict(patient=pid,delta=d,**class_metrics(y[a],p[a]),**compare(p[a],m[a],p0[a],m0[a]))
        row.update(N_prediction_to_V=int((a&(p0<=.5)&(p>.5)).sum()),V_prediction_to_N=int((a&(p0>.5)&(p<=.5)).sum()))
        for c,label in [(0,'N'),(1,'V')]:
            cm=a&(y==c);row[label+'_prediction_N_to_V']=int((cm&(p0<=.5)&(p>.5)).sum());row[label+'_prediction_V_to_N']=int((cm&(p0>.5)&(p<=.5)).sum())
        return row
    per=[];allshifts=[];classchanges=[]
    for pi,pid in list(enumerate(names))+[(-1,'pooled')]:
        mask=np.ones(len(y),bool) if pi==-1 else patient==pi
        z=class_metrics(y[mask],p0[mask]);auc,ap=auc_ap(y[mask],p0[mask]);ss=[shifted(mask,pid,d) for d in SHIFTS];allshifts.extend(ss)
        row=dict(patient=pid,zero=z,auc=auc,ap=ap,**robust(mask),worst_N_delta=max(ss,key=lambda r:r['N_false_positive'])['delta'] if any(r['N'] for r in ss) else None,worst_N_false_positive=max(r['N_false_positive'] for r in ss),worst_V_delta=min(ss,key=lambda r:r['V_detected'])['delta'] if any(r['V'] for r in ss) else None,worst_V_detected=min(r['V_detected'] for r in ss))
        if pi>=0:per.append(row)
        else:pooled=row
        for d in SHIFTS:
            p,m=predictions[d]
            for c,label in [(0,'N'),(1,'V')]:
                cm=mask&avail&(y==c);n=int(cm.sum());dp=p[cm].astype(float)-p0[cm];dm=m[cm].astype(float)-m0[cm]
                classchanges.append(dict(patient=pid,truth=label,delta=d,support=n,N_prediction_to_V=int(((p0[cm]<=.5)&(p[cm]>.5)).sum()),V_prediction_to_N=int(((p0[cm]>.5)&(p[cm]<=.5)).sum()),mean_abs_delta_p=float(np.abs(dp).mean()) if n else None,max_abs_delta_p=float(np.abs(dp).max()) if n else None,mean_signed_delta_p=float(dp.mean()) if n else None,mean_abs_delta_margin=float(np.abs(dm).mean()) if n else None,max_abs_delta_margin=float(np.abs(dm).max()) if n else None,mean_signed_delta_margin=float(dm.mean()) if n else None))
    overall=[r for r in allshifts if r['patient']=='pooled']
    for d in SHIFTS:
        ss=[r for r in allshifts if r['patient']!='pooled' and r['delta']==d];r=next(r for r in overall if r['delta']==d);r['macro_N_false_positive_rate']=mean_defined([x['N_false_positive_rate'] for x in ss]);r['macro_V_detection_rate']=mean_defined([x['V_detection_rate'] for x in ss])
    macro=dict(macro_N_false_positive_rate=mean_defined([r['zero']['N_false_positive_rate'] for r in per]),macro_V_detection_rate=mean_defined([r['zero']['V_detection_rate'] for r in per]),macro_N_any_wrong_rate=mean_defined([r['N_any_wrong_rate'] for r in per]),macro_V_all_detected_rate=mean_defined([r['V_all_detected_rate'] for r in per]),macro_AUC=mean_defined([r['auc'] for r in per if r['zero']['N'] and r['zero']['V']]),macro_AP=mean_defined([r['ap'] for r in per if r['zero']['N'] and r['zero']['V']]))
    return dict(identity_patients=len(names),NV_patients=sum(r['zero']['N']+r['zero']['V']>0 for r in per),N_patients=sum(r['zero']['N']>0 for r in per),V_patients=sum(r['zero']['V']>0 for r in per),zero=pooled['zero'],auc=pooled['auc'],ap=pooled['ap'],robust={k:pooled[k] for k in ['N_common','V_common','N_any_wrong','V_all_detected','N_any_wrong_rate','V_all_detected_rate','any_flip','any_flip_rate']},**macro,per_patient=per,shifts=overall,worst_N_delta=pooled['worst_N_delta'],worst_N_false_positive=pooled['worst_N_false_positive'],worst_V_delta=pooled['worst_V_delta'],worst_V_detected=pooled['worst_V_detected'],worst_macro_N_false_positive_rate=max((r['macro_N_false_positive_rate'] for r in overall if r['macro_N_false_positive_rate'] is not None),default=None),worst_macro_V_detection_rate=min((r['macro_V_detection_rate'] for r in overall if r['macro_V_detection_rate'] is not None),default=None)),allshifts,classchanges
def csvout(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def main(runid):
    bootstrap()
    from custody_prepare import release_gate
    gate=release_gate();pre=readj(ROOT/'independent_preparation.json');assert pre['release_gate_sha256']==sha(ROOT/'checkpoints_frozen_for_independent.json')
    import tensorflow as tf
    from models import build,FixedLowpass
    runtime=json.loads(json.dumps(configure_tf()));cfg=readj(ROOT/'frozen_config.json');assert runtime==cfg['runtime']
    for name,h in cfg['script_hashes'].items():assert sha(ROOT/name)==h
    run=next(r for r in readj(ROOT/'matrix.json')['runs'] if r['id']==runid);ck=gate['checkpoints'][runid]['sha256'];checkpoint=ROOT/'runs'/runid/'best.weights.h5';assert sha(checkpoint)==ck
    cache=Path(pre['data_root']);assert sha(cache/'meta.npz')==pre['meta_sha256']
    X=np.load(cache/'X.npy',mmap_mode='r')
    with np.load(cache/'meta.npz') as f:meta={k:f[k] for k in f.files}
    names=pre['patients'];out=ROOT/'independent'/runid;out.mkdir(parents=True,exist_ok=True);ev=out/('eval_'+ck[:16]);ev.mkdir(exist_ok=True)
    manifest=readj(ev/'manifest.json') if (ev/'manifest.json').exists() else dict(started=now(),run=run,checkpoint_sha256=ck,training_config_sha256=sha(ROOT/'frozen_config.json'),release_gate_sha256=sha(ROOT/'checkpoints_frozen_for_independent.json'),predictions={})
    assert manifest['checkpoint_sha256']==ck
    model=build(run['model'],run['seed']);model.load_weights(checkpoint)
    assert all(not l.bypass for l in model.layers if isinstance(l,FixedLowpass))
    forward=make_forward(model);bn0=bn_state(model);rows=np.arange(len(meta['y']));start=time.monotonic()
    def save(d,repeat=False):
        name=f'independent_{d:+d}'+('_repeat' if repeat else '');p=ev/(name+'.npz')
        if p.exists() and name in manifest['predictions']:assert sha(p)==manifest['predictions'][name]['sha256'];return
        pr,margin,nll,timing=infer(forward,X,meta,rows,SHIFTS.index(d));assert np.isfinite(pr).all() and np.isfinite(margin).all()
        np.savez(p,p=pr,margin=margin,rows=rows,**meta)
        manifest['predictions'][name]=dict(sha256=sha(p),**timing);writej(ev/'manifest.json',manifest);print('INDEPENDENT PREDICTION',runid,name,flush=True)
    save(0);save(0,True)
    with np.load(ev/'independent_+0.npz') as a,np.load(ev/'independent_+0_repeat.npz') as b:repeat=compare(a['p'],a['margin'],b['p'],b['margin'])
    assert repeat['flips']==0
    for d in SHIFTS:save(d)
    assert all(np.array_equal(a,b) for a,b in zip(bn0,bn_state(model)))
    predictions={}
    for d in SHIFTS:
        with np.load(ev/f'independent_{d:+d}.npz') as f:
            assert all(np.array_equal(f[k],meta[k]) for k in meta);predictions[d]=(f['p'],f['margin'])
    stats,patientshifts,changes=summarize_predictions(meta,names,predictions)
    manifest.update(bn_unchanged=True,repeat=repeat,finished=now(),seconds_this_process=time.monotonic()-start,runtime=runtime,resources=resources());writej(ev/'manifest.json',manifest)
    stats.update(run=run,checkpoint_sha256=ck,repeat=repeat,resources=manifest['resources'],selection='Fixed checkpoint chosen only on development CE; independent data never used for fitting/threshold/seed selection')
    writej(out/'metrics.json',stats);csvout(out/'per_patient_shift.csv',patientshifts);csvout(out/'class_shift_changes.csv',changes);writej(out/'evaluation.json',dict(directory=ev.name,checkpoint_sha256=ck));writej(out/'complete.json',dict(id=runid,finished=now(),checkpoint_sha256=ck,metrics_sha256=sha(out/'metrics.json'),release_gate_sha256=sha(ROOT/'checkpoints_frozen_for_independent.json')))
    print('INDEPENDENT RUN COMPLETE',runid,flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runid');main(p.parse_args().runid)
