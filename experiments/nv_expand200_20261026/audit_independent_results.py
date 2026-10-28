"""Recompute fixed independent predictions; never trains or changes thresholds."""
from common import *
from scipy.stats import rankdata
import math
def equal_rate(actual,num,den):
    if den:assert actual is not None and math.isclose(actual,num/den,rel_tol=0,abs_tol=1e-12)
    else:assert actual is None
def main():
    status=readj(ROOT/'independent_status.json');assert len(status['runs'])==12 and all(r['status']=='completed' for r in status['runs'])
    pre=readj(ROOT/'independent_preparation.json');cache=Path(pre['data_root']);assert sha(cache/'X.npy')==pre['X_sha256'] and sha(cache/'meta.npz')==pre['meta_sha256']
    with np.load(cache/'meta.npz') as f:meta={k:f[k] for k in f.files}
    y=meta['y'];av=meta['avail7'].astype(bool);rows=np.arange(len(y));gate=readj(ROOT/'checkpoints_frozen_for_independent.json');audits=[]
    for r in status['runs']:
        out=ROOT/'independent'/r['id'];m=readj(out/'metrics.json');done=readj(out/'complete.json');ev=out/readj(out/'evaluation.json')['directory'];manifest=readj(ev/'manifest.json')
        assert sha(out/'metrics.json')==done['metrics_sha256'] and m['checkpoint_sha256']==gate['checkpoints'][r['id']]['sha256']
        assert len(manifest['predictions'])==8 and manifest['bn_unchanged'] and manifest['repeat']['exact']
        for name,info in manifest['predictions'].items():assert sha(ev/(name+'.npz'))==info['sha256']
        ps=[]
        for d in SHIFTS:
            with np.load(ev/f'independent_{d:+d}.npz') as f:
                assert np.array_equal(f['rows'],rows) and all(np.array_equal(f[k],meta[k]) for k in meta);ps.append(f['p'])
        p0=ps[3];decision=np.stack([p>.5 for p in ps],1)
        fp=int(((y==0)&(p0>.5)).sum());tp=int(((y==1)&(p0>.5)).sum());nw=int(decision[av&(y==0)].any(1).sum());vd=int(decision[av&(y==1)].all(1).sum())
        assert (fp,tp,nw,vd)==(m['zero']['N_false_positive'],m['zero']['V_detected'],m['robust']['N_any_wrong'],m['robust']['V_all_detected'])
        nN=int((y==0).sum());nV=int((y==1).sum())
        assert (m['zero']['N'],m['zero']['V'])==(nN,nV)
        assert (m['robust']['N_common'],m['robust']['V_common'])==(int((av&(y==0)).sum()),int((av&(y==1)).sum()))
        equal_rate(m['zero']['N_false_positive_rate'],fp,nN);equal_rate(m['zero']['V_detection_rate'],tp,nV)
        equal_rate(m['robust']['N_any_wrong_rate'],nw,m['robust']['N_common']);equal_rate(m['robust']['V_all_detected_rate'],vd,m['robust']['V_common'])
        auc=float((rankdata(p0.astype(np.float64),method='average')[y==1].sum()-nV*(nV+1)/2)/(nV*nN)) if nN and nV else None
        if auc is not None:assert abs(auc-m['auc'])<1e-12
        rates={k:[] for k in ['macro_N_false_positive_rate','macro_V_detection_rate','macro_N_any_wrong_rate','macro_V_all_detected_rate']};supports=dict(NV_patients=0,N_patients=0,V_patients=0)
        assert len(m['per_patient'])==len(pre['patients']) and {p['patient'] for p in m['per_patient']}==set(pre['patients'])
        for pi,pid in enumerate(pre['patients']):
            mask=meta['patient']==pi;pp=next(p for p in m['per_patient'] if p['patient']==pid)
            assert int((mask&(y==0)&(p0>.5)).sum())==pp['zero']['N_false_positive'] and int((mask&(y==1)&(p0>.5)).sum())==pp['zero']['V_detected']
            assert int(decision[mask&av&(y==0)].any(1).sum())==pp['N_any_wrong'] and int(decision[mask&av&(y==1)].all(1).sum())==pp['V_all_detected']
            pn=int((mask&(y==0)).sum());pv=int((mask&(y==1)).sum());cn=int((mask&av&(y==0)).sum());cv=int((mask&av&(y==1)).sum())
            assert (pp['zero']['N'],pp['zero']['V'],pp['N_common'],pp['V_common'])==(pn,pv,cn,cv)
            supports['NV_patients']+=int(pn+pv>0);supports['N_patients']+=int(pn>0);supports['V_patients']+=int(pv>0)
            for key,value,num,den in [('macro_N_false_positive_rate',pp['zero']['N_false_positive_rate'],pp['zero']['N_false_positive'],pn),('macro_V_detection_rate',pp['zero']['V_detection_rate'],pp['zero']['V_detected'],pv),('macro_N_any_wrong_rate',pp['N_any_wrong_rate'],pp['N_any_wrong'],cn),('macro_V_all_detected_rate',pp['V_all_detected_rate'],pp['V_all_detected'],cv)]:
                equal_rate(value,num,den)
                if den:rates[key].append(num/den)
        assert m['identity_patients']==len(pre['patients']) and all(m[k]==v for k,v in supports.items())
        for key,values in rates.items():equal_rate(m[key],math.fsum(values),len(values))
        audits.append(dict(id=r['id'],all8_prediction_hashes_match=True,identities_match=True,counts_and_per_patient_robust_recomputed=True,all_primary_denominators_and_patient_equal_rates_recomputed=True,empty_and_single_class_identities_preserved=True,AUC_rank_sum_agrees=True,zero_repeat_exact=True,bn_unchanged=True,checkpoint_sha256=m['checkpoint_sha256']))
    writej(ROOT/'independent_prediction_audit.json',dict(passed=True,finished=now(),release_gate_sha256=sha(ROOT/'checkpoints_frozen_for_independent.json'),cache_hashes_unchanged=True,no_training_fit=True,no_threshold_or_checkpoint_changes=True,runs=audits));print('ALL96 INDEPENDENT PREDICTIONS AUDITED',flush=True)
if __name__=='__main__':main()
