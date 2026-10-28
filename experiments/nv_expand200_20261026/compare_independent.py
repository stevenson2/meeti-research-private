"""All-seed paired patient-cluster descriptive intervals; no fitting/selection."""
from common import *
import csv
from independent_evaluate import mean_defined
PAIRS=[('L_LP','L'),('L_LP','S_LP'),('S_LP','S'),('L','S')]
ENDPOINTS=['macro_N_false_positive_rate','macro_V_detection_rate','macro_N_any_wrong_rate','macro_V_all_detected_rate']
def csvout(path,rows):
    with path.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def rate(p,key):
    if key=='macro_N_false_positive_rate':return p['zero']['N_false_positive_rate']
    if key=='macro_V_detection_rate':return p['zero']['V_detection_rate']
    if key=='macro_N_any_wrong_rate':return p['N_any_wrong_rate']
    return p['V_all_detected_rate']
def main():
    status=readj(ROOT/'independent_status.json');assert len(status['runs'])==12 and all(r['status']=='completed' for r in status['runs'])
    protocol=readj(ROOT/'evaluation_protocol_frozen.json');pre=readj(ROOT/'independent_preparation.json');names=pre['patients'];models={};rows=[];patients=[]
    for r in status['runs']:
        out=ROOT/'independent'/r['id'];m=readj(out/'metrics.json');done=readj(out/'complete.json');assert sha(out/'metrics.json')==done['metrics_sha256'];models[(r['model'],r['seed'])]=m
        rows.append(dict(id=r['id'],model=r['model'],seed=r['seed'],identity_patients=m['identity_patients'],NV_patients=m['NV_patients'],N_patients=m['N_patients'],V_patients=m['V_patients'],**m['zero'],**m['robust'],auc=m['auc'],ap=m['ap'],**{k:m[k] for k in ENDPOINTS},macro_AUC=m['macro_AUC'],macro_AP=m['macro_AP'],worst_N_delta=m['worst_N_delta'],worst_N_false_positive=m['worst_N_false_positive'],worst_V_delta=m['worst_V_delta'],worst_V_detected=m['worst_V_detected']))
        for p in m['per_patient']:patients.append(dict(id=r['id'],patient=p['patient'],**p['zero'],**{k:v for k,v in p.items() if k not in ['patient','zero']}))
    boot=np.random.default_rng(protocol['intervals']['seed']).integers(0,len(names),size=(protocol['intervals']['draws'],len(names)));paired=[];changes=[];net=[]
    for left,right in PAIRS:
        raw={}
        for seed in [42,43,44]:
            a=models[(left,seed)];b=models[(right,seed)];am={p['patient']:p for p in a['per_patient']};bm={p['patient']:p for p in b['per_patient']};assert set(am)==set(bm)==set(names)
            dz={k:a['zero'][k]-b['zero'][k] if a['zero'][k] is not None and b['zero'][k] is not None else None for k in ['N_false_positive','V_detected','N_false_positive_rate','V_detection_rate']}
            dr={k:a['robust'][k]-b['robust'][k] if a['robust'][k] is not None and b['robust'][k] is not None else None for k in ['N_any_wrong','V_all_detected','N_any_wrong_rate','V_all_detected_rate']}
            net.append(dict(pair=left+'-'+right,seed=seed,**dz,**dr,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in ENDPOINTS+['auc','ap']}))
            for pid in names:
                ap=am[pid];bp=bm[pid];assert ap['zero']['N']==bp['zero']['N'] and ap['zero']['V']==bp['zero']['V']
                changes.append(dict(pair=left+'-'+right,seed=seed,patient=pid,N=ap['zero']['N'],V=ap['zero']['V'],delta_N_false_positive=ap['zero']['N_false_positive']-bp['zero']['N_false_positive'],delta_V_detected=ap['zero']['V_detected']-bp['zero']['V_detected'],delta_N_any_wrong=ap['N_any_wrong']-bp['N_any_wrong'],delta_V_all_detected=ap['V_all_detected']-bp['V_all_detected']))
            raw[seed]={key:np.array([(rate(am[p],key)-rate(bm[p],key)) if rate(am[p],key) is not None and rate(bm[p],key) is not None else np.nan for p in names]) for key in ENDPOINTS}
        for view in [42,43,44,'three_seed_mean']:
            for key in ENDPOINTS:
                if view=='three_seed_mean':
                    arrays=np.stack([raw[s][key] for s in [42,43,44]]);support=np.isfinite(arrays).all(0);delta=np.full(len(names),np.nan);delta[support]=arrays[:,support].mean(0)
                else:delta=raw[view][key];support=np.isfinite(delta)
                if support.any():
                    sampled=delta[boot];den=np.isfinite(sampled).sum(1);num=np.nansum(sampled,1);valid=den>0;dist=num[valid]/den[valid];point=float(delta[support].mean());lo,hi=(float(x) for x in np.quantile(dist,[.025,.975]))
                else:point=lo=hi=None;dist=[]
                paired.append(dict(pair=left+'-'+right,seed_view=view,endpoint=key,delta=point,delta_percentage_points=100*point if point is not None else None,CI95_low=lo,CI95_high=hi,CI95_low_pp=100*lo if lo is not None else None,CI95_high_pp=100*hi if hi is not None else None,support_patients=int(support.sum()),valid_bootstrap_draws=len(dist),multiplicity='unadjusted_descriptive_not_a_significance_winner_test'))
    csvout(ROOT/'independent_runs.csv',rows);csvout(ROOT/'independent_patients.csv',patients);csvout(ROOT/'independent_same_seed_pairs.csv',net);csvout(ROOT/'independent_same_seed_patient_changes.csv',changes);csvout(ROOT/'independent_paired_intervals.csv',paired)
    writej(ROOT/'independent_comparison.json',dict(finished=now(),training_config_sha256=sha(ROOT/'frozen_config.json'),release_gate_sha256=sha(ROOT/'checkpoints_frozen_for_independent.json'),evaluation_protocol_sha256=sha(ROOT/'evaluation_protocol_frozen.json'),runs=rows,same_seed_pairs=net,paired_intervals=paired,method=protocol['intervals'],no_post_test_training_or_threshold_selection=True))
    print('INDEPENDENT ALL12 COMPARISON AND CLUSTER INTERVALS COMPLETE',flush=True)
if __name__=='__main__':main()
