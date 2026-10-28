"""Descriptive development comparison from frozen saved outputs; no fitting."""
from common import *
import csv,argparse
def csvout(p,rows):
    with p.open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main(require_complete=False):
    frozen=ROOT/'development_comparison_frozen.json'
    if frozen.exists():
        prior=readj(frozen)
        assert sha(ROOT/'development_comparison.json')==prior['comparison_sha256']
        assert sha(ROOT/'development_runs.csv')==prior['runs_csv_sha256']
        assert sha(ROOT/'development_same_seed_pairs.csv')==prior['pairs_csv_sha256']
        print('DEVELOPMENT COMPARISON ALREADY FROZEN; unchanged');return
    status=readj(ROOT/'status.json');rows=[];patients=[];metrics={}
    for r in status['runs']:
        if r['status']!='completed':continue
        out=ROOT/'runs'/r['id'];m=readj(out/'metrics.json');t=readj(out/'training_summary.json');metrics[r['id']]=m
        ev=out/readj(out/'evaluation.json')['directory'];manifest=readj(ev/'manifest.json')
        for cohort,c in m['cohorts'].items():
            z=c['zero'];b=c['robust']
            rows.append(dict(id=r['id'],model=r['model'],seed=r['seed'],cohort=cohort,epochs=t['epoch'],best_epoch=t['best_epoch'],updates=t['epoch']*STEPS,best_selection_CE=t['best'],identity_patients=c['identity_patients'],N_patients=c['N_patients'],V_patients=c['V_patients'],**z,**b,macro_N_any_wrong_rate=c['macro_N_any_wrong_rate'],macro_V_all_detected_rate=c['macro_V_all_detected_rate'],train_compute_seconds=sum(d['train_compute_seconds'] for d in t['durations']),train_read_seconds=sum(d['read_seconds'] for d in t['durations']),selection_dev_seconds=sum(d['dev_seconds'] for d in t['durations']),fixed_evaluation_seconds=manifest['seconds_this_process']))
        for p in m['per_patient']:patients.append(dict(id=r['id'],patient=p['patient'],**p['zero'],**{k:v for k,v in p.items() if k not in ['patient','zero']}))
    pairs=[]
    cols=['N_false_positive','V_detected','N_false_positive_rate','V_detection_rate','macro_N_false_positive_rate','macro_V_detection_rate','N_any_wrong','V_all_detected','N_any_wrong_rate','V_all_detected_rate','macro_N_any_wrong_rate','macro_V_all_detected_rate','auc','ap']
    for left,right in [('L_LP','L'),('L_LP','S_LP'),('S_LP','S'),('L','S')]:
        for seed in [42,43,44]:
            for cohort in ['legacy_dev','new_dev','combined_dev']:
                a=next((x for x in rows if x['id']==f'{left}_C_s{seed}' and x['cohort']==cohort),None);b=next((x for x in rows if x['id']==f'{right}_C_s{seed}' and x['cohort']==cohort),None)
                if a is None or b is None:continue
                pairs.append(dict(pair=left+'-'+right,seed=seed,cohort=cohort,**{k:a[k]-b[k] if a[k] is not None and b[k] is not None else None for k in cols}))
    if rows:csvout(ROOT/'development_runs.csv',rows);csvout(ROOT/'development_patients.csv',patients)
    if pairs:csvout(ROOT/'development_same_seed_pairs.csv',pairs)
    summary=dict(updated=now(),training_config_sha256=sha(ROOT/'frozen_config.json'),completed=sum(r['status']=='completed' for r in status['runs']),failed=sum(r['status']=='failed' for r in status['runs']),runs=rows,pairs=pairs,independent_content_used=False)
    writej(ROOT/'development_comparison.json',summary)
    if require_complete:
        assert len(status['runs'])==12 and all(r['status']=='completed' for r in status['runs'])
        frozen=ROOT/'development_comparison_frozen.json'
        obj=dict(frozen_at=now(),training_config_sha256=sha(ROOT/'frozen_config.json'),comparison_sha256=sha(ROOT/'development_comparison.json'),runs_csv_sha256=sha(ROOT/'development_runs.csv'),pairs_csv_sha256=sha(ROOT/'development_same_seed_pairs.csv'),selected_checkpoints={r['id']:readj(ROOT/'runs'/r['id']/'complete.json')['checkpoint_sha256'] for r in status['runs']},no_independent_content_used=True,primary_pairs=['L_LP-L','L_LP-S_LP'],background_pairs=['S_LP-S','L-S'],interpretation='Descriptive all-seed same-cohort comparisons, no model/seed/threshold chosen from independent data')
        assert not frozen.exists(),'Never overwrite frozen development conclusions'
        writej(frozen,obj)
    print('DEVELOPMENT COMPARISON',summary['completed'],'/12 completed',len(pairs),'paired cohort rows')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args();main(a.freeze)
