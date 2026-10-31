import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import collections,csv,hashlib,json,math
import numpy as np
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/PC_REPLAY_REPAIR_20261030'
def read(p):
    with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
freeze=json.loads((O/'INFERENCE_FREEZE.json').read_text());progress=json.loads((O/'inference_progress.json').read_text())
assert progress['completed_segments']==20 and progress['model_invocations_done']==6*freeze['candidates']
config=json.loads((ROOT/'outputs/PTQ_REPAIR_20261029/deployment_config.json').read_text())
for key,entry in config.items():
    assert hashlib.sha256((ROOT/'outputs/PTQ_REPAIR_20261029'/entry['file']).read_bytes()).hexdigest()==freeze['models'][key]
product=Path(os.environ['ECG_PRODUCT_ROOT'])
for name,digest in json.loads((O/'source_snapshot.json').read_text()).items():
    if name.startswith('components/'):
        assert hashlib.sha256((product/name).read_bytes()).hexdigest()==digest
source=json.loads((O/'preprocess_progress.json').read_text())['segments'];keys=list(freeze['models'])
allrows=read(O/'review_lists.csv');assert len(allrows)==90
assert hashlib.sha256((W/'PREREG.json').read_bytes()).hexdigest()==freeze['recipe_sha256']
verified_scores=0;verified_candidates=0
for s in source:
    name=s['segment'];meta=np.load(W/'candidate_cache'/f'{name}_meta.npz');z=np.load(W/'scores'/f'{name}.npz')
    rec=json.loads((W/'scores'/f'{name}.json').read_text())
    assert rec['models']==freeze['models'] and rec['cache_sha256']==freeze['candidate_cache_sha256'][name]
    assert hashlib.sha256((W/'candidate_cache'/f'{name}.bin').read_bytes()).hexdigest()==rec['cache_sha256']
    assert hashlib.sha256((W/'scores'/f'{name}.npz').read_bytes()).hexdigest()==rec['score_sha256']
    assert z['q'].shape==(6,s['candidates'],2) and z['q'].dtype==np.int8
    for row in [r for r in allrows if r['segment']==name]:
        ki=keys.index(row['key']);wi=int(row['trigger_candidate_index']);g=int(row['grid_index'])
        assert int(meta['grid'][wi])==g and bool(meta['eligible'][wi])
        q=z['q'][ki,wi];assert [int(row['qN']),int(row['qV'])]==q.tolist()
        assert int(q[1])-int(q[0])==int(row['integer_margin'])
        assert [int(row['n_'+c]) for c in 'NSVQ']==meta['grid_reference_counts'][g].tolist()
        verified_scores+=1
    verified_candidates+=s['candidates'];meta.close();z.close()
ties=[];rare=[]
rs={r['patient']:r for r in read(O/'random_baseline_per_patient.csv')}
rr=read(O/'rare_V_results.csv')
for patient in sorted(rs):
    grid=np.load(W/'grid_scores'/f'{patient}.npz');b=grid['base'];m=grid['margin'];n=len(b);g=int(np.sum(b[:,4]>0))
    # Independent exact product form; avoids relying on the evaluation's comb() calculation.
    p=1-np.prod([(n-g-j)/(n-j) for j in range(3)])
    assert abs(float(rs[patient]['random_D3_probability'])-p)<1e-12
    assert n==int(rs[patient]['pool_grids']) and g==int(rs[patient]['V_supported_grids'])
    for ki,key in enumerate(keys):
        records=sorted([r for r in allrows if r['key']==key and r['patient']==patient],key=lambda r:int(r['rank']))
        # Python tuple sort, separate from NumPy lexsort used in primary evaluator.
        order=sorted(range(n),key=lambda j:(-int(m[ki,j]),source[int(b[j,0])]['segment'],int(b[j,1])))
        expected=[(source[int(b[j,0])]['segment'],int(b[j,1])) for j in order[:3]]
        assert [(r['segment'],int(r['grid_index'])) for r in records]==expected
        third=int(m[ki,order[2]]);top=int(m[ki,order[0]])
        ties.append({'key':key,'patient':patient,'top_margin':top,'third_margin':third,
            'grids_at_top_margin':int(np.sum(m[ki]==top)),'grids_at_third_margin':int(np.sum(m[ki]==third)),
            'grids_above_third_margin':int(np.sum(m[ki]>third)),
            'any_Q_top3_items':sum(int(r['n_Q'])>0 for r in records),
            'any_S_top3_items':sum(int(r['n_S'])>0 for r in records)})
        events=[r for r in rr if r['key']==key and r['patient']==patient]
        if events:
            matched=sum(r['has_real_QRS_match_within150ms']=='True' for r in events)
            captured=sum(r['reference_V_in_top3']=='True' for r in events)
            true_selected=[r for r in records if int(r['n_V'])>0]
            evgrids={(r['segment'],int(r['grid_index'])) for r in events}
            falsegrids=[int(m[ki,j]) for j in range(n) if (source[int(b[j,0])]['segment'],int(b[j,1])) not in evgrids]
            ranks=[int(r['reference_grid_rank']) for r in events if r['reference_grid_rank']]
            rare.append({'key':key,'patient':patient,'reference_V':len(events),'QRS_matched_V':matched,'V_in_top3':captured,
                'patient_hit':captured>0,'V_supported_top3_items':len(true_selected),
                'best_V_grid_rank':min(ranks) if ranks else '',
                'best_V_grid_margin':max(int(r['reference_grid_margin']) for r in events if r['reference_grid_margin']) if ranks else '',
                'third_top3_margin':third,'highest_no_referenceV_margin':max(falsegrids),
                'random_D3_probability':p,'random_expected_V_captured':3*sum(r['reference_grid_has_any_trigger']=='True' for r in events)/n})
    grid.close()
assert verified_scores==90 and verified_candidates==freeze['candidates']
write('score_tie_and_background_audit.csv',ties);write('rare_patient_summary.csv',rare)
random=json.loads((O/'random_baseline_exact.json').read_text());prs=read(ROOT/'outputs/PVC_EVIDENCE_REVIEW_PLAN_20261029/patient_inventory.csv')
groups={'rare_fixed5':sorted(rs)}
for group,people in groups.items():
    dist=np.array([1.])
    for patient in people:
        p=float(rs[patient]['random_D3_probability']);dist=np.convolve(dist,[1-p,p])
    random[group]['random_count_probability_mass']=dist.tolist()
    random[group]['probability_all_patients_hit']=float(dist[-1])
    random[group]['expected_count_95pct_range']=[int(np.searchsorted(np.cumsum(dist),.025)),int(np.searchsorted(np.cumsum(dist),.975))]
(O/'random_baseline_exact.json').write_text(json.dumps(random,indent=2),encoding='utf-8')
info={'status':'PASS','segments':20,'models':6,'generated_candidates':verified_candidates,'REF_invocations':6*verified_candidates,
    'review_rows_verified_to_raw_INT8_scores_and_full_reference_counts':verified_scores,
    'all_30_top3_lists_independently_recomputed':True,'random3_formula_independently_verified_all5':True,
    'model_cache_score_hashes_verified':True,'shared_product_source_hashes_unchanged':True,
    'ranking_uses_only_actual_INT8_margin_and_fixed_identity_tie_break':True,
    'scope':'Computational/reference-label audit, not human ECG readability adjudication'}
(O/'RESULT_AUDIT.json').write_text(json.dumps(info,indent=2),encoding='utf-8');print(json.dumps(info,indent=2))
