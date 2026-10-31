import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];B=R/'work/replay_repair';P=R/'outputs/PC_REPLAY_REPAIR_20261030';O=R/'outputs/IDEAL_R_FINAL_DIAGNOSTIC_20261030'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((O/'PREREG.json').read_text(encoding='utf-8'));assert json.loads((O/'OUTPUT_VERIFICATION.json').read_text())['status']=='PASS'
cases=json.loads((W/'cases.json').read_text());z=np.load(O/'diagnostic_outputs.npz');keys=z['keys'].tolist();q=z['q'].astype(np.int16);m=q[:,:,:,1]-q[:,:,:,0]
ss=sorted(json.loads((P/'preprocess_progress.json').read_text())['segments'],key=lambda x:(x['patient'],x['segment']));thresholds={(x['patient'],x['key']):x for x in read(O/'FROZEN_THIRD_BACKGROUND.csv')}
comparisons=[];lists=[];patient_rows=[];variants=['actual','referenceR_replace','optimistic_credit'];oldlists=read(P/'review_lists.csv')
def category(row):return 'V_supported' if row[4]>0 else 'Q_undetermined' if row[5]>0 else 'S_present_noV' if row[3]>0 else 'N_only_noV' if row[2]>0 else 'no_reference_beat'
for patient in protocol['patients']:
 grid=np.load(B/'grid_scores'/f'{patient}.npz');base=grid['base'];original=grid['margin'].copy();ref=original.copy();optimistic=original.copy();lookup={(int(b[0]),int(b[1])):j for j,b in enumerate(base)}
 assert sha(B/'grid_scores'/f'{patient}.npz')==protocol['baseline_grid_sha256'][patient]
 relevant=[(ci,c) for ci,c in enumerate(cases) if c['patient']==patient];bg=np.flatnonzero(base[:,4]==0)
 for ci,c in relevant:
  sid=c['segment_index'];g=c['grid_index'];gi=lookup[(sid,g)];meta=np.load(B/'candidate_cache'/f'{c["segment"]}_meta.npz');score=np.load(B/'scores'/f'{c["segment"]}.npz');wi=c['actual_candidate_index']
  assert int(meta['grid'][wi])==g and int(meta['matched_annotation_index'][wi])==c['annotation_index']
  ii=np.flatnonzero((meta['grid']==g)&meta['eligible']);rest=ii[ii!=wi];actual_margin=score['q'][:,:,1].astype(np.int16)-score['q'][:,:,0].astype(np.int16)
  assert np.array_equal(actual_margin[:,wi],m[:,0,ci])
  restmax=actual_margin[:,rest].max(axis=1) if len(rest) else np.full(6,-512,np.int16)
  ref[:,gi]=np.maximum(restmax,m[:,1,ci]);optimistic[:,gi]=np.maximum(original[:,gi],m[:,1,ci])
  for ki,key in enumerate(keys):
   t=thresholds[(patient,key)];cut=int(t['third_background_margin']);identity=(c['segment'],g)
   def bg_rank(scorevalue):
    return 1+sum(int(original[ki,j])>scorevalue or (int(original[ki,j])==scorevalue and (ss[int(base[j,0])]['segment'],int(base[j,1]))<identity) for j in bg)
   aa=int(m[ki,0,ci]);oo=int(m[ki,1,ci]);comparisons.append({**c,'key':key,'actual_qN':int(q[ki,0,ci,0]),'actual_qV':int(q[ki,0,ci,1]),'reference_qN':int(q[ki,1,ci,0]),'reference_qV':int(q[ki,1,ci,1]),
    'actual_margin':aa,'referenceR_margin':oo,'reference_minus_actual':oo-aa,'third_background_margin':cut,'actual_third_overall_margin':int(t['actual_third_overall_margin']),
    'actual_strictly_above_third_background':aa>cut,'reference_strictly_above_third_background':oo>cut,'reference_ties_third_background':oo==cut,
    'reference_below_third_background':oo<cut,'actual_input_rank_against_fixed_background':bg_rank(aa),'reference_input_rank_against_fixed_background':bg_rank(oo),
    'actual_grid_margin':int(original[ki,gi]),'reference_replaced_grid_margin':int(ref[ki,gi]),'optimistic_grid_margin':int(optimistic[ki,gi])})
  meta.close();score.close()
 for name,values in [('actual',original),('referenceR_replace',ref),('optimistic_credit',optimistic)]:
  assert np.array_equal(values[:,bg],original[:,bg]),'background changed'
  for ki,key in enumerate(keys):
   order=sorted(range(len(base)),key=lambda j:(-int(values[ki,j]),ss[int(base[j,0])]['segment'],int(base[j,1])))[:3];rr=[]
   for rank,j in enumerate(order,1):
    row=base[j];rec={'variant':name,'key':key,'patient':patient,'rank':rank,'segment':ss[int(row[0])]['segment'],'grid_index':int(row[1]),'start_seconds':int(row[1])*4,'end_seconds':int(row[1])*4+4,
      'margin':int(values[ki,j]),'n_N':int(row[2]),'n_S':int(row[3]),'n_V':int(row[4]),'n_Q':int(row[5]),'category':category(row)};lists.append(rec);rr.append(rec)
   co=collections.Counter(x['category'] for x in rr)
   patient_rows.append({'variant':name,'key':key,'patient':patient,'D3_hit':co['V_supported']>0,'referenceV_captured':sum(x['n_V'] for x in rr),
     'V_supported_items':co['V_supported'],'Q_noV_items':co['Q_undetermined'],'N_only_items':co['N_only_noV'],'S_noV_items':co['S_present_noV'],'no_reference_beat_items':co['no_reference_beat'],'review_items':3})
   if name=='actual':
    old=sorted([r for r in oldlists if r['key']==key and r['patient']==patient],key=lambda r:int(r['rank']))
    assert [(r['segment'],r['grid_index'],r['margin']) for r in rr]==[(r['segment'],int(r['grid_index']),int(r['integer_margin'])) for r in old]
 grid.close()
summary=[]
for key in keys:
 cc=[x for x in comparisons if x['key']==key];rec={'key':key,'referenceV':12,'actual_strict_crossings':sum(x['actual_strictly_above_third_background'] for x in cc),
  'reference_strict_crossings':sum(x['reference_strictly_above_third_background'] for x in cc),'reference_ties':sum(x['reference_ties_third_background'] for x in cc),
  'reference_below':sum(x['reference_below_third_background'] for x in cc),'reference_score_improved_beats':sum(x['reference_minus_actual']>0 for x in cc)}
 for variant in variants:
  rr=[x for x in patient_rows if x['key']==key and x['variant']==variant]
  for field in ['D3_hit','referenceV_captured','V_supported_items','Q_noV_items','N_only_items','S_noV_items','no_reference_beat_items','review_items']:
   rec[variant+'_'+field]=sum(x[field] for x in rr)
 summary.append(rec)
write('per_V_model_comparison.csv',comparisons);write('hypothetical_review_lists.csv',lists);write('patient_diagnostic_results.csv',patient_rows);write('model_diagnostic_summary.csv',summary)
strict=sum(x['reference_strictly_above_third_background'] for x in comparisons);ties=sum(x['reference_ties_third_background'] for x in comparisons)
weak=sum(x['optimistic_credit_D3_hit']<3 for x in summary);seal=(72-strict)>36 and weak>=4
decision={'status':'SEALED_NO_FURTHER_TWEAKS' if seal else 'DIAGNOSIS_COMPLETE_STOP_RULE_NOT_MET','reference_strict_crossings_out_of72':strict,'reference_ties_out_of72':ties,
 'reference_not_strictly_above_third_out_of72':72-strict,'models_optimistic_hit_fewer_than3of5':weak,'stop_rule_met':seal,
 'stop_rule_frozen_before_outputs':protocol['stop_rule'],'actual_REF_logit_pairs_verified':72,'referenceR_logit_pairs':72,
 'cases':12,'patients':5,'models':6,'referenceR_improved_score_pairs':sum(x['reference_minus_actual']>0 for x in comparisons),
 'no_Q_or_background_removed':True,'diagnostic_only_not_deployable_oracle':True,'no_additional_shifts_or_tuning':True}
(O/'DECISION.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
audit={'status':'PASS','actual12_INPUT_bytes_exact':True,'actual72_REF_output_pairs_exact':True,'actual90_review_rows_exact':True,'all_hypothetical_lists_exactly3_per_patient':len(lists)==270,
 'background_grid_scores_unchanged_all_models_variants':True,'frozen_thresholds_cases_protocol_verified':True,
 'original_grid_file_hashes_unchanged':True,'referenceV_unique_grid_count':len(set((c['patient'],c['segment'],c['grid_index']) for c in cases)),
 'scope':'computational diagnostic; not clinical relabel or deployment performance'}
assert sha(O/'fixed_12V_cases.csv')==protocol['input_cases_sha256'];assert sha(O/'FROZEN_THIRD_BACKGROUND.csv')==protocol['background_thresholds_sha256']
(O/'RESULT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps({'decision':decision,'models':summary},indent=2),flush=True)
