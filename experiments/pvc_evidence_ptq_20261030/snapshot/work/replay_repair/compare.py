import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import csv,json,hashlib,collections
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];O=R/'outputs/PC_REPLAY_REPAIR_20261030';P=R/'outputs/PVC_FULL_BACKGROUND_20261029';F=R/'work/full_background'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
people=json.loads((O/'PREREG.json').read_text(encoding='utf-8'))['patients'];new=read(O/'review_lists.csv');old=[x for x in read(P/'review_lists.csv') if x['patient'] in people]
dtype=np.dtype([('sample_index','<u4'),('input','i1',(250,))]);before={x['segment']:x for x in json.loads((P/'preprocess_progress.json').read_text())['segments']};after=json.loads((O/'preprocess_progress.json').read_text())['segments'];segments=[]
initial={x['segment']:x for x in json.loads((W/'initial_cold_init_bug/preprocess_progress.json').read_text())['segments']};reuse=[]
for s in after:
 name=s['segment'];b=before[name];a=np.fromfile(F/'candidate_cache'/f'{name}.bin',dtype=dtype);z=np.fromfile(W/'candidate_cache'/f'{name}.bin',dtype=dtype)
 shared,oi,ni=np.intersect1d(a['sample_index'],z['sample_index'],return_indices=True);exact=np.all(a['input'][oi]==z['input'][ni],axis=1)
 segments.append({'patient':s['patient'],'segment':name,'old_inputs':b['candidates'],'new_inputs':s['candidates'],'old_raw_QRS':b['raw_qrs_fired'],'new_raw_QRS':s['raw_qrs_fired'],
  'old_first_input_R_seconds':(int(a['sample_index'][0])-125)/250,'new_first_input_R_seconds':s['first_input_R_seconds'],
  'old_V_matched':b['reference_V_unique_QRS_matched'],'new_V_matched':s['reference_V_unique_QRS_matched'],'referenceV':s['reference_V'],
  'old_eligible_grids':b['eligible_4s_grids'],'new_eligible_grids':s['eligible_4s_grids'],'shared_window_end_samples':len(shared),'shared_anchor_INPUT_bytes_exact':int(exact.sum()),
  'first_R_recovery_updates':s['cold_recovery_updates']})
 score_rec=json.loads((W/'scores'/f'{name}.json').read_text());same=initial[name]['cache_sha256']==s['cache_sha256']
 assert score_rec['cache_sha256']==s['cache_sha256'] and sha(W/'scores'/f'{name}.npz')==score_rec['score_sha256']
 reuse.append({'segment':name,'exact_same_input_hash_after_cold_init_fix':same,'scored_or_reused':'reused_identical_input_and_models' if same else 'rescored_after_input_change','candidate_count':s['candidates'],'final_input_sha256':s['cache_sha256'],'final_score_sha256':score_rec['score_sha256']})
write('trigger_segment_comparison.csv',segments);write('checkpoint_reuse_audit.csv',reuse)
oldq={x['patient']:x for x in read(P/'QRS_patient_audit.csv')};newq={x['patient']:x for x in read(O/'QRS_patient_audit.csv')};patients=[]
for p in people:
 a=oldq[p];b=newq[p];ss=[x for x in segments if x['patient']==p]
 patients.append({'patient':p,'referenceV':b['reference_V'],'old_V_matched':a['unique_reference_V_matched'],'new_V_matched':b['unique_reference_V_matched'],
  'old_V_in_eligible_grids':int(a['reference_V'])-int(a['reference_V_in_no_trigger_complete_grid'])-int(a['reference_V_partial_tail']),
  'new_V_in_eligible_grids':int(b['reference_V'])-int(b['reference_V_in_no_trigger_complete_grid'])-int(b['reference_V_partial_tail']),
  'old_inputs':sum(x['old_inputs'] for x in ss),'new_inputs':sum(x['new_inputs'] for x in ss),'old_eligible_grids':a['QRS_eligible_grids'],'new_eligible_grids':b['QRS_eligible_grids'],
  'old_N_matched':a['unique_reference_N_matched'],'new_N_matched':b['unique_reference_N_matched'],'referenceN':b['reference_N']})
write('trigger_patient_comparison.csv',patients)
olde={(x['key'],x['patient'],x['segment'],x['annotation_index']):x for x in read(P/'rare_V_results.csv') if x['patient'] in people};events=[]
for b in read(O/'rare_V_results.csv'):
 a=olde[(b['key'],b['patient'],b['segment'],b['annotation_index'])]
 events.append({'key':b['key'],'patient':b['patient'],'segment':b['segment'],'annotation_index':b['annotation_index'],'reference_sample250':b['sample_orig'],
  'old_QRS_matched':a['has_real_QRS_match_within150ms'],'new_QRS_matched':b['has_real_QRS_match_within150ms'],
  'old_nearest_R_error_ms':a['nearest_trigger_distance_ms'],'new_nearest_R_error_ms':b['nearest_trigger_distance_ms'],
  'old_grid_available':a['reference_grid_has_any_trigger'],'new_grid_available':b['reference_grid_has_any_trigger'],
  'old_matched_V_margin':a['best_matched_trigger_margin'],'new_matched_V_margin':b['best_matched_trigger_margin'],
  'old_V_grid_rank':a['reference_grid_rank'],'new_V_grid_rank':b['reference_grid_rank'],'new_V_grid_margin':b['reference_grid_margin'],
  'old_in_top3':a['reference_V_in_top3'],'new_in_top3':b['reference_V_in_top3']})
write('V_before_after.csv',events)
beforepatient={(x['key'],x['patient']):x for x in read(P/'patient_results.csv') if x['patient'] in people and x['mode']=='primary_top3'};outcomes=[]
for key in sorted(set(x['key'] for x in new)):
 rr=[x for x in new if x['key']==key];aa=[x for x in old if x['key']==key];counter=collections.Counter(x['category'] for x in rr)
 oldset={(x['patient'],x['segment'],x['grid_index']) for x in aa};newset={(x['patient'],x['segment'],x['grid_index']) for x in rr}
 outcomes.append({'key':key,'old_D3_hits':sum(beforepatient[(key,p)]['patient_evidence_hit']=='True' for p in people),
  'new_D3_hits':sum(any(x['patient']==p and int(x['n_V'])>0 for x in rr) for p in people),'patients':5,
  'old_referenceV_captured':sum(int(x['n_V']) for x in aa),'new_referenceV_captured':sum(int(x['n_V']) for x in rr),'referenceV':12,
  'review_items':len(rr),'V_supported_items':counter['V_supported'],'Q_noV_items':counter['Q_undetermined'],'N_only_items':counter['N_only_noV'],
  'S_noV_items':counter['S_present_noV'],'no_reference_beat_items':counter['no_reference_beat'],'same_selected_windows_as_before':len(oldset&newset),
  'same_rank_slots_as_before':sum(any(a['patient']==x['patient'] and a['rank']==x['rank'] and a['segment']==x['segment'] and a['grid_index']==x['grid_index'] for a in aa) for x in rr)})
write('outcome_comparison.csv',outcomes)
summary={'status':'COMPLETE','patients':5,'segments':20,'referenceV':12,'old_V_matched':sum(int(x['old_V_matched']) for x in patients),'new_V_matched':sum(int(x['new_V_matched']) for x in patients),
 'old_V_in_candidate_grids':sum(x['old_V_in_eligible_grids'] for x in patients),'new_V_in_candidate_grids':sum(x['new_V_in_eligible_grids'] for x in patients),
 'old_inputs':sum(x['old_inputs'] for x in patients),'new_inputs':sum(x['new_inputs'] for x in patients),
 'old_pool_grids':sum(int(x['old_eligible_grids']) for x in patients),'new_pool_grids':sum(int(x['new_eligible_grids']) for x in patients),
 'final_model_input_outputs':6*sum(x['new_inputs'] for x in patients),'same_selected_windows_of90':sum(x['same_selected_windows_as_before'] for x in outcomes),
 'same_rank_slots_of90':sum(x['same_rank_slots_as_before'] for x in outcomes),'first_R_recovery_updates_real20':sum(x['first_R_recovery_updates'] for x in segments),
 'reused_byte_identical_checkpoints':sum(x['exact_same_input_hash_after_cold_init_fix'] for x in reuse),'rescored_after_cold_init_fix':sum(not x['exact_same_input_hash_after_cold_init_fix'] for x in reuse),
 'D3_unchanged_all6':all(x['old_D3_hits']==x['new_D3_hits'] for x in outcomes),'new_top3_referenceV_capture_unchanged_all6':all(x['old_referenceV_captured']==x['new_referenceV_captured'] for x in outcomes)}
(O/'COMPARISON_SUMMARY.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))
