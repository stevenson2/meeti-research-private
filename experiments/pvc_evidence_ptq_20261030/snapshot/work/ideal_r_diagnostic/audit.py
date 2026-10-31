import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
from pathlib import Path
import csv,json,hashlib
import numpy as np
W=Path(__file__).resolve().parent;R=W.parents[1];B=R/'work/replay_repair';P=R/'outputs/PC_REPLAY_REPAIR_20261030';O=R/'outputs/IDEAL_R_FINAL_DIAGNOSTIC_20261030'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
protocol=json.loads((O/'PREREG.json').read_text(encoding='utf-8'));cases=json.loads((W/'cases.json').read_text());comparisons=read(O/'per_V_model_comparison.csv');lists=read(O/'hypothetical_review_lists.csv');out=np.load(O/'diagnostic_outputs.npz');keys=out['keys'].tolist();m=out['q'][:,:,:,1].astype(int)-out['q'][:,:,:,0].astype(int)
ss=sorted(json.loads((P/'preprocess_progress.json').read_text())['segments'],key=lambda x:(x['patient'],x['segment']));checked_lists=0;checked_thresholds=0
for patient in protocol['patients']:
 z=np.load(B/'grid_scores'/f'{patient}.npz');base=z['base'];original=z['margin'];bg=base[:,4]==0;lookup={(int(x[0]),int(x[1])):j for j,x in enumerate(base)}
 assert sha(B/'grid_scores'/f'{patient}.npz')==protocol['baseline_grid_sha256'][patient]
 for ki,key in enumerate(keys):
  ref=original[ki].astype(int).copy();credit=original[ki].astype(int).copy();relevant=[(j,c) for j,c in enumerate(cases) if c['patient']==patient]
  third=int(np.partition(original[ki,bg],-3)[-3]);assert all(int(x['third_background_margin'])==third for x in comparisons if x['patient']==patient and x['key']==key);checked_thresholds+=1
  for ci,c in relevant:
   row=next(x for x in comparisons if x['key']==key and x['case_id']==c['case_id']);assert int(row['actual_margin'])==m[ki,0,ci] and int(row['referenceR_margin'])==m[ki,1,ci]
   assert (row['reference_strictly_above_third_background']=='True')==(m[ki,1,ci]>third)
   meta=np.load(B/'candidate_cache'/f'{c["segment"]}_meta.npz');scores=np.load(B/'scores'/f'{c["segment"]}.npz');allm=scores['q'][ki,:,1].astype(int)-scores['q'][ki,:,0].astype(int)
   where=np.flatnonzero((meta['grid']==c['grid_index'])&meta['eligible']);values=allm[where].copy();actualpos=np.flatnonzero(where==c['actual_candidate_index']);assert len(actualpos)==1
   assert int(values[actualpos[0]])==m[ki,0,ci];values[actualpos[0]]=m[ki,1,ci];gi=lookup[(c['segment_index'],c['grid_index'])]
   ref[gi]=int(values.max());credit[gi]=max(int(original[ki,gi]),int(m[ki,1,ci]));meta.close();scores.close()
  for variant,values in [('actual',original[ki]),('referenceR_replace',ref),('optimistic_credit',credit)]:
   assert np.array_equal(values[bg],original[ki,bg]);order=np.lexsort((base[:,1],base[:,0],-values))[:3]
   expected=[(ss[int(base[j,0])]['segment'],int(base[j,1]),int(values[j])) for j in order]
   got=sorted([x for x in lists if x['variant']==variant and x['key']==key and x['patient']==patient],key=lambda x:int(x['rank']))
   assert [(x['segment'],int(x['grid_index']),int(x['margin'])) for x in got]==expected;checked_lists+=1
 z.close()
cfg=json.loads((R/'outputs/PTQ_REPAIR_20261029/deployment_config.json').read_text())
for key,c in cfg.items():assert sha(R/'outputs/PTQ_REPAIR_20261029'/c['file'])==protocol['models'][key]
for name,expect in protocol['baseline_sources_sha256'].items():assert sha(B/'native'/name)==expect
decision=json.loads((O/'DECISION.json').read_text());summary=read(O/'model_diagnostic_summary.csv');strict=sum(x['reference_strictly_above_third_background']=='True' for x in comparisons);weak=sum(int(x['optimistic_credit_D3_hit'])<3 for x in summary)
assert strict==decision['reference_strict_crossings_out_of72'] and decision['stop_rule_met']==((72-strict)>36 and weak>=4)
audit={'status':'PASS','independently_checked_background_third_scores':checked_thresholds,'independently_reconstructed_top3_lists':checked_lists,
 'paired_V_model_margin_rows':len(comparisons),'frozen_models_and_repaired_source_hashes_unchanged':True,'same_fixed_background_all_variants':True,
 'stop_rule_independently_recomputed':True,'no_additional_model_calls_or_inputs':True}
(O/'INDEPENDENT_AUDIT.json').write_text(json.dumps(audit,indent=2),encoding='utf-8');print(json.dumps(audit,indent=2))
