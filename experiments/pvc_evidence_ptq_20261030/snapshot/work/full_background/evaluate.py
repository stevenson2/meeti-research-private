import os
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
import collections,csv,hashlib,json,math
from pathlib import Path
import numpy as np
from scipy.stats import beta
W=Path(__file__).resolve().parent;ROOT=W.parents[1];O=ROOT/'outputs/PVC_FULL_BACKGROUND_20261029'
CACHE=W/'candidate_cache';SCORES=W/'scores';GRID=W/'grid_scores';GRID.mkdir(exist_ok=True)
PRE=ROOT/'outputs/PVC_EVIDENCE_REVIEW_PLAN_20261029';PTQ=ROOT/'outputs/PTQ_REPAIR_20261029'
def read(p):
 with p.open(newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (O/name).open('w',newline='',encoding='utf-8-sig') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
def cp(k,n):return (0 if k==0 else float(beta.ppf(.025,k,n-k+1)),1 if k==n else float(beta.ppf(.975,k+1,n-k)))
progress=json.loads((O/'inference_progress.json').read_text());assert progress['completed_segments']==160
segments=sorted(json.loads((O/'preprocess_progress.json').read_text())['segments'],key=lambda x:(x['patient'],x['segment']))
config=json.loads((PTQ/'deployment_config.json').read_text());keys=list(config)
inventory={r['patient']:r for r in read(PRE/'patient_inventory.csv') if r['split']=='dev'}
people=sorted(inventory);positive=[p for p in people if int(inventory[p]['available_V'])>0]
low=[p for p in positive if float(inventory[p]['available_NSV_V_fraction'])<.01];rare=[p for p in positive if int(inventory[p]['available_V'])<=5]
code_names={0:'N',1:'S',2:'V',3:'Q',4:'unmatched'}
review=[];patients=[];random=[];qrs=[];rare_rows=[];allpoolstats=[]
def category(v):
 if v[2]>0:return 'V_supported'
 if v[3]>0:return 'Q_undetermined'
 if v[1]>0:return 'S_present_noV'
 if v[0]>0:return 'N_only_noV'
 return 'no_reference_beat'
for patient in people:
 ss=[(sid,s) for sid,s in enumerate(segments) if s['patient']==patient];stores={};base=[];scoreslist=[];winslist=[]
 ref_count=np.zeros(4,np.int64);covered_by_class=np.zeros(4,np.int64);trigger_by_class=np.zeros(5,np.int64)
 eligible_refV=set();total_candidates=0;total_grids=0;hours=0.;Vtail=0;extra_fired=0;duplicates=0;Vno_grid=0;Vno_grid_count=0
 for sid,s in ss:
  name=s['segment'];meta=np.load(CACHE/f'{name}_meta.npz');score=np.load(SCORES/f'{name}.npz')
  assert score['keys'].tolist()==keys and np.array_equal(score['sample_index'],meta['sample_index'])
  stores[sid]=(s,meta,score);m=score['q'][:,:,1].astype(np.int16)-score['q'][:,:,0].astype(np.int16)
  grid=meta['grid'];eligible=meta['eligible'];counts=meta['grid_reference_counts'];total_grids+=len(counts)
  present=np.zeros(len(counts),bool);present[np.unique(grid[eligible])]=True
  Vno_grid+=int(counts[~present,2].sum());Vno_grid_count+=int(np.sum(counts[~present,2]>0))
  total_candidates+=len(grid);hours+=s['seconds']/3600;Vtail+=s['reference_V_in_partial_tail'];extra_fired+=s['raw_qrs_fired']-s['candidates'];duplicates+=s['duplicate_reference_matches']
  labels=meta['matched_label'];trigger_by_class+=np.bincount(labels,minlength=5)
  reflabels=meta['reference_label'];refids=meta['reference_annotation_index'];ref_count+=np.bincount(reflabels,minlength=4)
  matched=np.unique(meta['matched_annotation_index'][meta['matched_annotation_index']>=0]);matched_mask=np.isin(refids,matched)
  covered_by_class+=np.bincount(reflabels[matched_mask],minlength=4)
  # Generated R coordinates are ordered, so grouping is label independent and exact.
  cuts=np.r_[0,np.flatnonzero(grid[1:]!=grid[:-1])+1,len(grid)]
  for start,end in zip(cuts[:-1],cuts[1:]):
   g=int(grid[start])
   if not eligible[start]:continue
   a=m[:,start:end];local=np.argmax(a,axis=1);win=local+start;best=a[np.arange(6),local]
   base.append((sid,g,*map(int,counts[g]),int(end-start)));scoreslist.append(best);winslist.append(win)
   if counts[g,2]>0:eligible_refV.add((sid,g))
 base=np.asarray(base,np.int32);sm=np.asarray(scoreslist,np.int16).T;winner=np.asarray(winslist,np.int32).T
 n=len(base);g=sum(row[4]>0 for row in base);assert n>=3
 prob=1-math.comb(n-g,3)/math.comb(n,3);classes=collections.Counter(category(row[2:6]) for row in base)
 randomrow={'patient':patient,'pool_grids':n,'V_supported_grids':g,'random_D3_probability':prob,'observed_hours':hours,
  'random_V_supported_items':3*g/n,**{f'random_{c}_items':3*classes[c]/n for c in ('N_only_noV','S_present_noV','Q_undetermined','no_reference_beat')}}
 random.append(randomrow)
 qrs.append({'patient':patient,'observed_hours':hours,'raw_QRS_fired':total_candidates+extra_fired,'generated_inputs':total_candidates,
  'fired_without_generated_input':extra_fired,'complete_4s_grids':total_grids,'QRS_eligible_grids':n,'reference_V_supported_eligible_grids':g,
  **{f'reference_{c}':int(ref_count[i]) for i,c in enumerate('NSVQ')},
  **{f'unique_reference_{c}_matched':int(covered_by_class[i]) for i,c in enumerate('NSVQ')},
  **{f'trigger_matched_{code_names[i]}':int(trigger_by_class[i]) for i in range(5)},
  'duplicate_reference_matches':duplicates,'reference_V_partial_tail':Vtail,
  'reference_V_in_no_trigger_complete_grid':Vno_grid,'V_supported_complete_grids_without_trigger':Vno_grid_count,'random_D3_probability':prob,
  'QRS_V_reference_coverage':float(covered_by_class[2]/ref_count[2]) if ref_count[2] else ''})
 ranks=np.empty_like(sm,np.int32);orders=[]
 for ki,key in enumerate(keys):
  order=np.lexsort((base[:,1],base[:,0],-sm[ki]));orders.append(order);ranks[ki,order]=np.arange(1,n+1)
  chosen=[]
  for rank,j in enumerate(order[:3],1):
   row=base[j];sid,gid=map(int,row[:2]);s,meta,score=stores[sid];w=int(winner[ki,j]);counts=row[2:6]
   rec={'key':key,'seed':int(key[4:6]),'arm':key[-1],'patient':patient,'rank':rank,'segment':s['segment'],'segment_index':sid,'grid_index':gid,
    'central_start_seconds':4*gid,'central_end_seconds':4*(gid+1),'integer_margin':int(sm[ki,j]),
    **{f'n_{c}':int(counts[i]) for i,c in enumerate('NSVQ')},'category':category(counts),
    'trigger_candidate_index':w,'trigger_sample_index':int(meta['sample_index'][w]),'trigger_anchor_sample':int(meta['anchor'][w]),
    'trigger_label':code_names[int(meta['matched_label'][w])],'trigger_matched_annotation_index':int(meta['matched_annotation_index'][w]),
    'trigger_match_distance_ms':float(meta['match_distance'][w]*4),'qN':int(score['q'][ki,w,0]),'qV':int(score['q'][ki,w,1]),
    'passes_existing_cutoff':int(sm[ki,j])>=config[key]['calibration_2pct_cutoff'],'existing_cutoff':config[key]['calibration_2pct_cutoff'],
    'source_base':str(Path(os.environ['ECG_DEV_SOURCE'])/patient/s['segment'])}
   review.append(rec);chosen.append(rec)
  for mode,returned in [('primary_top3',chosen),('existing_cutoff_gated',[x for x in chosen if x['passes_existing_cutoff']])]:
   c=collections.Counter(x['category'] for x in returned);supported=c['V_supported'];adj=sum(a['segment']==b['segment'] and abs(a['grid_index']-b['grid_index'])==1 for i,a in enumerate(returned) for b in returned[i+1:])
   patients.append({'key':key,'seed':int(key[4:6]),'arm':key[-1],'mode':mode,'patient':patient,'reference_V_positive':patient in positive,
    'available_V':int(inventory[patient]['available_V']),'observed_lt1pct':patient in low,'rare_1to5':patient in rare,
    'recommended_items':len(returned),'V_supported_items':supported,'patient_evidence_hit':supported>0,
    **{f'{cat}_items':c[cat] for cat in ('N_only_noV','S_present_noV','Q_undetermined','no_reference_beat')},
    'unsupported_items':len(returned)-supported,'V_supported_trigger_V_items':sum(x['category']=='V_supported' and x['trigger_label']=='V' for x in returned),
    'V_supported_other_trigger_items':sum(x['category']=='V_supported' and x['trigger_label']!='V' for x in returned),
    **{f'trigger_{code_names[i]}_items':sum(x['trigger_label']==code_names[i] for x in returned) for i in range(5)},
    'adjacent_pairs':adj,'best_V_window_rank':int(min(ranks[ki,j] for j,row in enumerate(base) if row[4]>0)) if g else '',
    'pool_grids':n,'V_supported_grids':g,'random_D3_probability':prob,'observed_hours':hours})
 np.savez(GRID/f'{patient}.npz',base=base,margin=sm,winner=winner,ranks=ranks,keys=np.array(keys))
 if patient in rare:
  index_lookup={(int(row[0]),int(row[1])):j for j,row in enumerate(base)}
  for sid,(s,meta,score) in stores.items():
   for vi in np.flatnonzero(meta['reference_label']==2):
    ai=int(meta['reference_annotation_index'][vi]);sample=int(meta['reference_sample'][vi]);gid=sample//1000
    where=index_lookup.get((sid,gid));matched=np.flatnonzero(meta['matched_annotation_index']==ai)
    nearest=int(np.argmin(abs(meta['anchor']-sample)))
    for ki,key in enumerate(keys):
     margin=score['q'][ki,:,1].astype(int)-score['q'][ki,:,0].astype(int)
     rare_rows.append({'key':key,'patient':patient,'segment':s['segment'],'annotation_index':ai,'sample_orig':sample,'grid_index':gid,
      'has_real_QRS_match_within150ms':len(matched)>0,'matched_QRS_count':len(matched),'nearest_trigger_distance_ms':int(abs(int(meta['anchor'][nearest])-sample)*4),
      'best_matched_trigger_margin':int(margin[matched].max()) if len(matched) else '',
      'reference_grid_has_any_trigger':where is not None,'reference_grid_margin':int(sm[ki,where]) if where is not None else '',
      'reference_grid_rank':int(ranks[ki,where]) if where is not None else '',
      'reference_V_in_top3':where is not None and ranks[ki,where]<=3})
 for _,meta,score in stores.values():meta.close();score.close()
write('review_lists.csv',review);write('patient_results.csv',patients);write('random_baseline_per_patient.csv',random);write('QRS_patient_audit.csv',qrs);write('rare_V_results.csv',rare_rows)
groups={'all_V_positive':positive,'observed_lt1pct_V':low,'observed_1to5_V':rare};metrics=[];burden=[];paired=[]
for key in keys:
 for mode in ('primary_top3','existing_cutoff_gated'):
  rr=[x for x in patients if x['key']==key and x['mode']==mode];n=sum(x['recommended_items'] for x in rr);v=sum(x['V_supported_items'] for x in rr)
  fields=['unsupported_items','N_only_noV_items','S_present_noV_items','Q_undetermined_items','no_reference_beat_items','V_supported_trigger_V_items','V_supported_other_trigger_items','adjacent_pairs']+[f'trigger_{code_names[i]}_items' for i in range(5)]
  burden.append({'key':key,'mode':mode,'recommended_items':n,'items_per_patient':n/40,'V_supported_items':v,'V_support_yield':v/n if n else 0,
   **{f:sum(x[f] for x in rr) for f in fields},'no_recommendation_patients':sum(x['recommended_items']==0 for x in rr),
   'reference_noV_patients_with_recommendation':sum(x['recommended_items']>0 for x in rr if not x['reference_V_positive'])})
  for name,ps in groups.items():
   k=sum(x['patient_evidence_hit'] for x in rr if x['patient'] in ps);lo,hi=cp(k,len(ps));rb=[x['random_D3_probability'] for x in random if x['patient'] in ps]
   metrics.append({'key':key,'mode':mode,'group':name,'hits':k,'patients':len(ps),'rate':k/len(ps),'CP95_low':lo,'CP95_high':hi,
    'random_expected_hits':sum(rb),'random_expected_rate':float(np.mean(rb))})
for mode in ('primary_top3','existing_cutoff_gated'):
 for name,ps in groups.items():
  aa=[];bb=[]
  for p in ps:
   aa.append(np.mean([x['patient_evidence_hit'] for x in patients if x['patient']==p and x['mode']==mode and x['arm']=='A']))
   bb.append(np.mean([x['patient_evidence_hit'] for x in patients if x['patient']==p and x['mode']==mode and x['arm']=='B']))
  a=np.array(aa);b=np.array(bb);d=b-a;rng=np.random.default_rng(20261029);draw=rng.integers(0,len(ps),(1000,len(ps)));ds=d[draw].mean(axis=1)
  paired.append({'mode':mode,'group':name,'patients':len(ps),'A_mean_rate':float(a.mean()),'B_mean_rate':float(b.mean()),'BA_pp':float(d.mean()*100),
   'CI_low_pp':float(np.quantile(ds,.025)*100),'CI_high_pp':float(np.quantile(ds,.975)*100)})
write('discovery_metrics.csv',metrics);write('review_burden.csv',burden);write('paired_patient_summary.csv',paired)
random_summary={name:{'patients':len(ps),'expected_hits':sum(x['random_D3_probability'] for x in random if x['patient'] in ps),
 'expected_rate':float(np.mean([x['random_D3_probability'] for x in random if x['patient'] in ps]))} for name,ps in groups.items()}
random_summary['burden']={c:sum(x[c] for x in random) for c in random[0] if c.startswith('random_') and c.endswith('_items')}
(O/'random_baseline_exact.json').write_text(json.dumps(random_summary,indent=2),encoding='utf-8')
print(json.dumps({'metrics':metrics,'burden':burden,'paired':paired,'random':random_summary,'QRS_V_covered':sum(x['unique_reference_V_matched'] for x in qrs)},indent=2))
