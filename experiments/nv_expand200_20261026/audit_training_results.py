"""Post-terminal independent counts and full-state restore audit; no fitting."""
from common import *
import argparse
def main():
    status=readj(ROOT/'status.json');assert len(status['runs'])==12 and all(r['status']=='completed' for r in status['runs'])
    auditpath=ROOT/'training_recovery_audit.json'
    if auditpath.exists():
        old=readj(auditpath);assert old['training_config_sha256']==sha(ROOT/'frozen_config.json')
        assert all(sha(ROOT/'runs'/r['id']/'best.weights.h5')==r['checkpoint_sha256'] for r in old['runs']);print('TRAINING AUDIT ALREADY PASSED; unchanged');return
    from run_matrix import other_training
    def others():return [p for p in other_training() if p['pid']!=os.getpid()]
    if others():
        print('RECOVERY AUDIT WAITING FOR LOCAL TRAINING TO FINISH; no GPU initialization yet',flush=True)
        while others():time.sleep(30)
    bootstrap()
    import tensorflow as tf
    from models import build,FixedLowpass
    runtime=json.loads(json.dumps(configure_tf()));cfg=readj(ROOT/'frozen_config.json');assert runtime==cfg['runtime']
    for name,h in cfg['script_hashes'].items():assert sha(ROOT/name)==h
    assert sha(ROOT/'development_qc.json')==cfg['development_qc_sha256'] and sha(ROOT/'prep_manifest.json')==cfg['prep_manifest_sha256']
    qc=readj(ROOT/'development_qc.json')
    for item in qc['cache_files'].values():assert sha(Path(item['path']))==item['sha256']
    meta=metadata('dev');names=patient_names('dev');n=len(meta['y']);y=meta['y'];avail=meta['avail7'].astype(bool);audits=[];traces={}
    init=readj(ROOT/'initializations.json');assert sha(ROOT/'initializations.json')==cfg['initializations_sha256']
    for r in status['runs']:
        out=ROOT/'runs'/r['id'];summary=readj(out/'training_summary.json');complete=readj(out/'complete.json');ck=sha(out/'best.weights.h5')
        assert ck==summary['best_checkpoint_sha256']==complete['checkpoint_sha256'] and sha(out/'metrics.json')==complete['metrics_sha256']
        assert summary['config_sha256']==sha(ROOT/'frozen_config.json') and summary['init_sha256']==init[str(r['seed'])]['models'][r['model']]['file_sha256']
        ev=out/readj(out/'evaluation.json')['directory'];manifest=readj(ev/'manifest.json');m=readj(out/'metrics.json')
        assert len(manifest['predictions'])==15 and manifest['bn_unchanged'] and m['repeat']['exact']
        for name,info in manifest['predictions'].items():assert sha(ev/(name+'.npz'))==info['sha256']
        predictions=[]
        for d in SHIFTS:
            with np.load(ev/f'dev_{d:+d}.npz') as f:
                assert np.array_equal(f['rows'],np.arange(n))
                assert all(np.array_equal(f[k],meta[k]) for k in ['y','patient','segment','annotation_index','avail7'])
                predictions.append(f['p'])
        p0=predictions[3];decisions=np.stack([p>.5 for p in predictions],1)
        fp=int(((y==0)&(p0>.5)).sum());tp=int(((y==1)&(p0>.5)).sum());nw=int(decisions[avail&(y==0)].any(1).sum());vd=int(decisions[avail&(y==1)].all(1).sum())
        assert (fp,tp,nw,vd)==(m['zero']['N_false_positive'],m['zero']['V_detected'],m['robust']['N_any_wrong'],m['robust']['V_all_detected'])
        for pi,pid in enumerate(names):
            mask=meta['patient']==pi;pp=next(q for q in m['per_patient'] if q['patient']==pid)
            assert int((mask&(y==0)&(p0>.5)).sum())==pp['zero']['N_false_positive']
            assert int((mask&(y==1)&(p0>.5)).sum())==pp['zero']['V_detected']
            assert int(decisions[mask&avail&(y==0)].any(1).sum())==pp['N_any_wrong']
            assert int(decisions[mask&avail&(y==1)].all(1).sum())==pp['V_all_detected']
        snap=out/readj(out/'last.json')['snapshot'];state=readj(snap/'state.json');model=tf.keras.models.load_model(snap/'model.keras')
        assert int(model.optimizer.iterations.numpy())==state['epoch']*STEPS
        assert ah(np.concatenate([v.numpy().ravel() for v in model.optimizer.variables]))==state['optimizer_sha256']
        for k in ['epoch','best','best_epoch','wait','finished_training','config_sha256']:assert state[k]==summary[k]
        assert state['finished_training'] and (state['wait']>=12 or state['epoch']==80)
        model.get_layer('do').seed_generator.state.assign(state['dropout_state']);assert model.get_layer('do').seed_generator.state.numpy().tolist()==state['dropout_state']
        for key in ['order','nonzero','gate']:
            rng=np.random.default_rng();rng.bit_generator.state=state['rng'][key];assert rng.bit_generator.state==state['rng'][key]
        lp=[l for l in model.layers if isinstance(l,FixedLowpass)]
        assert len(lp)==(3 if r['model'] in ['S_LP','L_LP'] else 0) and all(not l.bypass for l in lp)
        best=build(r['model'],r['seed']);best.load_weights(out/'best.weights.h5');best.save(out/'best_model.keras');loaded=tf.keras.models.load_model(out/'best_model.keras')
        X,_=data('dev');x=batch_input(X,np.arange(8),3);assert np.array_equal(best(x,training=False).numpy(),loaded(x,training=False).numpy());del X
        traces[(r['model'],r['seed'])]=summary['durations']
        audits.append(dict(id=r['id'],checkpoint_sha256=ck,counts_independently_recomputed=True,all_prediction_hashes_match=True,row_identities_match=True,zero_repeat_exact=True,bn_unchanged=True,snapshot=snap.name,epoch=state['epoch'],optimizer_iterations=state['epoch']*STEPS,optimizer_hash_match=True,sampling_rng_restored=True,dropout_restored=True,early_stop_state_restored=True,best_model_sha256=sha(out/'best_model.keras'),best_model_role='inference-only; full continuation uses epoch model.keras + state.json'))
        del model,best,loaded,predictions,decisions;tf.keras.backend.clear_session();print('TRAINING DELIVERY AUDIT',r['id'],'PASSED',flush=True)
    for seed in [42,43,44]:
        for epoch in range(min(len(traces[(kind,seed)]) for kind in ['S','L','S_LP','L_LP'])):
            rows=[traces[(kind,seed)][epoch] for kind in ['S','L','S_LP','L_LP']]
            assert all(row['epoch']==epoch+1 for row in rows)
            for key in ['order_sha256','candidate_sha256','gate_sha256','delta_sha256']:assert len({row[key] for row in rows})==1
    writej(auditpath,dict(passed=True,finished=now(),training_config_sha256=sha(ROOT/'frozen_config.json'),runtime=runtime,cache_hashes_unchanged=True,all_shared_seed_epoch_random_streams_match=True,independent_not_opened_when_audited=not (ROOT/'independent_opened.json').exists(),no_extra_training_fit=True,runs=audits))
    print('ALL12 TRAINING RESULTS AND FULL RECOVERY AUDIT PASSED',flush=True)
if __name__=='__main__':main()
