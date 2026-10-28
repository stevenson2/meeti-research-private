from common import *
bootstrap()
import tensorflow as tf
from models import build,copy_learned,FixedLowpass
runtime=configure_tf();old=ROOT.parent/'NV_LONGRUN_20261024'
assert not (ROOT/'frozen_config.json').exists(),'Never overwrite a frozen training protocol'
assert readj(ROOT/'development_qc.json')['independent_guard_rejection_test']
# Access guard must reject before opening an independent cache.
try:data('independent')
except AssertionError as e:assert 'INDEPENDENT_CONTENT_DENIED' in str(e)
else:raise AssertionError('Independent guard failed')
initdir=ROOT/'initializations';initdir.mkdir(exist_ok=True);initial={}
X,_=data('dev');x=batch_input(X,np.arange(16),3)
prior=readj(old/'initializations.json')
for seed in [42,43,44]:
    item={'models':{},'paired_matches':{}}
    for plain,filtered in [('S','S_LP'),('L','L_LP')]:
        src=old/'initializations'/f'{plain}_s{seed}.weights.h5'
        assert sha(src)==prior[str(seed)]['models'][plain]['file_sha256']
        base=build(plain,seed);base.load_weights(src);lp=build(filtered,seed);matched=copy_learned(base,lp)
        layers=[l for l in lp.layers if isinstance(l,FixedLowpass)]
        assert [l.name for l in layers]==['lp_stem','lp_b2','lp_b5']
        for l in layers:l.bypass=True
        assert np.array_equal(base(x,training=False).numpy(),lp(x,training=False).numpy())
        for l in layers:l.bypass=False
        for kind,m in [(plain,base),(filtered,lp)]:
            path=initdir/f'{kind}_s{seed}.weights.h5';m.save_weights(path)
            assert m.count_params()==(62834 if plain=='S' else 828514)
            item['models'][kind]=dict(parameters=m.count_params(),file_sha256=sha(path),source_fresh_initialization_sha256=sha(src),dropout_state=m.get_layer('do').seed_generator.state.numpy().tolist(),layer_shapes={l.name:str(l.output.shape) for l in m.layers})
        tmp=initdir/f'{filtered}_s{seed}.keras';lp.save(tmp);loaded=tf.keras.models.load_model(tmp)
        assert all(not l.bypass for l in loaded.layers if isinstance(l,FixedLowpass))
        assert np.array_equal(loaded(x,training=False).numpy(),lp(x,training=False).numpy())
        item['paired_matches'][filtered]=dict(source=plain,matched_layers=matched,bypass_exact=True,enabled_after_reload=True)
        tf.keras.backend.clear_session()
    initial[str(seed)]=item
writej(ROOT/'initializations.json',initial)
# Statistics guard for empty/single-class patient support.
from evaluate import auc_ap,class_metrics
assert class_metrics(np.array([],np.int8),np.array([],np.float32))['V_detection_rate'] is None
assert auc_ap(np.array([0,0]),np.array([.1,.2]))==(None,None)
assert compare(np.array([]),np.array([]),np.array([]),np.array([]))['flips']==0
assert STEPS==(FROZEN_DATA['counts']['train']['windows']+255)//256
assert CW.tolist()==FROZEN_DATA['class_weights']
scope=readj(ROOT/'scope.json');files=['common.py','models.py','source_model.py','train_one.py','evaluate.py']
cfg=dict(scope=scope,matrix_sha256=sha(ROOT/'matrix.json'),patient_split_sha256=sha(ROOT/'patient_split_frozen.json'),segment_split_sha256=sha(ROOT/'segments_frozen.json'),development_qc_sha256=sha(ROOT/'development_qc.json'),prep_manifest_sha256=sha(ROOT/'prep_manifest.json'),initializations_sha256=sha(ROOT/'initializations.json'),script_hashes={n:sha(ROOT/n) for n in files},runtime=runtime,training=dict(batch=BATCH,eval_batch=EVAL_BATCH,max_epochs=EPOCHS,steps_per_epoch=STEPS,decay_steps=STEPS*EPOCHS,class_weights=CW.tolist(),lr=3e-4,cosine_alpha=.01,patience=12,min_delta=1e-4,threshold='pV>0.50'),differences_from_previous=['More independent development patients; actual total samples/steps and class weights frozen after QC','Zero-only train cache uses column0 instead of old9-column cache column4; all C inputs equivalent','Old and new dev jointly select checkpoint with identical patient/class equal CE; separately reported','Shared periodic progress file only; optimizer/math/random-source semantics unchanged','Empty/single-class patient metrics represented as null; legacy data and old model results unchanged'])
writej(ROOT/'frozen_config.json',cfg)
writej(ROOT/'verification.json',dict(passed=True,runtime=runtime,independent_guard_rejected=True,initialization_pairs_passed=True,no_training_fit_performed=True,finished=now()))
print('TRAINING FROZEN: 12 RUNS',STEPS,'steps/epoch',CW.tolist(),flush=True)
