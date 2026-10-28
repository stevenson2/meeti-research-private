from common import *
bootstrap()
import tensorflow as tf
from models import build,FixedLowpass
import argparse,shutil

def main(runid):
    start=time.monotonic(); runtime=configure_tf()
    run=next(r for r in readj(ROOT/'matrix.json')['runs'] if r['id']==runid)
    seed,kind,aug=run['seed'],run['model'],run['augmentation']
    out=ROOT/'runs'/runid; out.mkdir(parents=True,exist_ok=True)
    frozen=readj(ROOT/'frozen_config.json')
    for name,h in frozen['script_hashes'].items(): assert sha(ROOT/name)==h, f'Frozen implementation drift: {name}'
    fingerprint=sha(ROOT/'frozen_config.json')
    X,tr=data('train'); D,dv=data('dev'); all_dev=np.arange(len(dv['y']))
    assert aug=='C' and X.shape[1]==1
    n_old_dev=FROZEN_DATA['counts']['dev']['legacy_windows']
    rng_order=np.random.default_rng(seed); rng_nonzero=np.random.default_rng(10000+seed); rng_gate=np.random.default_rng(20000+seed)
    state=dict(epoch=0,best=None,best_epoch=0,wait=0,finished_training=False,config_sha256=fingerprint,run=run,started=now(),durations=[])
    ptr=out/'last.json'
    if ptr.exists():
        snap=out/readj(ptr)['snapshot']; state=readj(snap/'state.json')
        assert state['config_sha256']==fingerprint and state['run']['id']==runid
        model=tf.keras.models.load_model(snap/'model.keras')
        model.get_layer('do').seed_generator.state.assign(state['dropout_state'])
        for rng,key in [(rng_order,'order'),(rng_nonzero,'nonzero'),(rng_gate,'gate')]: rng.bit_generator.state=state['rng'][key]
        assert int(model.optimizer.iterations.numpy())==state['epoch']*STEPS
        assert ah(np.concatenate([v.numpy().ravel() for v in model.optimizer.variables]))==state['optimizer_sha256']
        print('EXACT EPOCH RESTORE',runid,state['epoch'],flush=True)
    else:
        model=build(kind,seed)
        init=ROOT/'initializations'/f'{kind}_s{seed}.weights.h5'
        assert sha(init)==readj(ROOT/'initializations.json')[str(seed)]['models'][kind]['file_sha256']
        model.load_weights(init)
        model.compile(optimizer=tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.CosineDecay(3e-4,STEPS*EPOCHS,alpha=.01)),loss='sparse_categorical_crossentropy',jit_compile=False)
        state['init_sha256']=sha(init)
    opt=model.optimizer
    @tf.function(input_signature=[tf.TensorSpec([None,250,1],tf.float32),tf.TensorSpec([None],tf.int32)],jit_compile=False)
    def train_batch(x,y):
        with tf.GradientTape() as tape:
            p=model(x,training=True)
            loss=tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(y,p)*tf.gather(tf.constant(CW),y))
        grad=tape.gradient(loss,model.trainable_variables)
        opt.apply_gradients(zip(grad,model.trainable_variables))
        return loss,tf.reduce_sum(tf.cast(tf.equal(tf.argmax(p,axis=1,output_type=tf.int32),y),tf.int32))
    forward=make_forward(model)
    for ep in range(state['epoch'],EPOCHS):
        if state['finished_training']: break
        epoch_start=time.monotonic(); t=time.monotonic()
        order=rng_order.permutation(len(tr['y']))
        candidate=rng_nonzero.choice(np.array([-4,-3,-2,-1,1,2,3,4],np.int8),size=len(order))
        gate=rng_gate.random(len(order))
        delta=np.zeros(len(order),np.int8) if aug=='C' else np.where(gate<(1/9 if aug=='U9' else .5),0,candidate).astype(np.int8)
        rng_seconds=time.monotonic()-t
        row=dict(epoch=ep+1,order_sha256=ah(order),candidate_sha256=ah(candidate),gate_sha256=ah(gate),delta_sha256=ah(delta),offset_hist={str(d):int((delta==d).sum()) for d in range(-4,5)},rng_seconds=rng_seconds)
        read_seconds=0.; train_seconds=0.; first_batch_seconds=None; loss_sum=0.; correct=0
        for bi in range(STEPS):
            ix=order[bi*BATCH:(bi+1)*BATCH]; t=time.monotonic()
            bx=batch_input(X,ix,delta[ix].astype(np.int64)); by=tr['y'][ix].astype(np.int32)
            read_seconds+=time.monotonic()-t; t=time.monotonic()
            loss,acc=train_batch(bx,by); loss=float(loss.numpy()); acc=int(acc.numpy())
            elapsed=time.monotonic()-t; train_seconds+=elapsed
            if bi==0: first_batch_seconds=elapsed
            if not np.isfinite(loss): raise RuntimeError('NONFINITE_TRAIN_LOSS')
            loss_sum+=loss*len(ix); correct+=acc
            if (bi+1)%256==0 or bi+1==STEPS:
                writej(out/'progress.json',dict(phase='training',epoch=ep+1,batch=bi+1,batches=STEPS,seen=min((bi+1)*BATCH,len(order)),windows=len(order),epoch_seconds=time.monotonic()-epoch_start,updated=now()))
        assert int(opt.iterations.numpy())==(ep+1)*STEPS
        bn0=bn_state(model); t=time.monotonic()
        p,margin,nll,timing=infer(forward,D,dv,all_dev,3)
        bal,means=balanced(nll,dv); dev_seconds=time.monotonic()-t
        old_bal,_=balanced(nll[:n_old_dev],{k:dv[k][:n_old_dev] for k in ['y','patient']})
        new_bal,_=balanced(nll[n_old_dev:],{k:dv[k][n_old_dev:] for k in ['y','patient']})
        assert np.isfinite(bal) and all(np.array_equal(a,b) for a,b in zip(bn0,bn_state(model)))
        best=float('inf') if state['best'] is None else state['best']
        improved=(best-bal)>1e-4; new_min=bal<best
        if new_min:
            temp=out/'best.tmp.weights.h5'; model.save_weights(temp); os.replace(temp,out/'best.weights.h5')
            state['best']=bal; state['best_epoch']=ep+1
        state['wait']=0 if improved else state['wait']+1
        state['epoch']=ep+1; state['finished_training']=state['wait']>=12 or ep+1==EPOCHS
        row.update(dev_old_balanced_ce=old_bal,dev_new_balanced_ce=new_bal,train_loss=loss_sum/len(order),train_accuracy=correct/len(order),read_seconds=read_seconds,train_compute_seconds=train_seconds,first_batch_seconds=first_batch_seconds,dev_seconds=dev_seconds,dev_timing=timing,dev_balanced_ce=bal,dev_N_nll=means[0],dev_V_nll=means[1],dev_N_false_positive=int(np.sum(p[dv['y']==0]>.5)),dev_V_detected=int(np.sum(p[dv['y']==1]>.5)),new_min=new_min,best=state['best'],best_epoch=state['best_epoch'],wait=state['wait'],resources=resources())
        state['rng']={key:rng.bit_generator.state for key,rng in [('order',rng_order),('nonzero',rng_nonzero),('gate',rng_gate)]}
        state['dropout_state']=model.get_layer('do').seed_generator.state.numpy().tolist()
        state['optimizer_sha256']=ah(np.concatenate([v.numpy().ravel() for v in opt.variables]))
        state['runtime']=runtime; state['resources']=resources()
        row['epoch_seconds_before_checkpoint']=time.monotonic()-epoch_start
        state['durations'].append(row)
        snap=out/f'epoch_{ep+1:03d}'; snap.mkdir(exist_ok=True)
        t=time.monotonic(); model.save(snap/'model.keras')
        row['checkpoint_seconds']=time.monotonic()-t
        writej(snap/'state.json',state); writej(out/'last.json',{'snapshot':snap.name,'epoch':ep+1})
        with open(out/'epochs.jsonl','a') as f: f.write(json.dumps(row)+'\n')
        # Keep the two latest complete snapshots; only this run's own directory.
        for prior in sorted(out.glob('epoch_[0-9][0-9][0-9]'))[:-2]:
            assert prior.resolve().parent==out.resolve(); shutil.rmtree(prior)
        print(f"{runid} epoch={ep+1} CE={bal:.7f} best={state['best_epoch']} wait={state['wait']} train={train_seconds:.1f}s dev={dev_seconds:.1f}s",flush=True)
    state['training_process_seconds']=time.monotonic()-start
    state['best_checkpoint_sha256']=sha(out/'best.weights.h5')
    writej(out/'training_summary.json',state)
    # Fixed checkpoint inference is separated so statistical errors never retrain.
    model.load_weights(out/'best.weights.h5')
    from evaluate import evaluate_run
    evaluate_run(model,run,out,state)
    print('RUN COMPLETE',runid,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('runid'); args=parser.parse_args(); main(args.runid)
