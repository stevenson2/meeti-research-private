"""Fixed NV 27-run protocol. No training on import."""
import os, sys, site
from pathlib import Path

def bootstrap():
    if os.environ.get('NV_LONGRUN_LIBS') != '1':
        libs = sorted({str(p) for r in site.getsitepackages()+[site.getusersitepackages()] for p in Path(r).glob('nvidia/*/lib')})
        env = os.environ.copy()
        env.update(NV_LONGRUN_LIBS='1', TF_CPP_MIN_LOG_LEVEL='1', TF_ENABLE_ONEDNN_OPTS='0', PYTHONUNBUFFERED='1')
        env['LD_LIBRARY_PATH'] = ':'.join(libs+['/usr/lib/wsl/lib',env.get('LD_LIBRARY_PATH','')])
        os.execve(sys.executable, [sys.executable]+sys.argv, env)

import json, hashlib, time, resource
import numpy as np
ROOT = Path(__file__).resolve().parent
FROZEN_DATA = json.loads((ROOT/'development_qc.json').read_text())
DATA_ROOT = Path(FROZEN_DATA['cache_files']['train_X.npy']['path']).parent
R2 = ROOT.parent/'NV_BASELINE_R2_SHIFT'
R1 = ROOT
SHIFTS = [-4,-2,-1,0,1,2,4]
BATCH, EVAL_BATCH, EPOCHS, STEPS = 256,1024,80,FROZEN_DATA['steps_per_epoch']
CW = np.array(FROZEN_DATA['class_weights'],np.float32)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()

def ah(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def readj(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def writej(path,obj):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    os.replace(tmp,path)
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()

def configure_tf():
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.experimental.enable_op_determinism()
    tf.keras.mixed_precision.set_global_policy('float32')
    gpus=tf.config.list_physical_devices('GPU')
    assert gpus, 'GPU unavailable; cannot silently switch evaluation to CPU'
    tf.config.set_logical_device_configuration(gpus[0],[tf.config.LogicalDeviceConfiguration(memory_limit=3072)])
    return dict(tensorflow=tf.__version__,numpy=np.__version__,gpu=tf.config.experimental.get_device_details(gpus[0]),tf32=False,op_determinism=True,precision='float32',gpu_limit_mb=3072,intra=4,inter=1,eval_batch=1024,train_batch=256,device_prefetch=0,host_prefetch=0,reader_workers=1)

def resources():
    import tensorflow as tf
    d=dict(rss_peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024)
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith(('VmRSS:','RssAnon:','RssFile:')): d[line.split(':')[0]+'_mb']=int(line.split()[1])/1024
    try: d['tf_gpu_memory_bytes']=tf.config.experimental.get_memory_info('GPU:0')
    except Exception: pass
    return d

def metadata(split):
    assert split in ('train','dev'), 'INDEPENDENT_CONTENT_DENIED'
    with np.load(DATA_ROOT/f'{split}_meta.npz') as f: return {k:f[k] for k in f.files}
def data(split):
    assert split in ('train','dev'), 'INDEPENDENT_CONTENT_DENIED'
    return np.load(DATA_ROOT/f'{split}_X.npy',mmap_mode='r'),metadata(split)
def batch_input(X,rows,columns): return np.ascontiguousarray(X[rows,columns,:][...,None],np.float32)
def patient_names(split): return [p['patient'] for p in readj(R1/'prep_manifest.json')['patients'][split]]
def identity(meta,rows):
    seg=[s['segment'] for s in readj(R1/'prep_manifest.json')['segments']]
    return [f"{seg[int(meta['segment'][i])]} a{int(meta['annotation_index'][i])}" for i in rows]

def balanced(nll,meta):
    means=[]
    for c in (0,1):
        means.append(float(np.mean([np.mean(nll[(meta['patient']==p)&(meta['y']==c)],dtype=np.float64) for p in np.unique(meta['patient']) if np.any((meta['patient']==p)&(meta['y']==c))])))
    return 0.5*sum(means),means

def make_forward(model):
    import tensorflow as tf
    shared=tf.keras.Model(model.input,[model.output,model.get_layer('do').output])
    out=model.get_layer('out')
    @tf.function(input_signature=[tf.TensorSpec([None,250,1],tf.float32)],jit_compile=False)
    def f(x):
        p,feat=shared(x,training=False)
        logits=tf.linalg.matmul(feat,out.kernel)+out.bias
        return p,logits,tf.nn.log_softmax(logits)
    return f

def infer(forward,X,meta,rows,col):
    p=np.empty(len(rows),np.float32); margin=np.empty_like(p); nll=np.empty_like(p)
    read_seconds=0.; gpu_seconds=0.; first_batch_gpu_seconds=None
    for start in range(0,len(rows),EVAL_BATCH):
        ix=rows[start:start+EVAL_BATCH]; t=time.monotonic()
        x=batch_input(X,ix,col); read_seconds+=time.monotonic()-t
        t=time.monotonic(); pr,lg,ls=forward(x)
        pr,lg,ls=pr.numpy(),lg.numpy(),ls.numpy()
        end=start+len(ix); p[start:end]=pr[:,1]; margin[start:end]=lg[:,1]-lg[:,0]
        nll[start:end]=-ls[np.arange(len(ix)),meta['y'][ix].astype(int)]
        elapsed=time.monotonic()-t; gpu_seconds+=elapsed
        if start==0: first_batch_gpu_seconds=elapsed
    return p,margin,nll,dict(read_seconds=read_seconds,gpu_seconds=gpu_seconds,first_batch_gpu_seconds=first_batch_gpu_seconds)

def bn_state(model):
    import tensorflow as tf
    return [a.numpy().copy() for l in model.layers if isinstance(l,tf.keras.layers.BatchNormalization) for a in (l.moving_mean,l.moving_variance)]

def compare(p,m,p0,m0):
    if len(p)==0: return dict(probability_abs=None,margin_abs=None,flips=0,exact=True)
    dp=np.abs(p.astype(np.float64)-p0); dm=np.abs(m.astype(np.float64)-m0)
    def summary(v): return dict(mean=float(v.mean()),max=float(v.max()),q50=float(np.quantile(v,.5)),q95=float(np.quantile(v,.95)),q99=float(np.quantile(v,.99)))
    return dict(probability_abs=summary(dp),margin_abs=summary(dm),flips=int(np.sum((p>.5)!=(p0>.5))),exact=bool(np.array_equal(p,p0) and np.array_equal(m,m0)))
