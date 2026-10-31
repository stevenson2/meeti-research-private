import os,sys
os.environ.update(OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2',
 TF_NUM_INTRAOP_THREADS='2',TF_NUM_INTEROP_THREADS='2',TF_CPP_MIN_LOG_LEVEL='3',PYTHONDONTWRITEBYTECODE='1')
import copy,gc,hashlib,json,time
from pathlib import Path
import numpy as np,tensorflow as tf,flatbuffers
from tensorflow.lite.python import convert,schema_py_generated as schema

R=Path(os.environ['ECG_INT8_SOURCE'])
P=Path(os.environ['ECG_SMALLMODEL_SOURCE'])
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/PTQ_REPAIR_20261029'
(O/'models').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,os.environ['ECG_EXPAND200_SOURCE'])
import models
tf.config.threading.set_intra_op_parallelism_threads(2);tf.config.threading.set_inter_op_parallelism_threads(2)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def pack(m):
 b=flatbuffers.Builder(200000);offset=m.Pack(b);b.Finish(offset,file_identifier=b'TFL3');return bytes(b.Output())
def summary(d):return {'index':int(d['index']),'name':d['name'],'dtype':str(d['dtype']),
 'shape':d['shape'].tolist(),'shape_signature':d['shape_signature'].tolist(),
 'scale':float(d['quantization'][0]),'zero_point':int(d['quantization'][1])}
def write(name,obj):(W/name).write_text(json.dumps(obj,indent=2),encoding='utf-8')

def main(key):
 started=time.monotonic();source=R/'models'/f'{key}.weights.h5';source_sha=sha(source)
 m=models.build('S',int(key[4:6]));m.load_weights(str(source))
 assert m.count_params()==62834
 source_weights={f'{i}:{w.name}':hashlib.sha256(w.numpy().tobytes()).hexdigest() for i,w in enumerate(m.weights)}
 X=np.load(P/'data/selected_X.npy',mmap_mode='r');ids=np.load(R/'representative_ids.npy')
 def representative():
  for i in ids:yield [np.asarray(X[i],np.float32).reshape(1,250,1)]
 converter=tf.lite.TFLiteConverter.from_keras_model(m)
 converter.optimizations=[tf.lite.Optimize.DEFAULT];converter.representative_dataset=representative
 converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
 converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
 converter._experimental_calibrate_only=True
 calibrated=converter.convert();(W/f'{key}_calibrated_float.tflite').write_bytes(calibrated)
 graph=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(calibrated,0));sg=graph.subgraphs[0]
 assert len(sg.outputs)==1
 protected=set();softmax=None
 for op in sg.operators:
  code=graph.operatorCodes[op.opcodeIndex].builtinCode
  if code in (schema.BuiltinOperator.LOGISTIC,schema.BuiltinOperator.SOFTMAX):protected.update(map(int,op.outputs))
  if code==schema.BuiltinOperator.SOFTMAX and int(sg.outputs[0]) in op.outputs:softmax=op
 assert softmax is not None
 protected.add(int(softmax.inputs[0]))
 changed=True
 while changed:
  changed=False
  for op in sg.operators:
   if graph.operatorCodes[op.opcodeIndex].builtinCode in (schema.BuiltinOperator.RESHAPE,schema.BuiltinOperator.EXPAND_DIMS):
    if int(op.inputs[0]) in protected:
     for i in op.outputs:
      if int(i) not in protected:protected.add(int(i));changed=True
 hist={}
 for i,t in enumerate(sg.tensors):
  q=t.quantization
  buf=graph.buffers[t.buffer].data
  if t.type!=schema.TensorType.FLOAT32 or (buf is not None and len(buf)) or i in protected or q is None:continue
  if q.min is None or q.max is None or len(q.min)!=1 or len(q.max)!=1:continue
  lo=min(float(q.min[0]),0.);hi=max(float(q.max[0]),0.)
  if hi-lo<1e-12:continue
  hist[i]={'lo':lo,'hi':hi,'counts':np.zeros(4096,np.int64),'zeros':0,'elements':0,'outside_old_range':0}
 assert 0 in hist, 'input not eligible for MSE range calibration'
 it=tf.lite.Interpreter(model_content=calibrated,experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
    experimental_preserve_all_tensors=True,num_threads=1);it.allocate_tensors();inp=it.get_input_details()[0];out=it.get_output_details()[0]
 assert inp['dtype']==np.float32
 reference=np.load(R/'predictions'/f'{key}_keras_pre.npz');assert np.array_equal(ids,reference['ids'][:1000])
 float_parity_p=0.;float_parity_z=0.
 for j,beat in enumerate(ids):
  x=np.asarray(X[beat],np.float32).reshape(1,250,1);it.set_tensor(inp['index'],x);it.invoke()
  p=it.get_tensor(out['index'])[0];z=it.get_tensor(int(softmax.inputs[0]))[0]
  float_parity_p=max(float_parity_p,float(abs(p-reference['prob'][j]).max()))
  float_parity_z=max(float_parity_z,float(abs(z-reference['logits'][j]).max()))
  for index,h in hist.items():
   values=it.get_tensor(index).ravel();zero=values==0;h['zeros']+=int(zero.sum());h['elements']+=len(values)
   values=values[~zero];h['outside_old_range']+=int(np.sum((values<h['lo'])|(values>h['hi'])))
   h['counts']+=np.histogram(np.clip(values,h['lo'],h['hi']),bins=4096,range=(h['lo'],h['hi']))[0]
  if (j+1)%250==0:print(key,'histogram_fit',j+1,flush=True)
 del it;gc.collect()
 ranges=[];factors=np.linspace(.25,1.,97)
 for index,h in hist.items():
  centers=h['lo']+(np.arange(4096)+.5)*(h['hi']-h['lo'])/4096
  counts=h['counts'];best=None;unclipped=None
  for factor in factors:
   lo=h['lo']*factor;hi=h['hi']*factor;s=(hi-lo)/255
   zp=int(np.clip(np.floor(-128-lo/s+.5),-128,127))
   q=np.clip(np.rint(centers/s)+zp,-128,127);recovered=(q-zp)*s
   error=float(np.dot(counts.astype(float),(centers-recovered)**2))/h['elements']
   if factor==1.:unclipped=error
   if best is None or error<best[0]*(1-1e-12) or (abs(error-best[0])<=max(abs(best[0]),1e-300)*1e-12 and factor>best[1]):best=(error,float(factor),lo,hi)
  error,factor,lo,hi=best
  tensor=sg.tensors[index];tensor.quantization.min=np.array([lo],np.float32);tensor.quantization.max=np.array([hi],np.float32)
  ranges.append({'tensor_index':index,'name':tensor.name.decode() if isinstance(tensor.name,bytes) else tensor.name,
    'old_min':h['lo'],'old_max':h['hi'],'new_min':lo,'new_max':hi,'factor':factor,
    'mse_estimate':error,'original_range_mse_estimate':unclipped,'elements':h['elements'],
    'exact_zeros':h['zeros'],'outside_original_calibrated_range':h['outside_old_range']})
 adjusted=pack(graph);(W/f'{key}_mse_calibrated_float.tflite').write_bytes(adjusted)
 quantized=convert.mlir_quantize(adjusted,disable_per_channel=False,fully_quantize=True,
    input_data_type=tf.int8,output_data_type=tf.int8)
 full=schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(quantized,0));sgq=full.subgraphs[0]
 assert full.operatorCodes[sgq.operators[-1].opcodeIndex].builtinCode==schema.BuiltinOperator.SOFTMAX
 zi=int(sgq.operators[-1].inputs[0]);sgq.outputs=np.array([zi],np.int32);sgq.operators=sgq.operators[:-1]
 for sig in full.signatureDefs or []:
  for t in sig.outputs:t.tensorIndex=zi;t.name=b'logits_int8'
 for i in [int(sgq.inputs[0]),zi]:sgq.tensors[i].shapeSignature=sgq.tensors[i].shape.copy()
 model=pack(full);dest=O/'models'/f'{key}_mse_int8_logits.tflite';dest.write_bytes(model)
 it=tf.lite.Interpreter(model_path=str(dest),experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,num_threads=1)
 it.allocate_tensors();inp=it.get_input_details()[0];out=it.get_output_details()[0]
 types=sorted({str(d['dtype']) for d in it.get_tensor_details()});assert set(types)<={str(np.int8),str(np.int32)},types
 assert inp['shape_signature'].tolist()==[1,250,1] and out['shape_signature'].tolist()==[1,2]
 assert inp['dtype']==np.int8 and out['dtype']==np.int8
 assert len(out['quantization_parameters']['scales'])==1
 # Only pre/dev-independent representative+calibration inference is permitted in this phase.
 pre=np.load(R/'predictions'/f'{key}_int8_pre.npz');q=np.empty((len(pre['ids']),2),np.int8);s,zp=inp['quantization']
 for j,beat in enumerate(pre['ids']):
  x=np.asarray(X[beat],float);v=np.clip(np.rint(x/s)+zp,-128,127).astype(np.int8).reshape(1,250,1)
  it.set_tensor(inp['index'],v);it.invoke();q[j]=it.get_tensor(out['index'])[0]
 np.savez(W/f'{key}_repaired_pre.npz',ids=pre['ids'],zq=q)
 assert sha(source)==source_sha
 assert source_weights=={f'{i}:{w.name}':hashlib.sha256(w.numpy().tobytes()).hexdigest() for i,w in enumerate(m.weights)}
 info={'key':key,'recipe_config_sha256':sha(W/'PREREG.json'),'tensorflow':tf.__version__,
  'tf_onednn':os.environ.get('TF_ENABLE_ONEDNN_OPTS'),'source_weights_sha256':source_sha,
  'source_keras_weights_unchanged':True,'keras_params':m.count_params(),'representative_ids_sha256':sha(R/'representative_ids.npy'),
  'model_sha256':sha(dest),'bytes':dest.stat().st_size,'input':summary(inp),'output':summary(out),
  'tensor_types':types,'operators':[{ 'op_name':o['op_name'],'inputs':o['inputs'].tolist(),'outputs':o['outputs'].tolist()} for o in it._get_ops_details()],
  'n_range_candidates_per_tensor':97,'n_eligible_tensors':len(hist),'n_ranges_shrunk':sum(x['factor']<1 for x in ranges),
  'input_range':next(x for x in ranges if x['tensor_index']==int(sg.inputs[0])),
  'fit_float_reference_probability_max_abs':float_parity_p,'fit_float_reference_logit_max_abs':float_parity_z,
  'fit_float_reference_within_original_tolerances':float_parity_p<=1e-5 and float_parity_z<=1e-4,
  'ranges':ranges,'seconds':time.monotonic()-started,'calibration_inferred':9097,'dev_inferred':False}
 write(f'{key}_repair.json',info)
 print(json.dumps({k:info[k] for k in ('key','bytes','n_eligible_tensors','n_ranges_shrunk','input','output','fit_float_reference_probability_max_abs','seconds')}),flush=True)

if __name__=='__main__':main(sys.argv[1])
