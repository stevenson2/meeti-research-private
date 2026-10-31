import os
os.environ.update(TF_CPP_MIN_LOG_LEVEL='3',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
import hashlib,json
from pathlib import Path
import numpy as np
from tensorflow.lite.python import schema_py_generated as schema
R=Path(os.environ['ECG_INT8_SOURCE'])
W=Path(__file__).resolve().parent;O=W.parents[1]/'outputs/PTQ_REPAIR_20261029'
frozen=json.loads((O/'FREEZE.json').read_text());out={}
def model(p):return schema.ModelT.InitFromObj(schema.Model.GetRootAsModel(p.read_bytes(),0))
def code(m,op):return m.operatorCodes[op.opcodeIndex].builtinCode
def ops(m):return [op for op in m.subgraphs[0].operators if code(m,op) in (schema.BuiltinOperator.CONV_2D,schema.BuiltinOperator.DEPTHWISE_CONV_2D,schema.BuiltinOperator.FULLY_CONNECTED)]
def tensor(m,index):return m.subgraphs[0].tensors[int(index)]
def weight(m,op):
 t=tensor(m,op.inputs[1]);data=m.buffers[t.buffer].data
 return t,data
def mac(m):
 total=0
 for op in ops(m):
  t,_=weight(m,op);shape=list(t.shape);oshape=list(tensor(m,op.outputs[0]).shape)
  n=int(np.prod(oshape));c=code(m,op)
  if c==schema.BuiltinOperator.CONV_2D:total+=n*shape[1]*shape[2]*shape[3]
  elif c==schema.BuiltinOperator.DEPTHWISE_CONV_2D:total+=n*shape[1]*shape[2]
  else:total+=n*shape[-1]
 return int(total)
for key,digest in frozen['model_sha256'].items():
 path=O/'models'/f'{key}_mse_int8_logits.tflite';assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
 old=model(R/'models'/f'{key}_int8.tflite');new=model(path)
 before=model(W/f'{key}_calibrated_float.tflite');after=model(W/f'{key}_mse_calibrated_float.tflite')
 assert len(before.buffers)==len(after.buffers)
 assert all(np.array_equal(a.data,b.data) for a,b in zip(before.buffers,after.buffers))
 assert [code(old,x) for x in old.subgraphs[0].operators[:-1]]==[code(new,x) for x in new.subgraphs[0].operators]
 oldops,newops=ops(old),ops(new);assert len(oldops)==len(newops)
 matches=[];qparam=[];changed_bias=0
 for a,b in zip(oldops,newops):
  ta,da=weight(old,a);tb,db=weight(new,b)
  assert np.array_equal(ta.shape,tb.shape)
  matches.append(bool(np.array_equal(da,db)))
  qparam.append(bool(np.array_equal(ta.quantization.scale,tb.quantization.scale) and np.array_equal(ta.quantization.zeroPoint,tb.quantization.zeroPoint)))
  if len(a.inputs)>=3 and int(a.inputs[2])>=0 and int(b.inputs[2])>=0:
   ba=tensor(old,a.inputs[2]);bb=tensor(new,b.inputs[2])
   changed_bias+=int(not np.array_equal(old.buffers[ba.buffer].data,new.buffers[bb.buffer].data))
 oldmac,newmac=mac(old),mac(new);assert oldmac==newmac==1288192
 types=sorted({t.type for t in new.subgraphs[0].tensors});assert set(types)<={schema.TensorType.INT8,schema.TensorType.INT32}
 info=json.loads((W/f'{key}_repair.json').read_text())
 assert hashlib.sha256((R/'models'/f'{key}.weights.h5').read_bytes()).hexdigest()==info['source_weights_sha256']
 out[key]={'calibrated_float_all_constant_buffers_unchanged':True,'upstream_operator_types_unchanged':True,
  'conv_dense_MAC_original':oldmac,'conv_dense_MAC_repaired':newmac,'weight_tensor_count':len(matches),
  'weight_integer_buffers_equal_to_original':sum(matches),'weight_quantization_parameters_equal_to_original':sum(qparam),
  'integer_bias_buffers_changed_due_to_requantization':changed_bias,'only_int8_int32_tensor_types':True,
  'source_float_weights_sha256_unchanged':info['source_weights_sha256']}
(O/'graph_audit.json').write_text(json.dumps({'status':'PASS','models':out},indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
