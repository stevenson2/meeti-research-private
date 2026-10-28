"""One serial local-GPU evaluation loop after the custody gate; no training."""
from common import *
import subprocess,fcntl
from run_matrix import other_training
from custody_prepare import release_gate
def main():
    lock=open(ROOT/'independent.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    gate=release_gate();assert (ROOT/'independent_preparation.json').exists()
    path=ROOT/'independent_status.json';status=readj(path) if path.exists() else dict(started=now(),release_gate_sha256=sha(ROOT/'checkpoints_frozen_for_independent.json'),runs=[{**r,'status':'planned','attempts':[]} for r in readj(ROOT/'matrix.json')['runs']])
    assert status['release_gate_sha256']==sha(ROOT/'checkpoints_frozen_for_independent.json')
    for r in status['runs']:
        out=ROOT/'independent'/r['id'];out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():
            done=readj(out/'complete.json');assert done['checkpoint_sha256']==gate['checkpoints'][r['id']]['sha256'] and sha(out/'metrics.json')==done['metrics_sha256'];r['status']='completed';writej(path,status);continue
        if r['status']=='failed':continue
        while other_training():
            status['waiting_for_other_training']=True;writej(path,status);time.sleep(30)
        status.pop('waiting_for_other_training',None)
        if len(r['attempts'])>=2:
            status['recovery_budget_exhausted']=r['id'];writej(path,status)
            raise RuntimeError('Fixed independent process-attempt budget exhausted; no extra automatic evaluation')
        for retry in range(len(r['attempts']),2):
            r['status']='running';attempt=len(r['attempts'])+1;log=out/f'attempt_{attempt:02d}.log';info=dict(attempt=attempt,started=now(),log=log.name);r['attempts'].append(info);writej(path,status);start=time.monotonic();print('INDEPENDENT START',r['id'],flush=True)
            with log.open('w') as f:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'independent_evaluate.py'),r['id']],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT);info['pid']=proc.pid;writej(path,status);code=proc.wait()
            info.update(returncode=code,finished=now(),seconds=time.monotonic()-start)
            if code==0 and (out/'complete.json').exists():r['status']='completed';writej(path,status);print('INDEPENDENT COMPLETED',r['id'],flush=True);break
            tail=log.read_text(errors='replace')[-5000:];info['error_tail']=tail
            ordinary=code in [-9,137] or any(s in tail for s in ['ResourceExhaustedError','out of memory','Input/output error','OSError','IOError'])
            if ordinary and retry==0:writej(path,status);continue
            r['status']='failed';writej(path,status)
            if not ordinary:status['implementation_error_needs_review']=r['id'];writej(path,status);raise RuntimeError('Independent evaluation implementation/integrity error; no retraining or post-test tuning')
            break
    status['finished']=now();writej(path,status);print('INDEPENDENT MATRIX TERMINAL',flush=True)
if __name__=='__main__':main()
