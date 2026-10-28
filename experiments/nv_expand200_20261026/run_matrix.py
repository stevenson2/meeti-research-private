"""One simple sequential process loop. No service, no external scheduler."""
from common import *
import subprocess,fcntl,traceback

def other_training():
    found=[]
    for p in Path('/proc').glob('[0-9]*'):
        try:
            argv=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if not argv or 'python' not in Path(argv[0]).name.lower(): continue
            scripts=[x for x in argv[1:] if x.endswith('.py')]
            if any('train' in Path(x).name.lower() for x in scripts): found.append({'pid':int(p.name),'argv':argv})
        except (OSError,ValueError): pass
    return found

def main():
    lock=open(ROOT/'matrix.lock','w'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock.write(str(os.getpid())); lock.flush()
    assert readj(ROOT/'verification.json')['passed']
    fingerprint=sha(ROOT/'frozen_config.json'); statusfile=ROOT/'status.json'
    status=readj(statusfile) if statusfile.exists() else dict(started=now(),config_sha256=fingerprint,runs=[{**r,'attempts':[],'status':'planned'} for r in readj(ROOT/'matrix.json')['runs']])
    assert status['config_sha256']==fingerprint
    (ROOT/'runs').mkdir(exist_ok=True)
    for run in status['runs']:
        out=ROOT/'runs'/run['id']; out.mkdir(exist_ok=True)
        if (out/'complete.json').exists():
            complete=readj(out/'complete.json')
            assert complete['config_sha256']==fingerprint and sha(out/'best.weights.h5')==complete['checkpoint_sha256'] and sha(out/'metrics.json')==complete['metrics_sha256']
            run['status']='completed'; writej(statusfile,status); continue
        if run['status']=='failed': continue
        other=other_training()
        while other:
            status['waiting_for_other_training']=other; writej(statusfile,status)
            print('WAITING FOR OTHER TRAINING',other,flush=True); time.sleep(30); other=other_training()
        status.pop('waiting_for_other_training',None)
        if len(run['attempts'])>=2:
            status['recovery_budget_exhausted']=run['id'];writej(statusfile,status)
            raise RuntimeError('Original configuration process-attempt budget exhausted; no extra automatic fit')
        for retry in range(len(run['attempts']),2):
            run['status']='running'; attempt=len(run['attempts'])+1
            log=out/f'attempt_{attempt:02d}.log'; info=dict(attempt=attempt,started=now(),log=log.name,resume=(out/'last.json').exists())
            run['attempts'].append(info); writej(statusfile,status)
            print('START',run['id'],'attempt',attempt,flush=True); t=time.monotonic()
            with open(log,'w') as f:
                proc=subprocess.Popen([sys.executable,'-u',str(ROOT/'train_one.py'),run['id']],stdout=f,stderr=subprocess.STDOUT,cwd=ROOT)
                info['pid']=proc.pid; writej(statusfile,status); code=proc.wait()
            info.update(returncode=code,finished=now(),seconds=time.monotonic()-t)
            if code==0 and (out/'complete.json').exists():
                run['status']='completed'; writej(statusfile,status); print('COMPLETED',run['id'],flush=True); break
            tail=log.read_text(errors='replace')[-6000:]; info['error_tail']=tail
            ordinary=code in (-9,137) or any(x in tail for x in ['ResourceExhaustedError','out of memory','Input/output error','No space left','OSError','IOError'])
            if ordinary and retry==0:
                print('RETRY ORIGINAL CONFIG',run['id'],flush=True); writej(statusfile,status); continue
            run['status']='failed'; writej(statusfile,status); print('FAILED',run['id'],tail,flush=True)
            if not ordinary:
                status['implementation_error_needs_review']=run['id']; writej(statusfile,status)
                raise RuntimeError('Deterministic error: inspect and repair shared implementation before proceeding')
            break
    status['finished']=now(); writej(statusfile,status)
    print('MATRIX TERMINAL', {x:sum(r['status']==x for r in status['runs']) for x in ['completed','failed','planned','running']},flush=True)

if __name__=='__main__': main()
