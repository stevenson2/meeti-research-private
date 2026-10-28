"""One local continuation for this authorized fixed task, after the live matrix.

Waits without importing TensorFlow; each dependent step is a serial subprocess.
No scheduler, new chat, agent, extra fit or protocol change is created.
"""
from common import *
import subprocess,fcntl,argparse

STAGES=[('development_freeze',['compare_development.py','--freeze']),
        ('training_recovery_audit',['audit_training_results.py']),
        ('independent_release_prepare',['custody_prepare.py']),
        ('independent_serial_evaluation',['run_independent.py']),
        ('independent_prediction_audit',['audit_independent_results.py']),
        ('paired_patient_comparison',['compare_independent.py']),
        ('scientific_figures',['plot_independent.py']),
        ('final_report',['make_final_report.py']),
        ('delivery_inventory',['finalize_delivery.py'])]

def main(resume_waiting=False):
    lock=open(ROOT/'completion_pipeline.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.write(str(os.getpid()));lock.flush()
    path=ROOT/'completion_pipeline_status.json'
    assert readj(ROOT/'holdout_user_choice.json')['choice']=='one_shot_after_12'
    if path.exists():
        assert resume_waiting,'A continuation already has a durable record; inspect it before any restart'
        state=readj(path)
        assert state['status']=='waiting_for_all12_training' and all(s['status']=='pending' for s in state['stages']),'This recovery entry only resumes an untouched waiting continuation'
        assert not (Path('/proc')/str(state['pid'])).exists(),'Previous continuation PID is still alive'
        assert state['training_config_sha256']==sha(ROOT/'frozen_config.json')
        assert [(s['name'],s['command']) for s in state['stages']]==STAGES
        assert not (ROOT/'independent_opened.json').exists()
        state.setdefault('waiting_process_recoveries',[]).append(dict(observed=now(),previous_pid=state['pid'],new_pid=os.getpid(),reason='Previous process absent; all stages still pending; no independent content opened'))
        state['pid']=os.getpid()
    else:
        assert not resume_waiting,'No existing waiting continuation to resume'
        state=dict(started=now(),pid=os.getpid(),status='waiting_for_all12_training',stages=[dict(name=name,command=cmd,status='pending') for name,cmd in STAGES],training_config_sha256=sha(ROOT/'frozen_config.json'),no_additional_fits=True)
    writej(path,state);print('CONTINUATION WAITING: original12 must all complete before any stage',flush=True)
    try:
        while True:
            train=readj(ROOT/'status.json');assert train['config_sha256']==state['training_config_sha256']
            if train.get('implementation_error_needs_review') or any(r['status']=='failed' for r in train['runs']):raise RuntimeError('Training matrix needs attention; independent release remains blocked')
            if len(train['runs'])==12 and all(r['status']=='completed' for r in train['runs']):break
            time.sleep(30)
        state.update(status='running',training_terminal_observed=now());writej(path,state)
        for step in state['stages']:
            name,cmd=step['name'],step['command'];log=ROOT/f'completion_{name}.log'
            step.update(status='running',started=now(),script_sha256=sha(ROOT/cmd[0]),log=log.name);state['active_stage']=name;writej(path,state)
            print('CONTINUATION START',name,flush=True);start=time.monotonic()
            with log.open('w') as f:
                child=subprocess.Popen([sys.executable,'-u',str(ROOT/cmd[0]),*cmd[1:]],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);step['pid']=child.pid;writej(path,state);code=child.wait()
            step.update(returncode=code,finished=now(),seconds=time.monotonic()-start,log_sha256=sha(log))
            if code:
                step.update(status='failed',error_tail=log.read_text(errors='replace')[-6000:]);raise RuntimeError(f'Stage {name} failed with exit{code}; no automatic protocol repair or extra training')
            step['status']='completed';writej(path,state);print('CONTINUATION COMPLETE',name,flush=True)
        state.update(status='completed',finished=now());state.pop('active_stage',None);writej(path,state)
        print('FIXED TASK FULLY COMPLETE:12 training + one12-run independent evaluation, audited and delivered',flush=True)
    except Exception as e:
        state.update(status='needs_attention',error=f'{type(e).__name__}: {e}',stopped=now());writej(path,state);raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--resume-waiting',action='store_true');main(parser.parse_args().resume_waiting)
