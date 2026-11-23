"""Durable completion checks and fixed prior-only validation for the four runs."""
import pathlib,json,time,subprocess,os,hashlib,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/home/aigc/miniconda/envs/human_policy/bin/python'
RUNS={'visual_w01':'0','visual_w1':'2','readiness':'4','geometry':'6'}
processes={};validated=set();finished=set()

def latest(path):
    if not path.exists():return {}
    with path.open('rb') as f:
        f.seek(max(0,path.stat().st_size-16384));lines=f.read().splitlines()
    return json.loads(lines[-1]) if lines else {}

def launch_eval(name,gpu):
    f=open(ROOT/'logs'/(name+'_fixed_validation.log'),'ab',buffering=0)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONPATH=str(ROOT/'src'))
    p=subprocess.Popen([PYTHON,str(ROOT/'src/evaluate_experiment.py'),name],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,env=env)
    processes[name]=(p,f);print('VALIDATION_STARTED',name,p.pid,flush=True)

while True:
    progress={}
    for name,gpu in RUNS.items():
        ckpt=ROOT/'checkpoints'/(name+'_ckpt')
        row=latest(ckpt/'metrics.jsonl');progress[name]={k:round(v,5) if isinstance(v,float) else v for k,v in row.items() if k in ('step','train/loss','train/l1','train/readiness_loss','train/geometry_error_cm','train/future_dino_loss')}
        for k,v in row.items():
            if isinstance(v,float) and (v!=v or abs(v)==float('inf')):
                raise RuntimeError(f'Nonfinite metric {name} {k} {v}')
        status=ROOT/'results'/(name+'_status.json')
        if status.exists() and name not in finished:
            st=json.load(open(status));assert st['exit_code']==0,st
            assert row.get('step')==100000,(name,row)
            assert (ckpt/'policy_last.ckpt').stat().st_size>100_000_000
            finished.add(name);launch_eval(name,gpu)
    for name,(p,f) in list(processes.items()):
        code=p.poll()
        if code is not None:
            f.close();assert code==0,(name,code)
            validated.add(name);del processes[name];print('VALIDATION_FINISHED',name,flush=True)
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    state=dict(time=now,training=progress,finished=sorted(finished),validated=sorted(validated),validation_pids={n:p.pid for n,(p,f) in processes.items()})
    (ROOT/'results/progress.json').write_text(json.dumps(state,indent=2))
    print(json.dumps(state),flush=True)
    if len(finished)==4 and len(validated)==4:
        launch_eval('original','0')
        p,f=processes['original'];code=p.wait();f.close();assert code==0,code
        original_hashes=json.load(open(ROOT/'configs/original_source_sha256.json'))
        changed=[p for p,h in original_hashes.items() if hashlib.sha256((pathlib.Path('/home/aigc/human_policy')/p).read_bytes()).hexdigest()!=h]
        assert not changed,changed
        result=dict(completed_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_steps=100000,global_batch_size=64,original_code_verified_unchanged=True,training={n:json.load(open(ROOT/'results'/(n+'_status.json'))) for n in RUNS},validation={n:json.load(open(ROOT/'results'/(n+'_fixed_validation.json'))) for n in [*RUNS,'original']})
        (ROOT/'results/completed.json').write_text(json.dumps(result,indent=2));print('ALL_TRAINING_AND_VALIDATION_COMPLETE',flush=True);break
    time.sleep(30)
