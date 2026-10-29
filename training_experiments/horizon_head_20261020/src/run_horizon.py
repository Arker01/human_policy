"""Launch only inside this experiment; never stop or rewrite any other job."""
import os,sys,pathlib,json,subprocess,datetime,hashlib
ROOT=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/home/aigc/miniconda/envs/human_policy/bin/python'
name,gpus=sys.argv[1:3];assert name in ('original','geometry')
smoke='--smoke' in sys.argv
assert len(gpus.split(','))==2,'global batch 64 = 32 x 2'
run=name+('_smoke' if smoke else '')
labels=ROOT/'labels'/name
manifest=json.loads((labels/'manifest.json').read_text())
gate=json.loads((ROOT/'results'/name/'stage0_anchored_error.json').read_text())
override='--exploratory-stage0-override' in sys.argv
if not gate['passed']:
 assert override,'Stage0 failed; explicit exploratory override required'
 assert gate['all_p25_p75_gap']>=15,'Even the distribution spread is insufficient'
 print('EXPLORATORY ONLY: strict phase-ratio gate FAILED; user authorized correcting the design and starting training',flush=True)
h=hashlib.sha256()
with open(manifest['teacher'],'rb') as f:
 for b in iter(lambda:f.read(8<<20),b''):h.update(b)
assert h.hexdigest()==manifest['teacher_sha256']
original=next(j for j in json.loads((ROOT/'configs/original_jobs.json').read_text()) if j['name']=='b0930_20261004_replace50_nostate_vjepa_w0_lagvis')
args=original['command'][3:].copy()
for key,value in {'--batch_size':'32','--exptid':str(ROOT/'checkpoints'/run),'--dataset_json_path':str(ROOT/'configs/replace50.json'),'--model_cfg_path':str(ROOT/'configs/baseline.yaml'),'--future_dino_weight':'0.0'}.items():args[args.index(key)+1]=value
assert '--load_pretrained_path' not in args
ckpt=ROOT/'checkpoints'/(run+'_ckpt')
if ckpt.exists() and (ckpt/'metrics.jsonl').exists():raise RuntimeError('Refuse to overwrite/relaunch existing run '+str(ckpt))
ckpt.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpus,HDT_EXPERIMENT='horizon',HDT_HORIZON_BASE='geometry' if name=='geometry' else 'baseline',HDT_HORIZON_LABELS=str(labels),HDT_HORIZON_TEACHER_SHA256=manifest['teacher_sha256'],HDT_HORIZON_INITIAL_WEIGHTS=manifest['teacher'],PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'src/hdt'),PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
if smoke:env['HDT_SMOKE_STEPS']='3'
else:env.pop('HDT_SMOKE_STEPS',None)
command=[PYTHON,'-u','-m','torch.distributed.run','--nproc_per_node','2','--master_port',str(29820+int(gpus.split(',')[0])),'hdt/main.py',*args]
record=dict(name=run,variant='horizon',base_variant=env['HDT_HORIZON_BASE'],command=command,gpus=gpus,global_batch_size=64,steps=3 if smoke else 100000,initialization=manifest['teacher'],teacher_sha256=manifest['teacher_sha256'],labels=str(labels),stage0_strict_passed=gate['passed'],stage0_exploratory_override=override,stage0_override_reason='User permits correcting unreliable design assumptions and explicitly requires training; phase median ratio is not a mathematical test of label constancy; full dex5 IQR >=15 retained',horizon_weight=.1,geometry_weight=.1 if name=='geometry' else 0,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),launcher_pid=os.getpid(),checkpoint_dir=str(ckpt))
(ROOT/'configs'/(run+'.json')).write_text(json.dumps(record,indent=2))
with open(ROOT/'logs'/(run+'.log'),'ab',buffering=0) as log:
 p=subprocess.Popen(command,cwd=ROOT/'src',env=env,stdout=log,stderr=subprocess.STDOUT)
 record['training_pid']=p.pid;(ROOT/'configs'/(run+'.json')).write_text(json.dumps(record,indent=2));code=p.wait()
record.update(exit_code=code,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(ROOT/'results'/(run+'_status.json')).write_text(json.dumps(record,indent=2))
if code==0 and not smoke:
 post=dict(name=name,status='running',steps=100000)
 with open(ROOT/'logs'/(name+'_posttrain.log'),'ab',buffering=0) as log:
  for cmd in ([PYTHON,'-u',str(ROOT/'src/evaluate_horizon.py'),name],[PYTHON,'-u',str(ROOT/'src/mpjpe_comparison.py'),name],[PYTHON,'-u',str(ROOT/'src/mpjpe_comparison.py'),name+'_reference'],[PYTHON,'-u',str(ROOT/'src/verify_originals.py')]):
   result=subprocess.run(cmd,cwd=ROOT/'src',env=env,stdout=log,stderr=subprocess.STDOUT)
   if result.returncode:
    post.update(status='failed',command=cmd,exit_code=result.returncode);break
  else:post.update(status='completed',exit_code=0)
 (ROOT/'results'/(name+'_posttrain_status.json')).write_text(json.dumps(post,indent=2))
sys.exit(code)
