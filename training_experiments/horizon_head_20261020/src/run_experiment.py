import os,sys,json,pathlib,subprocess,datetime
ROOT=pathlib.Path(__file__).resolve().parents[1]
PYTHON='/home/aigc/miniconda/envs/human_policy/bin/python'
name=sys.argv[1];gpus=sys.argv[2];smoke='--smoke' in sys.argv
variants={'visual_w01':('baseline',.1),'visual_w1':('baseline',1.),'readiness':('readiness',0.),'geometry':('geometry',0.)}
variant,weight=variants[name]
ngpu=len(gpus.split(','));assert 64%ngpu==0
jobs=json.load(open('/home/aigc/human_policy/experiments/bottle0930_16runs_20261004/jobs.json'))
original=next(j for j in jobs if j['name']=='b0930_20261004_replace50_nostate_vjepa_w0_lagvis')
args=original['command'][3:].copy()
runname=name+('_smoke' if smoke else '')
(ROOT/'results'/f'{runname}_status.json').unlink(missing_ok=True)
ckpt=ROOT/'checkpoints'/runname
for key,value in {'--batch_size':str(64//ngpu),'--exptid':str(ckpt),'--dataset_json_path':str(ROOT/'configs/replace50.json'),'--model_cfg_path':str(ROOT/'configs/baseline.yaml'),'--future_dino_weight':str(weight)}.items():
 args[args.index(key)+1]=value
# The CLI appends _ckpt, so keep its resolved output path explicit.
ckpt=pathlib.Path(str(ckpt)+'_ckpt');ckpt.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=gpus,HDT_EXPERIMENT=variant,PYTHONPATH=str(ROOT/'src')+':'+str(ROOT/'src/hdt'),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false')
if smoke: env['HDT_SMOKE_STEPS']='3'
else: env.pop('HDT_SMOKE_STEPS',None)
command=[PYTHON,'-m','torch.distributed.run','--nproc_per_node',str(ngpu),'--master_port',str(29700+int(gpus.split(',')[0])),'hdt/main.py',*args] if ngpu>1 else [PYTHON,'-u','hdt/main.py',*args]
manifest=dict(name=runname,variant=variant,future_visual_weight=weight,original=original,command=command,global_batch_size=64,batch_size_per_rank=64//ngpu,gpus=gpus,steps=3 if smoke else 100000,initialization='from_scratch',readiness_weight=.1 if variant=='readiness' else 0,geometry_weight=.1 if variant=='geometry' else 0,started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),source_root=str(ROOT/'src'),checkpoint_dir=str(ckpt))
(ROOT/'configs'/f'{runname}.json').write_text(json.dumps(manifest,indent=2))
with open(ROOT/'logs'/f'{runname}.log','ab',buffering=0) as log:
 p=subprocess.Popen(command,cwd=ROOT/'src',env=env,stdout=log,stderr=subprocess.STDOUT)
 manifest['training_pid']=p.pid;(ROOT/'configs'/f'{runname}.json').write_text(json.dumps(manifest,indent=2))
 code=p.wait()
manifest.update(exit_code=code,finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
(ROOT/'results'/f'{runname}_status.json').write_text(json.dumps(manifest,indent=2))
sys.exit(code)
