"""Detached training completion checks for the two no-left-wrist runs."""
import pathlib,json,time,math,hashlib,datetime,os
ROOT=pathlib.Path(__file__).resolve().parents[1]
RUNS=['geometry','visual_w1_geometry']
def latest(path):
 if not path.exists():return {}
 with path.open('rb') as f:
  f.seek(max(0,path.stat().st_size-65536));lines=f.read().splitlines()
 for line in reversed(lines):
  try:return json.loads(line)
  except (ValueError,UnicodeError):pass
 return {}
try:
 while True:
  state={'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{}}
  all_done=True
  for name in RUNS:
   ckpt=ROOT/'checkpoints'/(name+'_ckpt');row=latest(ckpt/'metrics.jsonl')
   assert all(math.isfinite(v) for v in row.values() if isinstance(v,float)),(name,row)
   status=ROOT/'results'/(name+'_status.json')
   progress={'step':row.get('step',0),'target_steps':100000,'train_loss':row.get('train/loss')}
   if status.exists():
    result=json.loads(status.read_text());assert result['exit_code']==0,(name,result['exit_code'])
    assert row.get('step')==100000,(name,row)
    assert (ckpt/'policy_last.ckpt').stat().st_size>100000000
    progress['state']='complete'
   else:
    progress['state']='running';all_done=False
    manifest=ROOT/'configs'/(name+'.json')
    if manifest.exists():
     job=json.loads(manifest.read_text())
     for pid in [job['pid'],job.get('training_pid',job['pid'])]:
      os.kill(pid,0)
   state['runs'][name]=progress
  (ROOT/'results/progress.json').write_text(json.dumps(state,indent=2))
  if all_done:
   import torch
   expected=torch.ones(128);expected[80:89]=0;expected[98:128]=0
   for name in RUNS:
    weights=torch.load(ROOT/'checkpoints'/(name+'_ckpt')/'policy_last.ckpt',map_location='cpu',weights_only=True)
    assert torch.equal(weights['model.state_keep_mask'],expected),name
   hashes=json.loads((ROOT/'configs/original_source_sha256.json').read_text())
   assert all(hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()==h for p,h in hashes.items())
   state.update(original_source_unchanged=True,checkpoint_input_masks_verified=True)
   (ROOT/'results/completion_checks.json').write_text(json.dumps(state,indent=2))
   print('BOTH_TRAINING_RUNS_COMPLETE',json.dumps(state),flush=True)
   break
  time.sleep(30)
except Exception as exc:
 (ROOT/'results/monitor_error.json').write_text(json.dumps({'error':repr(exc),'time':datetime.datetime.now(datetime.timezone.utc).isoformat()},indent=2))
 raise
