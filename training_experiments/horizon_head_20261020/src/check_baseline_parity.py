"""Compare old vs isolated baseline inference in separate interpreters, no saved inputs."""
import os,sys,pathlib,subprocess,json
ROOT=pathlib.Path(__file__).resolve().parents[1]
CODE=r'''
import sys,json,torch,hashlib
sys.path[:0]=[sys.argv[1],sys.argv[1]+'/hdt']
from policy import ACTPolicy
torch.set_num_threads(4);torch.manual_seed(123)
c=json.load(open('/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_readiness_geometry/checkpoints/geometry_ckpt/training_config.json'))
a=c['config']['policy_config'].copy();a['experiment_variant']='baseline'
p=ACTPolicy(a).cuda().eval();p.load_state_dict(torch.load('/mnt/nvme0n1/human_policy_ckpt/b0930_20261004_replace50_nostate_vjepa_w0_lagvis_ckpt/policy_last.ckpt',map_location='cpu',weights_only=True),strict=True)
torch.manual_seed(123)
with torch.inference_mode():o=p(torch.rand(2,1,3,240,320,device='cuda'),torch.randn(2,128,device='cuda'))
print('PARITY_HASH',hashlib.sha256(o.cpu().numpy().tobytes()).hexdigest())
'''
env=os.environ.copy();env.update(HDT_EXPERIMENT='baseline',PYTHONDONTWRITEBYTECODE='1')
hashes=[]
for src in ['/mnt/nvme0n1/human_policy_ckpt/isolated_20261020_readiness_geometry/src',str(ROOT/'src')]:
 p=subprocess.run([sys.executable,'-c',CODE,src],env=env,capture_output=True,text=True)
 if p.returncode:raise RuntimeError(p.stdout+p.stderr)
 h=next(x for x in p.stdout.splitlines() if x.startswith('PARITY_HASH '));hashes.append(h)
assert hashes[0]==hashes[1],hashes
print('Baseline old/new bitwise parity verified:',hashes[0],flush=True)
