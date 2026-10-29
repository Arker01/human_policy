"""Three v2 runs sharing only the requested GPU 6/7; never alter v1 outputs."""
import runtime
import datetime,hashlib,json,os,subprocess,sys
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def main():
    root=runtime.ROOT
    bases=json.loads((root/'configs/bases.json').read_text())
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in usage.splitlines())}
    for gpu in (6,7):
        if memory[gpu]>100:raise RuntimeError(f'GPU {gpu} has another active job; inspect first')
    for job in runtime.EXPERIMENT['jobs']:
        name=job['name'];config=bases[name]
        if (root/f'logs/{name}_launch.json').exists() or (root/f'checkpoints/{name}').exists():
            raise RuntimeError(f'Run already exists: {name}')
        if digest(Path(config['checkpoint_dir'])/'policy_last.ckpt')!=config['weights_sha256']:
            raise RuntimeError(f'Wrong base: {name}')
    for job in runtime.EXPERIMENT['jobs']:
        name=job['name'];env=os.environ.copy()
        env.update(SCDP_BASE_VARIANT=name,CUDA_VISIBLE_DEVICES=','.join(map(str,job['gpus'])),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
        command=[sys.executable,'-u',str(root/'src/run_variant.py')]
        with (root/f'logs/{name}.log').open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=runtime.REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        metadata=dict(name=name,runner_pid=process.pid,gpus=job['gpus'],global_batch_size=64,
            command=command,base=bases[name],version='v2_five_scale',
            started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            source_sha256={str(p.relative_to(root)):digest(p) for p in sorted((root/'src').glob('*.py'))})
        (root/f'logs/{name}_launch.json').write_text(json.dumps(metadata,indent=2))
        print(json.dumps(metadata),flush=True)


if __name__=='__main__':main()
