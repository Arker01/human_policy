"""Launch the two selected SCDP bases, one GPU and global batch 64 each."""
import runtime
import datetime,hashlib,json,os,subprocess
from pathlib import Path


def sha256(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def main():
    jobs=[('geometry',6),('geometry_no_left_wrist',7)]
    configs=json.loads((runtime.ROOT/'configs/bases.json').read_text())
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    memory={int(a.strip()):int(b.strip()) for a,b in (line.split(',') for line in usage.splitlines())}
    for name,gpu in jobs:
        if memory[gpu]>100:raise RuntimeError(f'GPU {gpu} is occupied; do not interrupt it')
        metadata=runtime.ROOT/f'logs/{name}_launch.json'
        output=runtime.ROOT/f'checkpoints/{name}'
        if metadata.exists() or (output.exists() and any(output.iterdir())):
            raise RuntimeError(f'{name} already exists; inspect it instead of duplicating training')
        actual=sha256(Path(configs[name]['checkpoint_dir'])/'policy_last.ckpt')
        if actual!=configs[name]['weights_sha256']:raise RuntimeError(f'{name}: wrong original checkpoint hash')
    python='/home/aigc/miniconda/envs/human_policy/bin/python'
    for name,gpu in jobs:
        log_path=runtime.ROOT/f'logs/{name}.log';output=runtime.ROOT/f'checkpoints/{name}'
        command=[python,'-u',str(runtime.ROOT/'src/train.py'),'--steps','100000',
            '--batch-size','64','--lr','1e-5','--save-every','1000','--workers','4','--output',str(output)]
        env=os.environ.copy();env.update(SCDP_BASE_VARIANT=name,CUDA_VISIBLE_DEVICES=str(gpu),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',HDT_EXPERIMENT='geometry')
        with log_path.open('ab',buffering=0) as log:
            process=subprocess.Popen(command,cwd=runtime.REPO,env=env,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        source=Path(configs[name]['source_root'])
        info=dict(name=name,pid=process.pid,gpu=gpu,global_batch_size=64,steps=100000,
            command=command,log=str(log_path),checkpoint_dir=str(output),
            base_config=configs[name],calibration_manifest=str(runtime.CALIBRATION_MANIFEST),
            source_sha256={f:sha256(source/f) for f in ['hdt/policy.py','hdt/detr/models/detr_vae.py',
                'hdt/data_utils_hdt.py','hdt/auxiliary_tasks.py']},
            started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (runtime.ROOT/f'logs/{name}_launch.json').write_text(json.dumps(info,indent=2))
        print(json.dumps(info,indent=2),flush=True)


if __name__=='__main__':main()
