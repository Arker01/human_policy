"""Launch exactly one isolated two-GPU run; never terminate existing jobs."""
import runtime
import datetime,json,os,subprocess


def main():
    output=runtime.ROOT/'checkpoints/basic_20261021'
    log_path=runtime.ROOT/'logs/basic_20261021.log'
    metadata=runtime.ROOT/'logs/basic_20261021_launch.json'
    if metadata.exists() or (output.exists() and any(output.iterdir())):
        raise RuntimeError('This named run already exists; inspect it instead of launching a duplicate')
    usage=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used','--format=csv,noheader,nounits'],text=True)
    for row in usage.splitlines():
        index,used=[int(x.strip()) for x in row.split(',')]
        if index in (6,7) and used>100:
            raise RuntimeError(f'GPU {index} is in use ({used} MiB); leave that process running')
    python='/home/aigc/miniconda/envs/human_policy/bin/python'
    command=[python,'-m','torch.distributed.run','--nproc_per_node=2','--master_port=29868',
        str(runtime.ROOT/'src/train.py'),'--steps','100000','--batch-size','32',
        '--lr','1e-5','--save-every','1000','--output',str(output)]
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='6,7',OMP_NUM_THREADS='4',
        MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false')
    log_path.parent.mkdir(exist_ok=True)
    with log_path.open('ab',buffering=0) as log:
        process=subprocess.Popen(command,cwd=runtime.REPO,env=env,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    info=dict(pid=process.pid,command=command,gpus=[6,7],log=str(log_path),checkpoint_dir=str(output),
        started_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        baseline=str(runtime.BASE),calibration_manifest=str(runtime.ROOT/'calibration_sidecars_20261021/manifest.json'))
    metadata.write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))


if __name__=='__main__':main()
