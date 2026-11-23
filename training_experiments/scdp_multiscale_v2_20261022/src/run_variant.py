"""One independent v2 training run, followed by full original-protocol evaluation."""
import runtime
import json,os,subprocess,sys,time


def main():
    name=runtime.BASE_VARIANT
    job=next(j for j in runtime.EXPERIMENT['jobs'] if j['name']==name)
    command=[sys.executable,'-u']
    if len(job['gpus'])==2:
        command=[sys.executable,'-u','-m','torch.distributed.run','--nproc_per_node=2','--master_port=29878']
    output=runtime.ROOT/'checkpoints'/name
    command += [str(runtime.ROOT/'src/train.py'),'--batch-size',str(job['batch_size_per_process']),
                '--output',str(output)]
    process=subprocess.Popen(command,cwd=runtime.REPO)
    status=dict(phase='training',train_pid=process.pid,command=command,started_at=time.time())
    path=runtime.ROOT/f'logs/{name}_status.json';path.write_text(json.dumps(status,indent=2))
    code=process.wait()
    if code:
        status.update(phase='failed',exit_code=code);path.write_text(json.dumps(status,indent=2))
        raise SystemExit(code)
    status.update(phase='evaluating');path.write_text(json.dumps(status,indent=2))
    step=runtime.EXPERIMENT['training']['steps']
    evaluate=[sys.executable,'-u',str(runtime.ROOT/'src/evaluate_protocol.py'),
        '--checkpoint',str(output/f'step_{step}.pt'),'--output',str(runtime.ROOT/f'results/{name}/final_query0.json')]
    reference=runtime.BASE_CONFIG['v1_reference_checkpoint']
    if int(reference.rsplit('_',1)[-1].split('.')[0])==step:evaluate += ['--reference',reference]
    pairs=runtime.EXPERIMENT['evaluation']['bottle_pairs_manifest']
    if pairs:evaluate += ['--pairs',pairs]
    subprocess.run(evaluate,cwd=runtime.REPO,check=True)
    status.update(phase='complete',finished_at=time.time());path.write_text(json.dumps(status,indent=2))


if __name__=='__main__':main()
