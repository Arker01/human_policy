#!/usr/bin/env python3
"""Queue this experiment on an unused GPU, without interrupting existing jobs."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]


def stamp():
    return datetime.now(timezone.utc).isoformat()


def save(value):
    path=ROOT/'status.json.partial'
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    path.replace(ROOT/'status.json')


def verify_inputs(manifest):
    assert json.loads((ROOT/'preflight_report.json').read_text())['status']=='pass'
    def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    assert sha(ROOT/'configs/model.yaml')==manifest['model_sha256']
    assert sha(ROOT/'configs/tripod.json')==manifest['dataset_sha256']
    for name,digest in manifest['source_hashes'].items():
        assert sha(ROOT/'source'/name)==digest,name
    data_root=Path(manifest['data_root'])
    assert sha(data_root/'conversion_report.json')==manifest['conversion_report_sha256']
    conversion=json.loads((data_root/'conversion_report.json').read_text())
    assert conversion['status']=='pass' and conversion['both_hands_valid']
    for record in conversion['files']:
        assert sha(record['path'])==record['sha256'],record['path']


def idle_gpus():
    args=['nvidia-smi','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits']
    rows=list(csv.reader(subprocess.check_output(args,text=True,timeout=20).splitlines()))
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=20)
    occupied={row[0].strip() for row in csv.reader(apps.splitlines()) if row}
    return [dict(index=int(index),uuid=uuid.strip(),memory_mib=int(memory),utilization=int(util))
            for index,uuid,memory,util in rows
            if uuid.strip() not in occupied and int(memory)<512 and int(util)<5]


def worker():
    with (ROOT/'run.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        manifest=json.loads((ROOT/'manifest.json').read_text())
        verify_inputs(manifest)
        checkpoint=Path(manifest['checkpoint_dir'])
        if checkpoint.exists() or (ROOT/'logs/train.log').exists():
            raise RuntimeError('Training output already exists; refusing duplicate run or overwrite')
        status={'status':'waiting_for_gpu','queued_at':stamp(),'supervisor_pid':os.getpid(),
                'checkpoint_dir':str(checkpoint),'command':manifest['command'],
                'reason':'All existing GPU training jobs are left running; start only on an unused GPU.'}
        save(status)
        candidate=None
        while True:
            idle=idle_gpus()
            confirmed=next((gpu for gpu in idle if gpu['uuid']==candidate),None)
            if confirmed is not None:
                gpu=confirmed
                break
            candidate=idle[0]['uuid'] if idle else None
            status.update(last_gpu_check_at=stamp(),idle_candidates=idle)
            save(status)
            time.sleep(20)
        verify_inputs(manifest)
        checkpoint.parent.mkdir(parents=True,exist_ok=True)
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'],CUDA_DEVICE_ORDER='PCI_BUS_ID',
            PYTHONPATH=os.pathsep.join([str(ROOT/'source'),str(ROOT/'source/hdt'),str(ROOT/'source/hdt/detr'),str(REPO)]),
            OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
            TOKENIZERS_PARALLELISM='false',PYTHONDONTWRITEBYTECODE='1',
            MPLBACKEND='Agg',MPLCONFIGDIR=str(ROOT/'logs/matplotlib'),
            WANDB_MODE='disabled',HAT_CAMERA_SCALE='1.0')
        with (ROOT/'logs/train.log').open('xb',buffering=0) as log:
            process=subprocess.Popen(manifest['command'],cwd=REPO,env=env,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        status.update(status='running',started_at=stamp(),gpu=gpu,training_pid=process.pid)
        save(status)
        code=process.wait()
        status.update(status='completed' if code==0 else 'failed',exit_code=code,finished_at=stamp())
        save(status)
        raise SystemExit(code)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['queue','worker','status'])
    args=parser.parse_args()
    if args.action=='status':
        path=ROOT/'status.json'
        value=json.loads(path.read_text()) if path.exists() else {'status':'prepared'}
        pid=value.get('supervisor_pid')
        value['supervisor_alive']=pid is not None and Path(f'/proc/{pid}').exists()
        print(json.dumps(value,ensure_ascii=False,indent=2))
    elif args.action=='worker':
        try:
            worker()
        except BlockingIOError:
            raise SystemExit('An existing supervisor holds run.lock; no duplicate queued')
        except Exception as error:
            save({'status':'failed','error':repr(error),'finished_at':stamp(),'supervisor_pid':os.getpid()})
            raise
    else:
        manifest=json.loads((ROOT/'manifest.json').read_text())
        verify_inputs(manifest)
        if (ROOT/'status.json').exists():
            previous=json.loads((ROOT/'status.json').read_text())
            if previous.get('status') in ['waiting_for_gpu','running','completed']:
                raise SystemExit(f"Already {previous['status']}; no duplicate queued")
        with (ROOT/'logs/queue.log').open('ab',buffering=0) as log:
            process=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'worker'],
                cwd=REPO,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print(f'Queued supervisor PID {process.pid}; status: {ROOT / "status.json"}')


if __name__=='__main__':
    main()
