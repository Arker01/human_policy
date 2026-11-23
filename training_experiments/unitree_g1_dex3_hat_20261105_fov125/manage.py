#!/usr/bin/env python3
"""Prepare, launch, and inspect three independent standard HAT Dex3 runs."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
PYTHON = '/home/aigc/miniconda/envs/human_policy/bin/python'
CHECKPOINT_ROOT = Path('/mnt/nvme0n1/human_policy_ckpt/unitree_g1_dex3_hat_20261105_fov125')
TASKS = ['block_stacking', 'pick_apple', 'pouring']


def stamp():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def prepare():
    from prepare_runs import prepare_runs
    prepare_runs(ROOT, REPO, CHECKPOINT_ROOT, PYTHON, TASKS, stamp, write_json, digest)


def gpu_state():
    output = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True, timeout=20)
    result = {}
    for row in csv.reader(output.splitlines()):
        index, uuid, memory, utilization = [v.strip() for v in row]
        result[int(index)] = {'uuid': uuid, 'memory_mib': int(memory), 'utilization': int(utilization)}
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=20)
    occupied = {row[0].strip() for row in csv.reader(apps.splitlines()) if row}
    return result, occupied


def launch():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    if manifest['status'] != 'prepared' or (ROOT / 'launch.lock').exists():
        raise RuntimeError('Experiment already launched or not prepared; refusing duplicate training.')
    for report in ['data_preflight.json', 'left_hand_verification.json']:
        assert json.loads((ROOT / report).read_text())['status'] == 'pass', report
    gpus, occupied = gpu_state()
    for job in manifest['jobs'].values():
        gpu = gpus[job['gpu_index']]
        if gpu['uuid'] in occupied or gpu['memory_mib'] > 128 or gpu['utilization'] > 5:
            raise RuntimeError(f"GPU {job['gpu_index']} is occupied: {gpu}; no jobs launched.")
        if Path(job['checkpoint_dir']).exists() or Path(job['log']).exists():
            raise RuntimeError(f"Output already exists: {job['task']}")
        assert digest(job['dataset_config']) == job['dataset_sha256']
        job['gpu_uuid'] = gpu['uuid']
    assert digest(ROOT / 'configs/model.yaml') == manifest['model_sha256']
    for relative, sha in manifest['source_hashes'].items():
        assert digest(ROOT / 'source' / relative) == sha, relative
    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=False)
    (ROOT / 'checkpoints').symlink_to(CHECKPOINT_ROOT, target_is_directory=True)
    descriptor = os.open(ROOT / 'launch.lock', os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(descriptor)
    manifest.update(status='launching', launched_at=stamp(), gpu_preflight=gpus)
    write_json(ROOT / 'manifest.json', manifest)
    for task, job in manifest['jobs'].items():
        env = os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES=job['gpu_uuid'], CUDA_DEVICE_ORDER='PCI_BUS_ID',
                   OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                   TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1',
                   MPLBACKEND='Agg', MPLCONFIGDIR=str(ROOT / f'logs/matplotlib_{task}'),
                   WANDB_MODE='disabled',
                   PYTHONPATH=os.pathsep.join([str(ROOT / 'source'), str(ROOT / 'source/hdt'),
                                             str(ROOT / 'source/hdt/detr'), str(REPO)]))
        supervisor_command = [PYTHON, '-u', str(Path(__file__).resolve()), 'worker', '--task', task]
        with Path(job['log']).open('xb', buffering=0) as log:
            process = subprocess.Popen(supervisor_command, cwd=REPO, env=env,
                                       stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        job.update(status='launched', supervisor_pid=process.pid)
        write_json(ROOT / 'manifest.json', manifest)
        print(f"LAUNCHED {task} GPU {job['gpu_index']} supervisor={process.pid} log={job['log']}", flush=True)
    manifest['status'] = 'launched'
    write_json(ROOT / 'manifest.json', manifest)


def worker(task):
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    job = manifest['jobs'][task]
    assert os.environ['CUDA_VISIBLE_DEVICES'] == job['gpu_uuid']
    status = {'task': task, 'status': 'starting', 'started_at': stamp(),
              'supervisor_pid': os.getpid(), 'gpu_index': job['gpu_index'], 'gpu_uuid': job['gpu_uuid'],
              'command': job['command'], 'checkpoint_dir': job['checkpoint_dir'], 'log': job['log']}
    write_json(Path(job['status_file']), status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    process = subprocess.Popen(job['command'], cwd=REPO, stdin=subprocess.DEVNULL)
    status.update(status='running', training_pid=process.pid)
    write_json(Path(job['status_file']), status)
    code = process.wait()
    status.update(status='completed' if code == 0 else 'failed', exit_code=code, finished_at=stamp())
    write_json(Path(job['status_file']), status)
    print(json.dumps(status, ensure_ascii=False), flush=True)
    raise SystemExit(code)


def status():
    manifest = json.loads((ROOT / 'manifest.json').read_text())
    summaries = []
    for job in manifest['jobs'].values():
        path = Path(job['status_file'])
        summary = json.loads(path.read_text()) if path.exists() else {'task': job['task'], 'status': job['status']}
        log = Path(job['log']).read_text(errors='replace') if Path(job['log']).exists() else ''
        steps = re.findall(r'(\d+)/100000', log)
        metrics = Path(job['checkpoint_dir']) / 'metrics.csv'
        csv_steps = []
        finite = True
        if metrics.exists():
            for row in csv.reader(metrics.open()):
                try:
                    csv_steps.append(int(row[0]))
                except (ValueError, IndexError):
                    continue
                finite = finite and all(math.isfinite(float(v)) for v in row[1:] if v)
        summary['step'] = max([0] + [int(v) for v in steps] + csv_steps)
        summary['metrics_finite'] = finite
        summary['initial_val_losses'] = [float(v) for v in re.findall(r'Val loss:\s+([-+\d.eE]+)', log)]
        summary['has_traceback'] = 'Traceback (most recent call last)' in log
        summaries.append(summary)
    output = {'checked_at': stamp(), 'jobs': summaries}
    write_json(ROOT / 'status_summary.json', output)
    print(json.dumps(output, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'launch', 'worker', 'status'])
    parser.add_argument('--task', choices=TASKS)
    args = parser.parse_args()
    if args.action == 'worker':
        if args.task is None:
            parser.error('worker requires --task')
        worker(args.task)
    else:
        globals()[args.action]()
