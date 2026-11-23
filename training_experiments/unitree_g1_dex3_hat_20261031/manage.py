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
CHECKPOINT_ROOT = Path('/mnt/nvme0n1/human_policy_ckpt/unitree_g1_dex3_hat_20261031')
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
    import yaml
    if (ROOT / 'manifest.json').exists():
        raise RuntimeError('Prepared experiment already exists; inspect manifest instead of overwriting.')
    records_path = REPO / 'experiments/bottle0930_16runs_20261004/jobs.json'
    records = json.loads(records_path.read_text())
    baseline = max((j for j in records if j['status'] == 'completed'), key=lambda j: j['finished_at'])
    assert baseline['name'] == 'b0930_20261004_robotonly_nostate_vjepa_w0_lagvis'
    baseline_command = baseline['command']
    model_source = Path(baseline_command[baseline_command.index('--model_cfg_path') + 1])
    model = yaml.safe_load(model_source.read_text())
    assert model['common']['camera_names'] == ['top']
    assert model['common']['action_dim'] == model['common']['state_dim'] == 128
    assert model['model']['backbone'] == 'vjepa2_1_vitb'
    assert model['model']['future_dino']['weight'] == 0
    assert model['model']['zero_state_dims'] == '98:128'
    compatibility_path = REPO / 'data/unitree_g1_dex3/training_compatibility_report.json'
    compatibility = json.loads(compatibility_path.read_text())
    assert compatibility['status'] == 'pass'
    for folder in ['configs', 'logs', 'status', 'source']:
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(model_source, ROOT / 'configs/model.yaml')
    shutil.copyfile(compatibility_path, ROOT / 'configs/training_compatibility_report.json')
    def ignore_snapshot(directory, names):
        ignored = set(shutil.ignore_patterns('__pycache__', '*.pyc', '*_ckpt')(directory, names))
        return ignored | {name for name in names if (Path(directory) / name).is_symlink()}
    shutil.copytree(REPO / 'hdt', ROOT / 'source/hdt', ignore=ignore_snapshot, dirs_exist_ok=True)
    source_hashes = {str(p.relative_to(ROOT / 'source')): digest(p)
                     for p in sorted((ROOT / 'source').rglob('*')) if p.is_file()}
    jobs = {}
    for task, gpu in zip(TASKS, [1, 2, 3]):
        dataset_source = REPO / f'configs/datasets/unitree_g1_dex3_{task}.json'
        dataset_snapshot = ROOT / f'configs/{task}.json'
        shutil.copyfile(dataset_source, dataset_snapshot)
        config = json.loads(dataset_snapshot.read_text())
        counts = {split: sum(len(entry['file_list']) for entry in config[split]) for split in ['train', 'val']}
        for split in ['train', 'val']:
            for entry in config[split]:
                assert all((Path(entry['dataset_path']) / name).is_file() for name in entry['file_list'])
        command = baseline_command.copy()
        command[0] = PYTHON
        command[2] = str(ROOT / 'source/hdt/main.py')
        replacements = {'--exptid': str(CHECKPOINT_ROOT / task),
                        '--dataset_json_path': str(dataset_snapshot),
                        '--model_cfg_path': str(ROOT / 'configs/model.yaml')}
        for key, value in replacements.items():
            command[command.index(key) + 1] = value
        # All optimizer/model/augmentation/sampling arguments must match exactly.
        restored = command.copy()
        restored[2] = baseline_command[2]
        for key in replacements:
            restored[restored.index(key) + 1] = baseline_command[baseline_command.index(key) + 1]
        assert restored == baseline_command
        jobs[task] = {'task': task, 'gpu_index': gpu, 'command': command,
                      'dataset_config': str(dataset_snapshot), 'dataset_sha256': digest(dataset_snapshot),
                      'checkpoint_dir': str(CHECKPOINT_ROOT / (task + '_ckpt')),
                      'log': str(ROOT / f'logs/{task}.log'),
                      'status_file': str(ROOT / f'status/{task}.json'),
                      'train_episodes': counts['train'], 'val_episodes': counts['val'],
                      'status': 'prepared'}
    weights = [Path('/home/aigc/.cache/torch/hub/checkpoints/vjepa2_1_vitb_dist_vitG_384.pt'),
               Path('/home/aigc/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth')]
    for p in weights:
        assert p.is_file(), p
    patch = subprocess.check_output(['git', 'diff', '--', 'hdt/main.py'], cwd=REPO, text=True)
    (ROOT / 'training_loop_fix.patch').write_text(patch)
    manifest = {'status': 'prepared', 'prepared_at': stamp(), 'baseline_record': str(records_path),
                'baseline': baseline, 'user_selected_configuration': 'standard HAT',
                'checkpoint_root': str(CHECKPOINT_ROOT), 'source_hashes': source_hashes,
                'model_sha256': digest(ROOT / 'configs/model.yaml'),
                'pretrained_weights': {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in weights},
                'same_training_arguments_verified': True,
                'training_loop_fix': 'Keep validation val_data separate from pending training data; regression test passed.',
                'initialization': 'same as baseline: pretrained visual encoder, no policy checkpoint finetune',
                'jobs': jobs}
    write_json(ROOT / 'manifest.json', manifest)
    print(json.dumps({'status': 'prepared', 'baseline': baseline['name'], 'jobs': jobs}, indent=2), flush=True)


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
