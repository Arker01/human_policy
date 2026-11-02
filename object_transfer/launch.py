#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess


PYTHON = "/home/aigc/miniconda/envs/human_policy/bin/python"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("job")
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    jobs = json.loads((args.root / "jobs.json").read_text())
    source_root = Path(__file__).resolve().parent
    job = dict(jobs[args.job])
    steps = 3 if args.smoke else int(job["steps"])
    run_name = args.job + (".smoke" if args.smoke else "")
    exptid = args.root / "checkpoints" / run_name
    checkpoint_dir = Path(str(exptid) + "_ckpt")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (args.root / "logs").mkdir(parents=True, exist_ok=True)
    (args.root / "status").mkdir(parents=True, exist_ok=True)

    command = [
        PYTHON, "-u", "hdt/main.py",
        "--batch_size", "64",
        "--num_epochs", str(steps),
        "--lr", "1e-5",
        "--chunk_size", "100",
        "--seed", "0",
        "--exptid", str(exptid),
        "--dataset_json_path", job["dataset_json"],
        "--model_cfg_path", job["model_config"],
        "--base_dir", "/mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat",
        "--cond_mask_prob", "0.0",
        "--no_wandb",
        "--human_slow_down_factor", "1",
        "--use_future_dino_head",
        "--future_dino_weight", "0.0",
        "--future_dino_warmup_steps", "0",
        "--future_dino_horizon", "45",
        "--future_dino_ablation", "none",
        "--robot_target_fps", "30",
        "--action_representation", "original",
        "--lag_aug",
        "--zero_state_dims", "98:128",
        "--load_pretrained_path", job["pretrained"],
    ]
    env = os.environ.copy()
    env.update({
        "CUDA_VISIBLE_DEVICES": str(args.gpu),
        "HDT_EXPERIMENT": "baseline",
        "PYTHONPATH": (
            f"{source_root.parent}:{source_root / 'src'}:"
            f"{source_root / 'src/hdt'}"
        ),
        "OMP_NUM_THREADS": "4",
        "MKL_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    })
    manifest = {
        **job,
        "name": run_name,
        "gpu": str(args.gpu),
        "command": command,
        "checkpoint_dir": str(checkpoint_dir),
        "source_root": str(source_root / "src"),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
    }
    manifest_path = args.root / "status" / f"{run_name}.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    with (args.root / "logs" / f"{run_name}.log").open("ab", buffering=0) as log:
        process = subprocess.Popen(
            command,
            cwd=source_root / "src",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    manifest["training_pid"] = process.pid
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"job": run_name, "gpu": args.gpu, "pid": process.pid,
                      "log": str(args.root / 'logs' / f'{run_name}.log')}))


if __name__ == "__main__":
    main()
