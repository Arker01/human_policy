#!/usr/bin/env python3
"""Refresh run status/checkpoint hashes without modifying training outputs."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import h5py
import yaml


EXP = Path("/mnt/nvme0n1/human_policy_ckpt/sail_ablation_20261109")
SUPERSEDED = {
    "walk_wo_root_d1": "walk_wo_root_d1_corrected",
    "turn_wo_root_d1": "turn_wo_root_d1_corrected",
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


for path in sorted((EXP / "manifests").glob("*.json")):
    row = json.loads(path.read_text())
    if row.get("reused_existing_run"):
        continue
    name = row["name"]
    if name in SUPERSEDED:
        row["status"] = "superseded_wrong_data_pool"
        row["status_reason"] = (
            "Stopped without deletion: this partial run used the obsolete "
            "D1=SONIC+ScaleBridge definition"
        )
        row["replacement_run"] = SUPERSEDED[name]
        path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
        print(name, row["status"])
        continue
    final_checkpoint = row.get("final_checkpoint")
    checkpoint = Path(
        final_checkpoint["path"] if isinstance(final_checkpoint, dict)
        else row.get("checkpoint", "")
    )
    if not str(checkpoint):
        continue
    pid_path = EXP / f"status/{name}.pid"
    metrics_path = checkpoint.parent / "metrics.jsonl"
    if checkpoint.is_file():
        row["status"] = "complete"
        if isinstance(final_checkpoint, dict):
            row["final_checkpoint"]["sha256"] = sha256(checkpoint)
        else:
            row["checkpoint_sha256"] = sha256(checkpoint)
    elif pid_path.is_file():
        pid = int(pid_path.read_text().strip())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            row["status"] = "failed_or_stopped"
        else:
            row["status"] = "running"
            row["pid"] = pid
    if metrics_path.is_file():
        lines = [line for line in metrics_path.read_text().splitlines() if line]
        if lines:
            row["latest_metrics"] = json.loads(lines[-1])
    model_config_path = Path(row.get("model_config", ""))
    if model_config_path.is_file():
        row["model_config_full"] = yaml.safe_load(model_config_path.read_text())
    training_config_path = checkpoint.parent / "training_config.json"
    if training_config_path.is_file():
        row["resolved_training_config_full"] = json.loads(
            training_config_path.read_text())
    if row.get("variant") == "sonic_native":
        task = "walk" if row.get("task") == "walk_far" else "turn"
        native_dir = EXP / f"data_sources/sonic_native_{task}"
        for episode in row.get("episodes", []):
            generated = native_dir / episode["episode"]
            if generated.is_file():
                with h5py.File(generated, "r") as root:
                    episode["source_hdf5_sha256"] = str(
                        root.attrs["source_hdf5_sha256"])
                    episode["source_parquet_sha256"] = str(
                        root.attrs["source_parquet_sha256"])
    worktree = EXP / "worktree"
    row["git_commit"] = subprocess.check_output(
        ["git", "-c", f"safe.directory={worktree}", "-C", str(worktree),
         "rev-parse", "HEAD"], text=True).strip()
    code = {}
    for relative in ("hdt/main.py", "hdt/data_utils_hdt.py", "hdt/policy.py"):
        source = worktree / relative
        if source.is_file():
            code[relative] = sha256(source)
    row["training_code_sha256"] = code
    path.write_text(json.dumps(row, indent=2, sort_keys=True) + "\n")
    print(name, row["status"])
