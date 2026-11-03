#!/usr/bin/env python3
"""Precompute previous-plan chunks from a frozen random-OMEGA-only teacher."""

import argparse
import copy
import hashlib
import json
import os
import pickle
import time

import numpy as np
import torch
import yaml

from data_utils_hdt import load_data
from hat_conditioning import episode_cache_key
from main import make_visual_encoder
from policy import ACTPolicy


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def flush_batch(policy, dataset, records, outputs):
    if not records:
        return
    images = torch.stack([r[2] for r in records]).cuda(non_blocking=True)
    states = torch.stack([r[3] for r in records]).cuda(non_blocking=True)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = policy(images, states, conditioning_dict={}).float().cpu().numpy()
    for (episode_i, logical_t, _, _), normalized in zip(records, pred):
        embodiment = dataset.source_embodiment_list[episode_i]
        stats = dataset.norm_stats[embodiment]
        raw = normalized * (stats["action_std"][None] + 1e-6) + stats["action_mean"][None]
        outputs[episode_i].append((logical_t, raw.astype(np.float32)))
    records.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-ckpt", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    ckpt_root = os.path.abspath(args.baseline_ckpt)
    cfg = json.load(open(os.path.join(ckpt_root, "training_config.json")))
    run_args = cfg["args"]
    policy_cfg = copy.deepcopy(cfg["config"]["policy_config"])
    # The baseline's zero-weight Future-DINO head is irrelevant to action inference
    # and would load a second frozen teacher.  Removing only that silent head leaves
    # the actual V-JEPA action trunk and all loaded action weights unchanged.
    policy_cfg.pop("future_dino_config", None)
    _, preprocessor = make_visual_encoder("ACT", policy_cfg)
    with open(os.path.join(ckpt_root, "model.yaml"), "r") as f:
        model_cfg = yaml.safe_load(f)
    train_loader, _, _ = load_data(
        run_args["base_dir"], model_cfg["data"], run_args["dataset_json_path"],
        policy_cfg["camera_names"], policy_cfg["chunk_size"], 1, 1, preprocessor,
        0.0, run_args.get("human_slow_down_factor", 4),
        [run_args["processed_dir"]] if run_args.get("processed_dir") else None,
        dirty_start_check_frames=run_args.get("dirty_start_check_frames", 20),
        dirty_start_jump_threshold_m=run_args.get("dirty_start_jump_threshold_m", 0.3),
        dirty_start_settle_frames=run_args.get("dirty_start_settle_frames", 0),
        action_representation=run_args.get("action_representation", "original"),
        robot_target_fps=run_args.get("robot_target_fps", 30.0))
    dataset = train_loader.dataset
    dataset.train = False
    dataset.lag_aug = None
    dataset.visual_aug = None

    policy = ACTPolicy(policy_cfg).cuda().eval()
    latest = sorted(
        [d for d in os.listdir(ckpt_root) if d.startswith("policy_iter_")],
        key=lambda x: int(x.split("_")[2]))[-1]
    weights = os.path.join(ckpt_root, latest, "pytorch_model.bin")
    state = torch.load(weights, map_location="cuda", weights_only=True)
    incompatible = policy.load_state_dict(state, strict=False)
    if incompatible.missing_keys:
        raise RuntimeError(f"teacher checkpoint missing action-model keys: {incompatible.missing_keys[:8]}")

    os.makedirs(args.output_dir, exist_ok=True)
    outputs = [[] for _ in dataset.dataset_paths]
    batch = []
    started = time.time()
    total = sum((int(n) + args.stride - 1) // args.stride
                for n in dataset.episode_len_list)
    done = 0
    for episode_i, episode_len in enumerate(dataset.episode_len_list):
        for logical_t in range(0, int(episode_len), int(args.stride)):
            image, qpos, _, _, _ = dataset.read_one(episode_i, logical_t)
            batch.append((episode_i, logical_t, image, qpos))
            if len(batch) >= args.batch_size:
                flush_batch(policy, dataset, batch, outputs)
                done += args.batch_size
                if done % (args.batch_size * 20) == 0:
                    print(f"teacher cache {done}/{total} elapsed={time.time()-started:.1f}s", flush=True)
    if batch:
        n = len(batch)
        flush_batch(policy, dataset, batch, outputs)
        done += n

    episodes = {}
    for episode_i, rows in enumerate(outputs):
        path = os.path.abspath(dataset.dataset_paths[episode_i])
        key = episode_cache_key(path)
        times = np.asarray([r[0] for r in rows], dtype=np.int32)
        chunks = np.stack([r[1] for r in rows]).astype(np.float32)
        valid_lengths = np.minimum(100, int(dataset.episode_len_list[episode_i]) - times).astype(np.int16)
        filename = f"{key}.npz"
        np.savez(os.path.join(args.output_dir, filename), times=times, chunks=chunks,
                 valid_lengths=valid_lengths)
        episodes[key] = {"source_path": path, "file": filename, "num_predictions": len(times)}

    manifest = {
        "format_version": 1,
        "split": "train",
        "action_dim": 128,
        "chunk_size": 100,
        "cache_stride_frames": int(args.stride),
        "augment_seed": 0,
        "teacher_checkpoint": weights,
        "teacher_checkpoint_sha256": file_sha256(weights),
        "dataset_manifest": run_args["dataset_json_path"],
        "action_stats_source": os.path.join(ckpt_root, "dataset_stats.pkl"),
        "episodes": episodes,
    }
    tmp = os.path.join(args.output_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(args.output_dir, "manifest.json"))
    print(f"teacher cache complete: {done} predictions, {len(episodes)} episodes, "
          f"{time.time()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
