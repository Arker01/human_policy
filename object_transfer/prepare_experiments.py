#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import yaml


TASKS = ("wipe_table", "classify_gadgets")
OMEGA_ORIGINAL = Path("/mnt/nvme0n1/DATASETS/omega_HOME_4tasks_20261025/hat")
HUMAN_ALIGNED = Path("/mnt/nvme1n1/human_policy_derived/human_geometry_aligned_20261102")
DERIVED = Path("/mnt/nvme1n1/human_policy_derived/bg_variants_batch_20261029")
def read(path: Path) -> dict:
    return json.loads(path.read_text())


def replace_dataset_path(entries: list[dict], *, human: Path | None, robot: Path) -> list[dict]:
    result = []
    for entry in entries:
        updated = copy.deepcopy(entry)
        path = str(entry["dataset_path"])
        is_human = (
            str(entry.get("type", "")).lower() == "human"
            or "/human/" in path
            or "1029_" in path
            or "human_geometry" in path
        )
        if is_human:
            if human is None:
                continue
            updated["dataset_path"] = str(human)
        else:
            updated["dataset_path"] = str(robot)
        result.append(updated)
    return result


def assert_episode_split(config: dict) -> None:
    train = {name for entry in config["train"] for name in entry.get("file_list", [])}
    val = {name for entry in config["val"] for name in entry.get("file_list", [])}
    overlap = train & val
    if overlap:
        raise ValueError(f"episode leakage across train/val: {sorted(overlap)[:10]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-config-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    args = parser.parse_args()
    config_dir = args.output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    template = yaml.safe_load(args.template.read_text())
    jobs = {}

    for task in TASKS:
        sources = {
            "omega_original": read(args.source_config_root / f"{task}.random_omega_only.dataset.json"),
            "omega_human_original": read(args.source_config_root / f"{task}.orig_mix.dataset.json"),
            "omega_human_random": read(args.source_config_root / f"{task}.random_mix.dataset.json"),
        }
        for variant, source in sources.items():
            mixed = "human" in variant
            random_background = variant.endswith("random")
            robot_path = (DERIVED / "omega" / task / "randomized") \
                if random_background else (OMEGA_ORIGINAL / task)
            dataset = {
                split: replace_dataset_path(
                    source[split],
                    human=(HUMAN_ALIGNED / task) if mixed else None,
                    robot=robot_path,
                )
                for split in ("train", "val")
            }
            assert_episode_split(dataset)
            dataset_path = config_dir / f"{task}.{variant}.dataset.json"
            dataset_path.write_text(json.dumps(dataset, indent=2))

            model = copy.deepcopy(template)
            obj = model["model"]["object_transfer"]
            obj["task"] = task
            obj["human_image_override_root"] = (
                str(DERIVED / "human" / task / "randomized")
                if random_background else None
            )
            model_path = config_dir / f"{task}.{variant}.yaml"
            model_path.write_text(yaml.safe_dump(model, sort_keys=False))

            # Train the action model from random initialization.  Match the original
            # 100k Omega training budget.  With 50/50 embodiment sampling, a mixed
            # run needs 200k updates for about 100k robot-equivalent updates.
            steps = 200_000 if mixed else 100_000
            jobs[f"{task}.{variant}"] = {
                "task": task,
                "variant": variant,
                "mixed_human": mixed,
                "random_background": random_background,
                "steps": steps,
                "robot_equivalent_steps": 100_000,
                "initialization": "scratch_action_model",
                "experiment_variant": "geometry",
                "state_ablation": "noleftwrist_and_qpos_98_128",
                "dataset_json": str(dataset_path),
                "model_config": str(model_path),
                "pretrained": None,
            }

    (args.output_root / "jobs.json").write_text(json.dumps(jobs, indent=2))
    print(json.dumps(jobs, indent=2))


if __name__ == "__main__":
    main()

