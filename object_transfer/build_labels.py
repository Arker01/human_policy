#!/usr/bin/env python3
"""Build sparse, auditable object-point labels from saved detector refresh events.

This first pass intentionally does not use the old foreground mask: that mask is a
union of the task objects and the human/robot arm.  Only detector boxes whose prompt
names a task object are accepted.  The output is a separate NPZ per episode and never
modifies the source HDF5.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


WIPE_WORDS = ("cleaning cloth", "cleaning rag", "cloth", "rag")
CLASSIFY_WORDS = ("purple fruit", "orange", "banana", "roll of tape", "stapler")
HAND_WORDS = ("human hand", "human arm", "robot hand", "robot arm", "robot gripper")


def _matches(text: str, vocabulary: tuple[str, ...]) -> bool:
    text = text.lower().strip()
    return any(word in text for word in vocabulary)


def _box(record: dict, width: int, height: int, minimum_score: float) -> tuple[np.ndarray, float] | None:
    score = float(record.get("score", 0.0))
    values = np.asarray(record.get("box", ()), dtype=np.float32)
    if score < minimum_score or values.shape != (4,) or not np.isfinite(values).all():
        return None
    x0, y0, x1, y1 = values
    x0, x1 = sorted((float(x0), float(x1)))
    y0, y1 = sorted((float(y0), float(y1)))
    x0, x1 = np.clip((x0, x1), 0, width)
    y0, y1 = np.clip((y0, y1), 0, height)
    area_fraction = (x1 - x0) * (y1 - y0) / max(width * height, 1)
    if x1 - x0 < 3 or y1 - y0 < 3 or area_fraction > 0.35:
        return None
    return np.asarray((x0, y0, x1, y1), dtype=np.float32), score


def _center(box: np.ndarray) -> np.ndarray:
    return (box[:2] + box[2:]) * 0.5


def _edge_distance(a: np.ndarray, b: np.ndarray, diagonal: float) -> float:
    dx = max(float(a[0] - b[2]), float(b[0] - a[2]), 0.0)
    dy = max(float(a[1] - b[3]), float(b[1] - a[3]), 0.0)
    return math.hypot(dx, dy) / max(diagonal, 1.0)


def select_wipe(detections: list[dict], width: int, height: int):
    accepted = []
    for record in detections:
        text = f"{record.get('prompt', '')} {record.get('label', '')}"
        if not _matches(text, WIPE_WORDS):
            continue
        parsed = _box(record, width, height, 0.35)
        if parsed is not None:
            accepted.append(parsed)
    return max(accepted, key=lambda item: item[1]) if accepted else None


def select_classify(detections: list[dict], width: int, height: int):
    objects, hands = [], []
    for record in detections:
        text = f"{record.get('prompt', '')} {record.get('label', '')}"
        parsed = _box(record, width, height, 0.35)
        if parsed is None:
            continue
        if _matches(text, CLASSIFY_WORDS):
            objects.append(parsed)
        elif _matches(text, HAND_WORDS):
            hands.append(parsed)
    if not objects or not hands:
        return None
    diagonal = math.hypot(width, height)
    ranked = []
    for object_box, object_score in objects:
        distance = min(_edge_distance(object_box, hand_box, diagonal) for hand_box, _ in hands)
        ranked.append((distance, -object_score, object_box, object_score))
    distance, _, object_box, object_score = min(ranked, key=lambda item: (item[0], item[1]))
    # Only contact/near-contact frames are called active-object labels.  A distant
    # nearest object is not evidence of the next object the demonstrator will choose.
    if distance > 0.12:
        return None
    return object_box, object_score * max(0.25, 1.0 - distance / 0.12)


def select_hand(
    detections: list[dict],
    width: int,
    height: int,
    object_box: np.ndarray,
):
    accepted = []
    diagonal = math.hypot(width, height)
    for record in detections:
        text = f"{record.get('prompt', '')} {record.get('label', '')}"
        if not _matches(text, HAND_WORDS):
            continue
        parsed = _box(record, width, height, 0.35)
        if parsed is not None:
            hand_box, score = parsed
            accepted.append((
                _edge_distance(object_box, hand_box, diagonal),
                -score,
                hand_box,
                score,
            ))
    if not accepted:
        return None
    _, _, hand_box, score = min(accepted, key=lambda item: (item[0], item[1]))
    return hand_box, score


def build_one(path: Path, output_root: Path, minimum_quality: float) -> dict:
    metadata = json.loads(path.read_text())
    task = str(metadata["task"])
    domain = "human" if str(metadata["domain"]).lower() == "human" else "robot"
    frames = int(metadata["frames"])
    height, width = (int(v) for v in metadata["image_shape"][:2])
    uv = np.zeros((frames, 2), dtype=np.float32)
    target_valid = np.zeros(frames, dtype=np.uint8)
    target_quality = np.zeros(frames, dtype=np.float32)
    present = np.zeros(frames, dtype=np.uint8)
    present_valid = np.zeros(frames, dtype=np.uint8)
    object_to_hand_uv = np.zeros((frames, 2), dtype=np.float32)
    relation_valid = np.zeros(frames, dtype=np.uint8)
    relation_quality = np.zeros(frames, dtype=np.float32)

    events_by_frame: dict[int, list[dict]] = defaultdict(list)
    for event in metadata.get("events", []):
        if int(event.get("object_id", -1)) != 1 or "detections" not in event:
            continue
        events_by_frame[int(event["frame"])].extend(event["detections"])

    selector = select_wipe if task == "wipe_table" else select_classify
    for frame, detections in events_by_frame.items():
        if not 0 <= frame < frames:
            continue
        selected = selector(detections, width, height)
        if selected is None:
            continue
        box, quality = selected
        quality = float(np.clip(quality, 0.0, 1.0))
        if quality < minimum_quality:
            continue
        uv[frame] = _center(box) / np.asarray((width, height), dtype=np.float32)
        target_valid[frame] = 1
        target_quality[frame] = quality
        present[frame] = 1
        present_valid[frame] = 1
        hand = select_hand(detections, width, height, box)
        if hand is not None:
            hand_box, hand_quality = hand
            hand_uv = _center(hand_box) / np.asarray(
                (width, height), dtype=np.float32)
            object_to_hand_uv[frame] = hand_uv - uv[frame]
            relation_valid[frame] = 1
            relation_quality[frame] = min(quality, float(hand_quality))

    episode = Path(metadata["source"]).stem
    output_dir = output_root / task / domain
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"{episode}.npz"
    np.savez_compressed(
        output,
        uv=uv,
        target_valid=target_valid,
        target_quality=target_quality,
        present=present,
        present_valid=present_valid,
        object_to_hand_uv=object_to_hand_uv,
        relation_valid=relation_valid,
        relation_quality=relation_quality,
    )
    return {
        "episode": episode,
        "task": task,
        "domain": domain,
        "frames": frames,
        "valid": int(target_valid.sum()),
        "relation_valid": int(relation_valid.sum()),
        "valid_fraction": float(target_valid.mean()),
        "output": str(output),
        "source_metadata": str(path),
        "label_semantics": "detector_box_center_not_contact_point",
        "relation_semantics": "object_center_to_detected_hand_region_center",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--minimum-quality", type=float, default=0.35)
    args = parser.parse_args()
    paths = sorted(args.metadata_root.glob("*/*/metadata/*.json"))
    if not paths:
        raise SystemExit(f"no metadata JSON under {args.metadata_root}")
    records = [build_one(path, args.output_root, args.minimum_quality) for path in paths]
    summary = {
        "source": str(args.metadata_root),
        "label_kind": "sparse_detector_box_center",
        "explicitly_not": ["instance_mask", "contact_point", "combined_foreground"],
        "episodes": len(records),
        "frames": sum(item["frames"] for item in records),
        "valid": sum(item["valid"] for item in records),
        "relation_valid": sum(item["relation_valid"] for item in records),
        "by_task_domain": {},
        "records": records,
    }
    for task in ("wipe_table", "classify_gadgets"):
        for domain in ("robot", "human"):
            subset = [r for r in records if r["task"] == task and r["domain"] == domain]
            summary["by_task_domain"][f"{task}/{domain}"] = {
                "episodes": len(subset),
                "frames": sum(r["frames"] for r in subset),
                "valid": sum(r["valid"] for r in subset),
                "relation_valid": sum(r["relation_valid"] for r in subset),
            }
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "manifest.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["by_task_domain"], indent=2))


if __name__ == "__main__":
    main()

