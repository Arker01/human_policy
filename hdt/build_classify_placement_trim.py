#!/usr/bin/env python3
"""Build zero-copy classify episodes ending at the final release boundary.

The classify demonstrations finish by opening the right hand over the last box,
then moving the empty hand back across the workspace.  That return is precisely
the competing target this experiment removes.  The retained last frame is the
first raw-open frame confirmed by a short future hold; confirmation frames are
not retained, so a fast withdrawal cannot leak into the label.  Files are HDF5
virtual slices, so image payloads stay in the immutable source dataset.
"""

import argparse
import json
import os

import h5py
import numpy as np


def final_open(action, open_threshold):
    points = action[:, 40:58].reshape(-1, 6, 3)
    spread = np.linalg.norm(points[:, 1:] - points[:, :1], axis=2).mean(axis=1)
    smooth = np.convolve(spread, np.ones(7, dtype=np.float32) / 7.0, mode="same")
    candidates = []
    for index in range(20, len(action) - 20):
        was_closed = np.mean(smooth[index - 20:index] < open_threshold) >= 0.70
        stays_open = np.mean(smooth[index:index + 15] > open_threshold) >= 0.90
        if was_closed and stays_open:
            candidates.append(index)
    events = []
    for index in candidates:
        if not events or index - events[-1] > 30:
            events.append(index)
    if len(events) < 4:
        raise RuntimeError(f"expected at least four placement opens, found {events}")
    return events[-1], len(events), spread


def confirmed_open_frame(spread, event, open_threshold, hold_frames):
    """First open frame whose following hold window confirms the release."""
    for index in range(event, len(spread) - hold_frames + 1):
        if np.all(spread[index:index + hold_frames] > open_threshold):
            return index
    raise RuntimeError(
        f"no {hold_frames}-frame confirmed opening after final event {event}"
    )


def copy_attrs(source, target):
    for key, value in source.attrs.items():
        target.attrs[key] = value


def virtual_slice(source_path, output_path, cutoff, metadata):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as target:
        copy_attrs(source, target)
        target.attrs["placement_trim"] = json.dumps(metadata, sort_keys=True)

        def visit(group, target_group):
            copy_attrs(group, target_group)
            for name, value in group.items():
                if isinstance(value, h5py.Group):
                    visit(value, target_group.create_group(name))
                    continue
                if not value.shape or value.shape[0] < cutoff:
                    raise RuntimeError(
                        f"dataset {value.name} cannot be sliced to {cutoff}: {value.shape}"
                    )
                layout = h5py.VirtualLayout(
                    shape=(cutoff,) + value.shape[1:], dtype=value.dtype
                )
                virtual = h5py.VirtualSource(
                    os.path.abspath(source_path), value.name, shape=value.shape
                )
                layout[...] = virtual[:cutoff]
                created = target_group.create_virtual_dataset(name, layout)
                copy_attrs(value, created)

        visit(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--open-threshold", type=float, default=0.19)
    parser.add_argument("--open-hold-frames", type=int, default=3)
    args = parser.parse_args()
    manifest = json.load(open(args.source_manifest))
    result = {}
    summary = []
    for split, entries in manifest.items():
        result[split] = []
        for entry in entries:
            output_entry = dict(entry)
            output_entry["dataset_path"] = os.path.abspath(
                os.path.join(args.output_dir, split)
            )
            for filename in entry["file_list"]:
                source_path = os.path.join(entry["dataset_path"], filename)
                with h5py.File(source_path, "r") as source:
                    action = np.asarray(source["action"])
                event, event_count, spread = final_open(
                    action, float(args.open_threshold)
                )
                open_frame = confirmed_open_frame(
                    spread,
                    event,
                    float(args.open_threshold),
                    int(args.open_hold_frames),
                )
                # HDF5 slices are exclusive at the right edge.  Retain the first
                # confirmed-open frame, but not the look-ahead frames used to prove
                # it is a real release rather than threshold noise.
                cutoff = open_frame + 1
                metadata = {
                    "source": os.path.abspath(source_path),
                    "source_frames": len(action),
                    "cutoff_frames": cutoff,
                    "final_open_frame": event,
                    "confirmed_open_frame": open_frame,
                    "detected_open_events": event_count,
                    "open_threshold": float(args.open_threshold),
                    "open_hold_frames": int(args.open_hold_frames),
                }
                virtual_slice(
                    source_path,
                    os.path.join(output_entry["dataset_path"], filename),
                    cutoff,
                    metadata,
                )
                summary.append(dict(split=split, file=filename, **metadata))
            result[split].append(output_entry)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_manifest)), exist_ok=True)
    with open(args.output_manifest, "w") as stream:
        json.dump(result, stream, indent=2)
    with open(os.path.join(args.output_dir, "trim_summary.json"), "w") as stream:
        json.dump(summary, stream, indent=2)
    removed = np.asarray(
        [item["source_frames"] - item["cutoff_frames"] for item in summary]
    )
    print(
        f"built {len(summary)} virtual episodes; removed frames "
        f"min/median/max={removed.min()}/{np.median(removed):.0f}/{removed.max()}"
    )


if __name__ == "__main__":
    main()
