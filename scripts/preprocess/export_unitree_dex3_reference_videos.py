#!/usr/bin/env python3
"""Export HDF5 camera mosaics and frame-aligned keypoint comparisons.

Uses the episode selection in visualizations/sample_videos.json. All image pixels
are decoded from HDF5; original downloaded videos are not used by this exporter.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from fractions import Fraction
import json
import os
from pathlib import Path
import time

import av
import cv2
import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
CAMERA_ORDER = ("left", "right", "left_wrist", "right_wrist")
CAMERA_LABELS = {
    "left": "HEAD LEFT", "right": "HEAD RIGHT",
    "left_wrist": "WRIST LEFT", "right_wrist": "WRIST RIGHT",
}


def write_json(path, value):
    temporary = path.with_suffix(".json.partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def text_on_rgb(canvas, text, xy, scale=0.55):
    cv2.putText(canvas, text, xy, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def make_stream(container, width, height, fps):
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height = width, height
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 1) / fps
    stream.codec_context.thread_count = 2
    stream.options = {"crf": "18", "preset": "fast"}
    return stream


def encode_rgb(container, stream, rgb, index, fps):
    frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
    frame.pts = index
    frame.time_base = Fraction(1, 1) / fps
    for packet in stream.encode(frame):
        container.mux(packet)


def decode_hdf_image(dataset, index, color_order):
    value = dataset[index]
    if value.ndim == 1:
        rgb = cv2.imdecode(value, cv2.IMREAD_COLOR)
        if rgb is None:
            raise ValueError(f"Invalid JPEG: {dataset.name}[{index}]")
        # These files intentionally store JPEGs for the project's direct RGB
        # imdecode convention. Only standard BGR-decoded files need swapping.
        if not color_order.startswith("RGB"):
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    else:
        rgb = value
    return rgb


def verify_video(path, expected_frames, fps, size, expected_rgb=None):
    samples = {0, expected_frames // 2, expected_frames - 1}
    errors = []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        stream.thread_count = 2
        if (stream.width, stream.height) != size or stream.average_rate != fps:
            raise ValueError(f"Incorrect dimensions or frame rate: {path}")
        count = 0
        for i, frame in enumerate(container.decode(stream)):
            if abs(float(frame.pts * frame.time_base) - i / float(fps)) > 1e-5:
                raise ValueError(f"Video timestamp mismatch at frame {i}: {path}")
            if expected_rgb is not None and i in samples:
                actual = frame.to_ndarray(format="rgb24").astype(np.float32)
                target = expected_rgb[i].astype(np.float32)
                error = float(np.abs(actual - target).mean())
                if error > 8:
                    raise ValueError(f"Unexpected image/color difference ({error:.2f}): {path}")
                errors.append({"frame": i, "rgb_mean_absolute_error": error})
            count += 1
        if count != expected_frames:
            raise ValueError(f"Frame count {count} != {expected_frames}: {path}")
        codec = stream.codec_context.name
    return {"frames": count, "fps": float(fps), "width": size[0], "height": size[1],
            "codec": codec, "bytes": path.stat().st_size, "sample_rgb_checks": errors}


def export_episode(job):
    started = time.monotonic()
    source = Path(job["source"])
    keypoint_path = Path(job["output"])
    real_path = keypoint_path.with_name(keypoint_path.stem + "_real.mp4")
    compare_path = keypoint_path.with_name(keypoint_path.stem + "_compare.mp4")
    for path in (real_path, compare_path):
        if path.exists():
            raise FileExistsError(f"Output already exists: {path}")
    real_temp = real_path.with_suffix(".encoding.mp4")
    compare_temp = compare_path.with_suffix(".encoding.mp4")
    real_samples, compare_samples = {}, {}

    with ExitStack() as stack:
        f = stack.enter_context(h5py.File(source, "r"))
        keypoints = stack.enter_context(av.open(str(keypoint_path)))
        kp_stream = keypoints.streams.video[0]
        kp_stream.thread_type = "AUTO"
        kp_stream.thread_count = 2
        fps = Fraction(str(float(f.attrs["fps"]))).limit_denominator(100000)
        length = int(f["action"].shape[0])
        if length != job["frames"] or kp_stream.average_rate != fps:
            raise ValueError(f"HDF5/keypoint timing mismatch: {source}")
        cameras = [name for name in CAMERA_ORDER if f"observation.image.{name}" in f]
        datasets = {name: f[f"observation.image.{name}"] for name in cameras}
        if not cameras or any(len(ds) != length for ds in datasets.values()):
            raise ValueError(f"Camera frame count mismatch: {source}")
        color_order = str(f.attrs.get("image_decode_color_order", "BGR"))
        if not color_order.startswith("RGB"):
            raise ValueError(f"Review color convention before exporting: {source}")
        first = decode_hdf_image(datasets[cameras[0]], 0, color_order)
        tile_h, tile_w = first.shape[:2]
        label_h, header_h = 24, 32
        columns = min(2, len(cameras))
        rows = (len(cameras) + columns - 1) // columns
        mosaic_w = columns * tile_w
        mosaic_h = header_h + rows * (tile_h + label_h)
        kp_w, kp_h = kp_stream.width, kp_stream.height
        compare_size = (kp_w * 2, kp_h + 40)
        scale = min(kp_w / mosaic_w, kp_h / mosaic_h)
        scaled_w, scaled_h = int(round(mosaic_w * scale)), int(round(mosaic_h * scale))
        xpad, ypad = (kp_w - scaled_w) // 2, (kp_h - scaled_h) // 2

        real_out = stack.enter_context(av.open(str(real_temp), "w", format="mp4"))
        compare_out = stack.enter_context(av.open(str(compare_temp), "w", format="mp4"))
        real_stream = make_stream(real_out, mosaic_w, mosaic_h, fps)
        compare_stream = make_stream(compare_out, *compare_size, fps)
        kp_frames = keypoints.decode(kp_stream)
        sample_ids = {0, length // 2, length - 1}
        for i in range(length):
            try:
                kp_frame = next(kp_frames)
            except StopIteration as exc:
                raise ValueError(f"Short keypoint video: {keypoint_path}") from exc
            if abs(float(kp_frame.pts * kp_frame.time_base) - i / float(fps)) > 1e-5:
                raise ValueError(f"Keypoint timestamp mismatch at frame {i}: {keypoint_path}")
            kp_rgb = kp_frame.to_ndarray(format="rgb24")
            mosaic = np.zeros((mosaic_h, mosaic_w, 3), dtype=np.uint8)
            text_on_rgb(mosaic, f"{job['task']} / {job['episode']} | frame {i} | {i/float(fps):.2f}s", (8, 21), 0.48)
            for index, camera in enumerate(cameras):
                image = decode_hdf_image(datasets[camera], i, color_order)
                if image.shape != first.shape:
                    raise ValueError(f"Inconsistent camera size: {source}, {camera}, frame {i}")
                row, column = divmod(index, columns)
                y, x = header_h + row * (tile_h + label_h), column * tile_w
                text_on_rgb(mosaic, CAMERA_LABELS[camera], (x + 8, y + 17), 0.5)
                mosaic[y + label_h:y + label_h + tile_h, x:x + tile_w] = image
            comparison = np.zeros((compare_size[1], compare_size[0], 3), dtype=np.uint8)
            text_on_rgb(comparison, f"REAL CAMERA VIEWS | {job['task']} / {job['episode']}", (12, 27), 0.65)
            text_on_rgb(comparison, f"ACTION KEYPOINTS | frame {i} | {i/float(fps):.2f}s", (kp_w + 12, 27), 0.65)
            scaled = cv2.resize(mosaic, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            comparison[40 + ypad:40 + ypad + scaled_h, xpad:xpad + scaled_w] = scaled
            comparison[40:, kp_w:] = kp_rgb
            encode_rgb(real_out, real_stream, mosaic, i, fps)
            encode_rgb(compare_out, compare_stream, comparison, i, fps)
            if i in sample_ids:
                real_samples[i] = mosaic
                compare_samples[i] = comparison
        if next(kp_frames, None) is not None:
            raise ValueError(f"Keypoint video contains extra frames: {keypoint_path}")
        for packet in real_stream.encode():
            real_out.mux(packet)
        for packet in compare_stream.encode():
            compare_out.mux(packet)

    real_check = verify_video(real_temp, length, fps, (mosaic_w, mosaic_h), real_samples)
    compare_check = verify_video(compare_temp, length, fps, compare_size, compare_samples)
    os.replace(real_temp, real_path)
    os.replace(compare_temp, compare_path)
    return {"task": job["task"], "episode": job["episode"], "source_hdf5": str(source),
            "keypoint_video": str(keypoint_path), "real_video": str(real_path),
            "comparison_video": str(compare_path), "cameras": cameras,
            "source_color_order": color_order, "camera_tile_pixels": [tile_w, tile_h],
            "full_episode": True, "duration_seconds": length / float(fps),
            "real_validation": real_check, "comparison_validation": compare_check,
            "elapsed_seconds": round(time.monotonic() - started, 2), "status": "complete"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=REPO / "data/unitree_g1_dex3/visualizations/sample_videos.json")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    manifest = json.loads(args.manifest.read_text())
    jobs = manifest["videos"]
    report = {"status": "running", "source_manifest": str(args.manifest), "videos": [], "errors": []}
    report_path = args.manifest.parent / "real_world_videos.json"
    started = time.monotonic()
    write_json(report_path, report)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_episode, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                report["videos"].append(result)
                print(f"COMPLETE {job['task']}/{job['episode']}: {result['real_validation']['frames']} frames, "
                      f"{len(result['cameras'])} cameras, real + comparison", flush=True)
            except Exception as exc:
                error = {"task": job["task"], "episode": job["episode"], "error": repr(exc)}
                report["errors"].append(error)
                print(f"ERROR {error}", flush=True)
            write_json(report_path, report)
    report["videos"].sort(key=lambda item: (item["task"], item["episode"]))
    report["elapsed_seconds"] = round(time.monotonic() - started, 2)
    report["status"] = "complete" if not report["errors"] and len(report["videos"]) == len(jobs) else "failed"
    write_json(report_path, report)
    print(f"{report['status']}: {len(report['videos'])} episodes, {report['elapsed_seconds']} seconds", flush=True)
    if report["status"] != "complete":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
