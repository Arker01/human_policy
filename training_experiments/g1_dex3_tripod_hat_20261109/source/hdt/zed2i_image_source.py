"""Camera callback feeding the robot's ZED 2i stream into ``online_hat_service``.

The robot runs ``gear_sonic/scripts/zed2i_sender.py``, which publishes the
four-part TWIST2 RealSense message ``[b"realsense", pickle(meta), jpeg, depth]``
on a ZMQ PUB socket.  Its ``--view`` default is ``left``, so the JPEG already
carries only the left eye of the stereo pair.

Channel order matters.  The deployed ``BEST_CKPT`` uses the dex5 robot branch:
its JPEG frames are decoded by ``cv2.imdecode`` and passed to the model without
a channel conversion in ``data_utils_hdt.py``.  Consequently this checkpoint
actually saw BGR tensors for dex5, even though an old loader comment calls them
RGB.  This source therefore defaults to BGR to reproduce the checkpoint exactly.
Set ``HAT_CAMERA_CHANNEL_ORDER=rgb`` only for a checkpoint trained on RGB input.

Use it as ``--image-source hdt.zed2i_image_source:get_images``.  Run it directly
to check the link before starting the policy::

    python -m hdt.zed2i_image_source --save /tmp/hat_view.png
"""

from __future__ import annotations

import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import zmq


ENDPOINT = os.environ.get("HAT_CAMERA_ENDPOINT", "tcp://192.168.123.164:5555")
TOPIC = os.environ.get("HAT_CAMERA_TOPIC", "realsense").encode()
# Matches ``data.image_resolution_hw: [240, 320]`` in the checkpoint config.
HEIGHT = int(os.environ.get("HAT_CAMERA_HEIGHT", "240"))
WIDTH = int(os.environ.get("HAT_CAMERA_WIDTH", "320"))
# A frozen image is worse than a crash: ScaleBFM already freezes and latches the
# reference when chunks stop arriving, so dying loudly here is the safe failure.
MAX_AGE_S = float(os.environ.get("HAT_CAMERA_MAX_AGE_S", "1.0"))
STARTUP_TIMEOUT_S = float(os.environ.get("HAT_CAMERA_STARTUP_TIMEOUT_S", "20.0"))
POLICY_CHANNEL_ORDER = os.environ.get(
    "HAT_CAMERA_CHANNEL_ORDER", "bgr"
).strip().lower()
if POLICY_CHANNEL_ORDER not in {"bgr", "rgb"}:
    raise ValueError(
        "HAT_CAMERA_CHANNEL_ORDER must be 'bgr' or 'rgb', got "
        f"{POLICY_CHANNEL_ORDER!r}"
    )


class _SourceVideoRecorder:
    """Encode every received source JPEG without blocking camera reception."""

    _STOP = object()

    def __init__(self, output_path, fps):
        import imageio_ffmpeg

        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = float(fps)
        if self.fps <= 0:
            raise ValueError(f"video fps must be positive, got {self.fps}")
        # Four seconds of slack keeps a transient encoder slowdown away from
        # the latest-frame receiver.  If it fills, dropping video frames is
        # safer than delaying the images used to drive the robot.
        self._queue = queue.Queue(maxsize=max(8, int(round(self.fps * 4))))
        self._closed = False
        self._submitted = 0
        self._written = 0
        self._dropped = 0
        self._error = None
        # Resolve the bundled encoder synchronously so deployment fails before
        # motion starts if H.264 recording is unavailable.
        self._ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        self._thread = threading.Thread(
            target=self._encode_loop,
            name="zed-source-video-recorder",
            daemon=True,
        )
        self._thread.start()

    def submit(self, jpeg):
        if self._closed:
            return
        self._submitted += 1
        try:
            self._queue.put_nowait(bytes(jpeg))
        except queue.Full:
            self._dropped += 1

    def _encode_loop(self):
        process = None
        try:
            command = [
                self._ffmpeg,
                "-hide_banner", "-loglevel", "error", "-y",
                "-f", "mjpeg", "-framerate", f"{self.fps:g}",
                "-i", "pipe:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                str(self.output_path),
            ]
            # Keep ffmpeg out of the terminal's foreground process group.  A
            # Ctrl+C stops Python first; Python then closes this pipe cleanly
            # so ffmpeg can write the MP4 index before exiting.
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                start_new_session=True,
            )
            while True:
                item = self._queue.get()
                if item is self._STOP:
                    break
                process.stdin.write(item)
                self._written += 1
        except Exception as exc:
            self._error = exc
        finally:
            if process is not None:
                if process.stdin is not None and not process.stdin.closed:
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                try:
                    returncode = process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    process.terminate()
                    try:
                        returncode = process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        returncode = process.wait(timeout=5)
                if returncode != 0 and self._error is None:
                    self._error = RuntimeError(
                        f"ffmpeg exited with status {returncode}"
                    )

    def close(self):
        if not self._closed:
            self._closed = True
            deadline = time.monotonic() + 10.0
            while self._thread.is_alive():
                try:
                    self._queue.put(self._STOP, timeout=0.1)
                    break
                except queue.Full:
                    if time.monotonic() >= deadline:
                        if self._error is None:
                            self._error = RuntimeError(
                                "timed out queueing ZED video finalization"
                            )
                        break
        self._thread.join(timeout=30)
        if self._thread.is_alive() and self._error is None:
            self._error = RuntimeError("timed out finalizing ZED video")
        return {
            "path": str(self.output_path),
            "submitted": self._submitted,
            "written": self._written,
            "dropped": self._dropped,
            "error": self._error,
        }


class _Zed2iReceiver:
    """Background latest-frame reader.

    ``CONFLATE`` cannot be used here -- it silently drops all but the first part
    of a multipart message -- so the queue is kept short and drained by a thread
    instead.
    """

    def __init__(self, endpoint=ENDPOINT):
        self.endpoint = endpoint
        self._lock = threading.Lock()
        self._jpeg = None
        self._received_at = 0.0
        self._frames = 0
        self._error = None
        self._video_recorder = None
        self._context = zmq.Context.instance()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVHWM, 2)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.connect(endpoint)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        poller = zmq.Poller()
        poller.register(self._socket, zmq.POLLIN)
        while True:
            try:
                if not poller.poll(500):
                    continue
                parts = self._socket.recv_multipart()
                if len(parts) < 3 or parts[0] != TOPIC:
                    continue
                with self._lock:
                    self._jpeg = parts[2]
                    self._received_at = time.monotonic()
                    self._frames += 1
                    video_recorder = self._video_recorder
                if video_recorder is not None:
                    video_recorder.submit(parts[2])
            except Exception as exc:  # surfaced on the next get_images call
                with self._lock:
                    self._error = exc
                return

    def set_video_recorder(self, recorder):
        with self._lock:
            previous = self._video_recorder
            self._video_recorder = recorder
        return previous

    def latest_jpeg(self, timeout_s=0.0):
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                if self._error is not None:
                    raise RuntimeError(
                        f"ZED 2i receiver thread died: {self._error!r}"
                    )
                if self._jpeg is not None:
                    return self._jpeg, self._received_at, self._frames
            if time.monotonic() >= deadline:
                return None, 0.0, 0
            time.sleep(0.01)


_receiver = None
_receiver_lock = threading.Lock()
_video_recorder = None
_video_recorder_lock = threading.Lock()


def _get_receiver():
    global _receiver
    with _receiver_lock:
        if _receiver is None:
            _receiver = _Zed2iReceiver()
            print(f"[zed2i] subscribed to {ENDPOINT}", flush=True)
        return _receiver


def start_video_recording(output_path, fps=30.0):
    """Record every received source JPEG to an H.264 MP4 until stopped."""
    global _video_recorder
    with _video_recorder_lock:
        if _video_recorder is not None:
            raise RuntimeError("ZED source video recording is already active")
        recorder = _SourceVideoRecorder(output_path, fps)
        previous = _get_receiver().set_video_recorder(recorder)
        if previous is not None:
            recorder.close()
            raise RuntimeError("ZED receiver already has a video recorder")
        _video_recorder = recorder
    print(
        f"[zed2i] recording source stream at {float(fps):g} FPS to {output_path}",
        flush=True,
    )


def stop_video_recording():
    """Detach and finalize the active source MP4, if any."""
    global _video_recorder
    with _video_recorder_lock:
        recorder = _video_recorder
        if recorder is None:
            return None
        _get_receiver().set_video_recorder(None)
        _video_recorder = None
    result = recorder.close()
    if result["error"] is not None:
        print(
            f"[zed2i] video finalization failed: {result['error']!r}",
            flush=True,
        )
    else:
        print(
            "[zed2i] video saved to "
            f"{result['path']} ({result['written']} frames, "
            f"{result['dropped']} dropped)",
            flush=True,
        )
    return result


def latest_source_bgr():
    """Full-resolution decoded frame, exactly as the robot published it.

    Only for aiming and focus checks -- the policy never sees this size.
    """
    jpeg, _, _ = _get_receiver().latest_jpeg(timeout_s=STARTUP_TIMEOUT_S)
    if jpeg is None:
        raise RuntimeError(f"No ZED 2i frame on {ENDPOINT}")
    return cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)


def get_images(*, camera_names, robot_state=None):
    """Return uint8 NCHW frames in ``POLICY_CHANNEL_ORDER``.

    ``robot_state`` is part of the service's callback contract but this camera
    does not depend on it.
    """
    names = tuple(camera_names)
    if len(names) != 1:
        raise ValueError(
            f"Only one physical stream ({ENDPOINT}) is available, but the "
            f"checkpoint asks for cameras {names}"
        )
    receiver = _get_receiver()
    first = receiver._frames == 0
    jpeg, received_at, frames = receiver.latest_jpeg(
        timeout_s=STARTUP_TIMEOUT_S if first else 0.0
    )
    if jpeg is None:
        raise RuntimeError(
            f"No ZED 2i frame on {ENDPOINT} after {STARTUP_TIMEOUT_S:.0f}s. "
            "Is zed2i_sender.py running on the robot?"
        )
    age = time.monotonic() - received_at
    if age > MAX_AGE_S:
        raise RuntimeError(
            f"ZED 2i frame is {age:.2f}s stale (limit {MAX_AGE_S:.2f}s) after "
            f"{frames} frames; refusing to drive the policy on a frozen image"
        )
    bgr = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError("Failed to decode the ZED 2i JPEG payload")
    if bgr.shape[0] != HEIGHT or bgr.shape[1] != WIDTH:
        bgr = cv2.resize(bgr, (WIDTH, HEIGHT))
    policy_image = (
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if POLICY_CHANNEL_ORDER == "rgb"
        else bgr
    )
    return np.ascontiguousarray(policy_image.transpose(2, 0, 1))[None].astype(
        np.uint8
    )


def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--save", default="", help="write the policy input as PNG")
    parser.add_argument(
        "--save-source", default="", help="write the full-resolution frame as PNG"
    )
    args = parser.parse_args()

    if args.save_source:
        source = latest_source_bgr()
        cv2.imwrite(args.save_source, source)
        print(f"source frame {source.shape[1]}x{source.shape[0]} -> {args.save_source}")

    start = time.monotonic()
    count = 0
    last = None
    while time.monotonic() - start < args.seconds:
        last = get_images(camera_names=("top",))
        count += 1
        time.sleep(0.03)
    hz = count / max(time.monotonic() - start, 1e-6)
    print(f"shape={last.shape} dtype={last.dtype} calls={count} ({hz:.1f} Hz)")
    labels = POLICY_CHANNEL_ORDER.upper()
    print(
        f"per-channel mean ({labels}) = "
        f"{last[0].reshape(3, -1).mean(axis=1).round(1)}"
    )
    if args.save:
        policy_hwc = last[0].transpose(1, 2, 0)
        display_bgr = (
            cv2.cvtColor(policy_hwc, cv2.COLOR_RGB2BGR)
            if POLICY_CHANNEL_ORDER == "rgb"
            else policy_hwc
        )
        cv2.imwrite(args.save, display_bgr)
        print(f"wrote {args.save}")


if __name__ == "__main__":
    main()
