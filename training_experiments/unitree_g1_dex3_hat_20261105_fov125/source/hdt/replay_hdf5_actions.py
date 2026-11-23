"""Replay a recorded 128-D HAT action as an online ScaleBFM reference.

The service subscribes to ScaleBridge RobotState messages on port 5561 and
publishes sliding HatChunk windows on port 5562.  It intentionally uses the
same conversion function as live HAT inference, so waist/head/wrist poses and
wrist-local fingertips keep exactly the deployed online semantics.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import h5py
import numpy as np

from hdt import online_hat_service


def load_actions(
    path: Path,
    dataset: str,
    start_frame: int,
    end_frame: int | None,
) -> tuple[np.ndarray, float | None]:
    """Load and validate a finite ``(T, 128)`` action array."""
    with h5py.File(path, "r") as stream:
        if dataset not in stream:
            raise KeyError(f"{path} has no dataset {dataset!r}")
        actions = np.asarray(stream[dataset][()], dtype=np.float32)
        stored_fps = stream.attrs.get("fps")

    if actions.ndim != 2 or actions.shape[1] != 128:
        raise ValueError(f"Expected {dataset} shaped (T, 128), got {actions.shape}")
    if not np.isfinite(actions).all():
        raise ValueError(f"{dataset} contains NaN or infinity")
    if start_frame < 0:
        raise ValueError("--start-frame must not be negative")
    stop = len(actions) if end_frame is None else end_frame
    if stop <= start_frame or stop > len(actions):
        raise ValueError(
            f"Invalid frame range [{start_frame}, {stop}) for {len(actions)} frames"
        )
    fps = None if stored_fps is None else float(stored_fps)
    return actions[start_frame:stop], fps


def action_window(actions: np.ndarray, frame: int, window_frames: int) -> np.ndarray:
    """Return a fixed-size window, padding the recording's final frame."""
    frame = int(np.clip(frame, 0, len(actions) - 1))
    window = actions[frame : frame + window_frames]
    if len(window) < window_frames:
        padding = np.repeat(actions[-1:], window_frames - len(window), axis=0)
        window = np.concatenate((window, padding), axis=0)
    return window


def _wait_for_start(
    start_event: threading.Event, stop_event: threading.Event
) -> None:
    if not sys.stdin.isatty():
        print(
            "[hdf5-replay] stdin 不是交互终端；保持首帧。若需自动开始请加 "
            "--start-immediately。",
            flush=True,
        )
        return
    fd = sys.stdin.fileno()
    original = termios.tcgetattr(fd)
    print(
        "[hdf5-replay] ScaleBridge READY 后在本终端按空格开始回放...",
        flush=True,
    )
    try:
        tty.setcbreak(fd)
        while not stop_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            if sys.stdin.read(1) == " ":
                start_event.set()
                return
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, original)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="HDF5 file containing a 128-D action")
    parser.add_argument("--dataset", default="action")
    parser.add_argument("--fps", type=float, default=None, help="default: HDF5 attr, otherwise 30")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None, help="exclusive")
    parser.add_argument("--window-frames", type=int, default=64)
    parser.add_argument("--publish-period", type=float, default=0.2)
    parser.add_argument("--state-endpoint", default="tcp://127.0.0.1:5561")
    parser.add_argument("--chunk-endpoint", default="tcp://127.0.0.1:5562")
    parser.add_argument(
        "--scalebridge-root",
        type=Path,
        default=online_hat_service.DEFAULT_SCALEBRIDGE_ROOT,
    )
    parser.add_argument(
        "--start-immediately",
        action="store_true",
        help="start with the first RobotState instead of waiting for Enter",
    )
    parser.add_argument("--loop", action="store_true")
    parser.add_argument(
        "--exit-on-complete",
        action="store_true",
        help="exit after publishing one final-pose chunk instead of refreshing it",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.window_frames < 6:
        raise ValueError("--window-frames must be at least 6")
    if args.publish_period <= 0:
        raise ValueError("--publish-period must be positive")

    actions, stored_fps = load_actions(
        args.input.expanduser(), args.dataset, args.start_frame, args.end_frame
    )
    fps = args.fps if args.fps is not None else (stored_fps or 30.0)
    if fps <= 0:
        raise ValueError("--fps must be positive")
    if args.fps is None and stored_fps is None:
        print("[hdf5-replay] HDF5 未记录 fps，按 30 Hz 回放。", flush=True)

    online_hat_service.PROTOCOL_ROOT = args.scalebridge_root
    (
        _,
        monotonic_ns,
        validate_robot_state,
        LatestPublisher,
        LatestSubscriber,
    ) = online_hat_service._install_protocol(args.scalebridge_root)
    states = LatestSubscriber(args.state_endpoint, bind=False)
    chunks = LatestPublisher(args.chunk_endpoint, bind=True)

    start_event = threading.Event()
    stop_event = threading.Event()
    keyboard_thread = None
    if args.start_immediately:
        start_event.set()
    else:
        keyboard_thread = threading.Thread(
            target=_wait_for_start,
            args=(start_event, stop_event),
            daemon=True,
        )
        keyboard_thread.start()

    sequence_id = 0
    started_at = None
    next_publish = 0.0
    completion_reported = False
    ready_action = np.repeat(actions[0:1], args.window_frames, axis=0)
    duration = (len(actions) - 1) / fps
    print(
        f"[hdf5-replay] loaded {len(actions)} frames @ {fps:g} Hz "
        f"({duration:.2f}s) from {args.input}",
        flush=True,
    )
    print(
        f"[hdf5-replay] waiting for RobotState on {args.state_endpoint}; "
        f"publishing HatChunk on {args.chunk_endpoint}",
        flush=True,
    )

    try:
        while True:
            raw_state = states.receive_blocking(timeout_ms=1000)
            if raw_state is None:
                continue
            state = validate_robot_state(raw_state)
            now = time.monotonic()
            if now < next_publish:
                continue

            if not start_event.is_set():
                window = ready_action
                source_frame = 0
            else:
                if started_at is None:
                    started_at = now
                    print("[hdf5-replay] playback started", flush=True)
                source_frame = int((now - started_at) * fps)
                if source_frame >= len(actions):
                    if args.loop:
                        started_at = now
                        source_frame = 0
                        completion_reported = False
                        print("[hdf5-replay] looping", flush=True)
                    else:
                        source_frame = len(actions) - 1
                        if not completion_reported:
                            print(
                                "[hdf5-replay] playback complete; holding final pose",
                                flush=True,
                            )
                            completion_reported = True
                window = action_window(actions, source_frame, args.window_frames)

            chunk = online_hat_service.action_chunk_to_targets(
                window,
                state,
                fps,
                sequence_id,
                monotonic_ns(),
            )
            chunks.send(chunk)
            if sequence_id == 0 or sequence_id % 25 == 0:
                mode = "READY" if not start_event.is_set() else f"frame={source_frame}"
                print(f"[hdf5-replay] sent chunk {sequence_id} ({mode})", flush=True)
            sequence_id += 1
            next_publish = now + args.publish_period

            if completion_reported and args.exit_on_complete:
                break
    except KeyboardInterrupt:
        print("\n[hdf5-replay] stopped", flush=True)
    finally:
        stop_event.set()
        if keyboard_thread is not None:
            keyboard_thread.join(timeout=1.0)
        states.close()
        chunks.close()


if __name__ == "__main__":
    main()
