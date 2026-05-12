import argparse
import json

import h5py
import numpy as np


RETARGETTING_INDICES = [0, 4, 9, 14, 19, 24]
TIP_INDICES = [4, 9, 14, 19, 24]
NON_THUMB_TIP_INDICES = [9, 14, 19, 24]

OUTPUT_LEFT_EEF = np.arange(80, 89)
OUTPUT_RIGHT_EEF = np.arange(30, 39)
OUTPUT_LEFT_KEYPOINTS = np.arange(10, 10 + 3 * len(RETARGETTING_INDICES))
OUTPUT_RIGHT_KEYPOINTS = np.arange(40, 40 + 3 * len(RETARGETTING_INDICES))


def normalize(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=axis, keepdims=True) + 1e-12)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def expand_hand_25_from_tips(hand_kpts_25: np.ndarray) -> np.ndarray:
    hand_kpts_25 = np.asarray(hand_kpts_25, dtype=np.float64)
    out = hand_kpts_25.copy()
    palm = hand_kpts_25[:, 0:1, :]
    alphas = np.linspace(0.0, 1.0, 5, dtype=np.float64)
    for finger in range(5):
        base = finger * 5
        tip = base + 4
        tip_pos = hand_kpts_25[:, tip : tip + 1, :]
        for j, alpha in enumerate(alphas):
            idx = base + j
            if idx == 0:
                out[:, idx, :] = palm[:, 0, :]
            else:
                out[:, idx, :] = palm[:, 0, :] + alpha * (tip_pos[:, 0, :] - palm[:, 0, :])
    return out


def decode_actions(actions: np.ndarray) -> dict:
    left_R = rotation_6d_to_matrix(actions[:, OUTPUT_LEFT_EEF[3:]])
    right_R = rotation_6d_to_matrix(actions[:, OUTPUT_RIGHT_EEF[3:]])
    left_pos = actions[:, OUTPUT_LEFT_EEF[0:3]].astype(np.float64)
    right_pos = actions[:, OUTPUT_RIGHT_EEF[0:3]].astype(np.float64)

    left_local = np.zeros((actions.shape[0], 25, 3), dtype=np.float64)
    right_local = np.zeros((actions.shape[0], 25, 3), dtype=np.float64)
    left_local[:, RETARGETTING_INDICES, :] = actions[:, OUTPUT_LEFT_KEYPOINTS].reshape((-1, 6, 3))
    right_local[:, RETARGETTING_INDICES, :] = actions[:, OUTPUT_RIGHT_KEYPOINTS].reshape((-1, 6, 3))

    left_joints_local = expand_hand_25_from_tips(left_local)
    right_joints_local = expand_hand_25_from_tips(right_local)

    left_world = left_pos[:, None, :] + np.einsum("tij,tkj->tki", left_R, left_local)
    right_world = right_pos[:, None, :] + np.einsum("tij,tkj->tki", right_R, right_local)
    left_joints_world = left_pos[:, None, :] + np.einsum("tij,tkj->tki", left_R, left_joints_local)
    right_joints_world = right_pos[:, None, :] + np.einsum("tij,tkj->tki", right_R, right_joints_local)

    return {
        "left_R": left_R,
        "right_R": right_R,
        "left_pos": left_pos,
        "right_pos": right_pos,
        "left_world": left_world,
        "right_world": right_world,
        "left_joints_world": left_joints_world,
        "right_joints_world": right_joints_world,
    }


def summarize_scalar(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "p05": float(np.percentile(values, 5)),
        "p95": float(np.percentile(values, 95)),
    }


def check_per_frame(name: str, values: np.ndarray, pass_mask: np.ndarray, min_pass_rate: float) -> dict:
    pass_rate = float(np.mean(pass_mask))
    return {
        "name": name,
        "pass": bool(pass_rate >= min_pass_rate),
        "pass_rate": pass_rate,
        "failed_frames": int(np.size(pass_mask) - np.count_nonzero(pass_mask)),
        "stats": summarize_scalar(values),
    }


def axis_relation_check(name: str, values: np.ndarray, *, sign: str, epsilon: float, min_pass_rate: float) -> dict:
    if sign == "positive":
        per_joint_pass = values > epsilon
        margins = np.min(values, axis=1)
    elif sign == "negative":
        per_joint_pass = values < -epsilon
        margins = -np.max(values, axis=1)
    else:
        raise ValueError(sign)
    return check_per_frame(name, margins, np.all(per_joint_pass, axis=1), min_pass_rate)


def evaluate(decoded: dict, args: argparse.Namespace) -> dict:
    left_tip_rel = decoded["left_world"][:, TIP_INDICES] - decoded["left_pos"][:, None, :]
    right_tip_rel = decoded["right_world"][:, TIP_INDICES] - decoded["right_pos"][:, None, :]

    left_thumb = decoded["left_world"][:, 4]
    left_non_thumb = decoded["left_world"][:, NON_THUMB_TIP_INDICES]
    right_thumb = decoded["right_world"][:, 4]
    right_non_thumb = decoded["right_world"][:, NON_THUMB_TIP_INDICES]

    left_thumb_y_margin = np.min(left_non_thumb[:, :, 1], axis=1) - left_thumb[:, 1]
    right_thumb_y_margin = right_thumb[:, 1] - np.max(right_non_thumb[:, :, 1], axis=1)
    left_thumb_z_higher_margin = left_thumb[:, 2] - np.max(left_non_thumb[:, :, 2], axis=1)
    right_thumb_z_higher_margin = right_thumb[:, 2] - np.max(right_non_thumb[:, :, 2], axis=1)

    checks = [
        axis_relation_check(
            "left_all_fingertips_are_on_global_z_minus_side_of_wrist",
            left_tip_rel[:, :, 2],
            sign="negative",
            epsilon=args.coord_epsilon,
            min_pass_rate=args.min_pass_rate,
        ),
        axis_relation_check(
            "left_all_fingertips_are_on_global_x_plus_side_of_wrist",
            left_tip_rel[:, :, 0],
            sign="positive",
            epsilon=args.coord_epsilon,
            min_pass_rate=args.min_pass_rate,
        ),
        check_per_frame(
            "left_thumb_tip_global_y_is_less_than_all_other_left_fingertips",
            left_thumb_y_margin,
            left_thumb_y_margin > args.thumb_y_margin,
            args.min_pass_rate,
        ),
        check_per_frame(
            "left_thumb_tip_global_z_is_greater_than_all_other_left_fingertips",
            left_thumb_z_higher_margin,
            left_thumb_z_higher_margin > args.thumb_z_margin,
            args.min_pass_rate,
        ),
        axis_relation_check(
            "right_all_fingertips_are_on_global_z_minus_side_of_wrist",
            right_tip_rel[:, :, 2],
            sign="negative",
            epsilon=args.coord_epsilon,
            min_pass_rate=args.min_pass_rate,
        ),
        axis_relation_check(
            "right_all_fingertips_are_on_global_x_plus_side_of_wrist",
            right_tip_rel[:, :, 0],
            sign="positive",
            epsilon=args.coord_epsilon,
            min_pass_rate=args.min_pass_rate,
        ),
        check_per_frame(
            "right_thumb_tip_global_y_is_greater_than_all_other_right_fingertips",
            right_thumb_y_margin,
            right_thumb_y_margin > args.thumb_y_margin,
            args.min_pass_rate,
        ),
        check_per_frame(
            "right_thumb_tip_global_z_is_greater_than_all_other_right_fingertips",
            right_thumb_z_higher_margin,
            right_thumb_z_higher_margin > args.thumb_z_margin,
            args.min_pass_rate,
        ),
    ]

    return {
        "requirements": [
            "Left all fingertips must be on the global Z- side of the left wrist.",
            "Left all fingertips must be on the global X+ side of the left wrist.",
            "Left thumb tip global Y coordinate must be less than all other left fingertips.",
            "Left thumb tip global Z coordinate must be greater than all other left fingertips.",
            "Right all fingertips must be on the global Z- side of the right wrist.",
            "Right all fingertips must be on the global X+ side of the right wrist.",
            "Right thumb tip global Y coordinate must be greater than all other right fingertips.",
            "Right thumb tip global Z coordinate must be greater than all other right fingertips.",
        ],
        "checks": checks,
        "overall_pass": bool(all(c["pass"] for c in checks)),
    }


def frame_range(num_frames: int, args: argparse.Namespace) -> tuple[int, int, float]:
    fps = args.fps
    start = args.frame_start if args.frame_start is not None else 0
    end = args.frame_end if args.frame_end is not None else int(np.ceil(args.max_seconds * fps))
    start = max(0, min(num_frames, start))
    end = max(start, min(num_frames, end))
    return start, end, fps


def print_report(result: dict) -> None:
    print("Requirements checked:")
    for req in result["requirements"]:
        print(f"  - {req}")
    print(f"\nOverall: {'PASS' if result['overall_pass'] else 'FAIL'}")
    print("\nChecks:")
    for check in result["checks"]:
        status = "PASS" if check["pass"] else "FAIL"
        stats = check["stats"]
        print(
            f"  [{status}] {check['name']}: "
            f"pass_rate={check['pass_rate']:.3f}, failed_frames={check['failed_frames']}, "
            f"mean={stats['mean']:.6f}, min={stats['min']:.6f}, max={stats['max']:.6f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Inspire hand orientation constraints.")
    parser.add_argument("--file", required=True, help="HDF5 file containing action dataset")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS used to convert first seconds to frames")
    parser.add_argument("--max-seconds", type=float, default=4.0, help="Evaluate the first N seconds")
    parser.add_argument("--frame-start", type=int, default=None, help="Override start frame, inclusive")
    parser.add_argument("--frame-end", type=int, default=None, help="Override end frame, exclusive")
    parser.add_argument("--min-pass-rate", type=float, default=0.95)
    parser.add_argument("--coord-epsilon", type=float, default=1e-6)
    parser.add_argument("--thumb-y-margin", type=float, default=0.0)
    parser.add_argument("--thumb-z-margin", type=float, default=0.0)
    parser.add_argument("--json-out", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with h5py.File(args.file, "r") as f:
        actions = f["action"][:]
    start, end, fps = frame_range(actions.shape[0], args)
    result = evaluate(decode_actions(actions[start:end]), args)
    result["file"] = args.file
    result["num_frames_total"] = int(actions.shape[0])
    result["fps_used"] = float(fps)
    result["frame_range"] = {"start": int(start), "end_exclusive": int(end), "count": int(end - start)}
    result["seconds"] = {"start": 0.0, "end": float(args.max_seconds)}
    print_report(result)
    if args.json_out is not None:
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
