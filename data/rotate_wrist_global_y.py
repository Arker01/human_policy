import argparse

import h5py
import numpy as np


LEFT_EEF = np.arange(80, 89)
RIGHT_EEF = np.arange(30, 39)


def normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)


def rotation_6d_to_matrix(d6: np.ndarray) -> np.ndarray:
    d6 = np.asarray(d6, dtype=np.float64)
    a1 = d6[..., 0:3]
    a2 = d6[..., 3:6]
    b1 = normalize(a1)
    b2 = a2 - np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = normalize(b2)
    b3 = np.cross(b1, b2, axis=-1)
    return np.stack((b1, b2, b3), axis=-2)


def matrix_to_rotation_6d(mat: np.ndarray) -> np.ndarray:
    return np.asarray(mat, dtype=np.float32)[..., :2, :].reshape(*mat.shape[:-2], 6)


def rot_y(degrees: float) -> np.ndarray:
    theta = np.deg2rad(degrees)
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array(
        [
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ],
        dtype=np.float64,
    )


def rotate_wrist_fields(arr: np.ndarray, degrees: float) -> np.ndarray:
    out = np.array(arr, copy=True)
    correction = rot_y(degrees)
    for eef in (LEFT_EEF, RIGHT_EEF):
        old_R = rotation_6d_to_matrix(out[:, eef[3:]])
        new_R = correction @ old_R
        out[:, eef[3:]] = matrix_to_rotation_6d(new_R)
    return out


def copy_hdf5_with_rotated_wrist(src: str, dst: str, degrees: float) -> None:
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for name, item in fin.items():
            if isinstance(item, h5py.Dataset):
                data = item[()]
                if name in ("action", "observation.state"):
                    data = rotate_wrist_fields(data, degrees)
                fout.create_dataset(name, data=data, dtype=item.dtype)
            elif isinstance(item, h5py.Group):
                fin.copy(item, fout, name=name)

        for key, value in fin.attrs.items():
            fout.attrs[key] = value
        fout.attrs["wrist_correction"] = f"postprocess_global_y_{degrees:g}deg_left_multiply"
        fout.attrs["wrist_correction_source"] = src


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate left/right wrist rotations around the global Y axis in an existing HDF5.")
    parser.add_argument("--src", required=True, help="Source HDF5")
    parser.add_argument("--dst", required=True, help="Destination HDF5")
    parser.add_argument("--degrees", type=float, default=90.0, help="Global Y rotation angle in degrees")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copy_hdf5_with_rotated_wrist(args.src, args.dst, args.degrees)
    print(f"Wrote {args.dst}")


if __name__ == "__main__":
    main()
