"""Geometry checks against independent URDF chain composition, plus data repair."""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from scripts.preprocess.convert_unitree_dex3 import (
    Dex3Kinematics, joint_name, repair_isolated_telemetry,
)
from hdt.action_representation import rotation_6d_to_matrix

ASSETS = Path(__file__).resolve().parents[1] / "assets/g1_dex3"
NAMES = [
    "kLeftShoulderPitch", "kLeftShoulderRoll", "kLeftShoulderYaw", "kLeftElbow",
    "kLeftWristRoll", "kLeftWristPitch", "kLeftWristYaw",
    "kRightShoulderPitch", "kRightShoulderRoll", "kRightShoulderYaw", "kRightElbow",
    "kRightWristRoll", "kRightWristPitch", "kRightWristYaw",
    "kLeftHandThumb0", "kLeftHandThumb1", "kLeftHandThumb2",
    "kLeftHandMiddle0", "kLeftHandMiddle1", "kLeftHandIndex0", "kLeftHandIndex1",
    "kRightHandThumb0", "kRightHandThumb1", "kRightHandThumb2",
    "kRightHandIndex0", "kRightHandIndex1", "kRightHandMiddle0", "kRightHandMiddle1",
]


def independent_fk(angles):
    """Compose URDF matrices without Pinocchio or the conversion functions."""
    root = ET.parse(ASSETS / "g1_29dof_with_hand_rev_1_0.urdf").getroot()
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    transforms = {"pelvis": np.eye(4)}

    def visit(name):
        if name in transforms:
            return transforms[name]
        joint = by_child[name]
        origin = joint.find("origin")
        xyz = np.fromstring(origin.get("xyz", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        rpy = np.fromstring(origin.get("rpy", "0 0 0"), sep=" ") if origin is not None else np.zeros(3)
        fixed = np.eye(4)
        fixed[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
        fixed[:3, 3] = xyz
        motion = np.eye(4)
        if joint.get("type") == "revolute":
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            motion[:3, :3] = Rotation.from_rotvec(axis * angles.get(joint.get("name"), 0)).as_matrix()
        transforms[name] = visit(joint.find("parent").get("link")) @ fixed @ motion
        return transforms[name]
    for name in by_child:
        visit(name)
    return transforms


def test_named_joint_order_and_cartesian_geometry():
    kin = Dex3Kinematics(ASSETS)
    # Distinct joint values catch the dataset's right index/middle ordering.
    joints = np.random.default_rng(17).uniform(-0.6, 0.6, (3, 28)).astype(np.float32)
    result = kin.convert(joints, NAMES)
    for row, converted in zip(joints, result):
        angles = {joint_name(name): float(value) for name, value in zip(NAMES, row)}
        expected = independent_fk(angles)
        head = independent_fk({})["d435_link"][:3, 3]
        for side, pose_start, points_start in [("left", 80, 10), ("right", 30, 40)]:
            wrist = expected[f"{side}_wrist_yaw_link"]
            np.testing.assert_allclose(converted[pose_start:pose_start+3], wrist[:3, 3] - head, atol=1e-7)
            np.testing.assert_allclose(rotation_6d_to_matrix(converted[pose_start+3:pose_start+9]), wrist[:3, :3], atol=2e-7)
            for slot, finger in enumerate(["thumb", "index", "middle"], start=1):
                distal = 2 if finger == "thumb" else 1
                link = expected[f"{side}_hand_{finger}_{distal}_link"]
                offset = [0, -0.05 if side == "left" else 0.05, 0, 1] if finger == "thumb" else [0.05, 0, 0, 1]
                local = np.linalg.inv(wrist) @ link @ offset
                np.testing.assert_allclose(converted[points_start+3*slot:points_start+3*slot+3], local[:3], atol=1e-7)


def test_wrist_local_fingers_do_not_move_with_arm():
    kin = Dex3Kinematics(ASSETS)
    joints = np.zeros((2, 28), dtype=np.float32)
    joints[1, :14] = np.linspace(-0.4, 0.7, 14)
    action = kin.convert(joints, NAMES)
    np.testing.assert_allclose(action[0, 10:28], action[1, 10:28], atol=1e-7)
    np.testing.assert_allclose(action[0, 40:58], action[1, 40:58], atol=1e-7)
    assert not np.allclose(action[0, 80:83], action[1, 80:83])
    assert np.all(action[:, 98:] == 0)
    assert np.all(action[:, 22:28] == 0)
    assert np.all(action[:, 52:58] == 0)


def test_isolated_telemetry_repair_preserves_raw():
    raw = np.zeros((5, 28), dtype=np.float32)
    raw[1, 26] = 0.2
    raw[2, 26] = 428.8948
    raw[3, 26] = 0.4
    result, mask = repair_isolated_telemetry(raw)
    assert np.isclose(result[2, 26], 0.3)
    assert raw[2, 26] > 400
    assert mask.sum() == 1 and mask[2, 26]
    np.testing.assert_array_equal(result[~mask], raw[~mask])


@pytest.mark.parametrize("indices", [[0], [4], [1, 2]])
def test_unrepairable_telemetry_fails(indices):
    raw = np.zeros((5, 28), dtype=np.float32)
    raw[indices, 0] = 100
    with pytest.raises(ValueError, match="Non-isolated"):
        repair_isolated_telemetry(raw)
