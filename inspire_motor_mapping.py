"""Inspire-hand motor command (0-1000) -> URDF dof_pos (rad).

Loads the precomputed lookup `cmd_to_dofpos.csv` (1001 rows: cmd + 12 URDF joint
angles in radians, referenced so cmd=1000 OPEN -> 0). Numpy-only, no pandas/xls,
so it runs on the deploy host.

Source of the table: human_policy/inspirehand/关节角与0-1000 对应关系.xls
    dof_pos(cmd) = sign * deg2rad(angle(cmd) - angle(cmd=1000))

12 URDF joints (right hand; left mirrors). 6 actuated motors drive them via the
documented coupling: each finger motor -> proximal + intermediate (P2); thumb-bend
-> pitch + intermediate(pad) + distal(tip); thumb-rot -> proximal_yaw.
"""
import os
import numpy as np

CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inspire_motor_dof_lut.csv")

JOINT_NAMES = [
    "R_thumb_proximal_yaw", "R_thumb_proximal_pitch", "R_thumb_intermediate", "R_thumb_distal",
    "R_index_proximal", "R_index_intermediate",
    "R_middle_proximal", "R_middle_intermediate",
    "R_ring_proximal", "R_ring_intermediate",
    "R_pinky_proximal", "R_pinky_intermediate",
]
# angleSet motor order [pinky, ring, middle, index, thumb_bend, thumb_rot] -> driven joints
MOTOR_ORDER = ["pinky", "ring", "middle", "index", "thumb_bend", "thumb_rot"]
MOTOR_JOINTS = {
    "pinky": ["R_pinky_proximal", "R_pinky_intermediate"],
    "ring": ["R_ring_proximal", "R_ring_intermediate"],
    "middle": ["R_middle_proximal", "R_middle_intermediate"],
    "index": ["R_index_proximal", "R_index_intermediate"],
    "thumb_bend": ["R_thumb_proximal_pitch", "R_thumb_intermediate", "R_thumb_distal"],
    "thumb_rot": ["R_thumb_proximal_yaw"],
}

_data = np.loadtxt(CSV, delimiter=",", skiprows=1)   # (1001, 13)
CMD_GRID = _data[:, 0]                                # 0..1000
DOF_LUT = _data[:, 1:]                                # (1001, 12) rad
_JIDX = {n: i for i, n in enumerate(JOINT_NAMES)}


def cmd6_to_dofpos12(cmd6):
    """6 motor cmd [pinky,ring,middle,index,thumb_bend,thumb_rot] (0-1000)
    -> 12 URDF dof_pos (rad), in JOINT_NAMES order."""
    cmd6 = np.clip(np.asarray(cmd6, float), 0, 1000)
    motor_q = dict(zip(MOTOR_ORDER, cmd6))
    out = np.zeros(len(JOINT_NAMES))
    for motor, joints in MOTOR_JOINTS.items():
        for jn in joints:
            j = _JIDX[jn]
            out[j] = np.interp(motor_q[motor], CMD_GRID, DOF_LUT[:, j])
    return out


def cmd_to_dofpos_single(cmd, joint_name):
    """Scalar: one motor cmd (0-1000) -> that joint's dof_pos (rad)."""
    j = _JIDX[joint_name]
    return float(np.interp(np.clip(cmd, 0, 1000), CMD_GRID, DOF_LUT[:, j]))


if __name__ == "__main__":
    for tag, c in [("open", 1000), ("close", 0), ("half", 500)]:
        print(f"{tag:5s} cmd={c}: dofpos12 = {np.round(cmd6_to_dofpos12([c]*6), 3)}")
