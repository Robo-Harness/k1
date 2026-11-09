"""Robot-only measurement declarations; no scene objects or success predicates."""

import numpy as np


def mujoco_gripper_measurement(sim, gripper):
    if not gripper.dof:
        return dict(valid=False, reason="gripper_not_actuated", source="unavailable", unit=None)
    joints = list(gripper.joints)
    ids = [sim.model.joint_name2id(name) for name in joints]
    # mjJNT_SLIDE == 2. Summing hinge angles and calling them meters is invalid.
    if not ids or any(int(sim.model.jnt_type[i]) != 2 for i in ids):
        return dict(
            valid=False,
            reason="metric_gap_calibration_unavailable",
            source="unavailable",
            unit=None,
        )
    ranges = np.asarray([sim.model.jnt_range[i] for i in ids], dtype=float)
    low = float(sum(0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) for lo, hi in ranges))
    high = float(np.max(np.abs(ranges), axis=1).sum())
    value = float(sum(abs(float(sim.data.get_joint_qpos(j))) for j in joints))
    return dict(
        value=value,
        unit="m",
        source="measured_prismatic_finger_joints",
        limits=[low, high],
        valid=True,
    )
