"""Robot-only execution receipts shared by native adapters.

No task policy, object truth, reachability guess or grasp inference. Adapters
translate their own planner vocabulary; shared control reads only this contract.
"""

import numpy as np
from scipy.spatial.transform import Rotation


PLANNER_STATES = frozenset({"succeeded", "failed", "unknown", "not_applicable"})


def execution_receipt(*, planner_status, command_unit, start, end, physics_steps=None):
    if planner_status not in PLANNER_STATES:
        raise ValueError("Unknown normalized planner status")
    if command_unit not in ("native_trajectory_command", "control_step"):
        raise ValueError("Unknown command budget unit")
    if type(start) is not int or type(end) is not int or not 0 <= start <= end:
        raise ValueError("Invalid execution frame range")
    receipt = dict(
        version=1,
        planner_status=planner_status,
        command_unit=command_unit,
        commands_executed=end - start,
        start_frame_id=start,
        frame_id=end,
    )
    if physics_steps is not None:
        if type(physics_steps) is not int or physics_steps < 0:
            raise ValueError("Invalid physics step count")
        receipt["physics_steps_executed"] = physics_steps
    return receipt


def planner_failed(feedback):
    """Unknown / missing status is never converted to an explicit failure."""
    receipt = feedback.get("execution_contract", {})
    if receipt:
        if receipt.get("version") != 1:
            return False
        return receipt.get("planner_status") == "failed"
    # Preserve the explicit legacy adapter signal; not a heuristic text match.
    return feedback.get("status") == "ik_failure"


def robot_state_changed(before, after, *, commanded_grippers=None):
    """Conservative guard for failed-plan early return, including the other arm.

    A failed native plan can still advance physics and produce micrometer-scale
    settling. The 50 um / 0.005 degree deadbands are well below the 3 mm / 2 degree
    arrival tolerances; they apply ONLY after an explicit planner failure, not
    as a minimum speed for accepted motion. Missing evidence disables the stop.
    """
    if not before or set(before) != set(after):
        return True
    commanded_grippers = set(before) if commanded_grippers is None else set(commanded_grippers)
    try:
        for arm, old in before.items():
            new = after[arm]
            p0, p1 = [np.asarray(s["xyz_world_m"], dtype=float) for s in (old, new)]
            q0, q1 = [np.asarray(s["quaternion_xyzw"], dtype=float) for s in (old, new)]
            if (
                p0.shape != (3,)
                or p1.shape != (3,)
                or q0.shape != (4,)
                or q1.shape != (4,)
                or not all(np.isfinite(v).all() for v in (p0, p1, q0, q1))
            ):
                return True
            if np.linalg.norm(p1 - p0) > 5e-5:
                return True
            if (
                np.degrees((Rotation.from_quat(q1) * Rotation.from_quat(q0).inv()).magnitude())
                > 0.005
            ):
                return True
            if arm in commanded_grippers and (
                old.get("gripper_actuated", True) or new.get("gripper_actuated", True)
            ):
                widths = [float(s["gripper_opening_m"]) for s in (old, new)]
                if not np.isfinite(widths).all() or abs(widths[1] - widths[0]) > 5e-5:
                    return True
    except (KeyError, TypeError, ValueError):
        return True
    return False
