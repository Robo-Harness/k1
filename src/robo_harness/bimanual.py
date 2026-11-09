"""Robot-only multi-effector evidence. No object truth, task policy or safety gate."""

from contextlib import contextmanager
from itertools import combinations

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import execution_report


BIMANUAL_GUIDANCE = """
Multi-arm coordination: choose each arm's role from current evidence (manipulate,
support/hold, observe, or idle); record role changes and unverified grasp/transfer
claims in task progress when useful. These are optional plans, not verified facts.
If contact is occluded and another arm is free, consider positioning its existing
wrist camera for a complementary view, then keeping that observer arm steady while
the manipulating arm takes small steps. Verify that the new view actually helps.
Do not abandon an existing support/grasp just to obtain a view. Camera movement can
introduce collision or occlusion; no extra camera or guaranteed viewing pose exists.
For transfer of support, closing the receiving gripper is not proof it holds the
object. Observe contact/object response and consider gradual release before moving
the original support away. Choose the sequence yourself; no mandatory handover script.
For shared-object motion, simultaneous arm commands do NOT guarantee preservation of
relative pose along the path. Consider small compatible moves and inspect both arms'
actual pose changes. End-point relative-pose evidence is not a trajectory certificate.
""".strip()


def bimanual_guidance(observer_hand_advice=True):
    """Ablate only the optional observer-hand paragraph; no tool/view changes."""
    if type(observer_hand_advice) is not bool:
        raise ValueError("observer_hand_advice must be boolean")
    if observer_hand_advice:
        return BIMANUAL_GUIDANCE
    start = BIMANUAL_GUIDANCE.index("If contact is occluded")
    end = BIMANUAL_GUIDANCE.index("For transfer of support")
    return BIMANUAL_GUIDANCE[:start] + BIMANUAL_GUIDANCE[end:]


def arm_execution_evidence(
    start,
    goal,
    actual,
    *,
    requested,
    planner_statuses=(),
    position_tolerance_m=0.003,
    orientation_tolerance_deg=2.0,
):
    """Use absolute residuals for hold/tiny moves; expose raw planning outcome."""
    report = execution_report(start["xyz_world_m"], goal[0], actual["xyz_world_m"])
    start_q = Rotation.from_quat(start["quaternion_xyzw"])
    actual_q = Rotation.from_quat(actual["quaternion_xyzw"])
    goal_q = goal[1]
    distance = float(np.linalg.norm(report["actual_minus_target_m"]))
    angle = float(np.degrees((goal_q * actual_q.inv()).magnitude()))
    requested_distance = float(np.linalg.norm(report["commanded_translation_m"]))
    requested_angle = float(np.degrees((goal_q * start_q.inv()).magnitude()))
    small = requested_distance <= position_tolerance_m
    holding = small and requested_angle <= orientation_tolerance_deg
    within = distance <= position_tolerance_m and angle <= orientation_tolerance_deg
    # A 0.2 mm residual must not look worse than a 20 cm miss because of a ratio.
    if small:
        report["axial_progress_ratio"] = None
    report["status"] = (
        ("held" if holding else "target_reached")
        if within
        else ("hold_drift" if holding else "target_not_reached")
    )
    statuses = list(planner_statuses)
    report.update(
        requested=requested,
        intent="hold" if holding else "move",
        position_residual_m=distance,
        rotation_residual_deg=angle,
        target_reached=within,
        position_tolerance_m=position_tolerance_m,
        orientation_tolerance_deg=orientation_tolerance_deg,
        planning=dict(
            available=bool(statuses),
            calls=len(statuses),
            last_native_status=statuses[-1] if statuses else None,
            failure_count=sum(s != "Success" for s in statuses),
        ),
        meaning="Robot kinematics and native planner status only. A planner failure is not "
        "a diagnosed collision/IK cause; jaw closure and TCP arrival do not prove grasp.",
    )
    return report


def relative_pose_evidence(before, after):
    """Observed final relative transform change; not a claim of rigid attachment."""
    pairs = []
    for anchor, other in combinations(sorted(before), 2):

        def relative(states):
            a, b = states[anchor], states[other]
            qa = Rotation.from_quat(a["quaternion_xyzw"])
            qb = Rotation.from_quat(b["quaternion_xyzw"])
            delta = np.asarray(b["xyz_world_m"]) - a["xyz_world_m"]
            return qa.inv().apply(delta), qa.inv() * qb

        p0, q0 = relative(before)
        p1, q1 = relative(after)
        pairs.append(
            dict(
                anchor_arm=anchor,
                other_arm=other,
                translation_change_in_anchor_m=(p1 - p0).tolist(),
                translation_change_norm_m=float(np.linalg.norm(p1 - p0)),
                orientation_change_deg=float(np.degrees((q1 * q0.inv()).magnitude())),
            )
        )
    return dict(
        pairs=pairs,
        scope="observed start/end only; path not constrained or certified",
        meaning="Relative TCP motion, not object slip/contact evidence or a risk score.",
    )


@contextmanager
def capture_native_planning(robot, arms):
    """Observe planner receipts without changing inputs, return values or exceptions.

    Temporary instance wrappers are restored even when native execution fails.
    Only native status strings are retained; no paths/object geometry go to the VLM.
    """
    originals = {}
    receipts = {arm: [] for arm in arms}
    try:
        for arm in arms:
            name = f"{arm}_plan_path"
            original = getattr(robot, name, None)
            if original is None:
                continue
            originals[name] = (original, name in vars(robot))

            def wrapped(*args, _original=original, _arm=arm, **kwargs):
                result = _original(*args, **kwargs)
                status = result.get("status") if isinstance(result, dict) else None
                if status is not None:
                    receipts[_arm].append(str(status))
                return result

            setattr(robot, name, wrapped)
        yield receipts
    finally:
        for name, (original, was_instance_attribute) in originals.items():
            if was_instance_attribute:
                setattr(robot, name, original)
            else:
                delattr(robot, name)
