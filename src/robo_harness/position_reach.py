"""One actor call, one fixed absolute target, multiple native servo ticks.

No Cartesian chunking, delta_axis limit, inverse-kinematics oracle or path planner.
The adapter must implement direct absolute-position servoing through real dynamics.
"""

import numpy as np

from .episode_history import nonnegative_integer
from .motion import rotation
from .motion_limits import axis_limit
from .perception import finite_vector


def reach_config(max_steps=120, stall_steps=24, tolerance_m=0.003):
    nonnegative_integer(max_steps, "move_toward_max_steps")
    nonnegative_integer(stall_steps, "move_toward_stall_steps")
    if max_steps == 0:
        raise ValueError("move_toward_max_steps must be positive")
    return {
        "max_native_steps": max_steps,
        "stall_native_steps": stall_steps,
        "position_tolerance_m": axis_limit(tolerance_m),
        "orientation_tolerance_deg": 2.0,
        "settled_native_frames": 3,
        "progress_distance_m": 0.0005,
        "progress_angle_deg": 0.5,
    }


def reach_position(motion, obs, args, config):
    adapter = motion.adapter
    servo = getattr(adapter, "move_to_position", None)
    if not callable(servo):
        raise ValueError("Adapter does not support absolute position servo; use small-step tools")
    unsupported = set(args).intersection({"delta_axis", "max_distance_m", "steps"})
    if unsupported:
        raise ValueError(
            "move_toward reaches an absolute target; remove unsupported step-limit fields: "
            + ", ".join(sorted(unsupported))
        )
    arm = args["arm"]
    target = finite_vector(args["target_xyz_world_m"], 3)
    state = obs["arms"][arm]
    start = finite_vector(state["xyz_world_m"], 3)
    orientation = rotation(state)
    gripper = args.get("gripper")
    if gripper is not None and (not np.isfinite(gripper) or not 0 <= gripper <= 1):
        raise ValueError("gripper must be 0 closed .. 1 open")
    begin = int(adapter.frame)
    current = obs
    distance = float(np.linalg.norm(target - start))
    if not np.isfinite(distance):
        raise ValueError("Target displacement cannot be represented as a finite distance")
    angle = 0.0
    progress_distance, progress_angle, last_progress = distance, angle, begin
    settled = 0
    reason = "native_step_budget"
    native_feedback = {}
    ik_failure_reported = False
    motion._record(arm, state)
    while adapter.frame - begin < config["max_native_steps"]:
        if adapter.frame >= adapter.horizon:
            reason = "episode_step_budget"
            break
        if adapter.success():
            reason = "native_success"
            break
        previous_frame = adapter.frame
        # Full target every tick. Native action/velocity limits still apply.
        native_feedback = servo(arm, target.tolist(), orientation.as_quat().tolist(), gripper, 1)
        ik_failure_reported |= native_feedback.get("status") == "ik_failure"
        current = adapter.observe()
        after = current["arms"][arm]
        distance = float(np.linalg.norm(target - after["xyz_world_m"]))
        angle = float(np.degrees((orientation * rotation(after).inv()).magnitude()))
        motion._record(arm, after)
        if adapter.frame <= previous_frame:
            reason = "adapter_did_not_advance"
            break
        within = (
            distance <= config["position_tolerance_m"]
            and angle <= config["orientation_tolerance_deg"]
        )
        settled = settled + 1 if within else 0
        if settled >= config["settled_native_frames"]:
            reason = "target_reached"
            break
        if adapter.success():
            reason = "native_success"
            break
        if adapter.frame >= adapter.horizon:
            reason = "episode_step_budget"
            break
        if (
            progress_distance - distance >= config["progress_distance_m"]
            or progress_angle - angle >= config["progress_angle_deg"]
        ):
            last_progress = adapter.frame
            progress_distance, progress_angle = distance, angle
        if (
            not within
            and config["stall_native_steps"]
            and adapter.frame - last_progress >= config["stall_native_steps"]
        ):
            reason = "no_measurable_progress"
            break
    actual = finite_vector(current["arms"][arm]["xyz_world_m"], 3)
    # Keep recovery records and old orientation anchors consistent with actual motion.
    for goal in motion.goals.values():
        if goal["arm"] == arm:
            goal["xyz"] = actual.copy()
    motion.retreat_goals.pop(arm, None)
    motion.low.pop(arm, None)
    result = {
        "execution_mode": "absolute_target_native_servo",
        "delta_axis_applies": False,
        "waypoint_xyz_world_m": target.tolist(),
        "held_quaternion_xyzw": orientation.as_quat().tolist(),
        "actual_translation_m": (actual - start).tolist(),
        "remaining_delta_world_m": (target - actual).tolist(),
        "remaining_distance_m": distance,
        "remaining_degrees": angle,
        "target_reached": reason == "target_reached",
        "position_within_tolerance": distance <= config["position_tolerance_m"],
        "stop_reason": reason,
        "native_steps": int(adapter.frame) - begin,
        "start_frame_id": begin,
        "frame_id": int(adapter.frame),
        "execution_limits": dict(config),
        "native_ik_failure_reported": ik_failure_reported,
        "native_last_step_diagnostic": native_feedback,
        "meaning": "Measured robot arrival only, not object/task success. No IK feasibility or collision-free path certificate. "
        "A stall ends this call, not future retries; reobserve and choose the next action.",
    }
    motion.feedback[arm] = result
    # The last millimeter servo tick is not the whole transit's success/failure.
    # Present the aggregate outcome, retaining last-tick evidence for offline audit.
    adapter.last_feedback = result
    return result
