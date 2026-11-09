"""Explicit native waiting versus a true empty command; no task decisions."""

import numpy as np

from .perception import finite_vector


def is_zero_relative(args, obs):
    if args["arm"] not in obs["arms"]:
        raise ValueError("Unknown arm")
    if args.get("frame") not in ("world", "gripper"):
        raise ValueError("Specify world or gripper coordinate frame")
    delta = finite_vector(args["delta"], 3)
    rotation = finite_vector(args.get("rotation_deg", [0, 0, 0]), 3)
    if np.any(delta != 0) or np.any(rotation != 0):
        return False
    grip = args.get("gripper")
    if grip is None:
        return True
    if isinstance(grip, bool) or not np.isfinite(grip) or not 0 <= grip <= 1:
        raise ValueError("Gripper must be a finite value in 0..1")
    previous = obs["arms"][args["arm"]].get("gripper_command_open")
    return previous is not None and grip == previous


def hold_current(adapter, obs, steps):
    if type(steps) is not int or not 1 <= steps <= 120:
        raise ValueError("hold steps must be an integer in 1..120")
    targets = [
        dict(
            arm=arm,
            position=s["xyz_world_m"],
            quaternion_xyzw=s["quaternion_xyzw"],
            gripper=s.get("gripper_command_open"),
        )
        for arm, s in obs["arms"].items()
    ]
    if callable(getattr(adapter, "hold_effectors", None)):
        execution = adapter.hold_effectors(targets, steps=steps)
    elif callable(getattr(adapter, "move_effectors", None)):
        execution = adapter.move_effectors(targets, steps=steps)
    elif len(targets) == 1 and callable(getattr(adapter, "move_to_position", None)):
        t = targets[0]
        execution = adapter.move_to_position(
            t["arm"], t["position"], t["quaternion_xyzw"], gripper=t["gripper"], steps=steps
        )
    else:
        raise ValueError("Adapter cannot hold all effectors synchronously; no command sent")
    current = adapter.observe()
    elapsed = current["frame_id"] - obs["frame_id"]
    return dict(
        status="hold_completed" if elapsed == steps else "hold_stopped",
        frame_id=current["frame_id"],
        native_steps=current["frame_id"] - obs["frame_id"],
        requested_native_steps=steps,
        execution=execution,
        meaning="Native time advanced with current TCP targets and unchanged gripper commands. "
        "Actual objects and robot may still move; inspect new observations.",
    )
