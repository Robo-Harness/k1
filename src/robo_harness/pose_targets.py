"""Persistent, bounded SE(3) targets; no object/task knowledge or hidden planning."""

import numpy as np

from .motion import bounded_rotation, rotation
from .perception import finite_vector
from .motion_limits import bounded_translation, step_axis_limit


class PoseTargets:
    def __init__(self, motion):
        self.motion = motion
        self.goals = {}
        self.active = {}
        self.counter = 0

    def context(self, obs):
        result = {}
        for arm, key in self.active.items():
            if arm not in obs["arms"]:
                continue
            goal, state = self.goals[key], obs["arms"][arm]
            distance = float(np.linalg.norm(goal["xyz"] - state["xyz_world_m"]))
            angle = float(np.degrees((goal["rotation"] * rotation(state).inv()).magnitude()))
            result[arm] = {
                "pose_goal_id": key,
                "target_xyz_world_m": goal["xyz"].tolist(),
                "target_quaternion_xyzw": goal["rotation"].as_quat().tolist(),
                "remaining_distance_m": distance,
                "remaining_degrees": angle,
                "pose_reached": distance <= 0.003 and angle <= 2,
                "meaning": "Robot pose target only, not object arrival, attachment or task completion. Reobserve target features after motion.",
            }
        return result

    def step(self, obs, args):
        resolution = None
        if (
            getattr(self.motion, "advisory", False)
            and args.get("pose_goal_id")
            and "target_xyz_world_m" in args
        ):
            resolution = {
                "supplied_id": args["pose_goal_id"],
                "meaning": "Explicit XYZ defines a NEW target. The supplied label does not overwrite a saved target. Resume only the returned pose_goal_id next time.",
            }
            args = {k: v for k, v in args.items() if k != "pose_goal_id"}
        arm = args["arm"]
        limit = step_axis_limit(args, self.motion.delta_axis)
        angle_limit = float(args.get("max_degrees", 15))
        # Validate optional distance caps before registering a new persistent target.
        bounded_translation([0, 0, 0], limit, args.get("max_distance_m"))
        if not np.isfinite(angle_limit) or not 0 < angle_limit <= 15:
            raise ValueError("Rotation step limit must be in (0, 15]")
        if args.get("pose_goal_id"):
            if "target_xyz_world_m" in args or "target_quaternion_xyzw" in args:
                raise ValueError("Resume a pose_goal_id OR define a new target, not both")
            key = args["pose_goal_id"]
            goal = self.goals[key]
            if goal["arm"] != arm:
                raise ValueError("Pose target belongs to another arm")
        else:
            xyz = finite_vector(args["target_xyz_world_m"], 3)
            target = (
                rotation({"quaternion_xyzw": args["target_quaternion_xyzw"]})
                if "target_quaternion_xyzw" in args
                else rotation(obs["arms"][arm])
            )
            self.counter += 1
            key = f"T{self.counter}"
            goal = {"xyz": xyz.copy(), "rotation": target, "arm": arm}
            self.goals[key] = goal
        self.active[arm] = key
        # Parameter interpretation is reported, never silently changes an old goal.
        self.last_resolution = resolution
        # Retire only inactive targets; active targets of other arms remain resumable.
        for old in list(self.goals):
            if len(self.goals) <= 32:
                break
            if old not in self.active.values():
                del self.goals[old]
        state = obs["arms"][arm]
        delta = goal["xyz"] - state["xyz_world_m"]
        delta = bounded_translation(delta, limit, args.get("max_distance_m"))
        dr, _ = bounded_rotation(rotation(state), goal["rotation"], angle_limit)
        result = dict(
            self.motion.move(
                obs, arm, delta, dr, args.get("gripper"), args.get("steps", 12), check_sweep=True
            )
        )
        result.update(self.context(self.motion.adapter.observe())[arm])
        if resolution is not None:
            result["parameter_resolution"] = resolution
        return result
