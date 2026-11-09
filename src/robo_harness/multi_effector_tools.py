"""Capability-based simultaneous pose tools. No robot/task names or policy scripts."""

import numpy as np

from .control_reference import resolve_targets, pad_center_local
from .grasp_observation import surface_pad_relation


def install(runtime):
    base = runtime.ToolRegistry

    class Registry(base):
        single_motion_feedback = True

        def schemas(self, obs):
            specs = super().schemas(obs)
            if len(obs["arms"]) < 2 or not callable(getattr(self.adapter, "move_effectors", None)):
                return specs
            num = {"type": "number"}
            vec = {"type": "array", "items": num, "minItems": 3, "maxItems": 3}
            specs.append(
                runtime.schema(
                    "move_effectors",
                    "Move selected end-effectors simultaneously to absolute world poses; unspecified arms hold. "
                    "position_reference=native_eef uses the adapter TCP; pad_center uses the calibrated pad midpoint. "
                    "Orientation ALWAYS describes native TCP axes. Candidate TCP coordinates already use native_eef. "
                    "steps caps native control commands, not physics steps. No path planning, collision or attachment "
                    "guarantee; inspect actual residuals and both arms. Same end poses do not constrain relative motion along the path.",
                    {
                        "frame_id": {"type": "integer"},
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": len(obs["arms"]),
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "arm": {"type": "string", "enum": list(obs["arms"])},
                                    "position": vec,
                                    "quaternion_xyzw": {
                                        "type": "array",
                                        "items": num,
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "position_reference": {
                                        "type": "string",
                                        "enum": ["native_eef", "pad_center"],
                                    },
                                    "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["arm", "position", "quaternion_xyzw"],
                            },
                        },
                        "steps": {"type": "integer", "minimum": 1, "maximum": 120},
                        "decision_note": {
                            "type": "string",
                            "description": "Brief current evidence and intended observable change.",
                        },
                    },
                    ["frame_id", "targets", "steps", "decision_note"],
                )
            )
            if any(s.get("geometry", {}).get("calibration_verified") for s in obs["arms"].values()):
                specs.append(
                    runtime.schema(
                        "inspect_grasp_geometry",
                        "Read-only signed ranges of a CURRENT measured surface relative to calibrated pads. "
                        "Not a grasp score, object identity or attachment verdict.",
                        {
                            "frame_id": {"type": "integer"},
                            "region_id": {"type": "string"},
                            "arm": {
                                "type": "string",
                                "enum": [
                                    a
                                    for a, s in obs["arms"].items()
                                    if s.get("geometry", {}).get("calibration_verified")
                                ],
                            },
                            "decision_note": {"type": "string"},
                        },
                        ["frame_id", "region_id", "arm", "decision_note"],
                    )
                )
            return specs

        def _execute(self, name, args, obs):
            if name not in ("move_effectors", "inspect_grasp_geometry"):
                return super()._execute(name, args, obs)
            if type(args.get("frame_id")) is not int or args["frame_id"] != obs["frame_id"]:
                raise ValueError("Stale frame; reobserve")
            self.crop = None
            self.retrieved_history_images = []
            if name == "inspect_grasp_geometry":
                summary = self.regions.summary(args["region_id"], obs)
                if summary["status"] != "current_visible_surface_estimate":
                    return dict(available=False, reason="Region is not a current measured surface")
                return dict(
                    region_id=args["region_id"],
                    frame_id=obs["frame_id"],
                    **surface_pad_relation(
                        self.regions.entries[args["region_id"]]["xyz"], obs["arms"][args["arm"]]
                    ),
                )
            resolved = resolve_targets(obs, args["targets"])
            result = self.adapter.move_effectors(resolved, args["steps"])
            current = self.adapter.observe()
            self.motion.feedback.clear()
            for target in resolved:
                arm = target["arm"]
                self.pose_targets.record_external(
                    arm, target["position"], target["quaternion_xyzw"]
                )
                self.waypoints[arm] = target["position"]
                self.motion.last_goal.pop(arm, None)
                self.motion.low.pop(arm, None)
                self.motion._record(arm, current["arms"][arm])
                ref = target["control_reference"]
                actual = np.asarray(current["arms"][arm]["xyz_world_m"])
                if ref["kind"] == "pad_center":
                    actual = actual + np.asarray(
                        current["arms"][arm]["axes_world"]
                    ) @ pad_center_local(current["arms"][arm])
                result["arms"][arm]["control_reference"] = ref
                result["arms"][arm]["control_point_residual_m"] = (
                    actual - ref["requested_position_world_m"]
                ).tolist()
            result.update(
                frame_id=current["frame_id"],
                resolved_targets=resolved,
                active_pose_targets=self.pose_targets.context(current),
                action_record=dict(
                    tool=name, start_frame=obs["frame_id"], end_frame=current["frame_id"]
                ),
            )
            return result

    runtime.ToolRegistry = Registry
