"""Simultaneous Cartesian control for adapters that explicitly support it."""

from .runtime import ToolRegistry, schema


class MultiEffectorRegistry(ToolRegistry):
    def schemas(self, obs):
        specs = super().schemas(obs)
        if len(obs["arms"]) > 1 and hasattr(self.adapter, "move_effectors"):
            number = {"type": "number"}
            specs.append(
                schema(
                    "move_effectors",
                    "Move multiple end-effectors in the SAME native simulation steps. "
                    "World-frame absolute TCP positions and xyzw quaternions. Unspecified "
                    "arms hold position. No collision-free guarantee. Inspect per-arm residuals.",
                    {
                        "frame_id": {"type": "integer"},
                        "targets": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": len(obs["arms"]),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "arm": {"type": "string", "enum": list(obs["arms"])},
                                    "position": {
                                        "type": "array",
                                        "items": number,
                                        "minItems": 3,
                                        "maxItems": 3,
                                    },
                                    "quaternion_xyzw": {
                                        "type": "array",
                                        "items": number,
                                        "minItems": 4,
                                        "maxItems": 4,
                                    },
                                    "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                                },
                                "required": ["arm", "position", "quaternion_xyzw"],
                                "additionalProperties": False,
                            },
                        },
                        "steps": {"type": "integer", "minimum": 1, "maximum": 120},
                        "note": {"type": "string"},
                        "decision_note": {
                            "type": "string",
                            "description": "Brief observed-evidence decision summary in <THINK> tags.",
                        },
                    },
                    ["frame_id", "targets", "steps", "note", "decision_note"],
                )
            )
        if not any(a.get("gripper_actuated", True) for a in obs["arms"].values()):
            specs = [
                s for s in specs if s["function"]["name"] not in ("grasp_candidates", "set_gripper")
            ]
        return specs

    def _execute(self, name, args, obs):
        if name != "move_effectors":
            return super()._execute(name, args, obs)
        if args["frame_id"] != obs["frame_id"]:
            raise ValueError("Stale frame; reobserve")
        result = self.adapter.move_effectors(args["targets"], args["steps"])
        for target in args["targets"]:
            self.pose_targets.active.pop(target["arm"], None)
            self.waypoints.pop(target["arm"], None)
        return result
