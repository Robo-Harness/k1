"""Persistent jaw-command/gap receipts; never infer hidden contacts or grasps."""

from collections import deque

import numpy as np


class GripperEvidence:
    def __init__(self):
        self.arms = {}

    def update(self, obs):
        for arm, state in obs["arms"].items():
            gap, command = state.get("gripper_opening_m"), state.get("gripper_command_open")
            if gap is None or command is None:
                continue
            data = self.arms.setdefault(
                arm,
                {
                    "samples": deque(maxlen=3),
                    "events": deque(maxlen=8),
                    "reference": None,
                    "command": None,
                },
            )
            if data["samples"] and data["samples"][-1][0] == obs["frame_id"]:
                continue
            closed = command < 0.5
            if data["command"] is None or closed != data["command"]:
                data["events"].append(
                    {"frame_id": obs["frame_id"], "command": "close" if closed else "open"}
                )
                data["reference"] = None
                data["samples"].clear()
            data["command"] = closed
            data["samples"].append((obs["frame_id"], gap))
            samples = list(data["samples"])
            settled = (
                len(samples) == 3
                and samples[-1][0] - samples[0][0] == 2
                and np.ptp([p[1] for p in samples]) < 0.0005
            )
            if closed and settled:
                if data["reference"] is None or gap > data["reference"]:
                    data["reference"] = gap
            limits = state.get("gripper_limits_m", [0, 0.08])
            drop = max(0.0, (data["reference"] or gap) - gap)
            significant = (
                closed
                and data["reference"] is not None
                and drop > max(0.004, 0.5 * (data["reference"] - limits[0]))
            )
            data["current"] = {
                "command_events": list(data["events"]),
                "largest_settled_gap_since_close_m": data["reference"],
                "gap_decrease_since_reference_m": drop,
                "large_gap_decrease_while_command_still_closed": bool(significant),
                "meaning": "Command/gap history only. A large decrease can indicate lost or changed contact, deforming material, or a changing grasp; it is NOT a slip detector. Pause assumptions about carrying and check target motion/identity in current views. Nonzero gap does not prove attachment.",
            }

    def context(self):
        return {arm: data["current"] for arm, data in self.arms.items() if "current" in data}
