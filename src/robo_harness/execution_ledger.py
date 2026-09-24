"""Bounded factual decision receipts, independent of task names and simulator state."""

from collections import Counter, deque
from copy import deepcopy


class ExecutionLedger:
    def __init__(self):
        self.counts = Counter()
        self.recent = deque(maxlen=8)
        self.arms = {}

    def record(self, call_id, name, args, result, before, after):
        self.counts[name] += 1
        frame = before["frame_id"]
        receipt = {
            "call_id": call_id,
            "tool": name,
            "from_frame_id": frame,
            "to_frame_id": after["frame_id"],
            "error": result.get("error"),
        }
        # These strings describe a request, never an observed physical outcome.
        for key in ("region_id", "source_region_id", "target_region_id", "pose_goal_id"):
            if key in args:
                receipt[key] = args[key]
        if "pose_goal_id" in result:
            receipt["pose_goal_id"] = result["pose_goal_id"]
        self.recent.append(receipt)
        for arm, state in after["arms"].items():
            data = self.arms.setdefault(
                arm,
                {
                    "last_observed_command_transition": None,
                    "last_candidate_request": None,
                    "last_successful_candidate_receipt": None,
                    "candidate_requests_since_command_transition": 0,
                    "calls_since_command_transition": 0,
                    "observed_command": None,
                },
            )
            data["calls_since_command_transition"] += 1
            command = state.get("gripper_command_open")
            prior = before["arms"].get(arm, {}).get("gripper_command_open")
            if command is not None:
                closed = command < 0.5
                # Initialization is not a transition or an executed close.
                if prior is not None and closed != (prior < 0.5):
                    data["last_observed_command_transition"] = {
                        "call_id": call_id,
                        "frame_id": after["frame_id"],
                        "command": "closed" if closed else "open",
                    }
                    data["candidate_requests_since_command_transition"] = 0
                    data["calls_since_command_transition"] = 0
                data["observed_command"] = "closed" if closed else "open"
            if name == "grasp_candidates" and args.get("arm") == arm:
                data["candidate_requests_since_command_transition"] += 1
                data["last_candidate_request"] = {
                    "call_id": call_id,
                    "frame_id": frame,
                    "region_id": args.get("region_id"),
                    "returned_candidates": len(result.get("candidates", [])),
                    "error": result.get("error"),
                }
                if not result.get("error") and result.get("candidates"):
                    fields = (
                        "candidate_id",
                        "candidate_tcp_world_m",
                        "pregrasp_tcp_world_m",
                        "target_quaternion_xyzw",
                        "visibility_score",
                        "learned_confidence_not_success_probability",
                        "rotation_from_current_deg",
                    )
                    data["last_successful_candidate_receipt"] = {
                        "call_id": call_id,
                        "frame_id": frame,
                        "region_id": args.get("region_id"),
                        "historical_not_current": True,
                        "candidates": [
                            {k: deepcopy(c[k]) for k in fields if k in c}
                            for c in result["candidates"][:3]
                        ],
                    }

    def context(self):
        return deepcopy(
            {
                "tool_request_counts": dict(self.counts),
                "recent_receipts": list(self.recent),
                "arms": self.arms,
                "meaning": (
                    "Historical command/tool receipts, not object identity, contact or success. "
                    "A candidate request is NOT a grasp; a command transition is NOT attachment. "
                    "Counts include failed requests and do not force a next action. "
                    "Saved candidate poses are HISTORICAL hypotheses, never automatically tracked; "
                    "candidate IDs are scoped to their receipt frame and call. "
                    "Check CURRENT evidence before resuming any historical target."
                ),
            }
        )
