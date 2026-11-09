"""Portable tool registry + VLM loop; no task policies or simulator handles in actor input."""

import base64
import json
import time
from pathlib import Path

import cv2
import httpx
import numpy as np

from .geometry import unproject
from .roi import rectangle as roi_rectangle, schema as roi_schema, receipt as roi_receipt
from .motion import MotionController
from .pose_targets import PoseTargets
from .reference_alignment import measured_region_alignment, qualify_carry_alignment
from .tracked_points import TrackedPoints
from .regions import Regions
from .grasp_geometry import candidates as grasp_candidates
from .grasp_backends import learned_candidates
from .grasp_visualization import candidate_panel
from .grasp_presentation import present_candidates
from .interaction_evidence import InteractionEvidence
from .gripper_monitor import GripperMonitor
from .perception import GenericTools, finite_vector, pixel_points, tool_schemas
from .episode_history import EpisodeHistory, nonnegative_integer, validate_image_history_selection
from .task_progress import TaskProgress, memory_schemas
from .motion_limits import axis_limit
from .position_reach import reach_config, reach_position
from .bimanual import bimanual_guidance
from .completion_protocol import bind_protocol, resolve_mode, policy_success, native_terminated
from .output_budget import OutputBudget
from .point_targets import named_schema, point_schema, resolve as resolve_point_target
from .observation_contract import validate_reference

HARNESS_VERSION = "k1-v1.1"


def schema(name, description, properties, required):
    if not set(required) <= set(properties):
        raise ValueError(f"{name}: required tool parameters must be declared in properties")
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


def image_content(rgb):
    ok, encoded = cv2.imencode(
        ".jpg", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    if not ok:
        raise ValueError("Cannot encode camera")
    return {
        "type": "image_url",
        "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(encoded).decode()},
    }


def measurement_overlay(rgb, camera_name, frame_id, points):
    """Visual receipt of pixel selection; never modifies raw sensor/tracker images."""
    annotated = np.asarray(rgb).copy()
    h, w = annotated.shape[:2]
    for point in points[-10:]:
        if point.get("camera") != camera_name or point.get("frame_id") != frame_id:
            continue
        u, v = (int(x) for x in point["pixel"])
        if not (0 <= u < w and 0 <= v < h):
            continue
        cv2.drawMarker(annotated, (u, v), (255, 50, 50), cv2.MARKER_CROSS, 11, 1)
        label = str(point["id"])
        origin = (min(u + 5, max(0, w - 50)), max(12, v - 5))
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        cv2.putText(annotated, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
    return annotated


def history_window(history, limit=16, rounds=None):
    """Keep complete user / assistant-call / tool exchanges across truncation."""
    if rounds is not None:
        nonnegative_integer(rounds, "history_rounds")
        if rounds == 0:
            return []
        starts = [
            i
            for i, m in enumerate(history[:-1])
            if m.get("role") == "user" and history[i + 1].get("tool_calls")
        ]
        return history[starts[-rounds] :] if len(starts) >= rounds else history[:]
    window = history[-limit:]
    while window and window[0].get("role") != "user":
        window = window[1:]
    return window


class ToolRegistry:
    def __init__(
        self,
        adapter,
        motion_tools=True,
        arrival_guard=True,
        tracking_backend="lk",
        tracking_device="cpu",
        region_tools=False,
        grasp_provider=None,
        interaction_feedback=False,
        completion_feedback=False,
        advisory_controls=False,
        delta_axis=0.03,
        move_toward_max_steps=120,
        move_toward_stall_steps=24,
        move_toward_tolerance_m=0.003,
        completion_mode=None,
    ):
        self.reach_config = reach_config(
            move_toward_max_steps, move_toward_stall_steps, move_toward_tolerance_m
        )
        self.adapter = adapter
        self.perception = GenericTools()
        self.points = []
        self.memory = {}
        self.episode_history = None
        self.task_progress = None
        self.current_call_id = None
        self.retrieved_history_images = []
        self.crop = None
        self.crop_caption = "CURRENT enlarged crop (mapping in previous tool result)"
        self.advisory_controls = advisory_controls
        self.motion = MotionController(adapter, advisory=advisory_controls, delta_axis=delta_axis)
        self.pose_targets = PoseTargets(self.motion)
        self.motion_tools = motion_tools
        self.arrival_guard = arrival_guard
        self.waypoints = {}
        self.grasp_provider = grasp_provider
        self.interaction_feedback = interaction_feedback
        self.completion_protocol = bind_protocol(
            adapter, resolve_mode(completion_mode, completion_feedback)
        )
        self.completion_feedback = self.completion_protocol.mode == "feedback"
        self.completion_checks = []
        self.interaction_evidence = InteractionEvidence()
        self.gripper_monitor = GripperMonitor()
        self.tracked_points = TrackedPoints(backend=tracking_backend, device=tracking_device)
        self.regions = Regions(
            device=tracking_device, backend=tracking_backend, enabled=region_tools
        )
        if hasattr(adapter, "on_step"):
            # The core consumes every step for tracking and jaw settling. Legacy
            # evaluator-only adapters may otherwise call back with None.
            adapter.step_observations_required = True
            adapter.point_overlay = self.tracked_points.overlay
            previous_callback = adapter.on_step

            def on_step(obs, action):
                self.record_gripper(obs)
                self.tracked_points.update(obs)
                self.regions.update(obs)
                if previous_callback:
                    previous_callback(obs, action)

            adapter.on_step = on_step

    def record_gripper(self, obs):
        # Always-on proprioception, independent of optional visual interaction evidence.
        # This observer has NO access to robot actuation or motion stopping.
        self.gripper_monitor.update(obs)

    def gripper_context(self, obs):
        self.record_gripper(obs)
        return self.gripper_monitor.context()

    def arrival_context(self, obs):
        return {
            arm: {
                "target_xyz_world_m": target,
                "remaining_distance_m": float(
                    np.linalg.norm(np.asarray(target) - obs["arms"][arm]["xyz_world_m"])
                ),
                "meaning": "TCP goal only; not object position or task success",
            }
            for arm, target in self.waypoints.items()
        }

    def observe(self):
        obs = self.adapter.observe()
        self.record_gripper(obs)
        self.perception.update({"vision": obs["vision"]}, obs["frame_id"])
        self.tracked_points.update(obs)
        self.regions.update(obs, refresh=True)
        self.regions.refresh_cross_views(obs)
        if self.interaction_feedback:
            self.interaction_evidence.update(self.regions.entries, obs)
        return obs

    def schemas(self, obs):
        num = {"type": "number"}
        text = {"type": "string"}
        fid = {"type": "integer"}
        vec = {"type": "array", "items": num, "minItems": 3, "maxItems": 3}
        delta_vec = {
            "type": "array",
            "items": {
                "type": "number",
                "minimum": -self.motion.delta_axis,
                "maximum": self.motion.delta_axis,
            },
            "minItems": 3,
            "maxItems": 3,
        }
        uv = {"type": "array", "items": num, "minItems": 2, "maxItems": 2}
        camera = {"type": "string", "enum": list(obs["vision"])}
        space = {"type": "string", "enum": ["pixels", "normalized_01", "normalized_1000"]}
        specs = (
            self.regions.schemas(schema, obs["vision"])
            + tool_schemas(schema, obs["vision"])
            + [
                schema(
                    "grasp_candidates",
                    "Construct parallel-jaw POSE HYPOTHESES for a current slender region using visible RGB-D and calibrated hand geometry. Returns candidate IDs, quaternion, pregrasp and candidate TCP points. Evaluate each option independently from the original views and geometry; you may reject all candidates. No motion, object oracle or guaranteed grasp. Open jaws first; choose clearance and execute bounded primitives.",
                    {
                        "frame_id": fid,
                        "region_id": text,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                    },
                    ["frame_id", "region_id", "arm"],
                ),
                schema(
                    "set_gripper",
                    "Command gripper (0 close,1 open) while holding TCP pose. Intermediate values follow the arm's gripper_command_mode, NOT a guaranteed partial jaw width: position is a normalized target, binary thresholds at 0.5, directional closes below 0.5/opens above 0.5/holds at 0.5. Waits up to 24 native steps for measured opening to settle. Reports sourced metric gap OR normalized joint position; unavailable sensors stay unknown. Neither settling nor opening certifies attachment. Events are advisory, not motion interlocks.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                        "note": text,
                    },
                    ["frame_id", "arm", "gripper", "note"],
                ),
                schema(
                    "measure_depth",
                    "Measure selected visible surface in current calibrated RGB-D. Returns world point and delta from TCP, not a grasp pose.",
                    {"frame_id": fid, "camera": camera, "uv": uv, "coordinate_space": space},
                    ["frame_id", "camera", "uv", "coordinate_space"],
                ),
                schema(
                    "inspect_crop",
                    "Enlarge CURRENT original-image ROI; no physical camera movement.",
                    {
                        "frame_id": fid,
                        "camera": camera,
                        "roi": roi_schema(),
                    },
                    ["frame_id", "camera", "roi"],
                ),
                schema(
                    "move_relative",
                    f"RELATIVE INCREMENT: execute the supplied delta in meters in the selected world/gripper frame, NOT an absolute XYZ target. Each component independently satisfies abs(delta_i) <= {self.motion.delta_axis:g} m (delta_axis); diagonal length may be larger. Optional rotation_deg is a WORLD rotation-vector increment with total angle <=15deg. No persistent target. Optional gripper 0=closed,1=open. Use for direct visual-servo corrections; read actual result.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "delta": delta_vec,
                        "frame": {"type": "string", "enum": ["world", "gripper"]},
                        "rotation_deg": vec,
                        "gripper": num,
                        "steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        "note": text,
                    },
                    ["frame_id", "arm", "delta", "frame", "note"],
                ),
                schema(
                    "move_toward",
                    f"GO TO ABSOLUTE POSITION IN ONE TOOL CALL: supply target_xyz_world_m, NOT a relative delta. Send the FULL target to native control and continue internally up to {self.reach_config['max_native_steps']} native steps, holding the call's initial orientation. NOT limited by delta_axis and NOT a single 3cm chunk. Returns measured target_reached, residual and stop_reason; may stop without arrival on stall or budget. No IK feasibility or collision-free-path guarantee; no image-based replanning during execution. Use for deliberate transit with observed clearance; use move_relative/move_to_pose for small visual-servo/contact steps. Optional gripper command is applied from the start, not after arrival.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "target_xyz_world_m": named_schema(("x", "y", "z")),
                        "target_point": point_schema(),
                        "gripper": {"type": "number", "minimum": 0, "maximum": 1},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        "note": text,
                    },
                    ["frame_id", "arm", "note"],
                ),
                schema(
                    "rotate_toward",
                    "ABSOLUTE ORIENTATION ONLY: supply a target world quaternion (xyzw), or resume a returned R goal_id, NOT a relative rotation_deg increment. Execute ONE <=15deg rotation step while correcting drift toward the saved TCP position anchor; preserve gripper command. Reuse goal_id to keep the original orientation target. Returns actual rotation, remaining angle and position error. For changing position AND orientation use move_to_pose. Not a collision guarantee.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "target_quaternion_xyzw": {
                            "type": "array",
                            "items": num,
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "goal_id": text,
                        "max_degrees": {"type": "number", "exclusiveMinimum": 0, "maximum": 15},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        "note": text,
                    },
                    ["frame_id", "arm", "note"],
                ),
                schema(
                    "align_axis",
                    "Align a calibrated gripper approach/closing/lateral axis to a chosen WORLD direction with minimal twist. Executes one <=15deg step and saves an R orientation goal; continue via rotate_toward goal_id. Choose direction/clearance from evidence, not a task skill.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "axis": {"type": "string", "enum": ["approach", "closing", "lateral"]},
                        "direction_world": vec,
                        "max_degrees": {"type": "number", "exclusiveMinimum": 0, "maximum": 15},
                        "steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        "note": text,
                    },
                    ["frame_id", "arm", "axis", "direction_world", "note"],
                ),
                schema(
                    "retreat",
                    f"Take ONE step back toward an earlier actually observed pose, each world translation component <= delta_axis (default {self.motion.delta_axis:g} m), keeping gripper command unchanged. Continue until the current retreat waypoint is reached. Old path is NOT guaranteed safe: objects may have moved; check feedback. No random search or automatic release.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "delta_axis": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": self.motion.delta_axis,
                        },
                        "steps": {"type": "integer", "minimum": 1, "maximum": 12},
                        "note": text,
                    },
                    ["frame_id", "arm", "note"],
                ),
                schema(
                    "hold",
                    "Explicitly advance native simulation while holding ALL currently measured TCP poses and preserving gripper commands. Use to let physics settle or observe time evolution; this consumes native budget. Not a grasp, stability or collision guarantee. A zero move_relative is a no-op, not waiting.",
                    {
                        "frame_id": fid,
                        "steps": {"type": "integer", "minimum": 1, "maximum": 120},
                        "note": text,
                    },
                    ["frame_id", "steps", "note"],
                ),
                schema(
                    "read_guidance",
                    "Read optional general guidance about interpreting grasp, handover or motion evidence. No action or native steps. Not a task-specific recipe; current observations and decisions remain yours.",
                    {
                        "topic": {
                            "type": "string",
                            "enum": ["grasp_evidence", "handover_evidence", "motion_recovery"],
                        }
                    },
                    ["topic"],
                ),
                schema(
                    "remember",
                    "Save a short evidence-based working note; no success verification.",
                    {"key": text, "value": text},
                    ["key", "value"],
                ),
                schema(
                    "end_attempt",
                    "Stop this attempt without claiming task success, if you choose not to continue recovery. "
                    "Does not reset, retry, or alter native scoring. This is optional, not an automatic failure rule.",
                    {"reason": text},
                    ["reason"],
                ),
                schema(
                    "done",
                    "Claim completion; native success feedback may reject it and allow recovery. "
                    "Use end_attempt instead to stop without a success claim.",
                    {"summary": text},
                    ["summary"],
                ),
            ]
        )
        calibrated_arms = [
            arm for arm, state in obs["arms"].items() if state.get("geometry", {}).get("axes_local")
        ]
        for spec in specs:
            if spec["function"]["name"] == "align_axis":
                spec["function"]["parameters"]["properties"]["arm"]["enum"] = calibrated_arms
        specs = [
            spec for spec in specs if spec["function"]["name"] != "align_axis" or calibrated_arms
        ]
        if self.motion_tools:
            specs.append(
                schema(
                    "move_to_pose",
                    f"PERSISTENT ABSOLUTE POSE: supply world XYZ plus optional world quaternion to create a target, or resume a returned T pose_goal_id without XYZ/quaternion. Omitted quaternion snapshots current orientation ONCE; resuming keeps that original orientation rather than accumulated drift. Explicit XYZ creates a NEW target even if an ID is also supplied. Execute ONE straight translation step with each world component <= delta_axis (default {self.motion.delta_axis:g} m) and rotation <=15deg. No gripper change; use set_gripper separately. Prefer when both position and orientation must remain tied to one goal. Not path planning, grasp verification or object arrival; choose clearance and inspect residuals.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "pose_goal_id": text,
                        "target_xyz_world_m": vec,
                        "target_quaternion_xyzw": {
                            "type": "array",
                            "items": num,
                            "minItems": 4,
                            "maxItems": 4,
                        },
                        "delta_axis": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": self.motion.delta_axis,
                        },
                        "max_degrees": {"type": "number", "exclusiveMinimum": 0, "maximum": 15},
                        "note": text,
                    },
                    ["frame_id", "arm", "note"],
                )
            )
        if self.regions.enabled:
            specs.append(
                schema(
                    "align_region_references",
                    "Compute a TRANSLATION HYPOTHESIS aligning two CURRENT measured surface references while compensating source-to-TCP offset. No motion. For a carried object choose its visible region as source and destination region as target; set desired offset explicitly and select alignment axes (e.g. x,y preserves height). Surface references are NOT object centers. Requires rigid co-motion with unchanged orientation; remeasure lost regions and verify identity. Execute returned target_tcp_world_m using move_to_pose, then reobserve before release.",
                    {
                        "frame_id": fid,
                        "arm": {"type": "string", "enum": list(obs["arms"])},
                        "source_region_id": text,
                        "target_region_id": text,
                        "target_offset_world_m": vec,
                        "alignment_axes_world": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["x", "y", "z"]},
                            "minItems": 1,
                            "maxItems": 3,
                        },
                    },
                    [
                        "frame_id",
                        "arm",
                        "source_region_id",
                        "target_region_id",
                        "target_offset_world_m",
                    ],
                )
            )
        if not self.motion_tools:
            specs = [
                s
                for s in specs
                if s["function"]["name"] not in ("align_axis", "rotate_toward", "retreat")
            ]
        actuated = [
            arm for arm, state in obs["arms"].items() if state.get("gripper_actuated", True)
        ]
        for spec in specs:
            if spec["function"]["name"] == "set_gripper":
                spec["function"]["parameters"]["properties"]["arm"]["enum"] = actuated
        if not actuated:
            specs = [s for s in specs if s["function"]["name"] != "set_gripper"]
        if self.arrival_guard:
            for spec in specs:
                if spec["function"]["name"] in ("move_relative", "move_toward", "set_gripper"):
                    spec["function"]["parameters"]["properties"]["release_override_reason"] = {
                        "type": "string",
                        "description": "Optional explanation of changed intent; not required to release. Target residuals are advisory, not proof of object placement.",
                    }
                    spec["function"]["parameters"]["properties"]["close_override_reason"] = {
                        "type": "string",
                        "description": "Explicitly abandon a pending waypoint and close here for a deliberate non-arrival purpose. Explain observed reason; not proof of grasp.",
                    }
        if not self.regions.enabled:
            specs = [s for s in specs if s["function"]["name"] != "grasp_candidates"]
        if not callable(getattr(self.adapter, "move_to_position", None)):
            specs = [s for s in specs if s["function"]["name"] != "move_toward"]
        if self.episode_history is not None and self.task_progress is not None:
            specs.extend(memory_schemas(schema))
        # Low-level servo duration is not a VLM planning variable. Keep explicit
        # programmatic overrides for diagnostics, but expose only metric actions.
        for spec in specs:
            if (
                spec["function"]["name"] == "done"
                and self.completion_protocol.mode == "submission_only"
            ):
                spec["function"]["description"] = (
                    "Submit your final completion claim and end this attempt. "
                    "No environment success judgement is returned. Use end_attempt to stop without a success claim."
                )
            properties = spec["function"]["parameters"]["properties"]
            if "camera" in properties and "frame_id" in properties:
                properties["snapshot_id"] = {
                    "type": "string",
                    "description": "Optional exact sensor reference from a previous tool receipt; mismatches require reobservation.",
                }
            # A deliberate wait is different from a metric motion's internal
            # servo budget. Removing its required duration made hold impossible
            # to call while additionalProperties=False.
            if spec["function"]["name"] != "hold":
                spec["function"]["parameters"]["properties"].pop("steps", None)
            spec["function"]["parameters"]["properties"]["decision_note"] = {
                "type": "string",
                "description": "For each decision, provide <THINK>brief current visual/tool evidence; target ID and uncertainty; reason for the chosen tool/action; expected observable change</THINK>. Report your decision rationale, not assumed success.",
            }
            if self.grasp_provider and spec["function"]["name"] == "grasp_candidates":
                spec["function"]["description"] = (
                    "Propose grasps for a CURRENT measured target region using an optional learned point-cloud model, without requiring a long axis. Returns candidate IDs, nominal robot TCP pose hypotheses, and a schematic on the original camera image. Evaluate each option independently using the original views, target identity, visible jaw fit and approach direction; you may reject all candidates. No action, IK, path or grasp certificate. Open jaws, choose clearance, execute bounded moves."
                )
        for spec in specs:
            params = spec["function"]["parameters"]
            if spec["function"]["name"] == "move_toward":
                spec["function"]["description"] = (
                    'Choose exactly ONE target: prefer target_point={point_id:"P…",offset_world_m:{dx,dy,dz}} '
                    "for a CURRENT visible measured/tracked surface, avoiding copying XYZ. Offset is WORLD meters, "
                    "must be deliberately chosen for the TCP (surface is not a grasp pose). Lost/stale points "
                    "cannot be used as current evidence. Alternatively target_xyz_world_m={x,y,z} is an "
                    "explicit absolute world TCP position with named axes. Never submit both. "
                    + spec["function"]["description"].replace(
                        "supply target_xyz_world_m, NOT a relative delta.",
                        "resolve the selected absolute target once; do not chase a moving feature.",
                    )
                )
            if not set(params.get("required", [])) <= set(params.get("properties", {})):
                raise ValueError(
                    "Invalid tool schema: undeclared required parameters in "
                    + spec["function"]["name"]
                )
        if not self.grasp_provider:
            # A new robot must not inherit Panda's assumed geometry or advertise
            # a candidate tool which can only fail for missing calibration.
            available = [
                arm
                for arm, state in obs["arms"].items()
                if all(
                    key in state.get("geometry", {})
                    for key in ("axes_local", "sweep_probes_local_m", "finger_pad_centers_local_m")
                )
            ]
            specs = [s for s in specs if s["function"]["name"] != "grasp_candidates" or available]
            for spec in specs:
                if spec["function"]["name"] == "grasp_candidates":
                    spec["function"]["parameters"]["properties"]["arm"]["enum"] = available
        return specs

    def execute(self, name, args, obs):
        # Explanation metadata never becomes an actuator/perception argument.
        args = {k: v for k, v in args.items() if k != "decision_note"}
        resolved_target = None
        if name == "move_toward":
            # Validate before any target-memory mutation or native action.
            target, resolved_target = resolve_point_target(
                args, obs, self.tracked_points, self.points
            )
            args = {k: v for k, v in args.items() if k != "target_point"}
            args["target_xyz_world_m"] = target.tolist()
        sensor_reference = (
            validate_reference(args, obs) if "camera" in args and "frame_id" in args else None
        )
        before = None
        if self.advisory_controls and args.get("arm") in obs["arms"]:
            arm = args["arm"]
            before = {
                "frame_id": obs["frame_id"],
                "blocking": False,
                "waypoint": self.arrival_context(obs).get(arm),
                "pose_target": self.pose_targets.context(obs).get(arm),
                "meaning": "Pre-action robot target residuals, NOT object arrival or grasp evidence. You may change plan without an override. Check current contact/placement evidence before closing or releasing.",
            }
        result = self._execute(name, args, obs)
        if resolved_target is not None and isinstance(result, dict):
            result["resolved_target"] = resolved_target
        if sensor_reference is not None and isinstance(result, dict):
            result["sensor_reference"] = sensor_reference
        if before is not None and isinstance(result, dict):
            result["pre_action_target_advisory"] = before
        return result

    def _execute(self, name, args, obs):
        self.crop = None
        self.retrieved_history_images = []
        self.crop_caption = "CURRENT enlarged crop (mapping in previous tool result)"
        if name in ("search_history", "read_history", "update_task_progress"):
            if self.episode_history is None or self.task_progress is None:
                raise ValueError("Episode memory is not initialized")
            if name == "search_history":
                return self.episode_history.search(**args)
            if name == "update_task_progress":
                return self.task_progress.update(
                    args, self.episode_history, self.current_call_id, obs["frame_id"]
                )
            args = dict(args)
            include_images = args.pop("include_images", False)
            indices = args.pop("image_indices", None)
            if type(include_images) is not bool:
                raise ValueError("include_images must be boolean")
            if indices is not None and not include_images:
                raise ValueError("image_indices requires include_images=true")
            result = self.episode_history.read(**args)
            if include_images:
                self.retrieved_history_images = self.episode_history.image_blocks(
                    args["call_id"], obs["frame_id"], indices
                )
            result["images_scheduled_for_next_input"] = len(self.retrieved_history_images) // 2
            return result
        if name == "remember":
            self.memory[args["key"]] = args["value"]
            return {"saved": args["key"]}
        if name == "read_guidance":
            from .guidance import read_guidance

            return read_guidance(args["topic"])
        if name == "end_attempt":
            reason = args.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError("end_attempt requires a non-empty reason")
            return {
                "stop_requested": True,
                "termination": "model_ended_attempt",
                "reason": reason,
                "success_claimed": False,
                "meaning": "Attempt ended by model choice. Native evaluation determines success independently.",
            }
        if name == "done":
            if self.completion_feedback:
                check, reused = self.completion_protocol.check(self.adapter, obs)
                success = check["native_success"]
                self.completion_checks.append(check)
                return {
                    "stop_requested": success,
                    "summary": args["summary"],
                    "completion_check": check,
                    "completion_check_reused": reused,
                    "feedback": "Environment confirms task success."
                    if success
                    else "Environment has NOT judged the task successful. Your completion claim was rejected. Reobserve and choose recovery within remaining budgets, or use end_attempt if you choose to stop without success. Repeating done without new evidence does not change the outcome. No hidden failure reason or target coordinates are supplied.",
                }
            return {"stop_requested": True, "submitted": True, "summary": args["summary"]}
        if args.get("frame_id") != obs["frame_id"]:
            raise ValueError("Stale request; use CURRENT frame_id")
        if name == "hold":
            from .waiting import hold_current

            return hold_current(self.adapter, obs, args["steps"])
        if name == "move_relative":
            from .waiting import is_zero_relative

            if is_zero_relative(args, obs):
                return dict(
                    status="no_op",
                    native_steps=0,
                    frame_id=obs["frame_id"],
                    executed=False,
                    meaning="No translation, rotation or gripper command change requested. "
                    "No native command sent, no target replaced, and no new sensor evidence. Use hold to advance time.",
                )
        if name == "align_region_references":
            if not self.regions.enabled:
                raise ValueError("Region tools disabled")
            result = measured_region_alignment(self.regions, obs, args)
            evidence = self.interaction_evidence.context().get(
                args["source_region_id"], {"status": "unknown"}
            )
            result["current_target_hand_motion_evidence"] = evidence
            if self.advisory_controls:
                receipt = qualify_carry_alignment(result, evidence, args["arm"], obs["frame_id"])
                result["carry_applicability"] = receipt["carry_applicability"]
                result["carry_applicability"]["blocking"] = False
                return result
            return qualify_carry_alignment(result, evidence, args["arm"], obs["frame_id"])
        if name == "move_to_pose":
            if not self.motion_tools:
                raise ValueError("Motion extension disabled")
            if "gripper" in args:
                raise ValueError("move_to_pose preserves gripper; use set_gripper separately")
            try:
                return self.pose_targets.step(obs, args)
            finally:
                active = self.pose_targets.context(self.adapter.observe()).get(args["arm"])
                if self.arrival_guard and active:
                    self.waypoints[args["arm"]] = active["target_xyz_world_m"]
        if name == "grasp_candidates":
            if not self.regions.enabled:
                raise ValueError("Region tools disabled")
            if self.grasp_provider:
                config = self.grasp_provider
                raw = learned_candidates(
                    self.regions,
                    args["region_id"],
                    args["arm"],
                    obs,
                    config["endpoint"],
                    config["provider_axes"],
                    config["contact_reference_local_m"],
                )
                result = present_candidates(raw)
                camera = self.regions.entries[args["region_id"]]["camera"]
                self.crop = candidate_panel(result, obs, args["arm"], camera)
                self.crop_caption = "CURRENT original-view candidate schematics, side by side. These are hypothetical hand poses, NOT extra views or observed hands. Use original camera pixels for any measurement."
                result["visual_receipt"] = {
                    "camera": camera,
                    "type": "side_by_side_original_view_pose_hypotheses",
                    "yellow_dots": "nominal open finger-pad centers",
                    "pink": "schematic fingers, NOT full mesh",
                    "cyan": "pregrasp to TCP direction",
                    "measurement_coordinates": "Use original camera image pixels, never concatenated panel pixels",
                }
                return result
            return present_candidates(
                grasp_candidates(self.regions, args["region_id"], args["arm"], obs)
            )
        if name in ("find_regions", "inspect_region", "region_relation", "check_region_view"):
            if not self.regions.enabled:
                raise ValueError("Region tools disabled")
            if name == "find_regions":
                return self.regions.find(args, obs)
            if name == "inspect_region":
                return self.regions.inspect(args, obs)
            if name == "region_relation":
                return self.regions.relation(args, obs)
            return self.regions.cross_view(args["region_id"], args["camera"], obs)
        if not self.motion_tools and name in ("align_axis", "rotate_toward", "retreat"):
            raise ValueError("Motion extension disabled")
        if (
            self.arrival_guard
            and not self.advisory_controls
            and name in ("move_relative", "move_toward", "set_gripper")
        ):
            arm = args["arm"]
            previous = obs["arms"][arm].get("gripper_command_open", 1)
            if previous >= 0.5 and args.get("gripper", previous) < 0.5 and arm in self.waypoints:
                residual = self.arrival_context(obs)[arm]["remaining_distance_m"]
                reason = args.get("close_override_reason", "")
                if residual > 0.005 and (not isinstance(reason, str) or not reason.strip()):
                    raise ValueError(
                        f"Closing not executed: TCP is still {residual:.4f} m from the selected waypoint. Finish positioning before closing, or explicitly abandon it with close_override_reason. Arrival does NOT prove a grasp pose."
                    )
            if previous < 0.5 and args.get("gripper", previous) >= 0.5 and arm in self.waypoints:
                residual = self.arrival_context(obs)[arm]["remaining_distance_m"]
                if residual > 0.01:
                    reason = args.get("release_override_reason", "")
                    if not isinstance(reason, str) or not reason.strip():
                        raise ValueError(
                            f"Release not executed: previous TCP waypoint is still {residual:.4f} m away. Reobserve and correct, or explicitly abandon it with release_override_reason. This does not prove contact or task failure."
                        )
        if self.arrival_guard and not self.advisory_controls and name == "set_gripper":
            arm = args["arm"]
            previous = obs["arms"][arm].get("gripper_command_open", 1)
            opening = args["gripper"]
            active = self.pose_targets.context(obs).get(arm)
            if active and (previous >= 0.5) != (opening >= 0.5) and active["remaining_degrees"] > 5:
                key = "close_override_reason" if opening < 0.5 else "release_override_reason"
                reason = args.get(key)
                if not isinstance(reason, str) or not reason.strip():
                    raise ValueError(
                        f"Gripper change not executed: active pose still has {active['remaining_degrees']:.1f} degrees residual. Finish pose alignment or explicitly abandon with {key}."
                    )
        if name in ("move_relative", "move_toward", "rotate_toward", "align_axis", "retreat"):
            self.pose_targets.active.pop(args["arm"], None)
        if name in ("rotate_toward", "align_axis"):
            return self.motion.orient(name, args, obs)
        if name == "set_gripper":
            opening = float(args["gripper"])
            if not np.isfinite(opening) or not 0 <= opening <= 1:
                raise ValueError("Gripper opening must be 0..1")
            first_frame = obs["frame_id"]
            current = obs
            for _ in range(2):
                self.motion.move(current, args["arm"], [0, 0, 0], [0, 0, 0], opening, 12)
                current = self.observe()
                if (
                    self.gripper_context(current)[args["arm"]]["settled_last3_native_frames"]
                    is True
                ):
                    break
                if (
                    current["frame_id"] >= self.adapter.horizon
                    or policy_success(self.adapter)
                    or native_terminated(self.adapter)
                ):
                    break
            if args.get("release_override_reason") or args.get("close_override_reason"):
                self.waypoints.pop(args["arm"], None)
                self.pose_targets.active.pop(args["arm"], None)
            return {
                "native_steps": current["frame_id"] - first_frame,
                "gripper": self.gripper_context(current)[args["arm"]],
                "frame_id": current["frame_id"],
            }
        if name == "retreat":
            return self.motion.retreat(args, obs)
        if name == "measure_depth":
            camera = obs["vision"][args["camera"]]
            uv = pixel_points([args["uv"]], camera["color"].shape, args["coordinate_space"])[0]
            result = unproject(camera, uv)
            result.update(
                id=f"P{len(self.points) + 1}",
                frame_id=obs["frame_id"],
                camera=args["camera"],
                sensor_reference=validate_reference(args, obs),
            )
            result["position_world_m"] = dict(zip(("x", "y", "z"), result["xyz_world_m"]))
            result["delta_from_tcp_m"] = {
                arm: (np.array(result["xyz_world_m"]) - state["xyz_world_m"]).tolist()
                for arm, state in obs["arms"].items()
            }
            self.points.append(result)
            self.tracked_points.start(result, obs)
            result["tracking"] = (
                "Tracked as this persistent P ID. No verified identity. Lost tracks retain historical memory, not current coordinates."
            )
            return result
        if name == "inspect_crop":
            camera = obs["vision"][args["camera"]]
            h, w = camera["color"].shape[:2]
            x0, y0, x1, y1 = roi_rectangle(args["roi"], camera["color"].shape)
            scale = 384 / max(x1 - x0, y1 - y0)
            size = (max(1, round(scale * (x1 - x0))), max(1, round(scale * (y1 - y0))))
            self.crop = cv2.resize(camera["color"][y0:y1, x0:x1], size)
            self.crop_caption = f"CURRENT {args['camera']} crop, frame {obs['frame_id']}, original ROI {roi_receipt([x0, y0, x1, y1])}. Aspect ratio preserved. Measure in original pixels."
            return {
                "origin_uv": [int(x0), int(y0)],
                "roi_pixels": roi_receipt([x0, y0, x1, y1]),
                "scale_xy": [size[0] / (x1 - x0), size[1] / (y1 - y0)],
                "frame_id": obs["frame_id"],
            }
        if name == "move_toward":
            target = finite_vector(args["target_xyz_world_m"], 3)
            result = reach_position(self.motion, obs, args, self.reach_config)
            if self.arrival_guard:
                self.waypoints[args["arm"]] = target.tolist()
            return result
        if name == "move_relative":
            delta = finite_vector(args["delta"], 3)
            result = self.motion.move(
                obs,
                args["arm"],
                delta,
                args.get("rotation_deg", [0, 0, 0]),
                args.get("gripper"),
                args.get("steps", 12),
                check_sweep=bool(np.linalg.norm(args.get("rotation_deg", [0, 0, 0]))),
                delta_frame=args["frame"],
            )
            if self.arrival_guard and (
                args.get("release_override_reason") or args.get("close_override_reason")
            ):
                self.waypoints.pop(args["arm"], None)
            return result
        result = self.perception.execute(
            name, args, {"vision": obs["vision"]}, obs["frame_id"], self.points
        )
        if name == "fit_geometry":
            from .geometry_receipt import render

            self.crop, alignment = render(obs, args["camera"], result)
            self.crop_caption = f"CURRENT geometry on ORIGINAL {args['camera']}, frame {obs['frame_id']}. Yellow actual ROI; cyan unsigned fitted axis; magenta hand approach. Not a new view, identity or grasp score."
            result["current_hand_axis_comparison"] = alignment
        return result


SYSTEM = """You operate a robot using ONLY current RGB-D tools, proprioception and remembered evidence.
Every depth point starts causal tracking. Cyan P~ marks are CURRENT tracked feature estimates, not fresh measurements or verified identities. tracked_points reports WORLD displacement since initialization using current camera calibration/depth, not pixel displacement. Reuse current credible tracked XYZ when markers still match the intended feature; avoid remeasuring unchanged targets. Lost/ambiguous points have no current XYZ: remeasure when needed. Above 5mm is motion evidence, not proof of object motion; check correspondence visually. Compare manipulated-feature movement with hand movement before claiming effective manipulation.
Never use hidden task/object state. Follow the task instruction. Use tools to complete the task, not merely describe it.
World positions are meters; coordinate axes and arms come from capabilities. Gripper 0=closed,1=open.
Measure visible object/target surfaces before precise approaches. Pixels on top surfaces are NOT object centers:
choose geometry/clearance and contact positions yourself, accounting for the fingers. Maintain grasp by preserving closure command.
Current measurements are marked P1/P2/... on the current images. Verify the marker actually lands on the intended surface; a valid depth on background is not a valid target measurement. Do not repeat an unchanged query expecting new information.
Keep orientation unless a clear reason requires adjustment; short moves near contact, lift clear before lateral carrying motion.
Small-step translation limits apply independently to each component, not total vector length; see motion.translation_limit for delta_axis. move_toward is the exception: it goes toward a full absolute target within one call, without delta_axis chunking. Rotation-increment tools retain a total 15deg bound. Tool success is NOT task success.
Choose by INPUT TYPE: move_relative takes a small increment; move_toward attempts an entire absolute world XYZ transit with initial orientation held, returning on arrival/stall/budget; move_to_pose takes one small step toward a persistent XYZ+orientation goal; rotate_toward takes one rotation step toward a saved orientation. Long transit has no VLM reobservation mid-call; choose clear space. A surface measurement still needs a deliberately chosen clearance/contact offset.
Robot geometry describes approach/closing axes. Use align_axis to choose a contact orientation, then repeat rotate_toward with the saved R goal_id until the remaining angle is small. One 15deg step does not finish a 90deg goal. Choose clearance before rotating; sparse probe checks do not certify safety, especially with held objects.
Motion feedback counts low-progress attempts. After repeated undertracking, retreat along recorded observed waypoints or choose a meaningfully different approach/orientation; do not repeat the same blocked direction. Retreat preserves gripper command and is not guaranteed safe. Recovery decisions are yours, not a hidden task policy.
Track visibility is not verified identity/attachment. Re-measure moving features; never reuse historical coordinates as current.
Images explicitly labeled CURRENT are the only current images. Crops have their own origin/scale; use original pixels for measurement.
Use update_task_progress for your task stages, milestones and uncertainty; remember remains available for miscellaneous key/value notes. Avoid repeating identical failed actions without changing observation or approach.
Default to performing useful bounded motion after obtaining enough evidence. Use done only when you believe the task is complete.
"""


def run_agent(
    adapter,
    output,
    base_url,
    api_key,
    model="gemini-3.7-flash",
    max_calls=100,
    history_rounds=8,
    motion_tools=True,
    arrival_guard=True,
    tracking_backend="lk",
    tracking_device="cpu",
    region_tools=False,
    grasp_provider=None,
    interaction_feedback=False,
    completion_feedback=False,
    history_image_rounds=1,
    delta_axis=0.03,
    move_toward_max_steps=120,
    move_toward_stall_steps=24,
    move_toward_tolerance_m=0.003,
    observer_hand_advice=True,
    history_image_selection="decisions",
    completion_mode=None,
    max_output_tokens=None,
    truncation_max_output_tokens=7200,
    registry_class=None,
    client_factory=None,
    request_options=None,
):
    from .output_budget import default_output_budget

    if max_output_tokens is None:
        max_output_tokens = default_output_budget(model)
    output_budget = OutputBudget(max_output_tokens, truncation_max_output_tokens)
    completion_mode = resolve_mode(completion_mode, completion_feedback)
    completion_feedback = completion_mode == "feedback"
    bimanual_guidance(observer_hand_advice)  # validate even in a single-arm episode
    delta_axis = axis_limit(delta_axis)
    reach_config(move_toward_max_steps, move_toward_stall_steps, move_toward_tolerance_m)
    nonnegative_integer(history_rounds, "history_rounds")
    nonnegative_integer(history_image_rounds, "history_image_rounds")
    validate_image_history_selection(history_image_selection)
    if max_calls < 1:
        raise ValueError("max_calls must be positive")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    # Refuse an existing episode rather than overwriting its full evidence.
    for filename in (
        "agent.jsonl",
        "episode_history.jsonl",
        "task_progress.jsonl",
        "result.json",
        "gripper_events.jsonl",
        "gripper_samples.jsonl",
    ):
        if (output / filename).exists():
            raise FileExistsError(f"Use a fresh episode directory: {output / filename}")
    registry = (registry_class or ToolRegistry)(
        adapter,
        advisory_controls=True,
        motion_tools=motion_tools,
        arrival_guard=arrival_guard,
        tracking_backend=tracking_backend,
        tracking_device=tracking_device,
        region_tools=region_tools,
        grasp_provider=grasp_provider,
        interaction_feedback=interaction_feedback,
        completion_feedback=completion_feedback,
        completion_mode=completion_mode,
        delta_axis=delta_axis,
        move_toward_max_steps=move_toward_max_steps,
        move_toward_stall_steps=move_toward_stall_steps,
        move_toward_tolerance_m=move_toward_tolerance_m,
    )
    registry.episode_history = EpisodeHistory(output)
    registry.gripper_monitor.set_journal(output / "gripper_events.jsonl")
    registry.task_progress = TaskProgress(output, adapter.observe()["instruction"])
    system = SYSTEM
    system += "\nGripper receipts distinguish measured metric gap, measured normalized joint opening, and unavailable data. Gripper events are timestamped proprioceptive changes, not verdicts about attachment or task failure. They never stop a motion or change the gripper command. Check current images and decide whether to continue or recover; a blocked closure can be normal object contact. Old events in history are not new events.\n"
    if len(adapter.observe()["arms"]) > 1:
        system += "\n" + bimanual_guidance(observer_hand_advice) + "\n"
    system += "\nUse done to claim task completion; use end_attempt to stop an unsuccessful attempt by choice. Never confuse ending an attempt with achieving its objective.\n"
    history_source = (
        "model decisions"
        if history_image_selection == "decisions"
        else "distinct earlier physical states (latest archived camera set for each state)"
    )
    system += (
        f"\nEpisode memory: the prompt retains the latest {history_rounds} complete text tool exchanges "
        f"and camera images from the previous {history_image_rounds} {history_source}, plus CURRENT images. "
        "Historical images carry their call/frame/camera labels; same-frame comparisons contain no new sensor time. "
        "Camera motion and changed overlays may explain differences, not only object motion. Never treat historical "
        "coordinates/visibility as current. All completed decisions remain saved: search_history searches them, "
        "read_history retrieves full paginated evidence and optional archived images. Read more pages when needed. "
        "The original task and your task_progress are presented every decision independently of the text window. "
        "Use update_task_progress to record your own plan early, then important milestone transitions, failed "
        "attempts, changed goals and remaining uncertainties. Do not mechanically update every action. "
        "Cite completed call IDs for evidence; completed milestones are your claims, not environment certificates. "
        "Revise contradicted claims to uncertain/invalidated. Do not replace the original instruction. "
        "This memory is local to this episode, not transferred between tasks.\n"
    )
    if grasp_provider:
        system += "\nThe grasp_candidates provider is a learned visible-point-cloud model: it also supports non-slender regions. These are nominal pad-aligned hypotheses, not executed grasps or exact jaw-fit certificates. Inspect each candidate's original-view schematic and geometry independently; IDs identify poses, not a required choice. You may reject all candidates and choose your own pose. Do not repeatedly recompute a static candidate while alternating incompatible rotation goals. Commit to a plausible candidate with clearance, or reject it with observed evidence.\n"
    if interaction_feedback:
        system += "\nInteraction evidence persists across history truncation. A large jaw-gap decrease while still commanding closed, or target anchors not following the hand, challenges the carry/contact hypothesis. Check the current views and do a small safe diagnostic before continuing assumed transport; never infer attachment from gap alone. Anchor co-motion residual includes hand rotation, but depth-layer/correspondence errors remain possible. Command events are factual receipts: do not invent a pick-and-place sequence after forgetting prior turns. For multiple similar objects, verify the instruction's spatial relation to its reference object before selecting a target; segmentation confidence/ID order does not answer that question.\n"
    if region_tools:
        system += "\nFor a well-localized slender grasp target, grasp_candidates can compare approach/orientation hypotheses using measured region and robot geometry. It supplies poses, not a grasp action or verified free path; you still choose a candidate, move to clearance, rotate, approach and settle closure. Small-step tools use up to 12 native servo steps; full-target move_toward has its own execution budget. Inspect actual achieved displacement and residual rather than assuming exact execution.\n"
        system += "\nGripper semantics: APPROACH is the direction fingers point toward contact; CLOSING is the line joining the opposed pads and the direction jaws move together, NOT the long direction of a finger. LATERAL is perpendicular to both. For slender parts, closing across a thin dimension rather than along the long axis is often appropriate; still check available clearance. Robot A/C/L overlays are proprioceptive axes, not scene observations. Use set_gripper to let closure settle before assuming a grasp. Near the calibrated closed limit may indicate an empty grasp (or a very thin object); it does not justify claiming attachment. Verify target response with a small safe action and current visual evidence before continuing.\n"
        system += "\nCompletion discipline: a successful grasp, some target displacement, or a chosen number of movement steps is PROGRESS, not necessarily completion. For manipulation toward a mechanical end state, observe the intended terminal extent before releasing or declaring done. If motion remains effective and there is no contrary safety evidence, consider a further bounded continuation instead of stopping at the first apparent progress. Resistance alone is not proof of the correct endpoint. If target evidence is lost, reobserve using the existing cameras and acknowledge uncertainty. Orange S? projections are automatically refreshed depth-consistent candidates in the other existing view, not verified identities.\n"
        system += "\nFor initial semantic localization prefer find_regions with a short noun phrase, then select the intended S ID from labeled contours in the original view. Do not guess pixel boxes when uncertain. Candidate IDs are NOT ordered by height or task relevance. inspect_region is for refinements or when text search fails. Original-image tick marks use horizontal u and vertical v in pixels.\n"
        system += "\nUse inspect_region for a selected visible part when you need shape, orientation or relative position: one call returns those together. Green contours identify segmented regions, orange lines show unsigned geometric major axes, purple lines join TCP and surface reference; these are annotations on existing cameras only. Region summaries refresh using tracked image anchors and current RGB-D; uncertain regions require reinspection. Check mask identity visually. A visible-surface median is NOT a grasp pose, mechanical axis, full object shape or free space. Use relative_to_robot rather than repeatedly sampling the same part. Never infer the permitted manipulation direction from a PCA axis alone. Detect poor motion progress from measured feedback, reconsider assumptions, and do not claim completion without current visual evidence.\n"
    if not motion_tools:
        system = "\n".join(
            line
            for line in system.splitlines()
            if not line.startswith(("Robot geometry describes", "Motion feedback counts"))
        )
    if arrival_guard:
        system += "\nmove_toward attempts the entire waypoint within one call, but arrival is NOT guaranteed. Read target_reached and remaining_distance_m before release; do not treat an action note as observed completion. If undertracking persists, reobserve and choose the next action. TCP arrival does not prove object alignment. After release inspect CURRENT views for stable placement; never infer success from an open-gripper command alone.\n"
    if motion_tools:
        system += "\nFor repeated positioning with a chosen orientation, prefer move_to_pose and resume its pose_goal_id: it preserves the ORIGINAL absolute orientation across chunks rather than inheriting accumulated drift. Supplying XYZ without a quaternion snapshots current orientation. Choose clearance before simultaneous rotation/translation. It does not change the gripper; close/release separately after checking both position and angle. Legacy move/rotate/retreat tools abandon the active pose target (saved IDs remain resumable). A pose target is not an object goal or grasp certificate.\n"
    if region_tools and motion_tools:
        system += "\nWhen transporting a grasped object, its visible reference may be centimeters away from the TCP. Do not equate TCP-over-destination with object-over-destination. Use align_region_references on CURRENT intended source/destination regions to compute an offset-compensated translation hypothesis. Choose the desired source-reference offset explicitly; align x,y first if maintaining height is appropriate, then reobserve and choose vertical motion using measured surfaces and clearance. Surface statistics are not full object centers. This requires rigid co-motion and unchanged orientation; lost regions, changed contact or inconsistent correspondence require reinspection, not copying a stale target. Execute with move_to_pose and recheck the actual object relation before release or done.\n"
    if completion_feedback:
        system += "\nCompletion feedback is enabled: calling done requests a sparse environment success check. If it reports failure, the episode continues; reobserve and correct instead of merely repeating done. The check provides only a task-success boolean, never hidden coordinates or predicate thresholds. Every request consumes model-call budget.\n"
    elif completion_mode == "submission_only":
        system += "\nCompletion protocol is submission_only: done submits your final completion claim and ends this attempt. No task-success judgement is returned to you. Judge progress from current images and measured robot/tool evidence; host evaluation is recorded separately after the attempt. A native environment may terminate independently.\n"
    system += "\nDecision authority: target arrival, sparse sweep and low-progress evidence are ADVISORY, not vetoes. Choose to continue, reduce motion, change target, observe or retreat. No override reason is required. Keep intended contact distinct from obstacle contact; these estimates cannot certify safety or attachment. Before retrying, state what observable change you expect in the tool note. Identical-frame crops/searches provide no new sensor evidence; use current region geometry, another existing view or a purposeful small probe when appropriate. Do not confuse grasp confidence or co-motion with a verified grasp.\n"
    history = []
    system += "\nFor EVERY tool call, include decision_note='<THINK>...</THINK>': a concise decision rationale grounded in the CURRENT images and tool receipts. Identify the intended S/P target and why its visible location/shape matches the instruction when relevant; distinguish observations from assumptions; explain the chosen action and expected observable change. After failure, say what evidence changes your next decision. Do not repeat generic warnings, invent evidence, or treat the rationale as proof of success. THINK delimiters are text markers, not a separate hidden reasoning channel.\n"
    started = time.time()
    stopped = False
    errors = 0
    tool_execution_errors = missing_tool_calls = 0
    llm_calls = 0
    last_frame = -1
    stationary_calls = 0
    perception_budget = 3
    url = base_url.rstrip("/") + "/chat/completions"
    with (
        (client_factory or httpx.Client)(timeout=180) as client,
        (output / "agent.jsonl").open("w") as log,
    ):
        for call_id in range(max_calls):
            if (
                policy_success(adapter)
                or native_terminated(adapter)
                or adapter.frame >= adapter.horizon
            ):
                break
            obs = registry.observe()
            registry.current_call_id = call_id
            context = {k: v for k, v in obs.items() if k != "vision"}
            # Detailed probe coordinates serve the deterministic geometry check,
            # not the VLM prompt. Keep semantic axes and the stated limitations.
            context["arms"] = {
                arm: {
                    **state,
                    "geometry": {
                        k: v
                        for k, v in state.get("geometry", {}).items()
                        if k != "sweep_probes_local_m"
                    },
                }
                for arm, state in obs["arms"].items()
            }
            stationary_calls = stationary_calls + 1 if obs["frame_id"] == last_frame else 0
            last_frame = obs["frame_id"]
            context.update(
                decision_call_id=call_id,
                task_progress=registry.task_progress.context(),
                episode_history={
                    "completed_calls": len(registry.episode_history.records),
                    "text_history_rounds_N": history_rounds,
                    "previous_image_rounds_K": history_image_rounds,
                    "tools": ["search_history", "read_history", "update_task_progress"],
                },
                move_toward_execution=registry.reach_config,
                memory=registry.memory,
                measurements=registry.points[-10:],
                perception=registry.perception.context(),
                motion=registry.motion.context(),
                pose_targets=registry.pose_targets.context(obs),
                tracked_points=registry.tracked_points.context(),
                regions=registry.regions.context(obs),
                gripper_feedback=registry.gripper_context(obs),
                calls_remaining=max_calls - call_id,
                steps_remaining=adapter.horizon - adapter.frame,
            )
            pending_gripper_events = registry.gripper_monitor.drain_events()
            if pending_gripper_events:
                context["gripper_events"] = pending_gripper_events
            if interaction_feedback:
                context["target_hand_motion_evidence"] = registry.interaction_evidence.context()
            if completion_feedback and registry.completion_checks:
                context["last_requested_completion_check"] = registry.completion_checks[-1]
            # Freeze the actual decision input before tool execution mutates memory/feedback.
            if arrival_guard:
                context["arrival"] = registry.arrival_context(obs)
            if getattr(registry, "single_motion_feedback", False):
                # Root observation feedback is the current adapter receipt. The
                # motion-controller cache can describe an older tool invocation.
                context["motion"].pop("feedback", None)
            from .feedback_presentation import actor_feedback, PRESENTATION_VERSION

            diagnostic_context = json.loads(json.dumps(context))
            context = actor_feedback(diagnostic_context)
            content = [
                {
                    "type": "text",
                    "text": "CURRENT TASK: "
                    + obs["instruction"]
                    + "\nCURRENT observation "
                    + json.dumps(context),
                }
            ]
            for name, camera in obs["vision"].items():
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": f"CURRENT {name}, frame {obs['frame_id']}, ORIGINAL image {camera['color'].shape[1]}x{camera['color'].shape[0]}. u=x left-to-right, v=y top-to-bottom. P markers show only this frame's measured pixels; not object detections.",
                        },
                        image_content(
                            registry.regions.overlay(
                                registry.tracked_points.overlay(
                                    measurement_overlay(
                                        camera["color"], name, obs["frame_id"], registry.points
                                    ),
                                    name,
                                    obs["frame_id"],
                                ),
                                name,
                                obs,
                            )
                        ),
                    ]
                )
            if registry.crop is not None:
                content.extend(
                    [
                        {
                            "type": "text",
                            "text": registry.crop_caption,
                        },
                        image_content(registry.crop),
                    ]
                )
            current_camera_count = len(obs["vision"])
            history_image_call_ids = registry.episode_history.recent_image_call_ids(
                history_image_rounds, obs["frame_id"], history_image_selection
            )
            content.extend(
                registry.episode_history.recent_image_blocks(
                    history_image_rounds, obs["frame_id"], history_image_selection
                )
            )
            content.extend(registry.retrieved_history_images)
            # Retrieved images are a one-input response, not a persistent panel.
            registry.retrieved_history_images = []
            # Archive actual JPEG inputs, including overlays, for faithful video review.
            image_dir = output / "model_inputs"
            image_dir.mkdir(exist_ok=True)
            image_manifest = []
            caption = ""
            for item in content:
                if item.get("type") == "text":
                    caption = item["text"]
                if item.get("type") != "image_url":
                    continue
                image_index = len(image_manifest)
                path = image_dir / f"call_{call_id:03d}_view_{image_index}.jpg"
                path.write_bytes(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
                image_manifest.append(
                    {
                        "index": image_index,
                        "path": str(path.relative_to(output)),
                        "caption": caption,
                    }
                )
            messages = [
                {"role": "system", "content": system},
                *history_window(history, rounds=history_rounds),
                {"role": "user", "content": content},
            ]
            available = registry.schemas(obs)
            if stationary_calls >= perception_budget:
                content.append(
                    {
                        "type": "text",
                        "text": "Repeated perception at an unchanged physical state: no new sensor frame is available. All tools remain available. Decide what new evidence a different query/view could add, or choose a purposeful bounded probe; do not move merely to reset a counter.",
                    }
                )
            sensor_dir = output / "sensors"
            sensor_dir.mkdir(exist_ok=True)
            np.savez_compressed(
                sensor_dir / f"frame_{obs['frame_id']:04d}.npz",
                **{
                    f"{name}_{key}": v
                    for name, c in obs["vision"].items()
                    for key, v in c.items()
                    if not key.startswith("_")
                },
            )
            request = {
                "model": model,
                "messages": messages,
                "tools": available,
                "tool_choice": "required",
                "temperature": 0.2,
                "max_tokens": output_budget.current,
            }
            if request_options:
                if set(request_options) - {"provider", "seed"}:
                    raise ValueError(
                        "Only provider and sampling seed request options are supported"
                    )
                request.update(request_options)
            request_dir = output / "model_requests"
            response_dir = output / "model_responses"
            request_dir.mkdir(exist_ok=True)
            response_dir.mkdir(exist_ok=True)
            with (request_dir / f"call_{call_id:06d}.json").open("x") as stream:
                json.dump(request, stream, ensure_ascii=False)
            t = time.time()
            for attempt in range(3):
                try:
                    response = client.post(
                        url, headers={"Authorization": "Bearer " + api_key}, json=request
                    )
                    response.raise_for_status()
                    body = response.json()
                    break
                except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                    body_text = (
                        exc.response.text.replace(api_key, "[REDACTED]")[:1200]
                        if isinstance(exc, httpx.HTTPStatusError)
                        else type(exc).__name__
                    )
                    print(json.dumps({"api_retry": attempt, "error": body_text}), flush=True)
                    if "insufficient_user_quota" in body_text:
                        raise RuntimeError(
                            "Gemini relay quota exhausted; top up configured account or explicitly provide authorized credentials"
                        ) from exc
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            with (response_dir / f"call_{call_id:06d}.json").open("x") as stream:
                json.dump(body, stream, ensure_ascii=False)
            message = body["choices"][0]["message"]
            llm_calls += 1
            calls = message.get("tool_calls", [])
            row = {
                "call_id": call_id,
                "frame_id": obs["frame_id"],
                "model": body.get("model"),
                "response": message,
                "finish_reason": body["choices"][0].get("finish_reason"),
                "seconds": time.time() - t,
                "usage": body.get("usage"),
                "request_max_output_tokens": request["max_tokens"],
                "context": context,
                "diagnostic_context": diagnostic_context,
                "feedback_presentation": PRESENTATION_VERSION,
                "stationary_calls": stationary_calls,
                "available_tools": [s["function"]["name"] for s in available],
                "history_rounds": history_rounds,
                "history_image_rounds": history_image_rounds,
                "history_image_selection": history_image_selection,
                "history_image_source_call_ids": history_image_call_ids,
                "current_camera_count": current_camera_count,
                "image_manifest": image_manifest,
                "harness_version": HARNESS_VERSION,
                "observer_hand_advice": observer_hand_advice,
                "delta_axis": delta_axis,
                "history_tool_calls": sum(
                    m.get("role") == "assistant" and bool(m.get("tool_calls")) for m in messages
                ),
            }
            recovery = output_budget.after_response(row["finish_reason"], bool(calls))
            if recovery is not None:
                row["output_budget_recovery"] = recovery
            results = []
            # Execute one call per request. Multiple proposed calls cannot use future frames.
            if calls:
                call = calls[0]
                name = call["function"]["name"]
                from .decision_trace import decision_trace

                try:
                    trace_args = json.loads(call["function"]["arguments"])
                except (ValueError, TypeError):
                    trace_args = {}
                row["decision_trace"] = decision_trace(
                    message, trace_args if isinstance(trace_args, dict) else {}
                )
                try:
                    if name not in {s["function"]["name"] for s in available}:
                        raise ValueError(
                            "Tool not available in this decision; select one of the supplied tools"
                        )
                    args = json.loads(call["function"]["arguments"])
                    result = registry.execute(name, args, obs)
                except (ValueError, KeyError, TypeError, RuntimeError) as exc:
                    from .resource_errors import raise_if_infrastructure_error

                    raise_if_infrastructure_error(exc)
                    result = {"error": str(exc)}
                    errors += 1
                    tool_execution_errors += 1
                gripper_events = registry.gripper_monitor.drain_events()
                if gripper_events:
                    result["gripper_events"] = gripper_events
                results.append({"name": name, "result": result})
                history.extend(
                    [
                        {
                            "role": "user",
                            "content": f"Historical decision call {call_id}, frame {obs['frame_id']}; images omitted here. Task: {obs['instruction']}",
                        },
                        {
                            "role": "assistant",
                            "content": message.get("content"),
                            "tool_calls": [call],
                            **(
                                {"reasoning_details": message["reasoning_details"]}
                                if message.get("reasoning_details")
                                else {}
                            ),
                        },
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "name": name,
                            "content": json.dumps(actor_feedback(result)),
                        },
                    ]
                )
                stopped = name in {"done", "end_attempt"} and bool(
                    result.get("stop_requested", False)
                )
            else:
                errors += 1
                missing_tool_calls += 1
                row["response_error"] = dict(
                    kind="missing_tool_call",
                    finish_reason=row["finish_reason"],
                    native_command_sent=False,
                )
                history.append(
                    {
                        "role": "user",
                        "content": (
                            "Previous output reached its token limit before a tool call; NO action executed. "
                            "Select one tool with a concise decision summary. "
                            f"Next output cap is {output_budget.current} tokens."
                            if recovery is not None
                            else "No tool call received. Select a tool now."
                        ),
                    }
                )
            row["results"] = results
            row["model_results"] = actor_feedback(results)
            log.write(json.dumps(row) + "\n")
            log.flush()
            registry.episode_history.append(row, image_manifest, current_camera_count)
            print(
                json.dumps(
                    {
                        "call": call_id,
                        "frame": adapter.frame,
                        "tool": results,
                        "seconds": round(row["seconds"], 2),
                    }
                ),
                flush=True,
            )
            if stopped:
                break
    final_native_success = bool(adapter.success())  # Host only, after the last model call.
    result = {
        "harness_version": HARNESS_VERSION,
        "observer_hand_advice": observer_hand_advice,
        "delta_axis": delta_axis,
        "move_toward_execution": registry.reach_config,
        "history_rounds": history_rounds,
        "history_image_rounds": history_image_rounds,
        "history_image_selection": history_image_selection,
        "archived_completed_calls": len(registry.episode_history.records),
        "task_progress_revision": registry.task_progress.state["revision"],
        "model_requested": model,
        "output_token_budget": dict(
            base=max_output_tokens,
            truncation_cap=truncation_max_output_tokens,
            final_cap=output_budget.current,
            recovery="episode_high_water_mark; next logged requests only; no action replay or extra model calls",
        ),
        "completion_feedback_enabled": completion_feedback,
        "completion_mode": completion_mode,
        "requested_completion_native_queries": registry.completion_protocol.native_queries,
        "completion_checks": len(registry.completion_checks),
        "rejected_completion_checks": sum(
            not x["native_success"] for x in registry.completion_checks
        ),
        "native_success": final_native_success,
        "gripper_monitor": registry.gripper_monitor.protocol(),
        "gripper_event_count": len(registry.gripper_monitor.events),
        "native_steps": adapter.frame,
        "llm_calls": llm_calls,
        "tool_errors": errors,
        "tool_execution_errors": tool_execution_errors,
        "missing_tool_calls": missing_tool_calls,
        "stop_requested": stopped,
        "elapsed_seconds": time.time() - started,
        "termination": "native_success"
        if final_native_success
        else "model_ended_attempt"
        if stopped and name == "end_attempt"
        else "model_done"
        if stopped
        else "native_episode_ended"
        if native_terminated(adapter)
        else "budget",
    }
    (output / "result.json").write_text(json.dumps(result, indent=2))
    return result
