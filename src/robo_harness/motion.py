"""Bounded orientation goals and evidence-based recovery; no task/object handles."""

from collections import defaultdict, deque

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import calibration
from .perception import finite_vector
from .motion_limits import axis_limit, bounded_translation, step_axis_limit


def rotation(state):
    if "quaternion_xyzw" in state:
        q = finite_vector(state["quaternion_xyzw"], 4)
        if np.linalg.norm(q) < 1e-8:
            raise ValueError("Zero quaternion")
        return Rotation.from_quat(q)
    return Rotation.from_matrix(np.asarray(state["axes_world"], float))


def unit(vector):
    vector = finite_vector(vector, 3)
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        raise ValueError("Direction must be nonzero")
    return vector / norm


def align_rotation(current, local_axis, world_direction):
    """Shortest rotation aligning one axis, including a deterministic antipodal case."""
    a, b = current.apply(unit(local_axis)), unit(world_direction)
    cross = np.cross(a, b)
    dot = np.clip(a @ b, -1, 1)
    if np.linalg.norm(cross) < 1e-8:
        if dot > 0:
            return current
        basis = np.eye(3)[np.argmin(np.abs(a))]
        return Rotation.from_rotvec(unit(np.cross(a, basis)) * np.pi) * current
    return Rotation.from_rotvec(unit(cross) * np.arccos(dot)) * current


def bounded_rotation(current, target, maximum_deg=15):
    if not np.isfinite(maximum_deg) or not 0 < maximum_deg <= 15:
        raise ValueError("Rotation step limit must be in (0, 15] degrees")
    error = (target * current.inv()).as_rotvec()
    angle = np.linalg.norm(error)
    step = error * min(1, np.radians(maximum_deg) / max(angle, 1e-12))
    return np.degrees(step), float(np.degrees(angle))


def sweep_evidence(obs, arm, delta, dr):
    """Sparse visible-depth probe check, NOT a collision certificate.

    Probe trajectories in known free space are useful evidence. Occluded space,
    meshes between probes, the arm and held objects are not certified safe.
    Surfaces near initial robot probes are ambiguous (self/object), not obstacles.
    """
    state = obs["arms"][arm]
    probes = np.asarray(state.get("geometry", {}).get("sweep_probes_local_m", []), float)
    if probes.ndim != 2 or probes.shape[1:] != (3,) or not len(probes):
        return {
            "status": "unknown",
            "reason": "Robot footprint unavailable",
            "collision_free": False,
        }
    start = np.asarray(state["xyz_world_m"])
    r0 = rotation(state)
    initial = r0.apply(probes) + start
    paths = []
    for fraction in (0.25, 0.5, 0.75, 1):
        rr = Rotation.from_rotvec(np.radians(dr) * fraction) * r0
        paths.extend(rr.apply(probes) + start + np.asarray(delta) * fraction)
    paths = np.asarray(paths)
    free = np.zeros(len(paths), bool)
    risks = []
    for name, camera in obs["vision"].items():
        if "depth" not in camera:
            continue
        k, t = calibration(camera)
        depth = np.asarray(camera["depth"]).squeeze()
        h, w = depth.shape
        optical = (paths - t[:3, 3]) @ t[:3, :3]
        for i, p in enumerate(optical):
            if p[2] <= 0:
                continue
            uv = k @ (p / p[2])
            u, v = np.floor(uv[:2] + 0.5).astype(int)
            if not (0 <= u < w and 0 <= v < h):
                continue
            z = float(depth[v, u])
            if not np.isfinite(z) or z <= 0:
                continue
            if p[2] < z - 0.006:
                free[i] = True
            elif abs(p[2] - z) <= 0.004:
                surface = t[:3, :3] @ (np.linalg.solve(k, [u, v, 1]) * z) + t[:3, 3]
                # A sparse nearest-corner test misses the palm's own surfaces.
                # Within the current hand envelope, self vs held/contact object
                # is ambiguous: mark UNKNOWN, never declare this volume free.
                local_surface = r0.inv().apply(surface - start)
                in_hand_envelope = np.all(local_surface >= probes.min(axis=0) - 0.012) and np.all(
                    local_surface <= probes.max(axis=0) + 0.012
                )
                from .self_geometry import visible_self_mask

                own_hand = visible_self_mask(camera, arm, [surface])
                if own_hand is not None:
                    if own_hand[0]:
                        continue
                    risks.append({"camera": name, "pixel": [int(u), int(v)]})
                else:
                    if in_hand_envelope:
                        continue
                    if np.min(np.linalg.norm(initial - surface, axis=1)) > 0.015:
                        risks.append({"camera": name, "pixel": [int(u), int(v)]})
    return {
        "status": "possible_surface_intersection"
        if risks
        else "partly_observed_free"
        if free.any()
        else "unknown",
        "free_probe_fraction": float(free.mean()),
        "risk_pixels": risks[:6],
        "collision_free": False,
        "limitations": "Sparse current RGB-D only; self/occlusion ambiguous, arm and held-object extent unchecked",
    }


class MotionController:
    def __init__(self, adapter, adaptive_sweep=False, advisory=False, delta_axis=0.03):
        self.delta_axis = axis_limit(delta_axis)
        # The controller owns policy bounds; adapters retain a world-space transport envelope.
        adapter.delta_axis = self.delta_axis
        self.advisory = advisory
        self.adapter = adapter
        self.adaptive_sweep = adaptive_sweep
        self.goals = {}
        self.last_goal = {}
        self.paths = defaultdict(lambda: deque(maxlen=80))
        self.low = {}
        self.feedback = {}
        self.retreat_goals = {}

    def context(self):
        return {
            "translation_limit": {
                "delta_axis_m": self.delta_axis,
                "rule": "abs(each component) <= delta_axis in the requested world/gripper frame; target tools use world axes",
            },
            "orientation_goals": self.last_goal,
            "feedback": self.feedback,
            "recovery": {
                a: {
                    "low_progress_streak": v["count"],
                    "repeat_direction_blocked": not self.advisory and v["count"] >= 3,
                }
                for a, v in self.low.items()
            },
            "recorded_waypoints": {a: len(v) for a, v in self.paths.items()},
        }

    def _pose(self, state):
        return {"xyz": np.asarray(state["xyz_world_m"], float).copy(), "rotation": rotation(state)}

    def _record(self, arm, state):
        pose = self._pose(state)
        path = self.paths[arm]
        if (
            not path
            or np.linalg.norm(pose["xyz"] - path[-1]["xyz"]) >= 0.004
            or np.degrees((pose["rotation"] * path[-1]["rotation"].inv()).magnitude()) >= 3
        ):
            path.append(pose)

    def move(
        self,
        obs,
        arm,
        delta,
        dr,
        gripper=None,
        steps=8,
        recovery=False,
        check_sweep=False,
        preserve_orientation_anchor=False,
        delta_frame="world",
    ):
        delta, dr = finite_vector(delta, 3), finite_vector(dr, 3)
        if np.max(np.abs(delta)) > self.delta_axis + 1e-9:
            raise ValueError(
                f"Each translation component must be within +/-{self.delta_axis:g} m in {delta_frame} coordinates"
            )
        if np.linalg.norm(dr) > 15.000001:
            raise ValueError("Whole rotation limited to 15deg")
        state = obs["arms"][arm]
        if delta_frame == "gripper":
            delta = rotation(state).apply(delta)
        elif delta_frame != "world":
            raise ValueError("Unknown frame")
        requested_delta, requested_dr = delta.copy(), dr.copy()
        direction = np.r_[delta / self.delta_axis, dr / 15]
        norm = np.linalg.norm(direction)
        previous = self.low.get(arm)
        advisories = []
        if (
            norm > 1e-6
            and previous
            and previous["count"] >= 3
            and direction @ previous["direction"] / norm > 0.85
        ):
            if not self.advisory:
                raise ValueError(
                    "Repeated low-progress direction blocked. Retreat, change approach/orientation, or stop. This is not proof of contact."
                )
            advisories.append(
                {
                    "kind": "repeated_low_progress",
                    "frame_id": obs["frame_id"],
                    "count": previous["count"],
                    "blocking": False,
                    "suggestion": "Check actual object/hand motion. Continue deliberately, reduce the step, change approach or retreat; low progress does not prove contact.",
                }
            )
        evidence = sweep_evidence(obs, arm, delta, dr) if check_sweep else None
        prefix = None
        if (
            not self.advisory
            and self.adaptive_sweep
            and evidence
            and evidence["status"] == "possible_surface_intersection"
        ):
            prefix = {
                "requested_translation_m": requested_delta.tolist(),
                "requested_rotation_vector_world_deg": requested_dr.tolist(),
                "original_sweep": evidence,
                "tested_prefixes": [],
                "accepted_scale": None,
                "meaning": "Shorter prefix of the same requested motion, not a changed goal or a collision certificate. Reobserve after this step; absolute pose arrival remains pending.",
            }
            for scale in (0.5, 0.25, 0.125):
                trial = sweep_evidence(obs, arm, requested_delta * scale, requested_dr * scale)
                prefix["tested_prefixes"].append(
                    {
                        "scale": scale,
                        "status": trial["status"],
                        "free_probe_fraction": trial.get("free_probe_fraction", 0),
                    }
                )
                if (
                    trial["status"] == "partly_observed_free"
                    and trial.get("free_probe_fraction", 0) > 0
                ):
                    delta, dr, evidence = requested_delta * scale, requested_dr * scale, trial
                    prefix["accepted_scale"] = scale
                    break
        if self.advisory and evidence and evidence["status"] == "possible_surface_intersection":
            advisories.append(
                {
                    "kind": "possible_surface_intersection",
                    "blocking": False,
                    "frame_id": obs["frame_id"],
                    "evidence": evidence,
                    "suggestion": "Sparse depth is uncertain and intended contact may be valid. Decide whether to continue, use a smaller step or choose clearance; this is not collision certification.",
                }
            )
        if not self.advisory and evidence and evidence["status"] == "possible_surface_intersection":
            self.feedback[arm] = {"sweep": evidence, "executed": False}
            if prefix is not None:
                self.feedback[arm]["bounded_prefix"] = prefix
            raise ValueError(
                "Visible surface may intersect the proposed finger sweep. Move to clearance or choose a smaller/different motion; see motion feedback."
            )
        if not recovery:
            self._record(arm, state)
        start = self._pose(state)
        result = dict(self.adapter.move(arm, delta, dr, gripper, steps))
        if self.advisory:
            result["decision_advisories"] = advisories
        if prefix is not None:
            result["bounded_prefix"] = prefix
        if callable(getattr(self.adapter, "observe", None)):
            after = self.adapter.observe()["arms"][arm]
            actual = self._pose(after)
            actual_dr = np.degrees((actual["rotation"] * start["rotation"].inv()).as_rotvec())
            actual_delta = actual["xyz"] - start["xyz"]
            if not preserve_orientation_anchor and np.linalg.norm(delta) > 1e-9:
                # Saved orientation may resume after deliberate translation. Do not
                # pull the TCP back to a pre-translation position when it resumes.
                for goal in self.goals.values():
                    if goal["arm"] == arm:
                        goal["xyz"] = actual["xyz"].copy()
            trans_ratio = (
                float(actual_delta @ delta / (delta @ delta))
                if np.linalg.norm(requested_delta) >= 0.005 and np.linalg.norm(delta) > 1e-9
                else None
            )
            rot_ratio = (
                float(actual_dr @ dr / (dr @ dr))
                if np.linalg.norm(requested_dr) >= 3 and np.linalg.norm(dr) > 1e-9
                else None
            )
            stalled = (trans_ratio is not None and trans_ratio < 0.25) or (
                rot_ratio is not None and rot_ratio < 0.25
            )
            same = (
                previous is not None
                and norm > 1e-6
                and direction @ previous["direction"] / norm > 0.85
            )
            count = (previous["count"] if same else 0) + 1 if stalled else 0
            # A no-op does not erase an established blocked direction.
            if norm > 1e-6:
                self.low[arm] = {"count": count, "direction": direction / norm}
            result.update(
                actual_rotation_vector_world_deg=actual_dr.tolist(),
                commanded_rotation_vector_world_deg=dr.tolist(),
                rotation_step_error_deg=float(
                    np.degrees(
                        (
                            Rotation.from_rotvec(np.radians(dr))
                            * start["rotation"]
                            * actual["rotation"].inv()
                        ).magnitude()
                    )
                ),
                low_progress_streak=self.low.get(arm, {}).get("count", 0),
                recovery_hint="Change approach or retreat; low progress alone does not prove contact"
                if stalled
                else None,
            )
            if not recovery:
                self._record(arm, after)
                self.retreat_goals.pop(arm, None)
        if evidence:
            result["sweep"] = evidence
        self.feedback[arm] = result
        return result

    def orient(self, name, args, obs):
        arm = args["arm"]
        state = obs["arms"][arm]
        current = rotation(state)
        if name == "align_axis":
            axes = state.get("geometry", {}).get("axes_local", {})
            if args["axis"] not in axes:
                raise ValueError("Adapter has not calibrated this gripper axis")
            direction = unit(args["direction_world"])
            if (
                args["axis"] == "closing"
                and current.apply(unit(axes[args["axis"]])) @ direction < 0
            ):
                direction = -direction  # Parallel-jaw closing line has no intrinsic sign.
            target = align_rotation(current, axes[args["axis"]], direction)
        elif "target_quaternion_xyzw" in args:
            q = finite_vector(args["target_quaternion_xyzw"], 4)
            if np.linalg.norm(q) < 1e-8:
                raise ValueError("Zero target quaternion")
            target = Rotation.from_quat(q)
        elif "goal_id" in args:
            goal = self.goals[args["goal_id"]]
            if goal["arm"] != arm:
                raise ValueError("Orientation goal belongs to another arm")
            target = goal["rotation"]
        else:
            raise ValueError("Provide target_quaternion_xyzw or an existing goal_id")
        if name == "rotate_toward" and "goal_id" in args and "target_quaternion_xyzw" not in args:
            goal_id = args["goal_id"]
        else:
            goal_id = f"R{len(self.goals) + 1}"
            self.goals[goal_id] = {
                "arm": arm,
                "rotation": target,
                "xyz": np.asarray(state["xyz_world_m"], float).copy(),
            }
        anchor = self.goals[goal_id]["xyz"]
        correction = anchor - np.asarray(state["xyz_world_m"])
        correction = bounded_translation(correction, self.delta_axis)
        dr, remaining = bounded_rotation(current, target, args.get("max_degrees", 15))
        self.last_goal[arm] = {
            "goal_id": goal_id,
            "target_quaternion_xyzw": target.as_quat().tolist(),
            "remaining_degrees": remaining,
            "anchor_xyz_world_m": anchor.tolist(),
        }
        result = self.move(
            obs,
            arm,
            correction,
            dr,
            None,
            args.get("steps", 12),
            check_sweep=True,
            preserve_orientation_anchor=True,
        )
        after = self.adapter.observe()["arms"][arm]
        remaining = float(np.degrees((target * rotation(after).inv()).magnitude()))
        self.last_goal[arm]["remaining_degrees"] = remaining
        position_error = float(np.linalg.norm(anchor - after["xyz_world_m"]))
        self.last_goal[arm]["anchor_position_error_m"] = position_error
        result.update(self.last_goal[arm], goal_reached=remaining <= 2 and position_error <= 0.003)
        return result

    def retreat(self, args, obs):
        arm = args["arm"]
        current = self._pose(obs["arms"][arm])
        path = self.paths[arm]
        if arm not in self.retreat_goals:
            while (
                path
                and np.linalg.norm(path[-1]["xyz"] - current["xyz"]) < 0.003
                and np.degrees((path[-1]["rotation"] * current["rotation"].inv()).magnitude()) < 2
            ):
                path.pop()
            if not path:
                raise ValueError(
                    "No earlier observed waypoint is available; choose a new approach explicitly"
                )
            self.retreat_goals[arm] = path.pop()
        target = self.retreat_goals[arm]
        delta = target["xyz"] - current["xyz"]
        limit = step_axis_limit(args, self.delta_axis)
        delta = bounded_translation(delta, limit, args.get("max_distance_m"))
        dr, _ = bounded_rotation(current["rotation"], target["rotation"])
        result = self.move(
            obs, arm, delta, dr, None, args.get("steps", 12), recovery=True, check_sweep=True
        )
        after = self._pose(self.adapter.observe()["arms"][arm])
        distance = float(np.linalg.norm(target["xyz"] - after["xyz"]))
        angle = float(np.degrees((target["rotation"] * after["rotation"].inv()).magnitude()))
        result.update(
            retreat_remaining_m=distance,
            retreat_remaining_degrees=angle,
            gripper_preserved=True,
            path_is_collision_certified=False,
            path_sampling="Observed tool-boundary poses; not an exact time-reversed dynamics path",
        )
        if distance < 0.003 and angle < 2:
            self.retreat_goals.pop(arm, None)
        return result
