"""Causal image correspondence plus calibrated world displacement, never object truth."""

import cv2
import numpy as np

from .geometry import unproject
from .perception import FeatureTracker


class TrackedPoints:
    def __init__(self, capacity=12, backend="lk", device="cpu"):
        if backend not in ("lk", "tapnext"):
            raise ValueError("Tracker backend must be lk or tapnext")
        self.entries = {}
        self.capacity = capacity
        self.backend = backend
        self.device = device

    def start(self, point, obs):
        camera = point["camera"]
        tracker = FeatureTracker(self.backend, device=self.device)
        initial = tracker.start(obs["vision"][camera]["color"], [point["pixel"]], obs["frame_id"])
        # Keep point identity for the whole episode. `capacity` is retained as a
        # constructor compatibility argument, not permission to delete memory.
        self.entries[point["id"]] = {
            "tracker": tracker,
            "camera": camera,
            "origin": np.array(point["xyz_world_m"]),
            "last_xyz": np.array(point["xyz_world_m"]),
            "last_frame": obs["frame_id"],
            "lost": False,
            "latest": {
                "id": point["id"],
                "camera": camera,
                "frame_id": obs["frame_id"],
                "pixel": point["pixel"],
                "xyz_world_m": point["xyz_world_m"],
                "status": "initialized",
                "world_displacement_m": [0, 0, 0],
                "motion_evidence": "below_5mm_threshold",
                "identity_verified": False,
                "backend": self.backend,
            },
        }
        if not initial["visible"][0] or initial["needs_revalidation"][0]:
            self.invalidate([point["id"]], "initial_correspondence_uncertain", obs["frame_id"])

    def invalidate(self, point_ids, reason, frame):
        """Explicit retirement, e.g. observed contact/occlusion; never auto-reacquire."""
        if not reason or any(key not in self.entries for key in point_ids):
            raise ValueError("Supply existing point IDs and an invalidation reason")
        for key in point_ids:
            entry = self.entries[key]
            entry["lost"] = True
            entry["loss_reason"] = reason
            entry["latest"] = {
                "id": key,
                "camera": entry["camera"],
                "frame_id": frame,
                "backend": self.backend,
                "identity_verified": False,
                "status": "lost_remeasure",
                "motion_evidence": "unknown",
                "reason": reason,
            }

    def update(self, obs):
        for key, entry in self.entries.items():
            frame = obs["frame_id"]
            if frame == entry["last_frame"]:
                continue
            entry["last_frame"] = frame
            base = {
                "id": key,
                "camera": entry["camera"],
                "frame_id": frame,
                "identity_verified": False,
                "backend": "lk_forward_backward" if self.backend == "lk" else "tapnext",
            }
            if entry["lost"]:
                entry["latest"] = {
                    **base,
                    "status": "lost_remeasure",
                    "motion_evidence": "unknown",
                    "reason": entry.get("loss_reason", "previous_correspondence_lost"),
                }
                continue
            try:
                camera = obs["vision"][entry["camera"]]
                tracked = entry["tracker"].update(camera["color"], frame)
                if not tracked["visible"][0] or tracked["needs_revalidation"][0]:
                    raise ValueError("correspondence_lost_or_frame_gap")
                uv = tracked["points_uv"][0]
                measured = unproject(camera, uv)
                u, v = measured["pixel"]
                depth = np.asarray(camera["depth"]).squeeze()
                patch = depth[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2]
                if not np.isfinite(patch).all() or np.any(patch <= 0) or np.ptp(patch) > 0.02:
                    raise ValueError("ambiguous_depth_edge")
                xyz = np.array(measured["xyz_world_m"])
                if np.linalg.norm(xyz - entry["last_xyz"]) > 0.05:
                    raise ValueError("world_jump_revalidation_required")
                delta = xyz - entry["origin"]
                entry["last_xyz"] = xyz
                entry["latest"] = {
                    **base,
                    "pixel": measured["pixel"],
                    "xyz_world_m": xyz.tolist(),
                    "status": "tracked_estimate",
                    "world_displacement_m": delta.tolist(),
                    "world_distance_m": float(np.linalg.norm(delta)),
                    "motion_evidence": "above_5mm_threshold"
                    if np.linalg.norm(delta) > 0.005
                    else "below_5mm_threshold",
                    "meaning": "Same-feature estimate relative to initialization; camera pose compensated. Drift/identity switches remain possible; not verified object motion.",
                }
            except (ValueError, KeyError, RuntimeError, cv2.error) as error:
                entry["lost"] = True
                entry["loss_reason"] = str(error)
                entry["latest"] = {
                    **base,
                    "status": "lost_remeasure",
                    "motion_evidence": "unknown",
                    "reason": str(error),
                }

    def context(self):
        def compact(value):
            if isinstance(value, float):
                return round(value, 5)
            if isinstance(value, list):
                return [compact(v) for v in value]
            if isinstance(value, dict):
                return {k: compact(v) for k, v in value.items() if k != "meaning"}
            return value

        result = []
        for entry in self.entries.values():
            item = compact(entry["latest"])
            if entry["lost"]:
                item["historical_xyz_world_m_not_current"] = compact(entry["last_xyz"].tolist())
            result.append(item)
        return result

    def overlay(self, rgb, camera, frame):
        result = rgb.copy()
        for item in self.context():
            if item["camera"] != camera or item["frame_id"] != frame or "pixel" not in item:
                continue
            uv = tuple(item["pixel"])
            cv2.circle(result, uv, 5, (0, 220, 255), 1)
            cv2.putText(
                result,
                item["id"] + "~",
                (max(0, uv[0] - 10), max(12, uv[1] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 220, 255),
                1,
            )
        return result
