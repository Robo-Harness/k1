"""Task-independent, causal RGB-D geometry and feature tracking.

No object poses, rewards, task names, or future frames are inputs. Models are
optional and loaded read-only. References distinguish estimates from identity.
"""

from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .geometry import calibration, unproject


def finite_vector(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError(f"Expected {size} finite numbers")
    return value


def pixel_points(points, shape, coordinate_space="pixels"):
    """Explicit conversion only; never silently guess a VLM's coordinate convention."""
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError("Expected finite Nx2 image coordinates")
    if coordinate_space not in ("pixels", "normalized_01", "normalized_1000"):
        raise ValueError("Specify pixels, normalized_01 or normalized_1000")
    h, w = shape[:2]
    if coordinate_space != "pixels":
        limit = 1 if coordinate_space == "normalized_01" else 1000
        if np.any(points < 0) or np.any(points > limit):
            raise ValueError("Coordinates outside declared normalized range")
        points = points / limit * np.array([w - 1, h - 1])
    elif (
        len(points) and np.all((points >= 0) & (points <= 1)) and np.any(points != np.round(points))
    ):
        raise ValueError(
            "Ambiguous subpixel corner coordinates: declare normalized_01 explicitly if normalized"
        )
    return points


def fit_geometry(camera, roi, kind="plane", threshold_m=0.003):
    """Robust plane or PCA axis from a caller-selected current RGB-D rectangle."""
    if kind not in ("plane", "axis"):
        raise ValueError("Geometry kind must be plane or axis")
    x0, y0, x1, y1 = finite_vector(roi, 4).astype(int)
    if "depth" not in camera:
        raise ValueError("Metric geometry unavailable: this observation has no measured depth")
    depth = np.asarray(camera["depth"]).squeeze()
    h, w = depth.shape
    if not (0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h):
        raise ValueError("ROI must be a nonempty original-image rectangle")
    if not 0 < threshold_m <= 0.02:
        raise ValueError("Plane threshold must be in (0, 0.02] m")
    k, t = calibration(camera)
    stride = max(1, int(np.ceil(np.sqrt((x1 - x0) * (y1 - y0) / 3000))))
    v, u = np.mgrid[y0:y1:stride, x0:x1:stride]
    z = depth[v, u].ravel()
    valid = np.isfinite(z) & (z > 0) & (z < 10)
    uv1 = np.stack((u.ravel(), v.ravel(), np.ones(u.size)), axis=1)[valid]
    if len(uv1) < 20:
        raise ValueError("Too few valid depth samples in ROI")
    xyz = (np.linalg.solve(k, uv1.T).T * z[valid, None]) @ t[:3, :3].T + t[:3, 3]
    inliers = np.ones(len(xyz), dtype=bool)
    if kind == "plane":
        best = np.zeros(len(xyz), dtype=bool)
        rng = np.random.default_rng(0)
        for _ in range(120):
            p = xyz[rng.choice(len(xyz), 3, replace=False)]
            n = np.cross(p[1] - p[0], p[2] - p[0])
            length = np.linalg.norm(n)
            if length < 1e-9:
                continue
            n /= length
            selected = np.abs((xyz - p[0]) @ n) <= threshold_m
            if selected.sum() > best.sum():
                best = selected
        if best.sum() < 20 or best.mean() < 0.5:
            raise ValueError("No dominant plane; select a clearer ROI")
        inliers = best
    center = np.median(xyz[inliers], axis=0)
    _, s, vt = np.linalg.svd(xyz[inliers] - center, full_matrices=False)
    direction = vt[-1 if kind == "plane" else 0]
    if direction[np.argmax(np.abs(direction))] < 0:
        direction = -direction
    residual = (xyz[inliers] - center) @ direction
    result = {
        "kind": kind,
        "xyz_world_m": center.tolist(),
        "direction_world": direction.tolist(),
        "samples": len(xyz),
        "inliers": int(inliers.sum()),
        "inlier_fraction": float(inliers.mean()),
        "singular_values": s.tolist(),
        "roi": [int(x) for x in (x0, y0, x1, y1)],
        "meaning": "Geometry of selected visible surfaces; semantic identity and physical extent unverified",
    }
    if kind == "plane":
        result.update(
            offset_m=float(-center @ direction), rmse_m=float(np.sqrt(np.mean(residual**2)))
        )
    else:
        result.update(
            axis_ratio=float(s[0] / max(s[1], 1e-9)),
            visible_extent_m=float(np.ptp(residual)),
            axis_sign_ambiguous=True,
        )
    return result


def ray_plane(camera, uv, plane):
    """Intersect a pixel ray with an estimated plane, not the depth behind it."""
    uv = finite_vector(uv, 2)
    h, w = np.asarray(camera["depth"]).squeeze().shape
    if not (0 <= uv[0] < w and 0 <= uv[1] < h):
        raise ValueError("Pixel outside original image")
    k, t = calibration(camera)
    origin = t[:3, 3]
    ray = t[:3, :3] @ np.linalg.solve(k, [*uv, 1.0])
    n = finite_vector(plane["direction_world"], 3)
    denom = float(n @ ray)
    incidence = abs(denom) / (np.linalg.norm(ray) * np.linalg.norm(n))
    if incidence < 0.05:
        raise ValueError("Grazing ray-plane intersection is ill-conditioned")
    distance = -(float(n @ origin) + plane["offset_m"]) / denom
    if distance <= 0 or distance > 10:
        raise ValueError("Intersection behind camera or outside supported range")
    return {
        "kind": "constructed_point",
        "xyz_world_m": (origin + distance * ray).tolist(),
        "ray_plane_cosine": float(incidence),
        "meaning": "Ray-plane construction; not a measured surface or a verified object/hole center",
    }


def relation(a, b):
    delta = finite_vector(b["xyz_world_m"], 3) - finite_vector(a["xyz_world_m"], 3)
    out = {"delta_world_m": delta.tolist(), "distance_m": float(np.linalg.norm(delta))}
    if b.get("kind") == "plane":
        out["signed_distance_to_plane_m"] = float(
            np.dot(a["xyz_world_m"], b["direction_world"]) + b["offset_m"]
        )
    if "direction_world" in a and "direction_world" in b:
        u, v = np.array(a["direction_world"]), np.array(b["direction_world"])
        out["unsigned_axis_angle_deg"] = float(
            np.degrees(
                np.arccos(np.clip(abs(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)), 0, 1))
            )
        )
    return out


def execution_report(start, target, actual, ik_failures=0):
    start, target, actual = (finite_vector(x, 3) for x in (start, target, actual))
    command, achieved = target - start, actual - start
    length = float(np.linalg.norm(command))
    ratio = float(achieved @ command / length**2) if length > 1e-5 else None
    return {
        "commanded_translation_m": command.tolist(),
        "actual_translation_m": achieved.tolist(),
        "actual_minus_target_m": (actual - target).tolist(),
        "axial_progress_ratio": ratio,
        "lateral_displacement_m": float(np.linalg.norm(achieved - command * (ratio or 0))),
        "status": "ik_failure"
        if ik_failures
        else "no_translation"
        if ratio is None
        else "undertracking"
        if ratio < 0.5
        else "tracked",
        "meaning": "Kinematic evidence only; undertracking does not prove contact",
    }


@lru_cache(maxsize=2)
def _load_tapnext(repo, checkpoint, device):
    sys.path.insert(0, str(repo))
    from tapnet.tapnextpp.votsp2026.model import TAPNextPP

    return TAPNextPP.from_checkpoint(checkpoint, device=device)


class FeatureTracker:
    """Streaming point tracker. Lost LK points never silently reacquire identity."""

    def __init__(self, backend="lk", device="cpu"):
        self.backend, self.device = backend, device
        self.model = None
        if backend == "tapnext":
            repo = Path(
                os.environ.get(
                    "ROBO_HARNESS_TAPNEXT_REPO",
                    "./external/tapnextpp",
                )
            )
            checkpoint = Path(
                os.environ.get(
                    "ROBO_HARNESS_TAPNEXT_CHECKPOINT",
                    "./checkpoints/tapnextpp.pt",
                )
            )
            if not repo.is_dir() or not checkpoint.is_file():
                raise ValueError(
                    "TAPNext repository/checkpoint unavailable; configure explicit local paths"
                )
            self.model = _load_tapnext(str(repo), str(checkpoint), device)
        elif backend != "lk":
            raise ValueError("Tracker backend must be lk or tapnext")
        self.frame = None
        self.history = []

    def start(self, rgb, points, frame_id):
        points = np.asarray(points, dtype=np.float32)
        h, w = rgb.shape[:2]
        self.image_size_wh = [int(w), int(h)]
        if points.ndim != 2 or points.shape[1] != 2 or not 1 <= len(points) <= 64:
            raise ValueError("Select 1 to 64 original-image points")
        if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= [w, h]):
            raise ValueError("Tracking points outside original image")
        self.points = self.initial = points.copy()
        self.visible = np.ones(len(points), dtype=bool)
        self.continuity_broken = np.zeros(len(points), dtype=bool)
        self.last_frame_gap = 0
        self.gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.model is not None:
            self.points, self.visible, self.state = self.model.track_frame(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), query_points_xy=points
            )
        self.continuity_broken |= ~self.visible
        self.frame = int(frame_id)
        self.history = []
        return self._result()

    def update(self, rgb, frame_id):
        if self.frame is None or frame_id < self.frame:
            raise ValueError("Tracker requires initialization and monotonic frame IDs")
        if frame_id == self.frame:
            return self.history[-1]
        self.last_frame_gap = int(frame_id - self.frame)
        self.continuity_broken |= ~self.visible
        if self.last_frame_gap > 8:
            self.continuity_broken[:] = True
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if self.model is not None:
            self.points, self.visible, self.state = self.model.track_frame(
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), state=self.state
            )
        else:
            prev = self.points.astype(np.float32).reshape(-1, 1, 2)
            nxt, status, _ = cv2.calcOpticalFlowPyrLK(
                self.gray, gray, prev, None, winSize=(21, 21), maxLevel=3
            )
            if nxt is None:
                self.visible[:] = False
            else:
                back, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    gray, self.gray, nxt, None, winSize=(21, 21), maxLevel=3
                )
                if back is None:
                    self.visible[:] = False
                else:
                    err = np.linalg.norm(back.reshape(-1, 2) - self.points, axis=1)
                    self.visible &= (
                        status.ravel().astype(bool) & back_status.ravel().astype(bool) & (err < 1.5)
                    )
                self.points = nxt.reshape(-1, 2)
        h, w = rgb.shape[:2]
        self.visible &= (
            np.isfinite(self.points).all(axis=1)
            & (self.points >= 0).all(axis=1)
            & (self.points < [w, h]).all(axis=1)
        )
        self.continuity_broken |= ~self.visible
        self.gray, self.frame = gray, int(frame_id)
        return self._result()

    def _result(self):
        result = {
            "frame_id": self.frame,
            "backend": self.backend,
            "coordinate_space": "pixels",
            "image_size_wh": self.image_size_wh,
            "points_uv": [p.tolist() if np.isfinite(p).all() else None for p in self.points],
            "visible": self.visible.tolist(),
            "visible_count": int(self.visible.sum()),
            "needs_revalidation": self.continuity_broken.tolist(),
            "last_frame_gap": self.last_frame_gap,
            "identity_verified": False,
            "median_displacement_px": np.median(
                (self.points - self.initial)[self.visible], axis=0
            ).tolist()
            if self.visible.any()
            else None,
            "meaning": "Image feature correspondence; camera motion can also cause pixel motion; not verified object identity",
        }
        self.history.append(result)
        self.history = self.history[-32:]
        return result


class GenericTools:
    def __init__(self):
        self.references = {}
        self.tracks = {}
        self.active_reference = None

    def add(self, result, frame, camera=None):
        ref = f"G{len(self.references) + 1}"
        self.references[ref] = {**result, "id": ref, "frame_id": int(frame), "camera": camera}
        self.active_reference = ref
        return self.references[ref]

    def resolve(self, ref, frame, points=()):
        result = self.references.get(ref)
        if result is None:
            result = next((p for p in points if p["id"] == ref), None)
        if result is None:
            raise ValueError("Unknown geometry reference")
        if result["frame_id"] != frame:
            raise ValueError("Geometry reference is historical; re-measure in CURRENT frame")
        return result

    def update(self, obs, frame):
        for entry in self.tracks.values():
            if entry.get("failed"):
                entry["latest"]["frame_id"] = frame
                continue
            try:
                entry["latest"] = entry["tracker"].update(
                    np.asarray(obs["vision"][entry["camera"]]["color"])[..., :3], frame
                )
            except (ValueError, KeyError, RuntimeError, cv2.error) as exc:
                entry["failed"] = True
                entry["latest"] = {
                    "frame_id": frame,
                    "status": "tracker_failed_reset_required",
                    "error": str(exc),
                    "points_uv": [],
                    "visible": [],
                    "visible_count": 0,
                    "identity_verified": False,
                    "needs_revalidation": [],
                }

    def execute(self, name, args, obs, frame, points=()):
        if args.get("frame_id") != frame:
            raise ValueError("Stale tool request; use CURRENT frame_id")
        if name == "fit_geometry":
            camera = args["camera"]
            return self.add(
                fit_geometry(obs["vision"][camera], args["roi"], args["kind"]), frame, camera
            )
        if name == "project_to_plane":
            plane = self.resolve(args["plane_id"], frame, points)
            if plane.get("kind") != "plane":
                raise ValueError("Reference must be a fitted plane")
            cam = obs["vision"][args["camera"]]
            uv = pixel_points(
                [args["uv"]], np.asarray(cam["color"]).shape, args.get("coordinate_space", "pixels")
            )[0]
            return self.add(ray_plane(cam, uv, plane), frame, args["camera"])
        if name == "compute_relation":
            return relation(
                self.resolve(args["a"], frame, points), self.resolve(args["b"], frame, points)
            )
        if name == "track_features":
            key = args["track_id"]
            if not isinstance(key, str) or not 1 <= len(key) <= 80:
                raise ValueError("Use a short stable track ID")
            if key in self.tracks or len(self.tracks) >= 8:
                raise ValueError("Track ID exists or track limit reached; reset explicitly")
            camera = args["camera"]
            tracker = FeatureTracker(
                os.environ.get("ROBO_HARNESS_TRACKER_BACKEND", "lk"),
                os.environ.get("ROBO_HARNESS_TRACKER_DEVICE", "cpu"),
            )
            rgb = np.asarray(obs["vision"][camera]["color"])[..., :3]
            pixels = pixel_points(
                args["points_uv"], rgb.shape, args.get("coordinate_space", "pixels")
            )
            result = tracker.start(rgb, pixels, frame)
            self.tracks[key] = {"camera": camera, "tracker": tracker, "latest": result}
            return {"track_id": key, **result}
        if name == "read_track":
            entry = self.tracks[args["track_id"]]
            result = {"track_id": args["track_id"], **entry["latest"]}
            # Invalid depth stays invalid; never project an occluded point onto background.
            xyz = []
            for uv, visible, revalidate in zip(
                result["points_uv"], result["visible"], result["needs_revalidation"]
            ):
                try:
                    xyz.append(
                        unproject(obs["vision"][entry["camera"]], uv)["xyz_world_m"]
                        if visible and not revalidate
                        else None
                    )
                except (ValueError, KeyError):
                    xyz.append(None)
            return {
                **result,
                "surface_xyz_world_m": xyz,
                "depth_warning": "Current visible pixel surface; tracking visibility may be wrong",
            }
        if name == "reset_track":
            del self.tracks[args["track_id"]]
            return {"removed": args["track_id"]}
        raise ValueError("Unknown generic tool")

    def context(self):
        return {
            "geometry": list(self.references.values())[-8:],
            "tracks": {k: {"camera": v["camera"], **v["latest"]} for k, v in self.tracks.items()},
        }


def tool_schemas(schema, cameras):
    text, integer = {"type": "string"}, {"type": "integer"}

    def array(n):
        return {"type": "array", "items": {"type": "number"}, "minItems": n, "maxItems": n}

    camera = {"type": "string", "enum": list(cameras)}
    coordinates = {"type": "string", "enum": ["pixels", "normalized_01", "normalized_1000"]}
    specs = [
        (
            "fit_geometry",
            "Fit a plane or axis to a CURRENT RGB-D ROI. No semantic identity guarantee.",
            {
                "camera": camera,
                "roi": array(4),
                "kind": {"type": "string", "enum": ["plane", "axis"]},
            },
        ),
        (
            "project_to_plane",
            "Intersect CURRENT pixel ray with a CURRENT fitted plane. Returns constructed point, not depth behind pixel.",
            {"camera": camera, "uv": array(2), "plane_id": text, "coordinate_space": coordinates},
        ),
        (
            "compute_relation",
            "Distance/vector and optional plane distance or axis angle between CURRENT P/G references.",
            {"a": text, "b": text},
        ),
        (
            "track_features",
            "Initialize causal tracking of 1-64 visible feature pixels. Does not prove object identity. Runs on subsequent observations.",
            {
                "track_id": text,
                "camera": camera,
                "coordinate_space": coordinates,
                "points_uv": {"type": "array", "items": array(2), "minItems": 1, "maxItems": 64},
            },
        ),
        (
            "read_track",
            "Read current tracked pixels, visibility, displacement and available depth. Lost/occluded features are uncertain.",
            {"track_id": text},
        ),
        (
            "reset_track",
            "Explicitly discard a lost or unwanted tracker before reinitializing its ID.",
            {"track_id": text},
        ),
    ]
    return [
        schema(n, d, {"frame_id": integer, **props}, ["frame_id", *props]) for n, d, props in specs
    ]
