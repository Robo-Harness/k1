"""Observed RGB-D regions only: no simulator imports, hidden geometry, or new views."""

import os
from copy import deepcopy
from functools import lru_cache

import cv2
import numpy as np

from .geometry import calibration
from .perception import FeatureTracker, pixel_points


@lru_cache(maxsize=2)
def segmenter(device):
    import torch
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    torch.set_num_threads(6)
    if "cuda" in device:
        torch.cuda.set_device(device)
    model = (
        build_sam3_image_model(
            enable_inst_interactivity=True,
            checkpoint_path=os.environ.get("ROBO_HARNESS_SAM3_CHECKPOINT", "./checkpoints/sam3.pt"),
            load_from_HF=False,
        )
        .to(device)
        .eval()
    )
    # Disable optional compiled hole filling on this instance for portability.
    model.inst_interactive_predictor._transforms.max_hole_area = 0
    return model, Sam3Processor(model, device=device, confidence_threshold=0.0)


def segment(rgb, box, device):
    import torch
    from PIL import Image

    model, processor = segmenter(device)
    with (
        torch.inference_mode(),
        torch.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16),
    ):
        state = processor.set_image(Image.fromarray(rgb))
        masks, scores, _ = model.predict_inst(state, box=list(box), multimask_output=True)
    masks = np.asarray(masks).reshape(-1, *rgb.shape[:2]) > 0
    mask = masks[int(np.argmax(scores))]
    if mask.sum() < 20:
        raise ValueError("Too few segmented pixels; select a clearer region")
    return mask, float(np.max(scores))


def segment_text(rgb, query, device):
    import torch
    from PIL import Image

    _, processor = segmenter(device)
    with (
        torch.inference_mode(),
        torch.autocast("cuda" if "cuda" in device else "cpu", dtype=torch.bfloat16),
    ):
        state = processor.set_image(Image.fromarray(rgb))
        result = processor.set_text_prompt(state=state, prompt=query)
    scores = result["scores"].detach().float().cpu().numpy().reshape(-1)
    masks = result["masks"].detach().cpu().numpy().reshape(-1, *rgb.shape[:2]) > 0
    selected = []
    for i in np.argsort(-scores):
        if scores[i] < 0.5 or masks[i].sum() < 20:
            continue
        if any((masks[i] & m).sum() / max(1, (masks[i] | m).sum()) > 0.5 for m, _ in selected):
            continue
        selected.append((masks[i], float(scores[i])))
        if len(selected) == 4:
            break
    return selected


def cloud(camera, mask):
    depth = np.asarray(camera["depth"]).squeeze()
    valid = mask & np.isfinite(depth) & (depth > 0) & (depth < 10)
    # Erode boundary to reduce foreground/background contamination.
    valid &= cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    if valid.sum() < 15:
        # Thin regions can disappear under erosion. Accept only locally smooth
        # measured depth, never foreground/background depth discontinuities.
        safe_depth = np.where(np.isfinite(depth) & (depth > 0), depth, 100).astype(np.float32)
        spread = cv2.dilate(safe_depth, np.ones((3, 3), np.uint8)) - cv2.erode(
            safe_depth, np.ones((3, 3), np.uint8)
        )
        valid = mask & np.isfinite(depth) & (depth > 0) & (depth < 10) & (spread < 0.01)
    v, u = np.where(valid)
    if len(u) < 15:
        raise ValueError("Insufficient interior metric depth; region geometry unknown")
    stride = max(1, len(u) // 1500)
    u, v = u[::stride], v[::stride]
    k, t = calibration(camera)
    optical = np.linalg.solve(k, np.stack([u, v, np.ones_like(u)])).T * depth[v, u, None]
    xyz = optical @ t[:3, :3].T + t[:3, 3]
    return xyz, np.stack([u, v], axis=1), float(valid.sum() / max(1, mask.sum()))


def project(camera, xyz):
    k, t = calibration(camera)
    optical = (np.asarray(xyz) - t[:3, 3]) @ t[:3, :3]
    z = optical[:, 2]
    uv = (optical @ k.T)[:, :2] / np.maximum(z[:, None], 1e-9)
    h, w = camera["color"].shape[:2]
    inside = (
        (z > 0) & np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv < [w, h]).all(axis=1)
    )
    return uv, z, inside


def geometry(camera, mask):
    xyz, uv, fraction = cloud(camera, mask)
    center = np.median(xyz, axis=0)
    _, s, vt = np.linalg.svd(xyz - center, full_matrices=False)
    normal = vt[-1]
    _, pose = calibration(camera)
    if normal @ (pose[:3, 3] - center) < 0:
        normal = -normal
    local = (xyz - center) @ vt.T
    return (
        {
            "surface_reference_world_m": center.tolist(),
            "visible_extent_along_pca_axes_m": np.ptp(local, axis=0).tolist(),
            "major_axis_world_unsigned": vt[0].tolist(),
            "major_axis_ratio": float(s[0] / max(s[1], 1e-9)),
            "surface_normal_toward_camera_world": normal.tolist(),
            "plane_rmse_m": float(np.sqrt(np.mean(((xyz - center) @ normal) ** 2))),
            "plane_reliable": bool(np.sqrt(np.mean(((xyz - center) @ normal) ** 2)) < 0.003),
            "major_axis_reliable": bool(s[0] / max(s[1], 1e-9) > 2),
            "valid_interior_depth_fraction": fraction,
            "samples": len(xyz),
            "meaning": "Visible surface statistics, not full object center/extent, grasp pose, free space, or articulation axis. PCA may be ambiguous; verify mask.",
        },
        xyz,
        uv,
    )


class Regions:
    def __init__(self, device="cuda", backend="tapnext", enabled=True):
        self.device, self.backend, self.enabled = device, backend, enabled
        self.entries = {}
        self.counter = 0
        self.find_cache = {}
        self.find_cache_frame = None

    def inspect(self, args, obs):
        camera = args["camera"]
        rgb = obs["vision"][camera]["color"]
        roi = args.get("roi")
        if "box" in args:
            roi = [args["box"][name] for name in ("left", "top", "right", "bottom")]
        corners = pixel_points(np.asarray(roi).reshape(2, 2), rgb.shape, args["coordinate_space"])
        box = corners.flatten()
        if np.any(corners[1] <= corners[0]):
            raise ValueError("ROI must be [left,top,right,bottom]")
        mask, score = segment(rgb, box, self.device)
        return self.store(
            {
                **args,
                "provenance": {
                    "method": "bbox_prompt",
                    "input_box_pixels": box.tolist(),
                    "label_used_for_segmentation": False,
                },
            },
            obs,
            mask,
            score,
        )

    def find(self, args, obs):
        frame = obs["frame_id"]
        if self.find_cache_frame != frame:
            self.find_cache.clear()
            self.find_cache_frame = frame
        cache_key = (args["camera"], args["query"])
        if cache_key in self.find_cache:
            result = deepcopy(self.find_cache[cache_key])
            result["measurement_reused"] = True
            result["meaning"] += (
                " Same frame/query: no new sensor evidence. Reuse IDs or deliberately choose a different view/region."
            )
            return result
        masks = segment_text(obs["vision"][args["camera"]]["color"], args["query"], self.device)
        results = [
            self.store(
                {
                    "camera": args["camera"],
                    "label": args["query"],
                    "provenance": {
                        "method": "text_prompt",
                        "query": args["query"],
                        "label_used_for_segmentation": True,
                    },
                },
                obs,
                mask,
                score,
            )
            for mask, score in masks
        ]
        result = {
            "candidates": results,
            "meaning": "Candidates are unordered semantic hypotheses. Select intended S ID by current contours and spatial relations, not score or numbering. No candidate means unknown, not absence.",
        }
        self.find_cache[cache_key] = deepcopy(result)
        return result

    def store(self, args, obs, mask, score):
        # Explicit reinspection changes the region registry, invalidating cached receipts.
        self.find_cache.clear()
        camera = args["camera"]
        rgb = obs["vision"][camera]["color"]
        geometry_error = None
        try:
            stats, xyz, uv = geometry(obs["vision"][camera], mask)
        except ValueError as error:
            stats, xyz = None, None
            v, u = np.where(mask)
            uv = np.stack([u, v], axis=1)
            geometry_error = str(error)
        key = args.get("region_id")
        if key and key not in self.entries:
            raise ValueError("Unknown region_id; omit it to initialize")
        if not key:
            self.counter += 1
            key = f"S{self.counter}"
        # Track a distributed set of measured interior samples, not the box corners.
        indices = np.linspace(0, len(uv) - 1, min(8, len(uv))).astype(int)
        tracker = FeatureTracker(self.backend, device=self.device)
        initial = tracker.start(rgb, uv[indices], obs["frame_id"])
        v, u = np.where(mask)
        prior = self.entries.get(key)
        history = list(prior.get("reinspections", [])) if prior else []
        if prior:
            history.append(
                {
                    "frame_id": prior["last_refresh"],
                    "camera": prior["camera"],
                    "label_proposed_by_agent": prior["label"],
                    "geometry": deepcopy(prior["stats"]),
                    "provenance": deepcopy(prior.get("provenance", {})),
                }
            )
        entry = {
            "camera": camera,
            "label": args["label"],
            "mask": mask,
            "xyz": xyz,
            "frame": obs["frame_id"],
            "tracker": tracker,
            "anchor_uv": uv[indices],
            "track": initial,
            "box": np.array([u.min(), v.min(), u.max(), v.max()], float),
            "stats": stats,
            "score": score,
            "lost": False,
            "last_refresh": obs["frame_id"],
            "geometry_error": geometry_error,
            "provenance": deepcopy(args.get("provenance", {"method": "provided_mask"})),
            "reinspections": history,
        }
        self.entries[key] = entry
        # An S ID is episode memory, not a slot in a four-entry tracker cache.
        # New searches must never destroy another tool's still-referenced ID.
        return self.summary(key, obs)

    def update(self, obs, refresh=False):
        for entry in self.entries.values():
            if entry["lost"]:
                continue
            frame = obs["frame_id"]
            camera = obs["vision"][entry["camera"]]
            if frame != entry["frame"]:
                entry["track"] = entry["tracker"].update(camera["color"], frame)
                entry["frame"] = frame
            track = entry["track"]
            valid = np.asarray(track["visible"]) & ~np.asarray(track["needs_revalidation"])
            if valid.sum() < max(3, len(valid) // 2):
                entry["lost"] = True
                entry["reason"] = "region_anchor_correspondence_lost"
                continue
            if not refresh or frame == entry["last_refresh"]:
                continue
            try:
                current = np.asarray(track["points_uv"], float)[valid]
                offset = np.median(current - entry["anchor_uv"][valid], axis=0)
                h, w = camera["color"].shape[:2]
                box = np.clip(
                    entry["box"] + np.tile(offset, 2), [0, 0, 0, 0], [w - 1, h - 1, w - 1, h - 1]
                )
                mask, score = segment(camera["color"], box, self.device)
                pix = np.rint(current).astype(int)
                inside_mask = mask[np.clip(pix[:, 1], 0, h - 1), np.clip(pix[:, 0], 0, w - 1)]
                if inside_mask.mean() < 0.6:
                    raise ValueError("New mask does not agree with tracked anchors")
                try:
                    stats, xyz, _ = geometry(camera, mask)
                    error = None
                except ValueError as exc:
                    stats, xyz, error = None, None, str(exc)
                entry.update(
                    mask=mask,
                    score=score,
                    stats=stats,
                    xyz=xyz,
                    last_refresh=frame,
                    geometry_error=error,
                )
            except (ValueError, RuntimeError) as error:
                entry.update(lost=True, reason=str(error))

    def summary(self, key, obs):
        if key not in self.entries:
            raise ValueError(
                f"Unknown region_id {key}: this ID was not created in this episode. Use a returned S ID."
            )
        entry = self.entries[key]
        v, u = np.where(entry["mask"])
        base = {
            "id": key,
            "label_proposed_by_agent": entry["label"],
            "camera": entry["camera"],
            "identity_verified": False,
            "provenance": entry.get("provenance", {"method": "provided_mask"}),
            "last_observed_frame": entry["last_refresh"],
            "mask_box_pixels_at_last_observation": [
                int(u.min()),
                int(v.min()),
                int(u.max()),
                int(v.max()),
            ],
            "revision": len(entry.get("reinspections", [])),
        }
        if entry["lost"] or entry["last_refresh"] != obs["frame_id"]:
            return {
                **base,
                "status": "unknown_reinspect",
                "reason": entry.get("reason", "geometry_not_current"),
                "historical_measurement": {
                    "frame_id": entry["last_refresh"],
                    "surface_reference_world_m": entry["stats"]["surface_reference_world_m"]
                    if entry["stats"]
                    else None,
                    "usable_as_current_geometry": False,
                },
            }
        if entry["stats"] is None:
            return {
                **base,
                "status": "mask_visible_geometry_unknown",
                "frame_id": obs["frame_id"],
                "reason": entry["geometry_error"],
                "meaning": "Verify visible green contour; select another existing view or clearer region. No XYZ or orientation inferred.",
            }
        center = np.asarray(entry["stats"]["surface_reference_world_m"])
        relations = {}
        for arm, state in obs["arms"].items():
            delta = center - state["xyz_world_m"]
            rotation = np.asarray(state["axes_world"])
            relations[arm] = {
                "delta_world_m": delta.tolist(),
                "delta_eef_body_m": (rotation.T @ delta).tolist(),
                "distance_m": float(np.linalg.norm(delta)),
            }
            axes = state.get("geometry", {}).get("axes_local", {})
            relations[arm]["delta_along_semantic_axes_m"] = {
                n: float((rotation @ np.asarray(a)) @ delta) for n, a in axes.items()
            }
            major = np.asarray(entry["stats"]["major_axis_world_unsigned"])
            relations[arm]["unsigned_axis_angles_deg"] = {
                n: float(
                    np.degrees(np.arccos(np.clip(abs((rotation @ np.asarray(a)) @ major), 0, 1)))
                )
                for n, a in axes.items()
            }
        return {
            **base,
            "status": "current_visible_surface_estimate",
            "frame_id": obs["frame_id"],
            "mask_score_not_identity_probability": entry["score"],
            **entry["stats"],
            "relative_to_robot": relations,
        }

    def context(self, obs):
        results = []
        for key in self.entries:
            result = deepcopy(self.summary(key, obs))
            if result.get("major_axis_reliable") is False:
                result.pop("major_axis_world_unsigned", None)
                for relation in result.get("relative_to_robot", {}).values():
                    relation.pop("unsigned_axis_angles_deg", None)
                result["axis_advisory"] = (
                    "Visible shape has no reliable major axis; do not infer a grasp or motion direction from PCA."
                )
            if result.get("plane_reliable") is False:
                result.pop("surface_normal_toward_camera_world", None)
                result["plane_advisory"] = (
                    "Visible surface is not reliably planar; plane normal omitted from decision context."
                )
            results.append(result)
        return results

    def refresh_cross_views(self, obs):
        """Reproject CURRENT observed surfaces, never unseen completed geometry."""
        if not self.enabled:
            return
        for key, entry in self.entries.items():
            if self.summary(key, obs)["status"] != "current_visible_surface_estimate":
                continue
            for camera in obs["vision"]:
                if camera != entry["camera"]:
                    self.cross_view(key, camera, obs)

    def cross_view(self, key, target, obs):
        if self.summary(key, obs)["status"] != "current_visible_surface_estimate":
            raise ValueError("Region is not current; inspect_region again")
        entry = self.entries[key]
        if target == entry["camera"]:
            raise ValueError("Choose the other existing camera")
        camera = obs["vision"][target]
        uv, z, inside = project(camera, entry["xyz"])
        selected = np.where(inside)[0]
        if len(selected):
            pixels = np.floor(uv[selected]).astype(int)
            depth = np.asarray(camera["depth"]).squeeze()[pixels[:, 1], pixels[:, 0]]
            good = np.isfinite(depth) & (depth > 0) & (abs(depth - z[selected]) < 0.008)
            selected = selected[good]
        entry["cross"] = {"camera": target, "frame": obs["frame_id"], "pixels": uv[selected]}
        return {
            "region_id": key,
            "camera": target,
            "projected_in_image_samples": int(inside.sum()),
            "depth_consistent_samples": len(selected),
            "meaning": "Geometric corroboration only, NOT semantic correspondence. No fusion or unseen-surface completion. Orange dots on existing camera image require visual identity checking.",
        }

    def relation(self, args, obs):
        a = self.summary(args["region_id"], obs)
        if a["status"] != "current_visible_surface_estimate":
            raise ValueError("Region geometry stale; reinspect")
        result = {"region": a}
        if args.get("other_region_id"):
            b = self.summary(args["other_region_id"], obs)
            if b["status"] != "current_visible_surface_estimate":
                raise ValueError("Other region geometry stale")
            delta = np.asarray(b["surface_reference_world_m"]) - a["surface_reference_world_m"]
            normal = np.asarray(a["surface_normal_toward_camera_world"])
            result.update(
                delta_reference_world_m=delta.tolist(),
                distance_between_surface_references_m=float(np.linalg.norm(delta)),
                signed_distance_to_first_fitted_plane_m=float(delta @ normal),
                plane_distance_caveat="Only useful for low plane RMSE; not physical object clearance",
            )
        return result

    def overlay(self, rgb, camera, obs):
        result = rgb.copy()
        if self.enabled:
            h, w = result.shape[:2]
            for state in obs["arms"].values():
                tcp = np.asarray(state["xyz_world_m"])
                rotation = np.asarray(state["axes_world"])
                for name, local in state.get("geometry", {}).get("axes_local", {}).items():
                    axis = rotation @ np.asarray(local)
                    uv, _, valid = project(obs["vision"][camera], [tcp, tcp + 0.025 * axis])
                    if valid.all():
                        p, q = np.rint(uv).astype(int)
                        color = {
                            "approach": (255, 80, 80),
                            "closing": (80, 160, 255),
                            "lateral": (200, 100, 255),
                        }[name]
                        cv2.arrowedLine(result, tuple(p), tuple(q), color, 1, tipLength=0.2)
                        cv2.putText(result, name[0].upper(), tuple(q), 0, 0.4, color, 1)
            for x in range(50, w, 50):
                cv2.putText(result, str(x), (x - 8, 12), 0, 0.3, (255, 255, 255), 1)
                cv2.line(result, (x, 14), (x, 18), (255, 255, 255), 1)
            for y in range(50, h, 50):
                cv2.putText(result, str(y), (1, y), 0, 0.3, (255, 255, 255), 1)
        for key, entry in self.entries.items():
            if entry["lost"] or entry["last_refresh"] != obs["frame_id"]:
                continue
            if entry["camera"] == camera:
                contours, _ = cv2.findContours(
                    entry["mask"].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                cv2.drawContours(result, contours, -1, (80, 255, 80), 1)
                # Round bowls have no reliable PCA axis; their ID must still be
                # visible, independently of whether an orientation line is drawn.
                v, u = np.where(entry["mask"])
                cv2.putText(
                    result, key, (int(u.min()), max(12, int(v.min()))), 0, 0.5, (80, 255, 80), 1
                )
                if entry["stats"] is None:
                    v, u = np.where(entry["mask"])
                    cv2.putText(
                        result,
                        key + " ?",
                        (int(u.min()), max(12, int(v.min()))),
                        0,
                        0.5,
                        (80, 255, 80),
                        1,
                    )
                    continue
                center = np.asarray(entry["stats"]["surface_reference_world_m"])
                axis = np.asarray(entry["stats"]["major_axis_world_unsigned"])
                points = np.array([center - 0.02 * axis, center + 0.02 * axis])
                uv, _, valid = project(obs["vision"][camera], points)
                if valid.all() and entry["stats"].get("major_axis_reliable", False):
                    p, q = np.rint(uv).astype(int)
                    cv2.line(result, tuple(p), tuple(q), (255, 170, 30), 2)
                for state in obs["arms"].values():
                    uv, _, valid = project(obs["vision"][camera], [state["xyz_world_m"], center])
                    if valid.all():
                        p, q = np.rint(uv).astype(int)
                        cv2.line(result, tuple(p), tuple(q), (200, 100, 255), 1)
            cross = entry.get("cross", {})
            if cross.get("camera") == camera and cross.get("frame") == obs["frame_id"]:
                for pixel in cross["pixels"][:: max(1, len(cross["pixels"]) // 60)]:
                    cv2.circle(result, tuple(np.rint(pixel).astype(int)), 1, (255, 160, 20), -1)
                if len(cross["pixels"]):
                    center = np.rint(np.median(cross["pixels"], axis=0)).astype(int)
                    cv2.putText(result, key + "?", tuple(center), 0, 0.45, (255, 160, 20), 1)
        return result

    def schemas(self, schema, cameras):
        if not self.enabled:
            return []
        text = {"type": "string"}
        camera = {"type": "string", "enum": list(cameras)}
        fid = {"type": "integer"}
        return [
            schema(
                "find_regions",
                "Find visible candidate regions by a short object/part noun phrase using SAM3 on CURRENT original RGB, then return measured RGB-D geometry. Prefer this when pixel grounding is uncertain. Multiple matching parts receive S IDs: choose the intended one from labeled ORIGINAL view; score/order is not task identity.",
                {"frame_id": fid, "camera": camera, "query": text},
                ["frame_id", "camera", "query"],
            ),
            schema(
                "inspect_region",
                "Segment using the supplied BBOX on current RGB-D. Label is only your proposed name, NOT a semantic prompt or verified identity. Check the S-labeled contour matches your intended object. Returns measured surface geometry, not a grasp pose. Refresh an existing region_id only for the same intended part; new searches do not delete previous IDs.",
                {
                    "frame_id": fid,
                    "camera": camera,
                    "box": {
                        "type": "object",
                        "properties": {
                            "left": {
                                "type": "number",
                                "description": "Minimum horizontal u/x; not y",
                            },
                            "top": {"type": "number", "description": "Minimum vertical v/y; not x"},
                            "right": {"type": "number", "description": "Maximum horizontal u/x"},
                            "bottom": {"type": "number", "description": "Maximum vertical v/y"},
                        },
                        "required": ["left", "top", "right", "bottom"],
                        "additionalProperties": False,
                    },
                    "coordinate_space": {
                        "type": "string",
                        "enum": ["pixels", "normalized_01", "normalized_1000"],
                    },
                    "label": text,
                    "region_id": text,
                },
                ["frame_id", "camera", "box", "coordinate_space", "label"],
            ),
            schema(
                "region_relation",
                "Compute current region-to-robot relation, optionally region-to-region displacement and fitted plane distance. Not collision guarantee.",
                {"frame_id": fid, "region_id": text, "other_region_id": text},
                ["frame_id", "region_id"],
            ),
            schema(
                "check_region_view",
                "Project observed region into the OTHER existing RGB-D camera and depth-check it. No new camera, no completion, no semantic match guarantee; inspect orange points.",
                {"frame_id": fid, "region_id": text, "camera": camera},
                ["frame_id", "region_id", "camera"],
            ),
        ]
