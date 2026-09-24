"""Causal, non-oracle evidence of target/hand relative motion.

Uses matched visible RGB-D anchors, not changing segmentation centroids. No
attachment, contact, object identity, or task completion certificates are given.
"""

import numpy as np

from .geometry import calibration


def measured_anchors(entry, obs):
    camera = obs["vision"][entry["camera"]]
    track = entry["track"]
    uv = np.asarray(track["points_uv"], float)
    valid = np.asarray(track["visible"], bool) & ~np.asarray(track["needs_revalidation"], bool)
    h, w = camera["color"].shape[:2]
    valid &= np.isfinite(uv).all(axis=1) & (uv >= 0).all(axis=1) & (uv < [w - 1, h - 1]).all(axis=1)
    indices = np.where(valid)[0]
    pixels = np.rint(uv[indices]).astype(int)
    depth = np.asarray(camera["depth"]).squeeze()[pixels[:, 1], pixels[:, 0]]
    good = np.isfinite(depth) & (depth > 0) & (depth < 10)
    indices, pixels, depth = indices[good], pixels[good], depth[good]
    k, pose = calibration(camera)
    optical = np.linalg.solve(k, np.c_[pixels, np.ones(len(pixels))].T).T * depth[:, None]
    xyz = optical @ pose[:3, :3].T + pose[:3, 3]
    return dict(zip(indices.tolist(), xyz))


def compare_anchors(before, after, previous_arms, current_arms):
    common = sorted(set(before) & set(after))
    if len(common) < 3:
        return {"status": "unknown", "reason": "fewer_than_three_matched_depth_anchors"}
    old = np.array([before[i] for i in common])
    new = np.array([after[i] for i in common])
    displacements = new - old
    delta = np.median(displacements, axis=0)
    scatter = float(np.median(np.linalg.norm(displacements - delta, axis=1)))
    result = {
        "status": "measured_anchor_motion_hypothesis",
        "matched_anchor_count": len(common),
        "median_world_delta_m": delta.tolist(),
        "median_world_displacement_m": float(np.linalg.norm(delta)),
        "displacement_scatter_m": scatter,
        "possible_correspondence_or_rotation_change": scatter > 0.01,
        "relative_to_hands": {},
        "meaning": "Matched tracked pixels with current metric depth, NOT verified identity. Occlusion/depth-layer switches can mimic motion. Low hand-relative residual is compatible with co-motion, not proof of attachment. Inspect original views; do not infer task success.",
    }
    for arm in set(previous_arms) & set(current_arms):
        p, c = previous_arms[arm], current_arms[arm]
        old_tcp, new_tcp = np.asarray(p["xyz_world_m"]), np.asarray(c["xyz_world_m"])
        relative_rotation = np.asarray(c["axes_world"]) @ np.asarray(p["axes_world"]).T
        predicted = (old - old_tcp) @ relative_rotation.T + new_tcp
        expected = float(np.median(np.linalg.norm(predicted - old, axis=1)))
        residual = float(np.median(np.linalg.norm(new - predicted, axis=1)))
        result["relative_to_hands"][arm] = {
            "tcp_displacement_m": float(np.linalg.norm(new_tcp - old_tcp)),
            "expected_anchor_motion_if_rigidly_carried_m": expected,
            "rigid_co_motion_residual_m": residual,
            "inconsistent_with_rigid_carry_hypothesis": expected > 0.01
            and residual > max(0.008, 0.5 * expected),
        }
    return result


class InteractionEvidence:
    def __init__(self):
        self.previous = {}
        self.latest = {}

    def update(self, entries, obs):
        self.previous = {k: v for k, v in self.previous.items() if k in entries}
        self.latest = {k: v for k, v in self.latest.items() if k in entries}
        for key, entry in entries.items():
            if entry["lost"] or entry["frame"] != obs["frame_id"]:
                self.previous.pop(key, None)
                self.latest[key] = {"status": "unknown", "reason": "tracking_not_current"}
                continue
            prior = self.previous.get(key)
            if prior and prior["frame"] == obs["frame_id"] and prior["entry"] is entry:
                continue
            points = measured_anchors(entry, obs)
            result = {"status": "unknown", "reason": "need_another_physical_frame"}
            if prior and prior["entry"] is entry:
                result = compare_anchors(prior["points"], points, prior["arms"], obs["arms"])
                result["from_frame_id"] = prior["frame"]
            result["to_frame_id"] = obs["frame_id"]
            self.latest[key] = result
            self.previous[key] = {
                "entry": entry,
                "frame": obs["frame_id"],
                "points": points,
                "arms": obs["arms"],
            }

    def context(self):
        return self.latest.copy()
