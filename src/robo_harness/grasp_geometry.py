"""Generic parallel-jaw pose hypotheses from visible points and robot geometry.

No object models or simulator state; sparse visibility scores are NOT collision
certificates. The agent must choose/verify a candidate and execute bounded moves.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from .geometry import calibration
from .regions import project
from .self_geometry import visible_self_mask


def visibility_score(points, obs, arm):
    free = np.zeros(len(points), bool)
    risk = np.zeros(len(points), bool)
    state = obs["arms"][arm]
    local_probes = np.asarray(state["geometry"]["sweep_probes_local_m"])
    real_r = np.asarray(state["axes_world"])
    real_tcp = np.asarray(state["xyz_world_m"])
    for camera in obs["vision"].values():
        uv, z, inside = project(camera, points)
        indices = np.where(inside)[0]
        if not len(indices):
            continue
        pix = np.floor(uv[indices]).astype(int)
        depth = np.asarray(camera["depth"]).squeeze()[pix[:, 1], pix[:, 0]]
        valid = np.isfinite(depth) & (depth > 0)
        free[indices] |= valid & (z[indices] < depth - 0.005)
        near = valid & (abs(z[indices] - depth) < 0.005)
        k, t = calibration(camera)
        surface = (np.linalg.solve(k, np.c_[pix, np.ones(len(pix))].T).T * depth[:, None]) @ t[
            :3, :3
        ].T + t[:3, 3]
        local = (surface - real_tcp) @ real_r
        own_hand = visible_self_mask(camera, arm, surface)
        if own_hand is None:
            own_hand = (
                (local >= local_probes.min(axis=0) - 0.015)
                & (local <= local_probes.max(axis=0) + 0.015)
            ).all(axis=1)
        risk[indices] |= near & ~own_hand
    return {
        "observed_free_probe_fraction": float(free.mean()),
        "near_surface_probe_fraction": float(risk.mean()),
        "unknown_probe_fraction": float((~free & ~risk).mean()),
    }


def candidates(manager, region_id, arm, obs):
    summary = manager.summary(region_id, obs)
    if summary["status"] != "current_visible_surface_estimate":
        raise ValueError("Need a current measured region")
    if not summary["major_axis_reliable"]:
        raise ValueError(
            "Visible region has no reliable long axis; use manual orientation or a clearer part"
        )
    state = obs["arms"][arm]
    if state.get("gripper_opening_m", 0) < 0.025:
        raise ValueError("Open gripper before evaluating open-jaw candidates")
    geom = state["geometry"]
    local_basis = np.column_stack(
        [geom["axes_local"][n] for n in ["approach", "closing", "lateral"]]
    )
    probes = np.asarray(geom["sweep_probes_local_m"])
    pad_midpoint = np.mean(geom["finger_pad_centers_local_m"], axis=0)
    center = np.asarray(summary["surface_reference_world_m"])
    long = np.asarray(summary["major_axis_world_unsigned"])
    seed = np.eye(3)[np.argmin(abs(long))]
    a = np.cross(long, seed)
    a /= np.linalg.norm(a)
    b = np.cross(long, a)
    xyz = manager.entries[region_id]["xyz"]
    output = []
    for angle in np.arange(8) * np.pi / 4:
        approach = np.cos(angle) * a + np.sin(angle) * b
        closing = np.cross(approach, long)
        rotation = np.column_stack([approach, closing, long]) @ local_basis.T
        flipped = np.column_stack([approach, -closing, -long]) @ local_basis.T
        current_rotation = np.asarray(state["axes_world"])
        if (
            Rotation.from_matrix(flipped @ current_rotation.T).magnitude()
            < Rotation.from_matrix(rotation @ current_rotation.T).magnitude()
        ):
            rotation, closing = flipped, -closing
        target = center - rotation @ pad_midpoint + 0.003 * approach
        pre = target - 0.07 * approach
        width = float(np.ptp(xyz @ closing))
        samples = np.concatenate(
            [probes @ rotation.T + pre + f * (target - pre) for f in (0, 0.25, 0.5, 0.75, 1)]
        )
        score = visibility_score(samples, obs, arm)
        rank = (
            score["observed_free_probe_fraction"]
            - 2 * score["near_surface_probe_fraction"]
            - 0.25 * score["unknown_probe_fraction"]
        )
        fits = width < state["gripper_opening_m"] - 0.008
        if not fits:
            rank -= 1
        output.append(
            {
                "target_quaternion_xyzw": Rotation.from_matrix(rotation).as_quat().tolist(),
                "approach_world": approach.tolist(),
                "closing_world_unsigned": closing.tolist(),
                "pregrasp_tcp_world_m": pre.tolist(),
                "candidate_tcp_world_m": target.tolist(),
                "observed_width_along_closing_m": width,
                "visible_width_fits_open_jaws": fits,
                "score": rank,
                **score,
            }
        )
    output.sort(key=lambda x: x["score"], reverse=True)
    for i, candidate in enumerate(output[:3]):
        candidate["candidate_id"] = f"{region_id}-C{i + 1}"
    return {
        "region_id": region_id,
        "frame_id": obs["frame_id"],
        "candidates": output[:3],
        "meaning": "POSE HYPOTHESES, not verified grasps. Aligns pad midpoint to visible reference with 3mm inward hypothesis; unseen extent remains unknown. Sparse endpoint/approach probes omit meshes between samples, held objects and arm collisions. Pregrasp approach is not a planned path from current robot pose. Choose clearance, rotate in bounded steps, move, settle closure, verify attachment.",
    }
