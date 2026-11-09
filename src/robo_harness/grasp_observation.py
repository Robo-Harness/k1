"""Observed surface relative to nominal pads; no grasp score or unseen shape inference."""

import numpy as np
from scipy.spatial.transform import Rotation


def surface_pad_relation(points_world, state, position=None, quaternion=None):
    points = np.asarray(points_world, float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Need visible Nx3 world points")
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 3:
        return {"available": False, "reason": "Insufficient valid visible points"}
    g = state["geometry"]
    if not g.get("calibration_verified"):
        return {"available": False, "reason": "Unverified pad geometry"}
    p = np.asarray(state["xyz_world_m"] if position is None else position, float)
    r = (
        np.asarray(state["axes_world"])
        if quaternion is None
        else Rotation.from_quat(quaternion).as_matrix()
    )
    pads = np.asarray(g["finger_pad_centers_local_m"], float)
    middle = pads.mean(axis=0)
    local = (points - p) @ r - middle
    ranges = {
        axis: np.quantile(local @ np.asarray(g["axes_local"][axis]), [0.05, 0.95]).tolist()
        for axis in ["approach", "closing", "lateral"]
    }
    c = np.asarray(g["axes_local"]["closing"])
    bounds = sorted(((pads - middle) @ c).tolist())
    return {
        "available": True,
        "visible_points": len(points),
        "visible_surface_ranges_from_pad_midpoint_m_p05_p95": ranges,
        "open_pad_inner_planes_along_closing_m": bounds,
        "visible_width_along_closing_m_p05_p95": ranges["closing"][1] - ranges["closing"][0],
        "meaning": "Signed measured surface ranges, NOT scores. Positive approach is farther along hand approach; zero is pad-midpoint plane. Between closing planes alone does NOT imply pad contact, enclosure or attachment. Unseen object extent and true center unknown. Native hand orientation is NOT object orientation.",
    }
