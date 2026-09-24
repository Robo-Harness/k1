"""Meters, arbitrary shared world frame, OpenCV optical camera-to-world poses."""

import numpy as np


def calibration(camera):
    k = np.asarray(camera["intrinsic_matrix"], float)
    t = np.asarray(camera["extrinsic_matrix"], float)
    if (
        k.shape != (3, 3)
        or t.shape != (4, 4)
        or not np.isfinite(k).all()
        or not np.isfinite(t).all()
    ):
        raise ValueError("Invalid calibrated camera")
    if min(k[0, 0], k[1, 1]) <= 1:
        raise ValueError("Invalid focal length")
    return k, t


def unproject(camera, uv):
    if "depth" not in camera:
        raise ValueError("Measured metric depth unavailable")
    depth = np.asarray(camera["depth"]).squeeze()
    h, w = depth.shape
    uv = np.asarray(uv, float)
    if uv.shape != (2,) or not np.isfinite(uv).all() or np.any(uv < 0) or np.any(uv >= [w, h]):
        raise ValueError("Pixel outside original image")
    u, v = np.minimum(np.floor(uv + 0.5).astype(int), [w - 1, h - 1])
    z = float(depth[v, u])
    if not np.isfinite(z) or not 0 < z < 10:
        raise ValueError("Invalid metric depth")
    k, t = calibration(camera)
    p = t[:3, :3] @ (np.linalg.solve(k, [u, v, 1]) * z) + t[:3, 3]
    return {
        "kind": "measured_surface",
        "pixel": [int(u), int(v)],
        "depth_m": z,
        "xyz_world_m": p.tolist(),
        "meaning": "Visible surface, not object center or automatically a grasp target",
    }
