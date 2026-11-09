"""On-demand geometry overlay on the same original camera, no scores."""

import cv2
import numpy as np
from .geometry import calibration
from .roi import rectangle


def render(obs, camera_name, result):
    camera = obs["vision"][camera_name]
    rgb = np.asarray(camera["color"]).copy()
    roi = rectangle(result["roi"], rgb.shape)
    cv2.rectangle(rgb, tuple(roi[:2]), tuple(roi[2:]), (255, 200, 0), 2)
    center = np.asarray(result["xyz_world_m"])
    direction = np.asarray(result["direction_world"])
    k, t = calibration(camera)

    def pixel(point):
        p = t[:3, :3].T @ (point - t[:3, 3])
        if p[2] <= 1e-6:
            return None
        uv = (k @ p)[:2] / p[2]
        if not np.isfinite(uv).all() or np.max(np.abs(uv)) > 100000:
            return None
        return tuple(np.rint(uv).astype(int))

    origin = pixel(center)
    if origin:
        for sign in [-1, 1]:
            endpoint = pixel(center + sign * 0.05 * direction)
            if endpoint:
                cv2.arrowedLine(rgb, origin, endpoint, (0, 255, 255), 2, tipLength=0.2)
    evidence = {}
    for i, (arm, state) in enumerate(obs["arms"].items()):
        axis = state.get("geometry", {}).get("axes_local", {}).get("approach")
        if axis is None:
            continue
        world = np.asarray(state["axes_world"]) @ np.asarray(axis)
        world /= np.linalg.norm(world)
        angle = float(np.degrees(np.arccos(np.clip(abs(world @ direction), 0, 1))))
        evidence[arm] = dict(
            unsigned_angle_to_fitted_axis_deg=angle,
            meaning="0 means parallel to measured axis; not a recommended contact direction or collision-free pose",
        )
        a = pixel(np.asarray(state["xyz_world_m"]))
        b = pixel(np.asarray(state["xyz_world_m"]) + 0.05 * world)
        if a and b:
            cv2.arrowedLine(rgb, a, b, (255, 0, 255), 2, tipLength=0.2)
        cv2.putText(
            rgb,
            f"{arm}: axis angle {angle:.1f} deg",
            (8, 40 + 18 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )
    cv2.putText(
        rgb,
        "ROI yellow | fitted axis cyan (+/-) | hand magenta",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (255, 255, 255),
        1,
    )
    return rgb, evidence
