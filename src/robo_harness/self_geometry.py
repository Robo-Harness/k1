"""Robot-only self masking for sparse RGB-D queries, never an action guard.

Adapters supply convex parts in world coordinates. No object segmentation, scene
poses or native contacts are accepted. Absent geometry uses conservative fallback behavior.
Private geometry lives in camera observations, not in model-facing text.
"""

import numpy as np


def visible_self_mask(camera, arm, points):
    parts = camera.get("_robot_self_parts", {}).get(arm)
    if parts is None:
        return None
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    result = np.zeros(len(points), dtype=bool)
    for part in parts:
        lo, hi = part["world_bounds"]
        indices = np.flatnonzero(
            ~result & np.isfinite(points).all(1) & (points >= lo).all(1) & (points <= hi).all(1)
        )
        if not len(indices):
            continue
        local = (points[indices] - part["center"]) @ part["rotation"]
        eq = part["planes"]
        result[indices] |= (local @ eq[:, :3].T + eq[:, 3] <= part["tolerance"]).all(1)
    return result


class RobotSelfGeometry:
    """Build once from explicitly selected robot geom IDs; pose from kinematics.

    Currently supports mesh convex hulls and boxes. If any selected part is unsupported,
    the adapter must fall back rather than silently claiming complete self coverage.
    """

    def __init__(self, model, geom_ids, tolerance=0.002):
        from scipy.spatial import ConvexHull

        self.parts = []
        self.tolerance = tolerance
        for gid in geom_ids:
            kind = int(model.geom_type[gid])
            if kind == 7:
                mid = model.geom_dataid[gid]
                start, n = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
                vertices = np.array(model.mesh_vert[start : start + n], dtype=float)
                equations = ConvexHull(vertices).equations
            elif kind == 6:
                size = np.array(model.geom_size[gid])
                vertices = np.array(
                    [
                        [x, y, z]
                        for x in [-size[0], size[0]]
                        for y in [-size[1], size[1]]
                        for z in [-size[2], size[2]]
                    ]
                )
                equations = np.c_[np.r_[np.eye(3), -np.eye(3)], -np.r_[size, size]]
            else:
                raise ValueError(f"Unsupported robot self geometry type: {kind}")
            # Bounds of the OFFSET polytope, not bounds+tolerance: at acute corners
            # the latter can wrongly reject points accepted by the plane test.
            from scipy.optimize import linprog

            bounds = []
            for axis in np.eye(3):
                pair = [
                    linprog(
                        sign * axis,
                        A_ub=equations[:, :3],
                        b_ub=tolerance - equations[:, 3],
                        bounds=[(None, None)] * 3,
                        method="highs",
                    )
                    for sign in [1, -1]
                ]
                if not all(r.success for r in pair):
                    raise ValueError("Cannot bound robot self geometry")
                bounds.append([pair[0].fun, -pair[1].fun])
            bounds = np.asarray(bounds)
            corners = np.array([[x, y, z] for x in bounds[0] for y in bounds[1] for z in bounds[2]])
            self.parts.append((gid, equations, corners))

    def snapshot(self, data):
        result = []
        for gid, equations, corners in self.parts:
            center = np.array(data.geom_xpos[gid])
            rotation = np.array(data.geom_xmat[gid]).reshape(3, 3)
            world = corners @ rotation.T + center
            result.append(
                {
                    "center": center,
                    "rotation": rotation,
                    "planes": equations,
                    "tolerance": self.tolerance,
                    "world_bounds": (world.min(0) - 1e-9, world.max(0) + 1e-9),
                }
            )
        return result
