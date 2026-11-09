"""Compact passive-tool geometry from robot-owned, declared contact landmarks.

No scene geometry, native contact predicates, task labels or guessed grasp axes.
Adapters for other simulators can reuse describe_surface with their calibrated
robot landmarks. This descriptor does not change motion or certify contact.
"""

import numpy as np


def describe_surface(points_local_m, tcp_world_m, axes_world, marker_radii_m=None):
    points = np.asarray(points_local_m, dtype=float)
    tcp, axes = np.asarray(tcp_world_m, dtype=float), np.asarray(axes_world, dtype=float)
    if (
        points.ndim != 2
        or points.shape[1] != 3
        or not 3 <= len(points) <= 8
        or tcp.shape != (3,)
        or axes.shape != (3, 3)
        or not all(np.isfinite(v).all() for v in (points, tcp, axes))
    ):
        return dict(available=False, reason="invalid_or_excessive_robot_surface_landmarks")
    if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-5) or np.linalg.det(axes) < 0.999:
        return dict(available=False, reason="invalid_robot_frame_rotation")
    center = points.mean(axis=0)
    _, singular, vt = np.linalg.svd(points - center, full_matrices=False)
    if singular[1] < 1e-6 or singular[2] > 1e-5:
        return dict(available=False, reason="robot_landmarks_do_not_define_a_planar_surface")
    normal = vt[-1]
    # A repeatable SIGN CONVENTION, not an inferred outward/press direction.
    if normal[np.argmax(np.abs(normal))] < 0:
        normal = -normal
    result = dict(
        available=True,
        source="robot_model_declared_contact_landmarks",
        reference_frame="native_eef",
        center_local_m=center.tolist(),
        center_world_m=(tcp + axes @ center).tolist(),
        normal_axis_local_unsigned=normal.tolist(),
        normal_axis_world_unsigned=(axes @ normal).tolist(),
        landmark_offsets_local_m=points.tolist(),
        landmark_extent_local_xyz_m=np.ptp(points, axis=0).tolist(),
        meaning="Robot tool geometry only. Normal SIGN is unspecified. Landmark plane is not "
        "proof of contact, force, complete collision coverage or task success. "
        "TCP arrival alone does not mean the working surface is contacting the scene.",
    )
    if marker_radii_m is not None:
        radii = np.asarray(marker_radii_m, dtype=float)
        if radii.shape != (len(points),) or not np.isfinite(radii).all() or np.any(radii < 0):
            return dict(available=False, reason="invalid_robot_landmark_radii")
        result["spherical_landmark_radii_m"] = radii.tolist()
    return result


def mujoco_passive_surface(sim, gripper, tcp, axes):
    """Use a robot model's public corner declaration, not a task-name special case.

    Sphere centers/radii are declared robot collision landmarks. Unsupported or
    missing declarations yield unknown; normal gripper observations are unchanged.
    """
    if gripper.dof:
        return None
    names = gripper.important_geoms.get("corners", [])
    if not 3 <= len(names) <= 8 or len(set(names)) != len(names):
        return dict(available=False, reason="no_supported_robot_declared_surface")
    try:
        owned = {sim.model.body_name2id(n) for n in gripper.bodies}
        ids = [sim.model.geom_name2id(n) for n in names]
        if any(i < 0 or sim.model.geom_bodyid[i] not in owned for i in ids):
            return dict(available=False, reason="surface_landmark_not_owned_by_robot_tool")
        if any(int(sim.model.geom_type[i]) != 2 for i in ids):
            return dict(available=False, reason="unsupported_contact_landmark_primitive")
        local = (np.asarray(sim.data.geom_xpos)[ids] - np.asarray(tcp)) @ np.asarray(axes)
        return describe_surface(local, tcp, axes, np.asarray(sim.model.geom_size)[ids, 0])
    except (KeyError, ValueError, AttributeError, IndexError):
        return dict(available=False, reason="robot_surface_calibration_unavailable")
