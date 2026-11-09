"""Explicit measured-point references; never infer axis swaps or object identity."""

import numpy as np
from .perception import finite_vector
from .observation_contract import camera_reference


def named_schema(names):
    return dict(
        type="object",
        properties={n: dict(type="number") for n in names},
        required=list(names),
        additionalProperties=False,
    )


def named_vector(value, names):
    if not isinstance(value, dict) or set(value) != set(names):
        raise ValueError("Expected explicit named components: " + ", ".join(names))
    if any(isinstance(value[k], bool) for k in names):
        raise ValueError("Coordinates must be finite numbers, not booleans")
    return finite_vector([value[k] for k in names], 3)


def point_schema():
    return dict(
        type="object",
        properties={
            "point_id": dict(
                type="string",
                description="Current visible P ID from measure_depth / tracked_points",
            ),
            "offset_world_m": named_schema(("dx", "dy", "dz")),
        },
        required=["point_id", "offset_world_m"],
        additionalProperties=False,
    )


def resolve(args, obs, tracked_points, measured_points=()):
    """Resolve once at tool entry; a transit does not chase a moving feature."""
    selected = [k for k in ("target_point", "target_xyz_world_m") if k in args]
    if len(selected) != 1:
        raise ValueError(
            "Supply exactly one of target_point or target_xyz_world_m; no motion executed"
        )
    if "target_point" not in args:
        value = args["target_xyz_world_m"]
        # Legacy array replay remains exact; no guessing/permutation.
        xyz = (
            named_vector(value, ("x", "y", "z"))
            if isinstance(value, dict)
            else finite_vector(value, 3)
        )
        return xyz, dict(
            source="explicit_world_position", xyz_world_m=xyz.tolist(), axis_order="x,y,z"
        )
    ref = args["target_point"]
    if not isinstance(ref, dict) or set(ref) != {"point_id", "offset_world_m"}:
        raise ValueError("target_point requires point_id and explicit offset_world_m {dx,dy,dz}")
    offset = named_vector(ref["offset_world_m"], ("dx", "dy", "dz"))
    key = ref["point_id"]
    if not isinstance(key, str) or key not in tracked_points.entries:
        raise ValueError("Unknown point ID; measure the intended visible surface first")
    entry = tracked_points.entries[key]
    current = entry["latest"]
    # A textureless pixel can have valid current depth but cannot initialize
    # optical flow. That prevents FUTURE tracking, not use of this exact fresh
    # measurement. Explicit invalidation for contact/identity is still respected.
    if entry.get("lost") and entry.get("loss_reason") == "initial_correspondence_uncertain":
        fresh = next(
            (
                p
                for p in reversed(measured_points)
                if p["id"] == key
                and p["frame_id"] == obs["frame_id"]
                and p.get("sensor_reference") == camera_reference(obs, p["camera"])
            ),
            None,
        )
        if fresh is not None:
            current = dict(fresh, status="initialized")
    valid_fresh = current is not entry["latest"]
    if (
        (entry.get("lost") and not valid_fresh)
        or current.get("status") not in ("initialized", "tracked_estimate")
        or current.get("frame_id") != obs["frame_id"]
        or entry.get("sensor_reference") != camera_reference(obs, entry["camera"])
    ):
        raise ValueError(
            f"{key} has no current valid measured correspondence. Historical coordinates are retained but NOT used for motion; remeasure or specify an intentional explicit world target."
        )
    surface = finite_vector(current["xyz_world_m"], 3)
    xyz = surface + offset
    if not np.isfinite(xyz).all():
        raise ValueError("Resolved position is not finite")
    return xyz, dict(
        source="current_surface_point",
        point_id=key,
        frame_id=obs["frame_id"],
        camera=entry["camera"],
        surface_xyz_world_m=surface.tolist(),
        offset_world_m=ref["offset_world_m"],
        xyz_world_m=xyz.tolist(),
        identity_verified=False,
        meaning="Fixed TCP waypoint resolved once from current surface correspondence; not an object center, grasp pose, safe path or verified identity.",
    )
