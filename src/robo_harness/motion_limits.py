"""Translation limits: actor-frame axis bounds and a rotation-invariant transport envelope."""

import numpy as np

from .perception import finite_vector


def axis_limit(value):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("delta_axis must be a finite positive number in meters")
    limit = float(value)
    if not np.isfinite(limit) or limit <= 0:
        raise ValueError("delta_axis must be a finite positive number in meters")
    return limit


def step_axis_limit(args, maximum):
    limit = axis_limit(args.get("delta_axis", maximum))
    if limit > maximum:
        raise ValueError(f"delta_axis for this step must be <= configured {maximum:g} m")
    return limit


def bounded_translation(delta, limit, distance_cap=None):
    """Uniform scaling preserves the straight approach direction, unlike per-axis clipping.

    max_distance_m remains an explicit Euclidean cap for old programmatic callers,
    but is no longer exposed in the model schema. New callers use delta_axis.
    """
    limit = axis_limit(limit)
    delta = finite_vector(delta, 3)
    scale = min(1.0, limit / max(float(np.max(np.abs(delta))), 1e-12))
    if distance_cap is not None:
        distance = axis_limit(distance_cap)
        if distance > limit:
            raise ValueError(f"Legacy max_distance_m must be in (0, {limit:g}]")
        scale = min(scale, distance / max(float(np.linalg.norm(delta)), 1e-12))
    return delta * scale


def validate_transport_delta(delta, limit):
    """World-space adapter envelope enclosing every permitted rotated local cube.

    The controller already checks components in the requested world/gripper frame.
    Rechecking world components here would reject valid gripper-frame diagonals.
    This transport bound is NOT collision/workspace certification.
    """
    delta = finite_vector(delta, 3)
    if np.linalg.norm(delta) > np.sqrt(3) * axis_limit(limit) + 1e-9:
        raise ValueError("Translation exceeds the configured axis-cube transport envelope")
    return delta
