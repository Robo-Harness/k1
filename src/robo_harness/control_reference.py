"""Robot-calibrated control points; no scene geometry or task policy."""

import numpy as np
from scipy.spatial.transform import Rotation


def finite(value, size):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.isfinite(value).all():
        raise ValueError("Expected finite vector")
    return value


def pad_center_local(state):
    geometry = state["geometry"]
    if not geometry.get("calibration_verified", False):
        raise ValueError("pad_center requires verified robot geometry")
    pads = np.asarray(geometry["finger_pad_centers_local_m"], dtype=float)
    if pads.shape != (2, 3) or not np.isfinite(pads).all():
        raise ValueError("Expected two calibrated finger pad centers")
    return pads.mean(axis=0)


def resolve_targets(obs, targets):
    """Validate all targets before execution; snapshot pad offset for this command."""
    if not isinstance(targets, list) or not 1 <= len(targets) <= len(obs["arms"]):
        raise ValueError("Expected nonempty targets, at most one per arm")
    output, seen = [], set()
    for raw in targets:
        arm = raw["arm"]
        if arm not in obs["arms"] or arm in seen:
            raise ValueError("Unknown or duplicate arm")
        seen.add(arm)
        position = finite(raw["position"], 3)
        q = finite(raw["quaternion_xyzw"], 4)
        if np.linalg.norm(q) < 1e-8:
            raise ValueError("Zero quaternion")
        q = q / np.linalg.norm(q)
        reference = raw.get("position_reference", "native_eef")
        if reference not in ("native_eef", "pad_center"):
            raise ValueError("position_reference must be native_eef or pad_center")
        offset = pad_center_local(obs["arms"][arm]) if reference == "pad_center" else np.zeros(3)
        grip = raw.get("gripper")
        if grip is not None and (not np.isfinite(grip) or not 0 <= grip <= 1):
            raise ValueError("Gripper must be finite 0..1")
        output.append(
            dict(
                arm=arm,
                position=(position - Rotation.from_quat(q).apply(offset)).tolist(),
                quaternion_xyzw=q.tolist(),
                gripper=grip,
                control_reference=dict(
                    kind=reference,
                    requested_position_world_m=position.tolist(),
                    offset_local_m_at_command=offset.tolist(),
                    calibration_frame=obs["frame_id"],
                    meaning="Position converted using measured robot pad geometry at command start; "
                    "orientation is always native end-effector orientation, NOT an object orientation.",
                ),
            )
        )
    return output
