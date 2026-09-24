"""Task-independent translation hypotheses from current visible surface references."""

import numpy as np

from .perception import finite_vector


def qualify_carry_alignment(result, evidence, arm, frame_id):
    """Separate a measured geometric difference from an applicable carry target.

    Evidence comes only from current RGB-D correspondences and robot poses.
    Unknown tracking is not a rejection, and co-motion is not attachment proof.
    This function never changes the input receipt or chooses a recovery action.
    """
    qualified = dict(result)
    evidence = evidence or {}
    current = evidence.get("to_frame_id") == frame_id
    hand = evidence.get("relative_to_hands", {}).get(arm, {}) if current else {}
    conflict = hand.get("inconsistent_with_rigid_carry_hypothesis") is True
    qualified["carry_applicability"] = {
        "status": "contradicted" if conflict else "unverified",
        "reason": "current_source_motion_conflicts_with_rigid_carry"
        if conflict
        else "co_motion_does_not_verify_attachment"
        if hand
        else "no_current_matched_motion_evidence_for_requested_arm",
        "frame_id": frame_id,
        "arm": arm,
        "attachment_verified": False,
    }
    if conflict:
        qualified.pop("target_tcp_world_m", None)
        qualified["carry_applicability"]["next_step"] = (
            "Reinspect the source identity and grasp relationship in current views. "
            "The reference difference remains geometric information, not a carry command. "
            "Do not assume that moving the hand will move this source."
        )
    return qualified


def translation_hypothesis(source, target, tcp, target_offset, axes=("x", "y", "z")):
    source, target, tcp, offset = (
        finite_vector(value, 3) for value in (source, target, tcp, target_offset)
    )
    axes = list(axes)
    if not axes or len(axes) != len(set(axes)) or any(a not in ("x", "y", "z") for a in axes):
        raise ValueError("Choose distinct nonempty world axes from x, y, z")
    desired = target + offset
    delta = desired - source
    selected = np.array([a in axes for a in ("x", "y", "z")])
    translation = np.where(selected, delta, 0.0)
    return {
        "source_reference_world_m": source.tolist(),
        "target_reference_world_m": target.tolist(),
        "desired_source_reference_world_m": desired.tolist(),
        "source_minus_tcp_world_m": (source - tcp).tolist(),
        "full_reference_error_world_m": delta.tolist(),
        "alignment_axes_world": axes,
        "translation_world_m": translation.tolist(),
        "target_tcp_world_m": (tcp + translation).tolist(),
        "selected_reference_error_m": float(np.linalg.norm(translation)),
        "assumption": "Source feature follows the hand rigidly with unchanged orientation. This is a geometric hypothesis, NOT verified attachment, object-center alignment, collision clearance, or task completion. Unselected axes retain their present offsets. Reobserve after execution.",
    }


def measured_region_alignment(manager, obs, args):
    if args.get("frame_id") != obs["frame_id"]:
        raise ValueError("Stale request; use current frame_id")
    source_id, target_id = args["source_region_id"], args["target_region_id"]
    if source_id == target_id:
        raise ValueError("Choose distinct source and target references")
    source, target = (manager.summary(key, obs) for key in (source_id, target_id))
    if any(
        r["status"] != "current_visible_surface_estimate" or r.get("frame_id") != obs["frame_id"]
        for r in (source, target)
    ):
        raise ValueError(
            "Both region references must be currently measured; reinspect lost or stale regions"
        )
    state = obs["arms"][args["arm"]]
    result = translation_hypothesis(
        source["surface_reference_world_m"],
        target["surface_reference_world_m"],
        state["xyz_world_m"],
        args["target_offset_world_m"],
        args.get("alignment_axes_world", ("x", "y", "z")),
    )
    result.update(
        frame_id=obs["frame_id"],
        source_region_id=source_id,
        target_region_id=target_id,
        source_identity_verified=False,
        target_identity_verified=False,
        reference_kind="visible_surface_statistics_not_object_centers",
    )
    return result
