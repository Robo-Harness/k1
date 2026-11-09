"""Actor-only feedback reduction; measurements and diagnostic receipts stay intact.

Deliberately does not infer attachment, contact intent, or task-specific safety.
"""

REGION_GEOMETRY_VIEW = None
PRESENTATION_VERSION = "k1-v1.1-typed-gripper"


def actor_feedback(value):
    """Return a fresh compact tree without changing controller state or raw logs."""
    if isinstance(value, list):
        return [actor_feedback(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
        if "gripper_measurement" in value and key in {
            "gripper_opening_measured",
            "gripper_opening_normalized",
        }:
            # Legacy normalized fields can echo a command; the legacy boolean
            # sometimes means "metric gap available", not "any sensor exists".
            # Typed, sourced measurements are the sole model-visible authority.
            continue
        if isinstance(value.get("measurement"), dict) and "command_open" in value:
            if key == "settled_last3_native_frames":
                continue
            if (
                key in {"opening_m", "near_closed_mechanical_limit"}
                and value["measurement"].get("unit") != "m"
            ):
                continue
        if key in {
            "sweep",
            "decision_advisories",
            "recovery_hint",
            "pre_action_target_advisory",
            "native_last_step_diagnostic",
            "grasp_diagnostics",
        }:
            continue
        result[key] = actor_feedback(item)
    return REGION_GEOMETRY_VIEW(result) if REGION_GEOMETRY_VIEW is not None else result
