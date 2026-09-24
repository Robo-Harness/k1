"""Actor-only feedback reduction; measurements and diagnostic receipts stay intact.

Deliberately does not infer attachment, contact intent, or task-specific safety.
"""


def actor_feedback(value):
    """Return a fresh compact tree without changing controller state or raw logs."""
    if isinstance(value, list):
        return [actor_feedback(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    for key, item in value.items():
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
    return result
