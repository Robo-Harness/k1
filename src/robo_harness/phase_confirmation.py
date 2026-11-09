"""Generic observation-based phase confirmation; no task policy or action gate."""

PHASE_CONFIRMATION = (
    "\nBefore transport, compare the previous and current ORIGINAL camera images: did the intended "
    "object actually move with the hand? State a brief observation or uncertainty in decision_note, "
    "not the intended outcome. Empty closed jaws or an object still on the support contradict carrying. "
    "For precise placement, distinguish hand approach axis from the held object axis and its working "
    "tip; rotating the hand downward does not by itself make the object upright. Use visual servoing "
    "and inspect_grasp_geometry when the relationship is unclear. Surface depth is not an object center "
    "or automatically a hole-axis/tip target. If the current pose target is not reached, inspect residuals "
    "and images before continuing or releasing; release remains your choice, including deliberate recovery. "
    "Candidate poses and pad-range measurements are hypotheses/evidence, not verified grasps. "
    "Use update_task_progress for observed milestones and revise them after contradictions; "
    "do not advance a milestone solely because an action was commanded.\n"
)
