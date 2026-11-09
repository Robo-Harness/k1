"""Small optional explanations, never task scripts, motion gates or success rules."""

from copy import deepcopy

GUIDANCE = {
    "grasp_evidence": [
        "A close command, reached TCP pose or nonzero jaw gap does not establish a grasp.",
        "Compare original images before and after hand motion. Look for the intended part moving with the hand, not a different object or background.",
        "If a gentle observation motion is appropriate, you decide its direction and distance from visible clearance. There is no automatic lift or mandatory test.",
        "Occlusion and uncertain correspondence mean unknown. Revise progress notes when new images contradict carrying.",
    ],
    "handover_evidence": [
        "Distinguish donor and recipient hands, and track the intended object visually.",
        "Recipient closure or reaching a shared pose alone does not prove it can support the object independently.",
        "Consider keeping donor support until recipient contact is visually plausible; after release, check independent retention before assuming transfer.",
        "An observation hand can improve a native wrist view when appropriate. You choose motion and release; this guidance does not execute a sequence or enforce a gate.",
    ],
    "motion_recovery": [
        "Compare commanded and measured displacement, orientation residual, and the current original views.",
        "Explicit planner failure differs from slow tracking, contact, and unknown execution. Do not infer an obstacle or a grasp solely from one of these.",
        "If retrying, choose whether to change one meaningful variable: target, approach orientation, clearance or arm. No automatic substitute target is selected.",
        "A larger motion has less visual feedback during execution. Use short corrections when contact or geometry is uncertain. hold advances native time; a zero increment does not.",
    ],
}


def read_guidance(topic):
    if topic not in GUIDANCE:
        raise ValueError("Unknown guidance topic; choose one from the tool schema")
    return dict(
        topic=topic,
        principles=deepcopy(GUIDANCE[topic]),
        scope="General optional reasoning aid, not scene evidence, a task plan or proof of success",
        native_steps=0,
    )
