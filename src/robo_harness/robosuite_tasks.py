"""RoboSuite task wrapper bindings; external task sources are not redistributed."""

import hashlib
from pathlib import Path

TASKS = {
    "cube_lifting": (
        "robosuite_cube_lift",
        "FrankaRobosuiteCubeLiftLowLevel",
        "Pick up the red cube and lift it.",
        1500,
    ),
    "cube_restack": (
        "robosuite_cubes_restack",
        "FrankaRobosuiteCubesRestackLowLevel",
        "Place the red cube on top of the green cube and then open the gripper.",
        1500,
    ),
    "cube_stack": (
        "robosuite_cubes",
        "FrankaRobosuiteCubesLowLevel",
        "Place the red cube on top of the green cube and then open the gripper.",
        1500,
    ),
    "nut_assembly": (
        "robosuite_nut_assembly",
        "FrankaRobosuiteNutAssemblyVisual",
        "Grasp the brown square nut by its handle and insert its hollow center onto the square peg.",
        1500,
    ),
    "spill_wipe": (
        "robosuite_spill_wipe",
        "FrankaRobosuiteSpillWipeLowLevel",
        "Wipe up the brown spill. A sponge is already attached to the end-effector.",
        4000,
    ),
    "two_arm_handover": (
        "robosuite_handover",
        "RobosuiteHandoverEnv",
        "Arm 0 should pick up the hammer, lift it, and hand it to Arm 1. Arm 1 must grasp the handle, not the hammer head, and Arm 0 must release it.",
        5000,
    ),
    "two_arm_lift": (
        "robosuite_two_arm_lift",
        "RobosuiteTwoArmLiftEnv",
        "Coordinate both arms to grasp the two handles and lift the pot while keeping it level.",
        5000,
    ),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
