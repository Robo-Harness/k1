import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.motion import MotionController
from robo_harness.pose_targets import PoseTargets
from test_motion import Robot


def test_pose_target_resumes_without_resetting_orientation():
    robot = Robot()
    goals = PoseTargets(MotionController(robot))
    target = Rotation.from_euler("z", 40, degrees=True)
    result = goals.step(
        robot.observe(),
        {
            "arm": "hand",
            "target_xyz_world_m": [0.12, 0, 0.5],
            "target_quaternion_xyzw": target.as_quat().tolist(),
        },
    )
    key = result["pose_goal_id"]
    # External drift between chunks must be corrected toward the ORIGINAL orientation.
    robot.r = Rotation.from_euler("x", 7, degrees=True) * robot.r
    for _ in range(5):
        result = goals.step(robot.observe(), {"arm": "hand", "pose_goal_id": key})
    assert result["pose_reached"]
    np.testing.assert_allclose(robot.xyz, [0.12, 0, 0.5], atol=1e-10)
    assert (target * robot.r.inv()).magnitude() < 1e-10
    assert all(
        np.linalg.norm(d) <= 0.030000001 and np.linalg.norm(r) <= 15.000001
        for d, r, _ in robot.calls
    )


def test_default_orientation_is_snapshotted_and_not_replaced():
    robot = Robot()
    goals = PoseTargets(MotionController(robot))
    result = goals.step(robot.observe(), {"arm": "hand", "target_xyz_world_m": [0.09, 0, 0.5]})
    robot.r = Rotation.from_euler("y", 10, degrees=True)
    result = goals.step(robot.observe(), {"arm": "hand", "pose_goal_id": result["pose_goal_id"]})
    assert result["remaining_degrees"] < 1e-9


def test_invalid_pose_arguments_do_not_move_robot():
    robot = Robot()
    goals = PoseTargets(MotionController(robot))
    for extra in (
        {"max_distance_m": 0.04},
        {"max_degrees": 20},
        {"target_quaternion_xyzw": [0, 0, 0, 0]},
    ):
        with pytest.raises(ValueError):
            goals.step(robot.observe(), {"arm": "hand", "target_xyz_world_m": [0, 0, 0.5], **extra})
    assert not robot.calls


def test_no_progress_protection_is_preserved():
    robot = Robot()
    robot.scale = 0
    goals = PoseTargets(MotionController(robot))
    result = goals.step(robot.observe(), {"arm": "hand", "target_xyz_world_m": [0.2, 0, 0.5]})
    args = {"arm": "hand", "pose_goal_id": result["pose_goal_id"]}
    for _ in range(2):
        goals.step(robot.observe(), args)
    with pytest.raises(ValueError, match="Repeated low-progress"):
        goals.step(robot.observe(), args)
    assert len(robot.calls) == 3


def test_registry_pose_tool_guards_and_switches():
    from robo_harness.runtime import ToolRegistry

    robot = Robot()
    robot.gripper = 1
    registry = ToolRegistry(robot)
    args = {
        "arm": "hand",
        "frame_id": 0,
        "target_xyz_world_m": robot.xyz.tolist(),
        "target_quaternion_xyzw": Rotation.from_euler("z", 60, degrees=True).as_quat().tolist(),
    }
    result = registry.execute("move_to_pose", args, robot.observe())
    assert result["remaining_degrees"] == pytest.approx(45)
    with pytest.raises(ValueError, match="degrees residual"):
        registry.execute(
            "set_gripper", {"arm": "hand", "frame_id": robot.frame, "gripper": 0}, robot.observe()
        )
    with pytest.raises(ValueError, match="preserves gripper"):
        registry.execute(
            "move_to_pose", {**args, "frame_id": robot.frame, "gripper": 0}, robot.observe()
        )
    disabled = ToolRegistry(robot, motion_tools=False)
    assert "move_to_pose" not in {s["function"]["name"] for s in disabled.schemas(robot.observe())}
    with pytest.raises(ValueError, match="disabled"):
        disabled.execute("move_to_pose", {**args, "frame_id": robot.frame}, robot.observe())
