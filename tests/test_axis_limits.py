from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.libero_adapter import LiberoAdapter
from robo_harness.motion import MotionController
from robo_harness.motion_limits import bounded_translation, validate_transport_delta
from robo_harness.pose_targets import PoseTargets
from robo_harness.runtime import ToolRegistry, run_agent
from test_motion import Robot


@pytest.mark.parametrize("delta", [[0, 0.03, 0.01], [0.03, 0.03, 0.03], [-0.03, -0.03, -0.03]])
def test_relative_diagonal_does_not_require_norm_calculation(delta):
    robot = Robot()
    tools = ToolRegistry(robot, advisory_controls=True)
    tools.execute(
        "move_relative",
        {"arm": "hand", "frame_id": 0, "frame": "world", "delta": delta},
        robot.observe(),
    )
    np.testing.assert_allclose(robot.calls[-1][0], delta)
    assert np.linalg.norm(robot.calls[-1][0]) > 0.03


@pytest.mark.parametrize("limit", [0.01, 0.03, 0.06])
def test_config_propagates_to_relative_target_schemas_and_adapter(limit):
    robot = Robot()
    tools = ToolRegistry(robot, delta_axis=limit, advisory_controls=True)
    specs = {s["function"]["name"]: s["function"] for s in tools.schemas(robot.observe())}
    assert specs["move_relative"]["parameters"]["properties"]["delta"]["items"]["maximum"] == limit
    for name in ["move_to_pose", "retreat"]:
        props = specs[name]["parameters"]["properties"]
        assert props["delta_axis"]["maximum"] == limit
        assert "max_distance_m" not in props
    assert "delta_axis" not in specs["move_toward"]["parameters"]["properties"]
    assert robot.delta_axis == limit
    assert tools.motion.context()["translation_limit"]["delta_axis_m"] == limit
    motion = tools.motion
    motion.move(robot.observe(), "hand", [limit, -limit, limit], [0, 0, 0])
    with pytest.raises(ValueError, match="component"):
        motion.move(robot.observe(), "hand", [limit + 0.001, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="rotation"):
        motion.move(robot.observe(), "hand", [0, 0, 0], [15, 15, 0])


def test_gripper_axes_are_checked_before_rotation_not_after():
    robot = Robot()
    robot.r = Rotation.from_euler("z", 45, degrees=True)
    tools = ToolRegistry(robot, advisory_controls=True)
    local = np.array([0.03, -0.03, 0])
    expected_world = robot.r.apply(local)
    assert expected_world[0] > 0.04
    tools.execute(
        "move_relative",
        {"arm": "hand", "frame_id": 0, "frame": "gripper", "delta": local.tolist()},
        robot.observe(),
    )
    np.testing.assert_allclose(robot.calls[-1][0], expected_world, atol=1e-12)
    validate_transport_delta(expected_world, 0.03)
    with pytest.raises(ValueError, match="component"):
        tools.motion.move(robot.observe(), "hand", expected_world, [0, 0, 0])


@pytest.mark.parametrize("tool", ["move_to_pose"])
@pytest.mark.parametrize("limit", [0.01, 0.03, 0.06])
def test_absolute_targets_use_straight_per_axis_steps(tool, limit):
    robot = Robot()
    tools = ToolRegistry(robot, advisory_controls=True, delta_axis=limit)
    args = {"arm": "hand", "frame_id": 0, "target_xyz_world_m": [0.3, -0.6, 0.8]}
    tools.execute(tool, args, robot.observe())
    np.testing.assert_allclose(robot.calls[-1][0], [limit / 2, -limit, limit / 2], atol=1e-12)
    args["frame_id"] = robot.frame
    args["delta_axis"] = limit / 2
    tools.execute(tool, args, robot.observe())
    assert np.max(np.abs(robot.calls[-1][0])) == pytest.approx(limit / 2)


def test_small_absolute_target_not_rescaled_and_distance_cap_is_explicit():
    delta = [0, 0.03, 0.01]
    np.testing.assert_allclose(bounded_translation(delta, 0.03), delta)
    assert np.linalg.norm(bounded_translation(delta, 0.03, distance_cap=0.02)) == pytest.approx(
        0.02
    )
    np.testing.assert_allclose(bounded_translation([0, 0, 0], 0.03), 0)
    with pytest.raises(ValueError):
        bounded_translation([0, 0, 0], 0.03, distance_cap=0.04)


def test_pose_orientation_and_retreat_use_configured_bounds():
    robot = Robot()
    motion = MotionController(robot, advisory=True, delta_axis=0.01)
    goals = PoseTargets(motion)
    result = goals.step(robot.observe(), {"arm": "hand", "target_xyz_world_m": [0.04, 0.04, 0.54]})
    for _ in range(4):
        result = goals.step(
            robot.observe(), {"arm": "hand", "pose_goal_id": result["pose_goal_id"]}
        )
    assert result["pose_reached"]
    oriented = motion.orient(
        "rotate_toward",
        {
            "arm": "hand",
            "target_quaternion_xyzw": Rotation.from_euler("z", 40, degrees=True).as_quat().tolist(),
        },
        robot.observe(),
    )
    robot.xyz += [0.03, 0.03, 0.03]
    motion.orient("rotate_toward", {"arm": "hand", "goal_id": oriented["goal_id"]}, robot.observe())
    np.testing.assert_allclose(robot.calls[-1][0], [-0.01, -0.01, -0.01])
    motion.retreat({"arm": "hand"}, robot.observe())
    assert np.max(np.abs(robot.calls[-1][0])) <= 0.010000001
    assert all(np.max(np.abs(d)) <= 0.010000001 for d, _, _ in robot.calls)


@pytest.mark.parametrize("limit", [0, -1, float("nan"), float("inf"), True])
def test_invalid_axis_limit_rejected_before_output_or_api(tmp_path, limit):
    with pytest.raises(ValueError):
        run_agent(None, tmp_path / "unused", "unused", "unused", delta_axis=limit)
    assert not (tmp_path / "unused").exists()


def test_libero_transport_accepts_diagonal_without_real_simulation():
    # Exercise the actual adapter move method, using an ideal mock OSC backend.
    adapter = LiberoAdapter.__new__(LiberoAdapter)
    adapter.delta_axis = 0.03
    adapter.frame, adapter.horizon, adapter.gripper, adapter.on_step = 0, 12, 0.2, None
    adapter.raw = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": [0, 0, 0, 1]}

    def step(command):
        adapter.raw["robot0_eef_pos"] += command[:3] * 0.1
        return adapter.raw, 0, False, {}

    controller = SimpleNamespace(output_max=np.ones(6) * 0.1, output_min=-np.ones(6) * 0.1)
    adapter.env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller)], step=step, check_success=lambda: False
    )
    result = adapter.move("arm", [0.03, 0.03, 0.03])
    np.testing.assert_allclose(result["actual_translation_m"], [0.03, 0.03, 0.03])
    # A valid local-frame diagonal can have a world component greater than 3cm.
    adapter.frame = 0
    result = adapter.move("arm", [np.sqrt(2) * 0.03, 0, 0])
    np.testing.assert_allclose(result["actual_translation_m"], [np.sqrt(2) * 0.03, 0, 0])
    with pytest.raises(ValueError, match="transport"):
        adapter.move("arm", [0.06, 0, 0])
