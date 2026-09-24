from types import SimpleNamespace
import json

import httpx

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.position_reach import reach_config
from robo_harness.runtime import ToolRegistry, run_agent
from test_motion import Robot


def go(robot, tools, target):
    return tools.execute(
        "move_toward",
        {"frame_id": robot.frame, "arm": "hand", "target_xyz_world_m": target},
        robot.observe(),
    )


@pytest.mark.parametrize("delta_axis", [0.001, 0.03, 0.1])
def test_full_target_is_independent_of_axis_limit_and_orientation_is_fixed(delta_axis):
    robot = Robot()
    robot.r = Rotation.from_euler("xyz", [10, 20, 30], degrees=True)
    original = robot.r.as_quat().copy()
    tools = ToolRegistry(robot, advisory_controls=True, delta_axis=delta_axis)
    result = go(robot, tools, [0.4, -0.2, 0.6])
    assert result["target_reached"] and result["native_steps"] == 3
    np.testing.assert_allclose(robot.calls[0][0], [0.4, -0.2, 0.1])
    np.testing.assert_allclose(robot.r.as_quat(), original)
    assert robot.gripper == 0.25 and len(tools.motion.paths["hand"]) >= 2
    # The adjacent small-step API still enforces delta_axis.
    with pytest.raises(ValueError, match="component"):
        tools.execute(
            "move_relative",
            {
                "frame_id": robot.frame,
                "arm": "hand",
                "frame": "world",
                "delta": [delta_axis * 2, 0, 0],
            },
            robot.observe(),
        )


def test_stall_is_reported_and_does_not_block_a_later_retry():
    robot = Robot()
    robot.scale = 0
    tools = ToolRegistry(robot, advisory_controls=True, move_toward_stall_steps=5)
    first = go(robot, tools, [0.2, 0, 0.5])
    assert first["stop_reason"] == "no_measurable_progress" and first["native_steps"] == 5
    assert not first["target_reached"] and first["remaining_distance_m"] == pytest.approx(0.2)
    robot.scale = 1
    assert go(robot, tools, [0.2, 0, 0.5])["target_reached"]


@pytest.mark.parametrize(
    "episode_horizon,max_steps,reason,count",
    [(600, 7, "native_step_budget", 7), (4, 12, "episode_step_budget", 4)],
)
def test_budgets_never_overrun(episode_horizon, max_steps, reason, count):
    robot = Robot()
    robot.scale = 0
    robot.horizon = episode_horizon
    tools = ToolRegistry(robot, move_toward_max_steps=max_steps, move_toward_stall_steps=0)
    result = go(robot, tools, [0.2, 0, 0.5])
    assert result["stop_reason"] == reason and robot.frame == count
    assert not result["target_reached"]


def test_no_advance_and_native_success_are_not_fabricated_arrival():
    robot = Robot()
    robot.move_to_position = lambda *a: {}
    tools = ToolRegistry(robot)
    assert go(robot, tools, [0.2, 0, 0.5])["stop_reason"] == "adapter_did_not_advance"
    robot = Robot()
    robot.scale = 0
    robot.success = lambda: robot.frame >= 2
    tools = ToolRegistry(robot)
    result = go(robot, tools, [0.2, 0, 0.5])
    assert result["stop_reason"] == "native_success" and not result["target_reached"]
    assert result["native_steps"] == 2


def test_gripper_command_applies_from_start_and_original_orientation_is_held():
    robot = Robot()
    robot.scale = 0.5
    calls = []
    original = robot.move_to_position

    def servo(arm, target, quat, gripper, steps):
        calls.append((np.array(target), np.array(quat), gripper))
        return original(arm, target, quat, gripper, steps)

    robot.move_to_position = servo
    tools = ToolRegistry(robot)
    result = tools.execute(
        "move_toward",
        {"frame_id": 0, "arm": "hand", "gripper": 0, "target_xyz_world_m": [0.15, 0.1, 0.5]},
        robot.observe(),
    )
    assert result["target_reached"] and len(calls) > 3
    for target, quat, gripper in calls:
        np.testing.assert_allclose(target, [0.15, 0.1, 0.5])
        np.testing.assert_allclose(quat, [0, 0, 0, 1])
        assert gripper == 0


def test_unsupported_adapter_hides_long_move_and_does_not_emulate_chunks():
    robot = Robot()
    adapter = SimpleNamespace(observe=robot.observe, move=robot.move)
    tools = ToolRegistry(adapter)
    names = {s["function"]["name"] for s in tools.schemas(robot.observe())}
    assert "move_toward" not in names and "move_relative" in names
    with pytest.raises(ValueError, match="does not support"):
        go(robot, tools, [0.2, 0, 0.5])
    assert not robot.calls


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_steps": 0},
        {"max_steps": True},
        {"stall_steps": -1},
        {"tolerance_m": 0},
        {"tolerance_m": float("nan")},
    ],
)
def test_bad_reach_config(kwargs):
    with pytest.raises(ValueError):
        reach_config(**kwargs)


def test_bad_reach_config_fails_before_output(tmp_path):
    with pytest.raises(ValueError):
        run_agent(None, tmp_path / "unused", "unused", "unused", move_toward_max_steps=0)
    assert not (tmp_path / "unused").exists()


def test_libero_full_target_through_actual_adapter_servo_mock():
    from robo_harness.libero_adapter import LiberoAdapter

    env = LiberoAdapter.__new__(LiberoAdapter)
    env.delta_axis = 0.001
    env.frame, env.horizon, env.gripper = 0, 120, 0.5
    env.raw = {"robot0_eef_pos": np.zeros(3), "robot0_eef_quat": [0, 0, 0, 1]}
    commands, frames = [], []

    def step(command):
        commands.append(command)
        env.raw["robot0_eef_pos"] += command[:3] * 0.02
        return env.raw, 0, False, {}

    def observe():
        return {
            "frame_id": env.frame,
            "vision": {},
            "arms": {
                "arm": {
                    "xyz_world_m": env.raw["robot0_eef_pos"].tolist(),
                    "quaternion_xyzw": env.raw["robot0_eef_quat"],
                }
            },
        }

    env.observe = observe
    env.on_step = lambda obs, command: frames.append(obs["frame_id"])
    controller = SimpleNamespace(output_max=np.ones(6) * 0.02, output_min=-np.ones(6) * 0.02)
    env.env = SimpleNamespace(
        robots=[SimpleNamespace(controller=controller)], step=step, check_success=lambda: False
    )
    tools = ToolRegistry(env, advisory_controls=True, delta_axis=0.001)
    result = tools.execute(
        "move_toward", {"frame_id": 0, "arm": "arm", "target_xyz_world_m": [0.2, 0.1, 0]}, observe()
    )
    assert result["target_reached"] and len(commands) >= 10
    assert max(abs(commands[0][:3])) <= 1
    assert frames == list(range(1, env.frame + 1))


def test_last_tick_undertracking_does_not_become_whole_transit_failure():
    from robo_harness.feedback_presentation import actor_feedback

    robot = Robot()
    original = robot.move_to_position

    def servo(*args):
        original(*args)
        return {"status": "undertracking"}

    robot.move_to_position = servo
    result = go(robot, ToolRegistry(robot), [0.2, 0, 0.5])
    assert (
        result["target_reached"]
        and result["native_last_step_diagnostic"]["status"] == "undertracking"
    )
    assert "undertracking" not in json.dumps(actor_feedback(result))
    assert actor_feedback(robot.last_feedback)["target_reached"]


def test_one_model_request_can_complete_transit_and_archive_full_result(monkeypatch, tmp_path):
    import robo_harness.runtime as runtime

    robot = Robot()
    original = robot.observe
    robot.observe = lambda: {**original(), "instruction": "Transit test"}
    calls = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            calls.append(json)
            name = "move_toward" if len(calls) == 1 else "done"
            args = (
                {"frame_id": 0, "arm": "hand", "target_xyz_world_m": [0.2, 0.1, 0.5]}
                if name == "move_toward"
                else {"summary": "test finished"}
            )
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": str(len(calls)),
                        "type": "function",
                        "function": {"name": name, "arguments": __import__("json").dumps(args)},
                    }
                ],
            }
            return httpx.Response(
                200, json={"choices": [{"message": message}]}, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = run_agent(
        robot, tmp_path, "https://example.invalid", "fake", max_calls=2, delta_axis=0.001
    )
    assert result["llm_calls"] == 2 and result["native_steps"] == 3 and result["tool_errors"] == 0
    rows = [
        json.loads(line) for line in (tmp_path / "episode_history.jsonl").read_text().splitlines()
    ]
    assert rows[0]["executed_results"][0]["result"]["target_reached"]
    assert rows[1]["frame_id"] == 3
