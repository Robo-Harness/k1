import json
from types import SimpleNamespace

import numpy as np
import pytest

from robo_harness.geometry import unproject
from robo_harness.perception import fit_geometry
from robo_harness.runtime import ToolRegistry, history_window, measurement_overlay, run_agent


def scene():
    camera = {
        "color": np.zeros((100, 120, 3), np.uint8),
        "depth": np.ones((100, 120)),
        "intrinsic_matrix": np.array([[100, 0, 60], [0, 100, 50], [0, 0, 1.0]]),
        "extrinsic_matrix": np.eye(4),
    }
    return {
        "frame_id": 0,
        "vision": {"custom_camera": camera},
        "arms": {
            "custom_arm": {
                "xyz_world_m": [0, 0, 0],
                "axes_world": [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            }
        },
        "instruction": "arbitrary task",
    }


def test_portable_optical_world_axes():
    obs = scene()
    cam = obs["vision"]["custom_camera"]
    np.testing.assert_allclose(unproject(cam, [70, 60])["xyz_world_m"], [0.1, 0.1, 1])
    plane = fit_geometry(cam, [10, 10, 100, 90])
    json.dumps(plane)
    assert plane["xyz_world_m"][2] == pytest.approx(1)


def test_measurement_receipt_is_current_camera_only_and_preserves_sensor():
    raw = np.zeros((100, 100, 3), np.uint8)
    point = {"camera": "cam", "frame_id": 3, "pixel": [50, 50], "id": "P1"}
    assert measurement_overlay(raw, "cam", 3, [point]).any()
    assert not raw.any()
    assert not measurement_overlay(raw, "cam", 4, [point]).any()
    assert not measurement_overlay(raw, "other_cam", 3, [point]).any()


def test_history_truncation_never_orphans_a_tool_result():
    history = []
    for i in range(12):
        history.extend(
            [
                {"role": "user", "content": f"Decision {i}"},
                {"role": "assistant", "tool_calls": [{"id": str(i)}]},
                {"role": "tool", "tool_call_id": str(i), "name": "measure_depth"},
            ]
        )
        if i == 7:
            history.append({"role": "user", "content": "No tool call received"})
    window = history_window(history)
    assert window[0]["role"] != "tool"
    for i, message in enumerate(window):
        if message["role"] == "tool":
            assert window[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]


def test_history_keeps_eight_complete_interactions_despite_empty_responses():
    history = []
    for i in range(12):
        history.extend(
            [
                {"role": "user", "content": str(i)},
                {"role": "assistant", "tool_calls": [{"id": str(i)}]},
                {"role": "tool", "tool_call_id": str(i)},
            ]
        )
        if i % 2 == 0:
            history.append({"role": "user", "content": "No tool call received"})
    window = history_window(history, rounds=8)
    assert window[0]["content"] == "4"
    assert sum(m["role"] == "assistant" for m in window) == 8
    assert sum(m["role"] == "assistant" for m in history_window(history, rounds=5)) == 5


def test_agent_loop_keeps_protocol_pairs_and_counts_actual_requests(monkeypatch, tmp_path):
    import httpx
    import robo_harness.runtime as runtime

    obs = scene()
    adapter = SimpleNamespace(observe=lambda: obs, frame=0, horizon=600, success=lambda: False)
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            messages = json["messages"]
            for i, message in enumerate(messages):
                if message["role"] == "assistant" and message.get("tool_calls"):
                    assert messages[i - 1]["role"] in ("user", "tool")
                if message["role"] == "tool":
                    assert messages[i - 1]["role"] == "assistant"
                    assert messages[i - 1]["tool_calls"][0]["id"] == message["tool_call_id"]
                    assert message["name"]
            n = len(requests)
            requests.append(json)
            # One empty model response creates an odd-length history.
            if n == 6:
                message = {"role": "assistant", "content": ""}
            else:
                name = "done" if n == 19 else "move_relative"
                args = (
                    {"summary": "test complete"}
                    if name == "done"
                    else {
                        "frame_id": 0,
                        "arm": "custom_arm",
                        "delta": [0, 0, 0.01],
                        "frame": "world",
                        "note": "test",
                    }
                )
                message = {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": str(n),
                            "type": "function",
                            "function": {"name": name, "arguments": __import__("json").dumps(args)},
                        }
                    ],
                }
            return httpx.Response(
                200, json={"choices": [{"message": message}]}, request=httpx.Request("POST", url)
            )

    adapter.move = lambda *args: {"actual_translation_m": [0, 0, 0]}
    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = run_agent(adapter, tmp_path, "https://example.invalid/v1", "fake", max_calls=25)
    assert result["llm_calls"] == len(requests) == 20
    assert result["termination"] == "model_done"
    rows = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert rows[0]["context"]["motion"]["feedback"] == {}
    assert rows[0]["context"]["arrival"] == {}
    archived = [
        json.loads(line) for line in (tmp_path / "episode_history.jsonl").read_text().splitlines()
    ]
    assert len(archived) == 20
    assert archived[6]["response"]["content"] == ""
    assert archived[6]["executed_tool_count"] == 0
    assert (tmp_path / "model_responses/call_000006.json").exists()


def test_registry_accepts_arbitrary_arm_camera_and_rejects_stale_requests():
    obs = scene()
    adapter = SimpleNamespace(observe=lambda: obs)
    registry = ToolRegistry(adapter)
    schemas = registry.schemas(obs)
    assert "custom_camera" in json.dumps(schemas) and "custom_arm" in json.dumps(schemas)
    assert "cam_head" not in json.dumps(schemas)
    measured = registry.execute(
        "measure_depth",
        {"frame_id": 0, "camera": "custom_camera", "uv": [70, 60], "coordinate_space": "pixels"},
        obs,
    )
    np.testing.assert_allclose(measured["delta_from_tcp_m"]["custom_arm"], [0.1, 0.1, 1])
    with pytest.raises(ValueError, match="Stale"):
        registry.execute("measure_depth", {"frame_id": 1}, obs)


def test_generic_local_frame_conversion():
    obs = scene()
    captured = []
    adapter = SimpleNamespace(move=lambda *args: captured.append(args) or {})
    registry = ToolRegistry(adapter)
    registry.execute(
        "move_relative",
        {"frame_id": 0, "arm": "custom_arm", "frame": "gripper", "delta": [0.01, 0, 0]},
        obs,
    )
    np.testing.assert_allclose(captured[0][1], [0, 0.01, 0], atol=1e-12)


def test_waypoint_reaches_full_target_and_reports_remaining_distance():
    from test_motion import Robot

    robot = Robot()
    registry = ToolRegistry(robot)
    args = {"frame_id": 0, "arm": "hand", "target_xyz_world_m": [0.3, 0.4, 0.5]}
    result = registry.execute("move_toward", args, robot.observe())
    np.testing.assert_allclose(robot.calls[0][0], [0.3, 0.4, 0])
    assert result["remaining_distance_m"] == pytest.approx(0)
    assert result["target_reached"] and result["delta_axis_applies"] is False
    for key in ["max_distance_m", "delta_axis", "steps"]:
        with pytest.raises(ValueError, match="unsupported"):
            registry.execute(
                "move_toward", {**args, "frame_id": robot.frame, key: 0.01}, robot.observe()
            )
