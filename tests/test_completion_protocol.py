from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from robo_harness.completion_protocol import bind_protocol, policy_success
from robo_harness.runtime import ToolRegistry, run_agent
from test_portable_harness import scene


def fixture():
    obs = scene()
    calls = []

    def success():
        calls.append(True)
        return False

    adapter = SimpleNamespace(frame=0, horizon=1200, observe=lambda: deepcopy(obs), success=success)
    return adapter, obs, calls


def test_done_reuses_only_the_identical_physical_sensor_snapshot():
    adapter, obs, calls = fixture()
    registry = ToolRegistry(adapter, completion_mode="feedback")
    first = registry.execute("done", {"summary": "claim"}, obs)
    second = registry.execute("done", {"summary": "new wording is not new evidence"}, obs)
    assert len(calls) == 1 and second["completion_check_reused"]
    assert first["completion_check"] == second["completion_check"]
    obs["vision"]["custom_camera"]["color"][0, 0, 0] += 1
    registry.execute("done", {"summary": "changed sensor at same frame number"}, obs)
    assert len(calls) == 2
    obs["arms"]["custom_arm"]["xyz_world_m"][0] += 0.01
    registry.execute("done", {"summary": "changed robot measurement"}, obs)
    assert len(calls) == 3
    obs["frame_id"] += 1
    registry.execute("done", {"summary": "next physical frame"}, obs)
    assert len(calls) == 4


def test_completion_mode_is_fixed_and_submission_does_not_query_oracle():
    adapter, obs, calls = fixture()
    registry = ToolRegistry(adapter, completion_mode="submission_only")
    result = registry.execute("done", {"summary": "claim"}, obs)
    assert result["submitted"] and result["stop_requested"] and not calls
    assert "completion_check" not in result and not policy_success(adapter)
    with pytest.raises(ValueError):
        bind_protocol(adapter, "feedback")


def test_protocol_is_propagated_to_native_process_once_before_binding():
    adapter, obs, calls = fixture()
    modes = []
    adapter.configure_completion_protocol = lambda mode: modes.append(mode)
    bind_protocol(adapter, "submission_only")
    bind_protocol(adapter, "submission_only")
    assert modes == ["submission_only"]
    with pytest.raises(ValueError):
        bind_protocol(adapter, "feedback")
    assert modes == ["submission_only"]


def test_registry_requires_native_step_observations_for_tracking():
    adapter, obs, calls = fixture()
    adapter.on_step = None
    adapter.step_observations_required = False
    ToolRegistry(adapter)
    assert adapter.step_observations_required is True


def test_submission_runtime_and_macros_never_read_task_truth_before_final_evaluation(
    monkeypatch, tmp_path
):
    import robo_harness.runtime as runtime

    adapter, obs, oracle_calls = fixture()
    requests = []
    adapter.on_step = None

    def move_to(arm, xyz, q, gripper=None, steps=1):
        old = np.array(obs["arms"][arm]["xyz_world_m"])
        obs["arms"][arm]["xyz_world_m"] = list(xyz)
        obs["arms"][arm]["quaternion_xyzw"] = list(q)
        for _ in range(steps):
            adapter.frame += 1
            obs["frame_id"] = adapter.frame
            if adapter.on_step:
                adapter.on_step(deepcopy(obs), [])
        return {"actual_translation_m": (np.array(xyz) - old).tolist()}

    def move(arm, delta, rotation_deg=(0, 0, 0), gripper=None, steps=8):
        s = obs["arms"][arm]
        return move_to(
            arm, np.array(s["xyz_world_m"]) + delta, s["quaternion_xyzw"], gripper, steps
        )

    adapter.move_to_position = move_to
    adapter.move = move

    def success():
        assert len(requests) == 3, "Oracle read while the policy was still deciding"
        oracle_calls.append(True)
        return True

    adapter.success = success

    class Client:
        def __init__(self, **kw):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            index = len(requests)
            requests.append(deepcopy(json))
            if index == 0:
                name, args = (
                    "move_toward",
                    dict(
                        arm="custom_arm",
                        frame_id=adapter.frame,
                        target_xyz_world_m=(
                            np.array(obs["arms"]["custom_arm"]["xyz_world_m"]) + [0.01, 0, 0]
                        ).tolist(),
                    ),
                )
            elif index == 1:
                name, args = (
                    "set_gripper",
                    dict(arm="custom_arm", frame_id=adapter.frame, gripper=0.0),
                )
            else:
                name, args = "done", dict(summary="submit based on visible evidence")
            message = dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        id=str(index),
                        type="function",
                        function=dict(name=name, arguments=__import__("json").dumps(args)),
                    )
                ],
            )
            return httpx.Response(
                200, json={"choices": [{"message": message}]}, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = run_agent(
        adapter,
        tmp_path,
        "https://example.invalid",
        "unused",
        max_calls=4,
        completion_mode="submission_only",
    )
    assert result["llm_calls"] == 3 and result["tool_errors"] == 0
    assert result["native_success"] and result["completion_mode"] == "submission_only"
    assert len(oracle_calls) == 1 and result["requested_completion_native_queries"] == 0
    for request in requests:
        text = json.dumps(request)
        assert '"native_success"' not in text and '"completion_check"' not in text
    rows = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert rows[-1]["results"][0]["result"]["submitted"]


def test_conflicting_protocol_fails_before_episode_creation(tmp_path):
    with pytest.raises(ValueError):
        run_agent(
            None,
            tmp_path / "unused",
            "unused",
            "unused",
            completion_mode="submission_only",
            completion_feedback=True,
        )
    assert not (tmp_path / "unused").exists()
