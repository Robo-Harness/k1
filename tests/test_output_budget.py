import json
from copy import deepcopy
from types import SimpleNamespace
import httpx
import pytest

from robo_harness.output_budget import OutputBudget, default_output_budget
from robo_harness import runtime
from test_portable_harness import scene


def test_recovery_is_bounded_and_persists_for_episode():
    budget = OutputBudget()
    assert [budget.after_response("length", False)["next_max_tokens"] for _ in range(4)] == [
        3600,
        7200,
        7200,
        7200,
    ]
    assert budget.after_response("length", True) is None
    assert budget.current == 7200
    assert budget.after_response("stop", False) is None
    assert budget.current == 7200
    assert OutputBudget().current == 1800  # no cross-episode learning/state


def test_gemini38_profile_and_other_models():
    assert default_output_budget("google/gemini-3.8-flash") == 3600
    assert default_output_budget("gemini-3.8-flash") == 3600
    assert default_output_budget("google/gemini-3.7-flash") == 1800
    assert default_output_budget("local/unknown") == 1800


@pytest.mark.parametrize("base,maximum", [(0, 10), (True, 10), (100, 99), (100, False)])
def test_bad_budgets_fail_before_any_model_or_native_call(base, maximum):
    with pytest.raises(ValueError):
        OutputBudget(base, maximum)


def test_hold_required_duration_is_in_the_model_visible_schema():
    obs = scene()
    adapter = SimpleNamespace(observe=lambda: obs, frame=0, horizon=100, success=lambda: False)
    registry = runtime.ToolRegistry(adapter)

    def check(node):
        if isinstance(node, dict):
            if node.get("type") == "object":
                assert set(node.get("required", [])) <= set(node.get("properties", {}))
            for value in node.values():
                check(value)
        elif isinstance(node, list):
            for value in node:
                check(value)

    specs = registry.schemas(obs)
    check(specs)
    params = next(s["function"]["parameters"] for s in specs if s["function"]["name"] == "hold")
    assert "steps" in params["required"] and params["properties"]["steps"]["minimum"] == 1


@pytest.mark.parametrize(
    "settings,expected",
    [
        ({}, [1800, 3600, 3600]),
        ({"model": "google/gemini-3.8-flash"}, [3600, 7200, 7200]),
        ({"model": "google/gemini-3.8-flash", "max_output_tokens": 1000}, [1000, 2000, 2000]),
    ],
)
def test_runtime_retries_only_next_counted_decision_without_executing_motion(
    monkeypatch, tmp_path, settings, expected
):
    obs = scene()
    obs["capabilities"] = {}
    adapter = SimpleNamespace(
        frame=0, horizon=100, observe=lambda: deepcopy(obs), success=lambda: False
    )
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            requests.append(deepcopy(json))
            i = len(requests)
            message = dict(role="assistant", content="")
            if i > 1:
                name, args = (
                    ("remember", {"key": "observed", "value": "no action yet"})
                    if i == 2
                    else ("end_attempt", {"reason": "test complete"})
                )
                message["tool_calls"] = [
                    dict(
                        id=str(i),
                        type="function",
                        function=dict(name=name, arguments=__import__("json").dumps(args)),
                    )
                ]
            return httpx.Response(
                200,
                json=dict(
                    model=json["model"],
                    choices=[
                        dict(message=message, finish_reason="length" if i == 1 else "tool_calls")
                    ],
                ),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = runtime.run_agent(
        adapter, tmp_path, "https://invalid.example", "unused", max_calls=3, **settings
    )
    assert [r["max_tokens"] for r in requests] == expected
    assert (
        result["llm_calls"] == 3
        and result["native_steps"] == 0
        and result["missing_tool_calls"] == 1
    )
    rows = [json.loads(l) for l in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert rows[0]["output_budget_recovery"]["next_max_tokens"] == expected[1]
    assert all(r["frame_id"] == 0 for r in rows)
