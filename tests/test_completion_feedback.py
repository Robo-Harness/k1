import json
from types import SimpleNamespace

import httpx
import pytest

from robo_harness.runtime import run_agent
from test_portable_harness import scene


@pytest.mark.parametrize(
    "enabled,success_on_request,expected_calls,termination,rejections",
    [
        (False, None, 1, "model_done", 0),
        (True, 2, 2, "native_success", 1),
        (True, None, 3, "budget", 3),
    ],
)
def test_sparse_completion_feedback(
    monkeypatch, tmp_path, enabled, success_on_request, expected_calls, termination, rejections
):
    import robo_harness.runtime as runtime

    requests = []
    state = {"success": False}
    adapter = SimpleNamespace(
        observe=scene, frame=0, horizon=1200, success=lambda: state["success"]
    )

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            requests.append(json)
            if success_on_request == len(requests):
                state["success"] = True
            message = {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": str(len(requests)),
                        "type": "function",
                        "function": {"name": "done", "arguments": '{"summary":"finished"}'},
                    }
                ],
            }
            return httpx.Response(
                200, json={"choices": [{"message": message}]}, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = run_agent(
        adapter,
        tmp_path,
        "https://example.invalid/v1",
        "fake",
        max_calls=3,
        completion_feedback=enabled,
        interaction_feedback=True,
    )
    assert result["llm_calls"] == expected_calls
    assert result["termination"] == termination
    assert result["rejected_completion_checks"] == rejections
    if enabled:
        first = json.loads((tmp_path / "agent.jsonl").read_text().splitlines()[0])
        receipt = first["results"][0]["result"]
        assert receipt["completion_check"] == {"frame_id": 0, "native_success": False}
        assert not receipt["stop_requested"]
        assert "Environment has NOT" in str(requests[1]["messages"])
        second = json.loads((tmp_path / "agent.jsonl").read_text().splitlines()[1])
        assert second["context"]["last_requested_completion_check"]["native_success"] is False
