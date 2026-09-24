import base64
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from robo_harness.cli import read_config
from robo_harness.multi_effector import MultiEffectorRegistry
from robo_harness.release_audit import audit
from robo_harness.runtime import run_agent
from robo_harness.training.prepare import normalize, prepare, validate_manifest
from robo_harness.transport import APIClient, api_settings
from robo_harness.robosuite_tasks import sha
from test_portable_harness import scene


def test_task_source_hash_accepts_module_filename_string(tmp_path):
    path = tmp_path / "task.py"
    path.write_text("# synthetic task")
    assert sha(str(path)) == sha(path)


def test_configs_are_loadable():
    root = Path(__file__).resolve().parents[1]
    for name in ("libero-pro", "robosuite"):
        config = read_config(root / "configs" / f"{name}.yaml")
        assert config["harness"]["history_rounds"] == 8
        assert config["harness"]["history_image_rounds"] == 1
        assert config["harness"]["delta_axis"] == 0.03


def test_bad_configuration_is_not_silently_ignored(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("environment:\n  name: robosuite\n  task_typo: cube_lifting\n")
    with pytest.raises(ValueError):
        read_config(path)


def test_no_implicit_credentials_or_insecure_remote_urls(monkeypatch):
    monkeypatch.delenv("ROBO_HARNESS_API_KEY", raising=False)
    monkeypatch.delenv("ROBO_HARNESS_MODEL", raising=False)
    with pytest.raises(ValueError):
        api_settings()
    monkeypatch.setenv("ROBO_HARNESS_API_KEY", "synthetic-test-credential")
    monkeypatch.setenv("ROBO_HARNESS_MODEL", "example/model")
    monkeypatch.setenv("ROBO_HARNESS_BASE_URL", "http://example.org/v1")
    with pytest.raises(ValueError):
        api_settings()
    monkeypatch.setenv("ROBO_HARNESS_BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("ROBO_HARNESS_PROVIDERS", "example-provider")
    assert api_settings()[3]["provider"]["allow_fallbacks"] is False


def test_transport_recognizes_wrapped_provider_failure(monkeypatch):
    monkeypatch.delenv("ROBO_HARNESS_PROXY", raising=False)
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"error": {"code": 503}}))
    with APIClient(transport=transport) as client:
        assert client.post("https://example.org/v1", json={}).status_code == 503


def test_signature_preserved_but_not_archived_as_progress_evidence(tmp_path):
    obs = scene()
    adapter = SimpleNamespace(observe=lambda: obs, frame=0, horizon=50, success=lambda: False)
    received = []
    signature = [{"type": "reasoning.encrypted", "data": "synthetic-provider-signature"}]

    def handler(request):
        payload = json.loads(request.content)
        received.append(payload)
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "reasoning_details": signature,
                            "tool_calls": [
                                {
                                    "id": str(len(received)),
                                    "type": "function",
                                    "function": {
                                        "name": "search_history",
                                        "arguments": json.dumps({"query": "example"}),
                                    },
                                }
                            ],
                        },
                    }
                ]
            },
        )

    result = run_agent(
        adapter,
        tmp_path,
        "https://example.org/v1",
        "synthetic-credential",
        max_calls=2,
        client_factory=lambda **kw: httpx.Client(transport=httpx.MockTransport(handler), **kw),
    )
    assert result["llm_calls"] == 2
    assistant = next(m for m in received[1]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_details"] == signature
    assert "synthetic-provider-signature" not in (tmp_path / "episode_history.jsonl").read_text()
    assert "synthetic-credential" not in (tmp_path / "model_requests/call_000000.json").read_text()


def test_multieffector_dispatch_and_stale_frame():
    obs = scene()
    state = deepcopy(next(iter(obs["arms"].values())))
    obs["arms"] = {"left": state, "right": deepcopy(state)}
    dispatched = []
    adapter = SimpleNamespace(
        observe=lambda: obs,
        move_effectors=lambda t, s: dispatched.append((t, s)) or {"simultaneous": True},
    )
    registry = MultiEffectorRegistry(adapter)
    assert "move_effectors" in {s["function"]["name"] for s in registry.schemas(obs)}
    args = {
        "frame_id": 0,
        "targets": [{"arm": "left", "position": [0, 0, 0.2], "quaternion_xyzw": [0, 0, 0, 1]}],
        "steps": 4,
    }
    assert registry.execute("move_effectors", args, obs)["simultaneous"]
    with pytest.raises(ValueError):
        registry.execute("move_effectors", {**args, "frame_id": 5}, obs)
    assert len(dispatched) == 1


def test_split_validation_rejects_family_leakage():
    rows = [
        {"id": "a", "family": "shared", "split": "train"},
        {"id": "b", "family": "shared", "split": "test"},
    ]
    with pytest.raises(ValueError, match="leakage"):
        validate_manifest(rows)


def test_prepare_exact_request_success_filter_and_visible_targets(tmp_path):
    episode = tmp_path / "episode"
    episode.mkdir()
    requests = episode / "model_requests"
    requests.mkdir()
    (episode / "result.json").write_text(json.dumps({"native_success": True}))
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": "sample_tool", "arguments": '{"x": 1}'},
    }
    image = base64.b64encode(b"synthetic-jpeg").decode()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + image}}
            ],
        }
    ]
    request = {
        "messages": messages,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "sample_tool",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "number"}},
                        "required": ["x"],
                    },
                },
            }
        ],
    }
    (requests / "call_000000.json").write_text(json.dumps(request))
    row = {
        "call_id": 0,
        "frame_id": 0,
        "response": {
            "content": "Observed decision",
            "reasoning_details": "opaque",
            "tool_calls": [call],
        },
        "results": [{"name": "sample_tool", "result": {"measured": 1}}],
    }
    (episode / "agent.jsonl").write_text(json.dumps(row) + "\n")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps([{"id": "a", "path": "episode", "family": "f", "split": "train"}])
    )
    out = tmp_path / "prepared"
    assert prepare(manifest, out)["accepted_train"] == 1
    sample = json.loads((out / "train.jsonl").read_text())
    assert sample["target"]["tool_calls"][0]["function"]["arguments"] == {"x": 1}
    assert "reasoning_details" not in sample["target"]
    assert "measured" not in json.dumps(sample["messages"])
    assert Path(sample["messages"][0]["content"][0]["image"]).is_file()


def test_audit_rejects_secret_binary_and_symlink(tmp_path):
    (tmp_path / "sample.py").write_text('credential = "' + "sk-" + "x" * 25 + '"')
    (tmp_path / "weights.pt").write_bytes(b"not-a-checkpoint")
    (tmp_path / "alias").symlink_to(tmp_path / "sample.py")
    result = audit(tmp_path)
    assert not result["passed"]
    assert {x["rule"] for x in result["findings"]} >= {
        "credential",
        "unexpected_file_type",
        "symlink",
    }
    assert "x" * 25 not in json.dumps(result)
