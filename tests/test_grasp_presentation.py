from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from robo_harness.feedback_presentation import actor_feedback
from robo_harness.grasp_geometry import candidates
from robo_harness.grasp_presentation import PRIVATE_CANDIDATE_FIELDS, present_candidates
from robo_harness.grasp_visualization import candidate_panel
from test_grasp_geometry import fixture


def test_preserves_backend_pose_order_and_raw_diagnostics_without_exposing_scores():
    obs, manager = fixture()
    raw = candidates(manager, "S1", "hand", obs)
    before = deepcopy(raw)
    result = present_candidates(raw)
    assert raw == before
    visible = actor_feedback(result)
    assert "grasp_diagnostics" not in visible
    assert result["grasp_diagnostics"] == raw
    for source, target in zip(raw["candidates"], visible["candidates"]):
        assert target["target_quaternion_xyzw"] == source["target_quaternion_xyzw"]
        assert target["candidate_tcp_world_m"] == source["candidate_tcp_world_m"]
        assert target["pregrasp_tcp_world_m"] == source["pregrasp_tcp_world_m"]
        assert not (PRIVATE_CANDIDATE_FIELDS - {"candidate_id"}) & target.keys()
        assert target["candidate_id"].startswith("S1-H")
    assert not any(s in json.dumps(visible).lower() for s in ("score", "rank", "confidence"))

    # ID follows the pose, never its score or backend rank. Presentation does
    # not sort by the hash: output must retain this deliberately reversed order.
    changed = deepcopy(raw)
    changed["candidates"].reverse()
    for c in changed["candidates"]:
        c["score"] = -12345
        c["candidate_id"] = "old-rank-not-shown"
    reverse = actor_feedback(present_candidates(changed))
    assert [c["candidate_id"] for c in reverse["candidates"]] == [
        c["candidate_id"] for c in reversed(visible["candidates"])
    ]


@pytest.mark.parametrize("learned", [False, True])
def test_registry_both_backends_and_schematics_share_neutral_ids(monkeypatch, learned):
    import robo_harness.runtime as runtime

    obs, manager = fixture()
    camera = next(iter(obs["vision"]))
    manager.entries["S1"]["camera"] = camera
    raw = candidates(manager, "S1", "hand", obs)
    if learned:
        raw["provider"] = "graspgen_visible_point_cloud"
        raw["meaning"] = "confidence and visibility scores are used for ranking"
        for c in raw["candidates"]:
            c["visibility_score"] = c.pop("score")
            c["learned_confidence_not_success_probability"] = 0.88
    monkeypatch.setattr(
        runtime, "learned_candidates" if learned else "grasp_candidates", lambda *a: deepcopy(raw)
    )
    profile = dict(endpoint="unused", provider_axes={}, contact_reference_local_m=[])
    registry = runtime.ToolRegistry(
        SimpleNamespace(observe=lambda: obs),
        region_tools=True,
        grasp_provider=profile if learned else None,
    )
    registry.regions.entries = manager.entries
    result = registry.execute("grasp_candidates", dict(frame_id=0, region_id="S1", arm="hand"), obs)
    assert result["grasp_diagnostics"] == raw
    visible = actor_feedback(result)
    if learned:
        np.testing.assert_array_equal(registry.crop, candidate_panel(visible, obs, "hand", camera))
        # Pose lines, pixels and candidate order are unchanged; only the ID
        # header differs. No new camera/crop or hand geometry is synthesized.
        np.testing.assert_array_equal(
            registry.crop[24:], candidate_panel(raw, obs, "hand", camera)[24:]
        )
    else:
        assert registry.crop is None  # Existing geometric fallback behavior.
    description = next(
        s["function"]["description"]
        for s in registry.schemas(obs)
        if s["function"]["name"] == "grasp_candidates"
    )
    assert not any(s in description.lower() for s in ("score", "rank", "confidence"))


def test_loop_request_history_and_retrieval_never_leak_private_grasp_scores(monkeypatch, tmp_path):
    import robo_harness.runtime as runtime

    obs, manager = fixture()
    obs["instruction"] = "arbitrary pick and place"
    raw = candidates(manager, "S1", "hand", obs)
    raw["provider"] = "graspgen_visible_point_cloud"
    raw["meaning"] = "private score ranking formula"
    for c in raw["candidates"]:
        c["visibility_score"] = 0.8
        c["learned_confidence_not_success_probability"] = 0.9
    monkeypatch.setattr(runtime, "learned_candidates", lambda *a: deepcopy(raw))
    # Learned branch needs a known region only for selecting its camera. This
    # fixture sets it without running any segmentation model or oracle.
    original_init = runtime.ToolRegistry.__init__

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.regions.entries["S1"] = {"camera": next(iter(obs["vision"]))}
        self.regions.context = lambda *a: []
        self.regions.update = lambda *a, **kw: None
        self.regions.refresh_cross_views = lambda *a: None
        self.regions.overlay = lambda rgb, *a, **kw: rgb

    monkeypatch.setattr(runtime.ToolRegistry, "__init__", initialize)
    requests = []
    actions = [
        ("grasp_candidates", dict(frame_id=0, region_id="S1", arm="hand")),
        ("read_history", dict(call_id=0, max_chars=24000)),
        ("done", dict(summary="end mocked test")),
    ]

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            idx = len(requests)
            requests.append(deepcopy(json))
            name, args = actions[idx]
            message = dict(
                role="assistant",
                content="observable decision summary",
                tool_calls=[
                    dict(
                        id=str(idx),
                        type="function",
                        function=dict(name=name, arguments=__import__("json").dumps(args)),
                    )
                ],
            )
            return httpx.Response(
                200, json={"choices": [{"message": message}]}, request=httpx.Request("POST", url)
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    adapter = SimpleNamespace(observe=lambda: obs, frame=0, horizon=600, success=lambda: False)
    result = runtime.run_agent(
        adapter,
        tmp_path,
        "https://example.invalid/v1",
        "test-key",
        max_calls=3,
        region_tools=True,
        grasp_provider=dict(endpoint="unused", provider_axes={}, contact_reference_local_m=[]),
    )
    assert result["tool_errors"] == 0
    rows = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    assert rows[0]["results"][0]["result"]["grasp_diagnostics"] == raw
    forbidden = (
        "visibility_score",
        "learned_confidence_not_success_probability",
        "grasp_diagnostics",
        "observed_free_probe_fraction",
        "near_surface_probe_fraction",
        "unknown_probe_fraction",
        "private score ranking formula",
    )
    text = json.dumps(requests) + (tmp_path / "episode_history.jsonl").read_text()
    assert not any(key in text for key in forbidden)
    assert "S1-H" in text
    assert "executed_results" in rows[1]["results"][0]["result"]["serialized_json_slice"]
