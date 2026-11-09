"""Physical-state image history: no model, simulator, GPU or network required."""

import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from robo_harness.episode_history import EpisodeHistory
from robo_harness.runtime import run_agent
from test_portable_harness import scene


def append(archive, frame):
    n = len(archive.records)
    path = f"image_{n}.jpg"
    (archive.output / path).write_bytes(f"image-{n}".encode())
    archive.append(
        dict(
            call_id=n,
            frame_id=frame,
            context={"arms": {}},
            response={"content": f"observation-{n}"},
            model_results=[],
        ),
        [{"path": path, "caption": f"CURRENT camera frame {frame}"}],
        1,
    )


def archive_at(tmp_path, frames):
    a = EpisodeHistory(tmp_path)
    for frame in frames:
        append(a, frame)
    return a


def test_perception_only_calls_cannot_displace_last_pre_motion_image(tmp_path):
    a = archive_at(tmp_path, [0, 0, 0, 12, 12, 12])
    assert a.recent_image_call_ids(1, 12, "distinct_physical_states") == [2]
    assert a.recent_image_call_ids(1, 12, "decisions") == [5]
    assert len(a.records) == 6 and a.search("observation-5")["total_matches"] == 1


def test_no_previous_physical_state_means_no_duplicate_image(tmp_path):
    a = archive_at(tmp_path, [0, 0, 0])
    assert a.recent_image_blocks(1, 0, "distinct_physical_states") == []
    assert a.recent_image_call_ids(1, 0, "decisions") == [2]


@pytest.mark.parametrize("k,expected", [(0, []), (1, [5]), (2, [2, 5]), (20, [2, 5])])
def test_distinct_history_is_chronological_and_takes_latest_camera_set(k, expected, tmp_path):
    a = archive_at(tmp_path, [0, 0, 0, 12, 12, 12, 24, 24])
    assert a.recent_image_call_ids(k, 24, "distinct_physical_states") == expected
    blocks = a.recent_image_blocks(k, 24, "distinct_physical_states")
    payload = [
        base64.b64decode(b["image_url"]["url"].split(",")[1])
        for b in blocks
        if b["type"] == "image_url"
    ]
    assert payload == [f"image-{n}".encode() for n in expected]
    assert "SAME physical frame" not in str(blocks)
    assert a.image_blocks(7, 24)  # Same-frame data remains explicitly retrievable.


def test_selection_does_not_expose_future_frames_or_mutate_archive(tmp_path):
    a = archive_at(tmp_path, [0, 5, 10])
    before = a.path.read_bytes()
    records = deepcopy(a.records)
    assert a.recent_image_call_ids(10, 5, "distinct_physical_states") == [0]
    assert a.path.read_bytes() == before and a.records == records


@pytest.mark.parametrize(
    "k,current,selection",
    [
        (True, 1, "decisions"),
        (-1, 1, "decisions"),
        (1, False, "decisions"),
        (1, -1, "decisions"),
        (1, 1, "unknown"),
        (1, 1, None),
    ],
)
def test_bad_configuration_is_explicit(k, current, selection, tmp_path):
    a = archive_at(tmp_path, [0])
    with pytest.raises(ValueError):
        a.recent_image_call_ids(k, current, selection)


def test_invalid_runtime_mode_fails_before_creating_episode(tmp_path):
    out = tmp_path / "unused"
    with pytest.raises(ValueError):
        run_agent(None, out, "unused", "unused", history_image_selection="bad")
    assert not out.exists()


def test_default_image_selection_matches_decision_history(tmp_path):
    a = archive_at(tmp_path, [0, 0, 12, 12])
    for k in (0, 1, 2, 8):
        assert a.recent_image_blocks(k, 12) == a.recent_image_blocks(k, 12, "decisions")


@pytest.mark.parametrize("selection", ["decisions", "distinct_physical_states"])
@pytest.mark.parametrize("signed_transport", [False, True])
def test_full_runtime_retains_task_archive_and_labels_exact_input_images(
    monkeypatch, tmp_path, selection, signed_transport
):
    import robo_harness.runtime as runtime

    obs = scene()
    obs["vision"]["wrist"] = deepcopy(obs["vision"]["custom_camera"])
    obs["capabilities"] = {}
    adapter = SimpleNamespace(frame=0, horizon=600, success=lambda: False)
    adapter.observe = lambda: deepcopy(obs)

    def move(arm, delta, dr, gripper, steps):
        obs["arms"][arm]["xyz_world_m"] = (
            np.array(obs["arms"][arm]["xyz_world_m"]) + delta
        ).tolist()
        adapter.frame += 12
        obs["frame_id"] = adapter.frame
        for cam in obs["vision"].values():
            cam["color"][:] = adapter.frame
        return {"actual_translation_m": list(delta)}

    adapter.move = move
    actions = [
        ("update_task_progress", {"current_goal": "keep the intended object"}),
        ("remember", {"key": "original", "value": "preserved"}),
        (
            "move_relative",
            {
                "frame_id": 0,
                "arm": "custom_arm",
                "frame": "world",
                "delta": [0.01, 0, 0],
                "note": "probe",
            },
        ),
        ("remember", {"key": "after_move", "value": "observed"}),
        ("remember", {"key": "repeat", "value": "no new sensor time"}),
        (
            "move_relative",
            {
                "frame_id": 12,
                "arm": "custom_arm",
                "frame": "world",
                "delta": [0.01, 0, 0],
                "note": "probe",
            },
        ),
        ("remember", {"key": "second", "value": "observed"}),
        ("read_history", {"call_id": 0, "include_images": True}),
        ("done", {"summary": "end test"}),
    ]
    requests = []
    signatures = [
        {"type": "reasoning.encrypted", "data": f"opaque-test-signature-{i}", "index": 0}
        for i in range(len(actions))
    ]

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            i = len(requests)
            requests.append(deepcopy(json))
            name, args = actions[i]
            message = {
                "role": "assistant",
                "content": "brief decision summary",
                "tool_calls": [
                    {
                        "id": str(i),
                        "type": "function",
                        "function": {"name": name, "arguments": __import__("json").dumps(args)},
                    }
                ],
            }
            if signed_transport:
                message["reasoning_details"] = [signatures[i]]
            return httpx.Response(
                200,
                json={"model": json["model"], "choices": [{"message": message}]},
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = runtime.run_agent(
        adapter,
        tmp_path,
        "https://example.invalid",
        "unused-test-credential",
        max_calls=12,
        history_rounds=2,
        history_image_rounds=1,
        history_image_selection=selection,
    )
    assert result["tool_errors"] == 0 and result["llm_calls"] == 9
    assert result["task_progress_revision"] == 1 and result["history_image_selection"] == selection
    rows = [json.loads(line) for line in (tmp_path / "agent.jsonl").read_text().splitlines()]
    expected = (
        [[], [], [], [2], [2], [2], [5], [5], [5]]
        if selection == "distinct_physical_states"
        else [[], [0], [1], [2], [3], [4], [5], [6], [7]]
    )
    assert [r["history_image_source_call_ids"] for r in rows] == expected
    assert len((tmp_path / "episode_history.jsonl").read_text().splitlines()) == 9
    for i, (r, request) in enumerate(zip(rows, requests)):
        assert r["context"]["instruction"] == obs["instruction"]
        if i:
            assert r["context"]["task_progress"]["current_goal"] == "keep the intended object"
        assert sum(m["role"] == "assistant" for m in request["messages"]) <= 2
        images = [b for b in request["messages"][-1]["content"] if b["type"] == "image_url"]
        assert len(images) == 2 + 2 * len(expected[i]) + (2 if i == 8 else 0)
        archived = json.loads((tmp_path / f"model_requests/call_{i:06d}.json").read_text())
        if signed_transport:
            assert request == archived
            for message in request["messages"]:
                if message["role"] == "assistant":
                    prior = int(message["tool_calls"][0]["id"])
                    assert message["reasoning_details"] == [signatures[prior]]
        else:
            assert archived == request
