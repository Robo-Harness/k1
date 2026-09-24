import base64
from copy import deepcopy
import json
from types import SimpleNamespace

import httpx
import pytest

from robo_harness.episode_history import EpisodeHistory
from robo_harness.task_progress import TaskProgress
from robo_harness.runtime import ToolRegistry, history_window, run_agent
from test_portable_harness import scene


def append(history, n):
    filename = f"image_{n}.jpg"
    (history.output / filename).write_bytes(f"jpeg-{n}".encode())
    history.append(
        {
            "call_id": n,
            "frame_id": n,
            "context": {"instruction": "碗放盘子", "arms": {}},
            "diagnostic_context": {"secret_oracle": "not actor input"},
            "response": {
                "content": f"unique-{n}",
                "reasoning_details": "private",
                "tool_calls": [{"id": str(n)}, {"id": "unexecuted"}],
            },
            "model_results": [{"name": "measure_depth", "result": {"x": n}}],
        },
        [{"index": 0, "path": filename, "caption": f"CURRENT cam frame {n}"}],
        1,
    )


def milestone(key="pick", status="planned", refs=None):
    return {
        "id": key,
        "description": "check visible target response",
        "status": status,
        "evidence_call_ids": refs or [],
        "note": "model hypothesis",
    }


def test_full_archive_search_and_lossless_pages(tmp_path):
    archive = EpisodeHistory(tmp_path)
    for n in range(17):
        append(archive, n)
    assert len(archive.path.read_text().splitlines()) == 17
    assert archive.search("unique-0")["matches"][0]["call_id"] == 0
    page = archive.search(tool_name="measure_depth", limit=4)
    assert page["total_matches"] == 17 and page["next_offset"] == 4
    assert archive.search(offset=16)["next_offset"] is None
    assert archive.search(start_call=2, end_call=3)["total_matches"] == 2
    assert archive.search("SECRET_ORACLE")["total_matches"] == 0
    assert archive.search("private")["total_matches"] == 0
    offset, chunks = 0, []
    while offset is not None:
        page = archive.read(0, offset=offset, max_chars=37)
        chunks.append(page["serialized_json_slice"])
        offset = page["next_offset"]
    record = json.loads("".join(chunks))
    assert record == archive.get(0)
    assert record["proposed_tool_count"] == 2 and record["executed_tool_count"] == 1
    record["context"]["instruction"] = "mutated"
    assert archive.get(0)["context"]["instruction"] == "碗放盘子"
    with pytest.raises(FileExistsError):
        EpisodeHistory(tmp_path)


@pytest.mark.parametrize("bad", [-1, True, 0.5, "1", 8])
def test_no_future_or_invalid_citations(tmp_path, bad):
    archive = EpisodeHistory(tmp_path)
    append(archive, 0)
    with pytest.raises(ValueError):
        archive.get(bad)


def test_k_images_are_exact_nonrecursive_and_labeled(tmp_path):
    archive = EpisodeHistory(tmp_path)
    for n in range(3):
        append(archive, n)
    # Additional archived panels must not recursively become recent camera inputs.
    archive.records[2]["images"].append(archive.records[0]["images"][0])
    assert archive.recent_image_blocks(0, 2) == []
    for k, expected in [
        (1, [b"jpeg-2"]),
        (2, [b"jpeg-1", b"jpeg-2"]),
        (8, [b"jpeg-0", b"jpeg-1", b"jpeg-2"]),
    ]:
        blocks = archive.recent_image_blocks(k, 2)
        payloads = [
            base64.b64decode(b["image_url"]["url"].split(",")[1])
            for b in blocks
            if b["type"] == "image_url"
        ]
        assert payloads == expected
        assert "SAME physical frame" in str(blocks)
    assert "decision call 0" in archive.image_blocks(0, 10)[0]["text"]
    for indices in [[0, 0], [True], [[0]], [10]]:
        with pytest.raises(ValueError):
            archive.image_blocks(0, 10, indices)


def test_progress_upserts_retains_revisions_and_validates_atomically(tmp_path):
    archive = EpisodeHistory(tmp_path)
    append(archive, 0)
    progress = TaskProgress(tmp_path, "immutable original task")
    progress.update(
        {"current_goal": "pick target", "milestones": [milestone(), milestone("place")]},
        archive,
        1,
        0,
    )
    progress.update({"milestones": [milestone(status="completed", refs=[0])]}, archive, 2, 0)
    assert len(progress.context()["milestones"]) == 2
    assert (
        progress.context()["milestones"][0]["verification"]
        == "model_claim_not_environment_certificate"
    )
    saved = progress.path.read_text()
    state = progress.context()
    for patch in [
        {"original_instruction": "replace task"},
        {},
        {"current_stage": "bad", "milestones": [milestone(refs=[99])]},
    ]:
        with pytest.raises(ValueError):
            progress.update(patch, archive, 3, 0)
        assert progress.path.read_text() == saved and progress.context() == state
    progress.update({"milestones": [milestone(status="invalidated", refs=[0])]}, archive, 3, 0)
    revisions = [json.loads(line) for line in progress.path.read_text().splitlines()]
    assert revisions[1]["state"]["milestones"][0]["status"] == "completed"
    assert progress.context()["milestones"][0]["status"] == "invalidated"
    assert progress.context()["original_instruction"] == "immutable original task"


@pytest.mark.parametrize("n,k", [(2, 0), (2, 1), (2, 2), (0, 1)])
def test_loop_memory_beyond_window_and_exact_requests(monkeypatch, tmp_path, n, k):
    import robo_harness.runtime as runtime

    obs = scene()
    obs["vision"]["wrist"] = deepcopy(obs["vision"]["custom_camera"])
    obs["vision"]["wrist"]["color"][:] = 60
    adapter = SimpleNamespace(observe=lambda: obs, frame=0, horizon=600, success=lambda: False)
    requests = []
    actions = [
        (
            "update_task_progress",
            {"current_goal": "locate correct bowl", "milestones": [milestone()]},
        )
    ]
    actions += [("remember", {"key": f"test_{i}", "value": f"observation-{i}"}) for i in range(10)]
    actions += [
        ("search_history", {"query": "", "start_call": 0, "end_call": 0}),
        ("read_history", {"call_id": 0, "include_images": True, "max_chars": 24000}),
        ("update_task_progress", {"milestones": [milestone(status="uncertain", refs=[0])]}),
        ("done", {"summary": "end offline test"}),
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
            message = {
                "role": "assistant",
                "content": "test rationale",
                "tool_calls": [
                    {
                        "id": str(idx),
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
        adapter,
        tmp_path,
        "https://example.invalid/v1",
        "secret-test-key",
        max_calls=20,
        history_rounds=n,
        history_image_rounds=k,
        delta_axis=0.02,
    )
    assert result["llm_calls"] == 15 and result["tool_errors"] == 0
    assert result["delta_axis"] == 0.02
    assert result["task_progress_revision"] == 2 and not result["native_success"]
    rows = [
        json.loads(line) for line in (tmp_path / "episode_history.jsonl").read_text().splitlines()
    ]
    assert len(rows) == 15
    assert rows[0]["context"]["motion"]["translation_limit"]["delta_axis_m"] == 0.02
    assert rows[11]["executed_results"][0]["result"]["matches"][0]["call_id"] == 0
    assert rows[14]["context"]["task_progress"]["milestones"][0]["status"] == "uncertain"
    for idx, request in enumerate(requests):
        assert json.loads((tmp_path / f"model_requests/call_{idx:06d}.json").read_text()) == request
        assert (tmp_path / f"model_responses/call_{idx:06d}.json").exists()
        assert "secret-test-key" not in json.dumps(request)
        assert sum(m["role"] == "assistant" for m in request["messages"]) <= n
        content = request["messages"][-1]["content"]
        images = [b for b in content if b["type"] == "image_url"]
        assert len(images) == 2 + 2 * min(k, idx) + (2 if idx == 13 else 0)
        assert "CURRENT custom_camera" in content[1]["text"]
        assert "CURRENT wrist" in content[3]["text"]
        if idx:
            assert rows[idx]["context"]["task_progress"]["current_goal"] == "locate correct bowl"
        if idx == 13:
            assert "HISTORICAL image from decision call 0" in str(content)
            first_images = [
                b for b in requests[0]["messages"][-1]["content"] if b["type"] == "image_url"
            ]
            assert images[-2:] == first_images
    # The same directory cannot overwrite an earlier episode.
    with pytest.raises(FileExistsError):
        run_agent(adapter, tmp_path, "unused", "unused")


@pytest.mark.parametrize(
    "argument,value",
    [
        ("history_rounds", -1),
        ("history_rounds", True),
        ("history_image_rounds", -1),
        ("history_image_rounds", 1.5),
    ],
)
def test_invalid_window_rejected_before_execution(tmp_path, argument, value):
    with pytest.raises(ValueError):
        run_agent(None, tmp_path / "must_not_exist", "unused", "unused", **{argument: value})
    assert not (tmp_path / "must_not_exist").exists()
    assert history_window([{"role": "user"}], rounds=0) == []


def test_registry_memory_is_opt_in_and_retrieval_never_moves_robot(tmp_path):
    obs = scene()
    registry = ToolRegistry(SimpleNamespace(observe=lambda: obs))
    assert "search_history" not in {s["function"]["name"] for s in registry.schemas(obs)}
    registry.episode_history = EpisodeHistory(tmp_path)
    append(registry.episode_history, 0)
    registry.task_progress = TaskProgress(tmp_path, obs["instruction"])
    registry.current_call_id = 1
    assert "search_history" in {s["function"]["name"] for s in registry.schemas(obs)}
    result = registry.execute("read_history", {"call_id": 0, "include_images": True}, obs)
    assert result["images_scheduled_for_next_input"] == 1
    registry.execute("search_history", {}, obs)
    assert registry.retrieved_history_images == []
    with pytest.raises(ValueError):
        registry.execute("read_history", {"call_id": 0, "image_indices": [0]}, obs)
