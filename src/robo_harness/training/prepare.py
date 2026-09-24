"""Convert exact episode requests into supervised tool-use examples."""

import argparse
import base64
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path


def normalize(messages, image_dir):
    """Keep visible context; never label opaque provider reasoning/signatures."""
    result = copy.deepcopy(messages)
    for message in result:
        message.pop("reasoning_details", None)
        message.pop("reasoning", None)
        if message.get("content") is None:
            message["content"] = ""
        if isinstance(message.get("content"), list):
            for item in message["content"]:
                if item.get("type") != "image_url":
                    continue
                url = item["image_url"]["url"]
                if not url.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
                    raise ValueError("Only inline JPEG/PNG observation images are accepted")
                data = base64.b64decode(url.split(",", 1)[1], validate=True)
                suffix = ".png" if data.startswith(b"\x89PNG") else ".jpg"
                path = image_dir / (hashlib.sha256(data).hexdigest() + suffix)
                if not path.exists():
                    path.write_bytes(data)
                item.clear()
                item.update(type="image", image=str(path.resolve()))
        for call in message.get("tool_calls", []):
            if isinstance(call["function"]["arguments"], str):
                call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return result


def validate_manifest(episodes):
    ids, families = set(), {}
    for episode in episodes:
        if episode["id"] in ids:
            raise ValueError("Duplicate episode ID")
        ids.add(episode["id"])
        split = episode["split"]
        if split not in ("train", "validation", "test"):
            raise ValueError("split must be train, validation or test")
        family = episode["family"]
        if family in families and families[family] != split:
            raise ValueError("Task-family leakage across splits")
        families[family] = split
    if not ids:
        raise ValueError("Empty episode manifest")


def prepare(manifest, output):
    import jsonschema

    episodes = json.loads(manifest.read_text())
    validate_manifest(episodes)
    output.mkdir(parents=True, exist_ok=False)
    images = output / "images"
    images.mkdir()
    counts = Counter()
    streams = {
        name: (output / f"{name}.jsonl").open("x") for name in ("train", "validation", "test")
    }
    try:
        for episode in episodes:
            source = Path(episode["path"])
            if not source.is_absolute():
                source = manifest.resolve().parent / source
            result = json.loads((source / "result.json").read_text())
            if result.get("native_success") is not True:
                counts["excluded_unsuccessful_episode"] += 1
                continue
            for line in (source / "agent.jsonl").read_text().splitlines():
                row = json.loads(line)
                calls = row["response"].get("tool_calls", [])
                executed = row.get("results", [])
                counts["decisions_seen"] += 1
                if len(calls) != 1 or not executed:
                    counts["excluded_call_count"] += 1
                    continue
                call = copy.deepcopy(calls[0])
                name = call["function"]["name"]
                if executed[0]["name"] != name or "error" in executed[0].get("result", {}):
                    counts["excluded_tool_error"] += 1
                    continue
                if name == "done" and not executed[0]["result"].get("stop_requested"):
                    counts["excluded_false_done"] += 1
                    continue
                request_path = source / "model_requests" / f"call_{row['call_id']:06d}.json"
                request = json.loads(request_path.read_text())
                schema = next(
                    (
                        s["function"]["parameters"]
                        for s in request["tools"]
                        if s["function"]["name"] == name
                    ),
                    None,
                )
                try:
                    arguments = call["function"]["arguments"]
                    if isinstance(arguments, str):
                        arguments = json.loads(arguments)
                    if schema is None:
                        raise ValueError("Unknown tool")
                    jsonschema.validate(arguments, schema)
                except (ValueError, jsonschema.ValidationError):
                    counts["excluded_invalid_schema"] += 1
                    continue
                call["function"]["arguments"] = arguments
                sample = {
                    "id": f"{episode['id']}/call_{row['call_id']:06d}",
                    "episode": episode["id"],
                    "family": episode["family"],
                    "frame_id": row["frame_id"],
                    "messages": normalize(request["messages"], images),
                    "tools": request["tools"],
                    "target": {
                        "role": "assistant",
                        "content": row["response"].get("content") or "",
                        "tool_calls": [call],
                    },
                    "request_sha256": hashlib.sha256(request_path.read_bytes()).hexdigest(),
                }
                streams[episode["split"]].write(json.dumps(sample, ensure_ascii=False) + "\n")
                counts["accepted_" + episode["split"]] += 1
    finally:
        for stream in streams.values():
            stream.close()
    (output / "summary.json").write_text(
        json.dumps(
            {
                "counts": dict(counts),
                "hidden_reasoning_labels": False,
                "input_policy": "exact_request_without_future_evidence",
                "note": "Successful episodes can contain suboptimal actions; accepted is not an action optimality label.",
            },
            indent=2,
        )
    )
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="robo-harness k1 supervised-data preparation")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.output)))


if __name__ == "__main__":
    main()
