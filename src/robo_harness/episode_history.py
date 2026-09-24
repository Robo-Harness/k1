"""Append-only, episode-local actor evidence. Retrieval never supplies scene truth.

All completed decisions remain searchable independently of the prompt window.
Exact requests/responses and JPEGs are archived by the runtime; retrieval exposes
actor-visible records, not diagnostic_context or opaque provider reasoning fields.
"""

import base64
from copy import deepcopy
import json
from pathlib import Path


def nonnegative_integer(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


class EpisodeHistory:
    def __init__(self, output):
        self.output = Path(output)
        self.path = self.output / "episode_history.jsonl"
        # New episode only. Never silently overwrite or attach to an older run.
        with self.path.open("x"):
            pass
        self.records = []

    def get(self, call_id):
        nonnegative_integer(call_id, "call_id")
        if call_id >= len(self.records):
            raise ValueError("Only completed past calls in this episode can be retrieved")
        return deepcopy(self.records[call_id])

    def append(self, row, images, camera_count):
        if row["call_id"] != len(self.records):
            raise ValueError("History requires contiguous completed call IDs")
        response = row["response"]
        record = {
            "call_id": row["call_id"],
            "frame_id": row["frame_id"],
            "context": deepcopy(row["context"]),
            "response": {
                k: deepcopy(response[k]) for k in ("content", "tool_calls") if k in response
            },
            "executed_results": deepcopy(row["model_results"]),
            "decision_trace": deepcopy(row.get("decision_trace")),
            "images": deepcopy(images),
            "current_camera_count": camera_count,
            "proposed_tool_count": len(response.get("tool_calls") or []),
            "executed_tool_count": len(row["model_results"]),
            "request_file": f"model_requests/call_{row['call_id']:06d}.json",
            "response_file": f"model_responses/call_{row['call_id']:06d}.json",
            "meaning": "Historical actor-visible evidence. Proposed extra tools were NOT executed. "
            "Model descriptions/progress claims are not verified facts. Historical coordinates are not current measurements.",
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
        self.records.append(record)

    def search(self, query="", offset=0, limit=5, tool_name=None, start_call=0, end_call=None):
        if not isinstance(query, str) or len(query) > 2000:
            raise ValueError("query must be text up to 2000 characters")
        for name, value in [("offset", offset), ("start_call", start_call)]:
            nonnegative_integer(value, name)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError("limit must be 1..20; paginate to retrieve every match")
        if end_call is not None:
            nonnegative_integer(end_call, "end_call")
            if end_call < start_call:
                raise ValueError("end_call must be >= start_call (inclusive range)")
        if tool_name is not None and not isinstance(tool_name, str):
            raise ValueError("tool_name must be text")
        terms = query.casefold().split()
        matches = []
        for record in self.records:
            cid = record["call_id"]
            if cid < start_call or (end_call is not None and cid > end_call):
                continue
            names = [r["name"] for r in record["executed_results"]]
            if tool_name is not None and tool_name not in names:
                continue
            body = json.dumps(record, ensure_ascii=False).casefold()
            if not all(term in body for term in terms):
                continue
            matches.append(
                {
                    "call_id": cid,
                    "frame_id": record["frame_id"],
                    "executed_tools": names,
                    "tool_errors": [
                        r["result"].get("error")
                        for r in record["executed_results"]
                        if "error" in r["result"]
                    ],
                    "decision_excerpt": str((record.get("decision_trace") or {}).get("raw") or "")[
                        :320
                    ],
                    "details_tool": "read_history",
                }
            )
        page = matches[offset : offset + limit]
        return {
            "matches": page,
            "total_matches": len(matches),
            "offset": offset,
            "next_offset": offset + len(page) if offset + len(page) < len(matches) else None,
            "completed_calls": len(self.records),
            "search_method": "case-insensitive AND substring terms; empty query lists all",
            "meaning": "Episode-local past records only; snippets are not the complete evidence. Use read_history.",
        }

    def read(self, call_id, offset=0, max_chars=6000):
        nonnegative_integer(offset, "offset")
        if type(max_chars) is not int or not 1 <= max_chars <= 24000:
            raise ValueError("max_chars must be 1..24000; paginate without losing record content")
        record = self.get(call_id)
        text = json.dumps(record, ensure_ascii=False)
        if offset > len(text):
            raise ValueError("offset exceeds record length")
        end = min(offset + max_chars, len(text))
        return {
            "call_id": call_id,
            "frame_id": record["frame_id"],
            "offset": offset,
            "total_chars": len(text),
            "next_offset": end if end < len(text) else None,
            "serialized_json_slice": text[offset:end],
            "images": record["images"],
            "meaning": "Concatenate pages in offset order for the full JSON. Character offsets, not bytes. "
            "Images can be requested with include_images and optional image_indices. Nothing here is a fresh observation.",
        }

    def image_blocks(self, call_id, current_frame, indices=None):
        record = self.get(call_id)
        if indices is None:
            indices = list(range(record["current_camera_count"]))
        if not isinstance(indices, list) or len(indices) > 8:
            raise ValueError("Supply up to 8 distinct image_indices per retrieval")
        for index in indices:
            nonnegative_integer(index, "image_index")
        if len(set(indices)) != len(indices):
            raise ValueError("Supply distinct image_indices")
        blocks = []
        for index in indices:
            nonnegative_integer(index, "image_index")
            if index >= len(record["images"]):
                raise ValueError("Unknown historical image index")
            item = record["images"][index]
            payload = (self.output / item["path"]).read_bytes()
            caption = (
                f"HISTORICAL image from decision call {call_id}, frame {record['frame_id']}; "
                f"current frame is {current_frame}. NOT current measurement coordinates. "
                + (
                    "SAME physical frame: no new sensor time elapsed. "
                    if record["frame_id"] == current_frame
                    else ""
                )
                + f"Original archived caption (historical, even if it says CURRENT): {item['caption']}"
            )
            blocks.extend(
                [
                    {"type": "text", "text": caption},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "data:image/jpeg;base64," + base64.b64encode(payload).decode()
                        },
                    },
                ]
            )
        return blocks

    def recent_image_blocks(self, k, current_frame):
        nonnegative_integer(k, "history_image_rounds")
        blocks = []
        for record in self.records[max(0, len(self.records) - k) :] if k else []:
            # Only each decision's original current camera images. Never recursively
            # re-add older history panels or crops that were included in that input.
            arms = {
                arm: {
                    key: value
                    for key, value in state.items()
                    if key in ("xyz_world_m", "quaternion_xyzw", "gripper_opening_m")
                }
                for arm, state in record["context"].get("arms", {}).items()
            }
            blocks.append(
                {
                    "type": "text",
                    "text": f"HISTORICAL robot state at call {record['call_id']}: "
                    + json.dumps(arms)
                    + ". Camera motion may explain image differences.",
                }
            )
            blocks.extend(self.image_blocks(record["call_id"], current_frame))
        return blocks
