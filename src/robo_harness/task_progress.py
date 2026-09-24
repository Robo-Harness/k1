"""Model-managed within-episode progress, with append-only revisions and citations.

This module neither chooses milestones nor verifies them from simulator state.
It preserves revisions and validates citation existence, not their interpretation.
"""

from copy import deepcopy
import json
from pathlib import Path


class TaskProgress:
    def __init__(self, output, instruction):
        self.path = Path(output) / "task_progress.jsonl"
        with self.path.open("x"):
            pass
        self.state = {
            "original_instruction": instruction,
            "revision": 0,
            "current_goal": "",
            "current_stage": "",
            "next_step": "",
            "uncertainties": [],
            "milestones": [],
            "meaning": "Model-maintained task progress, NOT environment-verified completion. "
            "Keep observations, plans and claims distinct; revise when contradicted.",
        }

    def context(self):
        return deepcopy(self.state)

    def update(self, args, history, call_id, frame_id):
        allowed = {"current_goal", "current_stage", "next_step", "uncertainties", "milestones"}
        if not args or set(args) - allowed:
            raise ValueError(
                "Supply progress fields; original instruction and revision are harness-owned"
            )
        updated = self.context()

        def text(value, name):
            if not isinstance(value, str) or len(value) > 4000:
                raise ValueError(f"{name} must be text up to 4000 characters")
            return value

        for key in ("current_goal", "current_stage", "next_step"):
            if key in args:
                updated[key] = text(args[key], key)
        if "uncertainties" in args:
            if not isinstance(args["uncertainties"], list):
                raise ValueError("uncertainties must be a list of text")
            updated["uncertainties"] = [text(v, "uncertainty") for v in args["uncertainties"]]
        if "milestones" in args:
            if not isinstance(args["milestones"], list):
                raise ValueError("milestones must be a list of updates")
            milestones = {m["id"]: m for m in updated["milestones"]}
            changed_ids = set()
            for item in args["milestones"]:
                fields = {"id", "description", "status", "evidence_call_ids", "note"}
                if not isinstance(item, dict) or set(item) != fields:
                    raise ValueError(
                        "Each milestone needs id, description, status, evidence_call_ids, note"
                    )
                key = text(item["id"], "milestone id")
                if not key.strip() or len(key) > 80 or key in changed_ids:
                    raise ValueError("Use distinct nonempty milestone IDs up to 80 characters")
                changed_ids.add(key)
                if item["status"] not in (
                    "planned",
                    "in_progress",
                    "completed",
                    "uncertain",
                    "invalidated",
                ):
                    raise ValueError("Unknown milestone status")
                refs = item["evidence_call_ids"]
                if not isinstance(refs, list):
                    raise ValueError("evidence_call_ids must be a list of completed past call IDs")
                for ref in refs:
                    history.get(ref)
                milestones[key] = {
                    "id": key,
                    "description": text(item["description"], "description"),
                    "status": item["status"],
                    "evidence_call_ids": list(refs),
                    "note": text(item["note"], "note"),
                    "updated_at_call": call_id,
                    "updated_at_frame": frame_id,
                    "verification": "model_claim_not_environment_certificate",
                }
            # Unmentioned milestones persist. Retraction uses invalidated/uncertain,
            # not deletion; every previous revision remains in the append-only log.
            updated["milestones"] = list(milestones.values())
        updated["revision"] += 1
        event = {
            "revision": updated["revision"],
            "call_id": call_id,
            "frame_id": frame_id,
            "patch": deepcopy(args),
            "state": updated,
        }
        with self.path.open("a") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
        self.state = updated
        return {
            "revision": updated["revision"],
            "progress": self.context(),
            "citation_check": "IDs exist; their support for the model claim is NOT automatically verified",
        }


def memory_schemas(schema):
    text = {"type": "string"}
    integer = {"type": "integer", "minimum": 0}
    milestone = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": text,
            "description": text,
            "status": {
                "type": "string",
                "enum": ["planned", "in_progress", "completed", "uncertain", "invalidated"],
            },
            "evidence_call_ids": {"type": "array", "items": integer},
            "note": text,
        },
        "required": ["id", "description", "status", "evidence_call_ids", "note"],
    }
    return [
        schema(
            "search_history",
            "Search ALL completed actor-visible decisions in THIS episode. "
            "Keyword AND search or empty query; paginate all matches. No scene truth or cross-task memory.",
            {
                "query": text,
                "offset": integer,
                "limit": {"type": "integer", "minimum": 1, "maximum": 20},
                "tool_name": text,
                "start_call": integer,
                "end_call": integer,
            },
            [],
        ),
        schema(
            "read_history",
            "Read one historical decision in lossless text pages. "
            "Optionally display its archived images next turn; these are HISTORICAL, never current coordinates. "
            "Default images are that decision's current cameras; image_indices select any archived input panel.",
            {
                "call_id": integer,
                "offset": integer,
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 24000},
                "include_images": {"type": "boolean"},
                "image_indices": {"type": "array", "items": integer, "maxItems": 8},
            },
            ["call_id"],
        ),
        schema(
            "update_task_progress",
            "Maintain your OWN episode goal, stage, next step, uncertainties and milestones. "
            "Update after important transitions/failures, not mechanically every turn. "
            "Milestone IDs upsert; omitted milestones persist. Cite completed past call IDs. "
            "completed is YOUR claim, not verified success. Retract with uncertain/invalidated. "
            "Original task is immutable; no fixed task-specific stages are supplied.",
            {
                "current_goal": text,
                "current_stage": text,
                "next_step": text,
                "uncertainties": {"type": "array", "items": text},
                "milestones": {"type": "array", "items": milestone},
            },
            [],
        ),
    ]
