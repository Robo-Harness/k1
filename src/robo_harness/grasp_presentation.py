"""Score-free actor receipts; backend candidate selection/order remain untouched."""

from copy import deepcopy
import hashlib
import json


PRIVATE_CANDIDATE_FIELDS = frozenset(
    {
        "candidate_id",
        "score",
        "visibility_score",
        "learned_confidence_not_success_probability",
        "observed_free_probe_fraction",
        "near_surface_probe_fraction",
        "unknown_probe_fraction",
        "visible_width_fits_open_jaws",
    }
)


def present_candidates(raw):
    """Keep exact poses and backend order, hide evaluative numbers and rank IDs.

    Raw backend receipt is retained only in a diagnostic field, stripped from
    model inputs and retrievable episode history by actor_feedback(). No sorting,
    new filtering, random permutation, or controller mutation happens here.
    """
    result = {k: deepcopy(v) for k, v in raw.items() if k not in ("candidates", "meaning")}
    candidates = []
    for original in raw["candidates"]:
        candidate = {
            k: deepcopy(v) for k, v in original.items() if k not in PRIVATE_CANDIDATE_FIELDS
        }
        pose = {k: original[k] for k in ("candidate_tcp_world_m", "target_quaternion_xyzw")}
        code = hashlib.sha256(json.dumps(pose, sort_keys=True).encode()).hexdigest()[:12]
        candidate["candidate_id"] = f"{raw['region_id']}-H{code}"
        candidates.append(candidate)
    result["candidates"] = candidates
    result["meaning"] = (
        "POSE HYPOTHESES from the current visible target and calibrated hand geometry, "
        "not verified grasps. Candidate IDs identify poses only. Evaluate each option "
        "independently using the original views, target identity, visible jaw fit and "
        "approach direction; you may reject all options and choose your own pose. "
        "No exact mesh/jaw-fit, collision-free path, IK or attachment certificate. "
        "Choose clearance, approach, close, and verify actual target co-motion."
    )
    result["grasp_diagnostics"] = deepcopy(raw)
    return result
