"""Separate model completion submission from host-only task evaluation.

The selected protocol is fixed for one adapter lifetime. Legacy behavior remains
explicitly named; submission_only never polls success from policy/control code.
"""

from copy import deepcopy
import hashlib
import json

import numpy as np

from .observation_contract import camera_reference

MODES = ("feedback", "submission_only", "legacy_autostop")


def resolve_mode(mode, completion_feedback=False):
    if mode is None:
        return "feedback" if completion_feedback else "legacy_autostop"
    if mode not in MODES:
        raise ValueError("Unknown completion protocol")
    if completion_feedback and mode != "feedback":
        raise ValueError("completion_feedback conflicts with completion_mode")
    return mode


def state_key(obs):
    # Exclude model notes, tool overlays/feedback and proposed goals. They do not
    # change native task state. Include robot state and exact calibrated sensors.
    arms = {
        arm: {
            key: state.get(key)
            for key in (
                "xyz_world_m",
                "quaternion_xyzw",
                "joint_positions_rad",
                "gripper_opening_m",
                "gripper_joint_positions_m",
                "gripper_command_open",
            )
        }
        for arm, state in obs["arms"].items()
    }
    value = dict(
        frame_id=obs["frame_id"],
        arms=arms,
        cameras={name: camera_reference(obs, name)["snapshot_id"] for name in obs["vision"]},
    )
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, default=lambda v: np.asarray(v).tolist()
        ).encode()
    ).hexdigest()


class CompletionProtocol:
    def __init__(self, mode):
        self._mode = resolve_mode(mode)
        self._key = self._check = None
        self.native_queries = 0

    @property
    def mode(self):
        return self._mode

    def check(self, adapter, obs):
        if self.mode != "feedback":
            raise RuntimeError("Only feedback protocol may expose a completion check")
        key = state_key(obs)
        reused = self._key == key
        if not reused:
            check = dict(frame_id=obs["frame_id"], native_success=bool(adapter.success()))
            self.native_queries += 1
            self._key, self._check = key, check
        return deepcopy(self._check), reused


def bind_protocol(adapter, mode):
    existing = getattr(adapter, "_harness_completion_protocol", None)
    if existing is not None:
        if existing.mode != mode:
            raise ValueError(
                "Completion protocol cannot change within an adapter lifetime; construct a fresh episode"
            )
        return existing
    protocol = CompletionProtocol(mode)
    configure = getattr(adapter, "configure_completion_protocol", None)
    if callable(configure):
        # A process boundary must not silently keep its old success-polling mode.
        configure(mode)
    adapter._harness_completion_protocol = protocol
    return protocol


def policy_success(adapter):
    """For policy-facing control only. Final host evaluation calls success directly."""
    protocol = getattr(adapter, "_harness_completion_protocol", None)
    if protocol is not None and protocol.mode == "submission_only":
        return False
    return bool(adapter.success())


def native_terminated(adapter):
    """A native environment may terminate internally; do not ask why or retry it."""
    check = getattr(adapter, "episode_ended", None)
    return bool(check()) if callable(check) else False
