"""Passive, per-arm proprioception receipts. Never commands or stops a robot.

Commanded opening is NOT a measurement. Normalized joint position is NOT a
metric jaw gap. Events survive later recovery and are delivered once, while
the append-only journal retains all of them for audit/training.
"""

from collections import deque
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class MonitorConfig:
    settled_samples: int = 3
    metric_settle_tolerance: float = 0.0005
    normalized_settle_tolerance: float = 0.006
    metric_change_threshold: float = 0.004
    normalized_change_threshold: float = 0.08
    relative_change_threshold: float = 0.5
    command_change_tolerance: float = 1e-6
    no_response_seconds: float = 0.8
    confirmation_samples: int = 2


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def measurement(state):
    """Explicit adapter declaration wins, including unsupported / invalid data."""
    declared = state.get("gripper_measurement")
    if declared is not None:
        m = deepcopy(declared)
    elif (
        state.get("gripper_opening_m") is not None
        and state.get("gripper_opening_measured") is not False
    ):
        # Compatibility for older metric adapters, NOT normalized command echoes.
        m = dict(
            value=state["gripper_opening_m"],
            unit="m",
            source="legacy_metric_opening",
            valid=True,
            limits=state.get("gripper_limits_m"),
        )
    else:
        return dict(valid=False, reason="no_measured_opening", source="unavailable", unit=None)
    if not isinstance(m, dict):
        return dict(valid=False, reason="malformed_measurement", source="unavailable", unit=None)
    if m.get("valid") is not True:
        return dict(
            valid=False,
            reason=m.get("reason", "measurement_unavailable"),
            source=m.get("source", "unavailable"),
            unit=m.get("unit"),
        )
    if m.get("unit") not in ("m", "normalized") or not finite(m.get("value")):
        return dict(
            valid=False,
            reason="invalid_measurement_value_or_unit",
            source=m.get("source"),
            unit=m.get("unit"),
        )
    limits = m.get("limits")
    if limits is not None:
        if (
            not isinstance(limits, (list, tuple))
            or len(limits) != 2
            or not all(finite(v) for v in limits)
            or limits[1] <= limits[0]
        ):
            return dict(
                valid=False,
                reason="invalid_calibration_limits",
                source=m.get("source"),
                unit=m["unit"],
            )
        # Position outside a nominal actuator range is still a real measurement
        # (e.g. compliant joints under contact). Never call it a missing sensor.
        m["within_nominal_range"] = limits[0] <= m["value"] <= limits[1]
    if m["unit"] == "m" and m["value"] < 0:
        return dict(valid=False, reason="negative_opening", source=m.get("source"), unit=m["unit"])
    return m


def command_target(command, mode):
    """Interpret native command semantics for reporting only, never actuate."""
    if command is None:
        return None
    if mode == "position":
        return command
    if mode == "binary":
        return float(command >= 0.5)
    if mode == "directional":
        return None if command == 0.5 else float(command > 0.5)
    return None  # Unknown/native actuator semantics cannot imply a width target.


class GripperMonitor:
    def __init__(self, config=None):
        self.config = config or MonitorConfig()
        self.arms = {}
        self.events = []
        self.delivered = 0
        self.journal = None
        self.samples_journal = None

    def set_journal(self, path):
        samples_path = path.with_name("gripper_samples.jsonl")
        if path.exists() or samples_path.exists():
            raise FileExistsError(path)
        path.touch(exist_ok=False)
        samples_path.touch(exist_ok=False)
        self.journal = path
        self.samples_journal = samples_path

    def _event(self, arm, frame, kind, **values):
        event = dict(
            event_id=f"G{len(self.events) + 1}",
            arm=arm,
            frame_id=frame,
            kind=kind,
            advisory_only=True,
            **values,
        )
        self.events.append(event)
        if self.journal is not None:
            with self.journal.open("a") as stream:
                stream.write(json.dumps(event, allow_nan=False) + "\n")

    def drain_events(self):
        pending = deepcopy(self.events[self.delivered :])
        self.delivered = len(self.events)
        return pending

    def update(self, obs):
        frame = obs["frame_id"]
        caps = obs.get("capabilities", {})
        dt = caps.get("control_dt")
        if not finite(dt) or dt <= 0:
            hz = caps.get("control_hz")
            dt = 1 / hz if finite(hz) and hz > 0 else None
        for arm, state in obs["arms"].items():
            m = measurement(state)
            command = state.get("gripper_command_open")
            command = command if finite(command) and 0 <= command <= 1 else None
            mode = state.get("gripper_command_mode", "position")
            target = command_target(command, mode)
            signature = (m, command)
            d = self.arms.setdefault(
                arm,
                dict(
                    frame=None,
                    samples=deque(maxlen=self.config.settled_samples),
                    reference=None,
                    suspect=None,
                    command=None,
                    current={},
                    measurement=None,
                ),
            )
            if frame == d["frame"] and signature == d.get("signature"):
                continue
            if self.samples_journal is not None:
                with self.samples_journal.open("a") as stream:
                    stream.write(
                        json.dumps(
                            dict(
                                frame_id=frame,
                                arm=arm,
                                command_open=command,
                                measurement=m,
                                control_dt=dt,
                                command_mode=mode,
                                command_target_open=target,
                            ),
                            allow_nan=False,
                        )
                        + "\n"
                    )
            previous = d["measurement"]
            consecutive = d["frame"] is not None and frame == d["frame"] + 1
            comparable = (
                previous is not None
                and m.get("valid")
                and previous.get("valid")
                and m.get("source") == previous.get("source")
                and m.get("unit") == previous.get("unit")
                and m.get("calibration_id") == previous.get("calibration_id")
            )
            changed = (
                (command != d["command"])
                if command is None or d["command"] is None
                else abs(command - d["command"]) > self.config.command_change_tolerance
            )
            if not consecutive or not comparable or changed:
                d["samples"].clear()
                d["reference"] = d["suspect"] = None
                d["command_start"] = dict(frame=frame, value=m.get("value"), issued=False)
                d["no_response_reported"] = False
            if changed and comparable and consecutive:
                # The first observed tick can already move; retain pre-command value.
                d["command_start"] = dict(frame=d["frame"], value=previous["value"], issued=True)
            if previous and previous.get("valid") and not m.get("valid"):
                self._event(arm, frame, "measurement_unavailable", reason=m.get("reason"))
            elif previous and not previous.get("valid") and m.get("valid"):
                self._event(arm, frame, "measurement_restored", source=m["source"], unit=m["unit"])
            elif comparable is False and previous and previous.get("valid") and m.get("valid"):
                self._event(
                    arm, frame, "measurement_source_changed", source=m.get("source"), unit=m["unit"]
                )
            d.update(frame=frame, signature=deepcopy(signature), measurement=m, command=command)
            current = dict(
                measurement=deepcopy(m),
                command_open=command,
                settled=None,
                command_mode=mode,
                command_target_open=target,
                near_closed_mechanical_limit=None,
                frame_id=frame,
            )
            if not m.get("valid"):
                d["current"] = current
                continue
            value, unit = m["value"], m["unit"]
            limits = m.get("limits")
            tol = (
                self.config.metric_settle_tolerance
                if unit == "m"
                else self.config.normalized_settle_tolerance
            )
            threshold = (
                self.config.metric_change_threshold
                if unit == "m"
                else self.config.normalized_change_threshold
            )
            d["samples"].append((frame, value))
            values = [p[1] for p in d["samples"]]
            settled = (
                len(values) == self.config.settled_samples and max(values) - min(values) <= tol
            )
            current["settled"] = settled if len(values) == self.config.settled_samples else None
            if limits:
                current["near_closed_reference"] = value <= limits[0] + (
                    0.002 if unit == "m" else 0.02
                )
                if unit == "m":
                    current["near_closed_mechanical_limit"] = current["near_closed_reference"]
                position = (value - limits[0]) / (limits[1] - limits[0])
                current["command_position_error_normalized"] = (
                    None if target is None else abs(position - target)
                )
            ref = d["reference"]
            if ref and target is not None and target < 0.5:
                delta = value - ref["value"]
                relative = self.config.relative_change_threshold * max(
                    ref["value"] - (limits[0] if limits else 0), threshold
                )
                if abs(delta) > max(threshold, relative):
                    direction = 1 if delta > 0 else -1
                    suspect = d["suspect"]
                    if suspect is None or suspect["direction"] != direction:
                        suspect = dict(direction=direction, count=0, frame=frame)
                    suspect["count"] += 1
                    d["suspect"] = suspect
                    if suspect["count"] >= self.config.confirmation_samples:
                        self._event(
                            arm,
                            frame,
                            "opening_changed_while_command_held",
                            from_frame_id=ref["frame"],
                            first_change_frame_id=suspect["frame"],
                            before=ref["value"],
                            after=value,
                            unit=unit,
                            source=m["source"],
                            command_open=command,
                            elapsed_seconds=(frame - ref["frame"]) * dt if dt else None,
                            meaning="Contact may have changed; check current images. Not proof of slip, attachment or failure.",
                        )
                        d["reference"] = None
                        d["suspect"] = None
                else:
                    d["suspect"] = None
            if settled and d["suspect"] is None:
                # New stable contact becomes the reference; no repeated warning at a plateau.
                d["reference"] = dict(frame=frame, value=sum(values) / len(values))
            start = d.get("command_start", {})
            if (
                dt
                and target is not None
                and limits
                and start.get("issued")
                and not d["no_response_reported"]
                and finite(start.get("value"))
                and (frame - start["frame"]) * dt >= self.config.no_response_seconds
                and current["command_position_error_normalized"] > 0.1
                and abs(value - start["value"]) <= tol
                and settled
            ):
                self._event(
                    arm,
                    frame,
                    "no_measurable_response_to_command",
                    from_frame_id=start["frame"],
                    command_open=command,
                    value=value,
                    unit=unit,
                    meaning="Opening stayed unchanged. Contact can legitimately prevent closure; this is not a grasp or fault verdict.",
                )
                d["no_response_reported"] = True
            d["current"] = current

    def context(self):
        result = {}
        for arm, d in self.arms.items():
            current = deepcopy(d["current"])
            m = current.get("measurement", {})
            current["opening_m"] = (
                m.get("value") if m.get("valid") and m.get("unit") == "m" else None
            )
            current["settled_last3_native_frames"] = current["settled"]
            result[arm] = current
        return result

    def protocol(self):
        return dict(
            version="k1-v1.1-gripper-events",
            config=asdict(self.config),
            control_actions=False,
            interrupts_motion=False,
            delivery="next tool result or decision; never interrupt an in-flight action",
            measurement="Proprioception only; never certifies grasp, slip, or task success",
        )
