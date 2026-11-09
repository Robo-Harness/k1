from copy import deepcopy
import json
from types import SimpleNamespace
import numpy as np
import pytest
from robo_harness.gripper_monitor import GripperMonitor, measurement, command_target
from robo_harness.gripper_signals import mujoco_gripper_measurement


def obs(frame, value=0.8, command=0.0, unit="normalized", arm="left", source="driver_joint"):
    return dict(
        frame_id=frame,
        capabilities=dict(control_dt=1 / 15),
        arms={
            arm: dict(
                gripper_command_open=command,
                gripper_measurement=dict(
                    value=value,
                    unit=unit,
                    valid=True,
                    source=source,
                    limits=[0, 1] if unit == "normalized" else [0, 0.08],
                ),
            )
        },
    )


def test_command_echo_is_never_a_measured_gap():
    for flag in (False, True, None):
        assert not measurement(dict(gripper_opening_normalized=0.8, gripper_opening_measured=flag))[
            "valid"
        ]


@pytest.mark.parametrize("unit,scale", [("normalized", 1.0), ("m", 0.08)])
def test_event_survives_recovery_and_is_delivered_once(tmp_path, unit, scale):
    monitor = GripperMonitor()
    monitor.set_journal(tmp_path / "events.jsonl")
    for frame, value in enumerate([0.8] * 5 + [0.1, 0, 0, 0.8, 0.8, 0.8]):
        monitor.update(obs(frame, value * scale, unit=unit))
    events = monitor.drain_events()
    assert events[0]["kind"] == "opening_changed_while_command_held"
    assert events[0]["frame_id"] == 6 and events[0]["before"] == pytest.approx(0.8 * scale)
    assert events[0]["unit"] == unit and events[0]["advisory_only"]
    assert monitor.drain_events() == []
    assert [json.loads(l) for l in (tmp_path / "events.jsonl").read_text().splitlines()] == events
    assert monitor.protocol()["interrupts_motion"] is False


def test_normal_closing_and_opening_do_not_claim_slip():
    m = GripperMonitor()
    for i, value in enumerate(np.linspace(1, 0, 20).tolist() + [0] * 5):
        m.update(obs(i, float(value)))
    for i, value in enumerate(np.linspace(0, 1, 20).tolist() + [1] * 5, 25):
        m.update(obs(i, float(value), command=1))
    assert not m.events


def test_noise_and_one_frame_glitch_do_not_report_contact_change():
    m = GripperMonitor()
    for i, value in enumerate([0.8, 0.801, 0.799, 0.8, 0.1, 0.8, 0.799, 0.8]):
        m.update(obs(i, value))
    assert not m.events


@pytest.mark.parametrize("kind", ["gap", "same_frame", "reverse", "command", "source"])
def test_invalid_comparisons_do_not_create_changes(kind):
    m = GripperMonitor()
    for i in range(3):
        m.update(obs(i))
    start = {"gap": 20, "same_frame": 2, "reverse": 0}.get(kind, 3)
    for i in range(start, start + 3):
        m.update(
            obs(
                i,
                0.0,
                command=0.2 if kind == "command" else 0,
                source="new" if kind == "source" else "driver_joint",
            )
        )
    assert not any(e["kind"] == "opening_changed_while_command_held" for e in m.events)


def test_each_arm_has_its_own_reference():
    m = GripperMonitor()
    for i in range(8):
        a = obs(i, 0.8 if i < 4 else 0)
        a["arms"].update(obs(i, 0.7, arm="right")["arms"])
        m.update(a)
    assert len(m.events) == 1 and m.events[0]["arm"] == "left"


def test_dynamic_mesh_limit_estimate_is_not_a_new_sensor_each_frame():
    m = GripperMonitor()
    for i in range(12):
        a = obs(i, 0.04, unit="m")
        a["arms"]["left"]["gripper_measurement"]["limits"] = [
            0.001 + i * 0.0001,
            0.081 + i * 0.0001,
        ]
        m.update(a)
    assert not m.events and m.context()["left"]["settled"]


def test_missing_then_recovered_is_one_event_per_transition():
    m = GripperMonitor()
    for i in range(3):
        m.update(obs(i))
    for i in range(3, 8):
        a = obs(i)
        a["arms"]["left"]["gripper_measurement"]["valid"] = False
        m.update(a)
    m.update(obs(8))
    assert [e["kind"] for e in m.events] == ["measurement_unavailable", "measurement_restored"]


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_invalid_measurements_never_become_numeric_evidence(bad):
    m = GripperMonitor()
    m.update(obs(0, bad))
    assert not m.context()["left"]["measurement"]["valid"]
    json.dumps(m.context(), allow_nan=False)


def test_normalized_settling_without_fabricated_metric_gap():
    m = GripperMonitor()
    for i in range(3):
        m.update(obs(i, 0))
    c = m.context()["left"]
    assert c["settled_last3_native_frames"] is True
    assert c["near_closed_reference"] is True and c["opening_m"] is None
    assert c["near_closed_mechanical_limit"] is None
    assert "grasp_success" not in c


def test_no_response_report_is_about_measurement_not_failed_grasp():
    m = GripperMonitor()
    m.update(obs(0, 1, command=1))
    for i in range(1, 20):
        m.update(obs(i, 1, command=0))
    assert len(m.events) == 1 and m.events[0]["kind"] == "no_measurable_response_to_command"
    assert "not a grasp or fault verdict" in m.events[0]["meaning"]


def test_unchanged_initial_state_does_not_invent_an_issued_command():
    m = GripperMonitor()
    for i in range(30):
        m.update(obs(i, 0.8))
    assert not m.events


def test_measurement_observer_does_not_mutate_input():
    m = GripperMonitor()
    a = obs(0)
    before = deepcopy(a)
    m.update(a)
    c = m.context()
    c["left"]["measurement"]["value"] = 9
    assert a == before and m.context()["left"]["measurement"]["value"] == 0.8


@pytest.mark.parametrize("value", [-0.04, 1.0481817857785658, 2.0])
def test_nominal_range_excursion_keeps_measured_value_not_sensor_failure(value):
    m = GripperMonitor()
    for i in range(3):
        m.update(obs(i, 1.0, command=1))
    for i in range(3, 8):
        m.update(obs(i, value, command=1))
    measured = m.context()["left"]["measurement"]
    assert measured["valid"] and measured["value"] == value and not measured["within_nominal_range"]
    assert not any(e["kind"] == "measurement_unavailable" for e in m.events)


def test_negative_metric_gap_remains_invalid():
    assert not measurement(obs(0, -0.01, unit="m")["arms"]["left"])["valid"]


def test_actor_uses_typed_measurement_without_ambiguous_legacy_boolean():
    from robo_harness.feedback_presentation import actor_feedback

    raw = obs(0)["arms"]["left"]
    raw.update(gripper_opening_measured=False, gripper_opening_normalized=0.8)
    view = actor_feedback(raw)
    assert "gripper_opening_measured" not in view and "gripper_opening_normalized" not in view
    assert view["gripper_measurement"]["valid"]
    assert "gripper_opening_measured" in raw


@pytest.mark.parametrize(
    "mode,command,target",
    [
        ("position", 0.25, 0.25),
        ("binary", 0.25, 0),
        ("binary", 0.5, 1),
        ("directional", 0.25, 0),
        ("directional", 0.5, None),
        ("directional", 0.75, 1),
        ("native", 0.25, None),
    ],
)
def test_command_modes_do_not_invent_partial_width_targets(mode, command, target):
    assert command_target(command, mode) == target


def test_directional_hold_does_not_produce_a_no_response_event():
    m = GripperMonitor()
    for i in range(20):
        a = obs(i, 0.8, command=1 if i == 0 else 0.5)
        a["arms"]["left"]["gripper_command_mode"] = "directional"
        m.update(a)
    c = m.context()["left"]
    assert c["settled"] and c["command_target_open"] is None
    assert not m.events


def test_missing_command_does_not_erase_valid_settling_samples():
    m = GripperMonitor()
    for i in range(3):
        m.update(obs(i, 0.8, command=None))
    assert m.context()["left"]["settled"] and not m.events


def test_arbitrary_notes_do_not_crash_the_feedback_formatter():
    from robo_harness.feedback_presentation import actor_feedback

    assert actor_feedback(dict(measurement=None, command_open=0)) == dict(
        measurement=None, command_open=0
    )


def test_nonactuated_tool_does_not_advertise_gripper_actuation():
    from robo_harness.runtime import ToolRegistry
    from test_portable_harness import scene

    a = scene()
    for state in a["arms"].values():
        state["gripper_actuated"] = False
    r = ToolRegistry(
        SimpleNamespace(observe=lambda: a, frame=0, horizon=100, success=lambda: False)
    )
    assert "set_gripper" not in [s["function"]["name"] for s in r.schemas(a)]


@pytest.mark.parametrize("joint_type,actuated,valid", [(2, 1, True), (3, 1, False), (2, 0, False)])
def test_mujoco_angular_joints_or_nonactuated_tools_do_not_masquerade_as_meters(
    joint_type, actuated, valid
):
    sim = SimpleNamespace(
        model=SimpleNamespace(
            joint_name2id=lambda n: 0, jnt_type=[joint_type], jnt_range=[[0, 0.04]]
        ),
        data=SimpleNamespace(get_joint_qpos=lambda n: 0.02),
    )
    g = SimpleNamespace(dof=actuated, joints=["finger"])
    assert mujoco_gripper_measurement(sim, g)["valid"] == valid


def test_normalized_gripper_can_settle_in_twelve_steps_without_interaction_flag():
    from test_motion import Robot
    from robo_harness.runtime import ToolRegistry

    class Native(Robot):
        def __init__(self):
            super().__init__()
            self.on_step = None
            self.opening = 1

        def observe(self):
            a = super().observe()
            a["arms"]["hand"]["gripper_measurement"] = obs(0, self.opening)["arms"]["left"][
                "gripper_measurement"
            ]
            return a

        def move(self, arm, delta, dr, gripper, steps):
            self.gripper = gripper
            for _ in range(steps):
                self.opening = gripper
                self.frame += 1
                if self.on_step:
                    self.on_step(self.observe(), [])
            return dict(actual_translation_m=[0, 0, 0])

    a = Native()
    r = ToolRegistry(a, interaction_feedback=False)
    result = r.execute("set_gripper", dict(arm="hand", frame_id=0, gripper=0), a.observe())
    assert result["native_steps"] == 12 and result["gripper"]["settled"] is True


def test_long_motion_continues_and_event_reaches_model_tool_history_once(monkeypatch, tmp_path):
    import httpx
    from robo_harness import runtime
    from test_portable_harness import scene

    class Native:
        frame = 0
        horizon = 100
        on_step = None

        def observe(self):
            a = scene()
            a["frame_id"] = self.frame
            a["capabilities"] = {"control_hz": 15}
            arm = next(iter(a["arms"]))
            a["arms"][arm].update(
                gripper_command_open=0,
                gripper_measurement=dict(
                    value=0.8 if self.frame < 6 else 0,
                    unit="normalized",
                    source="driver",
                    limits=[0, 1],
                    valid=True,
                ),
            )
            return a

        def success(self):
            return False

        def move(self, arm, delta, rotation_deg, gripper, steps):
            # All requested native ticks execute despite a large opening change.
            for _ in range(steps):
                self.frame += 1
                self.on_step(self.observe(), [])
            return dict(actual_translation_m=[0, 0, 0])

    native = Native()
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            requests.append(deepcopy(json))
            i = len(requests)
            name = "move_relative" if i == 1 else "end_attempt"
            args = (
                dict(
                    frame_id=0,
                    arm=next(iter(native.observe()["arms"])),
                    delta=[0, 0, 0.01],
                    frame="world",
                )
                if i == 1
                else dict(reason="test complete")
            )
            msg = dict(
                role="assistant",
                content="",
                tool_calls=[
                    dict(
                        id=str(i),
                        type="function",
                        function=dict(name=name, arguments=__import__("json").dumps(args)),
                    )
                ],
            )
            return httpx.Response(
                200,
                json=dict(
                    model=json["model"], choices=[dict(message=msg, finish_reason="tool_calls")]
                ),
                request=httpx.Request("POST", url),
            )

    monkeypatch.setattr(runtime.httpx, "Client", Client)
    result = runtime.run_agent(native, tmp_path, "https://invalid.example", "unused", max_calls=2)
    assert native.frame == 12 and result["gripper_event_count"] == 1
    rows = [json.loads(s) for s in (tmp_path / "agent.jsonl").read_text().splitlines()]
    events = rows[0]["model_results"][0]["result"]["gripper_events"]
    assert events[0]["kind"] == "opening_changed_while_command_held"
    assert "gripper_events" not in rows[1]["context"]
    tool_messages = [m for m in requests[1]["messages"] if m.get("role") == "tool"]
    assert json.loads(tool_messages[-1]["content"])["gripper_events"] == events
    assert (tmp_path / "gripper_events.jsonl").read_text().count("\n") == 1
    assert len((tmp_path / "gripper_samples.jsonl").read_text().splitlines()) == 13
