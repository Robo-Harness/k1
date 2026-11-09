"""No native simulator is needed to test partial-execution handling."""

from types import SimpleNamespace

import pytest

from robo_harness.in_process_execution import close, motion_entry, native_step
from robo_harness.resource_errors import UncertainExecution
from robo_harness.robosuite_adapter import RobosuiteAdapter
from robo_harness.libero_adapter import LiberoAdapter


@pytest.mark.parametrize("cls", [LiberoAdapter, RobosuiteAdapter])
def test_native_exception_prevents_replay(cls):
    calls = []

    def fail(command):
        calls.append(command)
        raise RuntimeError("synthetic simulator failure after dispatch")

    adapter = cls.__new__(cls)
    adapter.env = SimpleNamespace(step=fail)
    adapter.on_step = None
    with pytest.raises(UncertainExecution):
        adapter._native([0])
    with pytest.raises(UncertainExecution):
        adapter._native([0])
    assert len(calls) == 1


def test_late_receipt_failure_after_native_ack_prevents_replay():
    class Adapter:
        @native_step
        def step(self):
            pass

        @motion_entry
        def move(self):
            self.step()
            raise ValueError("receipt failed")

    adapter = Adapter()
    with pytest.raises(UncertainExecution):
        adapter.move()
    with pytest.raises(UncertainExecution):
        adapter.move()
    assert adapter._execution_attempts == 1


def test_close_is_idempotent():
    calls = []
    adapter = SimpleNamespace(env=SimpleNamespace(close=lambda: calls.append(True)))
    close(adapter)
    close(adapter)
    assert calls == [True]


def test_invalid_second_target_does_not_change_first_gripper():
    adapter = RobosuiteAdapter.__new__(RobosuiteAdapter)
    adapter.joint = False
    adapter.grippers = {"left": 1.0, "right": 1.0}
    adapter.observe = lambda: {
        "arms": {
            name: {"xyz_world_m": [0, 0, 0], "quaternion_xyzw": [0, 0, 0, 1]}
            for name in adapter.grippers
        }
    }
    targets = [
        dict(arm=name, position=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1], gripper=0.0)
        for name in ("left", "unknown")
    ]
    with pytest.raises(ValueError):
        adapter.move_effectors(targets)
    assert adapter.grippers == {"left": 1.0, "right": 1.0}
