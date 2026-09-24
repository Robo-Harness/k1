import numpy as np
import pytest

from robo_harness.motion import MotionController
from robo_harness.pose_targets import PoseTargets
from test_motion import Robot


def evidence(status):
    return {
        "status": status,
        "free_probe_fraction": 0.5 if status == "partly_observed_free" else 0,
        "collision_free": False,
    }


def test_prefix_preserves_direction_rotation_ratio_and_gripper(monkeypatch):
    import robo_harness.motion as m

    monkeypatch.setattr(
        m,
        "sweep_evidence",
        lambda obs, arm, delta, dr: evidence(
            "possible_surface_intersection"
            if np.linalg.norm(delta) > 0.016
            else "partly_observed_free"
        ),
    )
    robot = Robot()
    motion = MotionController(robot, adaptive_sweep=True)
    result = motion.move(robot.observe(), "hand", [0.018, 0.024, 0], [0, 0, 12], check_sweep=True)
    d, r, grip = robot.calls[0]
    np.testing.assert_allclose(d, [0.009, 0.012, 0])
    np.testing.assert_allclose(r, [0, 0, 6])
    assert grip is None and robot.gripper == 0.25
    assert result["bounded_prefix"]["accepted_scale"] == 0.5
    assert not result["sweep"]["collision_free"]


@pytest.mark.parametrize("smaller_status", ["unknown", "possible_surface_intersection"])
def test_no_prefix_executes_if_unknown_or_risky(monkeypatch, smaller_status):
    import robo_harness.motion as m

    seen = []

    def check(obs, arm, delta, dr):
        seen.append(np.linalg.norm(delta))
        return evidence("possible_surface_intersection" if len(seen) == 1 else smaller_status)

    monkeypatch.setattr(m, "sweep_evidence", check)
    robot = Robot()
    motion = MotionController(robot, adaptive_sweep=True)
    with pytest.raises(ValueError, match="surface"):
        motion.move(robot.observe(), "hand", [0.03, 0, 0], [0, 0, 0], check_sweep=True)
    assert not robot.calls
    np.testing.assert_allclose(seen, [0.03, 0.015, 0.0075, 0.00375])
    assert motion.context()["feedback"]["hand"]["bounded_prefix"]["accepted_scale"] is None


def test_small_prefix_does_not_disable_no_progress_protection(monkeypatch):
    import robo_harness.motion as m

    monkeypatch.setattr(
        m,
        "sweep_evidence",
        lambda obs, arm, delta, dr: evidence(
            "possible_surface_intersection"
            if np.linalg.norm(delta) > 0.004
            else "partly_observed_free"
        ),
    )
    robot = Robot()
    robot.scale = 0
    motion = MotionController(robot, adaptive_sweep=True)
    for _ in range(3):
        result = motion.move(robot.observe(), "hand", [0.03, 0, 0], [0, 0, 0], check_sweep=True)
        assert result["bounded_prefix"]["accepted_scale"] == 0.125
    with pytest.raises(ValueError, match="Repeated low-progress"):
        motion.move(robot.observe(), "hand", [0.03, 0, 0], [0, 0, 0], check_sweep=True)
    assert len(robot.calls) == 3


def test_absolute_pose_target_is_not_replaced_by_short_prefix(monkeypatch):
    import robo_harness.motion as m

    monkeypatch.setattr(
        m,
        "sweep_evidence",
        lambda obs, arm, delta, dr: evidence(
            "possible_surface_intersection"
            if np.linalg.norm(delta) > 0.016
            else "partly_observed_free"
        ),
    )
    robot = Robot()
    targets = PoseTargets(MotionController(robot, adaptive_sweep=True))
    result = targets.step(robot.observe(), {"arm": "hand", "target_xyz_world_m": [0.09, 0, 0.5]})
    assert result["remaining_distance_m"] == pytest.approx(0.075)
    assert not result["pose_reached"]
    key = result["pose_goal_id"]
    for _ in range(8):
        result = targets.step(robot.observe(), {"arm": "hand", "pose_goal_id": key})
    assert result["pose_reached"]
    np.testing.assert_allclose(result["target_xyz_world_m"], [0.09, 0, 0.5])
