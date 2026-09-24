import numpy as np
import pytest

from test_motion import Robot
from robo_harness.motion import MotionController
from robo_harness.pose_targets import PoseTargets
from robo_harness.runtime import ToolRegistry
from robo_harness.regions import Regions


def test_arrival_is_not_a_release_veto():
    robot = Robot()
    registry = ToolRegistry(robot, advisory_controls=True)
    registry.execute(
        "move_toward",
        dict(arm="hand", frame_id=robot.frame, target_xyz_world_m=[0.09, 0, 0.5]),
        robot.observe(),
    )
    result = registry.execute(
        "move_relative",
        dict(arm="hand", frame_id=robot.frame, delta=[0, 0, 0], frame="world", gripper=1),
        robot.observe(),
    )
    assert robot.gripper == 1
    assert result["pre_action_target_advisory"]["blocking"] is False


def test_sweep_warns_but_preserves_requested_motion(monkeypatch):
    monkeypatch.setattr(
        "robo_harness.motion.sweep_evidence", lambda *a: {"status": "possible_surface_intersection"}
    )
    robot = Robot()
    motion = MotionController(robot, advisory=True)
    result = motion.move(robot.observe(), "hand", [0.01, 0, 0], [0, 0, 0], check_sweep=True)
    assert result["decision_advisories"][0]["kind"] == "possible_surface_intersection"
    assert len(robot.calls) == 1
    with pytest.raises(ValueError, match="translation component"):
        motion.move(robot.observe(), "hand", [1, 0, 0], [0, 0, 0])


def test_low_progress_is_not_veto():
    robot = Robot()
    motion = MotionController(robot, advisory=True)
    motion.low["hand"] = {"count": 3, "direction": np.array([1.0, 0, 0, 0, 0, 0])}
    result = motion.move(robot.observe(), "hand", [0.01, 0, 0], [0, 0, 0])
    assert result["decision_advisories"][0]["kind"] == "repeated_low_progress"
    assert len(robot.calls) == 1


def test_explicit_xyz_and_label_creates_new_goal_without_overwriting():
    robot = Robot()
    poses = PoseTargets(MotionController(robot, advisory=True))
    first = poses.step(robot.observe(), dict(arm="hand", target_xyz_world_m=[0.09, 0, 0.5]))
    old = poses.goals[first["pose_goal_id"]]["xyz"].copy()
    second = poses.step(
        robot.observe(),
        dict(arm="hand", pose_goal_id=first["pose_goal_id"], target_xyz_world_m=[0.06, 0.02, 0.5]),
    )
    assert second["pose_goal_id"] != first["pose_goal_id"]
    np.testing.assert_array_equal(poses.goals[first["pose_goal_id"]]["xyz"], old)
    assert "parameter_resolution" in second


def test_find_cache_is_frame_scoped_and_copied(monkeypatch):
    calls = []
    monkeypatch.setattr("robo_harness.regions.segment_text", lambda *a: calls.append(1) or [])
    regions = Regions()
    obs = {"frame_id": 0, "vision": {"cam": {"color": np.zeros((5, 5, 3))}}}
    args = {"camera": "cam", "query": "object"}
    regions.find(args, obs)
    result = regions.find(args, obs)
    assert result["measurement_reused"] is True and len(calls) == 1
    result["candidates"].append("mutation")
    assert regions.find(args, obs)["candidates"] == []
    obs["frame_id"] = 1
    regions.find(args, obs)
    assert len(calls) == 2


def test_uncertain_axes_not_in_model_context(monkeypatch):
    regions = Regions()
    regions.entries["S1"] = {}
    raw = {
        "major_axis_reliable": False,
        "major_axis_world_unsigned": [1, 0, 0],
        "plane_reliable": False,
        "surface_normal_toward_camera_world": [0, 0, 1],
        "relative_to_robot": {"hand": {"unsigned_axis_angles_deg": {"approach": 12}}},
    }
    monkeypatch.setattr(regions, "summary", lambda *a: raw)
    result = regions.context({})[0]
    assert "major_axis_world_unsigned" not in result
    assert "surface_normal_toward_camera_world" not in result
    assert "unsigned_axis_angles_deg" not in result["relative_to_robot"]["hand"]
    assert "major_axis_world_unsigned" in raw
