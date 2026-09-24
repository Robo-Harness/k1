import copy

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from robo_harness.motion import (
    MotionController,
    align_rotation,
    bounded_rotation,
    sweep_evidence,
)


class Robot:
    def __init__(self):
        self.xyz = np.array([0.0, 0.0, 0.5])
        self.r = Rotation.identity()
        self.gripper = 0.25
        self.frame = 0
        self.horizon = 600
        self.scale = 1.0
        self.calls = []

    def observe(self):
        return {
            "frame_id": self.frame,
            "vision": {},
            "arms": {
                "hand": {
                    "xyz_world_m": self.xyz.tolist(),
                    "quaternion_xyzw": self.r.as_quat().tolist(),
                    "axes_world": self.r.as_matrix().tolist(),
                    "gripper_command_open": self.gripper,
                    "geometry": {"axes_local": {"approach": [0, 0, 1], "closing": [0, 1, 0]}},
                }
            },
        }

    def move(self, arm, delta, dr, gripper, steps):
        self.calls.append((np.array(delta), np.array(dr), gripper))
        self.xyz += np.asarray(delta) * self.scale
        self.r = Rotation.from_rotvec(np.radians(dr) * self.scale) * self.r
        if gripper is not None:
            self.gripper = gripper
        self.frame += steps
        return {"actual_translation_m": (np.asarray(delta) * self.scale).tolist()}

    def success(self):
        return False

    def move_to_position(self, arm, target, quaternion, gripper, steps=1):
        delta = np.asarray(target) - self.xyz
        dr = np.degrees((Rotation.from_quat(quaternion) * self.r.inv()).as_rotvec())
        return self.move(arm, delta, dr, gripper, steps)


def test_arrival_guard_blocks_release_until_reached_and_allows_explicit_abandonment():
    from robo_harness.runtime import ToolRegistry

    robot = Robot()
    registry = ToolRegistry(robot)
    assert registry.arrival_guard is True

    def call(name, **args):
        return registry.execute(
            name, {"arm": "hand", "frame_id": robot.frame, **args}, robot.observe()
        )

    robot.scale = 0
    call("move_toward", target_xyz_world_m=[0.09, 0, 0.5])
    before = len(robot.calls)
    with pytest.raises(ValueError, match="Release not executed"):
        call("move_relative", delta=[0, 0, 0], frame="world", gripper=1)
    assert len(robot.calls) == before and robot.gripper == 0.25
    robot.scale = 1
    call("move_toward", target_xyz_world_m=[0.09, 0, 0.5])
    call("move_toward", target_xyz_world_m=[0.09, 0, 0.5])
    call("move_relative", delta=[0, 0, 0], frame="world", gripper=1)
    assert robot.gripper == 1
    robot.gripper = 0
    call("move_toward", target_xyz_world_m=[0.18, 0, 0.5])
    call(
        "move_relative",
        delta=[0, 0, 0],
        frame="world",
        gripper=1,
        release_override_reason="Abandon this waypoint to regrasp after observed slip",
    )
    assert robot.gripper == 1 and "hand" not in registry.waypoints


def test_motion_extension_switch_removes_and_rejects_tools():
    from robo_harness.runtime import ToolRegistry

    robot = Robot()
    registry = ToolRegistry(robot, motion_tools=False)
    names = {s["function"]["name"] for s in registry.schemas(robot.observe())}
    assert not names.intersection({"align_axis", "rotate_toward", "retreat"})
    with pytest.raises(ValueError, match="disabled"):
        registry.execute("retreat", {"frame_id": 0, "arm": "hand"}, robot.observe())


def test_orientation_goal_accumulates_to_90_without_position_or_gripper_change():
    robot = Robot()
    motion = MotionController(robot)
    start = robot.xyz.copy()
    result = motion.orient(
        "align_axis",
        {"arm": "hand", "axis": "approach", "direction_world": [1, 0, 0]},
        robot.observe(),
    )
    assert result["remaining_degrees"] == pytest.approx(75)
    goal = result["goal_id"]
    for _ in range(5):
        result = motion.orient("rotate_toward", {"arm": "hand", "goal_id": goal}, robot.observe())
    assert result["goal_reached"]
    np.testing.assert_allclose(robot.r.apply([0, 0, 1]), [1, 0, 0], atol=1e-8)
    np.testing.assert_allclose(robot.xyz, start)
    assert robot.gripper == 0.25
    assert all(np.linalg.norm(call[1]) <= 15.000001 for call in robot.calls)
    assert len(motion.goals) == 1


def test_resumed_orientation_does_not_undo_deliberate_translation():
    robot = Robot()
    motion = MotionController(robot)
    result = motion.orient(
        "align_axis",
        {"arm": "hand", "axis": "approach", "direction_world": [1, 0, 0]},
        robot.observe(),
    )
    motion.move(robot.observe(), "hand", [0.02, 0, 0], [0, 0, 0])
    translated = robot.xyz.copy()
    motion.orient("rotate_toward", {"arm": "hand", "goal_id": result["goal_id"]}, robot.observe())
    np.testing.assert_allclose(robot.xyz, translated)


def test_close_guard_requires_arrival_or_explicit_abandonment():
    from robo_harness.runtime import ToolRegistry

    robot = Robot()
    robot.gripper = 1
    robot.scale = 0
    registry = ToolRegistry(robot)
    registry.execute(
        "move_toward",
        {
            "frame_id": 0,
            "arm": "hand",
            "target_xyz_world_m": [0.03, 0, 0.5],
        },
        robot.observe(),
    )
    frame = robot.frame
    with pytest.raises(ValueError, match="Closing not executed"):
        registry.execute(
            "move_relative",
            {"frame_id": frame, "arm": "hand", "delta": [0, 0, 0], "frame": "world", "gripper": 0},
            robot.observe(),
        )
    assert robot.frame == frame
    registry.execute(
        "move_relative",
        {
            "frame_id": frame,
            "arm": "hand",
            "delta": [0, 0, 0],
            "frame": "world",
            "gripper": 0,
            "close_override_reason": "Abandon old waypoint for intentional close here",
        },
        robot.observe(),
    )
    assert "hand" not in registry.waypoints


def test_set_gripper_waits_for_proprioceptive_settling_without_claiming_attachment():
    from robo_harness.runtime import ToolRegistry

    class GripperRobot(Robot):
        def __init__(self):
            super().__init__()
            self.gripper = 1
            self.gap = 0.08
            self.on_step = None
            self.horizon = 100

        def observe(self):
            obs = super().observe()
            obs["arms"]["hand"].update(gripper_opening_m=self.gap, gripper_limits_m=[0, 0.08])
            return obs

        def success(self):
            return False

        def move(self, arm, delta, dr, gripper, steps):
            self.gripper = gripper
            for _ in range(steps):
                self.gap += 0.2 * (0.08 * gripper - self.gap)
                self.frame += 1
                if self.on_step:
                    self.on_step(self.observe(), [])
            return {"actual_translation_m": [0, 0, 0]}

    robot = GripperRobot()
    registry = ToolRegistry(robot)
    result = registry.execute(
        "set_gripper", {"frame_id": 0, "arm": "hand", "gripper": 0}, robot.observe()
    )
    assert result["native_steps"] == 24
    assert result["gripper"]["settled_last3_native_frames"] is True
    assert result["gripper"]["near_closed_mechanical_limit"] is True
    assert "grasp_success" not in result


def test_alignment_handles_180_degrees_and_nonidentity_frames():
    current = Rotation.from_euler("xyz", [20, 40, 70], degrees=True)
    target_axis = -current.apply([0, 1, 0])
    target = align_rotation(current, [0, 1, 0], target_axis)
    np.testing.assert_allclose(target.apply([0, 1, 0]), target_axis, atol=1e-8)
    delta, remaining = bounded_rotation(current, target)
    assert remaining == pytest.approx(180)
    assert np.linalg.norm(delta) == pytest.approx(15)
    with pytest.raises(ValueError):
        align_rotation(current, [0, 0, 0], target_axis)


def test_remaining_angle_uses_actual_not_commanded_rotation():
    robot = Robot()
    robot.scale = 0.5
    motion = MotionController(robot)
    result = motion.orient(
        "rotate_toward",
        {
            "arm": "hand",
            "target_quaternion_xyzw": Rotation.from_euler("z", 90, degrees=True).as_quat(),
        },
        robot.observe(),
    )
    assert result["remaining_degrees"] == pytest.approx(82.5)
    assert result["actual_rotation_vector_world_deg"][2] == pytest.approx(7.5)
    assert result["rotation_step_error_deg"] == pytest.approx(7.5)


def test_low_progress_blocks_repeated_push_and_retreat_does_not_bounce_forward():
    robot = Robot()
    motion = MotionController(robot)
    for _ in range(2):
        motion.move(robot.observe(), "hand", [0.02, 0, 0], [0, 0, 0])
    robot.scale = 0
    for _ in range(3):
        motion.move(robot.observe(), "hand", [0.02, 0, 0], [0, 0, 0])
    count = len(robot.calls)
    with pytest.raises(ValueError, match="blocked"):
        motion.move(robot.observe(), "hand", [0.02, 0, 0], [0, 0, 0])
    assert len(robot.calls) == count
    # Waiting is allowed, but cannot bypass the blocked-direction guard.
    motion.move(robot.observe(), "hand", [0, 0, 0], [0, 0, 0])
    with pytest.raises(ValueError, match="blocked"):
        motion.move(robot.observe(), "hand", [0.02, 0, 0], [0, 0, 0])
    robot.scale = 1
    for _ in range(4):
        motion.retreat({"arm": "hand", "max_distance_m": 0.01}, robot.observe())
    assert robot.xyz[0] == pytest.approx(0)
    assert robot.gripper == 0.25
    with pytest.raises(ValueError, match="No earlier"):
        motion.retreat({"arm": "hand"}, robot.observe())


def test_sparse_depth_sweep_reports_risk_without_claiming_collision_free():
    robot = Robot()
    obs = robot.observe()
    obs["arms"]["hand"]["xyz_world_m"] = [0, 0, 0.975]
    obs["arms"]["hand"]["geometry"]["sweep_probes_local_m"] = [[0, 0, 0]]
    obs["vision"] = {
        "cam": {
            "color": np.zeros((100, 100, 3), np.uint8),
            "depth": np.ones((100, 100)),
            "intrinsic_matrix": np.array([[100, 0, 50], [0, 100, 50], [0, 0, 1]]),
            "extrinsic_matrix": np.eye(4),
        }
    }
    result = sweep_evidence(obs, "hand", [0, 0, 0.025], [0, 0, 0])
    assert result["status"] == "possible_surface_intersection"
    assert result["collision_free"] is False
    motion = MotionController(robot)
    with pytest.raises(ValueError, match="surface"):
        motion.move(obs, "hand", [0, 0, 0.025], [0, 0, 0], check_sweep=True)
    assert not robot.calls
    hidden = copy.deepcopy(obs)
    hidden["vision"] = {}
    assert sweep_evidence(hidden, "hand", [0, 0, 0.025], [0, 0, 0])["status"] == "unknown"


def test_reject_unknown_axis_and_invalid_quaternion():
    robot = Robot()
    motion = MotionController(robot)
    with pytest.raises(ValueError, match="calibrated"):
        motion.orient(
            "align_axis",
            {"arm": "hand", "axis": "unavailable", "direction_world": [1, 0, 0]},
            robot.observe(),
        )
    with pytest.raises(ValueError, match="Zero"):
        motion.orient(
            "rotate_toward",
            {"arm": "hand", "target_quaternion_xyzw": [0, 0, 0, 0]},
            robot.observe(),
        )


def test_closing_line_sign_does_not_trigger_unnecessary_180_degree_rotation():
    robot = Robot()
    motion = MotionController(robot)
    result = motion.orient(
        "align_axis",
        {"arm": "hand", "axis": "closing", "direction_world": [0, -1, 0]},
        robot.observe(),
    )
    assert result["remaining_degrees"] == pytest.approx(0)
    assert np.linalg.norm(robot.calls[0][1]) == pytest.approx(0)
