"""LIBERO/MuJoCo adapter. Only calibrated RGB-D and robot proprioception leave observe()."""

import os
import sys
import types
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from .perception import execution_report, finite_vector
from .motion_limits import axis_limit, validate_transport_delta


def environment_instruction(env):
    """The actual BDDL instruction, not its potentially stale filename label."""
    instruction = getattr(env, "language_instruction", None)
    if isinstance(instruction, (list, tuple)) and all(isinstance(x, str) for x in instruction):
        instruction = " ".join(instruction)
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("Environment must expose its actual task language instruction")
    return instruction.strip()


def load_libero():
    root = Path(os.environ.get("LIBERO_ROOT", "./external/LIBERO-PRO"))
    if not (root / "libero/libero").is_dir():
        raise ValueError("Set LIBERO_ROOT to a local LIBERO-Pro checkout")
    config = os.environ.get("LIBERO_CONFIG_PATH")
    if not config or not (Path(config) / "config.yaml").is_file():
        raise ValueError("Prepare LIBERO_CONFIG_PATH using robo-harness configure-libero")
    if "libero.libero" not in sys.modules:
        package = types.ModuleType("libero")
        package.__path__ = [str(root / "libero")]
        sys.modules["libero"] = package
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    return benchmark, OffScreenRenderEnv


class LiberoAdapter:
    def __init__(
        self,
        suite="libero_goal",
        task_id=0,
        resolution=384,
        horizon=600,
        instruction_source="benchmark",
        delta_axis=0.03,
    ):
        self.delta_axis = axis_limit(delta_axis)
        if instruction_source not in ("benchmark", "environment"):
            raise ValueError("instruction_source must be benchmark or environment")
        self.instruction_source = instruction_source
        benchmark, env_type = load_libero()
        self.benchmark = benchmark.get_benchmark_dict()[suite]()
        self.task = self.benchmark.get_task(task_id)
        self.task_id = task_id
        self.suite = suite
        self.resolution = resolution
        self.horizon = horizon
        self.cameras = ["agentview", "robot0_eye_in_hand"]
        self.env = env_type(
            bddl_file_name=self.benchmark.get_task_bddl_file_path(task_id),
            camera_heights=resolution,
            camera_widths=resolution,
            camera_depths=True,
            camera_names=self.cameras,
            horizon=horizon + 20,
            ignore_done=True,
            control_freq=20,
        )
        self.frame = 0
        self.gripper = 1.0
        self.on_step = None
        self.last_feedback = {}
        self._self_geometry = None

    def reset(self, initial_state_id=0):
        import torch
        from libero.libero import get_libero_path

        path = (
            Path(get_libero_path("init_states"))
            / self.task.problem_folder
            / self.task.init_states_file
        )
        # Official local benchmark initialization archive, not a remotely supplied checkpoint.
        states = torch.load(path, weights_only=False, map_location="cpu")
        if not 0 <= initial_state_id < len(states):
            raise ValueError("Initial state ID outside benchmark")
        self.env.seed(initial_state_id)
        self.env.reset()
        self.raw = self.env.set_init_state(states[initial_state_id])
        # MuJoCo state restoration does not invalidate OSC's cached robot state.
        # Refresh before the first command so zero-delta settling holds this pose.
        for robot in self.env.robots:
            robot.controller.update(force=True)
            robot.controller.reset_goal()
        for _ in range(10):
            self.raw, *_ = self.env.step([0, 0, 0, 0, 0, 0, -1])
        self.frame = 0
        self.gripper = 1.0
        self.last_feedback = {}
        return self.observe()

    def observe(self):
        from robosuite.utils.camera_utils import (
            get_camera_extrinsic_matrix,
            get_camera_intrinsic_matrix,
            get_real_depth_map,
        )

        vision = {}
        for name in self.cameras:
            rgb = np.asarray(self.raw[name + "_image"])[::-1].copy()
            depth = (
                get_real_depth_map(self.env.sim, np.asarray(self.raw[name + "_depth"]))[::-1]
                .squeeze()
                .copy()
            )
            vision[name] = {
                "color": rgb,
                "depth": depth,
                "intrinsic_matrix": get_camera_intrinsic_matrix(
                    self.env.sim, name, self.resolution, self.resolution
                ),
                "extrinsic_matrix": get_camera_extrinsic_matrix(self.env.sim, name),
            }
        q = np.asarray(self.raw["robot0_eef_quat"])
        r = Rotation.from_quat(q).as_matrix()
        tcp = np.asarray(self.raw["robot0_eef_pos"])
        # Robot-only kinematics, not scene/object poses. The observed quaternion
        # is the robot eef BODY frame; the gripper site has a fixed extra rotation.
        robot = self.env.robots[0]
        sim = self.env.sim
        if self._self_geometry is None:
            from .self_geometry import RobotSelfGeometry

            # Only the robot's declared gripper bodies; never scene object IDs.
            body_ids = {sim.model.body_name2id(n) for n in robot.gripper.bodies}
            ids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] in body_ids]
            try:
                if not ids:
                    raise ValueError("Robot gripper geometry is unavailable")
                self._self_geometry = RobotSelfGeometry(sim.model, ids)
            except ValueError as exc:
                import warnings

                warnings.warn(f"Self geometry unavailable; using conservative mask fallback: {exc}")
                self._self_geometry = False
        if self._self_geometry:
            self_parts = {"arm": self._self_geometry.snapshot(sim.data)}
            for camera in vision.values():
                camera["_robot_self_parts"] = self_parts
        pads, probes = [], []
        for key in ("left_fingerpad", "right_fingerpad"):
            name = robot.gripper.important_geoms[key][0]
            gid = sim.model.geom_name2id(name)
            center = np.asarray(sim.data.geom_xpos[gid])
            axes = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3)
            size = np.asarray(sim.model.geom_size[gid])
            pads.append((center - tcp) @ r)
            for x in (-1, 1):
                for y in (-1, 1):
                    for z in (-1, 1):
                        probes.append((center + axes @ (size * [x, y, z]) - tcp) @ r)
        closing = np.asarray(pads[1]) - pads[0]
        closing /= max(np.linalg.norm(closing), 1e-12)
        site_r = np.asarray(sim.data.site_xmat[robot.eef_site_id]).reshape(3, 3)
        approach = r.T @ site_r[:, 2]
        ranges = np.array(
            [sim.model.jnt_range[sim.model.joint_name2id(name)] for name in robot.gripper.joints]
        )
        closed_limit = float(
            sum(0 if lo <= 0 <= hi else min(abs(lo), abs(hi)) for lo, hi in ranges)
        )
        open_limit = float(np.max(np.abs(ranges), axis=1).sum())
        for x in (-0.055, 0.055):
            for y in (-0.035, 0.035):
                for z in (-0.105, -0.045):
                    probes.append(r.T @ site_r @ [x, y, z])
        return {
            "frame_id": self.frame,
            "instruction": environment_instruction(self.env)
            if self.instruction_source == "environment"
            else self.task.language,
            "vision": vision,
            "arms": {
                "arm": {
                    "xyz_world_m": np.asarray(self.raw["robot0_eef_pos"]).tolist(),
                    "quaternion_xyzw": q.tolist(),
                    "axes_world": r.tolist(),
                    "gripper_opening_m": float(np.sum(np.abs(self.raw["robot0_gripper_qpos"]))),
                    "gripper_command_open": self.gripper,
                    "gripper_limits_m": [closed_limit, open_limit],
                    "geometry": {
                        "axes_local": {
                            "approach": approach.tolist(),
                            "closing": closing.tolist(),
                            "lateral": np.cross(closing, approach).tolist(),
                        },
                        "finger_pad_centers_local_m": [p.tolist() for p in pads],
                        "sweep_probes_local_m": [p.tolist() for p in probes],
                        "description": "TCP is grip_site position; orientation is robot eef body. Closing axis sign is arbitrary. Probes cover pad corners and approximate palm, not a full collision mesh or held object.",
                    },
                }
            },
            "feedback": self.last_feedback,
            "capabilities": {
                "metric_depth": True,
                "arms": ["arm"],
                "control_hz": 20,
                "world_axes": "MuJoCo world X,Y,Z; Z up; use calibrated point deltas, do not infer image/world axis correspondence",
                "rotation": "world-axis rotation vector in degrees; default orientation preserved",
            },
        }

    def move(self, arm, delta, rotation_deg=(0, 0, 0), gripper=None, steps=8):
        if arm != "arm":
            raise ValueError("Unknown arm")
        delta = validate_transport_delta(delta, self.delta_axis)
        dr = finite_vector(rotation_deg, 3)
        if np.linalg.norm(dr) > 15.000001:
            raise ValueError("Whole rotation limited to 15deg")
        if not 1 <= steps <= 12:
            raise ValueError("steps must be 1..12")
        if gripper is not None:
            if not np.isfinite(gripper) or not 0 <= gripper <= 1:
                raise ValueError("gripper 0 closed, 1 open")
            self.gripper = float(gripper)
        start = np.asarray(self.raw["robot0_eef_pos"]).copy()
        target = start + delta
        target_r = Rotation.from_rotvec(np.radians(dr)) * Rotation.from_quat(
            self.raw["robot0_eef_quat"]
        )
        return self._servo_absolute(target, target_r, steps)

    def move_to_position(
        self, arm, target_xyz_world_m, target_quaternion_xyzw, gripper=None, steps=1
    ):
        """Absolute goal through OSC/dynamics, not a teleport or an IK-feasibility query."""
        if arm != "arm":
            raise ValueError("Unknown arm")
        target = finite_vector(target_xyz_world_m, 3)
        q = finite_vector(target_quaternion_xyzw, 4)
        if np.linalg.norm(q) < 1e-8:
            raise ValueError("Zero quaternion")
        target_r = Rotation.from_quat(q)
        if type(steps) is not int or not 1 <= steps <= 12:
            raise ValueError("steps must be 1..12")
        if gripper is not None:
            if not np.isfinite(gripper) or not 0 <= gripper <= 1:
                raise ValueError("gripper 0 closed, 1 open")
            self.gripper = float(gripper)
        return self._servo_absolute(target, target_r, steps)

    def _servo_absolute(self, target, target_r, steps):
        start = np.asarray(self.raw["robot0_eef_pos"]).copy()
        controller = self.env.robots[0].controller
        scale = np.asarray(controller.output_max) - np.asarray(controller.output_min)
        scale = scale / 2
        for _ in range(steps):
            if self.frame >= self.horizon:
                break
            error = target - np.asarray(self.raw["robot0_eef_pos"])
            rot_error = (
                target_r * Rotation.from_quat(self.raw["robot0_eef_quat"]).inv()
            ).as_rotvec()
            command = np.r_[np.clip(np.r_[error, rot_error] / scale, -1, 1), 1 - 2 * self.gripper]
            self.raw, *_ = self.env.step(command)
            self.frame += 1
            if self.on_step:
                self.on_step(self.observe(), command)
            if self.success():
                break
        self.last_feedback = execution_report(start, target, self.raw["robot0_eef_pos"])
        self.last_feedback["rotation_residual_deg"] = float(
            np.degrees(
                (target_r * Rotation.from_quat(self.raw["robot0_eef_quat"]).inv()).magnitude()
            )
        )
        return self.last_feedback

    def success(self):
        return bool(self.env.check_success())

    def close(self):
        self.env.close()
