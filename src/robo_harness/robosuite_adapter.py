"""RATs task definitions with a sensor-only, embodiment-aware adapter.

Reuses original scene construction/success checks. Explicit adaptation seams:
headless render, extra native wrist view, robot model, controller and RNG seed.
Does NOT export wrapper object poses, instance masks or reward shaping to actor.
"""

import copy
import importlib
import types
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from .robosuite_tasks import TASKS, sha
from robo_harness.perception import execution_report, finite_vector
from robo_harness.motion_limits import validate_transport_delta
from robo_harness.self_geometry import RobotSelfGeometry


def controller_config(joint=False):
    import json
    import robosuite

    cfg = json.loads(
        (
            Path(robosuite.__file__).parent / "controllers/config/default/composite/basic.json"
        ).read_text()
    )
    arm = cfg["body_parts"]["arms"]["right"]
    if joint:
        arm = {
            "type": "JOINT_POSITION",
            "input_type": "absolute",
            "input_max": 10,
            "input_min": -10,
            "output_max": 10,
            "output_min": -10,
            "kp": 150,
            "damping_ratio": 1,
            "interpolation": None,
            "gripper": {"type": "GRIP"},
        }
    else:
        arm["input_ref_frame"] = "world"
    return {"type": "BASIC", "body_parts": {"right": arm}}


class SuiteProxy:
    """Proxy only the suite reference in a private task module, not global robosuite."""

    def __init__(self, real, override):
        self.real, self.override = real, override

    def __getattr__(self, name):
        value = getattr(self.real, name)
        if name == "make" or (
            isinstance(value, type)
            and name
            in ["Lift", "Stack", "NutAssemblySquare", "Wipe", "TwoArmHandover", "TwoArmLift"]
        ):

            def construct(*args, **kwargs):
                return value(*args, **self.override(kwargs))

            return construct
        if isinstance(value, types.ModuleType):
            return SuiteProxy(value, self.override)
        return value


class RobosuiteAdapter:
    def __init__(self, task="cube_lifting", robot="Panda", seed=271828, joint=False, horizon=None):
        self.task_name, self.robot_name, self.seed, self.joint = task, robot, seed, joint
        module_name, class_name, self.instruction, budget = TASKS[task]
        self.horizon = horizon or budget
        self.frame = 0
        self.delta_axis = 0.03
        self.on_step = None
        self.last_feedback = {}
        self._self_geometry = {}
        self.camera_names = []
        self.narms = 2 if task.startswith("two_arm") else 1
        if self.narms == 2 and robot != "Panda":
            raise ValueError("Only approved Panda dual-arm setup")
        self.arm_names = ["arm"] if self.narms == 1 else ["arm0", "arm1"]
        self.grippers = {a: 1.0 for a in self.arm_names}
        self._cached = None
        original = importlib.import_module("capx.envs.simulators." + module_name)
        private = types.ModuleType("transfer_" + module_name)
        private.__file__ = original.__file__
        # Original nut wrapper precomputes Panda names; use declared robot joints
        # instead. These addresses are unused by our execution, but must be valid.
        source = Path(original.__file__).read_text()
        source = source.replace(
            'joint_names = [f"robot0_joint{i}" for i in range(1, 8)]',
            "joint_names = self.robosuite_env.robots[0].robot_model.joints",
        )
        if task == "nut_assembly":
            # Defer the wrapper's constructor-time Panda joint-control reset.
            # Our reset applies the same Panda starting posture with the chosen
            # controller, without sending a 7-joint action into Cartesian OSC.
            assert source.count("        self.reset()") == 1
            source = source.replace(
                "        self.reset()", "        pass  # reset is owned by the adapter"
            )
        exec(compile(source, original.__file__, "exec"), private.__dict__)
        self.source_sha256 = sha(original.__file__)

        def override(kwargs):
            kwargs = dict(kwargs)
            kwargs["robots"] = [robot] * self.narms
            cams = list(kwargs.get("camera_names", []))
            self.camera_names = cams + [f"robot{i}_eye_in_hand" for i in range(self.narms)]
            kwargs.update(
                has_renderer=False,
                has_offscreen_renderer=True,
                use_camera_obs=True,
                camera_names=self.camera_names,
                camera_depths=True,
                camera_segmentations=None,
                camera_heights=384,
                camera_widths=384,
                control_freq=20,
                ignore_done=True,
                hard_reset=False,
                seed=seed,
                initialization_noise=None,
            )
            if task != "spill_wipe":
                kwargs["gripper_types"] = ["PandaGripper"] * self.narms
            cfg = controller_config(joint)
            kwargs["controller_configs"] = [copy.deepcopy(cfg) for _ in range(self.narms)]
            return kwargs

        import robosuite

        private.suite = SuiteProxy(robosuite, override)
        private.load_composite_controller_config = lambda **kw: controller_config(joint)
        cls = getattr(private, class_name)
        self.wrapper = cls(seed=seed, privileged=True, enable_render=True, max_steps=self.horizon)
        self.env = self.wrapper.robosuite_env
        self.cameras = self.camera_names
        self.reset()

    def _controller(self, robot):
        return robot.part_controllers["right"]

    def _gripper(self, robot):
        return robot.gripper["right"]

    def _site(self, robot):
        return robot.eef_site_id["right"]

    def reset(self, initial_state_id=None):
        # Bind the actual native RNG, not only the wrapper's unused _rng.
        self.env.rng = np.random.default_rng(self.seed)
        sampler = getattr(self.env, "placement_initializer", None)
        if sampler is not None:
            sampler.rng = self.env.rng
            for sub in getattr(sampler, "samplers", {}).values():
                sub.rng = self.env.rng
        self.raw = self.env.reset()
        sim = self.env.sim
        # Preserve original single-arm Panda reset orientation; never apply it
        # to joint index 6 of another embodiment (which could be a finger).
        if (
            self.narms == 1
            and self.robot_name == "Panda"
            and self.task_name in ["cube_lifting", "cube_stack", "cube_restack"]
        ):
            sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes[6]] -= np.pi
        if self.robot_name == "Panda" and self.task_name == "nut_assembly":
            sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes] = [
                0,
                -1.585,
                0,
                -2.645,
                0,
                1,
                0.785,
            ]
        if self.narms == 2:
            cid = sim.model.camera_name2id("agentview")
            sim.model.cam_pos[cid] = [1.5, 0, 2.5]
            sim.model.cam_quat[cid] = [0.653, 0.271, 0.271, 0.653]
        sim.forward()
        for robot in self.env.robots:
            ctrl = self._controller(robot)
            ctrl.update(force=True)
            ctrl.reset_goal()
        self.frame = 0
        self.last_feedback = {}
        self.grippers = {a: 1.0 for a in self.arm_names}
        # Physics settling only, without exposing privileged task state.
        for _ in range(10):
            commands = []
            for robot in self.env.robots:
                part = (
                    np.asarray(sim.data.qpos[robot._ref_joint_pos_indexes])
                    if self.joint
                    else np.zeros(6)
                )
                if self._gripper(robot).dof:
                    part = np.r_[part, -1.0]
                commands.extend(part)
            self.raw, *_ = self.env.step(np.asarray(commands))
        sim.forward()
        self.env.timestep = 0
        self._cached = None
        return self.observe()

    def restore(self, state):
        self.env.sim.set_state_from_flattened(state)
        self.env.sim.forward()
        for robot in self.env.robots:
            ctrl = self._controller(robot)
            ctrl.update(force=True)
            ctrl.reset_goal()
        self.env.timestep = 0
        self.frame = 0
        self._cached = None
        return self.observe()

    def observe(self):
        if self._cached is not None:
            return self._cached
        from robosuite.utils.camera_utils import (
            get_camera_intrinsic_matrix,
            get_camera_extrinsic_matrix,
            get_real_depth_map,
        )

        sim = self.env.sim
        self.raw = self.env._get_observations(force_update=True)
        vision = {}
        for name in self.cameras:
            rgb = np.asarray(self.raw[name + "_image"])[::-1].copy()
            depth = (
                get_real_depth_map(sim, np.asarray(self.raw[name + "_depth"]))[::-1]
                .squeeze()
                .copy()
            )
            vision[name] = {
                "color": rgb,
                "depth": depth,
                "intrinsic_matrix": get_camera_intrinsic_matrix(sim, name, *rgb.shape[:2]),
                "extrinsic_matrix": get_camera_extrinsic_matrix(sim, name),
            }
        arms = {}
        self_parts = {}
        for name, robot in zip(self.arm_names, self.env.robots):
            site = self._site(robot)
            tcp = np.asarray(sim.data.site_xpos[site])
            rot = np.asarray(sim.data.site_xmat[site]).reshape(3, 3)
            gripper = self._gripper(robot)
            body_ids = {sim.model.body_name2id(n) for n in gripper.bodies}
            gids = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] in body_ids]
            if name not in self._self_geometry:
                try:
                    self._self_geometry[name] = RobotSelfGeometry(sim.model, gids)
                except ValueError:
                    self._self_geometry[name] = None
            if self._self_geometry[name]:
                self_parts[name] = self._self_geometry[name].snapshot(sim.data)
            probes = []
            # All robot-declared gripper geometry, no hard-coded Panda palm box.
            for gid, equations, corners in (
                self._self_geometry[name].parts if self._self_geometry[name] else []
            ):
                center = np.asarray(sim.data.geom_xpos[gid])
                axes = np.asarray(sim.data.geom_xmat[gid]).reshape(3, 3)
                probes.extend((corners @ axes.T + center - tcp) @ rot)
            pads = []
            for key in ["left_fingerpad", "right_fingerpad"]:
                for geom in gripper.important_geoms.get(key, [])[:1]:
                    pads.append(
                        (np.asarray(sim.data.geom_xpos[sim.model.geom_name2id(geom)]) - tcp) @ rot
                    )
            geometry = {
                "description": "Robot-calibrated grip-site TCP and orientation; sparse gripper geometry only.",
                "sweep_probes_local_m": np.asarray(probes).tolist(),
                "self_geometry_available": bool(self._self_geometry[name]),
            }
            opening = 0.0
            limits = [0.0, 0.0]
            if len(pads) == 2:
                closing = pads[1] - pads[0]
                closing /= max(np.linalg.norm(closing), 1e-12)
                geometry.update(
                    axes_local={
                        "approach": [0, 0, 1],
                        "closing": closing.tolist(),
                        "lateral": np.cross(closing, [0, 0, 1]).tolist(),
                    },
                    finger_pad_centers_local_m=np.asarray(pads).tolist(),
                )
                ranges = np.array(
                    [sim.model.jnt_range[sim.model.joint_name2id(j)] for j in gripper.joints]
                )
                opening = float(sum(abs(sim.data.get_joint_qpos(j)) for j in gripper.joints))
                limits = [0.0, float(np.max(np.abs(ranges), axis=1).sum())]
            arms[name] = {
                "xyz_world_m": tcp.tolist(),
                "quaternion_xyzw": Rotation.from_matrix(rot).as_quat().tolist(),
                "axes_world": rot.tolist(),
                "gripper_opening_m": opening,
                "gripper_limits_m": limits,
                "gripper_command_open": self.grippers[name],
                "geometry": geometry,
                "joint_positions_rad": np.asarray(
                    sim.data.qpos[robot._ref_joint_pos_indexes]
                ).tolist(),
                "gripper_actuated": bool(gripper.dof),
            }
        for cam in vision.values():
            cam["_robot_self_parts"] = self_parts
        self._cached = {
            "frame_id": self.frame,
            "instruction": self.instruction,
            "vision": vision,
            "arms": arms,
            "feedback": self.last_feedback,
            "capabilities": {
                "metric_depth": True,
                "arms": self.arm_names,
                "control_hz": 20,
                "world_axes": "MuJoCo world XYZ, Z up",
                "rotation": "world-axis rotation vector degrees",
                "simultaneous_effectors": self.narms > 1,
            },
        }
        return self._cached

    def _native(self, command):
        self.raw, *_ = self.env.step(np.asarray(command))
        self.env.sim.forward()
        self.frame += 1
        self._cached = None
        if self.on_step:
            self.on_step(self.observe(), command)

    def move(self, arm, delta, rotation_deg=(0, 0, 0), gripper=None, steps=8):
        state = self.observe()["arms"][arm]
        delta = validate_transport_delta(delta, self.delta_axis)
        dr = finite_vector(rotation_deg, 3)
        if np.linalg.norm(dr) > 15.00001:
            raise ValueError("rotation exceeds 15 degrees")
        target = np.asarray(state["xyz_world_m"]) + delta
        q = (
            Rotation.from_rotvec(np.radians(dr)) * Rotation.from_quat(state["quaternion_xyzw"])
        ).as_quat()
        return self.move_to_position(arm, target, q, gripper, steps)

    def move_to_position(
        self, arm, target_xyz_world_m, target_quaternion_xyzw, gripper=None, steps=1
    ):
        result = self.move_effectors(
            [
                {
                    "arm": arm,
                    "position": target_xyz_world_m,
                    "quaternion_xyzw": target_quaternion_xyzw,
                    "gripper": gripper,
                }
            ],
            steps,
        )
        return result["arms"][arm]

    def move_effectors(self, targets, steps=12):
        if self.joint:
            raise ValueError("Cartesian tools unavailable in joint-policy mode")
        if type(steps) != int or not 1 <= steps <= 120:
            raise ValueError("steps must be 1..120")
        obs = self.observe()
        goals = {}
        starts = {}
        seen = set()
        for arm, state in obs["arms"].items():
            starts[arm] = np.asarray(state["xyz_world_m"])
            goals[arm] = (starts[arm].copy(), Rotation.from_quat(state["quaternion_xyzw"]))
        for target in targets:
            arm = target["arm"]
            if arm not in goals or arm in seen:
                raise ValueError("Unknown or duplicate arm")
            seen.add(arm)
            xyz = finite_vector(target["position"], 3)
            quat = finite_vector(target["quaternion_xyzw"], 4)
            if np.linalg.norm(quat) < 1e-8:
                raise ValueError("Zero quaternion")
            goals[arm] = (xyz, Rotation.from_quat(quat))
            grip = target.get("gripper")
            if grip is not None:
                if not np.isfinite(grip) or not 0 <= grip <= 1:
                    raise ValueError("gripper: 0 closed,1 open")
                self.grippers[arm] = float(grip)
        for _ in range(min(steps, self.horizon - self.frame)):
            sim = self.env.sim
            command = []
            for arm, robot in zip(self.arm_names, self.env.robots):
                site = self._site(robot)
                xyz, q = goals[arm]
                error = xyz - sim.data.site_xpos[site]
                drot = (
                    q * Rotation.from_matrix(sim.data.site_xmat[site].reshape(3, 3)).inv()
                ).as_rotvec()
                ctrl = self._controller(robot)
                scale = (np.asarray(ctrl.output_max) - np.asarray(ctrl.output_min)) / 2
                part = np.clip(np.r_[error, drot] / scale, -1, 1)
                if self._gripper(robot).dof:
                    part = np.r_[part, 1 - 2 * self.grippers[arm]]
                command.extend(part)
            self._native(command)
            if self.success():
                break
        now = self.observe()
        reports = {}
        for arm in seen:
            xyz, q = goals[arm]
            state = now["arms"][arm]
            report = execution_report(starts[arm], xyz, state["xyz_world_m"])
            report["rotation_residual_deg"] = float(
                np.degrees((q * Rotation.from_quat(state["quaternion_xyzw"]).inv()).magnitude())
            )
            reports[arm] = report
        self.last_feedback = {"arms": reports, "simultaneous": True, "native_frame": self.frame}
        self._cached = None
        return self.last_feedback

    def joint_action(self, target, gripper):
        if not self.joint or self.narms != 1:
            raise ValueError("Requires single-arm joint policy")
        robot = self.env.robots[0]
        target = finite_vector(target, len(robot._ref_joint_pos_indexes))
        ranges = np.asarray(self.env.sim.model.jnt_range)[robot._ref_joint_indexes]
        clipped = np.clip(target, ranges[:, 0], ranges[:, 1])
        command = clipped
        if self._gripper(robot).dof:
            command = np.r_[command, 2 * float(np.clip(gripper, 0, 1)) - 1]
        self.grippers["arm"] = 1 - float(np.clip(gripper, 0, 1))
        self._native(command)

    def success(self):
        return bool(self.wrapper.task_completed())

    def close(self):
        self.env.close()
