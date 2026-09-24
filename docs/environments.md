# Environment setup

Install this harness in two separate environments. LIBERO-Pro uses a single-controller RoboSuite API, while the RoboSuite transfer adapter uses composite controllers. A single universal dependency lock is intentionally not claimed.

## LIBERO-Pro

External source: https://github.com/Zxy-MLlab/LIBERO-PRO

The adapter was developed against source revision `eafdb809426b13153aa1e4c42d6601844217dfec`. Obtain that checkout and its assets/data under the upstream license. The benchmark source and datasets are not redistributed here.

```bash
git clone https://github.com/Zxy-MLlab/LIBERO-PRO.git external/LIBERO-PRO
git -C external/LIBERO-PRO checkout eafdb809426b13153aa1e4c42d6601844217dfec
```

Use upstream's LIBERO-compatible environment setup; it declares RoboSuite 1.4.0. Its older requirements also pin unrelated model packages. Do not blindly install those pins into the training or RoboSuite environment. In particular, controller APIs must remain LIBERO-compatible. Install this package after preparing that environment and verify imports and a smoke reset before starting evaluation.

Place user-supplied benchmark files in a directory containing `bddl_files/` and `init_files/`. Configure paths without modifying global user settings:

```bash
robo-harness configure-libero --repo external/LIBERO-PRO --data data/libero-pro --output configs/libero.local
export LIBERO_ROOT=./external/LIBERO-PRO
export LIBERO_CONFIG_PATH=./configs/libero.local
export MUJOCO_GL=egl
robo-harness run --config configs/libero-pro.yaml --output outputs/libero-smoke --smoke
```

Supported benchmark names include `libero_spatial_swap`, `libero_spatial_task`, `libero_goal_swap`, `libero_goal_task`, `libero_object_swap`, and `libero_object_task`, subject to installed task files. The configuration selects a task ID and initial-state index; valid ranges come from the installed benchmark, not an assumed fixed number. Task instructions come from the actual environment by default.

State archives are loaded using PyTorch's benchmark format. Only use trusted upstream/local archives; do not load arbitrary untrusted serialized state files.

## RoboSuite

The seven task bindings use the public RoboSuite wrappers in the RATs/CaP-X source tree, not bare task-name aliases. External source: https://github.com/Playful-RATs/RATs

Use revision `1df65a180562e91911214fa1caba7c7ed9407b3d`. The adapter expects its bundled RoboSuite source with composite-controller support (package metadata 1.5.1), and the `capx.envs.simulators` task modules. No third-party source is vendored in this release.

```bash
git clone https://github.com/Playful-RATs/RATs.git external/RATs
git -C external/RATs checkout 1df65a180562e91911214fa1caba7c7ed9407b3d
python -m pip install -e external/RATs/rats/third_party/robosuite
# Prepare CaP-X dependencies according to that checkout's instructions.
# Make its Python task modules importable without rewriting their source:
export PYTHONPATH="$PWD/external/RATs/capx-baseline${PYTHONPATH:+:$PYTHONPATH}"
export MUJOCO_GL=egl
robo-harness run --config configs/robosuite.yaml --output outputs/suite-smoke --smoke
```

The task modules additionally import CaP-X dependencies such as Viser. Installing only bare RoboSuite is insufficient. Follow the pinned source's environment instructions and resolve any missing upstream dependencies before smoke evaluation. This repository does not install unrelated third-party environments on import.

| Configuration task | Robot configurations |
|---|---|
| `cube_lifting`, `cube_restack`, `cube_stack`, `nut_assembly` | Panda, UR5e, IIWA |
| `spill_wipe` | Panda with native wiping tool |
| `two_arm_handover`, `two_arm_lift` | Two Panda arms |

These are supported adapter configurations, not claims that every configuration achieves a particular success rate. For the grasping tasks the adapter explicitly uses PandaGripper on the selected arm model; cross-arm tests are not simultaneous changes of arm and gripper. Wiping keeps its native non-actuated tool. Multi-effector commands synchronize native steps; they do not impose an automatic handover strategy.

The adapter uses wrapper scene construction and native success checks. It adds native wrist cameras, uses 384-pixel observations, a 20 Hz control loop, and explicit reset/controller adaptation. Therefore it must not be advertised as an identical-observation or identical-budget reproduction of another paper simply because task names match. Record all protocol choices in any new comparison.

## Perception backends

- SAM3: install its upstream package and provide `ROBO_HARNESS_SAM3_CHECKPOINT`. The required API includes `sam3.model_builder.build_sam3_image_model` and `Sam3Processor` with interactive/text segmentation.
- TAPNext++: provide `ROBO_HARNESS_TAPNEXT_REPO` and `ROBO_HARNESS_TAPNEXT_CHECKPOINT`. The expected class is `tapnet.tapnextpp.votsp2026.model.TAPNextPP` with `from_checkpoint` and stateful `track_frame` support used by `perception.py`. A different tracker release may require an interface adapter.
- GraspGen is optional. The client expects a loopback ZeroMQ service returning homogeneous grasp transforms and confidences. Install `.[grasp]`; deploy/calibrate the service separately. No model server or weights are implicitly downloaded.

Weights are deliberately not bundled. Check each model's licensing and access conditions. Device selection is explicit; the CLI does not start extra services, choose a proxy automatically or reserve GPUs.

## Reproducibility boundary

The pinned source revisions identify expected upstream interfaces, not a fully locked CUDA/rendering stack. Record Python, MuJoCo, controller, perception and model-serving versions for your runs. Complete native reset/render/action smoke tests in each environment. Unit tests alone do not validate headless rendering or hardware-specific kernels.
