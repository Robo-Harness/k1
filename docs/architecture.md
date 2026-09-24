# Architecture and tool semantics

## Decision loop

An environment adapter supplies calibrated RGB-D observations and robot proprioception. The runtime constructs the prompt from the original task, current measurements, recent tool exchanges, previous camera images and persistent task progress. The model selects one tool. The harness executes it, records actual outcomes, refreshes observations and repeats until native success, an accepted completion request, or a budget limit.

Scene-object ground-truth poses, instance masks and hidden success geometry are not exported as actor observations. Evaluator code may read native success and save simulator state for reproducibility. Completion feedback can report whether a requested termination actually satisfied the native task; it does not reveal the missing target pose.

## Components

| Component | Responsibility |
|---|---|
| `runtime.py` | Tool schemas, observation prompts, dispatch, model loop and archives |
| `contracts.py` | Environment-facing sensor/control contract |
| `libero_adapter.py` | LIBERO-Pro reset, calibrated camera conversion, native OSC execution |
| `robosuite_adapter.py` | RoboSuite task-wrapper binding, robot geometry and multi-arm execution |
| `perception.py`, `regions.py`, `tracked_points.py` | Pixel/depth measurements, visible geometry and correspondence memory |
| `motion.py`, `pose_targets.py`, `position_reach.py` | Coordinate conversion, bounded steps, pose targets and absolute transit |
| `grasp_geometry.py`, `grasp_backends.py`, `grasp_presentation.py` | Geometric/optional learned hypotheses and actor-visible filtering |
| `episode_history.py`, `task_progress.py` | Full evidence archive, retrieval and model-authored progress |
| `transport.py` | Explicit endpoint/proxy configuration |
| `training/` | Exact-request conversion and assistant-target LoRA training |

## Observation and perception

Camera pixels use `u` left-to-right and `v` top-to-bottom in the displayed original resolution. Depth is metric camera depth and is unprojected using camera intrinsics and extrinsics. TCP positions are world-frame meters; quaternions use xyzw. Optical and robot frames must not be interchanged.

Depth sampling and local plane/axis fitting describe the visible surface only. SAM3 can locate regions from text or refine a supplied box. Region IDs preserve identity records; tracking refreshes observed geometry, but visibility and correspondence uncertainty still need to be checked. A segmented surface center is not an object center, a grasp point or a mechanical joint axis.

LK is a lightweight tracker option. TAPNext++ is an optional learned streaming point tracker. The system compares world-coordinate estimates where reliable depth is available; image displacement alone is not treated as proof of physical object motion. Occlusion or a wrong depth layer can invalidate evidence. No extra camera viewpoints are synthesized as new sensor measurements.

## Motion tools

| Tool | Meaning |
|---|---|
| `move_relative` | Translate by a requested delta, optionally with bounded rotation; each requested translation component is limited by `delta_axis` |
| `move_toward` | Attempt a complete absolute position target within one model call, using native feedback-controlled steps and a separate execution budget |
| `move_to_pose` | Register/update an absolute pose intention and execute a bounded incremental pose step |
| `rotate_toward` | Rotate incrementally toward a quaternion while holding the position goal |
| `set_gripper` | Open or close the gripper while maintaining the TCP goal, then observe settling |
| `move_effectors` | RoboSuite multi-arm capability: advance multiple end-effector targets in the same native steps |

`delta_axis=0.03` means each translation component may be at most 3 cm; the three-dimensional vector norm can exceed 3 cm. Short chunks increase opportunities for visual feedback, but model/API latency means they are not a high-frequency real-time controller. The simulator's native controller closes the faster control loop.

`move_toward` is not limited by `delta_axis`. Its defaults are at most 120 native steps, 24 non-progress steps and 3 mm position tolerance. It does not teleport or guarantee reachability: inspect `target_reached`, achieved displacement and remaining error. Pose and rotation primitives also report residuals. This package uses native operational-space control; it does not expose a general collision-free IK planner.

Gripper opening alone cannot prove attachment. A small gap can mean an empty grasp or a thin object; a wide gap can mean obstruction. Evaluate object motion and current views. Advisory feedback reports evidence; it is not an alternative task policy. Numeric validation, frame freshness and explicit execution boundaries remain necessary.

## Grasp candidates

The geometric backend is intended for visible regions with a reliable elongated axis, not arbitrary-object grasp optimality. It constructs approach/orientation hypotheses, aligns the calibrated finger-pad midpoint to a visible reference, and evaluates sparse probes against existing RGB-D views. Visibility scores select and order candidates internally. The model sees neutral pose IDs, pose geometry and original-view schematics, not scalar scores or their ranking rationale. Neutral IDs do not make backend selection statistically unbiased.

An optional local GraspGen service can supply learned poses from a visible metric point cloud. Its frame conventions and nominal contact offset require explicit calibration for the provider and gripper. The example configuration is a schema illustration, not a calibrated grasp profile. Sparse geometry is not full-arm collision checking, and neither backend certifies attachment or a safe path.

## Memory and progress

- `history_rounds=N` keeps N complete recent text tool exchanges; default 8.
- `history_image_rounds=K` includes K previous decision image sets; default 1, plus current images.
- Full completed evidence is archived regardless of the active prompt window. `search_history` and `read_history` provide retrieval, pagination and optional archived images.
- `update_task_progress` lets the model retain milestones, evidence references, uncertainties and invalidated claims. The original task remains separately visible.
- History images have frame/call labels and are not silently treated as current sensor evidence. Progress is model-authored, not a native success certificate.

`decision_note` is a brief observable decision summary, conventionally wrapped in `<THINK>...</THINK>`. It is not privileged access to hidden model reasoning. Opaque provider signature fields are passed through for API continuity but are excluded from fine-tuning labels.

## Evaluation artifacts

Every episode uses a fresh output directory. Native actions, model requests/responses, image inputs, measurements, history and completion status are saved locally. `model_view.mp4` visualizes model inputs; its duration is not native simulation duration or wall-clock latency. No benchmark scores or private runs are bundled here.
