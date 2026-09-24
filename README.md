<p align="center">
  <img src="assets/logo.png" width="250" alt="robo-harness k1: a robot riding a horse with perception and measurement tools">
</p>

<h1 align="center">robo-harness k1</h1>

<p align="center">
  <strong>Harnessing Robot-Use Agents via Perception Augmentation</strong><br>
  Give vision-language models tools to measure, remember, and act.
</p>

<p align="center">
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/training.md">Fine-tuning</a> ·
  <a href="docs/usage_zh.md">中文</a> ·
  <a href="LICENSE">MIT License</a>
</p>

---

**robo-harness k1** turns a vision-language model into a robot-use agent: the model observes the scene, requests measurements, chooses an action, and checks what actually happened. Rather than asking the model to infer metric geometry from RGB alone, k1 exposes calibrated perception and robot control through a common tool interface.

This release supports **LIBERO-Pro** and **RoboSuite**, with optional interfaces for distilling successful tool-use trajectories into a smaller VLM. It contains code, configuration templates, documentation, selected paper illustrations, and synthetic tests—not datasets, model weights, episode recordings, credentials, or third-party simulator source code.

## Results

Selected results from the accompanying paper. **Success means the native simulator success criterion**, not the model's declaration of completion. Counts and comparison conditions are included below; see [evaluation details](docs/results.md) for the full tables and limitations.

### Perception-augmented frontier agents

On the **same 18 LIBERO-Pro task–variant–state configurations**:

| Model | Interface | Successful episodes | Success rate |
|:---|:---|---:|---:|
| GPT-6 Astra | RGB-only robot-use interface | 11 / 18 | 61.1% |
| Gemini 3.7 Flash | **k1** | 14 / 18 | 77.8% |
| GPT-6 Astra | **k1** | **16 / 18** | **88.9%** |

For Astra, replacing the RGB-only interface with k1 improves success by **27.8 percentage points** on this small matched subset. This is a comparison of complete interfaces, not an isolated depth-tool ablation.

The larger **Gemini + k1 evaluation succeeds in 139 / 180 episodes (77.2%)**: 30 base tasks across Spatial, Object, and Goal; two perturbations (`swap`, `task`); three initial states. The 14 / 18 Gemini result above is a subset of those 180 episodes, not an additional independent evaluation.

### Transfer across robot arms

**Gemini + k1, without target-environment fine-tuning**, on four shared RoboSuite tasks: cube lifting, restacking, stacking, and nut assembly. Each task has 20 trials per arm.

| Panda | UR5e | IIWA |
|:---:|:---:|:---:|
| **90.0%** · 72 / 80 | **88.8%** · 71 / 80 | **86.2%** · 69 / 80 |

All three arms use a PandaGripper and embodiment-specific adapters. This tests arm transfer, not arbitrary gripper transfer. Panda's broader seven-task evaluation, including wiping and two-arm tasks, reaches **101 / 140 (72.1%)**; [see the per-task breakdown](docs/results.md#robosuite-transfer).

### Learning robot tool use from 107 episodes

Models are trained from the **same pool of 107 successful teacher episodes** and evaluated at **epoch 5**:

| Model / policy | A: Trained task & state<br>43 episodes | B: New state, trained condition<br>43 episodes | C: Held-out conditions<br>36 episodes |
|:---|---:|---:|---:|
| **Qwen3.5-9B + k1** | **51.2%** | **44.2%** | **13.9%** |
| Qwen3.5-9B RGB-only robot-use agent | 7.0% | 4.7% | 0.0% |
| OpenVLA-7B | 20.9% | 30.2% | 0.0% |
| π0.5 | 7.0% | 7.0% | 0.0% |

Group B is the fixed **development set**. Group C holds out 12 task–variant conditions from fine-tuning, with three initial states each; these are not necessarily 12 distinct manipulation skills. The systems share source episodes, but differ in observations, action interfaces, and supervision. These are results under this training recipe, not a general ranking of VLMs and VLAs. [Split definitions and learning curves →](docs/results.md#learning-from-tool-use-trajectories)

## How it works

<p align="center">
  <img src="assets/perception-tools.png" width="1000" alt="Four k1 perception tools: region grounding, depth and local geometry, persistent anchor tracking, and projected grasp candidates">
</p>

<p align="center"><em>Tool outputs stay grounded in the existing camera views: labeled regions, metric geometry, tracked anchors, and grasp-pose schematics.</em></p>

**Observe → measure → act → verify.** The model remains responsible for selecting targets and deciding what to do next. Tools provide evidence and execution primitives—not task-specific solution scripts.

| Component | What the agent gets |
|:---|:---|
| **Ground & measure** | Segmentation-backed regions, calibrated RGB-D measurements, and local geometric estimates. |
| **Track & remember** | Persistent region identities, tracked visual anchors, and overlays on existing cameras. |
| **Inspect grasp hypotheses** | Projected gripper schematics with neutral candidate IDs. Heuristic scores select/order candidates internally but are not shown to the model. |
| **Move & verify** | Relative corrections, absolute-position servo execution, pose/rotation updates, gripper control, and measured execution feedback. |
| **Maintain progress** | Configurable recent text history (`N=8`), previous image history (`K=1`), full episode archives, history retrieval, and model-maintained milestones. |
| **Change the backend** | Environment adapters, optional simultaneous multi-effector control, and image/tool-capable OpenAI-compatible model endpoints. |

A grasp hypothesis is **not** a collision-free path or a guarantee of a successful grasp. See [architecture and execution semantics](docs/architecture.md) for tool behavior, coordinate conventions, and limitations.

## Quick start

### 1. Install the harness

Use **separate Python environments for LIBERO-Pro and RoboSuite**: their robot-controller APIs and simulator dependencies differ. Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

This installs only the harness and tests. Follow [environment setup](docs/environments.md) for simulator dependencies, task definitions and assets. Install SAM3 and TAPNext++ separately if using the provided full-perception configurations. For geometry-only smoke tests, set `region_tools: false`, `tracking_backend: lk`, and `tracking_device: cpu`; that is a reduced configuration, not equivalent to the full system.

FFmpeg is needed for model-view MP4 generation. Model weights and external code remain under their original licenses.

### 2. Connect a model and run

Set your API key in the process environment using your secret manager or a private, ignored file. Never put it in YAML, source code or the command line. `.env.example` documents variable names; the CLI does not automatically load `.env`.

```bash
export ROBO_HARNESS_BASE_URL=https://openrouter.ai/api/v1
export ROBO_HARNESS_MODEL='your-provider/model-id'
# Supply ROBO_HARNESS_API_KEY securely in the environment.
robo-harness run --config configs/libero-pro.yaml --output outputs/libero-example
robo-harness run --config configs/robosuite.yaml --output outputs/suite-example
```

The endpoint and model ID are configurable; no particular provider or model is silently substituted. For a local OpenAI-compatible server, use a loopback URL and supply its expected key. The server must support image inputs and function tools. Remote endpoints require HTTPS. Proxy use is opt-in via `ROBO_HARNESS_PROXY`.

Before using an API, test the simulator separately:

```bash
robo-harness run --config configs/robosuite.yaml --output outputs/suite-smoke --smoke
```

Smoke mode performs one native hold action and does not call a model. Existing episode output directories are not overwritten. Formal runs archive native commands, actual model inputs, requests/responses, history, progress and native success. Generated outputs are private by default and excluded from source sharing.

### 3. Distill tool-use trajectories

The training interface prepares exact-request supervised examples and supports Qwen3.5 visual-language LoRA fine-tuning. See [training](docs/training.md) for manifest format, filtering, epoch checkpoints, and serving the resulting model through the same harness.

All data and base weights must be supplied locally by the user. No training is started by installation or import. The VLA baseline implementations, benchmark datasets, and trained checkpoints shown in the paper are not bundled in this release.

## Documentation

- [Architecture, tools and execution semantics](docs/architecture.md)
- [Paper results, evaluation settings and comparison scope](docs/results.md)
- [Environment setup and dependency boundaries](docs/environments.md)
- [Training and model-serving interface](docs/training.md)
- [Security and release checklist](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Validation scope](docs/validation.md)
- [中文使用说明](docs/usage_zh.md)

The algorithm name is **robo-harness k1**. The Python module is `robo_harness`, the distribution is `robo-harness-k1`, and the command is `robo-harness`.

## Source validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
ruff check src tests
robo-harness-audit .
```

The audit flags generated files and metadata as well as likely secrets, private paths and unexpected binary artifacts. Only the explicitly approved, checksum-pinned README images are allowed as binary assets. Run it on a clean source export without `.git` or caches. It is a heuristic check, not a guarantee that arbitrary generated logs are safe to publish.

Unit tests use synthetic observations and mocked API responses. Passing them does not establish benchmark success rates, trainability of every model release or compatibility with arbitrary simulator versions. Benchmark results require explicit end-to-end evaluation with fixed tasks, initial states, budgets and native success checks.

## License

Project code is released under the [MIT License](LICENSE). External libraries, simulators, task definitions, datasets and weights are not relicensed by this repository.
