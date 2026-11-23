<p align="center">
  <img src="assets/logo.png" width="250" alt="Robo-Harness K1 logo">
</p>

<h1 align="center">Robo-Harness K1</h1>

<p align="center">
  <strong>Harnessing Robot-Use Agents via Perception Augmentation</strong>
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2609.29389">Paper (arXiv)</a> ·
  <a href="https://robo-harness.github.io/k1/">Project page</a> ·
  <a href="#results">Results</a> ·
  <a href="#how-it-works">How it works</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="docs/training.md">Fine-tuning</a> ·
  <a href="docs/usage_zh.md">中文</a> ·
  <a href="LICENSE">MIT License</a>
</p>

---

**Robo-Harness K1** turns a vision-language model into a robot-use agent: the model observes the scene, requests measurements, chooses an action, and checks what actually happened. Rather than asking the model to infer metric geometry from RGB alone, K1 exposes calibrated perception and robot control through a common tool interface.

This release supports **LIBERO-Pro** and **RoboSuite**, with optional interfaces for distilling successful tool-use trajectories into a smaller VLM. It contains code, configuration templates, documentation, selected paper illustrations, a real-robot demonstration GIF, and synthetic tests. Datasets, model weights, raw episode logs, credentials, and third-party simulator source code are supplied separately.

## News and updates

- **2026-11-23:** Updated Figure 1 to the **60-case paired comparison**, including Gemini RGB-only; added real-robot results and a demonstration GIF, RoboTwin and RoboCasa transfer results, and mixed-environment post-training results.
- **2026-11-09 · v1.1:** Updated the shared harness for cross-environment and cross-embodiment use, including single- and dual-arm control, more reliable perception references, gripper monitoring, physical-state image history, and bounded output-truncation recovery. Environment integration work now covers **LIBERO-Pro, RoboSuite, RoboCasa, RoboTwin, RoboDojo, and RoboLab**. LIBERO-Pro and RoboSuite adapters are included in this repository; the other four adapter integrations are pending public release. See the [upgrade guide](docs/releases/v1.1.md).
- **v1.0 archive:** The previous main branch is preserved at [v1.0](https://github.com/Robo-Harness/k1/tree/v1.0). Use that snapshot for the original public implementation.

## Results

Figures and results from the updated paper. Simulation accuracy uses native task criteria; real-robot outcomes use operator confirmation and image review. See [evaluation details](docs/results.md) for the settings.

### LIBERO-Pro

<p align="center">
  <img src="assets/frontier-results.png" width="1000" alt="LIBERO-Pro on 60 matched cases: GPT-6 Astra with K1 93.3%, Gemini 3.7 Flash with K1 71.7%, RGB-only Astra 63.3%, and RGB-only Gemini 16.7%; hatched bars show published references">
</p>

All four Gemini/Astra configurations use the same **60 task–variant–state combinations**: Spatial, Object, and Goal, ten tasks each, with `swap` and `task` variants at state 0.

| Model | RGB-only | K1 | Gain |
|:---|---:|---:|---:|
| Gemini 3.7 Flash | 10/60 · 16.7% | **43/60 · 71.7%** | **+55.0 pp** |
| GPT-6 Astra | 38/60 · 63.3% | **56/60 · 93.3%** | **+30.0 pp** |

Hatched bars are published references with their own evaluation protocols. Gemini + K1 achieves **139/180 (77.2%)** on the full three-state sweep, which includes these 60 cases. This is a comparison of complete agent interfaces; K1 adds perception, memory, motion tools, and completion feedback.

### RoboSuite

<p align="center">
  <img src="assets/robosuite-transfer.png" width="1000" alt="RoboSuite nut assembly with external and wrist camera observations across approach, regrasp, lifting, and alignment">
</p>

Gemini + K1 transfers without target-environment fine-tuning: **90.0%** on Panda, **88.8%** on UR5e, and **86.2%** on IIWA across the shared tasks. All arms use PandaGripper. [Transfer settings →](docs/results.md#robosuite-transfer)

### RoboTwin and RoboCasa

Gemini + K1 also transfers to **Aloha AgileX dual arms in RoboTwin** and the **PandaOmron mobile manipulator in RoboCasa**, without target-environment model fine-tuning.

| Environment | Evaluation | Success |
|:---|:---|---:|
| RoboTwin Easy | 10 tasks, task-macro average | **32.0%** |
| RoboTwin Hard | Same 10 tasks with randomized conditions | **28.0%** |
| RoboCasa | 24 kitchen tasks × 10 initial states | **53/240 · 22.1%** |

RoboTwin tests coordinated manipulation under changes in lighting, textures, clutter, and table height. RoboCasa extends the interface to base motion and torso height. [Task coverage and results →](docs/results.md#robotwin-and-robocasa-transfer)

### Real-World Robot Experiments

A physical **Piper arm** uses two external RGB-D cameras to perform can placement, bowl placement, and two-object sorting. Each method has three concluded trials per task.

<p align="center">
  <a href="https://robo-harness.github.io/k1/#real-robot"><img src="assets/real-robot-can-into-bowl.gif" width="960" alt="Successful physical Piper trial placing a cola can into a red bowl, with task query, current camera views, frame numbers, and tool calls"></a>
</p>

<p align="center"><em>Can into bowl: all 33 recorded decisions, with current-view overlays and tool results. Decision snapshots; API waiting time omitted. <a href="https://robo-harness.github.io/k1/#real-robot">Watch the MP4</a>.</em></p>

| Task | Clean baseline | K1 |
|:---|---:|---:|
| Can into bowl | 1/3 · 33.3% | **2/3 · 66.7%** |
| Bowl onto plate | 0/3 · 0.0% | **3/3 · 100.0%** |
| Two-object sorting | 0/3 · 0.0% | **3/3 · 100.0%** |
| **Overall** | **1/9 · 11.1%** | **8/9 · 88.9%** |

Both methods request Gemini 3.7 Flash and share the Piper control adapter. Outcomes are manually assessed. [Scoring and execution settings →](docs/results.md#real-world-robot-experiments)

### Agentic post-training

<p align="center">
  <img src="assets/learning-curves.png" width="1000" alt="Learning curves comparing Qwen with K1, RGB-only Qwen, OpenVLA, π0.5, and Qwen VLA on in-domain, new-state, and held-out-condition evaluations">
</p>

From **107 teacher episodes**, Qwen3.5-9B + K1 reaches **51.2% / 44.2% / 13.9%** on splits A / B / C at epoch 5; the untrained model scores **0%** on all three. B tests new initial states and C tests held-out task conditions. [Split definitions →](docs/results.md#learning-from-tool-use-trajectories)

Continued training on **316 trajectories across four environments** improves LIBERO-Pro B from **42.9% to 52.4%**, C from **11.1% to 27.8%**, and RoboSuite from **28.3% to 50.0%** in a separate before/after evaluation. [Mixed-environment post-training →](docs/results.md#mixed-environment-post-training)

## How it works

<p align="center">
  <img src="assets/perception-tools.png" width="1000" alt="K1 perception tools: region grounding, depth and local geometry, persistent anchor tracking, and projected grasp candidates">
</p>

<p align="center"><em>Tool outputs stay grounded in the existing camera views: labeled regions, metric geometry, tracked anchors, and grasp-pose schematics.</em></p>

**Observe → measure → act → verify.** The model remains responsible for selecting targets and deciding what to do next. Tools provide evidence and execution primitives—not task-specific solution scripts.

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

The algorithm name is **Robo-Harness K1**. The Python module is `robo_harness`, the distribution is `robo-harness-k1`, and the command is `robo-harness`.

## Source validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
ruff check src tests
robo-harness-audit .
```

The audit flags generated files and metadata as well as likely secrets, private paths and unexpected binary artifacts. Only the explicitly approved, checksum-pinned documentation images and demonstration GIF are allowed as binary assets. Run it on a clean source export without `.git` or caches. It is a heuristic check, not a guarantee that arbitrary generated logs are safe to publish.

Unit tests use synthetic observations and mocked API responses. Passing them does not establish benchmark success rates, trainability of every model release or compatibility with arbitrary simulator versions. Benchmark results require explicit end-to-end evaluation with fixed tasks, initial states, budgets and native success checks.

## License

Project code is released under the [MIT License](LICENSE). External libraries, simulators, task definitions, datasets and weights are not relicensed by this repository.
