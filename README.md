# robo-harness k1

**robo-harness k1** is a tool-augmented robot-control agent harness for **LIBERO-Pro** and **RoboSuite**. A vision-language model observes calibrated cameras and robot state, selects a perception or motion tool, reads its measured result, and decides again.

The algorithm name is **robo-harness k1**. The Python package is `robo_harness`; the distribution is `robo-harness-k1`; the command is `robo-harness`.

Repository: [Robo-Harness/k1](https://github.com/Robo-Harness/k1).

This repository contains source code, configuration templates, documentation and synthetic tests. It does **not** contain datasets, model weights, evaluation trajectories, videos, credentials, or third-party simulator source code.

## Capabilities

- Calibrated RGB-D measurement, local geometry, segmentation-backed region memory, point tracking, and existing-camera overlays.
- Relative motion, absolute-position servo execution, bounded pose/rotation updates, gripper control, and measured execution feedback.
- Grasp-pose hypotheses with neutral IDs and rendered schematics. Heuristic scores are retained for backend selection but hidden from the model.
- Configurable recent text history (`N=8`) and previous image history (`K=1`), complete local episode archives, historical retrieval and model-maintained task progress.
- Optional simultaneous multi-effector control for supported RoboSuite tasks.
- OpenAI-compatible chat/tool API transport, including preservation of opaque provider signature fields across tool turns.
- Exact-request supervised-data preparation and Qwen3.5 visual-language LoRA fine-tuning interfaces.

Tools provide measurements and execution primitives, not scripted task-solving policies. A grasp hypothesis is not a collision-free path, a grasp certificate or a guarantee of task success.

## Installation

Use **separate Python environments for LIBERO-Pro and RoboSuite**: their robot-controller APIs and simulator dependencies differ. Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest
```

This installs only the harness and tests. Follow [environment setup](docs/environments.md) for simulator dependencies, task definitions and assets. Install SAM3 and TAPNext++ separately if using the provided full-perception configurations. For geometry-only smoke tests, set `region_tools: false`, `tracking_backend: lk`, and `tracking_device: cpu`; that is a reduced configuration, not equivalent to the full system.

FFmpeg is needed for model-view MP4 generation. Model weights and external code remain under their original licenses.

## Run an episode

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

## Fine-tuning

See [training](docs/training.md) for manifest format, data filtering, LoRA training, epoch checkpoints and serving the resulting model through the same harness. All data and base weights must be supplied locally by the user. No training is started by installation or import.

## Documentation

- [Architecture, tools and execution semantics](docs/architecture.md)
- [Environment setup and dependency boundaries](docs/environments.md)
- [Training and model-serving interface](docs/training.md)
- [Security and release checklist](SECURITY.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)
- [Validation scope](docs/validation.md)
- [中文使用说明](docs/usage_zh.md)

## Source validation

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest
ruff check src tests
robo-harness-audit .
```

The audit flags generated files and metadata as well as likely secrets, private paths and unexpected binary artifacts. Run it on a clean source tree. It is a heuristic check, not a guarantee that arbitrary generated logs are safe to publish.

Unit tests use synthetic observations and mocked API responses. Passing them does not establish benchmark success rates, trainability of every model release or compatibility with arbitrary simulator versions. Benchmark results require explicit end-to-end evaluation with fixed tasks, initial states, budgets and native success checks.

## License

Project code is released under the [MIT License](LICENSE). External libraries, simulators, task definitions, datasets and weights are not relicensed by this repository.
