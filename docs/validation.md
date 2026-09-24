# Source validation

The source tree has been checked at the following levels:

| Check | Outcome | Scope |
|---|---|---|
| Unit and interface tests | 188 passed | Synthetic sensor/control inputs, mock API responses, memory, tool dispatch, data conversion and loss/gradient equivalence |
| Static checks | Passed | Python syntax and selected correctness checks |
| Package build and installation | Passed | Source distribution, wheel and command-line entry points |
| Source privacy audit | Passed | Likely credentials, private paths, nonlocal IPv4 addresses, unexpected binaries, symlinks and generated metadata |
| LIBERO-Pro native smoke | Passed | Spatial swap, task 0, state 0: reset, two cameras, one native hold action |
| RoboSuite native smoke | Passed | Panda cube lifting: reset, scene/wrist cameras, one native hold action |

The CPU loss test verifies that suffix-only assistant cross-entropy and its gradients match a full causal-loss calculation on a tiny synthetic model. It is not a full Qwen fine-tuning run. These two tensor tests require optional PyTorch; installations without it skip them.

Simulator smokes used separately configured existing dependencies. They do not establish reproducibility of every third-party installation on a clean machine, all robot configurations, segmentation/tracking checkpoints or full task success. Smoke mode does not call an API or load perception models. No paid-model benchmark or GPU fine-tuning job was launched as part of these checks.

The source audit is heuristic. Generated experiment logs, model inputs and training artifacts require a separate privacy review before release, even when this source tree passes.
