# Paper results and evaluation scope

[← README](../README.md)

This page summarizes the LIBERO-Pro and RoboSuite experiments in **Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation**. Values are transcribed from the manuscript's native-success ledger, matched frontier comparison, RoboSuite transfer table, and full checkpoint ledger. They describe the paper's experiments; installing this source release does not rerun or independently reproduce those experiments.

All percentages below measure **native simulator success**. Task completion declared by the language model is not sufficient. Episode recordings, evaluation manifests, training data, and model checkpoints are not included in this repository.

## LIBERO-Pro frontier evaluation

### Complete Gemini evaluation

The evaluation uses Spatial, Object, and Goal, with ten base task indices per category, two variants (`swap` and `task`), and stored initial-state indices 0, 1, and 2:

**3 categories × 10 task indices × 2 variants × 3 states = 180 episodes.**

| Category | Native successes | Success rate |
|:---|---:|---:|
| Spatial | 49 / 60 | 81.7% |
| Object | 52 / 60 | 86.7% |
| Goal | 38 / 60 | 63.3% |
| **Total: Gemini 3.7 Flash + k1** | **139 / 180** | **77.2%** |

The state numbers identify entries in the benchmark's stored initial-state files. They should not be interpreted as arbitrary pseudorandom seeds that necessarily recreate the same scenes. Benchmark conditions overlap with harness development; this sweep is not a wholly held-out test of harness design.

### Matched 18-episode comparison

Task indices 0–2 at state 0 are used in each of the six category–variant suites. All three rows below refer to the same task–variant–state configurations.

| Model | Interface | Native successes | Success rate |
|:---|:---|---:|---:|
| GPT-6 Astra | RGB-only robot-use interface | 11 / 18 | 61.1% |
| Gemini 3.7 Flash | k1 | 14 / 18 | 77.8% |
| GPT-6 Astra | k1 | 16 / 18 | 88.9% |

The Astra difference is **5 / 18 = 27.8 percentage points**. Matching initial states does not make the interfaces identical: k1 adds depth-backed perception tools, memory mechanisms, motion primitives, and completion feedback. This comparison does not isolate the effect of any single tool, equalize inference costs, or establish statistical significance on its own. There is no matched Gemini RGB-only row in this comparison.

Gemini's 18 cases are included in the 180-case sweep. The 77.8% and 77.2% figures are therefore different views of overlapping data, not independent replications.

### Native completion versus post-release completion

The reported scores use the native success criterion at evaluation time. A separate diagnostic holds the arm, opens the gripper for 20 native frames, and requires success over the final ten frames. Gemini's post-release result is **131 / 180 (72.8%)**, compared with 139 / 180 native successes: 11 native successes lose success and three native failures gain success. Both Astra groups retain their native successes. This diagnostic is separate because opening the gripper changes the state and can affect completion in either direction.

## RoboSuite transfer

Gemini 3.7 Flash uses k1 without target-environment model fine-tuning. Environment and embodiment adapters supply the appropriate sensing, coordinate transforms, robot geometry, and controller interface.

Each evaluated task–arm combination has **five reset configurations × four independent rollouts = 20 trials**. The arms all use **PandaGripper**; this is not a test of transfer to arbitrary gripper mechanisms. LIBERO-Pro and RoboSuite both use MuJoCo, so this transfer should not be described as cross-physics-engine generalization.

| Task | Panda | UR5e | IIWA |
|:---|---:|---:|---:|
| Cube lifting | 20 / 20 · 100.0% | 20 / 20 · 100.0% | 20 / 20 · 100.0% |
| Cube restacking | 20 / 20 · 100.0% | 20 / 20 · 100.0% | 20 / 20 · 100.0% |
| Cube stacking | 20 / 20 · 100.0% | 20 / 20 · 100.0% | 20 / 20 · 100.0% |
| Nut assembly | 12 / 20 · 60.0% | 11 / 20 · 55.0% | 9 / 20 · 45.0% |
| **Shared four-task total** | **72 / 80 · 90.0%** | **71 / 80 · 88.8%** | **69 / 80 · 86.2%** |
| Spill wiping | 13 / 20 · 65.0% | — | — |
| Two-arm handover | 1 / 20 · 5.0% | — | — |
| Two-arm lifting | 15 / 20 · 75.0% | — | — |
| **Panda seven-task total** | **101 / 140 · 72.1%** | — | — |

“—” means not evaluated, not a failed episode. Pooling the common four tasks across all three arms gives **212 / 240 (88.3%)**. The 90.0% value is Panda's shared-task result, not the pooled three-arm result.

Published CaP-Agent0 and RATs results discussed in the paper are contextual references, not matched reruns. Differences in trial counts, observations, controllers, and execution budgets prevent treating them as a controlled head-to-head comparison. They are not combined with the local scores above.

## Learning from tool-use trajectories

### Shared source episodes

The student experiments use **107 successful Gemini teacher episodes** after excluding fine-tuning-held-out conditions. Tool-use supervision contains **2,482 accepted targets from 2,564 raw turns**, excluding 12 execution-error turns, 16 call-format-error turns, and 54 rejected completion turns. The action-policy conversion uses the same physical source episodes, with 40,808 frames and 10,241 action windows.

The common episode pool does not imply identical observations, target counts, action spaces, model architectures, or optimization recipes. k1 students learn tool calls; action-policy baselines learn their respective action representations. The study measures performance at this data budget, not a general sample-complexity law.

### Evaluation splits

Each evaluated checkpoint uses the same **122 cases**:

| Split | Cases | Relationship to fine-tuning data |
|:---|---:|:---|
| **A: Trained task and state** | 43 | One initial state actually present in training for each trained task–variant condition. |
| **B: New state, trained condition** | 43 | The same 43 conditions at stored state index 3; the fixed development set. |
| **C: Fine-tuning-held-out conditions** | 36 | 12 held-out task–variant conditions at state indices 40, 41, and 42. |

A and B each contain 15 Spatial, 16 Object, and 12 Goal conditions. C contains four held-out conditions per category, with three states each. A condition includes the variant; matching task indices across `swap` and `task` need not mean matching instructions. Held-out conditions can share objects, scenes, and manipulation skills with training conditions.

### Checkpoint learning curves

Counts below preserve the full denominators. All four listed policies have complete epoch 1, 3, and 5 evaluations.

| Model / policy | Epoch | A · 43 cases | B · 43 cases | C · 36 cases |
|:---|---:|---:|---:|---:|
| Qwen3.5-9B + k1 | 1 | 7 / 43 · 16.3% | 10 / 43 · 23.3% | 3 / 36 · 8.3% |
| Qwen3.5-9B + k1 | 3 | 14 / 43 · 32.6% | **20 / 43 · 46.5%** | **5 / 36 · 13.9%** |
| **Qwen3.5-9B + k1** | **5** | **22 / 43 · 51.2%** | **19 / 43 · 44.2%** | **5 / 36 · 13.9%** |
| Qwen3.5-9B RGB-only robot-use agent | 1 | 0 / 43 · 0.0% | 0 / 43 · 0.0% | 0 / 36 · 0.0% |
| Qwen3.5-9B RGB-only robot-use agent | 3 | 0 / 43 · 0.0% | 0 / 43 · 0.0% | 0 / 36 · 0.0% |
| Qwen3.5-9B RGB-only robot-use agent | 5 | 3 / 43 · 7.0% | 2 / 43 · 4.7% | 0 / 36 · 0.0% |
| OpenVLA-7B | 1 | 5 / 43 · 11.6% | 3 / 43 · 7.0% | 0 / 36 · 0.0% |
| OpenVLA-7B | 3 | 6 / 43 · 14.0% | 4 / 43 · 9.3% | 0 / 36 · 0.0% |
| OpenVLA-7B | 5 | 9 / 43 · 20.9% | 13 / 43 · 30.2% | 0 / 36 · 0.0% |
| π0.5 | 1 | 2 / 43 · 4.7% | 1 / 43 · 2.3% | 0 / 36 · 0.0% |
| π0.5 | 3 | 1 / 43 · 2.3% | 2 / 43 · 4.7% | 0 / 36 · 0.0% |
| π0.5 | 5 | 3 / 43 · 7.0% | 3 / 43 · 7.0% | 0 / 36 · 0.0% |

The README reports the common epoch-5 comparison, rather than selecting each model's best epoch independently. k1's B result peaks at epoch 3, so improvement is not monotonic. There is no matched untrained-model row on this entire 122-case evaluation, and the five C successes cover four held-out conditions, not five new skills.

The paper additionally evaluates a custom Qwen continuous-action regression baseline. Its RGB-D epoch-5 run completes all 122 cases with zero successes. Some RGB-only groups are incomplete; unrun cases are not counted as failures. This is a result for that particular training recipe, not evidence that continuous-action policies in general cannot learn these tasks.

### What is provided here

This repository provides the k1 harness and [tool-use data preparation, LoRA training, and serving interfaces](training.md). It does not bundle the teacher episodes, student weights, VLA baseline implementations, or benchmark evaluation manifests. Reproducing the tables requires those experiment inputs and the paper's full settings; a smoke test only checks basic integration.
