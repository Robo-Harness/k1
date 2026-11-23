# Paper results and evaluation scope

[← README](../README.md)

This page summarizes the updated paper's frontier-model comparison, cross-environment transfer, real-robot experiments, and student training results in **Robo-Harness K1: Harnessing Robot-Use Agents via Perception Augmentation**.

Simulation results measure **native task success**; real-robot results use operator confirmation and image review. Selected demonstrations accompany the results. Raw episode logs, evaluation manifests, training data, and model checkpoints are supplied separately.

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

### Matched 60-episode comparison

All ten task indices at state 0 are used in each of the six category–variant suites: **3 categories × 10 tasks × 2 variants = 60 cases per system**. All four systems use the same configurations.

| Model | Interface | Native successes | Success rate |
|:---|:---|---:|---:|
| Gemini 3.7 Flash | RGB-only robot-use interface | 10 / 60 | 16.7% |
| Gemini 3.7 Flash | K1 | 43 / 60 | 71.7% |
| GPT-6 Astra | RGB-only robot-use interface | 38 / 60 | 63.3% |
| GPT-6 Astra | K1 | 56 / 60 | 93.3% |

K1 improves Gemini by **55.0 percentage points** and Astra by **30.0 points**. The comparison evaluates complete interfaces: K1 adds depth-backed perception tools, memory, motion primitives, and completion feedback.

| System | Spatial · 20 | Object · 20 | Goal · 20 |
|:---|---:|---:|---:|
| Gemini RGB-only | 1/20 · 5.0% | 8/20 · 40.0% | 1/20 · 5.0% |
| Gemini + K1 | 15/20 · 75.0% | 16/20 · 80.0% | 12/20 · 60.0% |
| Astra RGB-only | 9/20 · 45.0% | 16/20 · 80.0% | 13/20 · 65.0% |
| Astra + K1 | 18/20 · 90.0% | 20/20 · 100.0% | 18/20 · 90.0% |

Spatial shows the largest gain for both models: **+70 points for Gemini** and **+45 points for Astra**. Gemini's 60 paired cases are included in the 180-case sweep, giving 71.7% on the paired subset and 77.2% across all three states. Hatched Figure 1 bars are published references with their own evaluation protocols.

### Native completion versus post-release completion

The reported scores use the native success criterion at evaluation time. A separate diagnostic on the 180 Gemini episodes holds the arm, opens the gripper for 20 native frames, and requires success over the final ten frames. Gemini's post-release result is **131 / 180 (72.8%)**, compared with 139 / 180 native successes: 11 native successes lose success and three native failures gain success. This diagnostic is separate because opening the gripper changes the state and can affect completion in either direction.

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

## RoboTwin and RoboCasa transfer

Gemini 3.7 Flash uses K1 without target-environment model fine-tuning. These results extend the evaluation to Aloha AgileX dual arms in SAPIEN (RoboTwin) and PandaOmron mobile manipulation in MuJoCo (RoboCasa).

### RoboTwin

Ten tasks are evaluated under Easy and Hard conditions. Hard changes clutter, lighting, textures, and table height. After initialization-quality filtering, 182 episodes remain.

| Task | Easy | Hard |
|:---|---:|---:|
| Grab Roller | 8/10 · 80.0% | 9/10 · 90.0% |
| Handover Mic | 2/10 · 20.0% | 0/10 · 0.0% |
| Lift Pot | 2/10 · 20.0% | 2/10 · 20.0% |
| Move Can Pot | 4/8 · 50.0% | 5/8 · 62.5% |
| Place Dual Shoes | 0/8 · 0.0% | 0/6 · 0.0% |
| Place Phone Stand | 2/10 · 20.0% | 0/8 · 0.0% |
| Put Bottles Dustbin | 0/10 · 0.0% | 0/8 · 0.0% |
| Put Object Cabinet | 4/10 · 40.0% | 0/8 · 0.0% |
| Stack Blocks Two | 6/10 · 60.0% | 7/8 · 87.5% |
| Stack Bowls Two | 3/10 · 30.0% | 2/10 · 20.0% |
| **Episode-micro total** | **31/96 · 32.3%** | **25/86 · 29.1%** |
| **Task-macro average** | **32.0%** | **28.0%** |

The paper reports the task-macro average, weighting each task equally. Published RDT and π0 results, recomputed over these ten tasks from RoboTwin 2.0's Appendix K, are **43.1% / 15.9%** and **64.1% / 23.0%** on Easy / Hard. Those policies use task-specific training and their own evaluation protocols; these are published references. K1's strongest tasks are roller manipulation and block stacking, while handover and cluttered placement remain challenging.

### RoboCasa

The interface supports arm and gripper control, planar base translation, base rotation, and torso height. Each task uses ten distinct initial states.

| Category | Tasks | Successes | Success rate |
|:---|---:|---:|---:|
| Pick-and-place | 8 | 18/80 | 22.5% |
| Doors and drawers | 6 | 14/60 | 23.3% |
| Stove control | 2 | 4/20 | 20.0% |
| Faucet control | 3 | 10/30 | 33.3% |
| Microwave operation | 2 | 5/20 | 25.0% |
| Coffee preparation | 3 | 2/30 | 6.7% |
| **Overall** | **24** | **53/240** | **22.1%** |

Counter-to-sink placement reaches 80% and drawer closing reaches 60%. Microwave activation and coffee-machine button pressing remain unsolved in this evaluation, highlighting the difficulty of precise appliance contact.

## Real-robot manipulation

The physical setup uses a Piper arm, a parallel-jaw gripper, and two external RGB-D cameras. Both methods request Gemini 3.7 Flash. The Clean baseline uses RGB observations, measured arm pose, and four tools: `capture_scene`, `move`, `hold`, and `finish`. K1 adds perceptual tools and structured history with the same robot-control and IK adapters.

| Task | Clean success | K1 success | Clean milestone | K1 milestone |
|:---|---:|---:|---:|---:|
| Can into bowl | 1/3 · 33.3% | **2/3 · 66.7%** | 1.00 | **1.67** |
| Bowl onto plate | 0/3 · 0.0% | **3/3 · 100.0%** | 0.00 | **2.00** |
| Two-object sorting | 0/3 · 0.0% | **3/3 · 100.0%** | 0.50 | **2.00** |
| **Overall** | **1/9 · 11.1%** | **8/9 · 88.9%** | — | — |

Outcomes use operator confirmation and image review. Milestones are manually assigned on a 0–2 scale, covering approach, grasp-and-lift, transport, and stable placement. Can placement uses containment. The baseline has a 50-call limit; K1 uses 50 for can and bowl placement. The archived sorting demonstration extends to 100 calls and finishes at 88. One bowl trial includes external completion review and gripper-unit guidance.

The [real-robot demo](https://robo-harness.github.io/k1/#real-robot) shows the single-segment can-placement success. The GIF includes all 33 recorded decisions with task query, current views, separate observation/result frame numbers, and tool calls. It is a decision-snapshot presentation; API waiting time is omitted.

The experiment requested Gemini 3.7 Flash; the archived provider response for this demonstration is labeled `gemini-3.8-flash`, which is preserved in the video overlay.

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
| Qwen3.5-9B + K1 (before fine-tuning) | 0 | 0 / 43 · 0.0% | 0 / 43 · 0.0% | 0 / 36 · 0.0% |
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

The README plots checkpoint learning curves. K1's B result peaks at epoch 3. The base Qwen3.5-9B model was subsequently evaluated on the same 122 cases and scores zero on all three splits. The five C successes cover four held-out conditions.

The paper additionally evaluates a custom Qwen continuous-action regression baseline. Its RGB-D epoch-5 run completes all 122 cases with zero successes. Some RGB-only groups are incomplete; unrun cases are not counted as failures. This is a result for that particular training recipe, not evidence that continuous-action policies in general cannot learn these tasks.

## Mixed-environment post-training

The LIBERO-trained Qwen3.5-9B agent continues from its epoch-3 checkpoint for three additional LoRA epochs on **316 successful trajectories**: 107 LIBERO-Pro, 142 RoboSuite, 32 RoboCasa, and 35 RoboTwin. The objective remains next-tool-call prediction.

| Evaluation | Before | After | Gain |
|:---|---:|---:|---:|
| LIBERO-Pro B: new initial states | 18/42 · 42.9% | **22/42 · 52.4%** | **+9.5 pp** |
| LIBERO-Pro C: held-out task conditions | 4/36 · 11.1% | **10/36 · 27.8%** | **+16.7 pp** |
| RoboSuite | 17/60 · 28.3% | **30/60 · 50.0%** | **+21.7 pp** |
| RoboCasa | 4/48 · 8.3% | 2/48 · 4.2% | −4.2 pp |
| RoboTwin | 1/39 · 2.6% | 1/40 · 2.5% | — |

The before/after experiment uses fresh evaluations of the starting checkpoint with the same interface within each environment. Its starting scores are separate from the earlier learning curves. Diverse manipulation experience improves source-domain generalization and RoboSuite performance, while RoboCasa and RoboTwin remain difficult for the student.

## What is provided here

This repository provides the k1 harness and [tool-use data preparation, LoRA training, and serving interfaces](training.md). It does not bundle the teacher episodes, student weights, VLA baseline implementations, or benchmark evaluation manifests. Reproducing the tables requires those experiment inputs and the paper's full settings; a smoke test only checks basic integration.
