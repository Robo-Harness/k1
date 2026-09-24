# Fine-tuning and serving

The supplied trainer performs assistant-target LoRA SFT for compatible Qwen3.5 visual-language models. It is not a continuous-action VLA baseline. It preserves explicit tool calls and visible decision summaries; it does not train on hidden provider reasoning.

## 1. Supply local episodes and fixed splits

Each source episode needs `result.json`, `agent.jsonl` and `model_requests/call_XXXXXX.json` from the harness. Exact requests contain the images/context that were available before the labeled action. Do not reconstruct prompts using later evidence.

Create a local JSON manifest, for example:

```json
[
  {"id": "episode-a", "path": "episodes/a", "family": "family-a", "split": "train"},
  {"id": "episode-b", "path": "episodes/b", "family": "family-b", "split": "validation"},
  {"id": "episode-c", "path": "episodes/c", "family": "family-c", "split": "test"}
]
```

Relative episode paths are resolved against the manifest's directory. Choose family IDs that group related task variants so they cannot leak across splits. This converter rejects duplicate IDs and cross-split family membership. A different experimental split policy requires an explicit protocol change, not relabeling a test family as training.

```bash
python -m pip install -e '.[training]'
robo-harness-prepare --manifest data/episodes.json --output data/sft
```

Preparation retains native-success episodes and excludes malformed/multiple tool calls, execution errors, schema violations and rejected completion calls. It exports local image references, tools, observed messages and assistant targets. Opaque signatures/reasoning fields are removed. Earlier excluded actions remain in the exact subsequent context; history is not rewritten to look successful.

Important: this filtering does not prove every accepted action is optimal. Successful episodes can contain detours and redundant queries. Review the resulting distribution and keep all filtering counts. No data is included in this repository or uploaded by the converter.

## 2. Train LoRA

Use a separate training environment with a Transformers release supporting Qwen3.5 multimodal models. Supply a local checkpoint; remote code execution and automatic weight downloads are disabled. Optional FLA kernels can improve throughput but must match your Torch/CUDA stack.

```bash
CUDA_VISIBLE_DEVICES=0 robo-harness-train \
  --base-model checkpoints/qwen \
  --train data/sft/train.jsonl \
  --output outputs/sft-run \
  --epochs 5
```

Defaults: rank 16, alpha 32, dropout 0.05, learning rate 1e-4, accumulation 8, BF16, gradient checkpointing, frozen visual encoder, and maximum 32,768 tokens. Oversized samples fail explicitly; the trainer does not silently delete context or truncate an action. Loss applies only to the assistant target; prompt tokens are masked. Epoch checkpoints contain LoRA weights, processor files, optimizer state and RNG states, not another copy of base weights.

```bash
CUDA_VISIBLE_DEVICES=0 robo-harness-train \
  --base-model checkpoints/qwen \
  --train data/sft/train.jsonl \
  --output outputs/sft-run \
  --resume outputs/sft-run/epoch-2 \
  --epochs 5
```

Resume is at **completed epoch boundaries**. A partially completed epoch is repeated from its previous checkpoint. Resume requires the same data and training settings and the same base weights. Only load trusted optimizer checkpoints. This is a single-GPU trainer, not a distributed-training launcher; memory availability and full-model compatibility require a local forward/backward smoke test.

## 3. Evaluate through the same harness

Serve the base model plus the selected adapter through an OpenAI-compatible server that supports its multimodal chat template and function-tool serialization. The harness consumes ordinary `tool_calls` with JSON arguments, not unparsed raw XML text. Configuring a model server to emit that contract is required.

Set `ROBO_HARNESS_BASE_URL` to the local loopback endpoint, `ROBO_HARNESS_MODEL` to its served model ID, and `ROBO_HARNESS_API_KEY` to the key expected by that service. Use the normal `robo-harness run` command and unchanged evaluation YAML. There is no automatic merge, server launch or GPU allocation.

Compare base and fine-tuned checkpoints on identical task/initial-state sets and budgets. Keep training-state evaluation, new-state evaluation and held-out-task evaluation distinct. Teacher-forced loss is not a substitute for native closed-loop success. This source release includes no private scores, trained adapters or claims of training improvement.
