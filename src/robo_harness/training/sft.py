"""Single-GPU LoRA SFT on exact visual tool-use requests, with epoch checkpoints."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import random


def encode(processor, row, maximum=32768):
    import torch

    messages = copy.deepcopy(row["messages"])
    options = dict(
        tools=row["tools"],
        tokenize=True,
        enable_thinking=False,
        return_dict=True,
        return_tensors="pt",
    )
    prefix = processor.apply_chat_template(messages, add_generation_prompt=True, **options)
    batch = processor.apply_chat_template(
        [*messages, row["target"]], add_generation_prompt=False, **options
    )
    count = prefix["input_ids"].shape[-1]
    if not torch.equal(prefix["input_ids"], batch["input_ids"][:, :count]):
        raise ValueError("Assistant template prefix mismatch")
    if not count < batch["input_ids"].shape[-1] <= maximum:
        raise ValueError("Empty target or oversized sample; no silent truncation")
    labels = batch["input_ids"].clone()
    labels[:, :count] = -100
    batch["labels"] = labels
    return dict(batch), count


def target_loss(model, batch, prefix):
    """Suffix-only logits: exact next-token assistant CE without prompt loss."""
    import torch

    length = batch["input_ids"].shape[-1] - prefix
    labels = batch["labels"][:, prefix:]
    if length < 1 or prefix < 1 or not (labels >= 0).all():
        raise ValueError("Expected a nonempty unmasked assistant target")
    inputs = {k: v for k, v in batch.items() if k != "labels"}
    logits = model(**inputs, logits_to_keep=length + 1, use_cache=False).logits[:, :-1]
    return torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1)
    )


def train(args):
    import torch

    # Optional kernel preload avoids slow lazy-import fallback on compatible installations.
    try:
        import fla.ops.gated_delta_rule  # noqa: F401
    except ImportError:
        pass
    from transformers import AutoProcessor, AutoModelForImageTextToText
    from peft import LoraConfig, PeftModel, get_peft_model

    if not args.base_model.is_dir():
        raise ValueError("Provide a local model directory; automatic downloads are disabled")
    if args.epochs < 1 or args.accumulation < 1 or args.rank < 1:
        raise ValueError("epochs, accumulation and rank must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this trainer")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    rows = [json.loads(line) for line in args.train.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No training examples")
    data_hash = hashlib.sha256(args.train.read_bytes()).hexdigest()
    args.output.mkdir(parents=True, exist_ok=args.resume is not None)
    processor = AutoProcessor.from_pretrained(
        args.base_model, local_files_only=True, trust_remote_code=False
    )
    processor.image_processor.size = {"shortest_edge": 65536, "longest_edge": 262144}
    model = AutoModelForImageTextToText.from_pretrained(
        args.base_model,
        local_files_only=True,
        trust_remote_code=False,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        attn_implementation="sdpa",
    )
    modules = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
        and "language_model" in name
        and name.rsplit(".", 1)[-1]
        in {
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
        }
    ]
    if not modules:
        raise ValueError("No supported language LoRA modules found; use a compatible Qwen3.5 VLM")
    model = (
        PeftModel.from_pretrained(model, args.resume, is_trainable=True)
        if args.resume
        else get_peft_model(
            model,
            LoraConfig(
                r=args.rank,
                lora_alpha=args.rank * 2,
                lora_dropout=0.05,
                target_modules=modules,
                bias="none",
                task_type="CAUSAL_LM",
            ),
        )
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if any(p.requires_grad for name, p in model.named_parameters() if "visual" in name):
        raise RuntimeError("Vision encoder must remain frozen")
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    settings = {
        "algorithm": "robo-harness k1",
        "data_sha256": data_hash,
        "epochs": args.epochs,
        "rank": args.rank,
        "learning_rate": args.learning_rate,
        "accumulation": args.accumulation,
        "max_length": args.max_length,
        "seed": args.seed,
        "samples": len(rows),
        "base_model_type": model.config.model_type,
        "vision_frozen": True,
    }
    start, updates = 1, 0
    if args.resume:
        # Only load your own trusted checkpoint. Optimizer files use torch serialization.
        state = torch.load(args.resume / "trainer_state.pt", map_location="cpu", weights_only=False)
        for field in (
            "data_sha256",
            "rank",
            "learning_rate",
            "accumulation",
            "max_length",
            "seed",
            "base_model_type",
        ):
            if state["settings"][field] != settings[field]:
                raise ValueError("Resume configuration mismatch: " + field)
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        start, updates = state["epoch"] + 1, state["updates"]
    with (args.output / "settings.json").open("w") as stream:
        json.dump(settings, stream, indent=2)
    model.train()
    with (args.output / "training.jsonl").open("a") as log:
        for epoch in range(start, args.epochs + 1):
            order = list(range(len(rows)))
            random.Random(args.seed + epoch).shuffle(order)
            optimizer.zero_grad(set_to_none=True)
            gradient_seen = False
            for i, index in enumerate(order):
                batch, prefix = encode(processor, rows[index], args.max_length)
                batch = {k: v.to("cuda:0") for k, v in batch.items()}
                loss = target_loss(model, batch, prefix)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss")
                group_size = min(
                    args.accumulation, len(rows) - (i // args.accumulation) * args.accumulation
                )
                (loss / group_size).backward()
                if (i + 1) % args.accumulation == 0 or i + 1 == len(rows):
                    gradient_seen |= any(
                        p.grad is not None and bool(torch.any(p.grad != 0)) for p in parameters
                    )
                    norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
                    if not torch.isfinite(norm):
                        raise RuntimeError("Nonfinite gradient")
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    updates += 1
                log.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "sample": i,
                            "loss": float(loss.detach()),
                            "updates": updates,
                        }
                    )
                    + "\n"
                )
                log.flush()
            if not gradient_seen:
                raise RuntimeError("No nonzero gradient reached the LoRA parameters")
            checkpoint = args.output / f"epoch-{epoch}"
            checkpoint.mkdir(exist_ok=False)
            model.save_pretrained(checkpoint)
            processor.save_pretrained(checkpoint)
            torch.save(
                {
                    "epoch": epoch,
                    "updates": updates,
                    "settings": settings,
                    "optimizer": optimizer.state_dict(),
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all(),
                },
                checkpoint / "trainer_state.pt",
            )
            print(
                json.dumps({"epoch": epoch, "updates": updates, "checkpoint_saved": True}),
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description="robo-harness k1 Qwen3.5 LoRA SFT")
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=17)
    train(parser.parse_args())


if __name__ == "__main__":
    main()
