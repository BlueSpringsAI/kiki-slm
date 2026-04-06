#!/usr/bin/env python3
"""Kiki SLM SFT training script for Google Colab.

All configuration is read from configs/colab_config.yaml.
CLI args override config values.

Usage:
    python -u scripts/colab_train.py
    python -u scripts/colab_train.py --config configs/colab_config.yaml
    python -u scripts/colab_train.py --epochs 5 --lr 1e-4  # override config
    python -u scripts/colab_train.py --dry-run
    python -u scripts/colab_train.py --resume
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")

import logging
logging.getLogger("transformers.modeling_attn_mask_utils").setLevel(logging.ERROR)

import torch
import yaml
from datasets import load_dataset
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiki SLM SFT Training")
    parser.add_argument("--config", default="configs/colab_config.yaml", help="Config YAML path")
    # All overridable from CLI
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--eval-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--lora-r", type=int, default=None)
    parser.add_argument("--lora-alpha", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb", action="store_true", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def merge_config(cfg: dict, args: argparse.Namespace) -> dict:
    """Merge config file with CLI overrides. CLI wins."""
    # Flatten config for easy access
    c = {
        "train_file": args.train_file or cfg.get("data", {}).get("train_file", ""),
        "eval_file": args.eval_file or cfg.get("data", {}).get("eval_file", ""),
        "output_dir": args.output_dir or cfg.get("output", {}).get("adapter_dir", ""),
        # IMPORTANT: Use unsloth/ repo (not Qwen/) for Qwen3-4B-Thinking-2507.
        # Unsloth ships the patched chat template that correctly renders
        # reasoning_content as <think> blocks and handles tool calls with
        # reasoning together. The official Qwen repo template had bugs that
        # broke fine-tuning with assistant_only_loss (loss masking).
        # See: https://huggingface.co/unsloth/Qwen3-4B-Thinking-2507-GGUF/discussions/1
        "base_model": args.base_model or cfg.get("model", {}).get("name", "unsloth/Qwen3-4B-Thinking-2507"),
        "max_seq_length": args.max_seq_length or cfg.get("model", {}).get("max_seq_length", 2048),
        "load_in_4bit": cfg.get("model", {}).get("load_in_4bit", True),
        "lora_r": args.lora_r or cfg.get("lora", {}).get("r", 32),
        "lora_alpha": args.lora_alpha or cfg.get("lora", {}).get("alpha", 64),
        "lora_dropout": cfg.get("lora", {}).get("dropout", 0),
        "target_modules": cfg.get("lora", {}).get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
        "epochs": args.epochs or cfg.get("training", {}).get("epochs", 3),
        "lr": args.lr or cfg.get("training", {}).get("learning_rate", 2e-4),
        "lr_scheduler": cfg.get("training", {}).get("lr_scheduler", "cosine"),
        "warmup_ratio": cfg.get("training", {}).get("warmup_ratio", 0.03),
        "weight_decay": cfg.get("training", {}).get("weight_decay", 0.01),
        "packing": cfg.get("training", {}).get("packing", True),
        "optim": cfg.get("training", {}).get("optim", "adamw_8bit"),
        "logging_steps": cfg.get("training", {}).get("logging_steps", 10),
        "save_steps": cfg.get("training", {}).get("save_steps", 500),
        "save_total_limit": cfg.get("training", {}).get("save_total_limit", 3),
        "eval_steps": cfg.get("training", {}).get("eval_steps", 500),
        "seed": args.seed or cfg.get("training", {}).get("seed", 42),
        "gpu_profiles": cfg.get("training", {}).get("gpu_profiles", {}),
        "wandb_enabled": (args.wandb if args.wandb is not None else cfg.get("wandb", {}).get("enabled", False)) and not args.no_wandb,
        "wandb_project": args.wandb_project or cfg.get("wandb", {}).get("project", "kiki-slm"),
        "wandb_run_name": args.wandb_run_name or cfg.get("wandb", {}).get("run_name"),
        "resume": args.resume,
        "dry_run": args.dry_run,
    }
    return c


# ---------------------------------------------------------------------------
# GPU auto-detect with config profiles
# ---------------------------------------------------------------------------

def auto_detect_gpu(profiles: dict) -> dict:
    if not torch.cuda.is_available():
        print("  WARNING: No GPU detected")
        return {"gpu_name": "CPU", "gpu_mem_gb": 0, "batch_size": 1, "grad_accum": 4}

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if gpu_mem_gb >= 70 and "h100" in profiles:
        p = profiles["h100"]
    elif gpu_mem_gb >= 35 and "a100" in profiles:
        p = profiles["a100"]
    elif gpu_mem_gb >= 20 and "l4" in profiles:
        p = profiles["l4"]
    elif "t4" in profiles:
        p = profiles["t4"]
    else:
        p = {"batch_size": 2, "grad_accum": 8}

    return {"gpu_name": gpu_name, "gpu_mem_gb": round(gpu_mem_gb, 1),
            "batch_size": p.get("batch_size", 2), "grad_accum": p.get("grad_accum", 8)}


# ---------------------------------------------------------------------------
# Chat template helper
# ---------------------------------------------------------------------------

# Fields that Qwen3 chat template needs on assistant/tool messages.
# Must be preserved when normalizing — stripping these silently breaks
# tool calling and reasoning content rendering.
_MESSAGE_FIELDS = ("role", "content", "reasoning_content", "tool_calls", "tool_call_id", "name")


def _normalize_message(msg: dict) -> dict:
    """Normalize a message for the Qwen chat template without dropping fields.

    Preserves: role, content, reasoning_content, tool_calls, tool_call_id, name.
    Leaves content=None for tool-call-only assistant turns (Qwen template handles
    None correctly; empty string causes the model to learn to emit "").
    """
    out = {"role": str(msg.get("role", "user"))}
    # Content: preserve None/empty string as-is for tool-call turns.
    # If the assistant message has tool_calls, content SHOULD be None.
    # If it has a string, pass it through.
    if "content" in msg:
        content = msg["content"]
        if content is None:
            # Qwen template expects None (not "") for tool-call-only turns
            out["content"] = None
        else:
            out["content"] = str(content)
    else:
        out["content"] = None
    # Preserve reasoning_content (thinking models)
    if msg.get("reasoning_content") is not None:
        out["reasoning_content"] = str(msg["reasoning_content"])
    # Preserve tool_calls (list of dicts with id/type/function)
    if msg.get("tool_calls"):
        out["tool_calls"] = msg["tool_calls"]
    # Preserve tool_call_id (on tool-result messages)
    if msg.get("tool_call_id"):
        out["tool_call_id"] = str(msg["tool_call_id"])
    # Preserve tool name (on tool-result messages)
    if msg.get("name"):
        out["name"] = str(msg["name"])
    return out


def apply_chat_template_to_dataset(dataset, tokenizer):
    """Render ChatML examples through the Qwen3 chat template.

    Preserves tool_calls, reasoning_content, and tool_call_id so that
    tool-calling and thinking-mode examples render correctly. Passes the
    per-example `tools` list to apply_chat_template so tool schemas appear
    in the rendered text.
    """
    original_template = tokenizer.chat_template
    has_tools_column = "tools" in dataset.column_names

    def _apply(examples):
        texts = []
        msgs_batch = examples["messages"]
        tools_batch = examples.get("tools") if has_tools_column else None

        for i, msgs in enumerate(msgs_batch):
            normalized = [_normalize_message(m) for m in msgs]
            tools_arg = tools_batch[i] if tools_batch is not None else None

            try:
                # Pass tools= so the Qwen template injects the tool schema
                # into the system block. Without this, the model never sees
                # what tools are available during training.
                text = tokenizer.apply_chat_template(
                    normalized,
                    tools=tools_arg,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            except Exception as e:
                # Fallback: manual rendering (loses tool-call formatting,
                # but at least preserves content). Should rarely trigger.
                print(f"    WARN: chat template failed ({type(e).__name__}: {e}), using fallback")
                parts = [f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>" for m in normalized]
                text = "\n".join(parts)
            texts.append(text)
        return {"text": texts}

    cols_to_remove = [c for c in dataset.column_names if c != "text"]
    dataset = dataset.map(_apply, batched=True, remove_columns=cols_to_remove, desc="Applying chat template")
    return dataset, original_template


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class StepCountCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n{'='*60}")
        print(f"  TRAINING STARTED")
        print(f"  Total steps: {state.max_steps:,}")
        print(f"  Epochs: {args.num_train_epochs}")
        print(f"  Logging every: {args.logging_steps} steps")
        print(f"  Eval every: {args.eval_steps} steps")
        print(f"  Save every: {args.save_steps} steps")
        print(f"{'='*60}\n")

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        step = state.global_step
        total = state.max_steps
        pct = step / total * 100 if total > 0 else 0
        parts = []
        if "loss" in logs:
            parts.append(f"loss={logs['loss']:.4f}")
        if "eval_loss" in logs:
            parts.append(f"eval_loss={logs['eval_loss']:.4f}")
        if "learning_rate" in logs:
            parts.append(f"lr={logs['learning_rate']:.2e}")
        if parts:
            print(f"  [{step:>6d}/{total}] ({pct:5.1f}%) | {' | '.join(parts)}")


# ---------------------------------------------------------------------------
# W&B setup
# ---------------------------------------------------------------------------

def setup_wandb(c: dict, gpu_info: dict, train_size: int, eval_size: int) -> str | None:
    if not c["wandb_enabled"]:
        return None
    try:
        import wandb
        run_name = c["wandb_run_name"] or f"sft-r{c['lora_r']}-ep{c['epochs']}-{time.strftime('%m%d-%H%M')}"
        wandb.init(
            project=c["wandb_project"], name=run_name,
            config={**c, "gpu": gpu_info["gpu_name"], "gpu_mem_gb": gpu_info["gpu_mem_gb"],
                    "train_examples": train_size, "eval_examples": eval_size},
        )
        url = wandb.run.url
        print(f"  W&B: {url}")
        return url
    except Exception as e:
        print(f"  WARNING: W&B init failed ({e}), falling back to file logging")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Load config
    cfg = load_config(args.config) if os.path.exists(args.config) else {}
    c = merge_config(cfg, args)

    print(f"\n{'='*60}")
    print(f"  KIKI SLM — SFT TRAINING")
    print(f"  Config: {args.config}")
    print(f"{'='*60}")

    # GPU
    gpu_info = auto_detect_gpu(c["gpu_profiles"])
    batch_size = args.batch_size or gpu_info["batch_size"]
    grad_accum = args.grad_accum or gpu_info["grad_accum"]
    print(f"\n  GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_mem_gb']}GB)")
    print(f"  Batch: {batch_size} x {grad_accum} grad_accum = {batch_size * grad_accum} effective")
    print(f"  Model: {c['base_model']}")
    print(f"  LoRA: r={c['lora_r']}, alpha={c['lora_alpha']}, modules={c['target_modules']}")
    print(f"  Training: {c['epochs']} epochs, lr={c['lr']}, scheduler={c['lr_scheduler']}")

    # Validate files
    for label, path in [("Train", c["train_file"]), ("Eval", c["eval_file"])]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} file not found: {path}")
            sys.exit(1)
        print(f"  {label}: {path} ({os.path.getsize(path) / 1024 / 1024:.1f}MB)")

    # Load data
    print("\n  Loading data...")
    train_dataset = load_dataset("json", data_files=c["train_file"], split="train")
    eval_dataset = load_dataset("json", data_files=c["eval_file"], split="train")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Eval:  {len(eval_dataset):,} examples")

    if "source" in train_dataset.column_names:
        sources = Counter(train_dataset["source"])
        print(f"\n  Source distribution:")
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {src:<30s} {cnt:>6,} ({cnt / len(train_dataset) * 100:.1f}%)")

    # Dry run
    if c["dry_run"]:
        print(f"\n  DRY RUN — config validated, exiting.")
        print(f"  Would train {c['epochs']} epochs on {len(train_dataset):,} examples")
        return

    # W&B
    wandb_url = setup_wandb(c, gpu_info, len(train_dataset), len(eval_dataset))
    report_to = ["wandb"] if wandb_url else ["none"]

    # Load model
    from unsloth import FastLanguageModel

    print(f"\n  Loading model: {c['base_model']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=c["base_model"], max_seq_length=c["max_seq_length"],
        load_in_4bit=c["load_in_4bit"], dtype=None,
    )

    # LoRA
    print(f"  Applying LoRA (r={c['lora_r']}, alpha={c['lora_alpha']})")
    model = FastLanguageModel.get_peft_model(
        model, r=c["lora_r"], lora_alpha=c["lora_alpha"],
        target_modules=c["target_modules"], lora_dropout=c["lora_dropout"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=c["seed"],
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.1f}M / {total_params / 1e6:.1f}M ({trainable / total_params * 100:.2f}%)")

    # Chat template
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n  Applying chat template...")
    train_dataset, original_template = apply_chat_template_to_dataset(train_dataset, tokenizer)
    eval_dataset, _ = apply_chat_template_to_dataset(eval_dataset, tokenizer)
    tokenizer.chat_template = None

    # Trainer
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=c["output_dir"], dataset_text_field="text", packing=c["packing"],
        per_device_train_batch_size=batch_size, gradient_accumulation_steps=grad_accum,
        num_train_epochs=c["epochs"], learning_rate=c["lr"],
        lr_scheduler_type=c["lr_scheduler"], warmup_ratio=c["warmup_ratio"],
        bf16=torch.cuda.is_bf16_supported(), fp16=not torch.cuda.is_bf16_supported(),
        optim=c["optim"], weight_decay=c["weight_decay"], max_seq_length=c["max_seq_length"],
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=c["logging_steps"], logging_first_step=True,
        save_strategy="steps", save_steps=c["save_steps"], save_total_limit=c["save_total_limit"],
        eval_strategy="steps", eval_steps=c["eval_steps"],
        disable_tqdm=False, report_to=report_to, seed=c["seed"], dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model, tokenizer=tokenizer,
        train_dataset=train_dataset, eval_dataset=eval_dataset,
        args=training_args, callbacks=[StepCountCallback()],
    )

    # Loss masking: only compute loss on assistant turns, not on user tickets
    # or tool results. Without this, the model wastes capacity learning to
    # predict retrieved KB documents and customer ticket text.
    try:
        from unsloth.chat_templates import train_on_responses_only
        # Qwen3 ChatML format uses <|im_start|>{role}\n ... <|im_end|>
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        print("  Loss masking: enabled (train_on_responses_only)")
    except ImportError:
        print("  WARNING: unsloth.chat_templates.train_on_responses_only not available — loss computed on all tokens")
    except Exception as e:
        print(f"  WARNING: train_on_responses_only failed ({type(e).__name__}: {e}) — loss computed on all tokens")

    # Resume
    resume_checkpoint = None
    if c["resume"]:
        import glob
        checkpoints = sorted(glob.glob(os.path.join(c["output_dir"], "checkpoint-*")))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            print(f"\n  Resuming from: {resume_checkpoint}")
        else:
            print(f"\n  --resume: no checkpoints found, starting fresh")

    # Train
    print(f"  GPU memory: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
    start_time = time.time()
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    elapsed = time.time() - start_time

    final_loss = result.metrics.get("train_loss", 0)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Duration:    {elapsed / 3600:.1f}h ({elapsed / 60:.0f}m)")
    print(f"  Final loss:  {final_loss:.4f}")
    print(f"  Peak VRAM:   {peak_vram:.2f}GB")
    if wandb_url:
        print(f"  W&B:         {wandb_url}")

    # Eval
    eval_results = trainer.evaluate()
    print(f"\n  Eval results:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # Save
    os.makedirs(c["output_dir"], exist_ok=True)
    model.save_pretrained(c["output_dir"])
    tokenizer.save_pretrained(c["output_dir"])

    # Patch chat template
    tok_config = Path(c["output_dir"]) / "tokenizer_config.json"
    if tok_config.exists() and original_template:
        with open(tok_config) as f:
            tc = json.load(f)
        tc["chat_template"] = original_template
        with open(tok_config, "w") as f:
            json.dump(tc, f, indent=2, ensure_ascii=False)
        print(f"  Chat template restored in tokenizer_config.json")

    # Metrics
    metrics = {
        "config": c, "final_loss": final_loss, "eval_results": eval_results,
        "duration_hours": round(elapsed / 3600, 2), "gpu": gpu_info["gpu_name"],
        "peak_vram_gb": round(peak_vram, 2), "wandb_url": wandb_url,
        "train_examples": len(train_dataset), "eval_examples": len(eval_dataset),
    }
    metrics_path = Path(c["output_dir"]) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n  Adapter: {c['output_dir']}")
    print(f"  Metrics: {metrics_path}")

    if wandb_url:
        try:
            import wandb
            wandb.log({"peak_vram_gb": peak_vram})
            wandb.finish()
        except Exception:
            pass

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
