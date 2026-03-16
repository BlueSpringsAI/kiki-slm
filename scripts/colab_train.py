#!/usr/bin/env python3
"""Kiki SLM SFT training script for Google Colab.

Loads pre-formatted ChatML data, fine-tunes Qwen3-4B with QLoRA via Unsloth,
logs to W&B, saves adapter to output directory.

Self-contained — does NOT import from the kiki package.

Usage:
    python -u scripts/colab_train.py \
        --train-file /content/drive/MyDrive/kiki-slm/data/kiki_sft_chatml_train.jsonl \
        --eval-file /content/drive/MyDrive/kiki-slm/data/kiki_sft_chatml_eval.jsonl \
        --output-dir /content/drive/MyDrive/kiki-slm/adapters/kiki-sft-v1 \
        --wandb
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

# Unbuffered output for real-time tqdm in Colab subprocess
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from datasets import load_dataset
from transformers import TrainerCallback


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiki SLM SFT Training")
    # Data
    parser.add_argument("--train-file", required=True, help="Path to ChatML train JSONL")
    parser.add_argument("--eval-file", required=True, help="Path to ChatML eval JSONL")
    parser.add_argument("--output-dir", required=True, help="Where to save adapter + metrics")
    # Model
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="HF model ID")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    # LoRA
    parser.add_argument("--lora-r", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    # Training
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=None, help="Auto-detect if not set")
    parser.add_argument("--grad-accum", type=int, default=None, help="Auto-detect if not set")
    parser.add_argument("--seed", type=int, default=42)
    # W&B
    parser.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-project", default="kiki-slm", help="W&B project name")
    parser.add_argument("--wandb-run-name", default=None, help="W&B run name (auto if not set)")
    # Resume
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")
    # Misc
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# GPU auto-detect
# ---------------------------------------------------------------------------

def auto_detect_gpu() -> dict:
    """Auto-detect GPU and return batch_size, grad_accum settings."""
    if not torch.cuda.is_available():
        print("  WARNING: No GPU detected, using CPU defaults")
        return {"gpu_name": "CPU", "gpu_mem_gb": 0, "batch_size": 1, "grad_accum": 4}

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3

    if gpu_mem_gb >= 35:  # A100
        batch_size, grad_accum = 4, 8
    elif gpu_mem_gb >= 20:  # L4
        batch_size, grad_accum = 4, 4
    else:  # T4 or smaller
        batch_size, grad_accum = 2, 8

    return {
        "gpu_name": gpu_name,
        "gpu_mem_gb": round(gpu_mem_gb, 1),
        "batch_size": batch_size,
        "grad_accum": grad_accum,
    }


# ---------------------------------------------------------------------------
# Chat template helper (self-contained, no kiki package dependency)
# ---------------------------------------------------------------------------

def apply_chat_template_to_dataset(dataset, tokenizer):
    """Convert messages column to text column using chat template.

    Sanitizes each message to role+content only to avoid Jinja2 errors
    from mixed dataset formats (xlam, hermes, arcee).
    """
    original_template = tokenizer.chat_template

    def _apply(examples):
        texts = []
        for msgs in examples["messages"]:
            clean = [
                {"role": str(m.get("role", "user")), "content": str(m.get("content") or "")}
                for m in msgs
            ]
            try:
                text = tokenizer.apply_chat_template(clean, tokenize=False, add_generation_prompt=False)
            except Exception:
                parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in clean]
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
    """Print correct step count at training start and log progress to stdout."""

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

def setup_wandb(args, gpu_info: dict, train_size: int, eval_size: int) -> str | None:
    """Initialize W&B. Returns run URL or None on failure."""
    if not args.wandb:
        return None
    try:
        import wandb

        run_name = args.wandb_run_name or f"sft-r{args.lora_r}-ep{args.epochs}-{time.strftime('%m%d-%H%M')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "base_model": args.base_model,
                "lora_r": args.lora_r,
                "lora_alpha": args.lora_alpha,
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "max_seq_length": args.max_seq_length,
                "batch_size": gpu_info["batch_size"],
                "grad_accum": gpu_info["grad_accum"],
                "gpu": gpu_info["gpu_name"],
                "gpu_mem_gb": gpu_info["gpu_mem_gb"],
                "train_examples": train_size,
                "eval_examples": eval_size,
                "seed": args.seed,
            },
        )
        print(f"  W&B run initialized: {wandb.run.get_url()}")
        return wandb.run.get_url()
    except Exception as e:
        print(f"  WARNING: W&B init failed ({e}), falling back to file logging")
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  KIKI SLM — SFT TRAINING")
    print(f"{'='*60}")

    # 1. GPU auto-detect
    gpu_info = auto_detect_gpu()
    batch_size = args.batch_size or gpu_info["batch_size"]
    grad_accum = args.grad_accum or gpu_info["grad_accum"]
    print(f"\n  GPU: {gpu_info['gpu_name']} ({gpu_info['gpu_mem_gb']}GB)")
    print(f"  Batch: {batch_size} x {grad_accum} grad_accum = {batch_size * grad_accum} effective")

    # 2. Validate data files
    for label, path in [("Train", args.train_file), ("Eval", args.eval_file)]:
        if not os.path.exists(path):
            print(f"  ERROR: {label} file not found: {path}")
            sys.exit(1)
        size_mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {label}: {path} ({size_mb:.1f}MB)")

    # 3. Load data
    print("\n  Loading data...")
    train_dataset = load_dataset("json", data_files=args.train_file, split="train")
    eval_dataset = load_dataset("json", data_files=args.eval_file, split="train")
    print(f"  Train: {len(train_dataset):,} examples")
    print(f"  Eval:  {len(eval_dataset):,} examples")

    # Source distribution
    if "source" in train_dataset.column_names:
        sources = Counter(train_dataset["source"])
        print(f"\n  Source distribution:")
        for src, cnt in sorted(sources.items(), key=lambda x: -x[1]):
            print(f"    {src:<30s} {cnt:>6,} ({cnt / len(train_dataset) * 100:.1f}%)")

    # 4. Dry run check
    if args.dry_run:
        print(f"\n  DRY RUN — config validated, exiting without loading model.")
        print(f"  Would train {args.epochs} epochs on {len(train_dataset):,} examples")
        return

    # 5. W&B setup
    wandb_url = setup_wandb(args, gpu_info, len(train_dataset), len(eval_dataset))
    report_to = ["wandb"] if wandb_url else ["none"]

    # 6. Load model
    from unsloth import FastLanguageModel

    print(f"\n  Loading model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    # 7. Apply LoRA
    print(f"  Applying LoRA (r={args.lora_r}, alpha={args.lora_alpha})")
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable / 1e6:.1f}M / {total_params / 1e6:.1f}M ({trainable / total_params * 100:.2f}%)")

    # 8. Apply chat template
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("\n  Applying chat template...")
    train_dataset, original_template = apply_chat_template_to_dataset(train_dataset, tokenizer)
    eval_dataset, _ = apply_chat_template_to_dataset(eval_dataset, tokenizer)

    # Disable template for training (prevents Jinja2 errors in SFTTrainer)
    tokenizer.chat_template = None

    # 9. Configure trainer
    from trl import SFTConfig, SFTTrainer

    training_args = SFTConfig(
        output_dir=args.output_dir,
        dataset_text_field="text",
        packing=True,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        optim="adamw_8bit",
        weight_decay=0.01,
        max_seq_length=args.max_seq_length,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=500,
        disable_tqdm=False,
        report_to=report_to,
        seed=args.seed,
        dataloader_pin_memory=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        callbacks=[StepCountCallback()],
    )

    # 10. Train (with optional resume from checkpoint)
    resume_checkpoint = None
    if args.resume:
        import glob
        checkpoints = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-*")))
        if checkpoints:
            resume_checkpoint = checkpoints[-1]
            print(f"\n  Resuming from checkpoint: {resume_checkpoint}")
        else:
            print(f"\n  --resume passed but no checkpoints found in {args.output_dir}, starting fresh")

    print(f"  GPU memory before training: {torch.cuda.memory_allocated() / 1024**3:.2f}GB allocated")
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

    # 11. Eval
    eval_results = trainer.evaluate()
    print(f"\n  Eval results:")
    for k, v in eval_results.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")

    # 12. Save adapter
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # 13. Patch chat template back into saved tokenizer_config.json
    tokenizer_config_path = Path(args.output_dir) / "tokenizer_config.json"
    if tokenizer_config_path.exists() and original_template:
        with open(tokenizer_config_path) as f:
            config = json.load(f)
        config["chat_template"] = original_template
        with open(tokenizer_config_path, "w") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  Chat template restored in saved tokenizer_config.json")

    # 14. Save training metrics
    metrics = {
        "model": args.base_model,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "train_examples": len(train_dataset),
        "eval_examples": len(eval_dataset),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "final_loss": final_loss,
        "eval_results": eval_results,
        "duration_hours": round(elapsed / 3600, 2),
        "gpu": gpu_info["gpu_name"],
        "peak_vram_gb": round(peak_vram, 2),
        "wandb_url": wandb_url,
        "seed": args.seed,
    }
    metrics_path = Path(args.output_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\n  Adapter saved to: {args.output_dir}")
    print(f"  Metrics saved to: {metrics_path}")

    # 15. Finish W&B
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
