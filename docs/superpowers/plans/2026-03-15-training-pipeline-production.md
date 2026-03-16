# Training Pipeline Production Refactor — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 33-cell Colab notebook with a ~15-cell thin executor backed by two production scripts (`colab_train.py`, `colab_eval.py`) with W&B monitoring, correct step counting, and real-time tqdm progress.

**Architecture:** Self-contained scripts (no `kiki` package imports) called from notebook via `!python -u`. W&B for experiment tracking. Sequential model loading in eval to prevent OOM. Chat template patched on disk after saving.

**Tech Stack:** Unsloth, TRL SFTTrainer, W&B, HuggingFace datasets, argparse

**Spec:** `docs/superpowers/specs/2026-03-15-training-pipeline-production-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| CREATE | `scripts/colab_train.py` | SFT training entry point — model load, template apply, train, save, W&B |
| CREATE | `scripts/colab_eval.py` | Base vs fine-tuned eval — sequential model load, metrics, comparison table |
| REWRITE | `notebooks/kiki_sft_finetune.ipynb` | ~15-cell thin executor — paths, login, `!python -u` calls, display |

---

## Chunk 1: `scripts/colab_train.py`

### Task 1: Create `scripts/colab_train.py` with argument parsing and GPU auto-detect

**Files:**
- Create: `scripts/colab_train.py`

- [ ] **Step 1: Write the script skeleton with argparse**

```python
#!/usr/bin/env python3
"""Kiki SLM SFT training script for Google Colab.

Loads pre-formatted ChatML data, fine-tunes Qwen3-4B with QLoRA via Unsloth,
logs to W&B, saves adapter to output directory.

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
import re
import sys
import time
from collections import Counter
from pathlib import Path

# Unbuffered output for real-time tqdm in Colab subprocess
os.environ["PYTHONUNBUFFERED"] = "1"

import torch
from datasets import load_dataset
from transformers import TrainerCallback


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
    # Misc
    parser.add_argument("--dry-run", action="store_true", help="Validate config and exit")
    return parser.parse_args()


def auto_detect_gpu() -> dict:
    """Auto-detect GPU and return batch_size, grad_accum settings."""
    if not torch.cuda.is_available():
        print("WARNING: No GPU detected, using CPU defaults")
        return {"gpu_name": "CPU", "gpu_mem_gb": 0, "batch_size": 1, "grad_accum": 4}

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_mem / 1024**3

    if gpu_mem_gb >= 35:  # A100
        batch_size, grad_accum = 4, 8
    elif gpu_mem_gb >= 20:  # L4
        batch_size, grad_accum = 4, 4
    else:  # T4 or smaller
        batch_size, grad_accum = 2, 8

    return {"gpu_name": gpu_name, "gpu_mem_gb": round(gpu_mem_gb, 1), "batch_size": batch_size, "grad_accum": grad_accum}
```

- [ ] **Step 2: Add chat template sanitization helper (self-contained, no kiki import)**

```python
def apply_chat_template_to_dataset(dataset, tokenizer):
    """Convert messages column to text column using chat template.

    Self-contained — does not import from kiki package.
    Sanitizes each message to role+content only to avoid Jinja2 errors
    from mixed dataset formats (xlam, hermes, arcee).
    """
    original_template = tokenizer.chat_template

    def _apply(examples):
        texts = []
        for msgs in examples["messages"]:
            clean = [{"role": str(m.get("role", "user")), "content": str(m.get("content") or "")} for m in msgs]
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
```

- [ ] **Step 3: Add StepCountCallback and W&B setup**

```python
class StepCountCallback(TrainerCallback):
    """Print correct step count at training start and log progress."""

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


def setup_wandb(args, gpu_info: dict, train_size: int, eval_size: int) -> str | None:
    """Initialize W&B. Returns run URL or None."""
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
```

- [ ] **Step 4: Add the main training function**

```python
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
            print(f"    {src:<30s} {cnt:>6,} ({cnt/len(train_dataset)*100:.1f}%)")

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
    print(f"  Trainable: {trainable/1e6:.1f}M / {total_params/1e6:.1f}M ({trainable/total_params*100:.2f}%)")

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

    # 10. Train
    print(f"\n  GPU memory before training: {torch.cuda.memory_allocated()/1024**3:.2f}GB allocated")
    start_time = time.time()
    result = trainer.train()
    elapsed = time.time() - start_time

    final_loss = result.metrics.get("train_loss", 0)
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{'='*60}")
    print(f"  TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"  Duration:    {elapsed/3600:.1f}h ({elapsed/60:.0f}m)")
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
```

- [ ] **Step 5: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/colab_train.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add scripts/colab_train.py
git commit -m "feat: add colab_train.py — production SFT training with W&B"
```

---

## Chunk 2: `scripts/colab_eval.py`

### Task 2: Create `scripts/colab_eval.py` with sequential model loading

**Files:**
- Create: `scripts/colab_eval.py`

- [ ] **Step 1: Write the eval script**

```python
#!/usr/bin/env python3
"""Kiki SLM evaluation — base Qwen3 vs fine-tuned on gold test data.

Loads models sequentially (base first, then fine-tuned) to avoid OOM on T4.
Strips Qwen3 <think> tokens before JSON parsing.

Usage:
    python -u scripts/colab_eval.py \
        --adapter-path /content/drive/MyDrive/kiki-slm/adapters/kiki-sft-v1 \
        --gold-file /content/drive/MyDrive/kiki-slm/data/gold_100.jsonl \
        --output-file /content/drive/MyDrive/kiki-slm/adapters/eval_results.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
import time
from pathlib import Path

os.environ["PYTHONUNBUFFERED"] = "1"

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kiki SLM Base vs Fine-tuned Evaluation")
    parser.add_argument("--adapter-path", required=True, help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--base-model", default="Qwen/Qwen3-4B-Instruct-2507", help="Base model ID")
    parser.add_argument("--gold-file", required=True, help="Path to gold JSONL")
    parser.add_argument("--output-file", default=None, help="Where to save results JSON")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    return parser.parse_args()


EVAL_SYSTEM_PROMPT = """You are Kiki, an AI customer service agent. Analyze the customer message and respond with valid JSON containing: intent, urgency, workflow_steps, tools_required, reasoning, response."""


def load_gold_data(path: str) -> list[dict]:
    """Load gold JSONL. Uses built-in 5-example template if file missing."""
    if not os.path.exists(path):
        print(f"  WARNING: Gold file not found: {path}")
        print(f"  Using built-in 5-example template")
        return [
            {"ticket_id": "GOLD-001", "customer_message": "I placed order #12345 three days ago and it still shows processing.", "gold_intent": "order_status", "gold_urgency": "medium"},
            {"ticket_id": "GOLD-002", "customer_message": "I was charged twice for my subscription this month. I need a refund.", "gold_intent": "billing_inquiry", "gold_urgency": "high"},
            {"ticket_id": "GOLD-003", "customer_message": "My account has been locked and I cannot log in.", "gold_intent": "account_management", "gold_urgency": "critical"},
            {"ticket_id": "GOLD-004", "customer_message": "I want to return the headphones I bought last week.", "gold_intent": "return_request", "gold_urgency": "low"},
            {"ticket_id": "GOLD-005", "customer_message": "Someone made unauthorized purchases on my account totaling $500!", "gold_intent": "fraud_report", "gold_urgency": "critical"},
        ]

    tickets = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                tickets.append(json.loads(line))
    print(f"  Loaded {len(tickets)} gold tickets from {path}")
    return tickets


def parse_model_json(raw_text: str) -> dict | None:
    """Parse JSON from model output, stripping <think> tokens and markdown fences."""
    text = raw_text.strip()
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def run_inference(model, tokenizer, message: str) -> tuple[dict | None, str, float]:
    """Run inference, return (parsed_json, raw_text, latency_seconds)."""
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": EVAL_SYSTEM_PROMPT}, {"role": "user", "content": message}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    start = time.perf_counter()
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.1, do_sample=True)
    latency = time.perf_counter() - start

    raw = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    parsed = parse_model_json(raw)
    return parsed, raw, latency


def evaluate_model(model, tokenizer, tickets: list[dict], label: str) -> list[dict]:
    """Run a model on all gold tickets. Returns list of result dicts."""
    results = []
    for i, ticket in enumerate(tickets, 1):
        msg = ticket["customer_message"]
        tid = ticket.get("ticket_id", f"ticket_{i}")
        if i % 10 == 0 or i == 1 or i == len(tickets):
            print(f"    [{i}/{len(tickets)}] {tid}...")
        parsed, raw, latency = run_inference(model, tokenizer, msg)
        results.append({"ticket_id": tid, "parsed": parsed, "raw": raw, "latency": latency})
    return results


def compute_metrics(results: list[dict], tickets: list[dict]) -> dict:
    """Compute accuracy metrics."""
    n = len(results)
    intent_correct = urgency_correct = json_valid = 0
    latencies = []

    for res, ticket in zip(results, tickets):
        p = res["parsed"]
        latencies.append(res["latency"])
        if p is not None:
            json_valid += 1
            if p.get("intent", "").lower() == ticket.get("gold_intent", "").lower():
                intent_correct += 1
            if p.get("urgency", "").lower() == ticket.get("gold_urgency", "").lower():
                urgency_correct += 1

    return {
        "intent_accuracy": intent_correct / n if n else 0,
        "urgency_accuracy": urgency_correct / n if n else 0,
        "json_parse_rate": json_valid / n if n else 0,
        "avg_latency_s": sum(latencies) / len(latencies) if latencies else 0,
        "total": n,
    }


def free_model(model):
    """Delete model and free GPU memory."""
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def print_comparison(base_metrics: dict, ft_metrics: dict, base_results: list, ft_results: list, tickets: list):
    """Print formatted comparison table and per-ticket detail."""
    n = base_metrics["total"]
    print(f"\n{'='*65}")
    print(f"  BASE vs FINE-TUNED COMPARISON ({n} gold tickets)")
    print(f"{'='*65}")
    print(f"  {'Metric':<28s} {'Base Qwen3':>15s} {'Fine-tuned':>15s}")
    print(f"  {'-'*58}")

    for key, label, fmt in [
        ("intent_accuracy", "Intent Accuracy", "{:.1%}"),
        ("urgency_accuracy", "Urgency Accuracy", "{:.1%}"),
        ("json_parse_rate", "JSON Parse Rate", "{:.1%}"),
        ("avg_latency_s", "Avg Latency (s)", "{:.2f}"),
    ]:
        bv = fmt.format(base_metrics[key])
        fv = fmt.format(ft_metrics[key])
        print(f"  {label:<28s} {bv:>15s} {fv:>15s}")

    print(f"{'='*65}")

    for key, label in [("intent_accuracy", "Intent"), ("urgency_accuracy", "Urgency"), ("json_parse_rate", "JSON Parse")]:
        diff = ft_metrics[key] - base_metrics[key]
        arrow = "+" if diff >= 0 else ""
        print(f"  {label} improvement: {arrow}{diff:.1%}")

    print(f"\n{'='*65}")
    print(f"  PER-TICKET DETAIL")
    print(f"{'='*65}")
    for i, ticket in enumerate(tickets):
        tid = ticket.get("ticket_id", f"ticket_{i+1}")
        b = base_results[i]
        f = ft_results[i]
        b_intent = b["parsed"].get("intent", "PARSE_FAIL") if b["parsed"] else "PARSE_FAIL"
        f_intent = f["parsed"].get("intent", "PARSE_FAIL") if f["parsed"] else "PARSE_FAIL"
        gold = ticket.get("gold_intent", "?")
        b_ok = "Y" if b_intent.lower() == gold.lower() else "N"
        f_ok = "Y" if f_intent.lower() == gold.lower() else "N"
        print(f"\n  {tid} | Gold: {gold}")
        print(f"    Base:       {b_intent} [{b_ok}] ({b['latency']:.2f}s)")
        print(f"    Fine-tuned: {f_intent} [{f_ok}] ({f['latency']:.2f}s)")


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  KIKI SLM — BASE vs FINE-TUNED EVALUATION")
    print(f"{'='*60}")

    # Load gold data
    tickets = load_gold_data(args.gold_file)

    # --- Base model ---
    print(f"\n  Loading BASE model: {args.base_model}")
    from unsloth import FastLanguageModel

    base_model, base_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model, max_seq_length=args.max_seq_length, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(base_model)

    print(f"  Running base model on {len(tickets)} tickets...")
    base_results = evaluate_model(base_model, base_tokenizer, tickets, "Base")

    # Free base model VRAM before loading fine-tuned
    print(f"  Freeing base model VRAM...")
    free_model(base_model)
    del base_tokenizer

    # --- Fine-tuned model ---
    print(f"\n  Loading FINE-TUNED model: {args.adapter_path}")
    ft_model, ft_tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.adapter_path, max_seq_length=args.max_seq_length, load_in_4bit=True,
    )
    FastLanguageModel.for_inference(ft_model)

    print(f"  Running fine-tuned model on {len(tickets)} tickets...")
    ft_results = evaluate_model(ft_model, ft_tokenizer, tickets, "Fine-tuned")

    free_model(ft_model)

    # --- Metrics ---
    base_metrics = compute_metrics(base_results, tickets)
    ft_metrics = compute_metrics(ft_results, tickets)

    print_comparison(base_metrics, ft_metrics, base_results, ft_results, tickets)

    # --- Save ---
    output_file = args.output_file
    if output_file is None:
        output_file = str(Path(args.adapter_path) / "eval_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    output = {
        "base_metrics": base_metrics,
        "ft_metrics": ft_metrics,
        "base_results": [{k: v for k, v in r.items() if k != "raw"} for r in base_results],
        "ft_results": [{k: v for k, v in r.items() if k != "raw"} for r in ft_results],
    }
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {output_file}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify syntax**

Run: `python -c "import ast; ast.parse(open('scripts/colab_eval.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/colab_eval.py
git commit -m "feat: add colab_eval.py — base vs fine-tuned eval with sequential loading"
```

---

## Chunk 3: Notebook Rewrite

### Task 3: Rewrite `notebooks/kiki_sft_finetune.ipynb` as ~15-cell thin executor

**Files:**
- Rewrite: `notebooks/kiki_sft_finetune.ipynb`

- [ ] **Step 1: Write the new notebook with ~15 cells**

Cells in order:

**Cell 1 (markdown):** Title + prerequisites
**Cell 2 (code):** Install deps — uv + unsloth + wandb + datasets
**Cell 3 (code):** Verify GPU
**Cell 4 (code):** Mount Google Drive
**Cell 5 (code):** Clone/pull repo + `%cd` to repo root
**Cell 6 (code):** Configure paths + verify data files exist
**Cell 7 (code):** `wandb.login()`
**Cell 8 (code):** Inspect data (load JSONL, show stats — 5 lines inline)
**Cell 9 (markdown):** Training section header
**Cell 10 (code):** `!python -u scripts/colab_train.py --train-file ... --wandb`
**Cell 11 (code):** Display W&B link
**Cell 12 (markdown):** Evaluation section header
**Cell 13 (code):** `!python -u scripts/colab_eval.py --adapter-path ... --gold-file ...`
**Cell 14 (code):** Quick 3-message inference test (small inline)
**Cell 15 (markdown):** Done + next steps

- [ ] **Step 2: Verify notebook is valid JSON**

Run: `python -c "import json; json.load(open('notebooks/kiki_sft_finetune.ipynb')); print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add notebooks/kiki_sft_finetune.ipynb
git commit -m "refactor: rewrite notebook as thin executor with colab_train/eval scripts"
```

---

## Chunk 4: Final verification

### Task 4: Syntax check all scripts and verify no broken imports

- [ ] **Step 1: Syntax check all modified files**

Run:
```bash
python -c "
import ast
for f in ['scripts/colab_train.py', 'scripts/colab_eval.py']:
    ast.parse(open(f).read())
    print(f'{f}: OK')
import json
json.load(open('notebooks/kiki_sft_finetune.ipynb'))
print('notebook: OK')
"
```
Expected: All OK

- [ ] **Step 2: Verify no stale references**

Run: `grep -r "LiveLossPlotCallback\|ORIGINAL_CHAT_TEMPLATE\|loss_callback" notebooks/ scripts/colab_*.py`
Expected: No matches (old callback code fully removed)

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "chore: training pipeline production refactor complete"
```
