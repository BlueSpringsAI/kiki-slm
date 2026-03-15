#!/usr/bin/env python3
"""Script 2: QLoRA fine-tuning of Kiki SLM on annotated customer support data.

Loads annotated JSONL, converts to ChatML format, and fine-tunes a Qwen model
using Unsloth + TRL SFTTrainer with QLoRA. Optionally exports to GGUF.

Usage:
    python scripts/2_train.py                              # Full training run
    python scripts/2_train.py --dry-run                    # Format data only (no GPU)
    python scripts/2_train.py --skip-export                # Train without GGUF export
    python scripts/2_train.py --config configs/custom.yaml # Custom config
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import yaml

# ── Logging setup ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ── Config loading ────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


# ── Data formatting ───────────────────────────────────────────────────────

def format_training_example(record: dict) -> dict:
    """Convert an annotated record to ChatML messages format."""
    system_prompt = """You are Kiki, an AI customer service agent. When given a customer message, analyze it and respond with:
1. Your classification (intent, urgency)
2. The workflow steps needed to resolve this
3. Which tools to invoke with what parameters
4. A professional, empathetic response to the customer

Always respond in valid JSON with these fields:
- intent: string
- urgency: string (critical/high/medium/low)
- workflow_steps: list of strings
- tools_required: list of strings with parameters
- reasoning: brief explanation of your analysis
- response: the customer-facing reply"""

    assistant_output = json.dumps({
        "intent": record["intent"],
        "urgency": record["urgency"],
        "workflow_steps": record["workflow_steps"],
        "tools_required": record["tools_required"],
        "reasoning": f"Customer intent is {record['intent']} with {record['urgency']} urgency. "
                     f"Required workflow: {' → '.join(record['workflow_steps'])}.",
        "response": record["agent_response"]
    }, indent=2)

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record["customer_message"]},
            {"role": "assistant", "content": assistant_output}
        ]
    }


def load_and_format_data(data_path: str) -> list[dict]:
    """Load annotated JSONL and convert all records to ChatML format."""
    data_file = Path(data_path)
    if not data_file.exists():
        raise FileNotFoundError(f"Annotated data not found: {data_path}")

    records = []
    skipped = 0
    with open(data_file) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {line_num}: invalid JSON — {e}")
                skipped += 1
                continue

            # Validate required fields
            required = ["intent", "urgency", "workflow_steps", "tools_required",
                        "customer_message", "agent_response"]
            missing = [f for f in required if f not in record]
            if missing:
                logger.warning(f"Skipping line {line_num}: missing fields {missing}")
                skipped += 1
                continue

            try:
                formatted = format_training_example(record)
                records.append(formatted)
            except Exception as e:
                logger.warning(f"Skipping line {line_num}: formatting error — {e}")
                skipped += 1

    logger.info(f"Loaded {len(records)} records from {data_path} ({skipped} skipped)")
    return records


# ── Train/eval split ─────────────────────────────────────────────────────

def split_data(records: list[dict], train_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split records into train and eval sets."""
    random.seed(seed)
    indices = list(range(len(records)))
    random.shuffle(indices)

    split_idx = int(len(records) * train_ratio)
    train_indices = indices[:split_idx]
    eval_indices = indices[split_idx:]

    train_data = [records[i] for i in train_indices]
    eval_data = [records[i] for i in eval_indices]

    logger.info(f"Split: {len(train_data)} train, {len(eval_data)} eval "
                f"({train_ratio:.0%}/{1 - train_ratio:.0%})")
    return train_data, eval_data


def save_formatted_data(records: list[dict], output_path: Path) -> None:
    """Save formatted records to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    logger.info(f"Saved {len(records)} records to {output_path}")


# ── Data statistics ───────────────────────────────────────────────────────

def print_data_stats(train_data: list[dict], eval_data: list[dict]) -> None:
    """Print summary statistics for formatted data."""
    def compute_stats(data: list[dict]) -> dict:
        lengths = []
        for record in data:
            total_chars = sum(len(m["content"]) for m in record["messages"])
            lengths.append(total_chars)
        return {
            "count": len(data),
            "avg_chars": sum(lengths) / len(lengths) if lengths else 0,
            "min_chars": min(lengths) if lengths else 0,
            "max_chars": max(lengths) if lengths else 0,
        }

    train_stats = compute_stats(train_data)
    eval_stats = compute_stats(eval_data)

    print("\n" + "=" * 60)
    print("DATA STATISTICS")
    print("=" * 60)
    print(f"  Train examples:     {train_stats['count']}")
    print(f"  Eval examples:      {eval_stats['count']}")
    print(f"  Train avg chars:    {train_stats['avg_chars']:.0f}")
    print(f"  Train min/max:      {train_stats['min_chars']} / {train_stats['max_chars']}")
    print(f"  Eval avg chars:     {eval_stats['avg_chars']:.0f}")
    print(f"  Eval min/max:       {eval_stats['min_chars']} / {eval_stats['max_chars']}")
    print("=" * 60 + "\n")


# ── Training ──────────────────────────────────────────────────────────────

def run_training(config: dict, train_path: Path, eval_path: Path) -> tuple:
    """Load model, train with SFTTrainer, return (model, tokenizer, trainer)."""
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastLanguageModel

    # Load model
    logger.info(f"Loading model: {config['model']['name']}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config["model"]["name"],
        max_seq_length=config["model"]["max_seq_length"],
        load_in_4bit=config["model"]["load_in_4bit"],
        dtype=None,
    )
    logger.info("Model loaded successfully")

    # Apply LoRA
    logger.info("Applying LoRA adapters")
    model = FastLanguageModel.get_peft_model(
        model,
        r=config["lora"]["r"],
        lora_alpha=config["lora"]["lora_alpha"],
        target_modules=config["lora"]["target_modules"],
        lora_dropout=config["lora"]["lora_dropout"],
        bias=config["lora"]["bias"],
        use_gradient_checkpointing="unsloth",
        random_state=config["training"]["seed"],
    )
    logger.info("LoRA adapters applied")

    # Load datasets
    logger.info("Loading formatted datasets")
    dataset = load_dataset("json", data_files={
        "train": str(train_path),
        "eval": str(eval_path),
    })
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]
    logger.info(f"Datasets loaded — train: {len(train_dataset)}, eval: {len(eval_dataset)}")

    # Training arguments
    tc = config["training"]
    output_dir = "outputs/checkpoints/kiki-poc"

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=tc["per_device_train_batch_size"],
        gradient_accumulation_steps=tc["gradient_accumulation_steps"],
        num_train_epochs=tc["num_train_epochs"],
        learning_rate=tc["learning_rate"],
        lr_scheduler_type=tc["lr_scheduler_type"],
        warmup_ratio=tc["warmup_ratio"],
        bf16=tc["bf16"],
        optim=tc["optim"],
        packing=tc["packing"],
        max_seq_length=config["model"]["max_seq_length"],
        save_steps=tc["save_steps"],
        eval_steps=tc["eval_steps"],
        eval_strategy="steps",
        logging_steps=10,
        seed=tc["seed"],
        report_to="none",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=sft_config,
    )

    # Train
    logger.info("Starting training")
    start_time = time.time()
    train_result = trainer.train()
    elapsed = time.time() - start_time

    # Print training stats
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    print(f"  Training time:      {elapsed:.1f}s ({elapsed / 60:.1f}m)")
    print(f"  Train loss:         {train_result.training_loss:.4f}")
    print(f"  Train samples:      {train_result.metrics.get('train_samples_per_second', 'N/A')}")
    print(f"  Global steps:       {train_result.metrics.get('train_steps', trainer.state.global_step)}")
    print(f"  Epochs completed:   {train_result.metrics.get('epoch', tc['num_train_epochs'])}")
    print("=" * 60 + "\n")

    return model, tokenizer, trainer


# ── Save & Export ─────────────────────────────────────────────────────────

def save_adapter(model, tokenizer, output_dir: str = "outputs/adapters/kiki-poc") -> None:
    """Save LoRA adapter and tokenizer."""
    adapter_path = Path(output_dir)
    adapter_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    logger.info(f"Adapter and tokenizer saved to {adapter_path}")


def export_gguf(model, tokenizer, config: dict) -> None:
    """Export model to GGUF format."""
    export_cfg = config["export"]
    export_dir = Path(export_cfg["output_dir"]) / "kiki-poc-q4"
    export_dir.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Exporting to GGUF ({export_cfg['quantization']}) at {export_dir}")
    model.save_pretrained_gguf(
        str(export_dir),
        tokenizer,
        quantization_method=export_cfg["quantization"],
    )
    logger.info(f"GGUF export complete: {export_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning of Kiki SLM on annotated customer support data"
    )
    parser.add_argument(
        "--config", default="configs/poc_config.yaml",
        help="Path to YAML config file (default: configs/poc_config.yaml)"
    )
    parser.add_argument(
        "--data", default="data/annotated/annotated_5k.jsonl",
        help="Path to annotated JSONL data (default: data/annotated/annotated_5k.jsonl)"
    )
    parser.add_argument(
        "--skip-export", action="store_true",
        help="Skip GGUF export after training"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Format data and print stats without loading the model (no GPU needed)"
    )
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("KIKI SLM POC — QLoRA Fine-Tuning")
    print("=" * 60)

    # Step 1: Load config
    logger.info("[1/6] Loading configuration")
    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    train_split = config["data"].get("train_split", 0.9)
    seed = config["training"].get("seed", 42)

    # Step 2: Load and format data
    logger.info("[2/6] Loading and formatting annotated data")
    try:
        formatted_data = load_and_format_data(args.data)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not formatted_data:
        logger.error("No valid training examples found. Exiting.")
        sys.exit(1)

    # Step 3: Split data
    logger.info("[3/6] Splitting into train/eval sets")
    train_data, eval_data = split_data(formatted_data, train_split, seed)

    # Step 4: Save formatted data
    logger.info("[4/6] Saving formatted data")
    train_path = Path("data/formatted/train.jsonl")
    eval_path = Path("data/formatted/eval.jsonl")
    save_formatted_data(train_data, train_path)
    save_formatted_data(eval_data, eval_path)

    # Print data stats
    print_data_stats(train_data, eval_data)

    # Dry-run exits here
    if args.dry_run:
        print("=" * 60)
        print("DRY RUN COMPLETE")
        print("=" * 60)
        print(f"  Config:             {args.config}")
        print(f"  Model:              {config['model']['name']}")
        print(f"  LoRA r:             {config['lora']['r']}")
        print(f"  LoRA alpha:         {config['lora']['lora_alpha']}")
        print(f"  Target modules:     {', '.join(config['lora']['target_modules'])}")
        print(f"  Batch size:         {config['training']['per_device_train_batch_size']}")
        print(f"  Grad accum steps:   {config['training']['gradient_accumulation_steps']}")
        print(f"  Epochs:             {config['training']['num_train_epochs']}")
        print(f"  Learning rate:      {config['training']['learning_rate']}")
        print(f"  Scheduler:          {config['training']['lr_scheduler_type']}")
        print(f"  Max seq length:     {config['model']['max_seq_length']}")
        print(f"  Packing:            {config['training']['packing']}")
        print(f"  Seed:               {seed}")
        print(f"  Train file:         {train_path}")
        print(f"  Eval file:          {eval_path}")
        print("=" * 60)
        print("\nTo run full training, remove --dry-run flag.")
        return

    # Step 5: Train
    logger.info("[5/6] Starting QLoRA fine-tuning")
    try:
        model, tokenizer, trainer = run_training(config, train_path, eval_path)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Save adapter
    logger.info("Saving adapter and tokenizer")
    save_adapter(model, tokenizer)

    # Step 6: Export
    if args.skip_export:
        logger.info("[6/6] Skipping GGUF export (--skip-export flag set)")
    else:
        logger.info("[6/6] Exporting to GGUF")
        try:
            export_gguf(model, tokenizer, config)
        except Exception as e:
            logger.error(f"GGUF export failed: {e}")
            logger.info("Training was successful. You can retry export manually.")

    print("\n" + "=" * 60)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Adapter saved:      outputs/adapters/kiki-poc/")
    if not args.skip_export:
        print(f"  GGUF exported:      outputs/exports/kiki-poc-q4/")
    print(f"  Checkpoints:        outputs/checkpoints/kiki-poc/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
