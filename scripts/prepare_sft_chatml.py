#!/usr/bin/env python3
"""Download all 10 SFT datasets, convert to ChatML, mix, and save as JSONL.

Output: data/formatted/kiki_sft_chatml_train.jsonl  (90%)
        data/formatted/kiki_sft_chatml_eval.jsonl   (10%)

Upload these to Google Drive for Colab training.

Usage:
    python scripts/prepare_sft_chatml.py
    python scripts/prepare_sft_chatml.py --total-examples 30000
    python scripts/prepare_sft_chatml.py --hf-token hf_xxx
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 10 SFT datasets with converters and sampling weights
# ---------------------------------------------------------------------------

SFT_DATASETS = {
    # --- Real CS conversations (highest weight — these have real responses) ---
    "bitext_cs": {
        "hf_id": "bitext/Bitext-customer-support-llm-chatbot-training-dataset",
        "converter": "bitext",
        "weight": 0.20,
        "description": "General CS (27K)",
    },
    "bitext_ecom": {
        "hf_id": "bitext/Bitext-retail-ecommerce-llm-chatbot-training-dataset",
        "converter": "bitext",
        "weight": 0.20,
        "description": "E-commerce (50K+)",
    },
    "customer_support_tickets": {
        "hf_id": "Tobi-Bueck/customer-support-tickets",
        "converter": "ticket",
        "weight": 0.15,
        "filter_language": "en",
        "description": "Helpdesk tickets (61.8K, English subset — ONLY source of technical_support)",
    },
    # --- Vertical-specific CS (lower weight — useful but narrower domain) ---
    "bitext_banking": {
        "hf_id": "bitext/Bitext-retail-banking-llm-chatbot-training-dataset",
        "converter": "bitext",
        "weight": 0.05,
        "description": "Banking (37K+)",
    },
    "bitext_insurance": {
        "hf_id": "bitext/Bitext-insurance-llm-chatbot-training-dataset",
        "converter": "bitext",
        "weight": 0.05,
        "description": "Insurance (38K+)",
    },
    # --- Intent classification (lower weight — synthetic responses, useful for classification) ---
    "banking77": {
        "hf_id": "legacy-datasets/banking77",
        "converter": "banking77",
        "weight": 0.05,
        "description": "Banking intent 77-class (10K)",
    },
    "clinc_oos": {
        "hf_id": "clinc/clinc_oos",
        "subset": "plus",
        "converter": "clinc",
        "weight": 0.02,
        "filter_clinc_cs_only": True,
        "description": "Intent + OOS — CS-relevant intents only (filtered from 23.7K)",
    },
    # --- Tool calling / agentic (important for tool selection training) ---
    "arcee_agent": {
        "hf_id": "arcee-ai/agent-data",
        "converter": "arcee_agent",
        "weight": 0.15,
        "description": "Agent data (486K)",
    },
    "xlam_60k": {
        "hf_id": "Salesforce/xlam-function-calling-60k",
        "converter": "xlam",
        "weight": 0.08,
        "description": "Function calling (60K)",
    },
    "hermes_fc": {
        "hf_id": "NousResearch/hermes-function-calling-v1",
        "converter": "hermes",
        "weight": 0.05,
        "description": "Tool calling (11.6K)",
    },
}

# CS-relevant CLINC intents (filter out weather, jokes, recipes, etc.)
CLINC_CS_INTENTS = {
    "order_status", "order", "order_checks", "bill_balance", "bill_due",
    "pay_bill", "min_payment", "transactions", "spending_history", "balance",
    "credit_score", "credit_limit", "credit_limit_change", "interest_rate",
    "apr", "international_fees", "exchange_rate", "redeem_rewards",
    "rewards_balance", "income", "taxes", "w2", "payday", "rollover_401k",
    "card_declined", "transfer", "direct_deposit",
    "report_fraud", "report_lost_card", "freeze_account",
    "account_blocked", "pin_change", "reset_settings", "change_user_name",
    "cancel", "cancel_reservation",
    "new_card", "insurance", "insurance_change", "damaged_card",
    "replacement_card_duration", "lost_luggage",
    "oos",  # Keep OOS for general_inquiry training
}


def download_and_convert(
    name: str,
    config: dict,
    num_samples: int | None,
    token: str | None = None,
) -> list[dict]:
    """Download a dataset, convert ALL rows to ChatML, then subsample.

    Args:
        num_samples: Target number of examples. None = use all valid rows.
    """
    from kiki.data.processors import ChatMLConverter

    hf_id = config["hf_id"]
    converter_name = config["converter"]

    logger.info("Downloading '%s' (%s)...", name, hf_id)
    t0 = time.monotonic()

    kwargs = {"path": hf_id, "split": "train"}
    if config.get("subset"):
        kwargs["name"] = config["subset"]
    if token:
        kwargs["token"] = token

    try:
        ds = load_dataset(**kwargs)
    except Exception as exc:
        logger.error("Failed to download '%s': %s", name, exc)
        return []

    elapsed = time.monotonic() - t0
    logger.info("  Downloaded %d examples in %.1fs", len(ds), elapsed)

    # Filter by language if specified
    lang_filter = config.get("filter_language")
    if lang_filter and "language" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda x: x.get("language") == lang_filter)
        logger.info("  Filtered to language='%s': %d -> %d", lang_filter, before, len(ds))

    # Filter CLINC to CS-relevant intents only (remove weather, jokes, recipes, etc.)
    if config.get("filter_clinc_cs_only") and "intent" in ds.column_names:
        before = len(ds)
        # CLINC uses integer labels — resolve to names and filter
        features = ds.features.get("intent")
        if hasattr(features, "int2str"):
            ds = ds.filter(lambda x: features.int2str(x["intent"]) in CLINC_CS_INTENTS)
        else:
            ds = ds.filter(lambda x: str(x.get("intent", "")) in CLINC_CS_INTENTS)
        logger.info("  Filtered to CS-relevant intents: %d -> %d", before, len(ds))

    # Convert ALL rows to ChatML first — then subsample from valid results
    converter = ChatMLConverter.get_converter(converter_name)
    converted = []
    skipped = 0

    for example in ds:
        try:
            result = converter(dict(example))
            messages = result.get("messages", [])

            # Validate: need at least system + user + assistant
            if len(messages) < 2:
                skipped += 1
                continue

            # Filter out empty assistant messages
            has_assistant = any(
                m.get("role") == "assistant" and m.get("content", "").strip()
                for m in messages
            )
            if not has_assistant:
                skipped += 1
                continue

            converted.append({"messages": messages, "source": name})
        except Exception:
            skipped += 1

    logger.info("  Converted %d examples (%d skipped) from '%s'", len(converted), skipped, name)

    # Subsample AFTER conversion so failures don't eat into the target
    if num_samples is not None and len(converted) > num_samples:
        converted = random.sample(converted, num_samples)
        logger.info("  Subsampled to %d examples", num_samples)

    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all 10 SFT datasets as ChatML JSONL")
    parser.add_argument("--total-examples", type=int, default=50000, help="Target total examples (default: 50000)")
    parser.add_argument("--use-all", action="store_true", help="Use ALL valid examples from every dataset (ignore --total-examples)")
    parser.add_argument("--eval-ratio", type=float, default=0.10, help="Eval split ratio (default: 0.10)")
    parser.add_argument("--output-dir", type=str, default="data/formatted", help="Output directory")
    parser.add_argument("--hf-token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    # Calculate per-dataset samples
    all_examples: list[dict] = []
    total_weight = sum(c["weight"] for c in SFT_DATASETS.values())

    mode = "ALL available" if args.use_all else f"~{args.total_examples:,} target"
    print(f"\n{'='*70}")
    print(f"Preparing ChatML examples from 10 SFT datasets ({mode})")
    print(f"{'='*70}\n")

    for name, config in SFT_DATASETS.items():
        if args.use_all:
            num_samples = None  # Take everything
        else:
            num_samples = int(args.total_examples * config["weight"] / total_weight)
        examples = download_and_convert(name, config, num_samples, token)
        all_examples.extend(examples)

    # Deduplicate (hash on user+assistant content, ignoring shared system prompt)
    import hashlib
    seen = set()
    unique_examples = []
    for ex in all_examples:
        key_parts = [
            f"{m.get('role')}:{m.get('content', '')}"
            for m in ex.get("messages", [])
            if m.get("role") != "system"
        ]
        h = hashlib.md5("|".join(key_parts).encode()).hexdigest()
        if h not in seen:
            seen.add(h)
            unique_examples.append(ex)
    dupes_removed = len(all_examples) - len(unique_examples)
    if dupes_removed:
        logger.info("Deduplication: removed %d duplicates (%d -> %d)", dupes_removed, len(all_examples), len(unique_examples))
    all_examples = unique_examples

    # Shuffle
    random.shuffle(all_examples)

    # Split train/eval — ensure no overlap by using disjoint hash buckets
    eval_size = int(len(all_examples) * args.eval_ratio)
    eval_data = all_examples[:eval_size]
    train_data = all_examples[eval_size:]

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "kiki_sft_chatml_train.jsonl"
    eval_path = output_dir / "kiki_sft_chatml_eval.jsonl"

    with open(train_path, "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    with open(eval_path, "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    # Print summary
    print(f"\n{'='*70}")
    print(f"DONE! Saved {len(train_data)} train + {len(eval_data)} eval examples")
    print(f"{'='*70}")
    print(f"  Train: {train_path} ({train_path.stat().st_size / 1024 / 1024:.1f} MB)")
    print(f"  Eval:  {eval_path} ({eval_path.stat().st_size / 1024 / 1024:.1f} MB)")

    # Source distribution
    from collections import Counter
    source_counts = Counter(ex["source"] for ex in all_examples)
    print(f"\n  Source distribution:")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"    {source:<30s} {count:>6d} ({count/len(all_examples)*100:.1f}%)")

    print(f"\n  Upload these files to Google Drive for Colab training:")
    print(f"    1. {train_path}")
    print(f"    2. {eval_path}")
    print()


if __name__ == "__main__":
    main()
